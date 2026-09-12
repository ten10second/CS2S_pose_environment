#!/usr/bin/env python3
"""Migrate KITTI Utonia pixel caches to the canonical LiDAR projection.

This is a migration-only repair tool for caches built under the legacy sd21
NumPy 2.2.6 FP32 projection behavior.  It never recomputes Utonia features.
Instead, it recovers the legacy visible raw source-point indices, verifies that
an input NPZ exactly matches those legacy pixels/depths, and remaps the existing
feature rows onto the current canonical projection source-point order.

Frames whose canonical visible source points are not all present in the legacy
cache are written to a rebuild manifest and no migrated NPZ is created for them.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_pixel_feature_cache import (  # noqa: E402
    PIXEL_CACHE_FORMAT,
    PIXEL_DEPTH_KEY,
    PIXEL_FEATURE_KEY,
    PIXEL_INDEX_KEY,
    load_npz_pixel_cache,
    safe_sample_id,
    validate_pixel_arrays,
)
from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    load_raw_calibration,
    load_velodyne_points,
    project_velo_to_image,
    read_jsonl,
    zbuffer_visible_point_indices,
)
from dataloader.kitti_raw_lidar_utils import LIDAR_PROJECTION_VERSION  # noqa: E402


def atomic_savez(path: Path, **arrays) -> None:
    temp_path = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    temp_path.unlink(missing_ok=True)
    try:
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def rewrite_path(path: object, args: object) -> str:
    value = str(path)
    path_rewrite = str(getattr(args, "path_rewrite", "") or "")
    if path_rewrite:
        if "=" not in path_rewrite:
            raise ValueError("--path-rewrite must be OLD=NEW")
        old, new = path_rewrite.split("=", 1)
        value = value.replace(old, new, 1)
    kitti_root = str(getattr(args, "kitti_root", "") or "")
    if kitti_root:
        marker = "/KITTI_RAW/"
        if marker in value:
            value = str(Path(kitti_root) / value.split(marker, 1)[1])
    return value


def resolve_calib_dir(record: Mapping[str, object], args: object) -> str:
    value = Path(rewrite_path(record["calib_dir"], args))
    if (value / "calib_cam_to_cam.txt").is_file() and (value / "calib_velo_to_cam.txt").is_file():
        return str(value)

    date = str(record.get("date", ""))
    kitti_root = str(getattr(args, "kitti_root", "") or "")
    if kitti_root and date:
        candidates = (
            Path(kitti_root) / date / f"{date}_calib",
            Path(kitti_root) / date,
        )
        for candidate in candidates:
            if (candidate / "calib_cam_to_cam.txt").is_file() and (
                candidate / "calib_velo_to_cam.txt"
            ).is_file():
                return str(candidate)
    return str(value)

LEGACY_NUMPY_VERSION = "2.2.6"
LEGACY_PROJECTION_VERSION = "legacy_sd21_numpy226_fp32_matmul_v1"
MIGRATION_FORMAT = "kitti_utonia_pixel_cache_projection_migration_v1"
_CALIB_CACHE: Dict[str, Mapping[str, np.ndarray]] = {}


class MissingSourceFeaturesError(ValueError):
    """Raised when canonical source points are absent from the legacy cache."""

    def __init__(self, missing_indices: Sequence[int]):
        self.missing_indices = [int(value) for value in missing_indices]
        preview = ", ".join(str(value) for value in self.missing_indices[:10])
        suffix = "" if len(self.missing_indices) <= 10 else ", ..."
        super().__init__(
            f"canonical source points missing from legacy feature rows: {preview}{suffix}"
        )


def remap_features_by_source_index(
    features: np.ndarray,
    old_source_indices: np.ndarray,
    new_source_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return feature rows ordered by ``new_source_indices``.

    Every canonical source point must already exist in ``old_source_indices``.
    Missing rows are a hard error because this tool must not silently drop
    canonical visible supervision or invent Utonia features.
    """

    features = np.asarray(features)
    old_source_indices = np.asarray(old_source_indices, dtype=np.int64)
    new_source_indices = np.asarray(new_source_indices, dtype=np.int64)
    if old_source_indices.ndim != 1 or new_source_indices.ndim != 1:
        raise ValueError("source index arrays must be one-dimensional")
    if features.shape[0] != old_source_indices.shape[0]:
        raise ValueError(
            f"feature rows {features.shape[0]} != old source rows {old_source_indices.shape[0]}"
        )
    if np.unique(old_source_indices).size != old_source_indices.size:
        raise ValueError("old source indices must be unique")
    if np.unique(new_source_indices).size != new_source_indices.size:
        raise ValueError("new source indices must be unique")

    row_by_source = {int(source): row for row, source in enumerate(old_source_indices.tolist())}
    missing = [int(source) for source in new_source_indices.tolist() if int(source) not in row_by_source]
    if missing:
        raise MissingSourceFeaturesError(missing)
    rows = np.asarray([row_by_source[int(source)] for source in new_source_indices.tolist()], dtype=np.int64)
    return features[rows], rows


def legacy_velo_to_rect(points_xyz: np.ndarray, calib: Mapping[str, np.ndarray]) -> np.ndarray:
    """Copied legacy FP32 projection helper for migration verification only."""

    if points_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts_h = np.concatenate(
        [points_xyz[:, :3], np.ones((points_xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    ).T
    rect = (calib["R_rect_00_ext"] @ calib["Tr_velo_to_cam"] @ pts_h).T
    return rect[:, :3].astype(np.float32)


def legacy_project_velo_to_image(
    points_xyz: np.ndarray,
    calib: Mapping[str, np.ndarray],
    output_size: Tuple[int, int] = (128, 512),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copied legacy sd21/NumPy-2.2.6 FP32 projection, migration-only."""

    if points_xyz.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    out_h, out_w = output_size
    src_w, src_h = calib["S_rect_02"]

    rect_xyz = legacy_velo_to_rect(points_xyz, calib)
    rect_h = np.concatenate(
        [rect_xyz, np.ones((rect_xyz.shape[0], 1), dtype=np.float32)],
        axis=1,
    ).T
    pix = calib["P_rect_02"] @ rect_h

    depth = pix[2]
    safe_depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uv = (pix[:2] / safe_depth).T
    uv[:, 0] *= out_w / src_w
    uv[:, 1] *= out_h / src_h

    valid = (
        (depth > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < out_w)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < out_h)
    )
    return uv.astype(np.float32), depth.astype(np.float32), valid


def pixel_index_from_projected_uv(
    uv: np.ndarray,
    indices: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Convert projected UV to flattened pixel ids without changing UV precision."""

    out_h, out_w = output_size
    selected_uv = np.asarray(uv)[np.asarray(indices, dtype=np.int64)]
    x = np.rint(selected_uv[:, 0]).astype(np.int64)
    y = np.rint(selected_uv[:, 1]).astype(np.int64)
    x = np.clip(x, 0, int(out_w) - 1)
    y = np.clip(y, 0, int(out_h) - 1)
    return (y * int(out_w) + x).astype(np.int64, copy=False)


def legacy_zbuffer_visible_point_indices(
    uv: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    output_size: Tuple[int, int],
) -> np.ndarray:
    """Copied legacy z-buffer source-index selection, migration-only."""

    out_h, out_w = output_size
    valid = np.asarray(valid, dtype=bool)
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0.0)
    valid_indices = np.nonzero(valid & finite)[0]
    if valid_indices.size == 0:
        return np.zeros((0,), dtype=np.int64)

    x = np.rint(uv[valid_indices, 0]).astype(np.int64)
    y = np.rint(uv[valid_indices, 1]).astype(np.int64)
    x = np.clip(x, 0, int(out_w) - 1)
    y = np.clip(y, 0, int(out_h) - 1)
    pixel_index = y * int(out_w) + x

    order = np.lexsort((valid_indices, depth[valid_indices], pixel_index))
    ordered_pixels = pixel_index[order]
    keep = np.ones(order.shape[0], dtype=bool)
    keep[1:] = ordered_pixels[1:] != ordered_pixels[:-1]
    return np.sort(valid_indices[order[keep]]).astype(np.int64, copy=False)


def visible_projection_payload(
    points: np.ndarray,
    calib: Mapping[str, np.ndarray],
    output_size: Tuple[int, int],
    max_depth: float,
    projector,
    zbuffer_fn=zbuffer_visible_point_indices,
) -> Dict[str, np.ndarray]:
    """Compute Utonia-encodable full-scan front-surface source ids and pixels."""

    xyz = points[:, :3].astype(np.float32)
    ranges = np.linalg.norm(xyz, axis=1)
    encoder_mask = np.isfinite(xyz).all(axis=1) & np.isfinite(ranges) & (ranges > 0.0) & (
        ranges <= float(max_depth)
    )
    raw_uv, raw_depth, raw_projected = projector(xyz, calib, output_size)
    raw_projected = raw_projected & np.isfinite(raw_depth) & (raw_depth > 0.0)
    raw_zbuffer_indices = zbuffer_fn(raw_uv, raw_depth, raw_projected, output_size)
    keep = encoder_mask[raw_zbuffer_indices] if raw_zbuffer_indices.size else np.zeros((0,), dtype=bool)
    source_indices = raw_zbuffer_indices[keep].astype(np.int64, copy=False)
    return {
        "source_indices": source_indices,
        "pixel_index": pixel_index_from_projected_uv(raw_uv, source_indices, output_size),
        "depth": np.asarray(raw_depth[source_indices], dtype=np.float32),
        "encoder_point_count": np.asarray(int(encoder_mask.sum()), dtype=np.int32),
        "projected_point_count": np.asarray(int(raw_projected.sum()), dtype=np.int32),
        "raw_zbuffer_visible_point_count": np.asarray(int(raw_zbuffer_indices.size), dtype=np.int32),
        "zbuffer_visible_point_count": np.asarray(int(source_indices.size), dtype=np.int32),
    }


def load_calib_cached(calib_dir: str) -> Mapping[str, np.ndarray]:
    cached = _CALIB_CACHE.get(calib_dir)
    if cached is None:
        cached = load_raw_calibration(calib_dir)
        _CALIB_CACHE[calib_dir] = cached
    return cached


def verify_legacy_cache_exact(
    arrays: Mapping[str, np.ndarray],
    legacy_payload: Mapping[str, np.ndarray],
    sample_id: str,
) -> None:
    pixel_index = np.asarray(arrays[PIXEL_INDEX_KEY], dtype=np.int64)
    depth = np.asarray(arrays[PIXEL_DEPTH_KEY], dtype=np.float32)
    expected_pixel_index = np.asarray(legacy_payload["pixel_index"], dtype=np.int64)
    expected_depth = np.asarray(legacy_payload["depth"], dtype=np.float32)
    if not np.array_equal(pixel_index, expected_pixel_index):
        mismatch = int(np.nonzero(pixel_index != expected_pixel_index)[0][0]) if pixel_index.shape == expected_pixel_index.shape and pixel_index.size else -1
        raise ValueError(f"legacy pixel_index exact check failed for {sample_id} at row {mismatch}")
    if not np.array_equal(depth, expected_depth):
        if depth.shape == expected_depth.shape and depth.size:
            diff = np.abs(depth - expected_depth)
            mismatch = int(np.argmax(diff))
            max_err = float(diff[mismatch])
        else:
            mismatch = -1
            max_err = float("nan")
        raise ValueError(
            f"legacy depth exact check failed for {sample_id} at row {mismatch} max_err={max_err}"
        )


def read_npz_payload(path: Path, output_size: Tuple[int, int], feature_dim: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    arrays = load_npz_pixel_cache(path, output_size, feature_dim)
    metadata: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as payload:
        for key in payload.files:
            if key == PIXEL_FEATURE_KEY:
                metadata[key] = arrays[PIXEL_FEATURE_KEY]
            elif key == PIXEL_INDEX_KEY:
                metadata[key] = arrays[PIXEL_INDEX_KEY]
            elif key == PIXEL_DEPTH_KEY:
                metadata[key] = arrays[PIXEL_DEPTH_KEY]
            else:
                metadata[key] = np.asarray(payload[key])
    return arrays, metadata


def migrated_metadata(
    original_metadata: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    canonical_payload: Mapping[str, np.ndarray],
    legacy_payload: Mapping[str, np.ndarray],
    feature_dim: int,
    output_size: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    data = dict(original_metadata)
    data.update(
        {
            PIXEL_FEATURE_KEY: arrays[PIXEL_FEATURE_KEY],
            PIXEL_INDEX_KEY: arrays[PIXEL_INDEX_KEY],
            PIXEL_DEPTH_KEY: arrays[PIXEL_DEPTH_KEY],
            "format": np.asarray(PIXEL_CACHE_FORMAT),
            "image_height": np.asarray(int(output_size[0]), dtype=np.int32),
            "image_width": np.asarray(int(output_size[1]), dtype=np.int32),
            "feature_dim": np.asarray(int(feature_dim), dtype=np.int32),
            "encoder_point_count": canonical_payload["encoder_point_count"],
            "projected_point_count": canonical_payload["projected_point_count"],
            "raw_zbuffer_visible_point_count": canonical_payload["raw_zbuffer_visible_point_count"],
            "zbuffer_visible_point_count": canonical_payload["zbuffer_visible_point_count"],
            "source_point_index": canonical_payload["source_indices"].astype(np.int64, copy=False),
            "legacy_source_point_index": legacy_payload["source_indices"].astype(np.int64, copy=False),
            "projection_migration_format": np.asarray(MIGRATION_FORMAT),
            "legacy_projection_version": np.asarray(LEGACY_PROJECTION_VERSION),
            "projection_version": np.asarray(str(LIDAR_PROJECTION_VERSION)),
            "canonical_projection_version": np.asarray(str(LIDAR_PROJECTION_VERSION)),
        }
    )
    return data


def process_one(task: Tuple[int, Mapping[str, object], Mapping[str, object]]) -> Dict[str, object]:
    index, record, config = task
    args = SimpleNamespace(
        kitti_root=str(config.get("kitti_root", "")),
        path_rewrite=str(config.get("path_rewrite", "")),
    )
    sample_id = str(record["sample_id"])
    safe_id = safe_sample_id(sample_id)
    output_size = (int(config["image_height"]), int(config["image_width"]))
    feature_dim = int(config["feature_dim"])
    max_depth = float(config["max_depth"])
    in_path = Path(config["in_root"]) / f"{safe_id}.npz"
    out_path = Path(config["out_root"]) / f"{safe_id}.npz"
    if not in_path.is_file():
        raise FileNotFoundError(f"missing input cache for {sample_id}: {in_path}")
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite existing migrated cache: {out_path}")

    velodyne_path = rewrite_path(record["velodyne_path"], args)
    calib_dir = resolve_calib_dir(record, args)
    points = load_velodyne_points(velodyne_path)
    calib = load_calib_cached(str(calib_dir))

    arrays, metadata = read_npz_payload(in_path, output_size, feature_dim)
    legacy_payload = visible_projection_payload(
        points,
        calib,
        output_size,
        max_depth,
        legacy_project_velo_to_image,
        legacy_zbuffer_visible_point_indices,
    )
    verify_legacy_cache_exact(arrays, legacy_payload, sample_id)

    canonical_payload = visible_projection_payload(points, calib, output_size, max_depth, project_velo_to_image)
    try:
        remapped_features, legacy_rows = remap_features_by_source_index(
            arrays[PIXEL_FEATURE_KEY],
            legacy_payload["source_indices"],
            canonical_payload["source_indices"],
        )
    except MissingSourceFeaturesError as exc:
        return {
            "status": "needs_rebuild",
            "index": int(index),
            "sample_id": sample_id,
            "safe_id": safe_id,
            "missing_count": len(exc.missing_indices),
            "missing_source_indices": exc.missing_indices[:100],
            "record": dict(record),
        }

    migrated_arrays = validate_pixel_arrays(
        remapped_features,
        canonical_payload["pixel_index"],
        canonical_payload["depth"],
        output_size,
        feature_dim,
    )
    payload = migrated_metadata(
        metadata,
        migrated_arrays,
        canonical_payload,
        legacy_payload,
        feature_dim,
        output_size,
    )
    payload["legacy_feature_row_index"] = legacy_rows.astype(np.int64, copy=False)
    atomic_savez(out_path, **payload)
    return {
        "status": "migrated",
        "index": int(index),
        "sample_id": sample_id,
        "safe_id": safe_id,
        "rows": int(migrated_arrays[PIXEL_INDEX_KEY].shape[0]),
        "out_path": str(out_path),
        "record": dict(record),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair KITTI pixel cache projection by remapping legacy Utonia rows to canonical source-point pixels."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--in-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--kitti-root", default="")
    parser.add_argument("--path-rewrite", default="")
    parser.add_argument("--feature-dim", type=int, default=576)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--defer-manifest", action="append", default=[], help="Manifest(s) to pass through to rebuild_manifest without reading or writing NPZ files.")
    parser.add_argument("--rebuild-manifest-out", default="")
    parser.add_argument("--migrated-manifest-out", default="")
    parser.add_argument("--error-manifest-out", default="")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    temp_path = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    temp_path.unlink(missing_ok=True)
    try:
        with temp_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def result_manifest_records(results: Sequence[Mapping[str, object]], status: str) -> List[Mapping[str, object]]:
    rows = []
    for result in sorted(results, key=lambda item: int(item.get("index", 0))):
        if result.get("status") == status and isinstance(result.get("record"), dict):
            rows.append(result["record"])
    return rows


def main() -> None:
    args = parse_args()
    if np.__version__ != LEGACY_NUMPY_VERSION:
        raise SystemExit(
            f"This migration verifier must run under NumPy {LEGACY_NUMPY_VERSION}; found {np.__version__}"
        )
    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    if in_root == out_root:
        raise SystemExit("--in-root and --out-root must differ")
    out_root.mkdir(parents=True, exist_ok=True)

    all_records = list(enumerate(read_jsonl(args.manifest)))
    if int(args.limit) > 0:
        all_records = all_records[: int(args.limit)]
    seen_safe_ids = set()
    duplicate_safe_ids = set()
    for _, record in all_records:
        safe_id = safe_sample_id(record["sample_id"])
        if safe_id in seen_safe_ids:
            duplicate_safe_ids.add(safe_id)
        seen_safe_ids.add(safe_id)
    if duplicate_safe_ids:
        examples = ", ".join(sorted(duplicate_safe_ids)[:5])
        raise SystemExit(f"manifest contains duplicate sample ids; examples: {examples}")

    deferred_records_by_id = {}
    for manifest in args.defer_manifest or []:
        for record in read_jsonl(manifest):
            deferred_records_by_id[safe_sample_id(record["sample_id"])] = dict(record)
    records = []
    deferred_results: List[Mapping[str, object]] = []
    for index, record in all_records:
        safe_id = safe_sample_id(record["sample_id"])
        if safe_id in deferred_records_by_id:
            deferred_results.append(
                {
                    "status": "needs_rebuild",
                    "index": int(index),
                    "sample_id": str(record["sample_id"]),
                    "safe_id": safe_id,
                    "missing_count": None,
                    "deferred": True,
                    "record": dict(record),
                }
            )
        else:
            records.append((index, record))

    config = {
        "in_root": str(in_root),
        "out_root": str(out_root),
        "kitti_root": str(args.kitti_root),
        "path_rewrite": str(args.path_rewrite),
        "feature_dim": int(args.feature_dim),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "max_depth": float(args.max_depth),
    }
    tasks = [(index, record, config) for index, record in records]
    results: List[Mapping[str, object]] = list(deferred_results)
    errors: List[Mapping[str, object]] = []

    workers = max(1, int(args.workers))
    if workers == 1:
        for task in tasks:
            index, record, _ = task
            try:
                result = process_one(task)
                results.append(result)
                if len(results) == 1 or len(results) % max(1, int(args.progress_every)) == 0:
                    print(json.dumps(result, sort_keys=True), flush=True)
            except Exception as exc:
                error = {
                    "status": "error",
                    "index": int(index),
                    "sample_id": str(record.get("sample_id", "")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "record": dict(record),
                }
                errors.append(error)
                print(json.dumps(error, sort_keys=True), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_one, task): task for task in tasks}
            for future in as_completed(futures):
                index, record, _ = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if len(results) == 1 or len(results) % max(1, int(args.progress_every)) == 0:
                        print(json.dumps(result, sort_keys=True), flush=True)
                except Exception as exc:  # legacy verification failures land here and fail the run.
                    error = {
                        "status": "error",
                        "index": int(index),
                        "sample_id": str(record.get("sample_id", "")),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "record": dict(record),
                    }
                    errors.append(error)
                    print(json.dumps(error, sort_keys=True), flush=True)

    rebuild_rows = [result for result in results if result.get("status") == "needs_rebuild"]
    migrated_rows = [result for result in results if result.get("status") == "migrated"]
    rebuild_manifest_out = Path(args.rebuild_manifest_out) if args.rebuild_manifest_out else out_root / "rebuild_manifest.jsonl"
    migrated_manifest_out = Path(args.migrated_manifest_out) if args.migrated_manifest_out else out_root / "migrated_manifest.jsonl"
    error_manifest_out = Path(args.error_manifest_out) if args.error_manifest_out else out_root / "error_manifest.jsonl"
    write_jsonl(rebuild_manifest_out, result_manifest_records(rebuild_rows, "needs_rebuild"))
    write_jsonl(migrated_manifest_out, result_manifest_records(migrated_rows, "migrated"))
    if errors:
        write_jsonl(error_manifest_out, sorted(errors, key=lambda item: int(item.get("index", 0))))

    summary = {
        "complete": not errors,
        "records": len(all_records),
        "processed_records": len(tasks),
        "deferred_records": len(deferred_results),
        "migrated": len(migrated_rows),
        "needs_rebuild": len(rebuild_rows),
        "errors": len(errors),
        "out_root": str(out_root),
        "rebuild_manifest_out": str(rebuild_manifest_out),
        "migrated_manifest_out": str(migrated_manifest_out),
        "canonical_projection_version": str(LIDAR_PROJECTION_VERSION),
        "legacy_projection_version": LEGACY_PROJECTION_VERSION,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
