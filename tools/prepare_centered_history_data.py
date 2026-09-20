#!/usr/bin/env python3
"""Prepare centered adjacent-frame dense history references.

This is a data-preparation utility only.  It selects natural consecutive KITTI
raw frame pairs from the original geofence train/heldout manifests, builds a
dense previous-RGB reprojection using source-only depth calibration plus
target LiDAR depth verification, and stores the source pixel index needed to
reconstruct the warp after synchronized color augmentation.

Target RGB is loaded only after the geometry/reference artifact has been
computed.  It is saved as a native training target for downstream loaders; it
is not used for depth calibration, depth inference, or reprojection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.temporal_dense_reprojection import (  # noqa: E402
    _sparse_depth_map,
    build_dense_reference,
    calibrate_depth,
)
from tools.temporal_history_geometry import (  # noqa: E402
    consecutive_rows,
    get_geometry,
    load_velodyne,
    rebase_kitti_path,
)


DEFAULT_SETTINGS = "/mnt/shizhm/CS2S_run_control/temporal_static_p0p2_20260919/base_settings.json"
DEFAULT_FIXED_PAIRS = "/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/stage2_20260919/references/pairs.json"
DEFAULT_OUT = "/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/centered_a1_20260920_data"
DEFAULT_VENDOR = "/mnt/shizhm/third_party/Depth-Anything-V2-probe-a561b849"
DEFAULT_CHECKPOINT = "/mnt/shizhm/models/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth"
DEFAULT_PREV_RUN512 = "/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/dense_reprojection_20260919/run512"
IMAGE_SIZE = (512, 128)  # width, height


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def atomic_savez_compressed(path: str | Path, **arrays: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp_npz = Path(str(tmp) + ".npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp_npz, path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_settings_args(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if "args" not in data or not isinstance(data["args"], dict):
        raise ValueError(f"{path} must contain an 'args' object")
    return dict(data["args"])


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_fixed_pairs(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        pairs = data.get("pairs", [])
    else:
        pairs = data
    if not isinstance(pairs, list):
        raise ValueError(f"{path} must contain a pair list or {{'pairs': [...]}}")
    out: List[Dict[str, Any]] = []
    for item in pairs:
        if not all(k in item for k in ("name", "split", "previous", "current")):
            raise ValueError(f"fixed pair missing required fields: {item}")
        out.append(dict(item))
    return out


def sample_key(row: Mapping[str, Any]) -> str:
    return str(row["sample_id"])


def drive_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return str(row.get("date", "")), str(row.get("drive", ""))


def sorted_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (dict(r) for r in rows),
        key=lambda r: (str(r.get("date", "")), str(r.get("drive", "")), int(r.get("frame_index", r.get("frame_id")))),
    )


def adjacent_candidates(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    ordered = sorted_rows(rows)
    by_drive: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_drive[drive_key(row)].append(row)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for items in by_drive.values():
        for prev, cur in zip(items, items[1:]):
            if consecutive_rows(prev, cur):
                pairs.append((prev, cur))
    return pairs


def _linspace_indices(n: int, k: int) -> List[int]:
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    values = np.linspace(0, n - 1, k)
    out: List[int] = []
    seen = set()
    for value in values:
        idx = int(round(float(value)))
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    idx = 0
    while len(out) < k and idx < n:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
        idx += 1
    return sorted(out)


def select_even_pairs(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    used_sample_ids: Optional[Iterable[str]] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Select broadly spaced adjacent pairs without reusing frames."""
    if count <= 0:
        return []
    used = set(used_sample_ids or [])
    candidates = adjacent_candidates(rows)
    by_drive: Dict[Tuple[str, str], List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for prev, cur in candidates:
        if sample_key(prev) in used or sample_key(cur) in used:
            continue
        by_drive[drive_key(prev)].append((prev, cur))
    drives = sorted(by_drive)
    total = sum(len(by_drive[d]) for d in drives)
    if total < count:
        raise ValueError(f"not enough adjacent candidates after exclusions: need {count}, have {total}")

    quotas: Dict[Tuple[str, str], int] = {}
    remainders: List[Tuple[float, Tuple[str, str]]] = []
    remaining = count
    for drive in drives:
        exact = count * (len(by_drive[drive]) / float(total))
        q = min(len(by_drive[drive]), int(np.floor(exact)))
        quotas[drive] = q
        remaining -= q
        remainders.append((exact - q, drive))
    for _, drive in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if quotas[drive] < len(by_drive[drive]):
            quotas[drive] += 1
            remaining -= 1

    selected: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    used_local = set(used)
    for drive in drives:
        items = by_drive[drive]
        for idx in _linspace_indices(len(items), quotas.get(drive, 0)):
            prev, cur = items[idx]
            if sample_key(prev) in used_local or sample_key(cur) in used_local:
                continue
            selected.append((prev, cur))
            used_local.add(sample_key(prev))
            used_local.add(sample_key(cur))

    if len(selected) < count:
        for prev, cur in candidates:
            if sample_key(prev) in used_local or sample_key(cur) in used_local:
                continue
            selected.append((prev, cur))
            used_local.add(sample_key(prev))
            used_local.add(sample_key(cur))
            if len(selected) >= count:
                break
    if len(selected) < count:
        raise ValueError(f"could only select {len(selected)} non-reused pairs, requested {count}")
    return selected[:count]


def make_pair_entry(name: str, split: str, prev: Mapping[str, Any], cur: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "split": split,
        "previous": sample_key(prev),
        "current": sample_key(cur),
        "date": str(prev.get("date", "")),
        "drive": str(prev.get("drive", "")),
        "previous_frame_index": int(prev.get("frame_index", prev.get("frame_id"))),
        "current_frame_index": int(cur.get("frame_index", cur.get("frame_id"))),
    }


def build_pair_plan(
    settings_args: Mapping[str, Any],
    fixed_pairs_path: str | Path,
    train_count: int,
    heldout_count: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    rows_by_split = {
        "train": load_jsonl(settings_args["train_manifest"]),
        "heldout": load_jsonl(settings_args["val_manifest"]),
    }
    rows_by_id = {split: {sample_key(row): row for row in rows} for split, rows in rows_by_split.items()}
    fixed_pairs = load_fixed_pairs(fixed_pairs_path)
    used = set()
    observation: List[Dict[str, Any]] = []
    for pair in fixed_pairs:
        source_split = "heldout" if str(pair["split"]).lower() in ("heldout", "val", "test") else "train"
        if pair["previous"] not in rows_by_id[source_split] or pair["current"] not in rows_by_id[source_split]:
            raise ValueError(f"fixed pair {pair['name']} not found in {source_split} manifest")
        prev = rows_by_id[source_split][pair["previous"]]
        cur = rows_by_id[source_split][pair["current"]]
        if not consecutive_rows(prev, cur):
            raise ValueError(f"fixed pair {pair['name']} is not consecutive")
        used.add(str(pair["previous"]))
        used.add(str(pair["current"]))
        entry = make_pair_entry("obs_" + str(pair["name"]), "observation", prev, cur)
        entry["source_split"] = source_split
        entry["fixed_name"] = str(pair["name"])
        observation.append(entry)

    train_sel = select_even_pairs(rows_by_split["train"], train_count, used)
    used_after_train = set(used)
    train_entries: List[Dict[str, Any]] = []
    for i, (prev, cur) in enumerate(train_sel):
        train_entries.append(make_pair_entry(f"train_{i:04d}", "train", prev, cur))
        used_after_train.add(sample_key(prev))
        used_after_train.add(sample_key(cur))

    heldout_sel = select_even_pairs(rows_by_split["heldout"], heldout_count, used_after_train)
    heldout_entries = [
        make_pair_entry(f"heldout_{i:04d}", "heldout", prev, cur)
        for i, (prev, cur) in enumerate(heldout_sel)
    ]
    for entry in train_entries + heldout_entries:
        entry["source_split"] = entry["split"]

    entries = observation + train_entries + heldout_entries
    stats = {
        "available_adjacent_candidates": {
            split: len(adjacent_candidates(rows))
            for split, rows in rows_by_split.items()
        },
        "selected_counts": {
            "observation": len(observation),
            "train": len(train_entries),
            "heldout": len(heldout_entries),
        },
        "fixed_sample_ids_excluded_from_train_heldout": sorted(used),
    }
    return entries, rows_by_id, stats


def rebased(row: Mapping[str, Any], key: str, kitti_root: str | Path) -> str:
    return rebase_kitti_path(row[key], kitti_root)


def load_rgb_uint8(path: str | Path, image_size: Tuple[int, int], expected_native_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        native_size = tuple(rgb.size)
        if expected_native_size is not None and native_size != expected_native_size:
            raise ValueError(f"image size {native_size} does not match calibration {expected_native_size}: {path}")
        resized = rgb.resize(image_size, Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8), native_size


def source_flat_index_from_uv(source_uv: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    uv = np.asarray(source_uv, dtype=np.float32)
    mask = np.asarray(support_mask, dtype=bool)
    if uv.ndim != 3 or uv.shape[2] != 2:
        raise ValueError("source_uv must have shape [H,W,2]")
    if mask.shape != uv.shape[:2]:
        raise ValueError("support_mask shape must match source_uv")
    height, width = mask.shape
    out = np.full((height, width), -1, dtype=np.int64)
    if not mask.any():
        return out
    sx = np.rint(uv[..., 0]).astype(np.int64)
    sy = np.rint(uv[..., 1]).astype(np.int64)
    inside = mask & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
    if not np.array_equal(inside, mask):
        bad = int((mask & ~inside).sum())
        raise ValueError(f"{bad} supported pixels have out-of-bounds source_uv")
    out[mask] = sy[mask] * width + sx[mask]
    return out


def reconstruct_warp_from_source_index(prev_rgb_uint8: np.ndarray, source_flat_index: np.ndarray) -> np.ndarray:
    rgb = np.asarray(prev_rgb_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("prev_rgb_uint8 must be uint8 HWC RGB")
    index = np.asarray(source_flat_index, dtype=np.int64)
    if index.shape != rgb.shape[:2]:
        raise ValueError("source_flat_index must have shape [H,W]")
    flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    out = np.zeros(rgb.shape, dtype=np.float32)
    mask = index >= 0
    out[mask] = flat[index[mask]]
    return out


def validate_source_index(prev_rgb_uint8: np.ndarray, warped_rgb: np.ndarray, support_mask: np.ndarray, source_flat_index: np.ndarray) -> Dict[str, Any]:
    recon = reconstruct_warp_from_source_index(prev_rgb_uint8, source_flat_index)
    support = np.asarray(support_mask, dtype=bool)
    target = np.asarray(warped_rgb, dtype=np.float32)
    if target.shape != recon.shape:
        raise ValueError("warped_rgb and reconstructed warp shapes differ")
    outside_nonzero = float(np.abs(recon[~support]).max()) if np.any(~support) else 0.0
    if outside_nonzero > 1e-7:
        raise ValueError("reconstructed warp is nonzero outside support")
    max_abs = float(np.abs(recon[support] - target[support]).max()) if support.any() else 0.0
    # PIL uint8 -> float and build_dense_reference source colors should match
    # exactly up to float32 division by 255.
    if max_abs > 1.0 / 255.0 + 1e-6:
        raise ValueError(f"source_flat_index reconstruction mismatch: {max_abs}")
    return {"source_index_reconstruction_max_abs": max_abs, "outside_support_max_abs": outside_nonzero}


class DepthModel:
    def __init__(self, vendor: str | Path, checkpoint: str | Path, device: str, input_size: int = 518) -> None:
        import torch

        self.torch = torch
        self.device = str(device)
        self.input_size = int(input_size)
        if self.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(self.device))
        sys.path.insert(0, str(Path(vendor) / "metric_depth"))
        from depth_anything_v2.dpt import DepthAnythingV2

        self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=80)
        self.model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"), strict=True)
        self.model = self.model.to(self.device).eval()

    def infer(self, native_rgb_uint8: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            depth = self.model.infer_image(native_rgb_uint8[:, :, ::-1].copy(), input_size=self.input_size)
        return np.asarray(depth, dtype=np.float32)


def depth_cache_path(cache_dir: Path, sample_id: str, checkpoint_sha: str) -> Path:
    key = hashlib.sha256((sample_id + "|" + checkpoint_sha).encode("utf-8")).hexdigest()[:24]
    return cache_dir / (key + ".npy")


def load_or_infer_depth(
    *,
    native_rgb_uint8: np.ndarray,
    sample_id: str,
    checkpoint_sha: str,
    cache_dir: Path,
    depth_model: Optional[DepthModel],
    vendor: str,
    checkpoint: str,
    device: str,
) -> Tuple[np.ndarray, DepthModel, str, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = depth_cache_path(cache_dir, sample_id, checkpoint_sha)
    if path.is_file():
        return np.load(path), depth_model if depth_model is not None else None, str(path), True  # type: ignore[return-value]
    if depth_model is None:
        depth_model = DepthModel(vendor, checkpoint, device)
    depth = depth_model.infer(native_rgb_uint8)
    np.save(path, depth.astype(np.float32))
    return depth.astype(np.float32), depth_model, str(path), False


def build_one_reference(
    entry: Mapping[str, Any],
    rows_by_id: Mapping[str, Mapping[str, Mapping[str, Any]]],
    settings_args: Mapping[str, Any],
    out_dir: Path,
    depth_cache: Path,
    checkpoint_sha: str,
    vendor: str,
    checkpoint: str,
    device: str,
    depth_model: Optional[DepthModel],
    image_size: Tuple[int, int] = IMAGE_SIZE,
    depth_tol_m: float = 0.75,
) -> Tuple[Optional[DepthModel], Dict[str, Any]]:
    split = str(entry.get("source_split", entry["split"]))
    prev = rows_by_id[split][entry["previous"]]
    cur = rows_by_id[split][entry["current"]]
    if not consecutive_rows(prev, cur):
        raise ValueError(f"{entry['name']} is not a consecutive pair")
    kitti_root = settings_args["kitti_root"]

    geom = get_geometry(rebased(prev, "calib_dir", kitti_root))
    pose = geom.relative_velo_pose(rebased(prev, "oxts_path", kitti_root), rebased(cur, "oxts_path", kitti_root))
    prev_points = load_velodyne(rebased(prev, "velodyne_path", kitti_root))
    cur_points = load_velodyne(rebased(cur, "velodyne_path", kitti_root))

    prev_rgb_uint8, native_size = load_rgb_uint8(rebased(prev, "image_02_path", kitti_root), image_size, geom.image_size)
    # Depth and geometry use previous RGB / source lidar / target lidar only.
    native_rgb_uint8 = np.asarray(Image.open(rebased(prev, "image_02_path", kitti_root)).convert("RGB"), dtype=np.uint8)
    native_depth, depth_model, depth_path, depth_cached = load_or_infer_depth(
        native_rgb_uint8=native_rgb_uint8,
        sample_id=entry["previous"],
        checkpoint_sha=checkpoint_sha,
        cache_dir=depth_cache,
        depth_model=depth_model,
        vendor=vendor,
        checkpoint=checkpoint,
        device=device,
    )
    camera_z_shift = float((np.linalg.inv(geom.p_rect_02[:, :3]) @ geom.p_rect_02[:, 3])[2])
    depth_raw = cv2.resize(native_depth, image_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) - camera_z_shift
    source_depth = _sparse_depth_map(prev_points, geom, image_size)
    target_depth = _sparse_depth_map(cur_points, geom, image_size)
    height, width = image_size[1], image_size[0]
    yy, xx = np.indices((height, width))
    fit_partition = ((xx * 73856093 + yy * 19349663) % 5) != 0
    depth_scaled, fit = calibrate_depth(depth_raw, source_depth, fit_partition)
    depth_anchored = np.where(np.isfinite(source_depth), source_depth, depth_scaled).astype(np.float32)
    prev_rgb_float = prev_rgb_uint8.astype(np.float32) / 255.0
    ref = build_dense_reference(
        prev_rgb_float,
        depth_anchored,
        prev_points,
        cur_points,
        geom,
        pose,
        depth_tol_m=depth_tol_m,
    )
    source_flat_index = source_flat_index_from_uv(ref["source_uv"], ref["support_mask"])
    reconstruction = validate_source_index(prev_rgb_uint8, ref["warped_rgb"], ref["support_mask"], source_flat_index)

    # Target RGB is not a construction input; it is loaded after geometry is complete.
    current_rgb_uint8, _ = load_rgb_uint8(rebased(cur, "image_02_path", kitti_root), image_size)

    folder = out_dir / str(entry["name"])
    folder.mkdir(parents=True, exist_ok=True)
    atomic_savez_compressed(
        folder / "reference.npz",
        warped_rgb=np.asarray(ref["warped_rgb"], dtype=np.float32),
        support_mask=np.asarray(ref["support_mask"], dtype=bool),
        measured_mask=np.asarray(ref["measured_mask"], dtype=bool),
        estimated_mask=np.asarray(ref["estimated_mask"], dtype=bool),
        target_conflict_mask=np.asarray(ref["target_conflict_mask"], dtype=bool),
        projected_depth=np.asarray(ref["projected_depth"], dtype=np.float32),
        source_uv=np.asarray(ref["source_uv"], dtype=np.float32),
        source_flat_index=source_flat_index,
        prev_rgb=prev_rgb_uint8,
        current_rgb=current_rgb_uint8,
        depth_raw=depth_raw.astype(np.float32),
        depth_scaled=depth_scaled.astype(np.float32),
        depth_anchored=depth_anchored.astype(np.float32),
        source_depth=source_depth.astype(np.float32),
        target_depth=target_depth.astype(np.float32),
        previous=str(entry["previous"]),
        current=str(entry["current"]),
        name=str(entry["name"]),
        split=str(entry["split"]),
        source_split=split,
    )
    record = {
        **dict(entry),
        "path": str(folder / "reference.npz"),
        "valid_pixels": int(np.asarray(ref["support_mask"]).sum()),
        "measured_pixels": int(np.asarray(ref["measured_mask"]).sum()),
        "estimated_pixels": int(np.asarray(ref["estimated_mask"]).sum()),
        "target_conflict_pixels": int(np.asarray(ref["target_conflict_mask"]).sum()),
        "ego_translation_m": float(np.linalg.norm(pose[:3, 3])),
        "native_image_size": native_size,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "depth_cache_path": depth_path,
        "depth_cache_hit": bool(depth_cached),
        "fit": fit,
        "dense_diagnostics": ref["diagnostics"],
        **reconstruction,
    }
    write_json(folder / "metrics.json", record)
    return depth_model, record


def read_existing_record(folder: Path) -> Optional[Dict[str, Any]]:
    path = folder / "metrics.json"
    if not path.is_file() or not (folder / "reference.npz").is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument("--fixed-pairs", default=DEFAULT_FIXED_PAIRS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--vendor", default=DEFAULT_VENDOR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--heldout-count", type=int, default=64)
    parser.add_argument("--width", type=int, default=IMAGE_SIZE[0])
    parser.add_argument("--height", type=int, default=IMAGE_SIZE[1])
    parser.add_argument("--depth-cache", default="")
    parser.add_argument("--depth-tol-m", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="optional debug limit over planned entries")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    image_size = (int(args.width), int(args.height))
    if image_size != IMAGE_SIZE:
        raise ValueError("this centered data contract is fixed at 512x128")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_cache = Path(args.depth_cache) if args.depth_cache else out_dir / "depth_cache"
    settings_args = load_settings_args(args.settings)
    entries, rows_by_id, plan_stats = build_pair_plan(settings_args, args.fixed_pairs, args.train_count, args.heldout_count)
    if args.limit:
        entries = entries[: int(args.limit)]
    checkpoint_sha = sha256_file(args.checkpoint)
    contract = {
        "reference_npz_schema": {
            "prev_rgb": "uint8 HWC [128,512,3], previous frame resized with PIL bilinear",
            "current_rgb": "uint8 HWC [128,512,3], loaded after geometry construction",
            "warped_rgb": "float32 HWC [128,512,3] in [0,1], zero outside support_mask",
            "source_flat_index": "int64 HW [128,512], -1 outside support; valid values index prev_rgb.reshape(-1,3)",
            "support_mask": "bool HW, measured_mask|estimated_mask",
            "measured_mask": "bool HW, source and target LiDAR depth verified",
            "estimated_mask": "bool HW, dense-depth projected and not target-conflicted",
        },
        "target_rgb_used_for_geometry": False,
        "depth_calibration": "source LiDAR only, robust global multiplicative median scale",
        "dynamic_instance_exclusion": "not applied; dependency is ego-motion depth consistency without instance exclusion",
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "settings": str(args.settings),
        "fixed_pairs": str(args.fixed_pairs),
        "checkpoint_sha256": checkpoint_sha,
        "vendor": str(args.vendor),
        "checkpoint": str(args.checkpoint),
        "device": str(args.device),
        "plan_stats": plan_stats,
    }
    write_json(out_dir / "contract.json", contract)
    write_json(out_dir / "done.json", {"complete": False, "status": "running", "expected_references": len(entries)})
    write_json(out_dir / "pairs.json", entries)
    write_json(out_dir / "observation_pairs.json", [e for e in entries if e["split"] == "observation"])
    write_json(out_dir / "train_pairs.json", [e for e in entries if e["split"] == "train"])
    write_json(out_dir / "heldout_pairs.json", [e for e in entries if e["split"] == "heldout"])

    progress_path = out_dir / "progress.jsonl"
    records: List[Dict[str, Any]] = []
    depth_model: Optional[DepthModel] = None
    for idx, entry in enumerate(entries):
        folder = out_dir / str(entry["name"])
        if not args.overwrite:
            existing = read_existing_record(folder)
            if existing is not None:
                records.append(existing)
                continue
        depth_model, record = build_one_reference(
            entry,
            rows_by_id,
            settings_args,
            out_dir,
            depth_cache,
            checkpoint_sha,
            str(args.vendor),
            str(args.checkpoint),
            str(args.device),
            depth_model,
            image_size=image_size,
            depth_tol_m=float(args.depth_tol_m),
        )
        record["index"] = idx
        records.append(record)
        with progress_path.open("a") as stream:
            stream.write(json.dumps(json_safe(record), ensure_ascii=False) + "\n")
        print(
            f"[{idx + 1}/{len(entries)}] {entry['name']} valid={record['valid_pixels']} "
            f"measured={record['measured_pixels']} estimated={record['estimated_pixels']} "
            f"cache={'hit' if record['depth_cache_hit'] else 'miss'}",
            flush=True,
        )

    by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[str(record["split"])].append(record)
    manifest = {
        **contract,
        "total_references": len(records),
        "counts": {split: len(items) for split, items in sorted(by_split.items())},
        "references": records,
        "disk_budget_note": "RGB stored as uint8; depth cache is shared by previous sample id and checkpoint hash.",
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "done.json", {"complete": True, "total_references": len(records), "counts": manifest["counts"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
