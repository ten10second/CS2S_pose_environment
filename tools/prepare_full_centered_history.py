#!/usr/bin/env python3
"""Prepare compact full natural-pair centered history references."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_centered_history_data import (  # noqa: E402
    DepthModel,
    IMAGE_SIZE,
    _sparse_depth_map,
    atomic_savez_compressed,
    build_dense_reference,
    calibrate_depth,
    consecutive_rows,
    get_geometry,
    json_safe,
    load_jsonl,
    load_rgb_uint8,
    load_settings_args,
    load_velodyne,
    rebase_kitti_path,
    sha256_file,
    source_flat_index_from_uv,
    validate_source_index,
    write_json,
)

DEFAULT_SETTINGS = "/mnt/shizhm/CS2S_run_control/temporal_centered_a1_20260920/settings.json"
DEFAULT_ABC_SELECTION = "/mnt/shizhm/CS2S_run_control/centered_decoder_abc_20260920/selection.json"
DEFAULT_ABC_DETAILS = "/mnt/shizhm/CS2S_run_control/centered_decoder_abc_20260920/selection_details.json"
DEFAULT_OLD_REFERENCE_ROOT = "/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/centered_a1_20260920_data"
DEFAULT_OUT_ROOT = "/mnt/shizhm/CS2S_run_control/centered_decoder_full_20260920"
DEFAULT_VENDOR = "/mnt/shizhm/third_party/Depth-Anything-V2-probe-a561b849"
DEFAULT_CHECKPOINT = "/mnt/shizhm/models/depth_anything_v2/depth_anything_v2_metric_vkitti_vits.pth"
COMPACT_KEYS = ("prev_rgb", "current_rgb", "source_flat_index", "support_mask", "valid_mask", "measured_mask", "estimated_mask")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
    os.replace(tmp, path)


def sample_key(row: Mapping[str, Any]) -> str:
    return str(row["sample_id"])


def drive_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return str(row.get("date", "")), str(row.get("drive", ""))


def sorted_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(r) for r in rows), key=lambda r: (
        str(r.get("date", "")), str(r.get("drive", "")), int(r.get("frame_index", r.get("frame_id")))
    ))


def adjacent_candidates(rows: Sequence[Mapping[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_drive: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted_rows(rows):
        by_drive.setdefault(drive_key(row), []).append(row)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for items in by_drive.values():
        for prev, cur in zip(items, items[1:]):
            if consecutive_rows(prev, cur):
                pairs.append((prev, cur))
    return pairs


def pair_identity(previous: str, current: str) -> str:
    return hashlib.sha1((previous + "|" + current).encode("utf-8")).hexdigest()[:16]


def load_abc_entries(selection_path: str | Path, details_path: str | Path) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    selection = json.loads(Path(selection_path).read_text())
    details = json.loads(Path(details_path).read_text()).get("entries", {})
    for split in ("train", "heldout", "observation"):
        if split not in selection:
            raise ValueError(f"ABC selection missing {split}")
        for name in selection[split]:
            if name not in details:
                raise ValueError(f"ABC details missing {name}")
    return {k: list(v) for k, v in selection.items()}, {str(k): dict(v) for k, v in details.items()}


def entry_from_rows(name: str, split: str, prev: Mapping[str, Any], cur: Mapping[str, Any], source_split: str) -> dict[str, Any]:
    return {
        "name": name,
        "split": split,
        "source_split": source_split,
        "previous": sample_key(prev),
        "current": sample_key(cur),
        "date": str(prev.get("date", "")),
        "drive": str(prev.get("drive", "")),
        "previous_frame_index": int(prev.get("frame_index", prev.get("frame_id"))),
        "current_frame_index": int(cur.get("frame_index", cur.get("frame_id"))),
    }


def with_reference_path(entry: Mapping[str, Any], reference_root: Path) -> dict[str, Any]:
    out = dict(entry)
    out["pair_id"] = pair_identity(str(entry["previous"]), str(entry["current"]))
    out["reference_npz"] = str(reference_root / str(out["name"]) / "reference.npz")
    return out


def canonicalize_eval_reference_paths(
    eval_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    canonical_by_pair: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for group, items in eval_groups.items():
        out[group] = []
        for item in items:
            fixed = dict(item)
            pair = (str(fixed["previous"]), str(fixed["current"]))
            canonical = canonical_by_pair.get(pair)
            if canonical is not None:
                fixed["reference_npz"] = str(canonical["reference_npz"])
                fixed["canonical_reference_name"] = str(canonical["name"])
            out[group].append(fixed)
    return out


def scan_old_references(root: Path) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    if not root.is_dir():
        return out
    for metrics in root.glob("*/metrics.json"):
        try:
            row = json.loads(metrics.read_text())
        except Exception:
            continue
        ref = metrics.parent / "reference.npz"
        if ref.is_file() and row.get("previous") and row.get("current"):
            out[(str(row["previous"]), str(row["current"]))] = ref
    return out


def plan_entries(args: argparse.Namespace) -> dict[str, Any]:
    settings_args = load_settings_args(args.settings)
    train_rows = load_jsonl(settings_args["train_manifest"])
    heldout_rows = load_jsonl(settings_args["val_manifest"])
    rows_by_split = {
        "train": {sample_key(r): r for r in train_rows},
        "heldout": {sample_key(r): r for r in heldout_rows},
    }
    selection, abc_entries = load_abc_entries(args.abc_selection, args.abc_details)
    observation_frames: set[str] = set()
    for name in selection["observation"]:
        entry = abc_entries[name]
        observation_frames.add(str(entry["previous"]))
        observation_frames.add(str(entry["current"]))
    candidates = [
        (prev, cur) for prev, cur in adjacent_candidates(train_rows)
        if sample_key(prev) not in observation_frames and sample_key(cur) not in observation_frames
    ]
    order = list(range(len(candidates)))
    random.Random(int(args.seed)).shuffle(order)
    full_train = []
    for ordinal, source_order in enumerate(order):
        prev, cur = candidates[source_order]
        entry = entry_from_rows(f"full_train_{ordinal:06d}", "train", prev, cur, "train")
        entry["source_order"] = int(source_order)
        entry["shuffle_order"] = int(ordinal)
        full_train.append(entry)

    eval_groups: dict[str, list[dict[str, Any]]] = {}
    for group, names in (("eval_train", selection["train"]), ("heldout", selection["heldout"]), ("observation", selection["observation"])):
        eval_groups[group] = []
        for name in names:
            base = dict(abc_entries[name])
            source_split = str(base.get("source_split") or ("heldout" if base.get("split") == "heldout" else "train"))
            if base["previous"] not in rows_by_split[source_split] or base["current"] not in rows_by_split[source_split]:
                raise ValueError(f"ABC eval pair {name} not found in {source_split} manifest")
            base["name"] = name
            base["split"] = group
            base["source_split"] = source_split
            eval_groups[group].append(base)

    reference_root = Path(args.out_root) / "references"
    full_train = [with_reference_path(e, reference_root) for e in full_train]
    canonical_by_pair = {(str(item["previous"]), str(item["current"])): item for item in full_train}
    eval_groups = {k: [with_reference_path(e, reference_root) for e in v] for k, v in eval_groups.items()}
    eval_groups = canonicalize_eval_reference_paths(eval_groups, canonical_by_pair)
    produce_items = full_train + eval_groups["eval_train"] + eval_groups["heldout"] + eval_groups["observation"]
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in produce_items:
        unique.setdefault((str(item["previous"]), str(item["current"])), item)
    old_map = scan_old_references(Path(args.old_reference_root))
    stats = {
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "train_adjacent_candidates": len(adjacent_candidates(train_rows)),
        "heldout_adjacent_candidates": len(adjacent_candidates(heldout_rows)),
        "full_train_pairs_after_observation_exclusion": len(full_train),
        "observation_pairs": len(eval_groups["observation"]),
        "observation_frames_excluded_from_train": len(observation_frames),
        "eval_train_pairs": len(eval_groups["eval_train"]),
        "eval_heldout_pairs": len(eval_groups["heldout"]),
        "unique_reference_pairs_to_materialize": len(unique),
        "old_reference_pairs_reusable": sum(1 for key in unique if key in old_map),
        "compact_schema_keys": list(COMPACT_KEYS),
    }
    return {"settings_args": settings_args, "rows_by_split": rows_by_split, "full_train": full_train, "eval_groups": eval_groups, "produce_items": list(unique.values()), "stats": stats}


def validate_compact_arrays(arrays: Mapping[str, Any]) -> None:
    prev = np.asarray(arrays["prev_rgb"])
    cur = np.asarray(arrays["current_rgb"])
    index = np.asarray(arrays["source_flat_index"])
    support = np.asarray(arrays["support_mask"])
    if prev.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3) or prev.dtype != np.uint8:
        raise ValueError("prev_rgb must be uint8 [128,512,3]")
    if cur.shape != prev.shape or cur.dtype != np.uint8:
        raise ValueError("current_rgb must be uint8 [128,512,3]")
    if index.shape != prev.shape[:2] or index.dtype != np.int32:
        raise ValueError("source_flat_index must be int32 [128,512]")
    if support.shape != prev.shape[:2] or support.dtype != np.bool_:
        raise ValueError("support_mask must be bool [128,512]")
    if not np.array_equal(np.asarray(arrays["valid_mask"], dtype=bool), support):
        raise ValueError("valid_mask must match support_mask")
    if np.any(index[support] < 0) or np.any(index[support] >= IMAGE_SIZE[0] * IMAGE_SIZE[1]):
        raise ValueError("valid source_flat_index out of range")


def compact_arrays_from_npz(path: Path) -> dict[str, Any]:
    src = np.load(path)
    support = np.asarray(src["support_mask"], dtype=bool)
    arrays = {
        "prev_rgb": np.asarray(src["prev_rgb"], dtype=np.uint8),
        "current_rgb": np.asarray(src["current_rgb"], dtype=np.uint8),
        "source_flat_index": np.asarray(src["source_flat_index"], dtype=np.int32),
        "support_mask": support,
        "valid_mask": support,
        "measured_mask": np.asarray(src["measured_mask"], dtype=bool),
        "estimated_mask": np.asarray(src["estimated_mask"], dtype=bool),
    }
    validate_compact_arrays(arrays)
    return arrays


def write_compact_reference(path: Path, arrays: Mapping[str, Any], entry: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    payload.update(previous=str(entry["previous"]), current=str(entry["current"]), name=str(entry["name"]), split=str(entry["split"]), source_split=str(entry["source_split"]), compact_schema_version="full_centered_history_v1")
    atomic_savez_compressed(path, **payload)
    record = {**dict(entry), "reference_npz": str(path), "bytes": int(path.stat().st_size), "valid_pixels": int(np.asarray(arrays["support_mask"], dtype=bool).sum()), "measured_pixels": int(np.asarray(arrays["measured_mask"], dtype=bool).sum()), "estimated_pixels": int(np.asarray(arrays["estimated_mask"], dtype=bool).sum()), **dict(provenance)}
    write_json(path.parent / "metrics.json", record)
    return record


def rebased(row: Mapping[str, Any], key: str, kitti_root: str | Path) -> str:
    return rebase_kitti_path(row[key], kitti_root)


def build_compact_from_geometry(entry: Mapping[str, Any], rows_by_split: Mapping[str, Mapping[str, Mapping[str, Any]]], settings_args: Mapping[str, Any], depth_model: Optional[DepthModel], vendor: str, checkpoint: str, device: str, depth_tol_m: float) -> tuple[DepthModel, dict[str, Any], dict[str, Any]]:
    source_split = str(entry["source_split"])
    prev = rows_by_split[source_split][str(entry["previous"])]
    cur = rows_by_split[source_split][str(entry["current"])]
    if not consecutive_rows(prev, cur):
        raise ValueError(f"{entry['name']} is not consecutive")
    kitti_root = settings_args["kitti_root"]
    geom = get_geometry(rebased(prev, "calib_dir", kitti_root))
    pose = geom.relative_velo_pose(rebased(prev, "oxts_path", kitti_root), rebased(cur, "oxts_path", kitti_root))
    prev_points = load_velodyne(rebased(prev, "velodyne_path", kitti_root))
    cur_points = load_velodyne(rebased(cur, "velodyne_path", kitti_root))
    prev_rgb, native_size = load_rgb_uint8(rebased(prev, "image_02_path", kitti_root), IMAGE_SIZE, geom.image_size)
    current_rgb, _ = load_rgb_uint8(rebased(cur, "image_02_path", kitti_root), IMAGE_SIZE)
    native_rgb = np.asarray(Image.open(rebased(prev, "image_02_path", kitti_root)).convert("RGB"), dtype=np.uint8)
    if depth_model is None:
        depth_model = DepthModel(vendor, checkpoint, device)
    native_depth = depth_model.infer(native_rgb)
    camera_z_shift = float((np.linalg.inv(geom.p_rect_02[:, :3]) @ geom.p_rect_02[:, 3])[2])
    depth_raw = cv2.resize(native_depth, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) - camera_z_shift
    source_depth = _sparse_depth_map(prev_points, geom, IMAGE_SIZE)
    target_depth = _sparse_depth_map(cur_points, geom, IMAGE_SIZE)
    yy, xx = np.indices((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    fit_partition = ((xx * 73856093 + yy * 19349663) % 5) != 0
    depth_scaled, fit = calibrate_depth(depth_raw, source_depth, fit_partition)
    depth_anchored = np.where(np.isfinite(source_depth), source_depth, depth_scaled).astype(np.float32)
    ref = build_dense_reference(prev_rgb.astype(np.float32) / 255.0, depth_anchored, prev_points, cur_points, geom, pose, depth_tol_m=depth_tol_m)
    source_flat_index = source_flat_index_from_uv(ref["source_uv"], ref["support_mask"]).astype(np.int32)
    reconstruction = validate_source_index(prev_rgb, ref["warped_rgb"], ref["support_mask"], source_flat_index.astype(np.int64))
    support = np.asarray(ref["support_mask"], dtype=bool)
    arrays = {"prev_rgb": prev_rgb, "current_rgb": current_rgb, "source_flat_index": source_flat_index, "support_mask": support, "valid_mask": support, "measured_mask": np.asarray(ref["measured_mask"], dtype=bool), "estimated_mask": np.asarray(ref["estimated_mask"], dtype=bool)}
    validate_compact_arrays(arrays)
    provenance = {"source": "built_dense_depth", "native_image_size": native_size, "ego_translation_m": float(np.linalg.norm(pose[:3, 3])), "fit": fit, "dense_diagnostics": ref["diagnostics"], **reconstruction}
    return depth_model, arrays, provenance


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_plan(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = plan_entries(args)
    atomic_jsonl(out_root / "train_pairs.jsonl", plan["full_train"])
    write_json(out_root / "train_manifest.json", {"items": plan["full_train"], "count": len(plan["full_train"]), "ordered_like_producer": True})
    atomic_jsonl(out_root / "produce_manifest.jsonl", plan["produce_items"])
    write_json(out_root / "eval_selection.json", plan["eval_groups"])
    write_json(out_root / "selection.json", {"train": [x["name"] for x in plan["full_train"]], "eval_train": [x["name"] for x in plan["eval_groups"]["eval_train"]], "heldout": [x["name"] for x in plan["eval_groups"]["heldout"]], "observation": [x["name"] for x in plan["eval_groups"]["observation"]]})
    contract = {"schema_version": "full_centered_history_plan_v1", "settings": str(args.settings), "abc_selection": str(args.abc_selection), "abc_details": str(args.abc_details), "old_reference_root": str(args.old_reference_root), "out_root": str(out_root), "reference_root": str(out_root / "references"), "seed": int(args.seed), "compact_npz_schema": {key: "required" for key in COMPACT_KEYS}, "compact_note": "No depth_raw/depth_scaled/depth_anchored/source_uv/projected_depth arrays are retained.", "stats": plan["stats"]}
    write_json(out_root / "contract.json", contract)
    write_json(out_root / "done.json", {"done": False, "stage": "planned", "stats": plan["stats"]})
    return contract


def produce(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.out_root)
    if not (out_root / "produce_manifest.jsonl").is_file() or args.refresh_plan:
        write_plan(args)
    settings_args = load_settings_args(args.settings)
    train_rows = load_jsonl(settings_args["train_manifest"])
    heldout_rows = load_jsonl(settings_args["val_manifest"])
    rows_by_split = {"train": {sample_key(r): r for r in train_rows}, "heldout": {sample_key(r): r for r in heldout_rows}}
    items = read_jsonl(out_root / "produce_manifest.jsonl")
    if args.limit:
        items = items[: int(args.limit)]
    old_map = scan_old_references(Path(args.old_reference_root))
    checkpoint_sha = sha256_file(args.checkpoint)
    write_json(out_root / "producer_args.json", {**vars(args), "checkpoint_sha256": checkpoint_sha})
    write_json(out_root / "done.json", {"done": False, "stage": "producing", "expected": len(items), "started": time.time()})
    records = []
    depth_model: Optional[DepthModel] = None
    started = time.time()
    with (out_root / "progress.jsonl").open("a") as progress:
        for index, entry in enumerate(items):
            ref_path = Path(entry["reference_npz"])
            metrics_path = ref_path.parent / "metrics.json"
            free_gb = shutil.disk_usage(out_root).free / (1024 ** 3)
            if free_gb < float(args.min_free_gb):
                failure = {"error": "free space below guard", "free_gb": round(free_gb, 3), "min_free_gb": float(args.min_free_gb), "index": index, "name": entry["name"]}
                write_json(out_root / "failed.json", failure)
                raise RuntimeError(json.dumps(failure, sort_keys=True))
            if metrics_path.is_file() and ref_path.is_file() and not args.overwrite:
                record = json.loads(metrics_path.read_text())
                record["status"] = "exists"
            else:
                pair = (str(entry["previous"]), str(entry["current"]))
                if pair in old_map and not args.rebuild_existing:
                    arrays = compact_arrays_from_npz(old_map[pair])
                    record = write_compact_reference(ref_path, arrays, entry, {"source": "reused_old_reference", "old_reference_npz": str(old_map[pair])})
                else:
                    depth_model, arrays, provenance = build_compact_from_geometry(entry, rows_by_split, settings_args, depth_model, args.vendor, args.checkpoint, args.device, float(args.depth_tol_m))
                    record = write_compact_reference(ref_path, arrays, entry, provenance)
            record["index"] = index
            record["elapsed_seconds"] = round(time.time() - started, 2)
            records.append(record)
            progress.write(json.dumps(json_safe(record), sort_keys=True) + "\n")
            progress.flush()
            if index < 3 or (index + 1) % int(args.log_every) == 0:
                print(json.dumps({"index": index, "name": entry["name"], "source": record.get("source"), "bytes": record.get("bytes"), "elapsed": record["elapsed_seconds"]}, sort_keys=True), flush=True)
    total_bytes = sum(int(r.get("bytes", 0)) for r in records)
    summary = {"done": True, "stage": "complete", "total_references": len(records), "total_bytes": total_bytes, "total_gb": round(total_bytes / (1024 ** 3), 4), "reused_old_reference": sum(1 for r in records if r.get("source") == "reused_old_reference"), "built_dense_depth": sum(1 for r in records if r.get("source") == "built_dense_depth"), "exists": sum(1 for r in records if r.get("status") == "exists"), "finished": time.time()}
    write_json(out_root / "reference_manifest.json", {"items": records, "summary": summary})
    write_json(out_root / "done.json", summary)
    return summary


def selftest() -> None:
    eval_groups = {"eval_train": [{"name": "old_eval", "previous": "p", "current": "c", "reference_npz": "/tmp/eval/reference.npz"}]}
    canonical = {("p", "c"): {"name": "full_train_000001", "reference_npz": "/tmp/full/reference.npz"}}
    fixed = canonicalize_eval_reference_paths(eval_groups, canonical)["eval_train"][0]
    assert fixed["reference_npz"] == "/tmp/full/reference.npz"
    assert fixed["canonical_reference_name"] == "full_train_000001"

    arrays = {"prev_rgb": np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8), "current_rgb": np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8), "source_flat_index": np.full((IMAGE_SIZE[1], IMAGE_SIZE[0]), -1, dtype=np.int32), "support_mask": np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool), "valid_mask": np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool), "measured_mask": np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool), "estimated_mask": np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool)}
    arrays["support_mask"][0, 0] = True
    arrays["valid_mask"][0, 0] = True
    arrays["source_flat_index"][0, 0] = 0
    validate_compact_arrays(arrays)
    bad = dict(arrays)
    bad["valid_mask"] = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool)
    try:
        validate_compact_arrays(bad)
    except ValueError as exc:
        assert "valid_mask" in str(exc)
    else:
        raise AssertionError("invalid valid_mask was accepted")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "produce", "selftest"), default="plan")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS)
    parser.add_argument("--abc-selection", default=DEFAULT_ABC_SELECTION)
    parser.add_argument("--abc-details", default=DEFAULT_ABC_DETAILS)
    parser.add_argument("--old-reference-root", default=DEFAULT_OLD_REFERENCE_ROOT)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--vendor", default=DEFAULT_VENDOR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--depth-tol-m", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-existing", action="store_true")
    parser.add_argument("--refresh-plan", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "selftest":
        selftest()
        print("selftest ok")
        return 0
    if args.mode == "plan":
        contract = write_plan(args)
        print(json.dumps(contract["stats"], indent=2, sort_keys=True))
        return 0
    try:
        summary = produce(args)
    except Exception as exc:
        out_root = Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        failed = {"error": str(exc), "stage": "produce", "time": time.time()}
        if not (out_root / "failed.json").is_file():
            write_json(out_root / "failed.json", failed)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
