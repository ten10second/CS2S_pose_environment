#!/usr/bin/env python3
"""Visualize sparse static history reprojection for KITTI frame pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from dataloader.kitti_raw_lidar_utils import read_jsonl
from tools.temporal_history_geometry import consecutive_rows, rebase_kitti_path
from tools.temporal_static_geometry import build_static_pair


def _save_rgb(path: Path, rgb_hwc: np.ndarray) -> None:
    arr = np.clip(np.asarray(rgb_hwc, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def _save_gray(path: Path, values: np.ndarray, valid: np.ndarray | None = None) -> None:
    arr = np.asarray(values, dtype=np.float32)
    if valid is None:
        finite = np.isfinite(arr)
    else:
        finite = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.float32)
    if np.any(finite):
        vals = arr[finite]
        lo = float(vals.min())
        hi = float(vals.max())
        if hi > lo:
            out[finite] = (arr[finite] - lo) / (hi - lo)
        else:
            out[finite] = 1.0
    Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _save_unit_gray(path: Path, values: np.ndarray) -> None:
    arr = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="L").save(path)


def _load_current_gt(row: dict, kitti_root: str | None, image_size: Tuple[int, int]) -> np.ndarray:
    image_path = rebase_kitti_path(row["image_02_path"], kitti_root)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize(image_size, Image.BILINEAR)
    return np.asarray(rgb, dtype=np.float32) / 255.0


def _iter_pairs(rows: List[dict], start_sample_id: str | None, count: int) -> Iterable[Tuple[int, dict, dict]]:
    start = 0
    if start_sample_id:
        for idx, row in enumerate(rows):
            if row.get("sample_id") == start_sample_id:
                start = idx
                break
        else:
            raise ValueError(f"start sample id not found: {start_sample_id}")
    yielded = 0
    for idx in range(start, max(len(rows) - 1, 0)):
        prev, cur = rows[idx], rows[idx + 1]
        if not consecutive_rows(prev, cur):
            continue
        yield idx, prev, cur
        yielded += 1
        if yielded >= count:
            break


def _load_region_entries(path: str) -> List[dict]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        if "pairs" in payload and isinstance(payload["pairs"], list):
            return list(payload["pairs"])
        else:
            return [payload]
    elif isinstance(payload, list):
        return list(payload)
    else:
        raise ValueError("static region manifest must be a dict or list")


def _entry_prev_cur_ids(item: dict) -> Tuple[str | None, str | None]:
    prev_id = item.get("prev_sample_id") or item.get("previous_sample_id") or item.get("previous") or item.get("prev")
    cur_id = (
        item.get("cur_sample_id")
        or item.get("current_sample_id")
        or item.get("target_sample_id")
        or item.get("current")
        or item.get("cur")
    )
    return prev_id, cur_id


def _regions_from_entry(item: dict) -> dict | None:
    if "static_regions" not in item and not any(
        key in item
        for key in (
            "prev_regions",
            "prev_rects",
            "target_regions",
            "cur_regions",
            "target_rects",
            "cur_rects",
            "target",
        )
    ):
        return None
    static_regions = item.get("static_regions", item)
    return {
        "prev_regions": static_regions.get(
            "prev_regions",
            static_regions.get("prev_rects", static_regions.get("prev", static_regions.get("reference"))),
        ),
        "target_regions": static_regions.get(
            "target_regions",
            static_regions.get(
                "cur_regions",
                static_regions.get("target_rects", static_regions.get("cur_rects", static_regions.get("target", static_regions.get("current")))),
            ),
        ),
    }


def visualize_pair(
    prev: dict,
    cur: dict,
    out_dir: Path,
    kitti_root: str | None,
    image_size: Tuple[int, int],
    depth_tol_m: float,
    static_regions: dict | None = None,
) -> dict:
    out = build_static_pair(
        prev,
        cur,
        kitti_root=kitti_root,
        image_size=image_size,
        depth_tol_m=depth_tol_m,
        static_regions=static_regions,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_rgb = out["debug"]["prev_rgb"]
    cur_gt = _load_current_gt(cur, kitti_root, image_size)
    warped = out["warped_rgb"].transpose(1, 2, 0)
    support = out["support_mask"][0]
    strict = out["strict_mask"][0]
    confidence = out["confidence"][0]
    overlay = cur_gt.copy()
    overlay[support] = 0.5 * overlay[support] + 0.5 * warped[support]

    _save_rgb(out_dir / "prev_gt.png", prev_rgb)
    _save_rgb(out_dir / "current_gt_for_display.png", cur_gt)
    _save_rgb(out_dir / "warped_rgb.png", warped)
    _save_rgb(out_dir / "overlay_current_warped.png", overlay)
    _save_unit_gray(out_dir / "support_mask.png", support.astype(np.float32))
    _save_unit_gray(out_dir / "strict_mask.png", strict.astype(np.float32))
    _save_unit_gray(out_dir / "confidence.png", confidence)
    _save_gray(out_dir / "projected_depth.png", out["debug"]["projected_depth"], valid=support)
    _save_gray(out_dir / "target_depth.png", out["debug"]["target_depth"], valid=np.isfinite(out["debug"]["target_depth"]))
    _save_unit_gray(out_dir / "reference_roi_mask.png", out["debug"]["reference_roi_mask"].astype(np.float32))
    _save_unit_gray(out_dir / "target_roi_mask.png", out["debug"]["target_roi_mask"].astype(np.float32))

    diagnostics = dict(out["diagnostics"])
    diagnostics.update({"prev_sample_id": prev.get("sample_id"), "cur_sample_id": cur.get("sample_id")})
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--kitti-root", default="")
    parser.add_argument("--start-sample-id", default="")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--depth-tol-m", type=float, default=0.75)
    parser.add_argument("--static-regions-json", default="", help="Optional manual/static ROI manifest for offline diagnostics.")
    parser.add_argument("--pairs-json", default="", help="Alias for --static-regions-json with annotated frame pairs.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    rows = read_jsonl(args.manifest)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    region_entries = _load_region_entries(args.pairs_json or args.static_regions_json)
    summaries = []
    if region_entries:
        rows_by_id = {row.get("sample_id"): row for row in rows}
        missing = []
        for item in region_entries:
            prev_id, cur_id = _entry_prev_cur_ids(item)
            prev = rows_by_id.get(prev_id)
            cur = rows_by_id.get(cur_id)
            if prev is None or cur is None:
                missing.append({"previous": prev_id, "current": cur_id})
                continue
            if not consecutive_rows(prev, cur):
                raise RuntimeError(f"annotated pair is not consecutive: {prev_id} -> {cur_id}")
            safe_id = str(cur_id).replace("/", "__")
            pair_dir = out_root / f"{len(summaries):06d}_{safe_id}"
            summary = visualize_pair(
                prev,
                cur,
                pair_dir,
                args.kitti_root or None,
                (args.image_width, args.image_height),
                args.depth_tol_m,
                _regions_from_entry(item),
            )
            summary.update({"annotation_name": item.get("name"), "annotation_split": item.get("split")})
            summaries.append({"pair_dir": str(pair_dir), **summary})
            if len(summaries) >= args.count:
                break
        if missing:
            (out_root / "missing_pairs.json").write_text(json.dumps(missing, indent=2, sort_keys=True))
    else:
        for idx, prev, cur in _iter_pairs(rows, args.start_sample_id or None, args.count):
            safe_id = str(cur.get("sample_id", f"pair_{idx:06d}")).replace("/", "__")
            pair_dir = out_root / f"{idx:06d}_{safe_id}"
            summary = visualize_pair(
                prev,
                cur,
                pair_dir,
                args.kitti_root or None,
                (args.image_width, args.image_height),
                args.depth_tol_m,
            )
            summaries.append({"pair_dir": str(pair_dir), **summary})
    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))
    print(json.dumps({"out_dir": str(out_root), "pair_count": len(summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()
