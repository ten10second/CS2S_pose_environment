"""Audit sparse temporal-history geometry on KITTI manifest pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for item in (str(TOOLS_DIR), str(REPO_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from temporal_history_geometry import (  # noqa: E402
    build_pair_geometry,
    rebase_kitti_path,
    stream_consecutive_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit sparse current->previous temporal-history geometry.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kitti-root", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-pairs", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--grid-h", type=int, default=16)
    parser.add_argument("--grid-w", type=int, default=64)
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--depth-tol-m", type=float, default=0.75)
    parser.add_argument("--overlay-width", type=int, default=512)
    parser.add_argument("--overlay-height", type=int, default=128)
    return parser.parse_args()


def read_jsonl(path: str | Path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_sample_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "__")


def resize_image(path: str | Path, size):
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.BILINEAR)


def load_rgb_array(path: str | Path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def draw_current_overlay(image: Image.Image, geom_result: dict) -> Image.Image:
    out = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    h, w = geom_result["history_valid"].shape
    sx = out.size[0] / float(w)
    sy = out.size[1] / float(h)
    valid = geom_result["history_valid"]
    ground = geom_result["ground_proxy"]
    for y in range(h):
        for x in range(w):
            if not valid[y, x]:
                continue
            color = (32, 220, 80, 210) if ground[y, x] else (255, 160, 32, 220)
            draw.rectangle((x * sx, y * sy, (x + 1) * sx - 1, (y + 1) * sy - 1), outline=color, width=1)
    return out.convert("RGB")


def draw_previous_overlay(image: Image.Image, geom_result: dict) -> Image.Image:
    out = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(out)
    h, w = geom_result["history_valid"].shape
    sx = out.size[0] / float(w)
    sy = out.size[1] / float(h)
    valid = geom_result["history_valid"]
    grid_px = geom_result["history_grid_px"]
    for y, x in np.argwhere(valid):
        px = (float(grid_px[y, x, 0]) + 0.5) * sx
        py = (float(grid_px[y, x, 1]) + 0.5) * sy
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 64, 64, 220))
        draw.line((px, py, (x + 0.5) * sx, (y + 0.5) * sy), fill=(255, 64, 64, 120), width=1)
    return out.convert("RGB")


def save_panel(path: Path, prev_row: dict, cur_row: dict, geom_result: dict, args) -> None:
    size = (args.overlay_width, args.overlay_height)
    prev_img = resize_image(rebase_kitti_path(prev_row["image_02_path"], args.kitti_root), size)
    cur_img = resize_image(rebase_kitti_path(cur_row["image_02_path"], args.kitti_root), size)
    cur_overlay = draw_current_overlay(cur_img, geom_result)
    prev_overlay = draw_previous_overlay(prev_img, geom_result)
    label_h = 22
    canvas = Image.new("RGB", (size[0] * 2, size[1] + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(cur_overlay, (0, label_h))
    canvas.paste(prev_overlay, (size[0], label_h))
    draw.text((4, 4), "current valid cells: green=ground proxy, orange=non-ground proxy", fill=(0, 0, 0))
    draw.text((size[0] + 4, 4), "previous lookup coordinates, red dots", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def add_rgb_error_metrics(metrics: dict, prev_row: dict, cur_row: dict, geom_result: dict, args) -> dict:
    valid = geom_result["history_valid"]
    if not np.any(valid):
        metrics.update(
            {
                "rgb_error_mapped_mean": 0.0,
                "rgb_error_identity_mean": 0.0,
                "rgb_error_identity_minus_mapped_mean": 0.0,
                "rgb_error_num_points": 0,
            }
        )
        return metrics

    prev_img = load_rgb_array(rebase_kitti_path(prev_row["image_02_path"], args.kitti_root))
    cur_img = load_rgb_array(rebase_kitti_path(cur_row["image_02_path"], args.kitti_root))
    h, w = valid.shape
    prev_h, prev_w = prev_img.shape[:2]
    cur_h, cur_w = cur_img.shape[:2]
    cur_xy = geom_result["current_point_image_xy"][valid]
    prev_xy = geom_result["previous_point_image_xy"][valid]

    def gather(img, image_xy, img_w, img_h):
        x = np.clip(np.round(image_xy[:, 0]).astype(np.int64), 0, img_w - 1)
        y = np.clip(np.round(image_xy[:, 1]).astype(np.int64), 0, img_h - 1)
        return img[y, x]

    cur_rgb = gather(cur_img, cur_xy, cur_w, cur_h)
    mapped_rgb = gather(prev_img, prev_xy, prev_w, prev_h)
    identity_rgb = gather(prev_img, cur_xy, prev_w, prev_h)
    mapped_err = np.abs(cur_rgb - mapped_rgb).mean(axis=1)
    identity_err = np.abs(cur_rgb - identity_rgb).mean(axis=1)
    metrics.update(
        {
            "rgb_error_mapped_mean": float(mapped_err.mean()),
            "rgb_error_identity_mean": float(identity_err.mean()),
            "rgb_error_identity_minus_mapped_mean": float((identity_err - mapped_err).mean()),
            "rgb_error_num_points": int(len(mapped_err)),
        }
    )
    return metrics


def summarize(metrics):
    if not metrics:
        return {}
    keys = [
        "coverage_all",
        "coverage_of_current_lidar",
        "coverage_ground_proxy_all",
        "coverage_non_ground_proxy_all",
        "coverage_ground_proxy",
        "coverage_non_ground_proxy",
        "mean_reprojection_vs_identity_cells",
        "p95_reprojection_vs_identity_cells",
        "prev_to_cur_translation_m",
        "rgb_error_mapped_mean",
        "rgb_error_identity_mean",
        "rgb_error_identity_minus_mapped_mean",
    ]
    summary = {"num_pairs": len(metrics)}
    for key in keys:
        values = np.asarray([m[key] for m in metrics], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_min"] = float(values.min())
        summary[f"{key}_max"] = float(values.max())
    motion_pairs = [
        m for m in metrics
        if m["prev_to_cur_translation_m"] >= 0.2 and m["rgb_error_num_points"] > 0
    ]
    if motion_pairs:
        rgb_improvements = np.asarray(
            [m["rgb_error_identity_minus_mapped_mean"] for m in motion_pairs],
            dtype=np.float64,
        )
        median_rgb_improvement = float(np.median(rgb_improvements))
    else:
        median_rgb_improvement = 0.0
    summary.update(
        {
            "pilot_gate_valid_coverage_min": 0.05,
            "pilot_gate_non_ground_proxy_all_min": 0.05,
            "pilot_gate_motion_pair_min_count": 16,
            "pilot_gate_motion_pair_count": len(motion_pairs),
            "pilot_gate_rgb_identity_minus_mapped_median": median_rgb_improvement,
            "pilot_gate_valid_coverage_pass": summary["coverage_all_mean"] >= 0.05,
            "pilot_gate_non_ground_proxy_all_pass": summary["coverage_non_ground_proxy_all_mean"] >= 0.05,
            "pilot_gate_rgb_motion_pass": len(motion_pairs) >= 16 and median_rgb_improvement > 0.0,
        }
    )
    summary["note"] = "ground/non-ground is a geometric height proxy, not semantic facade labels"
    return summary


def select_pairs_round_robin(rows, start_index, stride, num_pairs):
    grouped = {}
    for prev_row, cur_row in stream_consecutive_pairs(rows):
        key = (cur_row.get("date", ""), cur_row.get("drive", ""))
        grouped.setdefault(key, []).append((prev_row, cur_row))
    selected = []
    keys = sorted(grouped)
    pos = int(start_index)
    stride = max(int(stride), 1)
    while keys and (num_pairs <= 0 or len(selected) < num_pairs):
        progressed = False
        for key in keys:
            bucket = grouped[key]
            if pos >= len(bucket):
                continue
            selected.append(bucket[pos])
            progressed = True
            if num_pairs > 0 and len(selected) >= num_pairs:
                break
        if not progressed:
            break
        pos += stride
    return selected


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)
    pairs = select_pairs_round_robin(rows, args.start_index, args.stride, args.num_pairs)
    if not pairs:
        raise RuntimeError("no consecutive manifest pairs selected")

    metrics_path = out_dir / "geometry_metrics.jsonl"
    all_metrics = []
    with metrics_path.open("w") as handle:
        for pair_index, (prev_row, cur_row) in enumerate(pairs):
            result = build_pair_geometry(
                prev_row,
                cur_row,
                kitti_root=args.kitti_root,
                grid=(args.grid_h, args.grid_w),
                max_range=args.max_range,
                depth_tol_m=args.depth_tol_m,
            )
            metrics = dict(result["metrics"])
            metrics["pair_index"] = pair_index
            metrics = add_rgb_error_metrics(metrics, prev_row, cur_row, result, args)
            all_metrics.append(metrics)
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            safe_id = safe_sample_id(metrics.get("cur_sample_id") or f"pair_{pair_index:06d}")
            save_panel(out_dir / "overlays" / f"{pair_index:04d}_{safe_id}.png", prev_row, cur_row, result, args)

    (out_dir / "summary.json").write_text(json.dumps(summarize(all_metrics), indent=2, sort_keys=True))
    print(json.dumps({"metrics": str(metrics_path), "summary": str(out_dir / "summary.json"), "num_pairs": len(all_metrics)}))


if __name__ == "__main__":
    main()
