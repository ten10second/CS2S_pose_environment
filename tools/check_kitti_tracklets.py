import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    box_corners_velo,
    generate_lidar_condition,
    load_raw_calibration,
    parse_tracklet_xml,
    project_velo_to_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Check KITTI raw XML tracklets and LiDAR projection.")
    parser.add_argument("--drive", required=True, help="Path to a KITTI raw *_sync drive.")
    parser.add_argument("--calib-dir", default="", help="Path to the matching *_calib directory.")
    parser.add_argument("--frame-id", default="", help="Frame id such as 0000000066. Defaults to first dynamic frame.")
    parser.add_argument("--output-overlay", default="", help="Optional output image for dynamic mask overlay.")
    return parser.parse_args()


def infer_calib_dir(drive: Path) -> Path:
    date = drive.parent.name
    return drive.parent / f"{date}_calib"


def projected_box_bounds(box, calib):
    corners = box_corners_velo(box)
    uv, _, valid = project_velo_to_image(corners, calib, (128, 512))
    if not np.any(valid):
        return None
    u = uv[valid, 0]
    v = uv[valid, 1]
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]


def read_local_bbox2d(drive: Path, frame_id: str):
    path = drive / "bbox_2d" / f"{frame_id}.txt"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6:
            rows.append(
                {
                    "y_min": float(parts[0]),
                    "y_max": float(parts[1]),
                    "x_min": float(parts[2]),
                    "x_max": float(parts[3]),
                    "class": parts[4],
                    "score": float(parts[5]),
                }
            )
    return rows


def save_overlay(image_path: Path, mask: np.ndarray, output_path: Path):
    image = Image.open(image_path).convert("RGB").resize((512, 128))
    rgb = np.asarray(image).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255.0
    alpha = np.clip(mask[..., None] * 0.45, 0.0, 0.45)
    overlay = rgb * (1.0 - alpha) + red * alpha
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def main():
    args = parse_args()
    drive = Path(args.drive)
    calib_dir = Path(args.calib_dir) if args.calib_dir else infer_calib_dir(drive)
    xml_path = drive / "tracklet_labels.xml"
    boxes_by_frame = parse_tracklet_xml(str(xml_path))
    calib = load_raw_calibration(str(calib_dir))

    class_counts = Counter()
    frame_box_hist = Counter()
    for boxes in boxes_by_frame.values():
        frame_box_hist[len(boxes)] += 1
        for box in boxes:
            class_counts[box.object_type] += 1

    if args.frame_id:
        frame_id = args.frame_id
    else:
        first_frame = min(boxes_by_frame) if boxes_by_frame else 0
        frame_id = f"{first_frame:010d}"
    frame_index = int(frame_id)
    boxes = boxes_by_frame.get(frame_index, [])

    sample = {
        "drive": str(drive),
        "calib_dir": str(calib_dir),
        "xml_path": str(xml_path),
        "num_dynamic_frames": len(boxes_by_frame),
        "class_counts": dict(class_counts),
        "frame_box_hist": {str(k): v for k, v in sorted(frame_box_hist.items())},
        "checked_frame_id": frame_id,
        "checked_frame_boxes": len(boxes),
        "projected_boxes_128x512_xyxy": [projected_box_bounds(box, calib) for box in boxes],
        "local_bbox2d_yyxxy": read_local_bbox2d(drive, frame_id),
    }

    velodyne_path = drive / "velodyne_points" / "data" / f"{frame_id}.bin"
    if velodyne_path.exists():
        cond = generate_lidar_condition(str(velodyne_path), boxes, calib, mode="dynamic_full")
        sample.update(
            {
                "num_projected_lidar_points": int(cond["num_projected_lidar_points"]),
                "num_projected_dynamic_points": int(cond["num_projected_dynamic_points"]),
                "dynamic_mask_coverage": float(cond["dynamic_mask"].mean()),
            }
        )
        if args.output_overlay:
            image_path = drive / "image_02" / "data" / f"{frame_id}.png"
            save_overlay(image_path, cond["dynamic_mask"][0], Path(args.output_overlay))
            sample["overlay_path"] = args.output_overlay

    print(json.dumps(sample, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
