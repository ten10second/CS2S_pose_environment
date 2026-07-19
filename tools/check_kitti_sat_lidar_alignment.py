import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader import KITTI_utils as kitti_utils  # noqa: E402
from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    generate_lidar_condition,
    load_raw_calibration,
    load_velodyne_points,
    project_velo_to_image,
    read_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visual/numeric KITTI satellite, camera pose, and LiDAR alignment check.")
    parser.add_argument("--manifest", default="", help="Optional sat-lidar manifest. If omitted, --date/--drive/--frame-id are used.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--kitti-root", default="/media/shizhm/Lenovo/KITTI_RAW")
    parser.add_argument("--date", default="2011_09_26")
    parser.add_argument("--drive", default="2011_09_26_drive_0002_sync")
    parser.add_argument("--frame-id", default="0000000000")
    parser.add_argument("--out-dir", default="results/kitti_sat_lidar_alignment")
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--max-draw-points", type=int, default=25000)
    return parser.parse_args()


OxtsKeys = (
    "lat",
    "lon",
    "alt",
    "roll",
    "pitch",
    "yaw",
    "vn",
    "ve",
    "vf",
    "vl",
    "vu",
    "ax",
    "ay",
    "az",
    "af",
    "al",
    "au",
    "wx",
    "wy",
    "wz",
    "wf",
    "wl",
    "wu",
    "pos_accuracy",
    "vel_accuracy",
    "navstat",
    "numsats",
    "posmode",
    "velmode",
    "orimode",
)


def read_oxts(path):
    values = [float(value) for value in Path(path).read_text().strip().split()]
    return {key: values[idx] for idx, key in enumerate(OxtsKeys[: len(values)])}


def parse_timestamp_line(value):
    value = value.strip()
    if not value:
        return None
    if "." in value:
        prefix, frac = value.split(".", 1)
        value = f"{prefix}.{frac[:6]}"
    return datetime.fromisoformat(value)


def timestamp_at(path, index):
    lines = Path(path).read_text().splitlines()
    if index >= len(lines):
        return None
    return parse_timestamp_line(lines[index])


def infer_calib_dir(drive_dir):
    date = drive_dir.parent.name
    return drive_dir.parent / f"{date}_calib"


def record_from_drive(args):
    drive_dir = Path(args.kitti_root) / args.date / args.drive
    calib_dir = infer_calib_dir(drive_dir)
    frame_id = Path(args.frame_id).stem
    return {
        "sample_id": f"{args.date}/{args.drive}/{frame_id}",
        "date": args.date,
        "drive": args.drive,
        "frame_id": frame_id,
        "frame_index": int(frame_id),
        "image_02_path": str(drive_dir / "image_02" / "data" / f"{frame_id}.png"),
        "satellite_path": str(drive_dir / "satellite" / f"{frame_id}.png"),
        "velodyne_path": str(drive_dir / "velodyne_points" / "data" / f"{frame_id}.bin"),
        "oxts_path": str(drive_dir / "oxts" / "data" / f"{frame_id}.txt"),
        "calib_dir": str(calib_dir),
    }


def load_records(args):
    if args.manifest:
        records = read_jsonl(args.manifest)
        start = min(max(args.sample_index, 0), max(len(records) - 1, 0))
        return records[start : start + args.num_samples]
    return [record_from_drive(args)]


def draw_cross(draw, xy, color, radius=5, width=2):
    x, y = xy
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)


def draw_arrow(draw, start, vec, color, width=3):
    x0, y0 = start
    x1, y1 = x0 + vec[0], y0 + vec[1]
    draw.line((x0, y0, x1, y1), fill=color, width=width)
    angle = math.atan2(vec[1], vec[0])
    length = max(8.0, math.hypot(vec[0], vec[1]) * 0.18)
    for delta in (math.pi * 0.78, -math.pi * 0.78):
        draw.line(
            (
                x1,
                y1,
                x1 + length * math.cos(angle + delta),
                y1 + length * math.sin(angle + delta),
            ),
            fill=color,
            width=width,
        )


def camera_forward_right(calib):
    if "cam2_imu_forward_right" in calib:
        offset = calib["cam2_imu_forward_right"]
        return float(offset[0]), float(offset[1])
    return float(kitti_utils.CameraGPS_shift_left[0]), float(kitti_utils.CameraGPS_shift_left[1])


def draw_raw_satellite(record, oxts, calib, side=512):
    image = Image.open(record["satellite_path"]).convert("RGB").resize((side, side), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    center = (side / 2.0, side / 2.0)
    draw_cross(draw, center, (255, 40, 40), radius=9, width=3)
    yaw = float(oxts["yaw"])
    draw_arrow(draw, center, (48.0 * math.cos(yaw), -48.0 * math.sin(yaw)), (255, 220, 40), width=4)
    meter_per_pixel = kitti_utils.get_meter_per_pixel()
    camera_forward, camera_right = camera_forward_right(calib)
    east = camera_forward * math.cos(yaw) + camera_right * math.sin(yaw)
    north = camera_forward * math.sin(yaw) - camera_right * math.cos(yaw)
    camera_xy = (center[0] + east / meter_per_pixel, center[1] - north / meter_per_pixel)
    draw_cross(draw, camera_xy, (40, 255, 80), radius=7, width=3)
    draw.text((8, 8), "raw satellite north-up, centered at IMU/OXTS", fill=(255, 255, 255))
    draw.text((8, 26), "red=IMU center, green=cam2 from calib, yellow=IMU yaw", fill=(255, 255, 255))
    return image


def draw_aligned_satellite(record, calib):
    dataset = SatLidarRawDataset.__new__(SatLidarRawDataset)
    dataset.align_satellite_to_camera = True
    dataset.sat_size = 256
    dataset.meter_per_pixel = kitti_utils.get_meter_per_pixel()
    with Image.open(record["satellite_path"]) as image:
        aligned = dataset._satellite_image(image, record["oxts_path"], calib).resize((512, 512), Image.BILINEAR)
    draw = ImageDraw.Draw(aligned)
    center = (256.0, 256.0)
    draw_cross(draw, center, (40, 255, 80), radius=9, width=3)
    draw_arrow(draw, center, (70.0, 0.0), (255, 220, 40), width=4)
    draw.text((8, 8), "model satellite input after yaw + IMU->cam2 shift", fill=(255, 255, 255))
    draw.text((8, 26), "center=cam2, arrow=canonical forward", fill=(255, 255, 255))
    return aligned


def draw_lidar_overlay(record, calib, output_size, max_draw_points):
    base = Image.open(record["image_02_path"]).convert("RGB").resize((output_size[1], output_size[0]), Image.BILINEAR)
    draw = ImageDraw.Draw(base)
    points = load_velodyne_points(record["velodyne_path"])
    points_xyz = points[:, :3] if points.size else np.zeros((0, 3), dtype=np.float32)
    uv, depth, valid = project_velo_to_image(points_xyz, calib, output_size)
    valid_indices = np.nonzero(valid)[0]
    if len(valid_indices) > max_draw_points:
        valid_indices = valid_indices[np.linspace(0, len(valid_indices) - 1, max_draw_points).astype(np.int64)]
    for idx in valid_indices:
        x = int(round(float(uv[idx, 0])))
        y = int(round(float(uv[idx, 1])))
        d = float(np.clip(depth[idx] / 80.0, 0.0, 1.0))
        color = (int(255 * (1.0 - d)), int(80 + 175 * d), 255)
        draw.point((x, y), fill=color)

    draw.text((8, 8), "RGB + LiDAR projection", fill=(255, 255, 255))
    draw.text((8, 26), "color encodes projected LiDAR depth", fill=(255, 255, 255))
    return base.resize((512, 128), Image.BILINEAR)


def condition_rgb(lidar_cond):
    cond = lidar_cond["lidar_cond"]
    rgb = np.stack([cond[2], cond[1], cond[6]], axis=-1)
    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255)).resize((512, 128), Image.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "current pointmap: depth / hit / camera-Z", fill=(255, 255, 255))
    return image


def make_panel(raw_sat, aligned_sat, lidar_overlay, cond_img):
    panel = Image.new("RGB", (1024, 640), (24, 24, 24))
    panel.paste(raw_sat, (0, 0))
    panel.paste(aligned_sat, (512, 0))
    panel.paste(lidar_overlay, (0, 512))
    panel.paste(cond_img, (512, 512))
    return panel


def timestamp_report(record):
    frame_index = int(record["frame_index"])
    drive_dir = Path(record["image_02_path"]).parents[2]
    image_ts = timestamp_at(drive_dir / "image_02" / "timestamps.txt", frame_index)
    velo_ts = timestamp_at(drive_dir / "velodyne_points" / "timestamps.txt", frame_index)
    oxts_ts = timestamp_at(drive_dir / "oxts" / "timestamps.txt", frame_index)
    report = {
        "image_02": image_ts.isoformat() if image_ts else "",
        "velodyne": velo_ts.isoformat() if velo_ts else "",
        "oxts": oxts_ts.isoformat() if oxts_ts else "",
    }
    if image_ts and velo_ts:
        report["velodyne_minus_image_ms"] = (velo_ts - image_ts).total_seconds() * 1000.0
    if image_ts and oxts_ts:
        report["oxts_minus_image_ms"] = (oxts_ts - image_ts).total_seconds() * 1000.0
    return report


def satellite_neighbor_shift(record):
    frame_index = int(record["frame_index"])
    next_frame = f"{frame_index + 1:010d}"
    sat_next = Path(record["satellite_path"]).with_name(f"{next_frame}.png")
    oxts_next = Path(record["oxts_path"]).with_name(f"{next_frame}.txt")
    if not sat_next.exists() or not oxts_next.exists():
        return {}

    img0 = np.asarray(Image.open(record["satellite_path"]).convert("L").resize((512, 512), Image.BILINEAR), dtype=np.float32)
    img1 = np.asarray(Image.open(sat_next).convert("L").resize((512, 512), Image.BILINEAR), dtype=np.float32)
    shift, response = cv2.phaseCorrelate(img0, img1)
    oxts0 = read_oxts(record["oxts_path"])
    oxts1 = read_oxts(str(oxts_next))
    dx_east, dy_south = kitti_utils.gps2meters(oxts0["lat"], oxts0["lon"], oxts1["lat"], oxts1["lon"])
    meter_per_pixel = kitti_utils.get_meter_per_pixel()
    return {
        "next_frame": next_frame,
        "phase_shift_x_px": float(shift[0]),
        "phase_shift_y_px": float(shift[1]),
        "phase_response": float(response),
        "gps_delta_east_m": float(dx_east),
        "gps_delta_south_m": float(dy_south),
        "gps_delta_px_x": float(dx_east / meter_per_pixel),
        "gps_delta_px_y": float(dy_south / meter_per_pixel),
    }


def check_record(record, args, out_dir):
    calib = load_raw_calibration(record["calib_dir"])
    oxts = read_oxts(record["oxts_path"])
    lidar = generate_lidar_condition(
        record["velodyne_path"],
        [],
        calib,
        output_size=(args.image_height, args.image_width),
        mode="raw_lidar_pointmap",
    )

    raw_sat = draw_raw_satellite(record, oxts, calib)
    aligned_sat = draw_aligned_satellite(record, calib)
    lidar_overlay = draw_lidar_overlay(record, calib, (args.image_height, args.image_width), args.max_draw_points)
    cond_img = condition_rgb(lidar)
    panel = make_panel(raw_sat, aligned_sat, lidar_overlay, cond_img)

    rel = Path(record["sample_id"]).with_suffix(".png")
    panel_path = out_dir / "panels" / rel
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(panel_path)

    points = load_velodyne_points(record["velodyne_path"])
    points_xyz = points[:, :3] if points.size else np.zeros((0, 3), dtype=np.float32)
    _, _, valid = project_velo_to_image(points_xyz, calib, (args.image_height, args.image_width))
    lidar_cond = lidar["lidar_cond"]
    summary = {
        "sample_id": record["sample_id"],
        "panel_path": str(panel_path),
        "lat": float(oxts["lat"]),
        "lon": float(oxts["lon"]),
        "yaw_rad": float(oxts["yaw"]),
        "yaw_deg": float(oxts["yaw"] / math.pi * 180.0),
        "satellite_path": record["satellite_path"],
        "image_02_path": record["image_02_path"],
        "velodyne_path": record["velodyne_path"],
        "oxts_path": record["oxts_path"],
        "meter_per_pixel_process_sat": float(kitti_utils.get_meter_per_pixel()),
        "calib_resolved_dir": str(calib.get("calib_dir", record["calib_dir"])),
        "transform_direction_imu_to_velo": "calib_imu_to_velo.txt: IMU -> Velodyne",
        "transform_direction_velo_to_cam": "calib_velo_to_cam.txt: Velodyne -> camera reference",
        "camera_02_in_imu_xyz_forward_left_up_m": (
            [float(value) for value in calib["cam2_rect_in_imu"]]
            if "cam2_rect_in_imu" in calib
            else []
        ),
        "camera_02_in_imu_forward_right_m": list(camera_forward_right(calib)),
        "projected_lidar_points": int(valid.sum()),
        "pointmap_hit_pixels": int((lidar_cond[1] > 0.0).sum()),
        "pointmap_xyz_pixels": int((np.abs(lidar_cond[4:7]).sum(axis=0) > 0.0).sum()),
        "timestamp": timestamp_report(record),
        "satellite_neighbor_shift": satellite_neighbor_shift(record),
    }
    return summary


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), dict)})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args)
    summaries = [check_record(record, args, out_dir) for record in records]
    write_json(out_dir / "alignment_summary.json", summaries)
    write_csv(out_dir / "alignment_summary.csv", summaries)
    print(json.dumps({"out_dir": str(out_dir), "records": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
