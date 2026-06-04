import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    DYNAMIC_CLASS_TO_ID,
    generate_lidar_condition,
    load_raw_calibration,
    parse_tracklet_xml,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build KITTI raw satellite/LiDAR dynamic manifests.")
    parser.add_argument("--kitti-root", default="/media/shizhm/Lenovo/KITTI_RAW")
    parser.add_argument("--date", default="2011_09_26")
    parser.add_argument("--out-dir", default="dataset/kitti_raw_sat_lidar")
    parser.add_argument("--require-tracklet", action="store_true", default=True)
    parser.add_argument("--include-no-tracklet", dest="require_tracklet", action="store_false")
    parser.add_argument("--val-every", type=int, default=5, help="Use every Nth sorted drive as validation.")
    parser.add_argument(
        "--val-drives",
        nargs="*",
        default=None,
        help="Explicit validation drive names. Overrides --val-every when provided.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-lidar-counts", action="store_true", help="Do not compute projected LiDAR point counts for manifest filtering.")
    return parser.parse_args()


def _ids_in_dir(path: Path, suffix: str) -> set:
    if not path.exists():
        return set()
    return {p.stem for p in path.glob(f"*{suffix}")}


def _drive_record_paths(drive: Path):
    return {
        "image_dir": drive / "image_02" / "data",
        "satellite_dir": drive / "satellite",
        "velodyne_dir": drive / "velodyne_points" / "data",
        "oxts_dir": drive / "oxts" / "data",
        "tracklet_xml": drive / "tracklet_labels.xml",
    }


def _normalize_drive_name(name: str, date: str) -> str:
    if name.startswith(f"{date}_drive_") and name.endswith("_sync"):
        return name
    if name.startswith("drive_") and name.endswith("_sync"):
        return f"{date}_{name}"
    if name.isdigit():
        return f"{date}_drive_{int(name):04d}_sync"
    return name


def collect_records(args):
    root = Path(args.kitti_root)
    date_dir = root / args.date
    calib_dir = date_dir / f"{args.date}_calib"
    if not calib_dir.exists():
        raise FileNotFoundError(f"Missing calibration directory: {calib_dir}")
    calib = None if args.skip_lidar_counts else load_raw_calibration(str(calib_dir))

    drives = sorted(p for p in date_dir.glob(f"{args.date}_drive_*_sync") if p.is_dir())
    if args.require_tracklet:
        drives = [p for p in drives if (p / "tracklet_labels.xml").exists()]
    val_drives = None
    if args.val_drives:
        val_drives = {_normalize_drive_name(drive, args.date) for drive in args.val_drives}
        missing = sorted(val_drives - {drive.name for drive in drives})
        if missing:
            raise ValueError(f"Validation drives not found in XML drive set: {missing}")

    train_records = []
    val_records = []
    stats = {
        "kitti_root": str(root),
        "date": args.date,
        "calib_dir": str(calib_dir),
        "num_drives": len(drives),
        "val_every": args.val_every,
        "val_drives": sorted(val_drives) if val_drives else [],
        "split_strategy": "explicit_val_drives" if val_drives else "sorted_drive_val_every",
        "frame_stride": args.frame_stride,
        "dynamic_classes": DYNAMIC_CLASS_TO_ID,
        "drive_stats": {},
    }
    class_counts = Counter()
    frame_box_hist = Counter()

    sample_count = 0
    for drive_index, drive in enumerate(drives):
        paths = _drive_record_paths(drive)
        boxes_by_frame = parse_tracklet_xml(str(paths["tracklet_xml"])) if paths["tracklet_xml"].exists() else {}
        image_ids = _ids_in_dir(paths["image_dir"], ".png")
        satellite_ids = _ids_in_dir(paths["satellite_dir"], ".png")
        velodyne_ids = _ids_in_dir(paths["velodyne_dir"], ".bin")
        oxts_ids = _ids_in_dir(paths["oxts_dir"], ".txt")
        frame_ids = sorted(image_ids & satellite_ids & velodyne_ids & oxts_ids)
        if args.frame_stride > 1:
            frame_ids = [f for i, f in enumerate(frame_ids) if i % args.frame_stride == 0]

        if val_drives is not None:
            split = "val" if drive.name in val_drives else "train"
        else:
            split = "val" if args.val_every > 0 and drive_index % args.val_every == 0 else "train"
        drive_class_counts = Counter()
        dynamic_frames = 0

        for frame_id in frame_ids:
            frame_int = int(frame_id)
            boxes = boxes_by_frame.get(frame_int, [])
            if boxes:
                dynamic_frames += 1
            for box in boxes:
                class_counts[box.object_type] += 1
                drive_class_counts[box.object_type] += 1
            frame_box_hist[len(boxes)] += 1

            record = {
                "sample_id": f"{args.date}/{drive.name}/{frame_id}",
                "date": args.date,
                "drive": drive.name,
                "frame_id": frame_id,
                "frame_index": frame_int,
                "image_02_path": str(paths["image_dir"] / f"{frame_id}.png"),
                "satellite_path": str(paths["satellite_dir"] / f"{frame_id}.png"),
                "velodyne_path": str(paths["velodyne_dir"] / f"{frame_id}.bin"),
                "oxts_path": str(paths["oxts_dir"] / f"{frame_id}.txt"),
                "tracklet_xml_path": str(paths["tracklet_xml"]) if paths["tracklet_xml"].exists() else "",
                "calib_cam_to_cam_path": str(calib_dir / "calib_cam_to_cam.txt"),
                "calib_velo_to_cam_path": str(calib_dir / "calib_velo_to_cam.txt"),
                "calib_dir": str(calib_dir),
                "has_dynamic_xml": bool(paths["tracklet_xml"].exists()),
                "num_dynamic_boxes": len(boxes),
                "split": split,
            }
            if calib is not None:
                lidar_counts = generate_lidar_condition(
                    record["velodyne_path"],
                    boxes,
                    calib,
                    mode="dynamic_points",
                )
                record["num_projected_lidar_points"] = int(lidar_counts["num_projected_lidar_points"])
                record["num_projected_dynamic_points"] = int(lidar_counts["num_projected_dynamic_points"])
            if split == "val":
                val_records.append(record)
            else:
                train_records.append(record)
            sample_count += 1
            if args.max_samples and sample_count >= args.max_samples:
                break
        stats["drive_stats"][drive.name] = {
            "split": split,
            "frames": len(frame_ids),
            "dynamic_frames": dynamic_frames,
            "class_counts": dict(drive_class_counts),
        }
        if args.max_samples and sample_count >= args.max_samples:
            break

    stats.update(
        {
            "num_train": len(train_records),
            "num_val": len(val_records),
            "num_total": len(train_records) + len(val_records),
            "train_drives": sorted({record["drive"] for record in train_records}),
            "actual_val_drives": sorted({record["drive"] for record in val_records}),
            "split_frame_counts": {
                "train": len(train_records),
                "val": len(val_records),
            },
            "split_zero_dynamic_frames": {
                "train": sum(1 for record in train_records if record["num_dynamic_boxes"] == 0),
                "val": sum(1 for record in val_records if record["num_dynamic_boxes"] == 0),
            },
            "split_dynamic_frames": {
                "train": sum(1 for record in train_records if record["num_dynamic_boxes"] > 0),
                "val": sum(1 for record in val_records if record["num_dynamic_boxes"] > 0),
            },
            "class_counts": dict(class_counts),
            "frame_box_hist": {str(k): v for k, v in sorted(frame_box_hist.items())},
        }
    )
    return train_records, val_records, stats


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    train_records, val_records, stats = collect_records(args)

    write_jsonl(str(out_dir / "train_manifest.jsonl"), train_records)
    write_jsonl(str(out_dir / "val_manifest.jsonl"), val_records)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    print(json.dumps(stats, indent=2, sort_keys=True))
    if not train_records or not val_records:
        raise SystemExit("Manifest build produced an empty train or val split.")


if __name__ == "__main__":
    main()
