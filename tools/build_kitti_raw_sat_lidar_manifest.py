import argparse
import json
import sys
from collections import Counter
from copy import copy
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
    parser.add_argument(
        "--date",
        nargs="+",
        default=["all"],
        help="One or more KITTI raw dates, or 'all' to use dates present in the split / KITTI root.",
    )
    parser.add_argument("--out-dir", default="dataset/kitti_raw_sat_lidar")
    parser.add_argument("--split-root", default="dataset/KITTI_location")
    parser.add_argument(
        "--split-mode",
        choices=("kitti_location", "drive_val_every"),
        default="kitti_location",
        help="Use official KITTI_location train/test split or legacy drive-based val split.",
    )
    parser.add_argument("--require-tracklet", action="store_true", default=False)
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


def _read_kitti_location_split(split_root: str) -> dict:
    split_dir = Path(split_root)
    split_lookup = {}
    for split_name, file_name in (
        ("train", "train_files.txt"),
        ("test1", "test1_files.txt"),
        ("test2", "test2_files.txt"),
    ):
        path = split_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing KITTI split file: {path}")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rel_path = line.split()[0]
            split_lookup[rel_path] = split_name
    return split_lookup


def _split_dates(split_lookup: dict) -> list:
    return sorted({key.split("/", 1)[0] for key in split_lookup})


def _root_dates(root: Path) -> list:
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / f"{path.name}_calib").exists()
    )


def _resolve_calib_dir(date_dir: Path, date: str) -> Path:
    calib_root = date_dir / f"{date}_calib"
    if (calib_root / "calib_cam_to_cam.txt").exists() and (calib_root / "calib_velo_to_cam.txt").exists():
        return calib_root
    for candidate in calib_root.rglob("calib_cam_to_cam.txt"):
        if (candidate.parent / "calib_velo_to_cam.txt").exists():
            return candidate.parent
    raise FileNotFoundError(f"Missing calibration files under: {calib_root}")


def _resolve_dates(args, split_lookup) -> list:
    requested = args.date
    if len(requested) == 1 and requested[0].lower() in {"all", "*"}:
        dates = _split_dates(split_lookup) if split_lookup is not None else _root_dates(Path(args.kitti_root))
    else:
        dates = requested
    available = set(_root_dates(Path(args.kitti_root)))
    missing = sorted(set(dates) - available)
    if missing:
        raise FileNotFoundError(f"Requested KITTI dates are missing under {args.kitti_root}: {missing}")
    return dates


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


def _collect_records_for_date(args, date: str, split_lookup):
    root = Path(args.kitti_root)
    date_dir = root / date
    calib_dir = _resolve_calib_dir(date_dir, date)
    calib = None if args.skip_lidar_counts else load_raw_calibration(str(calib_dir))

    drives = sorted(p for p in date_dir.glob(f"{date}_drive_*_sync") if p.is_dir())
    if args.require_tracklet:
        drives = [p for p in drives if (p / "tracklet_labels.xml").exists()]

    val_drives = None
    if args.split_mode == "drive_val_every" and args.val_drives:
        val_drives = {_normalize_drive_name(drive, date) for drive in args.val_drives}
        missing = sorted(val_drives - {drive.name for drive in drives})
        if missing:
            raise ValueError(f"Validation drives not found in XML drive set: {missing}")

    train_records = []
    val_records = []
    test1_records = []
    test2_records = []
    stats = {
        "kitti_root": str(root),
        "date": date,
        "calib_dir": str(calib_dir),
        "num_drives": len(drives),
        "split_mode": args.split_mode,
        "split_root": args.split_root,
        "require_tracklet": args.require_tracklet,
        "val_every": args.val_every,
        "val_drives": sorted(val_drives) if val_drives else [],
        "split_strategy": (
            "kitti_location_train_test1_test2"
            if args.split_mode == "kitti_location"
            else ("explicit_val_drives" if val_drives else "sorted_drive_val_every")
        ),
        "frame_stride": args.frame_stride,
        "dynamic_classes": DYNAMIC_CLASS_TO_ID,
        "drive_stats": {},
        "skipped_unmatched_split_frames": 0,
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

        drive_class_counts = Counter()
        dynamic_frames = 0
        drive_split_counts = Counter()

        for frame_id in frame_ids:
            if split_lookup is not None:
                split_key = f"{date}/{drive.name}/{frame_id}.png"
                split = split_lookup.get(split_key)
                if split is None:
                    stats["skipped_unmatched_split_frames"] += 1
                    continue
            elif val_drives is not None:
                split = "val" if drive.name in val_drives else "train"
            else:
                split = "val" if args.val_every > 0 and drive_index % args.val_every == 0 else "train"

            frame_int = int(frame_id)
            boxes = boxes_by_frame.get(frame_int, [])
            if boxes:
                dynamic_frames += 1
            for box in boxes:
                class_counts[box.object_type] += 1
                drive_class_counts[box.object_type] += 1
            frame_box_hist[len(boxes)] += 1

            record = {
                "sample_id": f"{date}/{drive.name}/{frame_id}",
                "date": date,
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
            elif split == "test1":
                test1_records.append(record)
            elif split == "test2":
                test2_records.append(record)
            else:
                train_records.append(record)
            drive_split_counts[split] += 1
            sample_count += 1
            if args.max_samples and sample_count >= args.max_samples:
                break
        stats["drive_stats"][drive.name] = {
            "split_counts": dict(drive_split_counts),
            "frames": sum(drive_split_counts.values()),
            "dynamic_frames": dynamic_frames,
            "class_counts": dict(drive_class_counts),
        }
        if args.max_samples and sample_count >= args.max_samples:
            break

    stats.update(
        {
            "num_train": len(train_records),
            "num_val": len(val_records),
            "num_test1": len(test1_records),
            "num_test2": len(test2_records),
            "num_total": len(train_records) + len(val_records) + len(test1_records) + len(test2_records),
            "train_drives": sorted({record["drive"] for record in train_records}),
            "actual_val_drives": sorted({record["drive"] for record in val_records}),
            "test1_drives": sorted({record["drive"] for record in test1_records}),
            "test2_drives": sorted({record["drive"] for record in test2_records}),
            "split_frame_counts": {
                "train": len(train_records),
                "val": len(val_records),
                "test1": len(test1_records),
                "test2": len(test2_records),
            },
            "split_zero_dynamic_frames": {
                "train": sum(1 for record in train_records if record["num_dynamic_boxes"] == 0),
                "val": sum(1 for record in val_records if record["num_dynamic_boxes"] == 0),
                "test1": sum(1 for record in test1_records if record["num_dynamic_boxes"] == 0),
                "test2": sum(1 for record in test2_records if record["num_dynamic_boxes"] == 0),
            },
            "split_dynamic_frames": {
                "train": sum(1 for record in train_records if record["num_dynamic_boxes"] > 0),
                "val": sum(1 for record in val_records if record["num_dynamic_boxes"] > 0),
                "test1": sum(1 for record in test1_records if record["num_dynamic_boxes"] > 0),
                "test2": sum(1 for record in test2_records if record["num_dynamic_boxes"] > 0),
            },
            "class_counts": dict(class_counts),
            "frame_box_hist": {str(k): v for k, v in sorted(frame_box_hist.items())},
        }
    )
    return train_records, val_records, test1_records, test2_records, stats


def collect_records(args):
    split_lookup = None
    if args.split_mode == "kitti_location":
        split_lookup = _read_kitti_location_split(args.split_root)
    dates = _resolve_dates(args, split_lookup)

    all_train_records = []
    all_val_records = []
    all_test1_records = []
    all_test2_records = []
    date_stats = {}
    for date in dates:
        date_args = copy(args)
        if args.max_samples:
            already = (
                len(all_train_records)
                + len(all_val_records)
                + len(all_test1_records)
                + len(all_test2_records)
            )
            remaining = args.max_samples - already
            if remaining <= 0:
                break
            date_args.max_samples = remaining
        train_records, val_records, test1_records, test2_records, stats = _collect_records_for_date(
            date_args,
            date,
            split_lookup,
        )
        all_train_records.extend(train_records)
        all_val_records.extend(val_records)
        all_test1_records.extend(test1_records)
        all_test2_records.extend(test2_records)
        date_stats[date] = stats

    class_counts = Counter()
    frame_box_hist = Counter()
    drive_stats = {}
    skipped_unmatched_split_frames = 0
    for stats in date_stats.values():
        class_counts.update(stats["class_counts"])
        frame_box_hist.update({int(key): value for key, value in stats["frame_box_hist"].items()})
        drive_stats.update(stats["drive_stats"])
        skipped_unmatched_split_frames += stats["skipped_unmatched_split_frames"]

    aggregate_stats = {
        "kitti_root": str(Path(args.kitti_root)),
        "dates": dates,
        "num_dates": len(dates),
        "split_mode": args.split_mode,
        "split_root": args.split_root,
        "require_tracklet": args.require_tracklet,
        "val_every": args.val_every,
        "val_drives": args.val_drives or [],
        "split_strategy": (
            "kitti_location_train_test1_test2"
            if args.split_mode == "kitti_location"
            else ("explicit_val_drives" if args.val_drives else "sorted_drive_val_every")
        ),
        "frame_stride": args.frame_stride,
        "dynamic_classes": DYNAMIC_CLASS_TO_ID,
        "drive_stats": drive_stats,
        "date_stats": date_stats,
        "skipped_unmatched_split_frames": skipped_unmatched_split_frames,
        "num_train": len(all_train_records),
        "num_val": len(all_val_records),
        "num_test1": len(all_test1_records),
        "num_test2": len(all_test2_records),
        "num_total": (
            len(all_train_records)
            + len(all_val_records)
            + len(all_test1_records)
            + len(all_test2_records)
        ),
        "train_drives": sorted({record["drive"] for record in all_train_records}),
        "actual_val_drives": sorted({record["drive"] for record in all_val_records}),
        "test1_drives": sorted({record["drive"] for record in all_test1_records}),
        "test2_drives": sorted({record["drive"] for record in all_test2_records}),
        "split_frame_counts": {
            "train": len(all_train_records),
            "val": len(all_val_records),
            "test1": len(all_test1_records),
            "test2": len(all_test2_records),
        },
        "split_zero_dynamic_frames": {
            "train": sum(1 for record in all_train_records if record["num_dynamic_boxes"] == 0),
            "val": sum(1 for record in all_val_records if record["num_dynamic_boxes"] == 0),
            "test1": sum(1 for record in all_test1_records if record["num_dynamic_boxes"] == 0),
            "test2": sum(1 for record in all_test2_records if record["num_dynamic_boxes"] == 0),
        },
        "split_dynamic_frames": {
            "train": sum(1 for record in all_train_records if record["num_dynamic_boxes"] > 0),
            "val": sum(1 for record in all_val_records if record["num_dynamic_boxes"] > 0),
            "test1": sum(1 for record in all_test1_records if record["num_dynamic_boxes"] > 0),
            "test2": sum(1 for record in all_test2_records if record["num_dynamic_boxes"] > 0),
        },
        "class_counts": dict(class_counts),
        "frame_box_hist": {str(k): v for k, v in sorted(frame_box_hist.items())},
    }
    return all_train_records, all_val_records, all_test1_records, all_test2_records, aggregate_stats


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    train_records, val_records, test1_records, test2_records, stats = collect_records(args)

    write_jsonl(str(out_dir / "train_manifest.jsonl"), train_records)
    write_jsonl(str(out_dir / "test1_manifest.jsonl"), test1_records)
    write_jsonl(str(out_dir / "test2_manifest.jsonl"), test2_records)
    default_eval_records = test1_records or test2_records or val_records
    write_jsonl(str(out_dir / "val_manifest.jsonl"), default_eval_records)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    print(json.dumps(stats, indent=2, sort_keys=True))
    if not train_records:
        raise SystemExit("Manifest build produced an empty train split.")
    if args.split_mode == "kitti_location" and not args.max_samples and not (test1_records or test2_records):
        raise SystemExit("Manifest build produced no KITTI_location test1/test2 records.")
    if args.split_mode == "drive_val_every" and not args.max_samples and not val_records:
        raise SystemExit("Manifest build produced an empty val split.")


if __name__ == "__main__":
    main()
