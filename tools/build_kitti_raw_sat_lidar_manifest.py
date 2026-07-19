import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import write_jsonl  # noqa: E402


SPLIT_FILES = {
    "train": "train_files.txt",
    "test1": "test1_files.txt",
    "test2": "test2_files.txt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the current KITTI raw satellite/LiDAR manifest from KITTI_location splits."
    )
    parser.add_argument("--kitti-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-root", default="dataset/KITTI_location")
    parser.add_argument(
        "--date",
        nargs="+",
        default=["all"],
        help="One or more KITTI raw dates, or 'all' to use every date in KITTI_location.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def _read_kitti_location_split(split_root):
    split_dir = Path(split_root)
    split_lookup = {}
    for split_name, file_name in SPLIT_FILES.items():
        path = split_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing KITTI split file: {path}")
        for line in path.read_text().splitlines():
            if line.strip():
                split_lookup[line.split()[0]] = split_name
    return split_lookup


def _root_dates(root):
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / f"{path.name}_calib").exists()
    )


def _resolve_dates(args, split_lookup):
    requested = list(args.date)
    if len(requested) == 1 and requested[0].lower() in {"all", "*"}:
        dates = sorted({key.split("/", 1)[0] for key in split_lookup})
    else:
        dates = requested
    available = set(_root_dates(Path(args.kitti_root)))
    missing = sorted(set(dates) - available)
    if missing:
        raise FileNotFoundError(f"Requested KITTI dates are missing under {args.kitti_root}: {missing}")
    return dates


def _resolve_calib_dir(date_dir, date):
    calib_root = date_dir / f"{date}_calib"
    direct_cam = calib_root / "calib_cam_to_cam.txt"
    direct_velo = calib_root / "calib_velo_to_cam.txt"
    if direct_cam.is_file() and direct_velo.is_file():
        return calib_root
    for camera_file in calib_root.rglob("calib_cam_to_cam.txt"):
        if (camera_file.parent / "calib_velo_to_cam.txt").is_file():
            return camera_file.parent
    raise FileNotFoundError(f"Missing calibration files under: {calib_root}")


def _ids_in_dir(path, suffix):
    if not path.is_dir():
        return set()
    return {item.stem for item in path.glob(f"*{suffix}")}


def _drive_paths(drive):
    return {
        "image_02_path": drive / "image_02" / "data",
        "satellite_path": drive / "satellite",
        "velodyne_path": drive / "velodyne_points" / "data",
        "oxts_path": drive / "oxts" / "data",
    }


def _frame_ids(paths, frame_stride):
    frame_ids = sorted(
        _ids_in_dir(paths["image_02_path"], ".png")
        & _ids_in_dir(paths["satellite_path"], ".png")
        & _ids_in_dir(paths["velodyne_path"], ".bin")
        & _ids_in_dir(paths["oxts_path"], ".txt")
    )
    if frame_stride > 1:
        frame_ids = frame_ids[::frame_stride]
    return frame_ids


def _frame_record(date, drive, frame_id, paths, calib_dir, split):
    return {
        "sample_id": f"{date}/{drive.name}/{frame_id}",
        "date": date,
        "drive": drive.name,
        "frame_id": frame_id,
        "frame_index": int(frame_id),
        "image_02_path": str(paths["image_02_path"] / f"{frame_id}.png"),
        "satellite_path": str(paths["satellite_path"] / f"{frame_id}.png"),
        "velodyne_path": str(paths["velodyne_path"] / f"{frame_id}.bin"),
        "oxts_path": str(paths["oxts_path"] / f"{frame_id}.txt"),
        "calib_cam_to_cam_path": str(calib_dir / "calib_cam_to_cam.txt"),
        "calib_velo_to_cam_path": str(calib_dir / "calib_velo_to_cam.txt"),
        "calib_dir": str(calib_dir),
        "split": split,
    }


def _collect_date(args, date, split_lookup, remaining):
    root = Path(args.kitti_root)
    date_dir = root / date
    calib_dir = _resolve_calib_dir(date_dir, date)
    records = {split: [] for split in SPLIT_FILES}
    drive_stats = {}
    skipped_unmatched = 0
    collected = 0

    drives = sorted(path for path in date_dir.glob(f"{date}_drive_*_sync") if path.is_dir())
    for drive in drives:
        paths = _drive_paths(drive)
        split_counts = Counter()
        for frame_id in _frame_ids(paths, int(args.frame_stride)):
            split_key = f"{date}/{drive.name}/{frame_id}.png"
            split = split_lookup.get(split_key)
            if split is None:
                skipped_unmatched += 1
                continue
            records[split].append(_frame_record(date, drive, frame_id, paths, calib_dir, split))
            split_counts[split] += 1
            collected += 1
            if remaining > 0 and collected >= remaining:
                break
        drive_stats[drive.name] = {
            "frames": int(sum(split_counts.values())),
            "split_counts": dict(split_counts),
        }
        if remaining > 0 and collected >= remaining:
            break

    return records, {
        "date": date,
        "calib_dir": str(calib_dir),
        "num_drives": len(drives),
        "num_train": len(records["train"]),
        "num_test1": len(records["test1"]),
        "num_test2": len(records["test2"]),
        "num_total": sum(len(items) for items in records.values()),
        "drive_stats": drive_stats,
        "skipped_unmatched_split_frames": skipped_unmatched,
    }


def collect_records(args):
    split_lookup = _read_kitti_location_split(args.split_root)
    dates = _resolve_dates(args, split_lookup)
    records = {split: [] for split in SPLIT_FILES}
    date_stats = {}

    for date in dates:
        remaining = 0
        if int(args.max_samples) > 0:
            remaining = int(args.max_samples) - sum(len(items) for items in records.values())
            if remaining <= 0:
                break
        date_records, stats = _collect_date(args, date, split_lookup, remaining)
        for split_name in SPLIT_FILES:
            records[split_name].extend(date_records[split_name])
        date_stats[date] = stats

    all_records = [record for split_name in SPLIT_FILES for record in records[split_name]]
    stats = {
        "kitti_root": str(Path(args.kitti_root)),
        "split_root": str(Path(args.split_root)),
        "split_strategy": "kitti_location_train_test1_test2",
        "dates": dates,
        "frame_stride": int(args.frame_stride),
        "num_train": len(records["train"]),
        "num_test1": len(records["test1"]),
        "num_test2": len(records["test2"]),
        "num_total": len(all_records),
        "train_drives": sorted({record["drive"] for record in records["train"]}),
        "test1_drives": sorted({record["drive"] for record in records["test1"]}),
        "test2_drives": sorted({record["drive"] for record in records["test2"]}),
        "skipped_unmatched_split_frames": sum(
            item["skipped_unmatched_split_frames"] for item in date_stats.values()
        ),
        "date_stats": date_stats,
    }
    return records["train"], [], records["test1"], records["test2"], stats


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_records, _, test1_records, test2_records, stats = collect_records(args)

    write_jsonl(str(out_dir / "train_manifest.jsonl"), train_records)
    write_jsonl(str(out_dir / "test1_manifest.jsonl"), test1_records)
    write_jsonl(str(out_dir / "test2_manifest.jsonl"), test2_records)
    write_jsonl(str(out_dir / "val_manifest.jsonl"), test1_records or test2_records)
    (out_dir / "manifest_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))

    if not train_records:
        raise SystemExit("Manifest build produced an empty train split.")
    if not int(args.max_samples) and not (test1_records or test2_records):
        raise SystemExit("Manifest build produced no KITTI_location test records.")


if __name__ == "__main__":
    main()
