import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import write_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build KITTI raw RGB-LiDAR manifests without satellite/XML requirements.")
    parser.add_argument("--kitti-root", default="/media/shizhm/Lenovo/KITTI_RAW")
    parser.add_argument("--out-dir", default="dataset/kitti_raw_lidar_normal")
    parser.add_argument("--dates", nargs="*", default=[], help="Optional explicit date folders, e.g. 2011_09_26.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--val-every-drive", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def _ids(path: Path, suffix: str):
    if not path.exists():
        return set()
    return {p.stem for p in path.glob(f"*{suffix}")}


def _date_dirs(root: Path, dates):
    if dates:
        return [root / date for date in dates]
    return sorted(p for p in root.glob("2011_*") if p.is_dir())


def _resolve_calib_dir(date_dir: Path) -> Path:
    date = date_dir.name
    root = date_dir / f"{date}_calib"
    if (root / "calib_cam_to_cam.txt").exists() and (root / "calib_velo_to_cam.txt").exists():
        return root
    if root.exists():
        for candidate in root.rglob("calib_cam_to_cam.txt"):
            if (candidate.parent / "calib_velo_to_cam.txt").exists():
                return candidate.parent
    return root


def collect_records(args):
    root = Path(args.kitti_root)
    train_records = []
    val_records = []
    date_stats = {}
    sample_count = 0
    global_drive_index = 0

    for date_dir in _date_dirs(root, args.dates):
        date = date_dir.name
        calib_dir = _resolve_calib_dir(date_dir)
        if not (calib_dir / "calib_cam_to_cam.txt").exists() or not (calib_dir / "calib_velo_to_cam.txt").exists():
            continue
        drives = sorted(p for p in date_dir.glob(f"{date}_drive_*_sync") if p.is_dir())
        date_stats[date] = {"num_drives": len(drives), "drives": {}}
        for drive in drives:
            image_dir = drive / "image_02" / "data"
            velodyne_dir = drive / "velodyne_points" / "data"
            frame_ids = sorted(_ids(image_dir, ".png") & _ids(velodyne_dir, ".bin"))
            if args.frame_stride > 1:
                frame_ids = [frame for i, frame in enumerate(frame_ids) if i % args.frame_stride == 0]
            split = "val" if args.val_every_drive > 0 and global_drive_index % args.val_every_drive == 0 else "train"
            global_drive_index += 1

            kept = 0
            for frame_id in frame_ids:
                record = {
                    "sample_id": f"{date}/{drive.name}/{frame_id}",
                    "date": date,
                    "drive": drive.name,
                    "frame_id": frame_id,
                    "frame_index": int(frame_id),
                    "image_02_path": str(image_dir / f"{frame_id}.png"),
                    "velodyne_path": str(velodyne_dir / f"{frame_id}.bin"),
                    "calib_cam_to_cam_path": str(calib_dir / "calib_cam_to_cam.txt"),
                    "calib_velo_to_cam_path": str(calib_dir / "calib_velo_to_cam.txt"),
                    "calib_dir": str(calib_dir),
                    "split": split,
                }
                if split == "val":
                    val_records.append(record)
                else:
                    train_records.append(record)
                kept += 1
                sample_count += 1
                if args.max_samples and sample_count >= args.max_samples:
                    break
            date_stats[date]["drives"][drive.name] = {"split": split, "frames": kept}
            if args.max_samples and sample_count >= args.max_samples:
                break
        if args.max_samples and sample_count >= args.max_samples:
            break

    stats = {
        "kitti_root": str(root),
        "frame_stride": args.frame_stride,
        "val_every_drive": args.val_every_drive,
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_total": len(train_records) + len(val_records),
        "dates": date_stats,
    }
    return train_records, val_records, stats


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    train_records, val_records, stats = collect_records(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out_dir / "train_manifest.jsonl"), train_records)
    write_jsonl(str(out_dir / "val_manifest.jsonl"), val_records)
    (out_dir / "manifest_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))
    if not train_records or not val_records:
        raise SystemExit("Manifest build produced an empty train or val split.")


if __name__ == "__main__":
    main()
