import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import DYNAMIC_CLASS_TO_ID, parse_tracklet_xml  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Build a small manifest subset containing a target dynamic class.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--class-name", default="")
    parser.add_argument("--class-id", type=int, default=-1)
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--per-drive-stride", type=int, default=1)
    parser.add_argument("--min-boxes", type=int, default=1)
    parser.add_argument("--selection", choices=("round_robin", "sequential"), default="round_robin")
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open("r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def resolve_class_id(args):
    if args.class_id >= 0:
        return args.class_id
    if not args.class_name:
        raise ValueError("Provide --class-id or --class-name")
    if args.class_name not in DYNAMIC_CLASS_TO_ID:
        raise ValueError(f"Unknown class name {args.class_name!r}; known: {sorted(DYNAMIC_CLASS_TO_ID)}")
    return int(DYNAMIC_CLASS_TO_ID[args.class_name])


def main():
    args = parse_args()
    class_id = resolve_class_id(args)
    records = read_jsonl(args.input_manifest)
    xml_cache = {}
    candidates_by_drive = defaultdict(list)
    stats = {
        "input_manifest": args.input_manifest,
        "out_manifest": args.out_manifest,
        "class_id": class_id,
        "class_name": args.class_name,
        "max_records": args.max_records,
        "per_drive_stride": args.per_drive_stride,
        "min_boxes": args.min_boxes,
        "input_records": len(records),
    }

    for record in records:
        xml_path = record.get("tracklet_xml_path", "")
        if not xml_path:
            continue
        if xml_path not in xml_cache:
            xml_cache[xml_path] = parse_tracklet_xml(xml_path)
        boxes = xml_cache[xml_path].get(int(record["frame_index"]), [])
        count = sum(1 for box in boxes if int(box.class_id) == class_id)
        if count >= args.min_boxes:
            row = dict(record)
            row["target_class_id"] = class_id
            row["target_class_box_count"] = count
            candidates_by_drive[record["drive"]].append(row)

    selected = []
    drive_counts = Counter()
    stride = max(1, int(args.per_drive_stride))
    sampled_by_drive = {
        drive: sorted(rows, key=lambda row: int(row["frame_index"]))[::stride]
        for drive, rows in candidates_by_drive.items()
    }
    if args.selection == "sequential":
        for drive in sorted(sampled_by_drive):
            for row in sampled_by_drive[drive]:
                selected.append(row)
                drive_counts[drive] += 1
                if args.max_records > 0 and len(selected) >= args.max_records:
                    break
            if args.max_records > 0 and len(selected) >= args.max_records:
                break
    else:
        cursor = 0
        drives = sorted(sampled_by_drive)
        while drives and (args.max_records <= 0 or len(selected) < args.max_records):
            next_drives = []
            for drive in drives:
                rows = sampled_by_drive[drive]
                if cursor < len(rows):
                    selected.append(rows[cursor])
                    drive_counts[drive] += 1
                    if args.max_records > 0 and len(selected) >= args.max_records:
                        break
                    if cursor + 1 < len(rows):
                        next_drives.append(drive)
            cursor += 1
            drives = next_drives

    write_jsonl(args.out_manifest, selected)
    stats.update(
        {
            "candidate_records": sum(len(rows) for rows in candidates_by_drive.values()),
            "selected_records": len(selected),
            "selected_drives": sorted(drive_counts),
            "selected_drive_counts": dict(drive_counts),
        }
    )
    stats_path = Path(args.out_manifest).with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))
    if not selected:
        raise SystemExit("No records selected.")


if __name__ == "__main__":
    main()
