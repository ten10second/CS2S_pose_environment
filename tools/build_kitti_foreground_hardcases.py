import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Select KITTI foreground hardcases from a manifest.")
    parser.add_argument("--manifest", default="dataset/kitti_raw_sat_lidar/test2_manifest.jsonl")
    parser.add_argument("--out", default="dataset/kitti_raw_sat_lidar/foreground_test2_hardcases.jsonl")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--min-dynamic-boxes", type=int, default=1)
    parser.add_argument("--min-projected-lidar-points", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    with Path(args.manifest).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("num_dynamic_boxes", 0)) < args.min_dynamic_boxes:
                continue
            projected_lidar = record.get("num_projected_lidar_points")
            if projected_lidar is not None and int(projected_lidar) < args.min_projected_lidar_points:
                continue
            records.append(record)

    records.sort(
        key=lambda item: (
            int(item.get("num_dynamic_boxes", 0)),
            int(item.get("num_projected_dynamic_points", 0)),
            int(item.get("num_projected_lidar_points", 0)),
        ),
        reverse=True,
    )
    selected = []
    for idx, record in enumerate(records[: args.max_samples]):
        item = dict(record)
        item["foreground_hardcase_index"] = idx
        item["foreground_source_manifest"] = args.manifest
        selected.append(item)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in selected:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    drives = sorted({f"{record.get('date')}/{record.get('drive')}" for record in selected})
    print(
        json.dumps(
            {
                "manifest": args.manifest,
                "out": str(out),
                "num_candidates": len(records),
                "num_selected": len(selected),
                "drives": drives,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
