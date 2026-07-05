import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_utils import gps2utm  # noqa: E402
from dataloader.kitti_raw_lidar_utils import write_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a route-disjoint KITTI manifest by removing train frames "
            "within a GPS buffer of the held-out route."
        )
    )
    parser.add_argument("--train-manifest", default="dataset/kitti_raw_sat_lidar/train_manifest.jsonl")
    parser.add_argument("--test-manifest", default="dataset/kitti_raw_sat_lidar/test2_manifest.jsonl")
    parser.add_argument("--out-dir", default="dataset/kitti_raw_sat_lidar_geofence_test2")
    parser.add_argument("--buffer-m", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--drop-missing-gps",
        action="store_true",
        help="Drop records with missing/malformed OXTS instead of failing.",
    )
    return parser.parse_args()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_oxts_lat_lon(path):
    with Path(path).open() as handle:
        values = handle.readline().strip().split()
    if len(values) < 2:
        raise ValueError(f"Malformed OXTS packet: {path}")
    return float(values[0]), float(values[1])


def with_gps(records, drop_missing=False):
    kept = []
    dropped = []
    for record in records:
        try:
            lat, lon = read_oxts_lat_lon(record["oxts_path"])
            utm_x, utm_y = gps2utm(lat, lon)
        except Exception as exc:
            if not drop_missing:
                raise
            dropped.append({**record, "gps_error": str(exc)})
            continue
        enriched = dict(record)
        enriched["gps_lat"] = float(lat)
        enriched["gps_lon"] = float(lon)
        enriched["gps_utm_x"] = float(utm_x)
        enriched["gps_utm_y"] = float(utm_y)
        kept.append(enriched)
    return kept, dropped


def nearest_distances(points, refs, chunk_size):
    if len(refs) == 0:
        return np.full((len(points),), np.inf, dtype=np.float32)
    dists = np.empty((len(points),), dtype=np.float32)
    refs = refs.astype(np.float32, copy=False)
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size].astype(np.float32, copy=False)
        diff = chunk[:, None, :] - refs[None, :, :]
        dists[start : start + len(chunk)] = np.sqrt(np.square(diff).sum(axis=2).min(axis=1))
    return dists


def drive_key(record):
    return f"{record.get('date', '')}/{record.get('drive', '')}"


def distance_summary(values):
    if len(values) == 0:
        return {}
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return {}
    return {
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    train = read_jsonl(args.train_manifest)
    test = read_jsonl(args.test_manifest)
    train, dropped_train = with_gps(train, drop_missing=args.drop_missing_gps)
    test, dropped_test = with_gps(test, drop_missing=args.drop_missing_gps)

    test_points = np.asarray([[r["gps_utm_x"], r["gps_utm_y"]] for r in test], dtype=np.float32)
    train_points = np.asarray([[r["gps_utm_x"], r["gps_utm_y"]] for r in train], dtype=np.float32)
    dists = nearest_distances(train_points, test_points, args.chunk_size)

    kept_train = []
    removed_train = []
    for record, dist in zip(train, dists):
        enriched = dict(record)
        enriched["nearest_test_route_distance_m"] = float(dist)
        if dist < args.buffer_m:
            enriched["geofence_removed"] = True
            removed_train.append(enriched)
        else:
            enriched["geofence_removed"] = False
            kept_train.append(enriched)

    test_out = []
    for record in test:
        enriched = dict(record)
        enriched["geofence_test_route"] = True
        test_out.append(enriched)

    write_jsonl(str(out_dir / "train_manifest.jsonl"), kept_train)
    write_jsonl(str(out_dir / "test_manifest.jsonl"), test_out)
    write_jsonl(str(out_dir / "val_manifest.jsonl"), test_out)
    write_jsonl(str(out_dir / "removed_train_near_test_manifest.jsonl"), removed_train)
    if dropped_train:
        write_jsonl(str(out_dir / "dropped_train_missing_gps_manifest.jsonl"), dropped_train)
    if dropped_test:
        write_jsonl(str(out_dir / "dropped_test_missing_gps_manifest.jsonl"), dropped_test)

    stats = {
        "train_manifest": args.train_manifest,
        "test_manifest": args.test_manifest,
        "out_dir": str(out_dir),
        "buffer_m": args.buffer_m,
        "num_train_input": len(train) + len(dropped_train),
        "num_test_input": len(test) + len(dropped_test),
        "num_train_with_gps": len(train),
        "num_test_with_gps": len(test),
        "num_train_kept": len(kept_train),
        "num_train_removed": len(removed_train),
        "removed_train_fraction": float(len(removed_train) / max(len(train), 1)),
        "num_dropped_train_missing_gps": len(dropped_train),
        "num_dropped_test_missing_gps": len(dropped_test),
        "train_drives_input": sorted({drive_key(r) for r in train}),
        "test_drives": sorted({drive_key(r) for r in test}),
        "kept_train_drives": sorted({drive_key(r) for r in kept_train}),
        "removed_train_drives": sorted({drive_key(r) for r in removed_train}),
        "drive_overlap_after_split": sorted({drive_key(r) for r in kept_train} & {drive_key(r) for r in test}),
        "nearest_test_route_distance_m": distance_summary(dists),
        "split_strategy": "remove_train_frames_within_gps_buffer_of_test_route",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps(stats, indent=2, sort_keys=True))

    if not kept_train:
        raise SystemExit("GPS-buffer split produced an empty train split.")
    if not test_out:
        raise SystemExit("GPS-buffer split produced an empty test split.")


if __name__ == "__main__":
    main()
