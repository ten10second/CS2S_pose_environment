import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_pixel_feature_cache import (  # noqa: E402
    PIXEL_CACHE_FORMAT,
    PIXEL_DEPTH_KEY,
    PIXEL_FEATURE_KEY,
    PIXEL_INDEX_KEY,
    pixel_index_from_uv,
    validate_pixel_arrays,
)
from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    LIDAR_PROJECTION_VERSION,
    load_raw_calibration,
    load_velodyne_points,
    project_velo_to_image,
    read_jsonl,
    zbuffer_visible_point_indices,
)
from tools.build_kitti_utonia_ray_cache import (  # noqa: E402
    atomic_savez,
    load_utonia,
    resolve_calib_dir,
    rewrite_path,
    safe_sample_id,
    upsample_point_features,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build z-buffer-visible per-pixel Utonia features for KITTI image_02."
    )
    parser.add_argument("--manifest", default="dataset/kitti_raw_sat_lidar/train_manifest.jsonl")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--kitti-root", default="")
    parser.add_argument("--path-rewrite", default="")
    parser.add_argument("--utonia-root", default="third_party/Utonia")
    parser.add_argument("--repo-id", default="Pointcept/Utonia")
    parser.add_argument("--ckpt", default="utonia")
    parser.add_argument("--enable-flash", dest="disable_flash", action="store_false")
    parser.set_defaults(disable_flash=True)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=576)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def process_record(record, model, transform, device, args):
    velodyne_path = rewrite_path(record["velodyne_path"], args)
    calib_dir = resolve_calib_dir(record, args)
    points = load_velodyne_points(velodyne_path)
    calib = load_raw_calibration(calib_dir)
    if points.size == 0:
        raise ValueError("empty Velodyne scan")

    xyz = points[:, :3].astype(np.float32)
    ranges = np.linalg.norm(xyz, axis=1)
    encoder_mask = np.isfinite(xyz).all(axis=1) & np.isfinite(ranges) & (ranges > 0.0) & (
        ranges <= float(args.max_depth)
    )
    encoder_indices = np.nonzero(encoder_mask)[0]
    encoder_xyz = xyz[encoder_indices]
    if encoder_xyz.size == 0:
        raise ValueError("no finite LiDAR points remain for Utonia encoding")

    output_size = (int(args.image_height), int(args.image_width))
    raw_uv, raw_depth, raw_projected = project_velo_to_image(xyz, calib, output_size)
    raw_projected = raw_projected & np.isfinite(raw_depth) & (raw_depth > 0.0)
    if not np.any(raw_projected):
        raise ValueError("no LiDAR points project into image_02")
    raw_zbuffer_indices = zbuffer_visible_point_indices(raw_uv, raw_depth, raw_projected, output_size)
    if raw_zbuffer_indices.size == 0:
        raise ValueError("no z-buffer-visible LiDAR points remain in image_02")
    encoder_row_by_raw_index = np.full((xyz.shape[0],), -1, dtype=np.int64)
    encoder_row_by_raw_index[encoder_indices] = np.arange(encoder_indices.shape[0], dtype=np.int64)
    zbuffer_encoder_rows = encoder_row_by_raw_index[raw_zbuffer_indices]
    keep = zbuffer_encoder_rows >= 0
    if not np.any(keep):
        raise ValueError("no z-buffer-visible LiDAR front-surface points remain for Utonia encoding")
    kept_raw_indices = raw_zbuffer_indices[keep]
    kept_encoder_rows = zbuffer_encoder_rows[keep]

    point = {
        "coord": encoder_xyz,
        "color": np.zeros((encoder_xyz.shape[0], 3), dtype=np.float32),
        "normal": np.zeros((encoder_xyz.shape[0], 3), dtype=np.float32),
    }
    point = transform(point)
    for key in list(point.keys()):
        if isinstance(point[key], torch.Tensor):
            point[key] = point[key].to(device, non_blocking=True)
    with torch.inference_mode():
        point = model(point)
        features = upsample_point_features(point).detach().float().cpu().numpy()
    if features.shape[0] != encoder_xyz.shape[0]:
        raise RuntimeError(
            f"Utonia output point count {features.shape[0]} != encoder input {encoder_xyz.shape[0]}"
        )

    pixel_payload = validate_pixel_arrays(
        features[kept_encoder_rows],
        pixel_index_from_uv(raw_uv, kept_raw_indices, output_size),
        raw_depth[kept_raw_indices],
        output_size,
        feature_dim=int(args.feature_dim),
    )
    pixel_count = int(pixel_payload[PIXEL_INDEX_KEY].size)
    return {
        PIXEL_FEATURE_KEY: pixel_payload[PIXEL_FEATURE_KEY],
        PIXEL_INDEX_KEY: pixel_payload[PIXEL_INDEX_KEY],
        PIXEL_DEPTH_KEY: pixel_payload[PIXEL_DEPTH_KEY],
        "source_point_index": kept_raw_indices,
        "projection_version": np.asarray(LIDAR_PROJECTION_VERSION),
        "format": np.asarray(PIXEL_CACHE_FORMAT),
        "image_height": np.asarray(int(args.image_height), dtype=np.int32),
        "image_width": np.asarray(int(args.image_width), dtype=np.int32),
        "feature_dim": np.asarray(int(args.feature_dim), dtype=np.int32),
        "encoder_point_count": np.asarray(encoder_xyz.shape[0], dtype=np.int32),
        "projected_point_count": np.asarray(int(raw_projected.sum()), dtype=np.int32),
        "raw_zbuffer_visible_point_count": np.asarray(int(raw_zbuffer_indices.size), dtype=np.int32),
        "zbuffer_visible_point_count": np.asarray(pixel_count, dtype=np.int32),
    }


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    np.random.seed(int(args.seed) + int(args.shard_index))
    torch.manual_seed(int(args.seed) + int(args.shard_index))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, transform = load_utonia(args, device)
    records = list(enumerate(read_jsonl(args.manifest)))[args.shard_index :: args.num_shards]
    if args.limit > 0:
        records = records[: int(args.limit)]

    written = skipped = failed = 0
    for index, record in records:
        sample_id = record["sample_id"]
        out_path = out_root / f"{safe_sample_id(sample_id)}.npz"
        if args.skip_existing and out_path.is_file():
            skipped += 1
            continue
        try:
            payload = process_record(record, model, transform, device, args)
            atomic_savez(out_path, sample_id=np.asarray(sample_id), **payload)
            written += 1
            if written == 1 or written % 25 == 0:
                print(
                    json.dumps(
                        {
                            "written": written,
                            "index": index,
                            "sample_id": sample_id,
                            "encoder_points": int(payload["encoder_point_count"]),
                            "projected_points": int(payload["projected_point_count"]),
                            "zbuffer_visible_points": int(payload["zbuffer_visible_point_count"]),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except Exception as exc:
            failed += 1
            print(
                json.dumps({"failed": failed, "index": index, "sample_id": sample_id, "error": str(exc)}),
                flush=True,
            )
    summary = {
        "complete": failed == 0,
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "out_root": str(out_root),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(f"Utonia pixel cache failed for {failed} samples")


if __name__ == "__main__":
    main()
