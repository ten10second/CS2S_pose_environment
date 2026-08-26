import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    load_raw_calibration,
    load_velodyne_points,
    project_velo_to_image,
    read_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build full-scan Utonia features pooled into KITTI camera ray-depth bins."
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
    parser.add_argument("--ray-depth-bins", type=int, default=4)
    parser.add_argument("--ray-height", type=int, default=8)
    parser.add_argument("--ray-width", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def safe_sample_id(sample_id):
    return str(sample_id).replace("/", "__")


def atomic_savez(path: Path, **arrays) -> None:
    temp_path = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    temp_path.unlink(missing_ok=True)
    try:
        np.savez_compressed(temp_path, **arrays)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def rewrite_path(path, args):
    value = str(path)
    if args.path_rewrite:
        if "=" not in args.path_rewrite:
            raise ValueError("--path-rewrite must be OLD=NEW")
        old, new = args.path_rewrite.split("=", 1)
        value = value.replace(old, new, 1)
    if args.kitti_root:
        marker = "/KITTI_RAW/"
        if marker in value:
            value = str(Path(args.kitti_root) / value.split(marker, 1)[1])
    return value


def resolve_calib_dir(record, args):
    """Resolve calibration directories from manifests with legacy path variants."""
    value = Path(rewrite_path(record["calib_dir"], args))
    if (value / "calib_cam_to_cam.txt").is_file() and (value / "calib_velo_to_cam.txt").is_file():
        return str(value)

    date = str(record.get("date", ""))
    if args.kitti_root and date:
        candidates = (
            Path(args.kitti_root) / date / f"{date}_calib",
            Path(args.kitti_root) / date,
        )
        for candidate in candidates:
            if (candidate / "calib_cam_to_cam.txt").is_file() and (
                candidate / "calib_velo_to_cam.txt"
            ).is_file():
                return str(candidate)
    return str(value)


def load_utonia(args, device):
    if args.utonia_root:
        sys.path.insert(0, str(Path(args.utonia_root).resolve()))
    import utonia  # noqa: WPS433

    custom_config = {"enable_flash": False} if args.disable_flash else None
    if args.ckpt == "utonia":
        model = utonia.model.load("utonia", repo_id=args.repo_id, custom_config=custom_config)
    else:
        model = utonia.model.load(args.ckpt, custom_config=custom_config)
    transform = utonia.transform.default(
        scale=float(args.scale),
        apply_z_positive=False,
        normalize_coord=False,
    )
    return model.eval().to(device), transform


def upsample_point_features(point):
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point.feat[point.inverse]


def pool_ray_depth_features(features, uv, depth, args):
    bins = int(args.ray_depth_bins)
    ray_h = int(args.ray_height)
    ray_w = int(args.ray_width)
    feat_dim = int(features.shape[1])
    row = np.floor(uv[:, 1] / float(args.image_height) * ray_h).astype(np.int64)
    col = np.floor(uv[:, 0] / float(args.image_width) * ray_w).astype(np.int64)
    row = np.clip(row, 0, ray_h - 1)
    col = np.clip(col, 0, ray_w - 1)
    depth_coord = np.log1p(depth) / np.log1p(float(args.max_depth))
    depth_bin = np.floor(depth_coord * bins).astype(np.int64)
    depth_bin = np.clip(depth_bin, 0, bins - 1)
    flat_index = depth_bin * ray_h * ray_w + row * ray_w + col

    sums = np.zeros((bins * ray_h * ray_w, feat_dim), dtype=np.float32)
    counts = np.zeros((bins * ray_h * ray_w,), dtype=np.float32)
    np.add.at(sums, flat_index, features.astype(np.float32, copy=False))
    np.add.at(counts, flat_index, 1.0)
    pooled = sums / np.maximum(counts[:, None], 1.0)
    pooled = pooled.reshape(bins, ray_h, ray_w, feat_dim).transpose(3, 0, 1, 2)
    mask = (counts.reshape(bins, ray_h, ray_w) > 0.0)[None]
    return pooled.astype(np.float16), mask.astype(np.uint8), counts


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

    uv, depth, visible = project_velo_to_image(
        encoder_xyz,
        calib,
        (int(args.image_height), int(args.image_width)),
    )
    visible = visible & np.isfinite(depth) & (depth > 0.0) & (depth <= float(args.max_depth))
    if not np.any(visible):
        raise ValueError("no LiDAR points project into image_02")

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

    pooled, mask, counts = pool_ray_depth_features(features[visible], uv[visible], depth[visible], args)
    return {
        "utonia_ray_feat": pooled,
        "utonia_ray_mask": mask,
        "encoder_point_count": np.asarray(encoder_xyz.shape[0], dtype=np.int32),
        "front_visible_point_count": np.asarray(int(visible.sum()), dtype=np.int32),
        "occupied_ray_depth_bins": np.asarray(int((counts > 0.0).sum()), dtype=np.int32),
        "feature_dim": np.asarray(features.shape[1], dtype=np.int32),
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
                            "front_visible_points": int(payload["front_visible_point_count"]),
                            "occupied_bins": int(payload["occupied_ray_depth_bins"]),
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
        raise SystemExit(f"Utonia ray cache failed for {failed} samples")


if __name__ == "__main__":
    main()
