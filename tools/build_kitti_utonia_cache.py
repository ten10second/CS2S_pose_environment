import argparse
import json
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
    velo_to_camera2,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build Utonia per-point feature cache for KITTI raw LiDAR.")
    parser.add_argument("--manifest", default="dataset/kitti_raw_sat_lidar/train_manifest.jsonl")
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--path-rewrite",
        default="",
        help="Optional OLD=NEW replacement for manifest paths, e.g. /media/a=/media/b.",
    )
    parser.add_argument("--utonia-root", default="", help="Optional cloned Pointcept/Utonia repo path.")
    parser.add_argument("--repo-id", default="Pointcept/Utonia")
    parser.add_argument("--ckpt", default="utonia", help="'utonia' for HF download or local ckpt path.")
    parser.add_argument("--enable-flash", dest="disable_flash", action="store_false")
    parser.set_defaults(disable_flash=True)
    parser.add_argument("--scale", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--wo-color", action="store_true", help="Use zero color even if RGB projection is later added.")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def apply_path_rewrite(path: str, rule: str) -> str:
    if not rule:
        return path
    if "=" not in rule:
        raise ValueError("--path-rewrite must be OLD=NEW")
    old, new = rule.split("=", 1)
    return path.replace(old, new, 1)


def safe_sample_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "__")


def import_utonia(utonia_root: str):
    if utonia_root:
        sys.path.insert(0, str(Path(utonia_root).resolve()))
    try:
        import utonia  # noqa: WPS433
    except Exception as exc:
        raise RuntimeError(
            "Utonia is not importable. Install or clone official Pointcept/Utonia first. "
            "Recommended: use a Python>=3.10 torch>=2.5 CUDA>=12 env, then either "
            "`git clone https://github.com/Pointcept/Utonia third_party/Utonia` and pass "
            "`--utonia-root third_party/Utonia`, or run `python setup.py install` inside that repo. "
            f"Original import error: {exc}"
        ) from exc
    return utonia


def load_utonia_model(utonia, ckpt: str, repo_id: str, device: torch.device, disable_flash: bool):
    custom_config = {"enable_flash": False} if disable_flash else None
    if ckpt == "utonia":
        model = utonia.model.load("utonia", repo_id=repo_id, custom_config=custom_config)
    else:
        model = utonia.model.load(ckpt, custom_config=custom_config)
    return model.eval().to(device)


def select_camera_visible_points(record, path_rewrite, max_depth, image_size, max_points):
    calib = load_raw_calibration(apply_path_rewrite(record["calib_dir"], path_rewrite))
    velodyne_path = apply_path_rewrite(record["velodyne_path"], path_rewrite)
    points = load_velodyne_points(velodyne_path)
    if points.size == 0:
        return None
    points_xyz = points[:, :3].astype(np.float32)
    uv, depth, valid = project_velo_to_image(points_xyz, calib, image_size)
    camera_xyz = velo_to_camera2(points_xyz, calib)
    ranges = np.linalg.norm(points_xyz, axis=1).astype(np.float32)
    valid = (
        valid
        & np.isfinite(depth)
        & np.isfinite(camera_xyz).all(axis=1)
        & (depth > 0.0)
        & (depth <= float(max_depth))
        & np.isfinite(ranges)
        & (ranges > 0.0)
    )
    valid_indices = np.nonzero(valid)[0]
    projected_count = int(valid_indices.size)
    max_points = int(max_points)
    if projected_count == 0:
        return None
    if projected_count > max_points:
        out_h, out_w = image_size
        flat_uv = uv[valid_indices, 1] * float(out_w) + uv[valid_indices, 0]
        order = np.argsort(flat_uv, kind="mergesort")
        take = np.linspace(0, projected_count - 1, max_points).round().astype(np.int64)
        selected = valid_indices[order[take]]
    else:
        selected = valid_indices
    return {
        "coord_lidar": points_xyz[selected].astype(np.float32),
        "coord_camera": camera_xyz[selected].astype(np.float32),
        "intensity": np.clip(points[selected, 3].astype(np.float32), 0.0, 1.0),
        "uv": uv[selected].astype(np.float32),
        "depth": depth[selected].astype(np.float32),
        "selected_count": int(selected.size),
        "projected_count": projected_count,
    }


def pad_features(features, mask, max_points):
    max_points = int(max_points)
    if features.ndim == 1:
        features = features[:, None]
    out = np.zeros((max_points, features.shape[1]), dtype=np.float32)
    out_mask = np.zeros((max_points,), dtype=np.float32)
    n = min(max_points, features.shape[0], mask.shape[0])
    out[:n] = features[:n]
    out_mask[:n] = mask[:n]
    return out, out_mask


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    utonia = import_utonia(args.utonia_root)
    model = load_utonia_model(utonia, args.ckpt, args.repo_id, device, bool(args.disable_flash))
    transform = utonia.transform.default(scale=float(args.scale), apply_z_positive=False, normalize_coord=False)

    records = read_jsonl(args.manifest)
    if int(args.limit) > 0:
        records = records[: int(args.limit)]
    written = 0
    skipped = 0
    failed = 0
    for idx, record in enumerate(records):
        sample_id = record["sample_id"]
        out_path = out_root / f"{safe_sample_id(sample_id)}.npz"
        if args.skip_existing and out_path.is_file():
            skipped += 1
            continue
        try:
            selected = select_camera_visible_points(
                record,
                args.path_rewrite,
                max_depth=float(args.max_depth),
                image_size=(int(args.image_height), int(args.image_width)),
                max_points=int(args.max_points),
            )
            if selected is None:
                empty_feat = np.zeros((int(args.max_points), 1), dtype=np.float32)
                empty_mask = np.zeros((int(args.max_points),), dtype=np.float32)
                np.savez_compressed(
                    out_path,
                    utonia_feat=empty_feat,
                    lidar_point_features_mask=empty_mask,
                    sample_id=np.asarray(sample_id),
                    selected_count=np.asarray(0, dtype=np.int32),
                    projected_count=np.asarray(0, dtype=np.int32),
                )
                written += 1
                continue
            coord = selected["coord_lidar"]
            point = {
                "coord": coord,
                "color": np.zeros((coord.shape[0], 3), dtype=np.float32),
                "normal": np.zeros((coord.shape[0], 3), dtype=np.float32),
            }
            point = transform(point)
            for key in list(point.keys()):
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].to(device, non_blocking=True)
            with torch.no_grad():
                point = model(point)
                while "pooling_parent" in point.keys():
                    parent = point.pop("pooling_parent")
                    inverse = point.pop("pooling_inverse")
                    parent.feat = point.feat[inverse]
                    point = parent
                feat = point.feat[point.inverse].detach().float().cpu().numpy()
            mask = np.ones((coord.shape[0],), dtype=np.float32)
            feat_padded, mask_padded = pad_features(feat.astype(np.float32), mask, int(args.max_points))
            np.savez_compressed(
                out_path,
                utonia_feat=feat_padded,
                lidar_point_features_mask=mask_padded,
                coord_lidar=coord.astype(np.float32),
                coord_camera=selected["coord_camera"].astype(np.float32),
                uv=selected["uv"].astype(np.float32),
                depth=selected["depth"].astype(np.float32),
                intensity=selected["intensity"].astype(np.float32),
                sample_id=np.asarray(sample_id),
                selected_count=np.asarray(selected["selected_count"], dtype=np.int32),
                projected_count=np.asarray(selected["projected_count"], dtype=np.int32),
                model=np.asarray("utonia"),
            )
            written += 1
            if written == 1 or written % 50 == 0:
                print(json.dumps({"written": written, "idx": idx, "sample_id": sample_id, "out": str(out_path)}))
        except Exception as exc:
            failed += 1
            print(json.dumps({"failed": failed, "idx": idx, "sample_id": sample_id, "error": str(exc)}))
    print(json.dumps({"complete": True, "written": written, "skipped": skipped, "failed": failed, "out_root": str(out_root)}))


if __name__ == "__main__":
    main()
