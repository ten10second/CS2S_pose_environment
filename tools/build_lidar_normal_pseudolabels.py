import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_lidar_normal import normal_label_path  # noqa: E402
from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    load_raw_calibration,
    load_velodyne_points,
    project_velo_to_image,
    read_jsonl,
    velo_to_rect,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-point LiDAR normal pseudo-label cache from RGB teacher.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", default="dataset/kitti_lidar_normal_labels")
    parser.add_argument("--teacher", default="metric3d_hub", choices=["metric3d_hub", "lidar_pca"])
    parser.add_argument("--metric3d-model", default="metric3d_vit_small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--depth-abs-tol", type=float, default=1.5)
    parser.add_argument("--depth-rel-tol", type=float, default=0.08)
    parser.add_argument("--max-depth", type=float, default=120.0)
    parser.add_argument("--min-label-points", type=int, default=64)
    parser.add_argument("--metric3d-input-height", type=int, default=616)
    parser.add_argument("--metric3d-input-width", type=int, default=1064)
    return parser.parse_args()


class Metric3DHubTeacher:
    def __init__(self, model_name: str, device: str, input_size):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        try:
            self.model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True, trust_repo=True)
        except TypeError:
            self.model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)
        self.model = self.model.to(self.device).eval()
        self.input_size = tuple(int(v) for v in input_size)
        self.mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)[:, None, None]

    def _prepare(self, image: Image.Image, fx: float):
        rgb_origin = np.asarray(image.convert("RGB"), dtype=np.float32)
        orig_h, orig_w = rgb_origin.shape[:2]
        input_h, input_w = self.input_size
        scale = min(input_h / orig_h, input_w / orig_w)
        resized_w = int(orig_w * scale)
        resized_h = int(orig_h * scale)
        resized = np.asarray(
            image.convert("RGB").resize((resized_w, resized_h), Image.BILINEAR),
            dtype=np.float32,
        )
        pad_value = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
        pad_h = input_h - resized_h
        pad_w = input_w - resized_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = np.pad(
            resized,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        if pad_top:
            padded[:pad_top, :, :] = pad_value
        if pad_bottom:
            padded[-pad_bottom:, :, :] = pad_value
        if pad_left:
            padded[:, :pad_left, :] = pad_value
        if pad_right:
            padded[:, -pad_right:, :] = pad_value
        tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float()
        tensor = (tensor - self.mean) / self.std
        tensor = tensor[None].to(self.device)
        return tensor, (orig_h, orig_w), (pad_top, pad_bottom, pad_left, pad_right), fx * scale

    @torch.no_grad()
    def predict(self, image: Image.Image, fx: float):
        rgb, orig_shape, pad, scaled_fx = self._prepare(image, fx)
        pred_depth, confidence, output_dict = self.model.inference({"input": rgb})
        pred_depth = pred_depth.squeeze()
        pad_top, pad_bottom, pad_left, pad_right = pad
        depth = pred_depth[
            pad_top : pred_depth.shape[0] - pad_bottom,
            pad_left : pred_depth.shape[1] - pad_right,
        ]
        depth = F.interpolate(depth[None, None], size=orig_shape, mode="bilinear", align_corners=False)[0, 0]
        depth = torch.clamp(depth * (scaled_fx / 1000.0), 0.0, 300.0)

        if "prediction_normal" not in output_dict:
            raise RuntimeError("Metric3D output_dict does not contain prediction_normal; use a v2 ViT model.")
        normal_pack = output_dict["prediction_normal"].squeeze(0)
        normal = normal_pack[:3]
        normal = normal[
            :,
            pad_top : normal.shape[1] - pad_bottom,
            pad_left : normal.shape[2] - pad_right,
        ]
        normal = F.interpolate(normal[None], size=orig_shape, mode="bilinear", align_corners=False)[0]
        normal = F.normalize(normal, dim=0, eps=1e-6)
        if normal_pack.shape[0] > 3:
            normal_conf = normal_pack[3:4]
            normal_conf = normal_conf[
                :,
                pad_top : normal_conf.shape[1] - pad_bottom,
                pad_left : normal_conf.shape[2] - pad_right,
            ]
            normal_conf = F.interpolate(normal_conf[None], size=orig_shape, mode="bilinear", align_corners=False)[0, 0]
            normal_conf = normal_conf.clamp(0.0, 1.0)
        else:
            normal_conf = torch.ones_like(depth)
        return (
            depth.detach().cpu().numpy().astype(np.float32),
            normal.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32),
            normal_conf.detach().cpu().numpy().astype(np.float32),
        )


def _zbuffer_keep(uv, depth, valid, image_shape):
    h, w = image_shape
    x = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
    flat = y * w + x
    nearest = np.full((h * w,), np.inf, dtype=np.float32)
    valid_indices = np.nonzero(valid)[0]
    np.minimum.at(nearest, flat[valid_indices], depth[valid_indices])
    keep = valid & (depth <= nearest[flat] + 1e-4)
    return keep, x, y


def _lidar_pca_normals(points_rect, keep_mask, k=24):
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:
        raise RuntimeError("lidar_pca teacher requires scipy.spatial.cKDTree") from exc
    kept = points_rect[keep_mask]
    if kept.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    query_k = min(k, kept.shape[0])
    tree = cKDTree(kept)
    _, nn_idx = tree.query(kept, k=query_k)
    if query_k == 1:
        nn_idx = nn_idx[:, None]
    normals = np.zeros_like(kept, dtype=np.float32)
    weights = np.zeros((kept.shape[0],), dtype=np.float32)
    for idx, neighbors in enumerate(nn_idx):
        local = kept[np.asarray(neighbors)]
        centered = local - local.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(local.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        normals[idx] = normal / max(np.linalg.norm(normal), 1e-6)
        weights[idx] = float(np.clip(1.0 - eigvals.min() / max(eigvals.sum(), 1e-6), 0.0, 1.0))
    return normals, weights


def build_one(record, args, teacher):
    image = Image.open(record["image_02_path"]).convert("RGB")
    image_w, image_h = image.size
    image_shape = (image_h, image_w)
    calib = load_raw_calibration(record["calib_dir"])
    p2 = calib["P_rect_02"]
    points = load_velodyne_points(record["velodyne_path"])
    points_xyz = points[:, :3] if points.size else np.zeros((0, 3), dtype=np.float32)
    intensity = points[:, 3] if points.size and points.shape[1] >= 4 else np.zeros((points_xyz.shape[0],), dtype=np.float32)
    points_rect = velo_to_rect(points_xyz, calib)
    uv, depth, valid = project_velo_to_image(points_xyz, calib, output_size=image_shape)
    valid = valid & (depth > 0.0) & (depth <= args.max_depth)
    z_keep, pix_x, pix_y = _zbuffer_keep(uv, depth, valid, image_shape)

    if args.teacher == "metric3d_hub":
        metric_depth, normal_map, normal_conf = teacher.predict(image, float(p2[0, 0]))
        sampled_depth = metric_depth[pix_y, pix_x]
        sampled_normal = normal_map[pix_y, pix_x]
        sampled_conf = normal_conf[pix_y, pix_x]
        depth_tol = np.maximum(args.depth_abs_tol, args.depth_rel_tol * np.maximum(depth, 1.0))
        depth_error = np.abs(depth - sampled_depth)
        visible = z_keep & np.isfinite(sampled_depth) & (sampled_depth > 0.0) & (depth_error <= depth_tol)
        depth_weight = np.exp(-depth_error / np.maximum(depth_tol, 1e-6)).astype(np.float32)
        label_weight = np.clip(sampled_conf * depth_weight, 0.0, 1.0).astype(np.float32)
        normals = sampled_normal.astype(np.float32)
    else:
        visible = z_keep
        pca_normals, pca_weights = _lidar_pca_normals(points_rect, visible)
        normals = np.zeros((points_rect.shape[0], 3), dtype=np.float32)
        label_weight = np.zeros((points_rect.shape[0],), dtype=np.float32)
        normals[visible] = pca_normals
        label_weight[visible] = pca_weights * 0.75

    normal_norm = np.linalg.norm(normals, axis=1)
    visible = visible & np.isfinite(normals).all(axis=1) & (normal_norm > 1e-6) & (label_weight > 1e-4)
    point_indices = np.nonzero(visible)[0].astype(np.int64)
    out = {
        "point_indices": point_indices,
        "points_rect": points_rect[visible].astype(np.float32),
        "intensity": intensity[visible].astype(np.float32),
        "uv_orig": uv[visible].astype(np.float32),
        "depth_rect": depth[visible].astype(np.float32),
        "normal_rect": (normals[visible] / np.maximum(normal_norm[visible, None], 1e-6)).astype(np.float32),
        "label_weight": label_weight[visible].astype(np.float32),
        "visibility_valid": visible[visible].astype(np.float32),
        "image_shape": np.asarray([image_h, image_w], dtype=np.int64),
    }
    return out


def main():
    args = parse_args()
    records = read_jsonl(args.manifest)
    selected = records[args.start_index :]
    if args.max_samples > 0:
        selected = selected[: args.max_samples]
    teacher = None
    if args.teacher == "metric3d_hub":
        teacher = Metric3DHubTeacher(
            args.metric3d_model,
            args.device,
            (args.metric3d_input_height, args.metric3d_input_width),
        )

    out_root = Path(args.out_root)
    stats = {
        "manifest": args.manifest,
        "teacher": args.teacher,
        "metric3d_model": args.metric3d_model,
        "num_requested": len(selected),
        "num_written": 0,
        "num_skipped_existing": 0,
        "num_too_few_points": 0,
        "num_failed": 0,
        "records": [],
    }
    for local_idx, record in enumerate(selected):
        out_path = normal_label_path(str(out_root), record["sample_id"])
        if out_path.exists() and not args.overwrite:
            stats["num_skipped_existing"] += 1
            continue
        try:
            payload = build_one(record, args, teacher)
            count = int(payload["points_rect"].shape[0])
            if count < args.min_label_points:
                stats["num_too_few_points"] += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, **payload)
            stats["num_written"] += 1
            stats["records"].append({"sample_id": record["sample_id"], "label_points": count, "path": str(out_path)})
            if stats["num_written"] % 25 == 0:
                print(json.dumps({"written": stats["num_written"], "last": record["sample_id"]}, sort_keys=True), flush=True)
        except Exception as exc:
            stats["num_failed"] += 1
            print(json.dumps({"failed": record.get("sample_id"), "error": repr(exc)}, sort_keys=True), flush=True)
        if local_idx % 100 == 0:
            torch.cuda.empty_cache()

    summary_path = out_root / "pseudolabel_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in stats.items() if k != "records"}, indent=2, sort_keys=True))
    if stats["num_written"] <= 0 and stats["num_skipped_existing"] <= 0:
        raise SystemExit("No pseudo-label files were available.")


if __name__ == "__main__":
    main()
