import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    load_raw_calibration,
    load_velodyne_points,
    parse_tracklet_xml,
    points_in_boxes,
    project_velo_to_image,
    scaled_camera_k,
)
from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler  # noqa: E402
from models.eval.dynamic_metrics import dynamic_masked_metrics  # noqa: E402
from models.eval.evaluate import Evaluate_indic  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402


MODES = ("none", "bbox_dynamic", "dynamic_full")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate KITTI raw sat-lidar checkpoints on fixed validation samples.")
    parser.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dynamic.yaml")
    parser.add_argument("--manifest", default="dataset/kitti_raw_sat_lidar/val_manifest.jsonl")
    parser.add_argument("--none-ckpt", default="")
    parser.add_argument("--bbox-dynamic-ckpt", default="")
    parser.add_argument("--dynamic-full-ckpt", default="")
    parser.add_argument("--out-dir", default="results/sat_lidar_dynamic/eval_compare")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--control-scale-override", type=float, default=None, help="Override LiDAR control residual scale at eval time.")
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--skip-metrics", action="store_true", help="Only generate and save images; skip full/dynamic metric computation.")
    parser.add_argument("--depth-consistency", action="store_true", help="Run MiDaS/DPT-vs-LiDAR depth consistency sanity metrics.")
    parser.add_argument("--depth-images-dir", default="", help="Existing eval images directory. Defaults to <out-dir>/images.")
    parser.add_argument("--depth-modes", default="none,dynamic_full", help="Comma-separated modes to read when --depth-consistency is used without checkpoints.")
    parser.add_argument("--midas-model", default="MiDaS_small", help="torch.hub MiDaS model name, e.g. MiDaS_small or DPT_Hybrid.")
    parser.add_argument("--midas-hub-dir", default="ckpt/midas_torch/hub", help="Local torch hub cache directory for MiDaS.")
    parser.add_argument("--depth-device", default="cuda")
    parser.add_argument("--depth-max-depth", type=float, default=80.0)
    parser.add_argument("--object-crop-panels", action="store_true", help="Save per-dynamic-box crop panels for visual diagnosis.")
    parser.add_argument("--object-crop-padding", type=int, default=8)
    parser.add_argument("--object-crop-size", type=int, default=160)
    parser.add_argument("--object-crop-max-boxes", type=int, default=8)
    return parser.parse_args()


def mode_ckpts(args):
    return {
        "none": args.none_ckpt,
        "bbox_dynamic": args.bbox_dynamic_ckpt,
        "dynamic_full": args.dynamic_full_ckpt,
    }


def configure_for_mode(cfg, mode):
    cfg.data.params.batch_size = 1
    cfg.data.params.num_workers = 0
    cfg.data.params.train.params.condition_mode = mode
    cfg.data.params.test.params.condition_mode = mode
    if mode == "none":
        cfg.model.params.use_lidar_cond = False
        cfg.model.params.freeze_for_lidar_control = False
    else:
        cfg.model.params.use_lidar_cond = True
        cfg.model.params.freeze_for_lidar_control = True
    return cfg


def configure_from_checkpoint(cfg, mode, payload):
    if mode != "none" and "control_grd" in payload:
        control_state = payload["control_grd"]
        control_keys = set(control_state.keys())
        if any(key.startswith(("stem.", "down_blocks.", "skip_outs.", "middle_out.")) for key in control_keys):
            unet = cfg.model.params.DDPM_config.params.unet_config.params
            control = cfg.model.params.DDPM_config.params.control_grd
            control.target = "models.KITTI_geo_ldm.lidar_condition_model.LidarMultiScaleControl"
            control.params.in_channels = 4
            control.params.model_channels = unet.model_channels
            control.params.channel_mult = list(unet.channel_mult)
            control.params.num_res_blocks = unet.num_res_blocks
            control.params.middle_channels = unet.model_channels * list(unet.channel_mult)[-1]
            if "stem.0.weight" in control_state:
                control.params.hidden_channels = int(control_state["stem.0.weight"].shape[0])
            elif "hidden_channels" not in control.params:
                control.params.hidden_channels = 128
            if "class_embedding.weight" in control_state:
                control.params.semantic_class_count = int(control_state["class_embedding.weight"].shape[0])
                control.params.semantic_class_scale = 1.0
        if "dynamic_class_tokens" in payload:
            tokens = payload["dynamic_class_tokens"]
            cfg.model.params.dynamic_class_token_weight = 1.0
            cfg.model.params.dynamic_class_token_count = int(tokens.shape[0])
            cfg.model.params.dynamic_class_token_dim = int(tokens.shape[1])
    return cfg


def load_model(config_path, mode, ckpt_path, control_scale_override=None):
    cfg = configure_for_mode(OmegaConf.load(config_path), mode)
    payload = torch.load(ckpt_path, map_location="cpu") if ckpt_path else None
    if payload is not None:
        cfg = configure_from_checkpoint(cfg, mode, payload)
    if control_scale_override is not None and mode != "none":
        cfg.model.params.DDPM_config.params.control_grd.params.control_scale = float(control_scale_override)
    model = instantiate_from_config(cfg.model).cuda().eval()
    if payload is not None:
        base_model_ckpt = payload.get("base_model_ckpt", "")
        if base_model_ckpt and mode != "none":
            base_payload = torch.load(base_model_ckpt, map_location="cpu")
            if "model" in base_payload:
                model.load_state_dict(base_payload["model"], strict=False)
            elif "state_dict" in base_payload:
                model.load_state_dict(base_payload["state_dict"], strict=False)
            else:
                raise ValueError(f"Unsupported base checkpoint format for {mode}: {base_model_ckpt}")
            del base_payload
            gc.collect()
        if mode == "none" and "model" in payload:
            model.load_state_dict(payload["model"], strict=False)
        elif "control_grd" in payload:
            model.DDPM.control_grd.load_state_dict(payload["control_grd"], strict=False)
            if "dynamic_class_tokens" in payload and getattr(model, "dynamic_class_tokens", None) is not None:
                model.dynamic_class_tokens.data.copy_(payload["dynamic_class_tokens"].to(model.dynamic_class_tokens.device))
            if "denoise_model_trainable" in payload:
                model.DDPM.denoise_model.load_state_dict(payload["denoise_model_trainable"], strict=False)
        elif "state_dict" in payload:
            model.load_state_dict(payload["state_dict"], strict=False)
        else:
            raise ValueError(f"Unsupported checkpoint format for {mode}: {ckpt_path}")
        del payload
        gc.collect()
    return model


def sample_to_batch(sample):
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).cuda(non_blocking=True)
        else:
            batch[key] = [value]
    return batch


def tensor_to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def safe_sample_id(sample_id):
    return sample_id.replace("/", "__")


def save_tensor_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    transforms.functional.to_pil_image(tensor.detach().cpu().clamp(0, 1)).save(path)


def make_lidar_overlay(gt, lidar_cond, dynamic_mask):
    base = transforms.functional.to_pil_image(gt.detach().cpu().clamp(0, 1)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    mask = dynamic_mask.detach().cpu()[0] > 0.5
    point = lidar_cond.detach().cpu()[1] > 0.5
    depth = lidar_cond.detach().cpu()[2].clamp(0, 1)
    width, height = base.size
    for y in range(height):
        for x in range(width):
            if bool(mask[y, x]):
                draw.point((x, y), fill=(255, 32, 32, 70))
            if bool(point[y, x]):
                d = int(255 * float(depth[y, x]))
                draw.point((x, y), fill=(32, 220, max(64, d), 220))
    return Image.alpha_composite(base, overlay).convert("RGB")


def make_cond_rgb(lidar_cond):
    cond = lidar_cond.detach().cpu().clamp(0, 1)
    if cond.shape[0] < 3:
        cond = F.pad(cond, (0, 0, 0, 0, 0, 3 - cond.shape[0]))
    return cond[:3]


@torch.no_grad()
def generate_prediction(model, batch, mode, ddim_steps, seed, guidance_scale, eta, temperature):
    inputs = model.get_input(batch, "sat_map").cuda()
    outputs = model.get_input(batch, "grd_left_imgs").cuda()
    lidar_cond = None
    if mode != "none" and model.lidar_condition_key in batch:
        lidar_cond = model.get_input(batch, model.lidar_condition_key).cuda()

    left_camera_k = model.get_input(batch, "left_camera_k").squeeze(-1).cuda()
    gt_shift_x = batch["gt_shift_x"].cuda()
    gt_shift_y = batch["gt_shift_y"].cuda()
    theta = batch["theta"].cuda()

    inputs = inputs * 2 - 1
    outputs = outputs * 2 - 1
    if hasattr(model, "make_condition"):
        cond_label = model.make_condition(inputs, batch).detach()
    else:
        cond_label = model.condition_model_sat(inputs)[:, 1:, :].detach()

    torch.manual_seed(seed)
    x_t = torch.randn((cond_label.shape[0], 4, 16, 64), device=inputs.device)
    sampler = KITTI_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor)
    samples_ddim, _ = sampler.sample(
        S=ddim_steps,
        cond_sat=None,
        cond_grd=None,
        conditioning=cond_label,
        batch_size=cond_label.shape[0],
        shape=[4, 16, 64],
        verbose=False,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=None,
        eta=eta,
        x_T=x_t,
        temperature=temperature,
        left_camera_k=left_camera_k,
        gt_shift_x=gt_shift_x,
        gt_shift_y=gt_shift_y,
        theta=theta,
        cond_init_grd=lidar_cond,
    )
    pred = model.pre_AE_model.decode(samples_ddim * (1 / model.scale_factor))
    pred = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0)
    target = torch.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)
    return pred, target


def full_metrics(evaluator, pred, target):
    log = evaluator(pred.clamp(0, 1), target.clamp(0, 1), split="test")
    return {
        "full_rmse": float(log["RMSE"]),
        "full_ssim": float(log["SSIM"]),
        "full_psnr": float(log["PSNR"]),
        "full_sd": float(log["SD"]),
        "full_lpips_alex": tensor_to_float(log["P_alex"]),
        "full_lpips_squeeze": tensor_to_float(log["P_squeeze"]),
    }


def dynamic_metrics(evaluator, pred, target, mask):
    log = dynamic_masked_metrics(pred, target, mask.to(pred.device), evaluator.loss_fn_alex)
    return {
        "dynamic_psnr": tensor_to_float(log["dynamic_psnr"]),
        "dynamic_ssim": tensor_to_float(log["dynamic_ssim"]),
        "dynamic_lpips": tensor_to_float(log["dynamic_lpips"]),
        "dynamic_mask_coverage": tensor_to_float(log["dynamic_mask_coverage"]),
        "dynamic_valid_images": int(log["dynamic_valid_images"].detach().cpu()),
    }


def mean_finite(values):
    finite = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def aggregate(records):
    numeric_keys = sorted(
        key
        for record in records
        for key, value in record.items()
        if isinstance(value, (int, float))
    )
    return {key: mean_finite([record.get(key, float("nan")) for record in records]) for key in numeric_keys}


def build_success_checks(summary):
    required = set(MODES)
    if not required.issubset(summary):
        return {"ready": False, "reason": "missing one or more required modes"}
    none = summary["none"]
    bbox = summary["bbox_dynamic"]
    full = summary["dynamic_full"]
    metric_keys = {"dynamic_lpips", "dynamic_psnr", "dynamic_ssim", "full_lpips_alex", "full_psnr"}
    if any(not metric_keys.issubset(record) for record in (none, bbox, full)):
        return {"ready": False, "reason": "metrics were skipped or incomplete"}
    checks = {
        "ready": True,
        "dynamic_full_lpips_lt_none": full["dynamic_lpips"] < none["dynamic_lpips"],
        "dynamic_full_psnr_gt_none": full["dynamic_psnr"] > none["dynamic_psnr"],
        "dynamic_full_better_than_bbox_any": (
            full["dynamic_lpips"] < bbox["dynamic_lpips"]
            or full["dynamic_psnr"] > bbox["dynamic_psnr"]
            or full["dynamic_ssim"] > bbox["dynamic_ssim"]
        ),
        "dynamic_full_full_lpips_le_none_plus_0_03": full["full_lpips_alex"] <= none["full_lpips_alex"] + 0.03,
        "dynamic_full_full_psnr_ge_none_minus_1db": full["full_psnr"] >= none["full_psnr"] - 1.0,
    }
    checks["passed"] = all(value for key, value in checks.items() if key not in {"ready", "passed"})
    return checks


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_panel(out_dir, sample_id, gt_path, overlay_path, pred_paths):
    labels = ["GT", "LiDAR overlay"] + list(pred_paths.keys())
    images = [Image.open(gt_path).convert("RGB"), Image.open(overlay_path).convert("RGB")]
    images.extend(Image.open(path).convert("RGB") for path in pred_paths.values())
    width, height = images[0].size
    label_h = 22
    panel = Image.new("RGB", (width * len(images), height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    for idx, (label, image) in enumerate(zip(labels, images)):
        panel.paste(image, (idx * width, label_h))
        draw.text((idx * width + 4, 4), label, fill=(0, 0, 0))
    panel_path = out_dir / "panels" / f"{safe_sample_id(sample_id)}.png"
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(panel_path)
    return str(panel_path)


def _draw_crop_with_box(image, crop_box, source_box, size):
    crop = image.crop(crop_box).convert("RGB")
    draw = ImageDraw.Draw(crop)
    rel_box = [
        max(0, int(source_box[0] - crop_box[0])),
        max(0, int(source_box[1] - crop_box[1])),
        min(crop.width - 1, int(source_box[2] - crop_box[0])),
        min(crop.height - 1, int(source_box[3] - crop_box[1])),
    ]
    if rel_box[2] > rel_box[0] and rel_box[3] > rel_box[1]:
        draw.rectangle(rel_box, outline=(255, 32, 32), width=2)
    return crop.resize((size, size), Image.BICUBIC)


def make_object_crop_panels(out_dir, sample_id, gt_path, overlay_path, pred_paths, boxes, valid, padding, size, max_boxes):
    images = {"GT": Image.open(gt_path).convert("RGB"), "LiDAR overlay": Image.open(overlay_path).convert("RGB")}
    images.update({mode: Image.open(path).convert("RGB") for mode, path in pred_paths.items()})
    width, height = images["GT"].size
    labels = list(images.keys())
    records = []
    label_h = 22
    valid_indices = [idx for idx, flag in enumerate(valid) if float(flag) > 0.5]
    for out_idx, box_idx in enumerate(valid_indices[:max_boxes]):
        box = boxes[box_idx]
        x0, y0, x1, y1, class_id = [float(value) for value in box[:5]]
        if x1 <= x0 or y1 <= y0:
            continue
        crop_box = (
            max(0, int(math.floor(x0)) - padding),
            max(0, int(math.floor(y0)) - padding),
            min(width, int(math.ceil(x1)) + padding + 1),
            min(height, int(math.ceil(y1)) + padding + 1),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            continue
        crops = [_draw_crop_with_box(image, crop_box, (x0, y0, x1, y1), size) for image in images.values()]
        panel = Image.new("RGB", (size * len(crops), size + label_h), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        for col, (label, crop) in enumerate(zip(labels, crops)):
            panel.paste(crop, (col * size, label_h))
            draw.text((col * size + 4, 4), label, fill=(0, 0, 0))
        panel_path = out_dir / "object_panels" / f"{safe_sample_id(sample_id)}__box{out_idx:02d}_c{int(class_id)}.png"
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel.save(panel_path)
        records.append(
            {
                "sample_id": sample_id,
                "box_index": int(box_idx),
                "class_id": int(class_id),
                "source_box_xyxy": [x0, y0, x1, y1],
                "crop_box_xyxy": list(crop_box),
                "panel_path": str(panel_path),
            }
        )
    for image in images.values():
        image.close()
    return records


def parse_mode_list(value):
    return [mode.strip() for mode in value.split(",") if mode.strip()]


def torch_hub_load(repo, name):
    try:
        return torch.hub.load(repo, name, trust_repo=True)
    except TypeError:
        return torch.hub.load(repo, name)


def load_midas_depth_model(model_name, device_name, hub_dir):
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    if hub_dir:
        torch.hub.set_dir(str((REPO_ROOT / hub_dir).resolve() if not Path(hub_dir).is_absolute() else Path(hub_dir)))
    model = torch_hub_load("intel-isl/MiDaS", model_name).to(device).eval()
    midas_transforms = torch_hub_load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform if model_name == "MiDaS_small" else midas_transforms.dpt_transform
    return model, transform, device


@torch.no_grad()
def estimate_relative_depth(image, midas_model, midas_transform, device, output_size):
    image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    batch = midas_transform(image_np)
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    pred = midas_model(batch.to(device))
    pred = F.interpolate(
        pred.unsqueeze(1),
        size=output_size,
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth = pred.detach().cpu().numpy().astype(np.float32)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def fit_aligned_depth(relative_depth, lidar_depth, fit_mask):
    valid = fit_mask & np.isfinite(relative_depth) & np.isfinite(lidar_depth) & (lidar_depth > 0.0)
    if int(valid.sum()) < 16:
        valid = np.isfinite(relative_depth) & np.isfinite(lidar_depth) & (lidar_depth > 0.0)
    if int(valid.sum()) < 16:
        return None, "insufficient"

    candidates = {
        "affine_rel": relative_depth,
        "affine_inv_rel": 1.0 / (relative_depth - float(np.nanmin(relative_depth)) + 1e-3),
    }
    best_depth = None
    best_name = ""
    best_error = float("inf")
    for name, feature in candidates.items():
        x = feature[valid].astype(np.float64)
        y = lidar_depth[valid].astype(np.float64)
        design = np.stack([x, np.ones_like(x)], axis=1)
        scale, bias = np.linalg.lstsq(design, y, rcond=None)[0]
        aligned = (scale * feature + bias).astype(np.float32)
        error = float(np.mean(np.abs(aligned[valid] - lidar_depth[valid])))
        if error < best_error:
            best_error = error
            best_depth = aligned
            best_name = name
    return best_depth, best_name


def backproject_to_velo(uv, depth, calib, output_size):
    k = scaled_camera_k(calib, output_size)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    x = (uv[:, 0] - cx) * depth / fx
    y = (uv[:, 1] - cy) * depth / fy
    rect = np.stack([x, y, depth, np.ones_like(depth)], axis=1).astype(np.float32)
    rect_to_velo = np.linalg.inv(calib["R_rect_00_ext"] @ calib["Tr_velo_to_cam"])
    return (rect_to_velo @ rect.T).T[:, :3]


def summarize_depth_errors(prefix, pred_depth, lidar_depth, pred_velo, lidar_velo, mask):
    valid = mask & np.isfinite(pred_depth) & np.isfinite(lidar_depth) & (lidar_depth > 0.0)
    if int(valid.sum()) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_abs_mean": float("nan"),
            f"{prefix}_abs_median": float("nan"),
            f"{prefix}_rel_mean": float("nan"),
            f"{prefix}_rel_median": float("nan"),
            f"{prefix}_xyz_l2_mean": float("nan"),
        }
    abs_error = np.abs(pred_depth[valid] - lidar_depth[valid])
    rel_error = abs_error / np.maximum(lidar_depth[valid], 1e-3)
    pred_valid = pred_velo[valid]
    lidar_valid = lidar_velo[valid]
    l2_error = np.linalg.norm(pred_valid - lidar_valid, axis=1)
    nn_l2_error, nn_index = cKDTree(lidar_valid).query(pred_valid, k=1)
    nn_depth_error = np.abs(pred_depth[valid] - lidar_depth[valid][nn_index])
    return {
        f"{prefix}_count": int(valid.sum()),
        f"{prefix}_abs_mean": float(np.mean(abs_error)),
        f"{prefix}_abs_median": float(np.median(abs_error)),
        f"{prefix}_rel_mean": float(np.mean(rel_error)),
        f"{prefix}_rel_median": float(np.median(rel_error)),
        f"{prefix}_xyz_l2_mean": float(np.mean(l2_error)),
        f"{prefix}_nn_abs_mean": float(np.mean(nn_depth_error)),
        f"{prefix}_nn_xyz_l2_mean": float(np.mean(nn_l2_error)),
    }


def depth_consistency_for_image(image, record, midas_model, midas_transform, device, max_depth):
    output_size = (image.height, image.width)
    calib = load_raw_calibration(record["calib_dir"])
    points = load_velodyne_points(record["velodyne_path"])
    points_xyz = points[:, :3] if points.size else np.zeros((0, 3), dtype=np.float32)
    boxes = parse_tracklet_xml(record.get("tracklet_xml_path", "")).get(int(record["frame_index"]), [])
    uv, lidar_depth, valid = project_velo_to_image(points_xyz, calib, output_size)
    valid &= lidar_depth > 0.0
    valid &= lidar_depth <= max_depth
    inside_dynamic, _ = points_in_boxes(points_xyz, boxes)
    if int(valid.sum()) == 0:
        return {"depth_valid_points": 0}

    xy = np.rint(uv[valid]).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, output_size[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, output_size[0] - 1)
    rel_map = estimate_relative_depth(image, midas_model, midas_transform, device, output_size)
    rel_samples = rel_map[xy[:, 1], xy[:, 0]]
    lidar_samples = lidar_depth[valid]
    dynamic_samples = inside_dynamic[valid]
    aligned_depth, fit_kind = fit_aligned_depth(rel_samples, lidar_samples, ~dynamic_samples)
    if aligned_depth is None:
        return {"depth_valid_points": int(valid.sum()), "depth_fit_kind": fit_kind}

    pred_velo = backproject_to_velo(uv[valid], aligned_depth, calib, output_size)
    lidar_velo = points_xyz[valid]
    metrics = {
        "depth_valid_points": int(valid.sum()),
        "depth_fit_kind": fit_kind,
        "depth_dynamic_points": int(dynamic_samples.sum()),
        "depth_static_points": int((~dynamic_samples).sum()),
    }
    metrics.update(summarize_depth_errors("depth_all", aligned_depth, lidar_samples, pred_velo, lidar_velo, np.ones_like(dynamic_samples, dtype=bool)))
    metrics.update(summarize_depth_errors("depth_dynamic", aligned_depth, lidar_samples, pred_velo, lidar_velo, dynamic_samples))
    metrics.update(summarize_depth_errors("depth_static", aligned_depth, lidar_samples, pred_velo, lidar_velo, ~dynamic_samples))
    return metrics


def compare_depth_modes(depth_records):
    by_mode = {}
    for record in depth_records:
        by_mode.setdefault(record["mode"], {})[record["sample_id"]] = record
    if "none" not in by_mode or "dynamic_full" not in by_mode:
        return {}

    comparison = {}
    common = sorted(set(by_mode["none"]) & set(by_mode["dynamic_full"]))
    for metric in ("depth_all_abs_mean", "depth_static_abs_mean", "depth_dynamic_abs_mean", "depth_all_rel_mean", "depth_dynamic_rel_mean"):
        deltas = []
        for sample_id in common:
            a = by_mode["none"][sample_id].get(metric, float("nan"))
            b = by_mode["dynamic_full"][sample_id].get(metric, float("nan"))
            if math.isfinite(a) and math.isfinite(b):
                deltas.append(b - a)
        if not deltas:
            continue
        wins = sum(delta < 0.0 for delta in deltas)
        n = len(deltas)
        tail = sum(math.comb(n, k) for k in range(0, min(wins, n - wins) + 1)) / float(2**n)
        comparison[metric] = {
            "mean_delta_dynamic_full_minus_none": float(sum(deltas) / n),
            "dynamic_full_better_count": int(wins),
            "paired_count": int(n),
            "sign_test_two_sided_p": float(min(1.0, 2.0 * tail)),
        }
    return comparison


def run_depth_consistency_from_images(args, out_dir):
    records = []
    manifest_records = []
    with Path(args.manifest).open("r") as handle:
        for line in handle:
            if line.strip():
                manifest_records.append(json.loads(line))
            if len(manifest_records) >= args.num_samples:
                break

    midas_model, midas_transform, device = load_midas_depth_model(args.midas_model, args.depth_device, args.midas_hub_dir)
    modes = parse_mode_list(args.depth_modes)
    image_root = Path(args.depth_images_dir) if args.depth_images_dir else out_dir / "images"
    for mode in modes:
        for record in manifest_records:
            rel = Path(record["sample_id"]).with_suffix(".png")
            image_path = image_root / mode / "pred" / rel
            if not image_path.exists():
                continue
            with Image.open(image_path) as image:
                metrics = depth_consistency_for_image(image.convert("RGB"), record, midas_model, midas_transform, device, args.depth_max_depth)
            row = {"mode": mode, "sample_id": record["sample_id"], "image_path": str(image_path)}
            row.update(metrics)
            records.append(row)
            print(json.dumps(row, sort_keys=True))

    summary = {mode: aggregate([record for record in records if record["mode"] == mode]) for mode in modes}
    comparison = compare_depth_modes(records)
    metrics_dir = out_dir / "metrics"
    write_json(metrics_dir / "depth_consistency_per_sample.json", records)
    write_csv(metrics_dir / "depth_consistency_per_sample.csv", records)
    write_json(metrics_dir / "depth_consistency_summary.json", {"summary": summary, "comparison": comparison})
    write_csv(metrics_dir / "depth_consistency_summary.csv", [dict({"mode": mode}, **values) for mode, values in summary.items()])
    print(json.dumps({"depth_consistency_summary": summary, "comparison": comparison}, indent=2, sort_keys=True))


def main():
    args = parse_args()
    ckpts = {mode: path for mode, path in mode_ckpts(args).items() if path}
    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "metrics"
    image_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "eval_config.json", vars(args))
    if not ckpts:
        if args.depth_consistency:
            run_depth_consistency_from_images(args, out_dir)
            return
        raise SystemExit("At least one checkpoint path is required.")

    evaluator = None if args.skip_metrics else Evaluate_indic().cuda().eval()
    midas_model = midas_transform = midas_device = None
    depth_records = []
    if args.depth_consistency:
        midas_model, midas_transform, midas_device = load_midas_depth_model(args.midas_model, args.depth_device, args.midas_hub_dir)
    sample_ids = []
    all_records = {}
    gt_paths = {}
    overlay_paths = {}
    pred_paths_by_sample = {}
    object_boxes_by_sample = {}

    for mode, ckpt_path in ckpts.items():
        dataset = SatLidarRawDataset(args.manifest, condition_mode=mode)
        model = load_model(args.config, mode, ckpt_path, args.control_scale_override)
        mode_records = []
        limit = min(args.num_samples, len(dataset))
        for sample_idx in range(limit):
            sample = dataset[sample_idx]
            batch = sample_to_batch(sample)
            sample_id = sample["sample_id"]
            if sample_id not in sample_ids:
                sample_ids.append(sample_id)
            if sample_id not in object_boxes_by_sample:
                object_boxes_by_sample[sample_id] = {
                    "boxes": sample["dynamic_boxes"].detach().cpu().numpy(),
                    "valid": sample["dynamic_box_valid"].detach().cpu().numpy(),
                }

            pred, target = generate_prediction(
                model,
                batch,
                mode=mode,
                ddim_steps=args.ddim_steps,
                seed=args.seed + sample_idx,
                guidance_scale=args.guidance_scale,
                eta=args.eta,
                temperature=args.temperature,
            )
            record = {
                "mode": mode,
                "sample_id": sample_id,
                "checkpoint": ckpt_path,
                "ddim_steps": args.ddim_steps,
                "num_dynamic_boxes": int(batch["num_dynamic_boxes"].detach().cpu().item()),
                "num_projected_dynamic_points": int(batch["num_projected_dynamic_points"].detach().cpu().item()),
                "num_projected_lidar_points": int(batch["num_projected_lidar_points"].detach().cpu().item()),
            }
            if not args.skip_metrics:
                record.update(full_metrics(evaluator, pred, target))
                record.update(dynamic_metrics(evaluator, pred, target, batch["dynamic_mask"]))
            if args.depth_consistency:
                pred_image = transforms.functional.to_pil_image(pred[0].detach().cpu().clamp(0, 1)).convert("RGB")
                depth_record = {
                    "mode": mode,
                    "sample_id": sample_id,
                    "checkpoint": ckpt_path,
                }
                depth_record.update(
                    depth_consistency_for_image(
                        pred_image,
                        dataset.records[sample_idx],
                        midas_model,
                        midas_transform,
                        midas_device,
                        args.depth_max_depth,
                    )
                )
                record.update(depth_record)
                depth_records.append(depth_record)
            mode_records.append(record)
            print(json.dumps(record, sort_keys=True))

            rel = Path(sample_id)
            pred_path = image_dir / mode / "pred" / rel.with_suffix(".png")
            save_tensor_image(pred[0], pred_path)
            pred_paths_by_sample.setdefault(sample_id, {})[mode] = pred_path

            if sample_id not in gt_paths:
                gt_path = image_dir / "gt" / rel.with_suffix(".png")
                save_tensor_image(target[0], gt_path)
                gt_paths[sample_id] = gt_path
            if mode == "dynamic_full" or sample_id not in overlay_paths:
                overlay_path = image_dir / "lidar_overlay" / mode / rel.with_suffix(".png")
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                make_lidar_overlay(sample["grd_left_imgs"], sample["lidar_cond"], sample["dynamic_mask"]).save(overlay_path)
                cond_path = image_dir / "lidar_cond_rgb" / mode / rel.with_suffix(".png")
                save_tensor_image(make_cond_rgb(sample["lidar_cond"]), cond_path)
                overlay_paths[sample_id] = overlay_path

        all_records[mode] = mode_records
        del model
        torch.cuda.empty_cache()

    rows = [record for records in all_records.values() for record in records]
    summary = {mode: aggregate(records) for mode, records in all_records.items()}
    checks = build_success_checks(summary)

    write_json(metrics_dir / "per_sample_metrics.json", rows)
    write_json(metrics_dir / "summary.json", {"summary": summary, "success_checks": checks})
    write_csv(metrics_dir / "per_sample_metrics.csv", rows)
    summary_rows = [dict({"mode": mode}, **metrics) for mode, metrics in summary.items()]
    write_csv(metrics_dir / "summary.csv", summary_rows)
    if args.depth_consistency:
        depth_modes = sorted({record["mode"] for record in depth_records})
        depth_summary = {
            mode: aggregate([record for record in depth_records if record["mode"] == mode])
            for mode in depth_modes
        }
        depth_comparison = compare_depth_modes(depth_records)
        write_json(metrics_dir / "depth_consistency_per_sample.json", depth_records)
        write_csv(metrics_dir / "depth_consistency_per_sample.csv", depth_records)
        write_json(metrics_dir / "depth_consistency_summary.json", {"summary": depth_summary, "comparison": depth_comparison})
        write_csv(metrics_dir / "depth_consistency_summary.csv", [dict({"mode": mode}, **values) for mode, values in depth_summary.items()])

    panel_paths = []
    object_panel_records = []
    for sample_id in sample_ids:
        pred_paths = {mode: pred_paths_by_sample[sample_id][mode] for mode in MODES if mode in pred_paths_by_sample[sample_id]}
        panel_paths.append(make_panel(out_dir, sample_id, gt_paths[sample_id], overlay_paths[sample_id], pred_paths))
        if args.object_crop_panels:
            object_panel_records.extend(
                make_object_crop_panels(
                    out_dir,
                    sample_id,
                    gt_paths[sample_id],
                    overlay_paths[sample_id],
                    pred_paths,
                    object_boxes_by_sample[sample_id]["boxes"],
                    object_boxes_by_sample[sample_id]["valid"],
                    padding=args.object_crop_padding,
                    size=args.object_crop_size,
                    max_boxes=args.object_crop_max_boxes,
                )
            )
    write_json(out_dir / "panels.json", {"panels": panel_paths})
    if args.object_crop_panels:
        write_json(out_dir / "object_panels.json", {"object_panels": object_panel_records})
    print(json.dumps({"summary": summary, "success_checks": checks, "out_dir": str(out_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
