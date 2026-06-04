import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from dataloader.kitti_raw_lidar_utils import lidar_condition_channels, lidar_condition_gate_channel  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Gradient and influence sanity check for KITTI LiDAR control checkpoints.")
    parser.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dynamic.yaml")
    parser.add_argument("--manifest", default="dataset/kitti_raw_sat_lidar/val_manifest.jsonl")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--compare-ckpt", default="")
    parser.add_argument(
        "--mode",
        default="raw_lidar",
        choices=["bbox_dynamic", "dynamic_points", "raw_lidar", "dynamic_full"],
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def configure_for_mode(cfg, mode):
    cfg.data.params.batch_size = 1
    cfg.data.params.num_workers = 0
    cfg.data.params.train.params.condition_mode = mode
    cfg.data.params.test.params.condition_mode = mode
    cfg.model.params.use_lidar_cond = True
    cfg.model.params.freeze_for_lidar_control = True
    raw_geometry_modes = set()
    semantic_free_modes = {"dynamic_points"}
    geometry_gate_channel = lidar_condition_gate_channel(mode)
    cfg.model.params.dynamic_class_token_weight = 0.0 if mode in semantic_free_modes else cfg.model.params.get("dynamic_class_token_weight", 0.0)
    cfg.model.params.static_teacher_consistency_weight = 0.25 if mode in raw_geometry_modes else 0.0
    cfg.model.params.static_teacher_gate_channel = geometry_gate_channel
    if mode in raw_geometry_modes:
        cfg.model.params.dynamic_loss_weight = 0.0
        cfg.model.params.dynamic_x0_loss_weight = 0.0
        cfg.model.params.dynamic_point_loss_weight = 0.0
        cfg.model.params.dynamic_point_x0_loss_weight = 0.0
    control = cfg.model.params.DDPM_config.params.control_grd
    unet = cfg.model.params.DDPM_config.params.unet_config.params
    control.target = "models.KITTI_geo_ldm.lidar_condition_model.LidarMultiScaleControl"
    control.params.in_channels = lidar_condition_channels(mode)
    control.params.model_channels = unet.model_channels
    control.params.channel_mult = list(unet.channel_mult)
    control.params.num_res_blocks = unet.num_res_blocks
    control.params.middle_channels = unet.model_channels * list(unet.channel_mult)[-1]
    control.params.semantic_class_count = 0 if mode in semantic_free_modes else int(control.params.get("semantic_class_count", 0))
    control.params.gate_channel = geometry_gate_channel
    control.params.gate_residuals = geometry_gate_channel >= 0
    return cfg


def load_full_model_checkpoint(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    if "model" in payload:
        model.load_state_dict(payload["model"], strict=False)
    elif "state_dict" in payload:
        model.load_state_dict(payload["state_dict"], strict=False)
    else:
        raise ValueError(f"Unsupported base checkpoint format: {ckpt_path}")


def load_control_checkpoint(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    base_model_ckpt = payload.get("base_model_ckpt", "")
    if base_model_ckpt:
        load_full_model_checkpoint(model, base_model_ckpt)
    if "control_grd" not in payload:
        raise ValueError(f"Checkpoint has no control_grd state: {ckpt_path}")
    model.DDPM.control_grd.load_state_dict(payload["control_grd"], strict=True)
    if "dynamic_class_tokens" in payload and getattr(model, "dynamic_class_tokens", None) is not None:
        model.dynamic_class_tokens.data.copy_(payload["dynamic_class_tokens"].to(model.dynamic_class_tokens.device))
    if "denoise_model_trainable" in payload:
        model.DDPM.denoise_model.load_state_dict(payload["denoise_model_trainable"], strict=False)
    return payload


def move_batch_to_cuda(sample):
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).cuda(non_blocking=True)
        else:
            batch[key] = [value]
    return batch


def module_grad_stats(module):
    param_count = 0
    trainable_count = 0
    grad_param_count = 0
    param_sq = 0.0
    grad_sq = 0.0
    max_abs_grad = 0.0
    for param in module.parameters():
        param_count += param.numel()
        param_sq += float(param.detach().float().pow(2).sum().cpu())
        if param.requires_grad:
            trainable_count += param.numel()
        if param.grad is not None:
            grad = param.grad.detach().float()
            grad_param_count += param.numel()
            grad_sq += float(grad.pow(2).sum().cpu())
            max_abs_grad = max(max_abs_grad, float(grad.abs().max().cpu()))
    return {
        "param_count": param_count,
        "trainable_count": trainable_count,
        "grad_param_count": grad_param_count,
        "param_l2": math.sqrt(param_sq),
        "grad_l2": math.sqrt(grad_sq),
        "max_abs_grad": max_abs_grad,
    }


def parameter_grad_stats(param):
    if param is None:
        return {}
    result = {
        "param_count": param.numel(),
        "param_l2": float(param.detach().float().norm().cpu()),
        "max_abs": float(param.detach().float().abs().max().cpu()),
    }
    if param.grad is not None:
        grad = param.grad.detach().float()
        result.update(
            {
                "grad_param_count": param.numel(),
                "grad_l2": float(grad.norm().cpu()),
                "max_abs_grad": float(grad.abs().max().cpu()),
            }
        )
    else:
        result.update({"grad_param_count": 0, "grad_l2": 0.0, "max_abs_grad": 0.0})
    return result


def optimizer_state_stats(payload):
    state = payload.get("optimizer", {}).get("state", {})
    exp_avg_sq = 0.0
    exp_avg_l2_sq = 0.0
    tensor_count = 0
    for value in state.values():
        if not isinstance(value, dict):
            continue
        exp_avg = value.get("exp_avg")
        exp_avg_sq_value = value.get("exp_avg_sq")
        if torch.is_tensor(exp_avg):
            tensor_count += 1
            exp_avg_l2_sq += float(exp_avg.float().pow(2).sum().cpu())
        if torch.is_tensor(exp_avg_sq_value):
            exp_avg_sq += float(exp_avg_sq_value.float().sum().cpu())
    return {
        "optimizer_state_entries": len(state),
        "optimizer_exp_avg_tensor_count": tensor_count,
        "optimizer_exp_avg_l2": math.sqrt(exp_avg_l2_sq),
        "optimizer_exp_avg_sq_sum": exp_avg_sq,
    }


def control_weight_delta(model, compare_ckpt):
    if not compare_ckpt:
        return {}
    payload = torch.load(compare_ckpt, map_location="cpu")
    other = payload["control_grd"]
    delta_sq = 0.0
    base_sq = 0.0
    max_abs = 0.0
    for name, param in model.DDPM.control_grd.state_dict().items():
        current = param.detach().cpu().float()
        previous = other[name].detach().cpu().float()
        delta = current - previous
        delta_sq += float(delta.pow(2).sum())
        base_sq += float(previous.pow(2).sum())
        max_abs = max(max_abs, float(delta.abs().max()))
    return {
        "compare_ckpt": compare_ckpt,
        "control_delta_l2": math.sqrt(delta_sq),
        "control_compare_l2": math.sqrt(base_sq),
        "control_delta_relative_l2": math.sqrt(delta_sq) / max(math.sqrt(base_sq), 1e-12),
        "control_delta_max_abs": max_abs,
    }


def get_training_tensors(model, batch):
    inputs = model.get_input(batch, "sat_map").cuda() * 2 - 1
    outputs = model.get_input(batch, "grd_left_imgs").cuda() * 2 - 1
    lidar_cond = model.get_input(batch, model.lidar_condition_key).cuda()
    left_camera_k = model.get_input(batch, "left_camera_k").squeeze(-1).cuda()
    if hasattr(model, "make_condition"):
        cond_label = model.make_condition(inputs, batch).detach()
    else:
        cond_label = model.condition_model_sat(inputs)[:, 1:, :].detach()
    latent = model.pre_AE_model.encode(outputs).sample().detach() * model.scale_factor
    return {
        "latent": latent,
        "lidar_cond": lidar_cond,
        "cond_label": cond_label,
        "left_camera_k": left_camera_k,
        "gt_shift_x": batch["gt_shift_x"].cuda(),
        "gt_shift_y": batch["gt_shift_y"].cuda(),
        "theta": batch["theta"].cuda(),
    }


def denoise_output(model, tensors, t, noise, lidar_cond):
    x_noisy = model.DDPM.q_sample(x_start=tensors["latent"], t=t, noise=noise)
    control = None
    if lidar_cond is not None:
        control = model.DDPM.control_grd(
            x_noisy,
            t,
            cond_init_grd=lidar_cond,
            cond_sat=None,
            cond_txt=tensors["cond_label"],
        )
    return model.DDPM.denoise_model(
        x_noisy,
        t,
        context=tensors["cond_label"],
        control_grd=control,
        left_camera_k=tensors["left_camera_k"],
        gt_shift_x=tensors["gt_shift_x"],
        gt_shift_y=tensors["gt_shift_y"],
        theta=tensors["theta"],
    )


def tensor_diff_stats(prefix, a, b):
    diff = (a - b).detach().float()
    return {
        f"{prefix}_mean_abs": float(diff.abs().mean().cpu()),
        f"{prefix}_max_abs": float(diff.abs().max().cpu()),
        f"{prefix}_rmse": float(diff.pow(2).mean().sqrt().cpu()),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    cfg = configure_for_mode(OmegaConf.load(args.config), args.mode)
    model = instantiate_from_config(cfg.model).cuda()
    payload = load_control_checkpoint(model, args.ckpt)
    model.learning_rate = 7e-5
    optimizer = model.configure_optimizers()[0]
    model.train()

    dataset = SatLidarRawDataset(args.manifest, condition_mode=args.mode)
    sample = dataset[args.sample_index]
    batch = move_batch_to_cuda(sample)

    optimizer.zero_grad(set_to_none=True)
    loss = model.training_step(batch, args.sample_index)
    loss.backward()

    tensors = get_training_tensors(model, batch)
    torch.manual_seed(args.seed)
    t = torch.randint(0, model.DDPM.num_timesteps, (tensors["latent"].shape[0],), device=tensors["latent"].device).long()
    noise = torch.randn_like(tensors["latent"])
    with torch.no_grad():
        out_with = denoise_output(model, tensors, t, noise, tensors["lidar_cond"])
        out_without = denoise_output(model, tensors, t, noise, None)
        out_zero = denoise_output(model, tensors, t, noise, torch.zeros_like(tensors["lidar_cond"]))
        target = noise

    result = {
        "checkpoint": args.ckpt,
        "checkpoint_step": int(payload.get("step", -1)),
        "condition_mode": payload.get("condition_mode", args.mode),
        "base_model_ckpt": payload.get("base_model_ckpt", ""),
        "sample_id": sample["sample_id"],
        "loss": float(loss.detach().cpu()),
        "loss_with_control_fixed_noise": float(F.mse_loss(out_with, target).detach().cpu()),
        "loss_without_control_fixed_noise": float(F.mse_loss(out_without, target).detach().cpu()),
        "loss_zero_control_fixed_noise": float(F.mse_loss(out_zero, target).detach().cpu()),
        "num_dynamic_boxes": int(batch["num_dynamic_boxes"].detach().cpu().item()),
        "num_projected_dynamic_points": int(batch["num_projected_dynamic_points"].detach().cpu().item()),
        "num_projected_lidar_points": int(batch["num_projected_lidar_points"].detach().cpu().item()),
        "modules": {
            "control_grd": module_grad_stats(model.DDPM.control_grd),
            "denoise_model": module_grad_stats(model.DDPM.denoise_model),
            "condition_model_sat": module_grad_stats(model.condition_model_sat),
            "pre_AE_model": module_grad_stats(model.pre_AE_model),
        },
    }
    if getattr(model, "dynamic_class_tokens", None) is not None:
        result["modules"]["dynamic_class_tokens"] = parameter_grad_stats(model.dynamic_class_tokens)
    result.update(tensor_diff_stats("denoise_with_vs_without_control", out_with, out_without))
    result.update(tensor_diff_stats("denoise_with_vs_zero_control", out_with, out_zero))
    result.update(optimizer_state_stats(payload))
    result.update(control_weight_delta(model, args.compare_ckpt))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
