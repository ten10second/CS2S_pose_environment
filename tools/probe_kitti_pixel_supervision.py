import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import autocast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.KITTI_geo_ldm_diffusion.latent_diffusion import resize_masked_lidar_depth  # noqa: E402
from tools.generate_kitti_raea_samples import load_checkpoint_into_model  # noqa: E402
from tools.train_kitti_raea import TrainingStepModule  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402


TIMESTEP_CYCLE = (50, 250, 500, 750)


def parse_args():
    parser = argparse.ArgumentParser(description="Read-only V2.1 pixel LiDAR depth supervision diagnostic.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default="", help="Optional current checkpoint. If absent, probes fresh config baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pixel-cache-root", required=True)
    parser.add_argument("--image-semantic-cache-root", required=True)
    parser.add_argument("--kitti-root", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--gradient-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--depth-loss-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def scalar(value):
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() != 1:
            return float(value.float().mean().cpu())
        return float(value.float().cpu())
    return float(value)


def tensor_finite(value):
    return bool(torch.is_tensor(value) and torch.isfinite(value.detach()).all().item())


def gradient_summary(g_depth, g_total):
    if g_depth is None or g_total is None:
        return {
            "bottleneck_h_grad_connected": False,
            "bottleneck_h_grad_depth_norm": 0.0,
            "bottleneck_h_grad_total_norm": 0.0,
            "bottleneck_h_grad_other_norm": 0.0,
            "bottleneck_h_grad_depth_to_other_ratio": 0.0,
            "bottleneck_h_grad_depth_total_cosine": 0.0,
            "bottleneck_h_grad_depth_other_cosine": 0.0,
            "bottleneck_h_grad_finite": False,
        }
    g_depth = g_depth.detach().float()
    g_total = g_total.detach().float()
    g_other = g_total - g_depth
    depth_norm = g_depth.norm()
    total_norm = g_total.norm()
    other_norm = g_other.norm()
    denom = depth_norm * total_norm
    cosine = torch.sum(g_depth * g_total) / denom.clamp_min(1e-12)
    other_denom = depth_norm * other_norm
    other_cosine = torch.sum(g_depth * g_other) / other_denom.clamp_min(1e-12)
    ratio = depth_norm / other_norm.clamp_min(1e-12)
    finite = torch.isfinite(g_depth).all() and torch.isfinite(g_total).all() and torch.isfinite(g_other).all()
    return {
        "bottleneck_h_grad_connected": True,
        "bottleneck_h_grad_depth_norm": scalar(depth_norm),
        "bottleneck_h_grad_total_norm": scalar(total_norm),
        "bottleneck_h_grad_other_norm": scalar(other_norm),
        "bottleneck_h_grad_depth_to_other_ratio": scalar(ratio),
        "bottleneck_h_grad_depth_total_cosine": scalar(cosine),
        "bottleneck_h_grad_depth_other_cosine": scalar(other_cosine),
        "bottleneck_h_grad_finite": bool(finite.item()),
    }


def masked_mean(raw, mask):
    weighted = raw * mask
    denom = mask.sum().clamp_min(1e-6)
    return weighted.sum() / denom


def depth_loss_stats(
    depth_pred,
    lidar_depth_target,
    lidar_depth_mask,
    mode="masked_area",
    eps=1e-3,
    weight=0.1,
):
    depth_pred = depth_pred.float().clamp(1e-6, 1.0)
    target, target_support = resize_masked_lidar_depth(
        lidar_depth_target,
        lidar_depth_mask,
        depth_pred.shape[-2:],
        mode=mode,
    )
    target = target.to(device=depth_pred.device, dtype=depth_pred.dtype)
    resized_mask = F.interpolate(lidar_depth_mask.float(), size=depth_pred.shape[-2:], mode="area")
    depth_mask = resized_mask.to(device=depth_pred.device, dtype=depth_pred.dtype)
    if target_support is not None:
        target_support = target_support.to(device=depth_pred.device, dtype=depth_pred.dtype)
        depth_mask = depth_mask * (target_support > 0.0).to(depth_mask.dtype)
    invalid_zero_targets = ((depth_mask > 0.0) & (target <= 0.0)).sum()
    target_clamped = target.clamp(1e-6, 1.0)
    raw = (torch.log(depth_pred.clamp_min(float(eps))) - torch.log(target_clamped.clamp_min(float(eps)))).abs()
    log_l1 = masked_mean(raw, depth_mask)
    weighted = float(weight) * log_l1
    return {
        "loss": weighted,
        "log_l1": log_l1,
        "target": target,
        "mask": depth_mask,
        "invalid_zero_target_count_before_clamp": int(invalid_zero_targets.detach().cpu()),
        "support_count": int((depth_mask > 0.0).sum().detach().cpu()),
        "target_mean": scalar(masked_mean(target_clamped, depth_mask)),
        "pred_mean": scalar(masked_mean(depth_pred, depth_mask)),
        "weighted_depth_contribution": scalar(weighted),
    }


def patch_cfg(cfg, args):
    for split in ("train", "test"):
        if not hasattr(cfg.data.params, split):
            continue
        params = getattr(cfg.data.params, split).params
        params.manifest = args.manifest
        params.kitti_root = args.kitti_root
        params.lidar_pixel_feature_cache_root = args.pixel_cache_root
        params.lidar_pixel_feature_cache_suffix = ".npz"
        params.lidar_pixel_feature_dim = 576
        params.image_semantic_cache_root = args.image_semantic_cache_root
        params.image_semantic_cache_suffix = ".npz"
        params.image_semantic_feature_key = "dino_feat"
        params.image_semantic_feature_dim = 384
        params.image_semantic_height = 8
        params.image_semantic_width = 32
        params.include_range_image = False
        params.include_raw_lidar_points = False
        params.include_tracklets = False
    cfg.model.params.lidar_depth_loss_weight = float(args.depth_loss_weight)
    cfg.model.params.lidar_depth_output_scale = 0.0
    cfg.model.params.lidar_depth_bottleneck_scale = 1.0
    cfg.model.params.lidar_depth_resample_mode = "masked_area"
    return cfg


def batch_to_device(sample, device):
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device=device, non_blocking=True)
        else:
            batch[key] = [value]
    return batch


@contextmanager
def fixed_timestep(model, t_value):
    original = model.DDPM.p_losses
    captured = {"kwargs": None}

    def fixed_p_losses(x_start, t, *args, **kwargs):
        captured["kwargs"] = {
            "lidar_depth_target": kwargs.get("lidar_depth_target"),
            "lidar_depth_mask": kwargs.get("lidar_depth_mask"),
            "lidar_depth_loss_weight": kwargs.get("lidar_depth_loss_weight"),
            "lidar_depth_bottleneck_scale": kwargs.get("lidar_depth_bottleneck_scale"),
            "lidar_depth_resample_mode": kwargs.get("lidar_depth_resample_mode"),
            "lidar_depth_log_eps": kwargs.get("lidar_depth_log_eps"),
        }
        return original(x_start, torch.full_like(t, int(t_value)), *args, **kwargs)

    model.DDPM.p_losses = fixed_p_losses
    try:
        yield captured
    finally:
        model.DDPM.p_losses = original


@contextmanager
def middle_block_capture(model):
    state = {"h": None}

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        state["h"] = h

    handle = model.DDPM.denoise_model.middle_block.register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def run_with_rng(device, seed):
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    else:
        devices = []
    return torch.random.fork_rng(devices=devices, enabled=True)


def metric_agreement(computed, reported, atol=2e-4):
    if reported is None:
        return {"reported": None, "abs_error": None, "matches": False}
    reported_value = float(reported)
    computed_value = float(computed)
    error = abs(computed_value - reported_value)
    return {"reported": reported_value, "abs_error": error, "matches": bool(error <= float(atol))}


def evaluate_sample(model, trainer, sample, sample_index, device, want_grad, args):
    t_value = TIMESTEP_CYCLE[sample_index % len(TIMESTEP_CYCLE)]
    batch = batch_to_device(sample, device)
    seed = int(args.seed) + int(sample_index)
    autocast_ctx = autocast(enabled=device.type == "cuda")
    grad_ctx = torch.enable_grad() if want_grad else torch.no_grad()
    model.zero_grad(set_to_none=True)
    with run_with_rng(device, seed):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with grad_ctx, fixed_timestep(model, t_value) as captured_p_losses, middle_block_capture(model) as capture:
            with autocast_ctx:
                total_loss = trainer(batch, 0)
            head = model.DDPM.denoise_model.last_lidar_bottleneck_depth_pred
            depth_kwargs = captured_p_losses["kwargs"] or {}
            lidar_depth_target = depth_kwargs.get("lidar_depth_target")
            lidar_depth_mask = depth_kwargs.get("lidar_depth_mask")
            if lidar_depth_target is None or lidar_depth_mask is None:
                raise RuntimeError("p_losses did not receive lidar_depth_target/lidar_depth_mask")
            stats = depth_loss_stats(
                head,
                lidar_depth_target,
                lidar_depth_mask,
                mode=str(depth_kwargs.get("lidar_depth_resample_mode") or "masked_area"),
                eps=float(depth_kwargs.get("lidar_depth_log_eps") or getattr(model, "lidar_depth_log_eps", 1e-3)),
                weight=float(depth_kwargs.get("lidar_depth_loss_weight") or args.depth_loss_weight)
                * float(depth_kwargs.get("lidar_depth_bottleneck_scale") or 1.0),
            )
            grad_stats = {}
            if want_grad:
                h = capture["h"]
                if h is None or not torch.is_tensor(h):
                    raise RuntimeError("middle_block hook did not capture a tensor output")
                g_depth = torch.autograd.grad(stats["loss"], h, retain_graph=True, allow_unused=True)[0]
                g_total = torch.autograd.grad(total_loss, h, retain_graph=False, allow_unused=True)[0]
                grad_stats = gradient_summary(g_depth, g_total)
            finite = {
                "total_loss_finite": torch.isfinite(total_loss.detach()).item(),
                "head_depth_finite": tensor_finite(head),
                "weighted_depth_loss_finite": torch.isfinite(stats["loss"].detach()).item(),
            }
    metrics = {
        key: scalar(value) if torch.is_tensor(value) else value
        for key, value in getattr(model.DDPM, "last_loss_metrics", {}).items()
    }
    log_l1_agreement = metric_agreement(
        scalar(stats["log_l1"]),
        metrics.get("loss_lidar_bottleneck_depth_log_l1"),
    )
    contrib_agreement = metric_agreement(
        scalar(stats["weighted_depth_contribution"]),
        metrics.get("loss_lidar_bottleneck_depth_log_l1_contrib"),
    )
    record = {
        "sample_index": int(sample_index),
        "sample_id": str(sample.get("sample_id", "")),
        "seed": int(seed),
        "timestep": int(t_value),
        "reported_total_loss": scalar(total_loss),
        "reported_loss_metrics": metrics,
        "head_depth_shape": list(head.shape),
        "head_depth_log_l1": scalar(stats["log_l1"]),
        "head_depth_weighted_contribution_at_0p1": scalar(stats["weighted_depth_contribution"]),
        "head_depth_pred_mean": stats["pred_mean"],
        "head_depth_target_mean": stats["target_mean"],
        "head_depth_support_count": int(stats["support_count"]),
        "invalid_zero_target_count_before_clamp": int(stats["invalid_zero_target_count_before_clamp"]),
        "reported_bottleneck_depth_log_l1_agreement": log_l1_agreement,
        "reported_bottleneck_depth_contrib_agreement": contrib_agreement,
        "finite_checks": finite,
        "gradient_scope": "bottleneck_feature_h" if want_grad else "not_computed",
    }
    record.update(grad_stats)
    return record


def aggregate(records):
    numeric_keys = [
        "reported_total_loss",
        "head_depth_log_l1",
        "head_depth_weighted_contribution_at_0p1",
        "bottleneck_h_grad_depth_norm",
        "bottleneck_h_grad_total_norm",
        "bottleneck_h_grad_other_norm",
        "bottleneck_h_grad_depth_to_other_ratio",
        "bottleneck_h_grad_depth_total_cosine",
        "bottleneck_h_grad_depth_other_cosine",
    ]
    out = {"num_records": len(records)}
    for key in numeric_keys:
        values = [float(record[key]) for record in records if key in record and math.isfinite(float(record[key]))]
        if values:
            out[f"{key}_mean"] = sum(values) / len(values)
            out[f"{key}_min"] = min(values)
            out[f"{key}_max"] = max(values)
    for key in ("loss_eps_base", "loss_lidar_hit_eps", "loss_lidar_hit_eps_contrib"):
        values = [float(record["reported_loss_metrics"][key]) for record in records
                  if key in record.get("reported_loss_metrics", {})]
        if values:
            out[f"reported_{key}_mean"] = sum(values) / len(values)
    out["finite_all"] = all(all(record.get("finite_checks", {}).values()) for record in records)
    out["gradient_finite_all"] = all(
        record.get("gradient_scope") != "bottleneck_feature_h" or bool(record.get("bottleneck_h_grad_finite", False))
        for record in records
    )
    out["gradient_connected_all"] = all(
        record.get("gradient_scope") != "bottleneck_feature_h" or bool(record.get("bottleneck_h_grad_connected", False))
        for record in records
    )
    out["reported_depth_metrics_match_all"] = all(
        record.get("reported_bottleneck_depth_log_l1_agreement", {}).get("matches", False)
        and record.get("reported_bottleneck_depth_contrib_agreement", {}).get("matches", False)
        for record in records
    )
    out["invalid_zero_target_count_before_clamp_total"] = sum(
        int(record.get("invalid_zero_target_count_before_clamp", 0)) for record in records
    )
    out["probe_passed"] = (
        bool(out["finite_all"])
        and bool(out["gradient_finite_all"])
        and bool(out["gradient_connected_all"])
        and bool(out["reported_depth_metrics_match_all"])
        and int(out["invalid_zero_target_count_before_clamp_total"]) == 0
    )
    return out


def main():
    args = parse_args()
    if args.num_samples < 1 or args.gradient_samples < 0:
        raise ValueError("num-samples must be positive and gradient-samples nonnegative")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        if device.index is None:
            device = torch.device("cuda:0")
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(int(args.seed))

    cfg = patch_cfg(OmegaConf.load(args.config), args)
    (out_dir / "probe_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    dataset = instantiate_from_config(cfg.data.params.train)
    model = instantiate_from_config(cfg.model).to(device)
    if args.ckpt:
        load_checkpoint_into_model(model, args.ckpt)
    model.eval()
    trainer = TrainingStepModule(model, 0.0).to(device)

    records = []
    count = min(int(args.num_samples), len(dataset))
    try:
        with (out_dir / "samples.jsonl").open("w") as handle:
            for index in range(count):
                record = evaluate_sample(
                    model,
                    trainer,
                    dataset[index],
                    index,
                    device,
                    want_grad=index < int(args.gradient_samples),
                    args=args,
                )
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
    finally:
        model.zero_grad(set_to_none=True)

    aggregate_stats = aggregate(records)
    summary = {
        "diagnostic_scope": "fixed training-sample diagnostic subset",
        "not_a_validation_claim": True,
        "not_rgb_geometry_proof": True,
        "checkpoint": str(args.ckpt) if args.ckpt else "",
        "baseline": "checkpoint" if args.ckpt else "fresh_config_sd_baseline_seed_3407",
        "manifest": str(args.manifest),
        "pixel_cache_root": str(args.pixel_cache_root),
        "image_semantic_cache_root": str(args.image_semantic_cache_root),
        "kitti_root": str(args.kitti_root),
        "num_samples": int(count),
        "gradient_samples": min(int(args.gradient_samples), int(count)),
        "timestep_cycle": list(TIMESTEP_CYCLE),
        "depth_loss_weight_used_for_probe": float(args.depth_loss_weight),
        "gradient_label": "bottleneck-feature gradient wrt UNet middle_block output h",
        "aggregate": aggregate_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not aggregate_stats.get("probe_passed", False):
        raise SystemExit("Pixel supervision probe failed diagnostic checks; see summary.json")


if __name__ == "__main__":
    main()
