import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    lidar_condition_channels,
    lidar_condition_gate_channel,
    lidar_condition_uses_pointmap,
)
from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from tools.generate_kitti_lidar_dominant_samples import (  # noqa: E402
    generate_prediction,
    make_condition_rgb,
    make_lidar_overlay,
    make_panel,
    safe_sample_id,
    sample_to_batch,
    save_tensor_image,
)
from utils.util import instantiate_from_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-attention LiDAR KITTI sat-to-street training.")
    parser.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dual_attn.yaml")
    parser.add_argument("--train-manifest", default="dataset/kitti_raw_sat_lidar/train_manifest.jsonl")
    parser.add_argument("--val-manifest", default="dataset/kitti_raw_sat_lidar/test2_manifest.jsonl")
    parser.add_argument(
        "--condition-mode",
        default="raw_lidar_pointmap",
        choices=["none", "bbox_dynamic", "dynamic_points", "raw_lidar", "dynamic_full", "raw_lidar_pointmap"],
    )
    parser.add_argument("--sd-base-ckpt", default="/home/shizhm/Downloads/sd-v1-4.ckpt")
    parser.add_argument("--cs2s-init-ckpt", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--out-root", default="results/kitti_xlidar_overfit")
    parser.add_argument("--steps", type=int, default=36000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=3000)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=0,
        help="Run inline visual sampling every N steps. 0 disables sampling.",
    )
    parser.add_argument("--sample-manifest", default="", help="Fixed manifest used for inline sample panels.")
    parser.add_argument("--sample-num-samples", type=int, default=2)
    parser.add_argument("--sample-ddim-steps", type=int, default=2)
    parser.add_argument(
        "--sample-probes",
        default="normal,zero",
        help="Comma-separated inline sample probes. Supported probes: normal,zero.",
    )
    parser.add_argument(
        "--sample-shift-fraction",
        type=float,
        default=0.12,
        help="Deprecated no-op kept for old launch scripts; global LiDAR shift probes are disabled.",
    )
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--no-save-optimizer", dest="save_optimizer", action="store_false")
    parser.set_defaults(save_optimizer=True)
    parser.add_argument(
        "--keep-step-checkpoints",
        type=int,
        default=-1,
        help="Keep only the newest N step_*.pt checkpoints. Use -1 to keep all, 0 to keep none.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument(
        "--lidar-warmstart-ckpt",
        default="",
        help="Load LiDAR trainable weights from a previous run without loading optimizer state or step.",
    )

    parser.add_argument("--control-hidden-channels", type=int, default=192)
    parser.add_argument("--control-scale", type=float, default=1.5)
    parser.add_argument("--control-gate-channel", type=int, default=None)
    parser.add_argument("--control-gate-dilation", type=int, default=8)
    parser.add_argument("--range-feature-channels", type=int, default=192)
    parser.add_argument("--range-hidden-channels", type=int, default=192)
    parser.add_argument("--range-depth-samples", type=int, default=16)
    parser.add_argument("--range-residual-scale", type=float, default=1.5)
    parser.add_argument("--geo-xattn-scales", default="1,2,4,8")
    parser.add_argument("--lidar-injection", choices=["dual_attention"], default="dual_attention")
    parser.add_argument("--lidar-token-height", type=int, default=8)
    parser.add_argument("--lidar-token-width", type=int, default=32)
    parser.add_argument("--lidar-token-hidden-channels", type=int, default=128)
    parser.add_argument("--lidar-token-dim", type=int, default=768)
    parser.add_argument("--lidar-token-range-depth-samples", type=int, default=8)
    parser.add_argument("--lidar-attn-gate-init", type=float, default=1e-3)
    parser.add_argument(
        "--lidar-attention-mode",
        choices=["token", "reference"],
        default="token",
        help=(
            "token uses global pooled LiDAR token cross-attention; reference uses "
            "Pointmap-style local attention aligned to the camera-view token grid."
        ),
    )
    parser.add_argument(
        "--lidar-reference-window",
        type=int,
        default=3,
        help="Odd local window size for Pointmap-style LiDAR reference attention.",
    )
    parser.add_argument("--lidar-gate-cap-start", type=float, default=1.0)
    parser.add_argument("--lidar-gate-cap-end", type=float, default=1.0)
    parser.add_argument(
        "--lidar-gate-warmup-steps",
        type=int,
        default=0,
        help="Linearly ramp the effective LiDAR attention gate cap from start to end over this many steps.",
    )
    parser.add_argument(
        "--force-lidar-gate",
        type=float,
        default=-1.0,
        help="If in (0, 1), overwrite all loaded LiDAR gate logits after warmstart/resume.",
    )
    parser.add_argument("--lidar-context-lr-scale", type=float, default=1.0)
    parser.add_argument("--lidar-evidence-dilation", type=int, default=4)
    parser.add_argument("--lidar-evidence-free-space-dilation", type=int, default=14)
    parser.add_argument(
        "--lidar-token-output-norm",
        choices=["none", "layernorm", "center_layernorm"],
        default="none",
        help="Normalize LiDAR tokens at the source before they enter denoise attention.",
    )
    parser.add_argument(
        "--lidar-token-structure-loss-weight",
        type=float,
        default=0.0,
        help="Weight for penalizing collapse of raw LiDAR token centered/mean ratio.",
    )
    parser.add_argument(
        "--lidar-token-structure-target-ratio",
        type=float,
        default=0.08,
        help="Minimum raw token centered/mean ratio encouraged by the structure regularizer.",
    )
    parser.add_argument(
        "--no-lidar-pointmap-pe",
        dest="lidar_pointmap_pe",
        action="store_false",
        help="Disable PointmapDiff-style Fourier encoding of camera-frame XYZ LiDAR point maps.",
    )
    parser.set_defaults(lidar_pointmap_pe=True)

    parser.add_argument("--dynamic-loss-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-x0-loss-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--dynamic-crop-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--lidar-support-loss-weight", type=float, default=1.0)
    parser.add_argument("--lidar-support-x0-loss-weight", type=float, default=0.5)
    parser.add_argument("--lidar-support-image-loss-weight", type=float, default=2.0)
    parser.add_argument("--lidar-support-dilation", type=int, default=8)
    parser.add_argument(
        "--lidar-depth-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary log-depth L1 weight on latent-resolution LiDAR hit cells.",
    )
    parser.add_argument(
        "--lidar-depth-log-eps",
        type=float,
        default=1e-3,
        help="Epsilon used by normalized log-depth supervision.",
    )
    parser.add_argument("--static-teacher-consistency-weight", type=float, default=0.0)
    parser.add_argument("--foreground-mask-root", default="")
    parser.add_argument("--foreground-mask-suffix", default="_foreground.png")
    parser.add_argument("--foreground-loss-weight", type=float, default=0.0)
    parser.add_argument("--foreground-x0-loss-weight", type=float, default=0.0)
    parser.add_argument("--foreground-image-loss-weight", type=float, default=0.0)
    parser.add_argument("--foreground-lpips-loss-weight", type=float, default=0.0)
    parser.add_argument("--foreground-lpips-padding", type=int, default=8)
    parser.add_argument("--foreground-lpips-size", type=int, default=96)
    parser.add_argument("--foreground-lidar-intersection", action="store_true")
    parser.add_argument(
        "--lidar-zero-reconstruction-loss-weight",
        type=float,
        default=0.0,
        help="Extra base denoise reconstruction weight for the zero-LiDAR counterfactual branch.",
    )
    parser.add_argument(
        "--lidar-zero-reconstruction-mask-mode",
        choices=["all", "background"],
        default="all",
        help=(
            "Where zero-LiDAR reconstruction is applied. all keeps the legacy full-latent loss; "
            "background applies it only outside the foreground/LiDAR counterfactual mask."
        ),
    )
    parser.add_argument(
        "--lidar-counterfactual-weight",
        type=float,
        default=0.0,
        help="Weight for forcing correct LiDAR to beat zero LiDAR on foreground/LiDAR-hit image loss.",
    )
    parser.add_argument("--lidar-counterfactual-margin", type=float, default=0.02)
    parser.add_argument("--lidar-counterfactual-probes", default="zero")
    parser.add_argument(
        "--lidar-counterfactual-train-negative",
        dest="lidar_counterfactual_stop_negative",
        action="store_false",
        help="Allow gradients to push wrong-LiDAR branches away from GT. Default keeps negative branches stop-grad.",
    )
    parser.set_defaults(lidar_counterfactual_stop_negative=True)
    parser.add_argument("--lidar-counterfactual-separation-weight", type=float, default=0.0)
    parser.add_argument("--lidar-counterfactual-exist-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-lidar-counterfactual-point-fallback",
        dest="lidar_counterfactual_point_fallback",
        action="store_false",
        help="Use only SAM foreground ∩ LiDAR hit for counterfactual forcing; default falls back to LiDAR hit if foreground is empty.",
    )
    parser.set_defaults(lidar_counterfactual_point_fallback=True)
    parser.add_argument(
        "--no-tracklets",
        dest="include_tracklets",
        action="store_false",
        help="Skip KITTI raw tracklet XML loading; dynamic boxes become empty while raw LiDAR/SAM foreground supervision remains active.",
    )
    parser.set_defaults(include_tracklets=True)

    parser.add_argument("--train-sat-condition", action="store_true", default=False)
    parser.add_argument("--no-train-sat-condition", dest="train_sat_condition", action="store_false")
    parser.add_argument("--sat-lr-scale", type=float, default=1.0)
    parser.add_argument(
        "--satellite-condition-dropout-prob",
        type=float,
        default=0.0,
        help="Training-only probability of zeroing all satellite tokens for a sample so LiDAR must carry geometry.",
    )
    parser.add_argument(
        "--cs2s-train-scope",
        action="store_true",
        help=(
            "Follow the original CS2S KITTI optimizer scope: train full denoise UNet, "
            "satellite encoder, and LiDAR context together instead of the LiDAR-only freeze branch."
        ),
    )
    parser.add_argument("--train-denoise", choices=["none", "partial", "all"], default="none")
    parser.add_argument("--lidar-unfreeze-output-blocks", type=int, default=2)
    parser.add_argument("--lidar-unfreeze-out", action="store_true")
    parser.add_argument(
        "--lidar-unfreeze-transformers",
        default="output_middle",
        choices=["none", "input", "middle", "output", "output_middle", "all"],
    )
    parser.add_argument("--lidar-unet-lr-scale", type=float, default=0.05)
    parser.add_argument(
        "--lidar-unet-new-lr-scale",
        type=float,
        default=1.0,
        help="LR scale for newly added UNet LiDAR attention/router/gate params only.",
    )
    return parser.parse_args()


def move_batch_to_cuda(batch):
    return {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def foreground_lidar_mask_stats(batch, dilation, use_intersection):
    foreground = batch.get("foreground_mask")
    lidar_cond = batch.get("lidar_cond")
    if not torch.is_tensor(foreground):
        return {
            "foreground_lidar_hit_coverage": 0.0,
            "foreground_loss_mask_coverage": 0.0,
            "foreground_loss_mask_to_foreground_ratio": 0.0,
        }
    foreground = foreground.float()
    if foreground.ndim == 3:
        foreground = foreground[:, None]
    foreground = foreground.clamp(0.0, 1.0)
    loss_mask = foreground
    lidar_hit = torch.zeros_like(foreground)
    if torch.is_tensor(lidar_cond) and lidar_cond.shape[1] >= 2:
        point_channel = 7 if lidar_cond.shape[1] >= 8 else 1
        lidar_hit = (lidar_cond[:, point_channel : point_channel + 1].float() > 0.0).float()
        if lidar_hit.shape[-2:] != foreground.shape[-2:]:
            lidar_hit = F.interpolate(lidar_hit, size=foreground.shape[-2:], mode="nearest")
        dilation = int(dilation)
        if dilation > 0:
            kernel = 2 * dilation + 1
            lidar_hit = F.max_pool2d(lidar_hit, kernel_size=kernel, stride=1, padding=dilation)
        lidar_hit = lidar_hit.clamp(0.0, 1.0)
        if use_intersection:
            loss_mask = foreground * lidar_hit
    foreground_sum = foreground.sum().clamp_min(1e-6)
    return {
        "foreground_lidar_hit_coverage": scalar(lidar_hit.mean()),
        "foreground_loss_mask_coverage": scalar(loss_mask.mean()),
        "foreground_loss_mask_to_foreground_ratio": scalar(loss_mask.sum() / foreground_sum),
    }


def lidar_support_mask_stats(batch, dilation, latent_size=(16, 64)):
    lidar_cond = batch.get("lidar_cond")
    if not torch.is_tensor(lidar_cond) or lidar_cond.shape[1] < 2:
        return {
            "lidar_support_image_mask_coverage": 0.0,
            "lidar_support_latent_mask_coverage": 0.0,
        }
    point_channel = 7 if lidar_cond.shape[1] >= 8 else 1
    image_mask = (lidar_cond[:, point_channel : point_channel + 1].float() > 0.0).float()
    dilation = int(dilation)
    if dilation > 0:
        kernel = 2 * dilation + 1
        image_mask = F.max_pool2d(image_mask, kernel_size=kernel, stride=1, padding=dilation)
    image_mask = image_mask.clamp(0.0, 1.0)
    latent_h, latent_w = latent_size
    if image_mask.shape[-2] % latent_h == 0 and image_mask.shape[-1] % latent_w == 0:
        kernel = (image_mask.shape[-2] // latent_h, image_mask.shape[-1] // latent_w)
        latent_mask = F.max_pool2d(image_mask, kernel_size=kernel, stride=kernel).clamp(0.0, 1.0)
    else:
        latent_mask = F.interpolate(image_mask, size=(latent_h, latent_w), mode="area").clamp(0.0, 1.0)
    return {
        "lidar_support_image_mask_coverage": scalar(image_mask.mean()),
        "lidar_support_latent_mask_coverage": scalar(latent_mask.mean()),
    }


def _proc_rss_mb():
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        return -1.0


def _mem_available_mb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def resource_metrics():
    metrics = {
        "cpu_rss_mb": round(_proc_rss_mb(), 2),
        "mem_available_mb": round(_mem_available_mb(), 2),
    }
    if torch.cuda.is_available():
        metrics.update(
            {
                "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
                "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
                "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
            }
        )
    return metrics


def configure_cfg(cfg, args):
    cfg.model.base_learning_rate = args.lr
    cfg.model.params.pre_sat2grd_model_path = None
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.use_lidar_cond = args.condition_mode != "none"
    cfg.model.params.freeze_for_lidar_control = bool(args.condition_mode != "none" and not args.cs2s_train_scope)
    cfg.model.params.lidar_context_lr_scale = float(args.lidar_context_lr_scale)
    cfg.model.params.lidar_train_sat_condition = bool(args.train_sat_condition)
    cfg.model.params.lidar_sat_lr_scale = float(args.sat_lr_scale)
    cfg.model.params.lidar_unfreeze_denoise_all = args.train_denoise == "all"
    cfg.model.params.lidar_unfreeze_output_blocks = 0 if args.train_denoise == "none" else int(args.lidar_unfreeze_output_blocks)
    cfg.model.params.lidar_unfreeze_out = bool(args.lidar_unfreeze_out and args.train_denoise != "none")
    cfg.model.params.lidar_unfreeze_transformers = "none" if args.train_denoise == "none" else args.lidar_unfreeze_transformers
    cfg.model.params.lidar_unet_lr_scale = float(args.lidar_unet_lr_scale)
    cfg.model.params.lidar_unet_new_lr_scale = float(args.lidar_unet_new_lr_scale)

    cfg.model.params.dynamic_loss_weight = float(args.dynamic_loss_weight)
    cfg.model.params.dynamic_x0_loss_weight = float(args.dynamic_x0_loss_weight)
    cfg.model.params.dynamic_image_loss_weight = float(args.dynamic_image_loss_weight)
    cfg.model.params.dynamic_crop_image_loss_weight = float(args.dynamic_crop_image_loss_weight)
    cfg.model.params.dynamic_point_loss_weight = float(args.lidar_support_loss_weight)
    cfg.model.params.dynamic_point_x0_loss_weight = float(args.lidar_support_x0_loss_weight)
    cfg.model.params.dynamic_point_image_loss_weight = float(args.lidar_support_image_loss_weight)
    cfg.model.params.dynamic_point_dilation = int(args.lidar_support_dilation)
    cfg.model.params.lidar_depth_loss_weight = float(args.lidar_depth_loss_weight)
    cfg.model.params.lidar_depth_log_eps = float(args.lidar_depth_log_eps)
    cfg.model.params.static_teacher_consistency_weight = float(args.static_teacher_consistency_weight)
    cfg.model.params.static_teacher_gate_channel = int(lidar_condition_gate_channel(args.condition_mode))
    cfg.model.params.foreground_loss_weight = float(args.foreground_loss_weight)
    cfg.model.params.foreground_x0_loss_weight = float(args.foreground_x0_loss_weight)
    cfg.model.params.foreground_image_loss_weight = float(args.foreground_image_loss_weight)
    cfg.model.params.foreground_lpips_loss_weight = float(args.foreground_lpips_loss_weight)
    cfg.model.params.foreground_lpips_padding = int(args.foreground_lpips_padding)
    cfg.model.params.foreground_lpips_size = int(args.foreground_lpips_size)
    cfg.model.params.foreground_lidar_intersection = bool(args.foreground_lidar_intersection)
    cfg.model.params.lidar_zero_reconstruction_loss_weight = float(args.lidar_zero_reconstruction_loss_weight)
    cfg.model.params.lidar_zero_reconstruction_mask_mode = args.lidar_zero_reconstruction_mask_mode
    cfg.model.params.lidar_counterfactual_weight = float(args.lidar_counterfactual_weight)
    cfg.model.params.lidar_counterfactual_margin = float(args.lidar_counterfactual_margin)
    cfg.model.params.lidar_counterfactual_probes = args.lidar_counterfactual_probes
    cfg.model.params.lidar_counterfactual_stop_negative = bool(args.lidar_counterfactual_stop_negative)
    cfg.model.params.lidar_counterfactual_separation_weight = float(args.lidar_counterfactual_separation_weight)
    cfg.model.params.lidar_counterfactual_point_fallback = bool(args.lidar_counterfactual_point_fallback)
    cfg.model.params.lidar_counterfactual_exist_weight = float(args.lidar_counterfactual_exist_weight)
    cfg.model.params.satellite_condition_dropout_prob = float(args.satellite_condition_dropout_prob)

    unet = cfg.model.params.DDPM_config.params.unet_config.params
    unet.use_checkpoint = False
    cfg.model.params.DDPM_config.params.control_grd = None
    use_dual_attention = args.condition_mode != "none"
    unet.use_lidar_cross_attention = bool(use_dual_attention)
    unet.lidar_context_dim = int(args.lidar_token_dim)
    unet.lidar_gate_init = float(args.lidar_attn_gate_init)
    unet.lidar_evidence_channels = 8 if use_dual_attention else 0
    unet.lidar_attention_mode = args.lidar_attention_mode
    unet.lidar_reference_window = int(args.lidar_reference_window)
    if use_dual_attention:
        cfg.model.params.Lidar_context_config = {
            "target": "models.KITTI_geo_ldm.lidar_condition_model.LidarRangeTokenEncoder",
            "params": {
                "front_in_channels": int(lidar_condition_channels(args.condition_mode)),
                "range_in_channels": 3,
                "hidden_channels": int(args.lidar_token_hidden_channels),
                "range_feature_channels": int(args.range_feature_channels),
                "token_dim": int(args.lidar_token_dim),
                "token_grid": [int(args.lidar_token_height), int(args.lidar_token_width)],
                "range_depth_samples": int(args.lidar_token_range_depth_samples),
                "image_size": [
                    int(cfg.data.params.train.params.image_height),
                    int(cfg.data.params.train.params.image_width),
                ],
                "use_evidence_maps": True,
                "evidence_dilation": int(args.lidar_evidence_dilation),
                "evidence_free_space_dilation": int(args.lidar_evidence_free_space_dilation),
                "token_output_norm": args.lidar_token_output_norm,
                "token_structure_target_ratio": float(args.lidar_token_structure_target_ratio),
                "use_pointmap_pe": bool(
                    lidar_condition_uses_pointmap(args.condition_mode) and args.lidar_pointmap_pe
                ),
            },
        }
    else:
        cfg.model.params.Lidar_context_config = None

    cfg.data.params.batch_size = int(args.batch_size)
    cfg.data.params.num_workers = int(args.num_workers)
    cfg.data.params.train.params.manifest = args.train_manifest
    cfg.data.params.train.params.condition_mode = args.condition_mode
    cfg.data.params.train.params.include_range_image = args.condition_mode != "none"
    cfg.data.params.train.params.foreground_mask_root = args.foreground_mask_root
    cfg.data.params.train.params.foreground_mask_suffix = args.foreground_mask_suffix
    cfg.data.params.train.params.include_tracklets = bool(args.include_tracklets)
    cfg.data.params.test.params.manifest = args.val_manifest
    cfg.data.params.test.params.condition_mode = args.condition_mode
    cfg.data.params.test.params.include_range_image = args.condition_mode != "none"
    cfg.data.params.test.params.foreground_mask_root = args.foreground_mask_root
    cfg.data.params.test.params.foreground_mask_suffix = args.foreground_mask_suffix
    cfg.data.params.test.params.include_tracklets = bool(args.include_tracklets)
    return cfg


def _load_lidar_trainable_payload(model, payload):
    loaded = {"step": int(payload.get("step", 0))}
    if "control_grd" in payload and getattr(model.DDPM, "control_grd", None) is not None:
        loaded["control_grd"] = load_state_dict_compatible(model.DDPM.control_grd, payload["control_grd"])
    if "lidar_context_model" in payload and getattr(model, "lidar_context_model", None) is not None:
        loaded["lidar_context_model"] = load_state_dict_compatible(model.lidar_context_model, payload["lidar_context_model"])
    if "condition_model_sat" in payload:
        loaded["condition_model_sat"] = load_state_dict_compatible(model.condition_model_sat, payload["condition_model_sat"])
    if "denoise_model_trainable" in payload:
        loaded["denoise_model_trainable"] = load_state_dict_compatible(
            model.DDPM.denoise_model,
            payload["denoise_model_trainable"],
        )
    if "dynamic_class_tokens" in payload and getattr(model, "dynamic_class_tokens", None) is not None:
        if tuple(model.dynamic_class_tokens.shape) == tuple(payload["dynamic_class_tokens"].shape):
            model.dynamic_class_tokens.data.copy_(payload["dynamic_class_tokens"])
            loaded["dynamic_class_tokens"] = True
        else:
            loaded["dynamic_class_tokens"] = False
    return loaded


def load_state_dict_compatible(module, state_dict):
    current = module.state_dict()
    compatible = {}
    skipped = []
    for name, tensor in state_dict.items():
        if name in current and tuple(current[name].shape) == tuple(tensor.shape):
            compatible[name] = tensor
        else:
            skipped.append(name)
    missing, unexpected = module.load_state_dict(compatible, strict=False)
    return {
        "loaded_tensors": len(compatible),
        "source_tensors": len(state_dict),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "skipped_shape_or_missing": len(skipped),
    }


def load_lidar_trainable_checkpoint(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    loaded = _load_lidar_trainable_payload(model, payload)
    del payload
    gc.collect()
    return loaded


def load_training_checkpoint(model, optimizer, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    _load_lidar_trainable_payload(model, payload)
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    step = int(payload.get("step", 0))
    del payload
    gc.collect()
    return step


def _prefixed_state_dict(state_dict, prefix):
    return {name[len(prefix):]: value for name, value in state_dict.items() if name.startswith(prefix)}


def load_cs2s_backbone(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    loaded = {}
    denoise_state = _prefixed_state_dict(state_dict, "DDPM.denoise_model.")
    if denoise_state:
        missing, unexpected = model.DDPM.denoise_model.load_state_dict(denoise_state, strict=False)
        loaded["denoise_model"] = {"loaded_tensors": len(denoise_state), "missing": len(missing), "unexpected": len(unexpected)}
    sat_state = _prefixed_state_dict(state_dict, "condition_model_sat.")
    if sat_state:
        missing, unexpected = model.condition_model_sat.load_state_dict(sat_state, strict=False)
        loaded["condition_model_sat"] = {"loaded_tensors": len(sat_state), "missing": len(missing), "unexpected": len(unexpected)}
    ae_state = _prefixed_state_dict(state_dict, "pre_AE_model.")
    if ae_state:
        missing, unexpected = model.pre_AE_model.load_state_dict(ae_state, strict=False)
        loaded["pre_AE_model"] = {"loaded_tensors": len(ae_state), "missing": len(missing), "unexpected": len(unexpected)}
    del payload
    gc.collect()
    return loaded


def trainable_state_dict(module):
    return {name: param.detach().cpu() for name, param in module.named_parameters() if param.requires_grad}


def save_checkpoint(path, model, optimizer, step, args, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "args": vars(args),
        "metadata": metadata,
    }
    if args.save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    if getattr(model.DDPM, "control_grd", None) is not None:
        payload["control_grd"] = model.DDPM.control_grd.state_dict()
    if getattr(model, "lidar_context_model", None) is not None:
        payload["lidar_context_model"] = model.lidar_context_model.state_dict()
    sat_state = trainable_state_dict(model.condition_model_sat)
    if sat_state:
        payload["condition_model_sat"] = model.condition_model_sat.state_dict()
    denoise_state = trainable_state_dict(model.DDPM.denoise_model)
    if denoise_state:
        payload["denoise_model_trainable"] = denoise_state
    if getattr(model, "dynamic_class_tokens", None) is not None:
        payload["dynamic_class_tokens"] = model.dynamic_class_tokens.detach().cpu()
    torch.save(payload, path)


def prune_step_checkpoints(ckpt_dir, keep_count):
    keep_count = int(keep_count)
    if keep_count < 0:
        return []
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"))
    remove_count = len(checkpoints) - keep_count
    if remove_count <= 0:
        return []
    removed = []
    for path in checkpoints[:remove_count]:
        path.unlink(missing_ok=True)
        removed.append(str(path))
    return removed


def build_inline_sample_dataset(args):
    if int(args.sample_every) <= 0 or not args.sample_manifest:
        return None
    return SatLidarRawDataset(
        manifest=args.sample_manifest,
        condition_mode=args.condition_mode,
        image_height=128,
        image_width=512,
        sat_size=256,
        max_depth=80.0,
        align_satellite_to_camera=True,
        include_range_image=True,
    )


@torch.no_grad()
def run_inline_samples(model, sample_dataset, args, out_dir, step):
    if sample_dataset is None:
        return None
    probes = [item.strip() for item in str(args.sample_probes).split(",") if item.strip()]
    unsupported = [probe for probe in probes if probe not in {"normal", "zero"}]
    if unsupported:
        raise ValueError(
            "Unsupported sample probes: "
            + ", ".join(unsupported)
            + ". Global LiDAR shift_x was removed because it moves road/background geometry; use normal,zero."
        )
    if not probes:
        return None
    sample_out = Path(out_dir) / "samples" / f"step_{int(step):06d}"
    sample_out.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    records = []
    try:
        for idx in range(min(int(args.sample_num_samples), len(sample_dataset))):
            sample = sample_dataset[idx]
            sample_id = sample["sample_id"]
            safe_id = safe_sample_id(sample_id)
            gt_path = sample_out / "images" / "gt" / f"{safe_id}.png"
            overlay_path = sample_out / "images" / "lidar_overlay" / f"{safe_id}.png"
            cond_path = sample_out / "images" / "lidar_cond" / f"{safe_id}.png"
            target = sample["grd_left_imgs"].unsqueeze(0).clamp(0.0, 1.0)
            save_tensor_image(target[0], gt_path)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            make_lidar_overlay(target[0], sample["lidar_cond"], sample.get("dynamic_mask")).save(overlay_path)
            save_tensor_image(make_condition_rgb(sample["lidar_cond"]), cond_path)

            image_paths = {"GT": gt_path, "LiDAR overlay": overlay_path}
            attention_by_probe = {}
            batch = sample_to_batch(sample)
            try:
                for probe in probes:
                    pred, _, attention_stats, key_structure_stats = generate_prediction(
                        model,
                        batch,
                        probe=probe,
                        ddim_steps=int(args.sample_ddim_steps),
                        seed=int(args.sample_seed) + int(step) + idx,
                        guidance_scale=7.5,
                        eta=1.0,
                        temperature=1.0,
                        use_lidar=True,
                        key_stats_max_tokens=256,
                    )
                    pred_path = sample_out / "images" / probe / f"{safe_id}.png"
                    save_tensor_image(pred[0], pred_path)
                    image_paths[f"trained:{probe}"] = pred_path
                    attention_by_probe[f"trained:{probe}"] = {
                        "attention": attention_stats,
                        "key_structure": key_structure_stats,
                    }
                    del pred
                    torch.cuda.empty_cache()
                panel_path = make_panel(sample_out, sample_id, image_paths)
                records.append(
                    {
                        "sample_id": sample_id,
                        "panel_path": str(panel_path),
                        "step": int(step),
                        "lidar_attention_stats": attention_by_probe,
                        **{key: str(value) for key, value in image_paths.items()},
                    }
                )
            finally:
                del batch
                torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()
    (sample_out / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True))
    return {"sample_step": int(step), "sample_out": str(sample_out), "num_samples": len(records)}


def count_trainable(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def lidar_gate_mean(model):
    gates = []
    for name, param in model.DDPM.denoise_model.named_parameters():
        if name.endswith("lidar_gate"):
            gates.append(torch.sigmoid(param.detach()).float().mean().cpu())
    if not gates:
        return 0.0
    return float(torch.stack(gates).mean())


def iter_lidar_gate_modules(model):
    for module in model.DDPM.denoise_model.modules():
        if hasattr(module, "lidar_gate"):
            yield module


def set_lidar_gate_cap(model, cap):
    cap = max(0.0, min(1.0, float(cap)))
    count = 0
    for module in iter_lidar_gate_modules(model):
        module.lidar_gate_cap = cap
        count += 1
    return {"cap": cap, "count": count}


def scheduled_lidar_gate_cap(args, step):
    start = max(0.0, min(1.0, float(args.lidar_gate_cap_start)))
    end = max(0.0, min(1.0, float(args.lidar_gate_cap_end)))
    warmup_steps = max(0, int(args.lidar_gate_warmup_steps))
    if warmup_steps <= 0:
        return end
    progress = max(0.0, min(1.0, float(step) / float(warmup_steps)))
    return start + (end - start) * progress


def lidar_gate_stats(model):
    raw_gates = []
    caps = []
    effective_gates = []
    for module in iter_lidar_gate_modules(model):
        raw = torch.sigmoid(module.lidar_gate.detach()).float().mean().cpu()
        cap = float(getattr(module, "lidar_gate_cap", 1.0))
        raw_gates.append(raw)
        caps.append(torch.tensor(cap, dtype=torch.float32))
        effective_gates.append(raw * cap)
    if not raw_gates:
        return {
            "lidar_gate_raw_mean": 0.0,
            "lidar_gate_cap_mean": 0.0,
            "lidar_gate_effective_mean": 0.0,
        }
    return {
        "lidar_gate_raw_mean": float(torch.stack(raw_gates).mean()),
        "lidar_gate_cap_mean": float(torch.stack(caps).mean()),
        "lidar_gate_effective_mean": float(torch.stack(effective_gates).mean()),
    }


def lidar_gate_grad_stats(model):
    grads = []
    for name, param in model.DDPM.denoise_model.named_parameters():
        if name.endswith("lidar_gate") and param.grad is not None:
            grads.append(param.grad.detach().float().reshape(-1).cpu())
    if not grads:
        return {
            "lidar_gate_grad_count": 0,
            "lidar_gate_grad_mean": 0.0,
            "lidar_gate_grad_abs_mean": 0.0,
            "lidar_gate_grad_abs_max": 0.0,
            "lidar_gate_grad_pos_frac": 0.0,
            "lidar_gate_grad_neg_frac": 0.0,
        }
    grad = torch.cat(grads)
    return {
        "lidar_gate_grad_count": int(grad.numel()),
        "lidar_gate_grad_mean": float(grad.mean()),
        "lidar_gate_grad_abs_mean": float(grad.abs().mean()),
        "lidar_gate_grad_abs_max": float(grad.abs().max()),
        "lidar_gate_grad_pos_frac": float((grad > 0).float().mean()),
        "lidar_gate_grad_neg_frac": float((grad < 0).float().mean()),
    }


def module_stat_float(module, name):
    value = getattr(module, name)
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def lidar_attention_stats(model):
    entropy = []
    max_mean = []
    std_mean = []
    sim_std = []
    sim_range = []
    token_count = []
    query_count = []
    for module in model.DDPM.denoise_model.modules():
        if hasattr(module, "last_attn_entropy_norm"):
            entropy.append(torch.tensor(module_stat_float(module, "last_attn_entropy_norm"), dtype=torch.float32))
            max_mean.append(torch.tensor(module_stat_float(module, "last_attn_max_mean"), dtype=torch.float32))
            std_mean.append(torch.tensor(module_stat_float(module, "last_attn_std_mean"), dtype=torch.float32))
            sim_std.append(torch.tensor(module_stat_float(module, "last_sim_std_mean"), dtype=torch.float32))
            sim_range.append(torch.tensor(module_stat_float(module, "last_sim_range_mean"), dtype=torch.float32))
            token_count.append(torch.tensor(module_stat_float(module, "last_attn_token_count"), dtype=torch.float32))
            query_count.append(torch.tensor(module_stat_float(module, "last_attn_query_count"), dtype=torch.float32))
    if not entropy:
        return {
            "lidar_attn_modules": 0,
            "lidar_sim_std_mean": 0.0,
            "lidar_sim_range_mean": 0.0,
            "lidar_attn_entropy_norm_mean": 0.0,
            "lidar_attn_max_mean": 0.0,
            "lidar_attn_std_mean": 0.0,
            "lidar_attn_token_count_mean": 0.0,
            "lidar_attn_query_count_mean": 0.0,
        }
    return {
        "lidar_attn_modules": len(entropy),
        "lidar_sim_std_mean": float(torch.stack(sim_std).mean()),
        "lidar_sim_range_mean": float(torch.stack(sim_range).mean()),
        "lidar_attn_entropy_norm_mean": float(torch.stack(entropy).mean()),
        "lidar_attn_max_mean": float(torch.stack(max_mean).mean()),
        "lidar_attn_std_mean": float(torch.stack(std_mean).mean()),
        "lidar_attn_token_count_mean": float(torch.stack(token_count).mean()),
        "lidar_attn_query_count_mean": float(torch.stack(query_count).mean()),
    }


def lidar_context_token_stats(model):
    context_model = getattr(model, "lidar_context_model", None)
    if context_model is None:
        return {
            "lidar_token_mean_norm": 0.0,
            "lidar_token_centered_norm": 0.0,
            "lidar_token_centered_to_mean_ratio": 0.0,
            "lidar_token_var_mean": 0.0,
            "lidar_token_proj_bias_norm": 0.0,
            "lidar_token_output_mean_norm": 0.0,
            "lidar_token_output_centered_norm": 0.0,
            "lidar_token_output_centered_to_mean_ratio": 0.0,
            "lidar_token_output_var_mean": 0.0,
            "lidar_pointmap_coverage": 0.0,
        }
    return {
        "lidar_token_mean_norm": scalar(getattr(context_model, "last_token_mean_norm", torch.tensor(0.0))),
        "lidar_token_centered_norm": scalar(
            getattr(context_model, "last_token_centered_norm", torch.tensor(0.0))
        ),
        "lidar_token_centered_to_mean_ratio": scalar(
            getattr(context_model, "last_token_centered_to_mean_ratio", torch.tensor(0.0))
        ),
        "lidar_token_var_mean": scalar(getattr(context_model, "last_token_var_mean", torch.tensor(0.0))),
        "lidar_token_proj_bias_norm": scalar(
            getattr(context_model, "last_token_proj_bias_norm", torch.tensor(0.0))
        ),
        "lidar_token_output_mean_norm": scalar(
            getattr(context_model, "last_token_output_mean_norm", torch.tensor(0.0))
        ),
        "lidar_token_output_centered_norm": scalar(
            getattr(context_model, "last_token_output_centered_norm", torch.tensor(0.0))
        ),
        "lidar_token_output_centered_to_mean_ratio": scalar(
            getattr(context_model, "last_token_output_centered_to_mean_ratio", torch.tensor(0.0))
        ),
        "lidar_token_output_var_mean": scalar(
            getattr(context_model, "last_token_output_var_mean", torch.tensor(0.0))
        ),
        "lidar_pointmap_coverage": scalar(getattr(context_model, "last_pointmap_coverage", torch.tensor(0.0))),
    }


def lidar_token_structure_loss_tensor(model):
    context_model = getattr(model, "lidar_context_model", None)
    if context_model is not None:
        loss = getattr(context_model, "last_token_structure_loss", None)
        if torch.is_tensor(loss):
            return loss
    return torch.zeros((), device=next(model.parameters()).device)


def force_lidar_gate(model, gate_value):
    if not (0.0 < float(gate_value) < 1.0):
        return {"enabled": False, "count": 0}
    gate = torch.tensor(float(gate_value), dtype=torch.float32)
    logit = torch.log(gate / (1.0 - gate))
    count = 0
    for name, param in model.DDPM.denoise_model.named_parameters():
        if name.endswith("lidar_gate"):
            param.data.copy_(logit.to(device=param.device, dtype=param.dtype))
            count += 1
    return {"enabled": True, "count": count, "gate": float(gate_value)}


def build_loader(dataset, args):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": bool(args.shuffle),
        "num_workers": args.num_workers,
        "drop_last": False,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main():
    args = parse_args()
    effective_train_denoise = "all" if args.cs2s_train_scope else args.train_denoise
    if args.cs2s_train_scope and args.train_denoise != "all":
        print(
            json.dumps(
                {
                    "warning": "--cs2s-train-scope follows original CS2S optimizer scope and trains the full denoise UNet.",
                    "requested_train_denoise": args.train_denoise,
                    "effective_train_denoise": effective_train_denoise,
                },
                sort_keys=True,
            )
        )
    if args.resume_ckpt and args.lidar_warmstart_ckpt:
        raise ValueError("--resume-ckpt and --lidar-warmstart-ckpt are mutually exclusive")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = configure_cfg(OmegaConf.load(args.config), args)
    run_name = args.run_name or f"lidar_dominant_sdinit_{args.steps}step"
    out_dir = Path(args.out_root) / run_name
    metrics_dir = out_dir / "metrics"
    ckpt_dir = out_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    train_dataset = instantiate_from_config(cfg.data.params.train)
    loader = build_loader(train_dataset, args)
    iterator = iter(loader)
    inline_sample_dataset = build_inline_sample_dataset(args)

    model = instantiate_from_config(cfg.model).cuda()
    cs2s_loaded = {}
    if args.cs2s_init_ckpt:
        cs2s_loaded = load_cs2s_backbone(model, args.cs2s_init_ckpt)
        print(json.dumps({"cs2s_init_ckpt": args.cs2s_init_ckpt, "loaded": cs2s_loaded}, sort_keys=True))
    model.learning_rate = args.lr
    optimizer = model.configure_optimizers()[0]
    scaler = GradScaler(enabled=bool(args.amp and torch.cuda.is_available()))
    start_step = 0
    warmstart_loaded = {}
    if args.lidar_warmstart_ckpt:
        warmstart_loaded = load_lidar_trainable_checkpoint(model, args.lidar_warmstart_ckpt)
        print(json.dumps({"lidar_warmstart_ckpt": args.lidar_warmstart_ckpt, "loaded": warmstart_loaded}, sort_keys=True))
    if args.resume_ckpt:
        start_step = load_training_checkpoint(model, optimizer, args.resume_ckpt)
        print(json.dumps({"resumed": args.resume_ckpt, "start_step": start_step}, sort_keys=True))
    forced_gate = force_lidar_gate(model, args.force_lidar_gate)
    if forced_gate["enabled"]:
        print(
            json.dumps(
                {"forced_lidar_gate": forced_gate, "lidar_gate_mean": lidar_gate_mean(model), **lidar_gate_stats(model)},
                sort_keys=True,
            )
        )
    initial_gate_cap = set_lidar_gate_cap(model, scheduled_lidar_gate_cap(args, start_step))
    model.train()

    metadata = {
        "sd_base_ckpt": args.sd_base_ckpt,
        "cs2s_init_ckpt": args.cs2s_init_ckpt,
        "lidar_warmstart_ckpt": args.lidar_warmstart_ckpt,
        "resume_ckpt": args.resume_ckpt,
        "start_step": int(start_step),
        "save_optimizer": bool(args.save_optimizer),
        "keep_step_checkpoints": int(args.keep_step_checkpoints),
        "sample_every": int(args.sample_every),
        "sample_manifest": args.sample_manifest,
        "sample_num_samples": int(args.sample_num_samples),
        "sample_ddim_steps": int(args.sample_ddim_steps),
        "sample_probes": args.sample_probes,
        "deprecated_sample_shift_fraction_noop": float(args.sample_shift_fraction),
        "uses_result_kitti_ckpt": bool(args.cs2s_init_ckpt),
        "cs2s_loaded": cs2s_loaded,
        "lidar_warmstart_loaded": warmstart_loaded,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "train_size": len(train_dataset),
        "steps_per_epoch_batch1": len(train_dataset),
        "lidar_injection": args.lidar_injection,
        "use_lidar_control_residual": False,
        "use_lidar_cross_attention": bool(cfg.model.params.DDPM_config.params.unet_config.params.use_lidar_cross_attention),
        "lidar_evidence_channels": int(cfg.model.params.DDPM_config.params.unet_config.params.lidar_evidence_channels),
        "train_sat_condition": bool(args.train_sat_condition),
        "train_denoise": args.train_denoise,
        "effective_train_denoise": effective_train_denoise,
        "cs2s_train_scope": bool(args.cs2s_train_scope),
        "freeze_for_lidar_control": bool(cfg.model.params.freeze_for_lidar_control),
        "optimizer_scope": "lidar_freeze_branch" if cfg.model.params.freeze_for_lidar_control else "cs2s_full_denoise_sat_lidar",
        "base_lr": float(args.lr),
        "effective_lidar_unet_lr": float(args.lr)
        if not cfg.model.params.freeze_for_lidar_control
        else float(args.lr) * float(args.lidar_unet_lr_scale),
        "effective_lidar_unet_new_lr": float(args.lr)
        if not cfg.model.params.freeze_for_lidar_control
        else float(args.lr) * float(args.lidar_unet_new_lr_scale),
        "effective_old_unet_lr": float(args.lr)
        if not cfg.model.params.freeze_for_lidar_control
        else float(args.lr) * float(args.lidar_unet_lr_scale),
        "lidar_unet_lr_scale": float(args.lidar_unet_lr_scale),
        "lidar_unet_new_lr_scale": float(args.lidar_unet_new_lr_scale),
        "lidar_context_lr": float(args.lr) * float(args.lidar_context_lr_scale),
        "lidar_token_output_norm": args.lidar_token_output_norm,
        "lidar_token_structure_loss_weight": float(args.lidar_token_structure_loss_weight),
        "lidar_token_structure_target_ratio": float(args.lidar_token_structure_target_ratio),
        "lidar_attention_mode": args.lidar_attention_mode,
        "lidar_reference_window": int(args.lidar_reference_window),
        "force_lidar_gate": float(args.force_lidar_gate),
        "forced_lidar_gate": forced_gate,
        "lidar_gate_cap_start": float(args.lidar_gate_cap_start),
        "lidar_gate_cap_end": float(args.lidar_gate_cap_end),
        "lidar_gate_warmup_steps": int(args.lidar_gate_warmup_steps),
        "initial_lidar_gate_cap": initial_gate_cap,
        "foreground_mask_root": args.foreground_mask_root,
        "foreground_mask_suffix": args.foreground_mask_suffix,
        "include_tracklets": bool(args.include_tracklets),
        "foreground_loss_weight": float(args.foreground_loss_weight),
        "foreground_x0_loss_weight": float(args.foreground_x0_loss_weight),
        "foreground_image_loss_weight": float(args.foreground_image_loss_weight),
        "foreground_lpips_loss_weight": float(args.foreground_lpips_loss_weight),
        "foreground_lidar_intersection": bool(args.foreground_lidar_intersection),
        "lidar_zero_reconstruction_loss_weight": float(args.lidar_zero_reconstruction_loss_weight),
        "lidar_zero_reconstruction_mask_mode": args.lidar_zero_reconstruction_mask_mode,
        "lidar_depth_loss_weight": float(args.lidar_depth_loss_weight),
        "lidar_depth_log_eps": float(args.lidar_depth_log_eps),
        "satellite_condition_dropout_prob": float(args.satellite_condition_dropout_prob),
        "lidar_counterfactual_weight": float(args.lidar_counterfactual_weight),
        "lidar_counterfactual_margin": float(args.lidar_counterfactual_margin),
        "lidar_counterfactual_probes": args.lidar_counterfactual_probes,
        "lidar_counterfactual_stop_negative": bool(args.lidar_counterfactual_stop_negative),
        "lidar_counterfactual_separation_weight": float(args.lidar_counterfactual_separation_weight),
        "lidar_counterfactual_point_fallback": bool(args.lidar_counterfactual_point_fallback),
        "lidar_counterfactual_exist_weight": float(args.lidar_counterfactual_exist_weight),
        "trainable_lidar_context_params": count_trainable(model.lidar_context_model) if getattr(model, "lidar_context_model", None) is not None else 0,
        "trainable_sat_condition_params": count_trainable(model.condition_model_sat),
        "trainable_denoise_params": count_trainable(model.DDPM.denoise_model),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({"start": True, **metadata, **resource_metrics()}, sort_keys=True))

    metrics_path = metrics_dir / "train_metrics.jsonl"
    mode = "a" if args.resume_ckpt and start_step > 0 else "w"
    with metrics_path.open(mode) as metrics_file:
        window = {
            "steps": 0,
            "foreground_available": 0,
            "foreground_mask_coverage": 0.0,
            "foreground_lidar_hit_coverage": 0.0,
            "foreground_loss_mask_coverage": 0.0,
            "foreground_loss_mask_nonzero_steps": 0,
            "foreground_loss_mask_to_foreground_ratio": 0.0,
            "lidar_support_image_mask_coverage": 0.0,
            "lidar_support_latent_mask_coverage": 0.0,
            "lidar_counterfactual_applied_steps": 0,
            "lidar_counterfactual_exist_active_steps": 0,
            "lidar_counterfactual_exist_loss": 0.0,
            "lidar_counterfactual_zero_minus_normal_l1": 0.0,
            "lidar_counterfactual_zero_minus_normal_depth_log_l1": 0.0,
            "lidar_depth_log_l1": 0.0,
            "lidar_depth_log_l1_contrib": 0.0,
            "lidar_depth_mask_coverage": 0.0,
            "lidar_bottleneck_depth_log_l1": 0.0,
            "lidar_bottleneck_depth_log_l1_contrib": 0.0,
            "lidar_bottleneck_depth_mask_coverage": 0.0,
            "satellite_condition_dropout_applied_steps": 0,
            "satellite_condition_dropout_fraction": 0.0,
        }
        for step in range(start_step + 1, args.steps + 1):
            lidar_gate_cap = scheduled_lidar_gate_cap(args, step)
            set_lidar_gate_cap(model, lidar_gate_cap)
            batch_cpu, iterator = next_batch(loader, iterator)
            batch = move_batch_to_cuda(batch_cpu)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(args.amp and torch.cuda.is_available())):
                loss = model.training_step(batch, step)
                token_structure_loss = lidar_token_structure_loss_tensor(model)
                token_structure_loss_contrib = (
                    float(args.lidar_token_structure_loss_weight) * token_structure_loss
                )
                loss = loss + token_structure_loss_contrib
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gate_grad_stats = lidar_gate_grad_stats(model)
            scaler.step(optimizer)
            scaler.update()

            foreground_mask_stats = foreground_lidar_mask_stats(
                batch,
                dilation=args.lidar_support_dilation,
                use_intersection=bool(args.foreground_lidar_intersection),
            )
            lidar_support_stats = lidar_support_mask_stats(batch, dilation=args.lidar_support_dilation)
            foreground_mask_coverage = scalar(
                batch.get("foreground_mask", torch.zeros(1, 1, 1, 1, device="cuda")).float().mean()
            )
            foreground_available = int(
                batch.get("foreground_mask_available", torch.zeros(1, device="cuda")).sum().detach().cpu()
            )
            window["steps"] += 1
            window["foreground_available"] += foreground_available
            window["foreground_mask_coverage"] += foreground_mask_coverage
            window["foreground_lidar_hit_coverage"] += foreground_mask_stats["foreground_lidar_hit_coverage"]
            window["foreground_loss_mask_coverage"] += foreground_mask_stats["foreground_loss_mask_coverage"]
            window["foreground_loss_mask_to_foreground_ratio"] += foreground_mask_stats[
                "foreground_loss_mask_to_foreground_ratio"
            ]
            window["lidar_support_image_mask_coverage"] += lidar_support_stats["lidar_support_image_mask_coverage"]
            window["lidar_support_latent_mask_coverage"] += lidar_support_stats["lidar_support_latent_mask_coverage"]
            if foreground_mask_stats["foreground_loss_mask_coverage"] > 0.0:
                window["foreground_loss_mask_nonzero_steps"] += 1
            counterfactual_metrics = getattr(model, "last_lidar_counterfactual_metrics", {})
            counterfactual_applied = int(
                float(counterfactual_metrics.get("lidar_counterfactual_applied", 0.0)) > 0.0
            )
            counterfactual_exist_loss = float(
                counterfactual_metrics.get("lidar_counterfactual_exist_loss", 0.0)
            )
            counterfactual_zero_minus_normal_l1 = float(
                counterfactual_metrics.get("lidar_counterfactual_zero_l1", 0.0)
            ) - float(counterfactual_metrics.get("lidar_counterfactual_normal_l1", 0.0))
            counterfactual_zero_minus_normal_depth_l1 = float(
                counterfactual_metrics.get("lidar_counterfactual_zero_minus_normal_depth_log_l1", 0.0)
            )
            window["lidar_counterfactual_applied_steps"] += counterfactual_applied
            if counterfactual_applied:
                window["lidar_counterfactual_exist_loss"] += counterfactual_exist_loss
                window[
                    "lidar_counterfactual_zero_minus_normal_l1"
                ] += counterfactual_zero_minus_normal_l1
                window[
                    "lidar_counterfactual_zero_minus_normal_depth_log_l1"
                ] += counterfactual_zero_minus_normal_depth_l1
                if counterfactual_exist_loss > 1e-8:
                    window["lidar_counterfactual_exist_active_steps"] += 1
            loss_metrics = getattr(model.DDPM, "last_loss_metrics", {})
            window["lidar_depth_log_l1"] += float(loss_metrics.get("loss_lidar_depth_log_l1", 0.0))
            window["lidar_depth_log_l1_contrib"] += float(loss_metrics.get("loss_lidar_depth_log_l1_contrib", 0.0))
            window["lidar_depth_mask_coverage"] += float(loss_metrics.get("lidar_depth_mask_coverage", 0.0))
            window["lidar_bottleneck_depth_log_l1"] += float(
                loss_metrics.get("loss_lidar_bottleneck_depth_log_l1", 0.0)
            )
            window["lidar_bottleneck_depth_log_l1_contrib"] += float(
                loss_metrics.get("loss_lidar_bottleneck_depth_log_l1_contrib", 0.0)
            )
            window["lidar_bottleneck_depth_mask_coverage"] += float(
                loss_metrics.get("lidar_bottleneck_depth_mask_coverage", 0.0)
            )
            satellite_dropout_metrics = getattr(model, "last_satellite_condition_dropout_metrics", {})
            satellite_dropout_applied = int(
                float(satellite_dropout_metrics.get("satellite_condition_dropout_applied", 0.0)) > 0.0
            )
            satellite_dropout_fraction = float(
                satellite_dropout_metrics.get("satellite_condition_dropout_fraction", 0.0)
            )
            window["satellite_condition_dropout_applied_steps"] += satellite_dropout_applied
            window["satellite_condition_dropout_fraction"] += satellite_dropout_fraction

            if step % args.log_every == 0 or step == 1:
                window_steps = max(1, int(window["steps"]))
                window_counterfactual_applied_steps = max(
                    1, int(window["lidar_counterfactual_applied_steps"])
                )
                record = {
                    "step": step,
                    "loss": scalar(loss),
                    "condition_mode": args.condition_mode,
                    "num_projected_lidar_points": int(batch.get("num_projected_lidar_points", torch.zeros(1, device="cuda")).sum().detach().cpu()),
                    "num_dynamic_boxes": int(batch.get("num_dynamic_boxes", torch.zeros(1, device="cuda")).sum().detach().cpu()),
                    "lidar_support_loss_weight": args.lidar_support_loss_weight,
                    "lidar_support_x0_loss_weight": args.lidar_support_x0_loss_weight,
                    "lidar_support_image_loss_weight": args.lidar_support_image_loss_weight,
                    "dynamic_loss_weight": args.dynamic_loss_weight,
                    "dynamic_image_loss_weight": args.dynamic_image_loss_weight,
                    "foreground_mask_available": foreground_available,
                    "foreground_mask_coverage": foreground_mask_coverage,
                    **foreground_mask_stats,
                    **lidar_support_stats,
                    "window_steps": window_steps,
                    "window_foreground_available": int(window["foreground_available"]),
                    "window_foreground_nonzero_steps": int(window["foreground_loss_mask_nonzero_steps"]),
                    "window_foreground_mask_coverage_mean": window["foreground_mask_coverage"] / window_steps,
                    "window_foreground_lidar_hit_coverage_mean": window["foreground_lidar_hit_coverage"] / window_steps,
                    "window_foreground_loss_mask_coverage_mean": window["foreground_loss_mask_coverage"] / window_steps,
                    "window_foreground_loss_mask_to_foreground_ratio_mean": window[
                        "foreground_loss_mask_to_foreground_ratio"
                    ]
                    / window_steps,
                    "window_lidar_support_image_mask_coverage_mean": window[
                        "lidar_support_image_mask_coverage"
                    ]
                    / window_steps,
                    "window_lidar_support_latent_mask_coverage_mean": window[
                        "lidar_support_latent_mask_coverage"
                    ]
                    / window_steps,
                    "lidar_counterfactual_exist_active": float(counterfactual_exist_loss > 1e-8),
                    "lidar_counterfactual_zero_minus_normal_l1": counterfactual_zero_minus_normal_l1,
                    "window_lidar_counterfactual_applied_frac": window[
                        "lidar_counterfactual_applied_steps"
                    ]
                    / window_steps,
                    "window_lidar_counterfactual_exist_active_frac": window[
                        "lidar_counterfactual_exist_active_steps"
                    ]
                    / window_counterfactual_applied_steps,
                    "window_lidar_counterfactual_exist_loss_mean": window[
                        "lidar_counterfactual_exist_loss"
                    ]
                    / window_counterfactual_applied_steps,
                    "window_lidar_counterfactual_zero_minus_normal_l1_mean": window[
                        "lidar_counterfactual_zero_minus_normal_l1"
                    ]
                    / window_counterfactual_applied_steps,
                    "lidar_counterfactual_zero_minus_normal_depth_log_l1": counterfactual_zero_minus_normal_depth_l1,
                    "window_lidar_counterfactual_zero_minus_normal_depth_log_l1_mean": window[
                        "lidar_counterfactual_zero_minus_normal_depth_log_l1"
                    ]
                    / window_counterfactual_applied_steps,
                    "window_lidar_depth_log_l1_mean": window["lidar_depth_log_l1"] / window_steps,
                    "window_lidar_depth_log_l1_contrib_mean": window["lidar_depth_log_l1_contrib"] / window_steps,
                    "window_lidar_depth_mask_coverage_mean": window["lidar_depth_mask_coverage"] / window_steps,
                    "window_lidar_bottleneck_depth_log_l1_mean": window[
                        "lidar_bottleneck_depth_log_l1"
                    ]
                    / window_steps,
                    "window_lidar_bottleneck_depth_log_l1_contrib_mean": window[
                        "lidar_bottleneck_depth_log_l1_contrib"
                    ]
                    / window_steps,
                    "window_lidar_bottleneck_depth_mask_coverage_mean": window[
                        "lidar_bottleneck_depth_mask_coverage"
                    ]
                    / window_steps,
                    "window_satellite_condition_dropout_applied_frac": window[
                        "satellite_condition_dropout_applied_steps"
                    ]
                    / window_steps,
                    "window_satellite_condition_dropout_fraction_mean": window[
                        "satellite_condition_dropout_fraction"
                    ]
                    / window_steps,
                    "foreground_loss_weight": args.foreground_loss_weight,
                    "foreground_x0_loss_weight": args.foreground_x0_loss_weight,
                    "foreground_image_loss_weight": args.foreground_image_loss_weight,
                    "foreground_lpips_loss_weight": args.foreground_lpips_loss_weight,
                    "lidar_zero_reconstruction_loss_weight": float(args.lidar_zero_reconstruction_loss_weight),
                    "lidar_zero_reconstruction_mask_mode": args.lidar_zero_reconstruction_mask_mode,
                    "lidar_counterfactual_weight": args.lidar_counterfactual_weight,
                    "lidar_counterfactual_margin": args.lidar_counterfactual_margin,
                    "lidar_counterfactual_exist_weight": float(args.lidar_counterfactual_exist_weight),
                    "lidar_depth_loss_weight": float(args.lidar_depth_loss_weight),
                    "lidar_depth_log_eps": float(args.lidar_depth_log_eps),
                    "train_sat_condition": bool(args.train_sat_condition),
                    "train_denoise": args.train_denoise,
                    "effective_train_denoise": effective_train_denoise,
                    "lidar_token_output_norm": args.lidar_token_output_norm,
                    "lidar_token_structure_loss_weight": float(args.lidar_token_structure_loss_weight),
                    "lidar_token_structure_target_ratio": float(args.lidar_token_structure_target_ratio),
                    "lidar_token_structure_loss": scalar(token_structure_loss),
                    "lidar_token_structure_loss_contrib": scalar(token_structure_loss_contrib),
                    "lidar_token_valid_sample_ratio": scalar(getattr(getattr(model, "lidar_context_model", None), "last_valid_sample_ratio", torch.tensor(0.0))),
                    "lidar_evidence_hit_coverage": scalar(getattr(getattr(model, "lidar_context_model", None), "last_hit_coverage", torch.tensor(0.0))),
                    "lidar_evidence_empty_coverage": scalar(getattr(getattr(model, "lidar_context_model", None), "last_empty_coverage", torch.tensor(0.0))),
                    **lidar_context_token_stats(model),
                    "lidar_gate_mean": lidar_gate_mean(model),
                    **lidar_gate_stats(model),
                    **gate_grad_stats,
                    **lidar_attention_stats(model),
                    **satellite_dropout_metrics,
                    **loss_metrics,
                    **getattr(model, "last_lidar_counterfactual_metrics", {}),
                    **resource_metrics(),
                }
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                metrics_file.flush()
                print(json.dumps(record, sort_keys=True))
                window = {
                    "steps": 0,
                    "foreground_available": 0,
                    "foreground_mask_coverage": 0.0,
                    "foreground_lidar_hit_coverage": 0.0,
                    "foreground_loss_mask_coverage": 0.0,
                    "foreground_loss_mask_nonzero_steps": 0,
                    "foreground_loss_mask_to_foreground_ratio": 0.0,
                    "lidar_support_image_mask_coverage": 0.0,
                    "lidar_support_latent_mask_coverage": 0.0,
                    "lidar_counterfactual_applied_steps": 0,
                    "lidar_counterfactual_exist_active_steps": 0,
                    "lidar_counterfactual_exist_loss": 0.0,
                    "lidar_counterfactual_zero_minus_normal_l1": 0.0,
                    "lidar_counterfactual_zero_minus_normal_depth_log_l1": 0.0,
                    "lidar_depth_log_l1": 0.0,
                    "lidar_depth_log_l1_contrib": 0.0,
                    "lidar_depth_mask_coverage": 0.0,
                    "lidar_bottleneck_depth_log_l1": 0.0,
                    "lidar_bottleneck_depth_log_l1_contrib": 0.0,
                    "lidar_bottleneck_depth_mask_coverage": 0.0,
                    "satellite_condition_dropout_applied_steps": 0,
                    "satellite_condition_dropout_fraction": 0.0,
                }

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step, args, metadata)
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, step, args, metadata)
                removed = prune_step_checkpoints(ckpt_dir, args.keep_step_checkpoints)
                if removed:
                    print(json.dumps({"pruned_checkpoints": removed, "step": step}, sort_keys=True))
            if int(args.sample_every) > 0 and step % int(args.sample_every) == 0:
                sample_record = run_inline_samples(model, inline_sample_dataset, args, out_dir, step)
                if sample_record is not None:
                    print(json.dumps(sample_record, sort_keys=True))

            del batch, batch_cpu, loss, token_structure_loss, token_structure_loss_contrib

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, args.steps, args, metadata)
    print(json.dumps({"complete": True, "last_ckpt": str(ckpt_dir / "last.pt"), "steps": args.steps}, sort_keys=True))


if __name__ == "__main__":
    main()
