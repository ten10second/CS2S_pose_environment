import argparse
import gc
import json
import os
import shutil
import sys
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from tools.generate_kitti_raea_samples import (  # noqa: E402
    generate_prediction,
    make_condition_rgb,
    make_lidar_overlay,
    make_panel,
    safe_sample_id,
    sample_to_batch,
    save_tensor_image,
)
from utils.util import instantiate_from_config  # noqa: E402


CONDITION_MODE = "raw_lidar_pointmap"
LIDAR_GEOM_MODE = "ray_depth_inv"
LIDAR_CONTEXT_BACKBONE = "utonia_zbuffer_visible_ray"
LIDAR_PIXEL_CONTEXT_BACKBONE = "utonia_pixel_lidar_v21"
LIDAR_POINT_IN_CHANNELS = 10
LIDAR_POINT_FEATURE_DIM = 576
LIDAR_RAY_CACHE_PLANES = 1
LIDAR_PIXEL_SIZE = (128, 512)
IMAGE_SEMANTIC_DIM = 384
IMAGE_SEMANTIC_SIZE = (8, 32)
LIDAR_TOKEN_DIM = 768
LIDAR_TOKEN_HIDDEN_CHANNELS = 128
LIDAR_TOKEN_GRID = (8, 32)
LIDAR_PIXEL_PYRAMID_CHANNELS = (64, 128, 256, 256)
LIDAR_PIXEL_HIDDEN_CHANNELS = 64
LIDAR_SEMANTIC_MASK_MODE = "lidar_hit"
RAY_EVIDENCE_MASK_MODE = "lidar_hit"
LIDAR_TOKEN_OUTPUT_NORM = "center_layernorm"
LIDAR_PIXEL_CONTEXT_TARGET = "models.KITTI_geo_ldm.lidar_pixel_condition.LidarPixelConditionEncoder"


def is_lidar_pixel_config(cfg):
    target = OmegaConf.select(cfg, "model.params.Lidar_context_config.target")
    return str(target) == LIDAR_PIXEL_CONTEXT_TARGET


def uses_native_depth_head(cfg):
    return OmegaConf.select(cfg, "model.params.DDPM_config.params.unet_config.params.lidar_depth_head_mode") == "pixel"


def delete_config_key(config, key):
    if key in config:
        del config[key]


def parse_args():
    parser = argparse.ArgumentParser(description="Ray-posterior satellite-LiDAR KITTI street-view training.")
    parser.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml")
    parser.add_argument("--train-manifest", default="dataset/kitti_raw_sat_lidar/train_manifest.jsonl")
    parser.add_argument("--val-manifest", default="dataset/kitti_raw_sat_lidar/test2_manifest.jsonl")
    parser.add_argument(
        "--kitti-root",
        default="",
        help="Optional KITTI_RAW root used to rebase machine-specific paths stored in manifests.",
    )
    parser.add_argument("--sd-base-ckpt", default="/home/shizhm/Downloads/sd-v1-4.ckpt")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--out-root", default="results/kitti_xlidar_overfit")
    parser.add_argument("--steps", type=int, default=36000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument(
        "--dist-backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Distributed backend used under torchrun. auto selects NCCL on CUDA and Gloo otherwise.",
    )
    parser.add_argument(
        "--dist-timeout-seconds",
        type=int,
        default=600,
        help="Abort distributed collectives after this many seconds instead of hanging indefinitely.",
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_true",
        help="Allow conditional trainable branches to be unused in an individual DDP step.",
    )
    parser.add_argument(
        "--no-ddp-find-unused-parameters",
        dest="ddp_find_unused_parameters",
        action="store_false",
    )
    parser.set_defaults(ddp_find_unused_parameters=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument(
        "--dataloader-timeout",
        type=int,
        default=180,
        help="Seconds to wait for a worker batch before failing the run. Ignored with zero workers.",
    )
    parser.add_argument(
        "--parallel-model-init",
        action="store_true",
        help="Load base/resume checkpoints on all ranks concurrently. Disabled by default to limit host RAM and I/O spikes.",
    )
    parser.add_argument(
        "--allow-ddp-inline-sampling",
        action="store_true",
        help="Allow rank-0 inline DDIM sampling while other ranks wait. Prefer offline sampling on multi-GPU runs.",
    )
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
        default="normal",
        help="Comma-separated inline sample probes. Supported probes: normal,zero.",
    )
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--sample-fixed-seed", action="store_true", help="Reuse the same generation noise across checkpoints.")
    parser.add_argument("--sample-eta", type=float, default=1.0)
    parser.add_argument(
        "--keep-step-checkpoints",
        type=int,
        default=-1,
        help="Keep only the newest N step_*.pt checkpoints. Use -1 to keep all, 0 to keep none.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume-ckpt", default="")
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=50.0,
        help="Refuse checkpoint writes when output storage has less free space than this threshold.",
    )
    parser.add_argument(
        "--min-free-host-memory-gb",
        type=float,
        default=12.0,
        help="Require this much available host RAM before each rank loads model or resume state.",
    )

    parser.add_argument(
        "--lidar-ray-feature-cache-root",
        default="",
        help="Required float16 cache containing z-buffer-visible Utonia [576,1,8,32] features.",
    )
    parser.add_argument(
        "--lidar-pixel-feature-cache-root",
        default="",
        help="Required by V2.1 configs: float16 cache containing pixel LiDAR [576,128,512] features.",
    )
    parser.add_argument(
        "--image-semantic-cache-root",
        default="",
        help="Required float16 memmap cache containing 384x8x32 DINO features per frame.",
    )
    parser.add_argument("--lidar-semantic-alignment-weight", type=float, default=0.2)
    parser.add_argument(
        "--lidar-reference-window",
        type=int,
        default=3,
        help="Odd local window size for Pointmap-style LiDAR reference attention.",
    )
    parser.add_argument("--lidar-evidence-dilation", type=int, default=4)
    parser.add_argument("--lidar-evidence-free-space-dilation", type=int, default=14)
    parser.add_argument(
        "--lidar-token-structure-loss-weight",
        type=float,
        default=0.01,
        help="Weight for penalizing collapse of raw LiDAR token centered/mean ratio.",
    )
    parser.add_argument(
        "--lidar-token-structure-target-ratio",
        type=float,
        default=0.08,
        help="Minimum raw token centered/mean ratio encouraged by the structure regularizer.",
    )
    parser.add_argument("--lidar-support-loss-weight", type=float, default=1.0)
    parser.add_argument("--lidar-support-dilation", type=int, default=8)
    parser.add_argument(
        "--lidar-depth-loss-weight",
        type=float,
        default=0.1,
        help="Auxiliary log-depth L1 weight on latent-resolution LiDAR hit cells.",
    )
    parser.add_argument(
        "--lidar-depth-log-eps",
        type=float,
        default=1e-3,
        help="Epsilon used by normalized log-depth supervision.",
    )
    return parser.parse_args()


def init_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    distributed = world_size > 1
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside the {torch.cuda.device_count()} visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed:
        backend = args.dist_backend
        if backend == "auto":
            backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=max(1, int(getattr(args, "dist_timeout_seconds", 600)))),
        )
    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "device": device,
        "is_main": rank == 0,
    }


def distributed_barrier(distributed):
    if distributed and dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def validate_runtime_args(args, world_size):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.dataloader_timeout < 0:
        raise ValueError("--dataloader-timeout must be non-negative")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")
    if args.min_free_disk_gb < 0:
        raise ValueError("--min-free-disk-gb must be non-negative")
    if args.min_free_host_memory_gb < 0:
        raise ValueError("--min-free-host-memory-gb must be non-negative")
    cpu_count = os.cpu_count() or 1
    total_workers = int(args.num_workers) * int(world_size)
    if total_workers > cpu_count:
        raise ValueError(
            f"DataLoader would start {total_workers} workers across {world_size} ranks, "
            f"but only {cpu_count} logical CPUs are available"
        )
    if world_size > 1 and int(args.sample_every) > 0 and not args.allow_ddp_inline_sampling:
        raise ValueError(
            "Inline sampling is disabled by default for DDP because non-main ranks would wait at a "
            "collective during 50-step DDIM. Set --sample-every 0 and sample checkpoints offline, "
            "or explicitly pass --allow-ddp-inline-sampling."
        )


def require_host_memory(min_free_host_memory_gb, operation):
    available_mb = _mem_available_mb()
    threshold_mb = float(min_free_host_memory_gb) * 1024.0
    if available_mb >= 0.0 and available_mb < threshold_mb:
        raise RuntimeError(
            f"Refusing {operation} with only {available_mb / 1024.0:.1f} GiB host RAM available; "
            f"required at least {float(min_free_host_memory_gb):.1f} GiB"
        )


def instantiate_model_safely(
    cfg,
    device,
    distributed,
    rank,
    world_size,
    parallel_model_init,
    min_free_host_memory_gb,
):
    if not distributed or parallel_model_init:
        require_host_memory(min_free_host_memory_gb, "model initialization")
        return instantiate_from_config(cfg.model).to(device)
    model = None
    for owner_rank in range(world_size):
        if rank == owner_rank:
            require_host_memory(min_free_host_memory_gb, f"model initialization on rank {rank}")
            model = instantiate_from_config(cfg.model).to(device)
        distributed_barrier(distributed)
    return model


def load_training_checkpoint_safely(
    model,
    optimizer,
    ckpt_path,
    scaler,
    distributed,
    rank,
    world_size,
    parallel_model_init,
    min_free_host_memory_gb,
    expected_architecture=None,
):
    if not distributed or parallel_model_init:
        require_host_memory(min_free_host_memory_gb, "checkpoint resume")
        return load_training_checkpoint(
            model, optimizer, ckpt_path, scaler=scaler, expected_architecture=expected_architecture
        )
    start_step = 0
    for owner_rank in range(world_size):
        if rank == owner_rank:
            require_host_memory(min_free_host_memory_gb, f"checkpoint resume on rank {rank}")
            start_step = load_training_checkpoint(
                model, optimizer, ckpt_path, scaler=scaler, expected_architecture=expected_architecture
            )
        distributed_barrier(distributed)
    return start_step


def synchronized_loss_finite(loss, distributed):
    finite = torch.isfinite(loss.detach()).all()
    if not distributed:
        return bool(finite.item())
    finite_flag = finite.to(device=loss.device, dtype=torch.int32)
    dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
    return bool(finite_flag.item())


def move_batch_to_device(batch, device):
    return {
        key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


class TrainingStepModule(torch.nn.Module):
    """Expose the custom Lightning training_step through a normal DDP forward."""

    def __init__(self, model, token_structure_loss_weight):
        super().__init__()
        self.model = model
        self.token_structure_loss_weight = float(token_structure_loss_weight)

    def forward(self, batch, step):
        loss = self.model.training_step(batch, step)
        token_structure_loss = lidar_token_structure_loss_tensor(self.model)
        return loss + self.token_structure_loss_weight * token_structure_loss


def merge_distributed_records(records):
    if not records:
        return None
    merged = {}
    keys = set().union(*(record.keys() for record in records))
    for key in keys:
        values = [record[key] for record in records if key in record]
        first = values[0]
        if all(value == first for value in values):
            merged[key] = first
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            total = sum(float(value) for value in values)
            if key == "num_projected_lidar_points":
                merged[key] = total
            else:
                merged[key] = total / len(values)
        else:
            merged[key] = first
    merged["distributed_metrics_world_size"] = len(records)
    return merged


def gather_distributed_record(record, distributed, rank, world_size):
    if not distributed:
        return record
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(record, gathered, dst=0)
    return merge_distributed_records(gathered) if rank == 0 else None


def scalar(value):
    return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)


def lidar_support_mask_stats(batch, dilation, latent_size=(16, 64)):
    lidar_cond = batch.get("lidar_cond")
    if not torch.is_tensor(lidar_cond) or lidar_cond.shape[1] < 2:
        return {
            "lidar_support_image_mask_coverage": 0.0,
            "lidar_support_latent_mask_coverage": 0.0,
        }
    image_mask = (lidar_cond[:, 1:2].float() > 0.0).float()
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
    use_lidar_pixel = is_lidar_pixel_config(cfg)
    native_depth = uses_native_depth_head(cfg)
    if native_depth and not use_lidar_pixel:
        raise ValueError("The V2.2 native depth head requires the pixel LiDAR configuration")
    cfg.model.base_learning_rate = args.lr
    cfg.model.params.pre_sat2grd_model_path = None
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.use_lidar_cond = True
    cfg.model.params.lidar_geom_mode = LIDAR_GEOM_MODE

    # Global eps + LiDAR-hit eps + the selected depth head + DINO alignment.
    cfg.model.params.dynamic_point_loss_weight = float(args.lidar_support_loss_weight)
    cfg.model.params.dynamic_point_dilation = int(args.lidar_support_dilation)
    cfg.model.params.lidar_depth_loss_weight = float(args.lidar_depth_loss_weight)
    cfg.model.params.lidar_depth_resample_mode = "native" if native_depth else "masked_area"
    cfg.model.params.lidar_depth_output_scale = 1.0 if native_depth else 0.0
    cfg.model.params.lidar_depth_bottleneck_scale = 0.0 if native_depth else 1.0
    cfg.model.params.lidar_depth_log_eps = float(args.lidar_depth_log_eps)
    cfg.model.params.ray_evidence_mask_mode = RAY_EVIDENCE_MASK_MODE
    cfg.model.params.lidar_semantic_alignment_weight = float(args.lidar_semantic_alignment_weight)
    cfg.model.params.lidar_semantic_alignment_key = "image_semantic_feat"
    cfg.model.params.lidar_semantic_alignment_mask_mode = LIDAR_SEMANTIC_MASK_MODE

    unet = cfg.model.params.DDPM_config.params.unet_config.params
    unet.use_checkpoint = False
    cfg.model.params.DDPM_config.params.control_grd = None
    unet.ray_fusion_mode = "ray_posterior"
    unet.use_lidar_ray_posterior = True
    unet.lidar_posterior_log_depth_sigma = 0.35
    unet.lidar_posterior_strength = 2.0
    unet.lidar_message_gate_bias = -2.0
    if use_lidar_pixel:
        unet.use_lidar_cross_attention = False
        unet.lidar_spatial_channels = list(LIDAR_PIXEL_PYRAMID_CHANNELS)
        delete_config_key(unet, "lidar_context_dim")
        delete_config_key(unet, "lidar_reference_window")
        unet.ray_fusion_mode = "ray_posterior"
        unet.use_lidar_ray_posterior = True
        cfg.model.params.Lidar_context_config = {
            "target": LIDAR_PIXEL_CONTEXT_TARGET,
            "params": {
                "point_feature_dim": LIDAR_POINT_FEATURE_DIM,
                "hidden_channels": LIDAR_PIXEL_HIDDEN_CHANNELS,
                "pyramid_channels": list(LIDAR_PIXEL_PYRAMID_CHANNELS),
                "image_size": list(LIDAR_PIXEL_SIZE),
                "token_grid": list(LIDAR_TOKEN_GRID),
                "semantic_feature_dim": IMAGE_SEMANTIC_DIM,
                "evidence_dilation": int(args.lidar_evidence_dilation),
                "evidence_free_space_dilation": int(args.lidar_evidence_free_space_dilation),
            },
        }
    else:
        unet.use_lidar_cross_attention = True
        unet.lidar_context_dim = LIDAR_TOKEN_DIM
        unet.lidar_reference_window = int(args.lidar_reference_window)
        cfg.model.params.Lidar_context_config = {
            "target": "models.KITTI_geo_ldm.lidar_condition_model.LidarVisibleRaySemanticTokenEncoder",
            "params": {
                "point_in_channels": LIDAR_POINT_IN_CHANNELS,
                "front_in_channels": 5,
                "hidden_channels": LIDAR_TOKEN_HIDDEN_CHANNELS,
                "token_dim": LIDAR_TOKEN_DIM,
                "token_grid": list(LIDAR_TOKEN_GRID),
                "image_size": [
                    int(cfg.data.params.train.params.image_height),
                    int(cfg.data.params.train.params.image_width),
                ],
                "use_evidence_maps": True,
                "evidence_dilation": int(args.lidar_evidence_dilation),
                "evidence_free_space_dilation": int(args.lidar_evidence_free_space_dilation),
                "token_output_norm": LIDAR_TOKEN_OUTPUT_NORM,
                "token_structure_target_ratio": float(args.lidar_token_structure_target_ratio),
                "use_pointmap_pe": True,
                "point_pretrained_ckpt": "",
                "point_feature_dim": LIDAR_POINT_FEATURE_DIM,
                "semantic_feature_dim": IMAGE_SEMANTIC_DIM,
            },
        }

    cfg.data.params.batch_size = int(args.batch_size)
    cfg.data.params.num_workers = int(args.num_workers)
    for split, manifest in (("train", args.train_manifest), ("test", args.val_manifest)):
        params = getattr(cfg.data.params, split).params
        params.manifest = manifest
        params.kitti_root = args.kitti_root
        params.condition_mode = CONDITION_MODE
        # The visible-ray encoder consumes lidar_cond plus the offline Utonia cache and
        # explicitly discards range_img. Avoid building and transferring it per sample.
        params.include_range_image = False
        params.include_raw_lidar_points = False
        params.lidar_point_feature_cache_root = ""
        params.lidar_point_feature_dim = 0
        if use_lidar_pixel:
            for key in (
                "lidar_ray_feature_cache_root",
                "lidar_ray_feature_cache_suffix",
                "lidar_ray_feature_dim",
                "lidar_ray_depth_bins",
                "lidar_ray_height",
                "lidar_ray_width",
            ):
                delete_config_key(params, key)
        else:
            params.lidar_ray_feature_cache_root = args.lidar_ray_feature_cache_root
            params.lidar_ray_feature_cache_suffix = ".npz"
            params.lidar_ray_feature_dim = LIDAR_POINT_FEATURE_DIM
            params.lidar_ray_depth_bins = LIDAR_RAY_CACHE_PLANES
            params.lidar_ray_height = LIDAR_TOKEN_GRID[0]
            params.lidar_ray_width = LIDAR_TOKEN_GRID[1]
        params.image_semantic_cache_root = args.image_semantic_cache_root
        params.image_semantic_cache_suffix = ".npz"
        params.image_semantic_feature_key = "dino_feat"
        params.image_semantic_feature_dim = IMAGE_SEMANTIC_DIM
        params.image_semantic_height = IMAGE_SEMANTIC_SIZE[0]
        params.image_semantic_width = IMAGE_SEMANTIC_SIZE[1]
        params.include_tracklets = False
        if use_lidar_pixel:
            params.lidar_pixel_feature_cache_root = getattr(args, "lidar_pixel_feature_cache_root", "")
            params.lidar_pixel_feature_cache_suffix = ".npz"
            params.lidar_pixel_feature_dim = LIDAR_POINT_FEATURE_DIM
    return cfg


def load_training_checkpoint(model, optimizer, ckpt_path, scaler=None, expected_architecture=None):
    payload = torch.load(ckpt_path, map_location="cpu")
    required = {
        "denoise_model",
        "condition_model_sat",
        "lidar_context_model",
        "optimizer",
        "grad_scaler",
        "step",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(
            "Checkpoint is not a current ray-posterior training checkpoint; "
            f"missing keys: {missing}"
        )
    if expected_architecture is not None:
        metadata = payload.get("metadata", {})
        actual_architecture = metadata.get("architecture") if isinstance(metadata, dict) else None
        if actual_architecture != expected_architecture:
            raise RuntimeError(
                "Checkpoint architecture mismatch for strict resume: "
                f"expected {expected_architecture!r}, got {actual_architecture!r}"
            )
    model.DDPM.denoise_model.load_state_dict(payload["denoise_model"], strict=True)
    model.condition_model_sat.load_state_dict(payload["condition_model_sat"], strict=True)
    if model.lidar_context_model is None:
        raise RuntimeError("Current ray-posterior model is missing lidar_context_model")
    model.lidar_context_model.load_state_dict(payload["lidar_context_model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(payload["grad_scaler"])
    step = int(payload.get("step", 0))
    del payload
    gc.collect()
    return step


def free_disk_gb(path):
    path = Path(path)
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024 ** 3)


def require_checkpoint_disk_space(path, min_free_disk_gb):
    free_gb = free_disk_gb(path)
    if free_gb < float(min_free_disk_gb):
        raise RuntimeError(
            f"Refusing checkpoint write with only {free_gb:.1f} GiB free at {Path(path).parent}; "
            f"required at least {float(min_free_disk_gb):.1f} GiB"
        )
    return free_gb


def atomic_torch_save(payload, path):
    path = Path(path)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp_path.unlink(missing_ok=True)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def update_checkpoint_alias(source, alias):
    source = Path(source)
    alias = Path(alias)
    temp_alias = alias.with_name(f".{alias.name}.tmp-{os.getpid()}")
    temp_alias.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temp_alias)
            mode = "hardlink"
        except OSError:
            shutil.copy2(source, temp_alias)
            mode = "copy"
        os.replace(temp_alias, alias)
    finally:
        temp_alias.unlink(missing_ok=True)
    return mode


def save_checkpoint(path, model, optimizer, scaler, step, args, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    free_gb_before = require_checkpoint_disk_space(path, args.min_free_disk_gb)
    payload = {
        "step": int(step),
        "args": vars(args),
        "metadata": metadata,
    }
    if model.lidar_context_model is None:
        raise RuntimeError("Current ray-posterior model is missing lidar_context_model")
    payload.update(
        {
            "denoise_model": model.DDPM.denoise_model.state_dict(),
            "condition_model_sat": model.condition_model_sat.state_dict(),
            "lidar_context_model": model.lidar_context_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict(),
        }
    )
    atomic_torch_save(payload, path)
    return {
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_free_disk_gb_before": round(free_gb_before, 2),
    }


def save_step_checkpoint(ckpt_dir, model, optimizer, scaler, step, args, metadata):
    step_path = ckpt_dir / f"step_{step:06d}.pt"
    save_stats = save_checkpoint(step_path, model, optimizer, scaler, step, args, metadata)
    alias_mode = update_checkpoint_alias(step_path, ckpt_dir / "last.pt")
    return step_path, {**save_stats, "checkpoint_alias_mode": alias_mode}


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


def ensure_fresh_run_directory(out_dir, resume_ckpt):
    if resume_ckpt or not out_dir.exists():
        return
    if any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to start a fresh run in non-empty directory: {out_dir}. "
            "Choose a new --run-name or pass --resume-ckpt."
        )


def ensure_fresh_run_directory_distributed(out_dir, resume_ckpt, distributed, is_main):
    error_message = None
    if is_main:
        try:
            ensure_fresh_run_directory(out_dir, resume_ckpt)
        except FileExistsError as error:
            error_message = str(error)
    if distributed:
        error_payload = [error_message]
        dist.broadcast_object_list(error_payload, src=0)
        error_message = error_payload[0]
    if error_message:
        raise FileExistsError(error_message)


def ensure_output_disk_space_distributed(path, min_free_disk_gb, distributed, is_main):
    error_message = None
    if is_main:
        try:
            require_checkpoint_disk_space(path, min_free_disk_gb)
        except RuntimeError as error:
            error_message = str(error)
    if distributed:
        error_payload = [error_message]
        dist.broadcast_object_list(error_payload, src=0)
        error_message = error_payload[0]
    if error_message:
        raise RuntimeError(error_message)


def build_inline_sample_dataset(args, use_lidar_pixel=False):
    if int(args.sample_every) <= 0 or not args.sample_manifest:
        return None
    lidar_ray_feature_cache_root = "" if use_lidar_pixel else args.lidar_ray_feature_cache_root
    return SatLidarRawDataset(
        manifest=args.sample_manifest,
        condition_mode=CONDITION_MODE,
        image_height=128,
        image_width=512,
        sat_size=256,
        max_depth=80.0,
        align_satellite_to_camera=True,
        include_range_image=False,
        include_raw_lidar_points=False,
        lidar_ray_feature_cache_root=lidar_ray_feature_cache_root,
        lidar_ray_feature_cache_suffix=".npz",
        lidar_ray_feature_dim=LIDAR_POINT_FEATURE_DIM,
        lidar_ray_depth_bins=LIDAR_RAY_CACHE_PLANES,
        lidar_ray_height=LIDAR_TOKEN_GRID[0],
        lidar_ray_width=LIDAR_TOKEN_GRID[1],
        lidar_pixel_feature_cache_root=getattr(args, "lidar_pixel_feature_cache_root", ""),
        lidar_pixel_feature_cache_suffix=".npz",
        lidar_pixel_feature_dim=LIDAR_POINT_FEATURE_DIM,
        image_semantic_cache_root=args.image_semantic_cache_root,
        image_semantic_cache_suffix=".npz",
        image_semantic_feature_key="dino_feat",
        image_semantic_feature_dim=IMAGE_SEMANTIC_DIM,
        image_semantic_height=IMAGE_SEMANTIC_SIZE[0],
        image_semantic_width=IMAGE_SEMANTIC_SIZE[1],
        include_tracklets=False,
        kitti_root=args.kitti_root,
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
            + ". Supported probes: normal,zero."
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
            sat_path = sample_out / "images" / "satellite" / f"{safe_id}.png"
            overlay_path = sample_out / "images" / "lidar_overlay" / f"{safe_id}.png"
            cond_path = sample_out / "images" / "lidar_cond" / f"{safe_id}.png"
            target = sample["grd_left_imgs"].unsqueeze(0).clamp(0.0, 1.0)
            save_tensor_image(target[0], gt_path)
            save_tensor_image(sample["sat_map"], sat_path)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            make_lidar_overlay(target[0], sample["lidar_cond"]).save(overlay_path)
            save_tensor_image(make_condition_rgb(sample["lidar_cond"]), cond_path)

            image_paths = {
                "Satellite input": sat_path,
                "LiDAR projection on RGB": overlay_path,
                "GT": gt_path,
            }
            attention_by_probe = {}
            batch = sample_to_batch(sample)
            try:
                for probe in probes:
                    sample_seed = int(args.sample_seed) + idx
                    if not getattr(args, "sample_fixed_seed", False):
                        sample_seed += int(step)
                    # Monitoring must not change subsequent training noise, and
                    # AMP sampling must fit alongside the live optimizer state.
                    rng_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                    with torch.random.fork_rng(devices=rng_devices), autocast(
                        enabled=bool(args.amp and torch.cuda.is_available())
                    ):
                        pred, _, attention_stats, key_structure_stats = generate_prediction(
                            model,
                            batch,
                            probe=probe,
                            ddim_steps=int(args.sample_ddim_steps),
                            seed=sample_seed,
                            guidance_scale=7.5,
                            eta=float(getattr(args, "sample_eta", 1.0)),
                            temperature=1.0,
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


def ray_fusion_parameter_stats(model):
    values = {"to_q": [], "to_k": [], "ray_proj": []}
    posterior_gate = []
    for module in model.DDPM.denoise_model.modules():
        if module.__class__.__name__ == "RayAlignedEvidenceAttention":
            values["to_q"].append(module.to_q.weight.detach().float().norm().cpu())
            values["to_k"].append(module.to_k.weight.detach().float().norm().cpu())
            values["ray_proj"].append(module.ray_proj.weight.detach().float().norm().cpu())
        elif module.__class__.__name__ == "RayPosteriorEvidenceFusion":
            posterior_gate.append(module.lidar_gate.weight.detach().float().norm().cpu())
    stats = {
        f"ray_evidence_{name}_weight_norm_mean": float(torch.stack(items).mean()) if items else 0.0
        for name, items in values.items()
    }
    stats["ray_posterior_gate_weight_norm_mean"] = (
        float(torch.stack(posterior_gate).mean()) if posterior_gate else 0.0
    )
    return stats


def ray_fusion_grad_stats(model):
    grads = {"to_q": [], "to_k": [], "ray_proj": []}
    posterior_gate = []
    for name, param in model.DDPM.denoise_model.named_parameters():
        for key in grads:
            if f".ray_evidence_attn.{key}.weight" in name and param.grad is not None:
                grads[key].append(param.grad.detach().float().abs().mean().cpu())
        if ".ray_posterior_fusion.lidar_gate.weight" in name and param.grad is not None:
            posterior_gate.append(param.grad.detach().float().abs().mean().cpu())
    stats = {
        f"ray_evidence_{name}_grad_abs_mean": float(torch.stack(items).mean()) if items else 0.0
        for name, items in grads.items()
    }
    stats["ray_posterior_gate_grad_abs_mean"] = (
        float(torch.stack(posterior_gate).mean()) if posterior_gate else 0.0
    )
    return stats


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
    evidence_sat = []
    evidence_lidar = []
    evidence_lidar_std = []
    evidence_lidar_min = []
    evidence_lidar_max = []
    evidence_null = []
    evidence_entropy = []
    evidence_lidar_mask = []
    evidence_lidar_masked = []
    evidence_lidar_background = []
    posterior_hit = []
    posterior_prior_entropy = []
    posterior_entropy = []
    posterior_weight_shift = []
    posterior_depth_error = []
    posterior_lidar_confidence = []
    posterior_lidar_mask = []
    posterior_lidar_message_ratio = []
    for module in model.DDPM.denoise_model.modules():
        if hasattr(module, "last_attn_entropy_norm"):
            entropy.append(torch.tensor(module_stat_float(module, "last_attn_entropy_norm"), dtype=torch.float32))
            max_mean.append(torch.tensor(module_stat_float(module, "last_attn_max_mean"), dtype=torch.float32))
            std_mean.append(torch.tensor(module_stat_float(module, "last_attn_std_mean"), dtype=torch.float32))
            sim_std.append(torch.tensor(module_stat_float(module, "last_sim_std_mean"), dtype=torch.float32))
            sim_range.append(torch.tensor(module_stat_float(module, "last_sim_range_mean"), dtype=torch.float32))
            token_count.append(torch.tensor(module_stat_float(module, "last_attn_token_count"), dtype=torch.float32))
            query_count.append(torch.tensor(module_stat_float(module, "last_attn_query_count"), dtype=torch.float32))
        if hasattr(module, "last_evidence_sat_weight"):
            evidence_sat.append(torch.tensor(module_stat_float(module, "last_evidence_sat_weight"), dtype=torch.float32))
            evidence_lidar.append(torch.tensor(module_stat_float(module, "last_evidence_lidar_weight"), dtype=torch.float32))
            evidence_lidar_std.append(torch.tensor(module_stat_float(module, "last_evidence_lidar_weight_std"), dtype=torch.float32))
            evidence_lidar_min.append(torch.tensor(module_stat_float(module, "last_evidence_lidar_weight_min"), dtype=torch.float32))
            evidence_lidar_max.append(torch.tensor(module_stat_float(module, "last_evidence_lidar_weight_max"), dtype=torch.float32))
            evidence_null.append(torch.tensor(module_stat_float(module, "last_evidence_null_weight"), dtype=torch.float32))
            evidence_entropy.append(torch.tensor(module_stat_float(module, "last_evidence_entropy_norm"), dtype=torch.float32))
            if hasattr(module, "last_evidence_lidar_mask_mean"):
                evidence_lidar_mask.append(torch.tensor(module_stat_float(module, "last_evidence_lidar_mask_mean"), dtype=torch.float32))
                evidence_lidar_masked.append(torch.tensor(module_stat_float(module, "last_evidence_lidar_weight_masked"), dtype=torch.float32))
                evidence_lidar_background.append(torch.tensor(module_stat_float(module, "last_evidence_lidar_weight_background"), dtype=torch.float32))
        if bool(getattr(module, "use_lidar_ray_posterior", False)) and hasattr(
            module, "last_ray_posterior_hit_coverage"
        ):
            posterior_hit.append(torch.tensor(module_stat_float(module, "last_ray_posterior_hit_coverage"), dtype=torch.float32))
            posterior_prior_entropy.append(torch.tensor(module_stat_float(module, "last_ray_posterior_prior_entropy"), dtype=torch.float32))
            posterior_entropy.append(torch.tensor(module_stat_float(module, "last_ray_posterior_entropy"), dtype=torch.float32))
            posterior_weight_shift.append(torch.tensor(module_stat_float(module, "last_ray_posterior_weight_shift"), dtype=torch.float32))
            posterior_depth_error.append(torch.tensor(module_stat_float(module, "last_ray_posterior_depth_error"), dtype=torch.float32))
        if module.__class__.__name__ == "RayPosteriorEvidenceFusion":
            posterior_lidar_confidence.append(torch.tensor(module_stat_float(module, "last_lidar_confidence"), dtype=torch.float32))
            posterior_lidar_mask.append(torch.tensor(module_stat_float(module, "last_lidar_mask_mean"), dtype=torch.float32))
            posterior_lidar_message_ratio.append(torch.tensor(module_stat_float(module, "last_lidar_message_ratio"), dtype=torch.float32))

    spatial_modules = [
        module for module in model.DDPM.denoise_model.modules()
        if module.__class__.__name__ == "LidarSpatialResidual"
    ]
    spatial_stats = {"lidar_spatial_modules": len(spatial_modules)}
    for metric, attr in (
        ("lidar_spatial_confidence_mean", "last_lidar_confidence"),
        ("lidar_spatial_support_mean", "last_lidar_mask_mean"),
        ("lidar_spatial_message_ratio_mean", "last_lidar_message_ratio"),
    ):
        values = [float(getattr(module, attr).detach().float().cpu()) for module in spatial_modules]
        spatial_stats[metric] = sum(values) / len(values) if values else 0.0

    posterior_stats = {
        **spatial_stats,
        "ray_posterior_modules": len(posterior_hit),
        "ray_posterior_hit_coverage_mean": float(torch.stack(posterior_hit).mean()) if posterior_hit else 0.0,
        "ray_posterior_prior_entropy_mean": float(torch.stack(posterior_prior_entropy).mean()) if posterior_prior_entropy else 0.0,
        "ray_posterior_entropy_mean": float(torch.stack(posterior_entropy).mean()) if posterior_entropy else 0.0,
        "ray_posterior_weight_shift_mean": float(torch.stack(posterior_weight_shift).mean()) if posterior_weight_shift else 0.0,
        "ray_posterior_depth_log_error_mean": float(torch.stack(posterior_depth_error).mean()) if posterior_depth_error else 0.0,
        "ray_posterior_lidar_confidence_mean": float(torch.stack(posterior_lidar_confidence).mean()) if posterior_lidar_confidence else 0.0,
        "ray_posterior_lidar_mask_mean": float(torch.stack(posterior_lidar_mask).mean()) if posterior_lidar_mask else 0.0,
        "ray_posterior_lidar_message_ratio_mean": float(torch.stack(posterior_lidar_message_ratio).mean()) if posterior_lidar_message_ratio else 0.0,
    }
    if not entropy:
        stats = {
            "lidar_attn_modules": 0,
            "lidar_sim_std_mean": 0.0,
            "lidar_sim_range_mean": 0.0,
            "lidar_attn_entropy_norm_mean": 0.0,
            "lidar_attn_max_mean": 0.0,
            "lidar_attn_std_mean": 0.0,
            "lidar_attn_token_count_mean": 0.0,
            "lidar_attn_query_count_mean": 0.0,
            "ray_evidence_modules": len(evidence_sat),
            "ray_evidence_sat_weight_mean": float(torch.stack(evidence_sat).mean()) if evidence_sat else 0.0,
            "ray_evidence_lidar_weight_mean": float(torch.stack(evidence_lidar).mean()) if evidence_lidar else 0.0,
            "ray_evidence_lidar_weight_std_mean": float(torch.stack(evidence_lidar_std).mean()) if evidence_lidar_std else 0.0,
            "ray_evidence_lidar_weight_min_mean": float(torch.stack(evidence_lidar_min).mean()) if evidence_lidar_min else 0.0,
            "ray_evidence_lidar_weight_max_mean": float(torch.stack(evidence_lidar_max).mean()) if evidence_lidar_max else 0.0,
            "ray_evidence_null_weight_mean": float(torch.stack(evidence_null).mean()) if evidence_null else 0.0,
            "ray_evidence_entropy_norm_mean": float(torch.stack(evidence_entropy).mean()) if evidence_entropy else 0.0,
            "ray_evidence_lidar_mask_mean": float(torch.stack(evidence_lidar_mask).mean()) if evidence_lidar_mask else 0.0,
            "ray_evidence_lidar_weight_masked_mean": float(torch.stack(evidence_lidar_masked).mean()) if evidence_lidar_masked else 0.0,
            "ray_evidence_lidar_weight_background_mean": float(torch.stack(evidence_lidar_background).mean()) if evidence_lidar_background else 0.0,
        }
        stats.update(posterior_stats)
        return stats
    stats = {
        "lidar_attn_modules": len(entropy),
        "lidar_sim_std_mean": float(torch.stack(sim_std).mean()),
        "lidar_sim_range_mean": float(torch.stack(sim_range).mean()),
        "lidar_attn_entropy_norm_mean": float(torch.stack(entropy).mean()),
        "lidar_attn_max_mean": float(torch.stack(max_mean).mean()),
        "lidar_attn_std_mean": float(torch.stack(std_mean).mean()),
        "lidar_attn_token_count_mean": float(torch.stack(token_count).mean()),
        "lidar_attn_query_count_mean": float(torch.stack(query_count).mean()),
    }
    stats.update(
        {
            "ray_evidence_modules": len(evidence_sat),
            "ray_evidence_sat_weight_mean": float(torch.stack(evidence_sat).mean()) if evidence_sat else 0.0,
            "ray_evidence_lidar_weight_mean": float(torch.stack(evidence_lidar).mean()) if evidence_lidar else 0.0,
            "ray_evidence_lidar_weight_std_mean": float(torch.stack(evidence_lidar_std).mean()) if evidence_lidar_std else 0.0,
            "ray_evidence_lidar_weight_min_mean": float(torch.stack(evidence_lidar_min).mean()) if evidence_lidar_min else 0.0,
            "ray_evidence_lidar_weight_max_mean": float(torch.stack(evidence_lidar_max).mean()) if evidence_lidar_max else 0.0,
            "ray_evidence_null_weight_mean": float(torch.stack(evidence_null).mean()) if evidence_null else 0.0,
            "ray_evidence_entropy_norm_mean": float(torch.stack(evidence_entropy).mean()) if evidence_entropy else 0.0,
            "ray_evidence_lidar_mask_mean": float(torch.stack(evidence_lidar_mask).mean()) if evidence_lidar_mask else 0.0,
            "ray_evidence_lidar_weight_masked_mean": float(torch.stack(evidence_lidar_masked).mean()) if evidence_lidar_masked else 0.0,
            "ray_evidence_lidar_weight_background_mean": float(torch.stack(evidence_lidar_background).mean()) if evidence_lidar_background else 0.0,
        }
    )
    stats.update(posterior_stats)
    return stats


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
            "lidar_point_valid_ratio": 0.0,
            "lidar_point_feature_valid_ratio": 0.0,
            "lidar_point_count_mean": 0.0,
            "lidar_ray_token_coverage": 0.0,
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
        "lidar_point_valid_ratio": scalar(getattr(context_model, "last_point_valid_ratio", torch.tensor(0.0))),
        "lidar_point_feature_valid_ratio": scalar(
            getattr(context_model, "last_point_feature_valid_ratio", torch.tensor(0.0))
        ),
        "lidar_point_count_mean": scalar(getattr(context_model, "last_point_count_mean", torch.tensor(0.0))),
        "lidar_ray_token_coverage": scalar(getattr(context_model, "last_ray_token_coverage", torch.tensor(0.0))),
    }


def lidar_token_structure_loss_tensor(model):
    context_model = getattr(model, "lidar_context_model", None)
    if context_model is not None:
        loss = getattr(context_model, "last_token_structure_loss", None)
        if torch.is_tensor(loss):
            return loss
    return torch.zeros((), device=next(model.parameters()).device)


def build_loader(dataset, args, rank=0, world_size=1):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=bool(args.shuffle),
            seed=int(args.seed),
            drop_last=False,
        )
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": bool(args.shuffle) if sampler is None else False,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "drop_last": False,
        "pin_memory": bool(args.pin_memory),
        "timeout": int(getattr(args, "dataloader_timeout", 0)) if args.num_workers > 0 else 0,
    }
    if args.num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs), sampler


def next_batch(loader, iterator, sampler=None, data_epoch=0):
    try:
        return next(iterator), iterator, data_epoch
    except StopIteration:
        data_epoch += 1
        if sampler is not None:
            sampler.set_epoch(data_epoch)
        iterator = iter(loader)
        return next(iterator), iterator, data_epoch


def retryable_backward_shape_error(error):
    message = str(error)
    return "returned an invalid gradient" in message and "expected shape compatible" in message


def batch_sample_ids(batch):
    sample_ids = batch.get("sample_id", [])
    if isinstance(sample_ids, str):
        return [sample_ids]
    if isinstance(sample_ids, (list, tuple)):
        return [str(sample_id) for sample_id in sample_ids]
    return [str(sample_ids)]


def validate_memmap_cache(root, kind, feature_shape, mask_shape, manifests):
    root = Path(root)
    meta_path = root / "memmap_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Required {kind} memmap metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if meta.get("format") != "kitti_feature_memmap_v1" or meta.get("kind") != kind:
        raise ValueError(f"Invalid {kind} memmap metadata: {meta_path}")
    if tuple(meta.get("feature_shape", ())) != tuple(feature_shape):
        raise ValueError(
            f"{kind} feature shape {tuple(meta.get('feature_shape', ()))} != expected {tuple(feature_shape)}"
        )
    if tuple(meta.get("mask_shape", ())) != tuple(mask_shape):
        raise ValueError(
            f"{kind} mask shape {tuple(meta.get('mask_shape', ()))} != expected {tuple(mask_shape)}"
        )
    for key in ("features_file", "masks_file"):
        cache_file = root / str(meta.get(key, ""))
        if not cache_file.is_file():
            raise FileNotFoundError(f"Required {kind} memmap file not found: {cache_file}")

    index = meta.get("index", {})
    required_ids = set()
    for manifest in manifests:
        if not manifest:
            continue
        for line in Path(manifest).read_text().splitlines():
            if line.strip():
                required_ids.add(safe_sample_id(json.loads(line)["sample_id"]))
    missing = sorted(required_ids.difference(index))
    if missing:
        examples = ", ".join(missing[:5])
        raise RuntimeError(
            f"{kind} memmap cache misses {len(missing)}/{len(required_ids)} required samples; "
            f"examples: {examples}"
        )
    return {
        f"{kind}_cache_rows": len(index),
        f"{kind}_cache_required_rows": len(required_ids),
        f"{kind}_cache_root": str(root),
    }


def validate_pixel_ragged_cache(root, feature_dim, image_size, manifests):
    from dataloader.kitti_pixel_feature_cache import preflight_pixel_ragged_cache

    return preflight_pixel_ragged_cache(root, feature_dim, image_size, manifests)


def validate_pixel_cache_distributed(root, feature_dim, image_size, manifests, distributed, is_main):
    if not distributed:
        return validate_pixel_ragged_cache(root, feature_dim, image_size, manifests)
    payload = [None]
    if is_main:
        try:
            payload[0] = {
                "ok": True,
                "metadata": validate_pixel_ragged_cache(root, feature_dim, image_size, manifests),
            }
        except Exception as exc:
            payload[0] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not result.get("ok", False):
        raise RuntimeError(
            f"Rank-0 pixel cache preflight failed with {result.get('error_type', 'Exception')}: "
            f"{result.get('error', '')}"
        )
    return result["metadata"]


def new_metric_window():
    return {
        "steps": 0,
        "optimizer_skipped_steps": 0,
        "lidar_support_image_mask_coverage": 0.0,
        "lidar_support_latent_mask_coverage": 0.0,
        "lidar_depth_log_l1": 0.0,
        "lidar_depth_log_l1_contrib": 0.0,
        "lidar_depth_mask_coverage": 0.0,
        "lidar_bottleneck_depth_log_l1": 0.0,
        "lidar_bottleneck_depth_log_l1_contrib": 0.0,
        "lidar_bottleneck_depth_mask_coverage": 0.0,
        "lidar_semantic_alignment_loss": 0.0,
        "lidar_semantic_alignment_contrib": 0.0,
        "lidar_semantic_alignment_mask_coverage": 0.0,
        "lidar_semantic_alignment_target_available": 0.0,
        "lidar_semantic_alignment_applied_steps": 0,
    }


def run_training(args, dist_info):
    distributed = dist_info["distributed"]
    world_size = dist_info["world_size"]
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    device = dist_info["device"]
    is_main = dist_info["is_main"]
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = configure_cfg(OmegaConf.load(args.config), args)
    use_lidar_pixel = is_lidar_pixel_config(cfg)
    architecture = (
        "satellite_lidar_pixel_v22_ray_posterior"
        if uses_native_depth_head(cfg)
        else
        "satellite_lidar_pixel_v21_ray_posterior"
        if use_lidar_pixel
        else "satellite_lidar_zbuffer_visible_ray_posterior"
    )
    cache_manifests = [args.train_manifest]
    if int(args.sample_every) > 0 and args.sample_manifest:
        cache_manifests.append(args.sample_manifest)
    cache_metadata = {}
    if use_lidar_pixel:
        cache_metadata.update(
            validate_pixel_cache_distributed(
                args.lidar_pixel_feature_cache_root,
                LIDAR_POINT_FEATURE_DIM,
                LIDAR_PIXEL_SIZE,
                cache_manifests,
                distributed,
                is_main,
            )
        )
    else:
        cache_metadata.update(
            validate_memmap_cache(
                args.lidar_ray_feature_cache_root,
                "ray",
                (LIDAR_POINT_FEATURE_DIM, LIDAR_RAY_CACHE_PLANES, *LIDAR_TOKEN_GRID),
                (1, LIDAR_RAY_CACHE_PLANES, *LIDAR_TOKEN_GRID),
                cache_manifests,
            )
        )
    cache_metadata.update(
        validate_memmap_cache(
            args.image_semantic_cache_root,
            "image",
            (IMAGE_SEMANTIC_DIM, *IMAGE_SEMANTIC_SIZE),
            (1, *IMAGE_SEMANTIC_SIZE),
            cache_manifests,
        )
    )

    default_run_suffix = "pixel_lidar_v22" if uses_native_depth_head(cfg) else ("pixel_lidar_v21" if use_lidar_pixel else "ray_posterior")
    run_name = args.run_name or f"kitti_{default_run_suffix}_sd14_fresh_{args.steps}step"
    out_dir = Path(args.out_root) / run_name
    metrics_dir = out_dir / "metrics"
    ckpt_dir = out_dir / "checkpoints"
    ensure_fresh_run_directory_distributed(
        out_dir,
        args.resume_ckpt,
        distributed=distributed,
        is_main=is_main,
    )
    if is_main:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg))
        (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    ensure_output_disk_space_distributed(
        ckpt_dir,
        args.min_free_disk_gb,
        distributed,
        is_main,
    )
    distributed_barrier(distributed)

    train_dataset = instantiate_from_config(cfg.data.params.train)
    loader, distributed_sampler = build_loader(train_dataset, args, rank=rank, world_size=world_size)
    inline_sample_dataset = build_inline_sample_dataset(args, use_lidar_pixel=use_lidar_pixel) if is_main else None

    model = instantiate_model_safely(
        cfg,
        device,
        distributed,
        rank,
        world_size,
        args.parallel_model_init,
        args.min_free_host_memory_gb,
    )
    model.learning_rate = args.lr
    optimizer = model.configure_optimizers()[0]
    scaler = GradScaler(enabled=bool(args.amp and torch.cuda.is_available()))
    start_step = 0
    if args.resume_ckpt:
        start_step = load_training_checkpoint_safely(
            model,
            optimizer,
            args.resume_ckpt,
            scaler,
            distributed,
            rank,
            world_size,
            args.parallel_model_init,
            args.min_free_host_memory_gb,
            expected_architecture=architecture,
        )
        if is_main:
            print(json.dumps({"resumed": args.resume_ckpt, "start_step": start_step}, sort_keys=True))
    model.train()
    token_structure_loss_weight = 0.0 if use_lidar_pixel else float(args.lidar_token_structure_loss_weight)
    training_model = TrainingStepModule(model, token_structure_loss_weight).to(device)
    if distributed:
        ddp_kwargs = {
            "find_unused_parameters": bool(args.ddp_find_unused_parameters),
        }
        if device.type == "cuda":
            ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
        training_model = DistributedDataParallel(training_model, **ddp_kwargs)

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    data_epoch = start_step // max(len(loader), 1)
    if distributed_sampler is not None:
        distributed_sampler.set_epoch(data_epoch)
    iterator = iter(loader)

    metadata = {
        "architecture": architecture,
        "backbone": LIDAR_PIXEL_CONTEXT_BACKBONE if use_lidar_pixel else LIDAR_CONTEXT_BACKBONE,
        "sd_base_ckpt": args.sd_base_ckpt,
        "resume_ckpt": args.resume_ckpt,
        "start_step": int(start_step),
        "strict_same_version_resume": True,
        "keep_step_checkpoints": int(args.keep_step_checkpoints),
        "sample_every": int(args.sample_every),
        "sample_manifest": args.sample_manifest,
        "sample_num_samples": int(args.sample_num_samples),
        "sample_ddim_steps": int(args.sample_ddim_steps),
        "sample_probes": args.sample_probes,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "kitti_root": args.kitti_root,
        "train_size": len(train_dataset),
        "steps_per_epoch_batch1": len(train_dataset),
        "distributed": bool(distributed),
        "world_size": int(world_size),
        "local_batch_size": int(args.batch_size),
        "global_batch_size": int(args.batch_size) * int(world_size),
        "steps_per_epoch": len(loader),
        "ddp_find_unused_parameters": bool(args.ddp_find_unused_parameters),
        "dist_timeout_seconds": int(args.dist_timeout_seconds),
        "dataloader_timeout": int(args.dataloader_timeout),
        "parallel_model_init": bool(args.parallel_model_init),
        "min_free_disk_gb": float(args.min_free_disk_gb),
        "min_free_host_memory_gb": float(args.min_free_host_memory_gb),
        "lidar_attention_mode": "spatial_pyramid" if use_lidar_pixel else "local_reference",
        "lidar_fusion_mode": "ray_posterior",
        "lidar_geom_mode": LIDAR_GEOM_MODE,
        "lidar_context_backbone": LIDAR_PIXEL_CONTEXT_BACKBONE if use_lidar_pixel else LIDAR_CONTEXT_BACKBONE,
        "lidar_raw_point_count": "all_front_points_for_geometry",
        "lidar_point_in_channels": LIDAR_POINT_IN_CHANNELS,
        "lidar_point_feature_dim": LIDAR_POINT_FEATURE_DIM,
        "image_semantic_feature_dim": IMAGE_SEMANTIC_DIM,
        "image_semantic_size": list(IMAGE_SEMANTIC_SIZE),
        "ray_evidence_mask_mode": RAY_EVIDENCE_MASK_MODE,
        "use_lidar_control_residual": False,
        "use_lidar_cross_attention": bool(cfg.model.params.DDPM_config.params.unet_config.params.use_lidar_cross_attention),
        "optimizer_scope": "full_denoise_satellite_lidar",
        "base_lr": float(args.lr),
        "lidar_token_output_norm": None if use_lidar_pixel else LIDAR_TOKEN_OUTPUT_NORM,
        "lidar_token_structure_loss_weight": 0.0
        if use_lidar_pixel
        else float(args.lidar_token_structure_loss_weight),
        "lidar_token_structure_target_ratio": float(args.lidar_token_structure_target_ratio),
        "lidar_reference_window": None if use_lidar_pixel else int(args.lidar_reference_window),
        "lidar_support_loss_weight": float(args.lidar_support_loss_weight),
        "lidar_support_dilation": int(args.lidar_support_dilation),
        "lidar_depth_loss_weight": float(args.lidar_depth_loss_weight),
        "lidar_depth_resample_mode": cfg.model.params.lidar_depth_resample_mode,
        "lidar_depth_output_scale": float(cfg.model.params.lidar_depth_output_scale),
        "lidar_depth_bottleneck_scale": float(cfg.model.params.lidar_depth_bottleneck_scale),
        "lidar_depth_head_mode": getattr(model.DDPM.denoise_model, "lidar_depth_head_mode", "latent"),
        "lidar_depth_log_eps": float(args.lidar_depth_log_eps),
        "lidar_ray_feature_cache_root": "" if use_lidar_pixel else args.lidar_ray_feature_cache_root,
        "lidar_ray_cache_planes": 0 if use_lidar_pixel else LIDAR_RAY_CACHE_PLANES,
        "lidar_pixel_feature_cache_root": getattr(args, "lidar_pixel_feature_cache_root", ""),
        "lidar_pixel_size": list(LIDAR_PIXEL_SIZE),
        "lidar_spatial_channels": list(LIDAR_PIXEL_PYRAMID_CHANNELS) if use_lidar_pixel else [],
        "image_semantic_cache_root": args.image_semantic_cache_root,
        "lidar_semantic_alignment_weight": float(args.lidar_semantic_alignment_weight),
        "lidar_semantic_alignment_mask_mode": LIDAR_SEMANTIC_MASK_MODE,
        "trainable_lidar_context_params": count_trainable(model.lidar_context_model) if getattr(model, "lidar_context_model", None) is not None else 0,
        "trainable_sat_condition_params": count_trainable(model.condition_model_sat),
        "trainable_denoise_params": count_trainable(model.DDPM.denoise_model),
        **cache_metadata,
    }
    if is_main:
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
        print(json.dumps({"start": True, **metadata, **resource_metrics()}, sort_keys=True))

    metrics_path = metrics_dir / "train_metrics.jsonl"
    mode = "a" if args.resume_ckpt and start_step > 0 else "w"
    metrics_context = metrics_path.open(mode) if is_main else nullcontext(None)
    with metrics_context as metrics_file:
        window = new_metric_window()
        last_saved_step = None
        for step in range(start_step + 1, args.steps + 1):
            batch_cpu, iterator, data_epoch = next_batch(
                loader,
                iterator,
                sampler=distributed_sampler,
                data_epoch=data_epoch,
            )
            batch = move_batch_to_device(batch_cpu, device)
            for backward_attempt in range(2):
                optimizer.zero_grad(set_to_none=True)
                try:
                    with autocast(enabled=bool(args.amp and torch.cuda.is_available())):
                        loss = training_model(batch, step)
                        token_structure_loss = lidar_token_structure_loss_tensor(model)
                        token_structure_loss_contrib = (
                            token_structure_loss_weight * token_structure_loss
                        )
                    if not synchronized_loss_finite(loss, distributed):
                        raise FloatingPointError(
                            f"Non-finite loss at step {step}; rank={rank}; "
                            f"sample_ids={batch_sample_ids(batch_cpu)}"
                        )
                    scaler.scale(loss).backward()
                    break
                except RuntimeError as error:
                    retryable = retryable_backward_shape_error(error)
                    failure = {
                        "backward_attempt": backward_attempt + 1,
                        "error": str(error),
                        "retryable": retryable,
                        "sample_ids": batch_sample_ids(batch_cpu),
                        "step": step,
                    }
                    failure["rank"] = rank
                    print(json.dumps({"backward_failure": failure}, sort_keys=True), flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    if distributed or not retryable or backward_attempt > 0:
                        raise
                    del loss, token_structure_loss, token_structure_loss_contrib
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            scaler.unscale_(optimizer)
            router_grad_stats = ray_fusion_grad_stats(model)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_skipped = scaler.get_scale() < scale_before
            window["optimizer_skipped_steps"] += int(optimizer_skipped)

            lidar_support_stats = lidar_support_mask_stats(batch, dilation=args.lidar_support_dilation)
            window["steps"] += 1
            window["lidar_support_image_mask_coverage"] += lidar_support_stats["lidar_support_image_mask_coverage"]
            window["lidar_support_latent_mask_coverage"] += lidar_support_stats["lidar_support_latent_mask_coverage"]
            loss_metrics = getattr(model.DDPM, "last_loss_metrics", {})
            semantic_metrics = getattr(model, "last_lidar_semantic_alignment_metrics", {})
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
            semantic_applied = int(float(semantic_metrics.get("lidar_semantic_alignment_applied", 0.0)) > 0.0)
            window["lidar_semantic_alignment_applied_steps"] += semantic_applied
            window["lidar_semantic_alignment_loss"] += float(
                semantic_metrics.get("lidar_semantic_alignment_loss", 0.0)
            )
            window["lidar_semantic_alignment_contrib"] += float(
                semantic_metrics.get("lidar_semantic_alignment_contrib", 0.0)
            )
            window["lidar_semantic_alignment_mask_coverage"] += float(
                semantic_metrics.get("lidar_semantic_alignment_mask_coverage", 0.0)
            )
            window["lidar_semantic_alignment_target_available"] += float(
                semantic_metrics.get("lidar_semantic_alignment_target_available", 0.0)
            )

            if step % args.log_every == 0 or step == 1:
                window_steps = max(1, int(window["steps"]))
                record = {
                    "step": step,
                    "loss": scalar(loss),
                    "condition_mode": CONDITION_MODE,
                    "lidar_fusion_mode": "ray_posterior",
                    "lidar_geom_mode": LIDAR_GEOM_MODE,
                    "ray_evidence_mask_mode": RAY_EVIDENCE_MASK_MODE,
                    "num_projected_lidar_points": int(batch.get("num_projected_lidar_points", torch.zeros(1, device=device)).sum().detach().cpu()),
                    "lidar_support_loss_weight": args.lidar_support_loss_weight,
                    **lidar_support_stats,
                    "window_steps": window_steps,
                    "optimizer_skipped_steps_in_window": window["optimizer_skipped_steps"],
                    "amp_scale": float(scaler.get_scale()),
                    "lidar_depth_head_mode": getattr(model.DDPM.denoise_model, "lidar_depth_head_mode", "latent"),
                    "window_lidar_depth_log_l1_mean": window["lidar_depth_log_l1"] / window_steps,
                    "window_lidar_depth_log_l1_contrib_mean": window["lidar_depth_log_l1_contrib"] / window_steps,
                    "window_lidar_depth_mask_coverage_mean": window["lidar_depth_mask_coverage"] / window_steps,
                    "window_lidar_support_image_mask_coverage_mean": window[
                        "lidar_support_image_mask_coverage"
                    ]
                    / window_steps,
                    "window_lidar_support_latent_mask_coverage_mean": window[
                        "lidar_support_latent_mask_coverage"
                    ]
                    / window_steps,
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
                    "lidar_depth_loss_weight": float(args.lidar_depth_loss_weight),
                    "lidar_depth_output_scale": float(cfg.model.params.lidar_depth_output_scale),
                    "lidar_depth_bottleneck_scale": float(cfg.model.params.lidar_depth_bottleneck_scale),
                    "lidar_depth_log_eps": float(args.lidar_depth_log_eps),
                    "lidar_semantic_alignment_weight": float(args.lidar_semantic_alignment_weight),
                    "lidar_semantic_alignment_mask_mode": LIDAR_SEMANTIC_MASK_MODE,
                    "window_lidar_semantic_alignment_applied_frac": window[
                        "lidar_semantic_alignment_applied_steps"
                    ]
                    / window_steps,
                    "window_lidar_semantic_alignment_loss_mean": window[
                        "lidar_semantic_alignment_loss"
                    ]
                    / window_steps,
                    "window_lidar_semantic_alignment_contrib_mean": window[
                        "lidar_semantic_alignment_contrib"
                    ]
                    / window_steps,
                    "window_lidar_semantic_alignment_mask_coverage_mean": window[
                        "lidar_semantic_alignment_mask_coverage"
                    ]
                    / window_steps,
                    "window_lidar_semantic_alignment_target_available_mean": window[
                        "lidar_semantic_alignment_target_available"
                    ]
                    / window_steps,
                    "lidar_token_output_norm": None if use_lidar_pixel else LIDAR_TOKEN_OUTPUT_NORM,
                    "lidar_token_structure_loss_weight": token_structure_loss_weight,
                    "lidar_token_structure_target_ratio": float(args.lidar_token_structure_target_ratio),
                    "lidar_token_structure_loss": scalar(token_structure_loss),
                    "lidar_token_structure_loss_contrib": scalar(token_structure_loss_contrib),
                    "lidar_token_valid_sample_ratio": scalar(getattr(getattr(model, "lidar_context_model", None), "last_valid_sample_ratio", torch.tensor(0.0))),
                    "lidar_evidence_hit_coverage": scalar(getattr(getattr(model, "lidar_context_model", None), "last_hit_coverage", torch.tensor(0.0))),
                    "lidar_evidence_empty_coverage": scalar(getattr(getattr(model, "lidar_context_model", None), "last_empty_coverage", torch.tensor(0.0))),
                    **lidar_context_token_stats(model),
                    **router_grad_stats,
                    **ray_fusion_parameter_stats(model),
                    **lidar_attention_stats(model),
                    **semantic_metrics,
                    **loss_metrics,
                    **resource_metrics(),
                }
                record.update(
                    {
                        "data_epoch": int(data_epoch),
                        "global_batch_size": int(args.batch_size) * int(world_size),
                    }
                )
                record = gather_distributed_record(record, distributed, rank, world_size)
                if is_main:
                    metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
                    metrics_file.flush()
                    print(json.dumps(record, sort_keys=True))
                window = new_metric_window()

            if args.save_every > 0 and step % args.save_every == 0:
                distributed_barrier(distributed)
                if is_main:
                    _, save_stats = save_step_checkpoint(
                        ckpt_dir,
                        model,
                        optimizer,
                        scaler,
                        step,
                        args,
                        metadata,
                    )
                    last_saved_step = step
                    removed = prune_step_checkpoints(ckpt_dir, args.keep_step_checkpoints)
                    print(
                        json.dumps(
                            {
                                "checkpoint_saved": True,
                                "pruned_checkpoints": removed,
                                "step": step,
                                **save_stats,
                            },
                            sort_keys=True,
                        )
                    )
                distributed_barrier(distributed)
            if int(args.sample_every) > 0 and step % int(args.sample_every) == 0:
                distributed_barrier(distributed)
                if is_main:
                    sample_record = run_inline_samples(model, inline_sample_dataset, args, out_dir, step)
                    if sample_record is not None:
                        print(json.dumps(sample_record, sort_keys=True))
                distributed_barrier(distributed)

            del batch, batch_cpu, loss, token_structure_loss, token_structure_loss_contrib

    distributed_barrier(distributed)
    if is_main:
        if last_saved_step != args.steps:
            _, save_stats = save_step_checkpoint(
                ckpt_dir,
                model,
                optimizer,
                scaler,
                args.steps,
                args,
                metadata,
            )
            removed = prune_step_checkpoints(ckpt_dir, args.keep_step_checkpoints)
            print(
                json.dumps(
                    {
                        "checkpoint_saved": True,
                        "pruned_checkpoints": removed,
                        "step": args.steps,
                        **save_stats,
                    },
                    sort_keys=True,
                )
            )
        print(json.dumps({"complete": True, "last_ckpt": str(ckpt_dir / "last.pt"), "steps": args.steps}, sort_keys=True))
    distributed_barrier(distributed)


def main():
    args = parse_args()
    try:
        dist_info = init_distributed(args)
        validate_runtime_args(args, dist_info["world_size"])
        run_training(args, dist_info)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
