"""Utilities for persistent two-frame temporal-history training.

This module is intentionally small and model-agnostic. The UNet owns the actual
``configure_temporal_history`` implementation; this file only prepares history,
selects trainable parameters, computes the standard diffusion epsilon loss, and
serializes checkpoints for the history reader and decoder.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINT_VERSION = "temporal_persistent_pairs_v1"
TEMPORAL_PREFIX = "temporal_history."
DECODER_PREFIXES = ("output_blocks.", "out.")


def training_contract(task: str, history_policy: str, history_dropout: float) -> Dict[str, Any]:
    """Keep mandatory-reference training separate from explicit ablations."""
    if task not in ("gt_next_frame", "history_ablation"):
        raise ValueError("unknown training task: " + str(task))
    if history_policy not in ("correct", "off"):
        raise ValueError("unknown history policy: " + str(history_policy))
    if not 0.0 <= float(history_dropout) <= 1.0:
        raise ValueError("history dropout must be in [0,1]")
    if task == "gt_next_frame" and (history_policy != "correct" or float(history_dropout) != 0.0):
        raise ValueError("gt_next_frame requires correct GT history and history-dropout=0; use history_ablation for controls")
    return {
        "task": task,
        "history_source": "previous_gt" if history_policy == "correct" else "none",
        "target_source": "current_gt",
        "history_required": task == "gt_next_frame",
        "history_dropout": float(history_dropout),
        "objective": "current_frame_epsilon_mse",
        "primary_validation": "independent_gt_pairs" if task == "gt_next_frame" else "ablation",
    }


def validate_resume_settings(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Resume is exact continuation; task changes require weights-only initialization."""
    saved_task = saved.get("training_task", "history_ablation")
    current_task = current.get("training_task", "gt_next_frame")
    if saved_task != current_task:
        raise ValueError("resume training task differs; use --init-temporal-ckpt for a new GT-pair run")
    for key in ("seed", "batch_size", "train_manifest", "history_policy", "history_dropout",
                "geometry_depth_candidates", "latent_grid_height", "latent_grid_width", "config",
                "temporal_lr", "decoder_lr", "world_size", "kitti_root", "sd_base_ckpt",
                "lidar_pixel_feature_cache_root", "image_semantic_cache_root"):
        if key in saved and saved[key] != current.get(key):
            raise ValueError("resume setting differs: " + key)


def raw_unet(model: Any) -> torch.nn.Module:
    unet = model.DDPM.denoise_model if hasattr(model, "DDPM") else model
    return unet.module if hasattr(unet, "module") else unet


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(value.shape)).encode())
    h.update(str(value.dtype).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def base_identity(checkpoint: str | Path, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    identity = {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}
    if payload is not None:
        if "step" in payload:
            identity["step"] = int(payload["step"])
        if "denoise_model" in payload and isinstance(payload["denoise_model"], Mapping):
            identity["denoise_model_sha256"] = state_dict_sha256(payload["denoise_model"])
    return identity


def assert_same_base(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    for key in ("sha256", "denoise_model_sha256", "step"):
        if key in expected or key in observed:
            if expected.get(key) != observed.get(key):
                raise ValueError("base checkpoint identity mismatch for %s" % key)


def is_trainable_name(name: str) -> bool:
    return name.startswith(TEMPORAL_PREFIX) or name.startswith(DECODER_PREFIXES)


def configure_temporal_pair_trainables(model: Any, mode: str = "geometry", hidden_dim: int = 64) -> Dict[str, List[torch.nn.Parameter]]:
    """Freeze the base, then train temporal history plus existing decoder blocks."""
    if mode not in ("geometry", "content"):
        raise ValueError("mode must be 'geometry' or 'content'")
    model.eval().requires_grad_(False)
    unet = raw_unet(model)
    if not hasattr(unet, "configure_temporal_history"):
        raise ValueError("UNet lacks configure_temporal_history")
    unet.configure_temporal_history(enabled=True, mode=mode, hidden_dim=int(hidden_dim))
    groups: Dict[str, List[torch.nn.Parameter]] = {"temporal": [], "decoder": []}
    for name, parameter in unet.named_parameters():
        if name.startswith(TEMPORAL_PREFIX):
            parameter.requires_grad_(True); groups["temporal"].append(parameter)
        elif name.startswith(DECODER_PREFIXES):
            parameter.requires_grad_(True); groups["decoder"].append(parameter)
    if not groups["temporal"]:
        raise ValueError("no temporal_history.* parameters found")
    if not groups["decoder"]:
        raise ValueError("no output_blocks./out. decoder parameters found")
    return groups


def flatten_trainable_groups(groups: Mapping[str, Sequence[torch.nn.Parameter]]) -> List[torch.nn.Parameter]:
    return [p for values in groups.values() for p in values]


def make_optimizer_param_groups(groups: Mapping[str, Sequence[torch.nn.Parameter]], temporal_lr: float, decoder_lr: float):
    return [
        {"name": "temporal", "params": list(groups["temporal"]), "lr": float(temporal_lr)},
        {"name": "decoder", "params": list(groups["decoder"]), "lr": float(decoder_lr)},
    ]


def trainable_state_dict(model: Any) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in raw_unet(model).state_dict().items() if is_trainable_name(k)}


def validate_trainable_selection(model: Any, state: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    current = raw_unet(model).state_dict()
    selected = [k for k in current if is_trainable_name(k)]
    extra = sorted(k for k in state if not is_trainable_name(k))
    missing = sorted(set(selected) - set(state))
    unknown = sorted(k for k in state if k not in current)
    if extra or missing or unknown:
        raise ValueError("trainable state mismatch extra=%d missing=%d unknown=%d" % (len(extra), len(missing), len(unknown)))
    return {
        "temporal": sum(1 for k in state if k.startswith(TEMPORAL_PREFIX)),
        "decoder": sum(1 for k in state if k.startswith(DECODER_PREFIXES)),
        "total": len(state),
    }


def load_temporal_state(model: Any, state: Mapping[str, torch.Tensor], strict: bool = True):
    if not state:
        raise ValueError("empty trainable state")
    unet = raw_unet(model)
    current = unet.state_dict()
    if strict:
        validate_trainable_selection(model, state)
    else:
        bad = [k for k in state if not is_trainable_name(k)]
        if bad:
            raise ValueError("non-trainable keys in checkpoint: %s" % bad[:8])
    for key, value in state.items():
        if key in current:
            if current[key].shape != value.shape or current[key].dtype != value.dtype:
                raise ValueError("shape/dtype mismatch for %s" % key)
            if not torch.isfinite(value).all():
                raise ValueError("non-finite checkpoint tensor: " + key)
            current[key] = value
    return unet.load_state_dict(current, strict=True)

def _np_to_tensor(value: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if torch.is_tensor(value):
        out = value.detach().to(device=device)
    else:
        out = torch.as_tensor(np.asarray(value), device=device)
    return out.to(dtype=dtype) if dtype is not None else out


def build_history(previous_latent: torch.Tensor, geometries: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    """Build the contract history dict from per-sample numpy geometry outputs."""
    if previous_latent.ndim != 4 or previous_latent.shape[0] != len(geometries):
        raise ValueError("previous_latent must be [B,C,H,W] with one geometry per sample")
    b, _, h, w = previous_latent.shape
    device = previous_latent.device
    fields = {"history_grid": [], "sat_grid": [], "valid": [], "sat_valid": [], "positions": []}
    for geo in geometries:
        for key in fields:
            if key not in geo:
                raise ValueError("geometry missing %s" % key)
        fields["history_grid"].append(_np_to_tensor(geo["history_grid"], device, torch.float32))
        fields["sat_grid"].append(_np_to_tensor(geo["sat_grid"], device, torch.float32))
        fields["valid"].append(_np_to_tensor(geo["valid"], device).bool())
        fields["sat_valid"].append(_np_to_tensor(geo["sat_valid"], device).bool())
        fields["positions"].append(_np_to_tensor(geo["positions"], device, torch.float32))
    history = {key: torch.stack(values, 0) for key, values in fields.items()}
    if history["history_grid"].ndim != 5 or history["history_grid"].shape[:3] != (b, h, w) or history["history_grid"].shape[-1] != 2:
        raise ValueError("history_grid shape does not match latent")
    if history["sat_grid"].shape != history["history_grid"].shape:
        raise ValueError("sat_grid shape does not match history_grid")
    candidate_shape = history["history_grid"].shape[:-1]
    if candidate_shape[-1] < 1:
        raise ValueError("at least one candidate is required")
    for key in ("valid", "sat_valid"):
        if history[key].shape != candidate_shape:
            raise ValueError(key + " shape does not match candidates")
    if history["positions"].shape != (*candidate_shape, 4):
        raise ValueError("positions shape does not match candidates")
    for key in ("history_grid", "sat_grid", "positions"):
        if not torch.isfinite(history[key]).all():
            raise ValueError("non-finite geometry: " + key)
    history["latent"] = previous_latent.detach()
    return history


def apply_history_dropout(history: Mapping[str, Any], dropout: float, seed: int, device: Optional[torch.device] = None) -> Dict[str, Any]:
    if not 0.0 <= float(dropout) <= 1.0:
        raise ValueError("history dropout must be in [0,1]")
    latent = history.get("latent")
    if not torch.is_tensor(latent) or latent.ndim != 4:
        raise ValueError("history latent must be [B,C,H,W]")
    dev = device or latent.device
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    enabled = torch.rand((latent.shape[0],), generator=gen, device=dev) >= float(dropout)
    out = dict(history)
    out["enabled"] = enabled
    return out


def seed_training_step(seed: int, device: Optional[torch.device] = None) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def anchor_trainable_loss(loss: torch.Tensor, params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    anchor = loss.new_zeros(())
    for parameter in params:
        if parameter.requires_grad:
            anchor = anchor + parameter.reshape(-1)[0].float() * 0.0
    return loss + anchor


def epsilon_prediction_loss(ddpm: Any, x_start: torch.Tensor, cond: Mapping[str, Any], history: Optional[Mapping[str, Any]], seed: int):
    if x_start.ndim != 4 or not torch.isfinite(x_start).all():
        raise ValueError("x_start must be finite NCHW")
    gen = torch.Generator(device=x_start.device).manual_seed(int(seed))
    if hasattr(ddpm, "num_timesteps"):
        num_timesteps = int(ddpm.num_timesteps)
    elif hasattr(ddpm, "timesteps"):
        num_timesteps = int(ddpm.timesteps)
    else:
        raise ValueError("DDPM object must expose num_timesteps or timesteps")
    t = torch.randint(0, num_timesteps, (x_start.shape[0],), generator=gen, device=x_start.device).long()
    noise = torch.randn(x_start.shape, generator=gen, device=x_start.device, dtype=x_start.dtype)
    x_noisy = ddpm.q_sample(x_start=x_start, t=t, noise=noise)
    allowed = {
        "context", "lidar_context", "lidar_evidence", "lidar_geometry_mask",
        "control_grd", "left_camera_k", "gt_shift_x", "gt_shift_y", "theta",
    }
    denoise_cond = {key: value for key, value in dict(cond).items() if key in allowed}
    model_out = ddpm.denoise_model(x_noisy, t, history=history, **denoise_cond)
    loss = F.mse_loss(model_out.float(), noise.float())
    return loss, {"t": t.detach(), "noise": noise.detach(), "model_out": model_out}


def save_temporal_checkpoint(path: str | Path, model: Any, optimizer: Any, scaler: Any, step: int,
                             base: Mapping[str, Any], args: Mapping[str, Any], epoch: int = 0,
                             dataloader_offset: int = 0, rng_state: Optional[Mapping[str, Any]] = None) -> None:
    state = trainable_state_dict(model)
    payload = {
        "version": CHECKPOINT_VERSION,
        "artifact_kind": "temporal_pair_trainable",
        "base_checkpoint": dict(base),
        "trainable_state": state,
        "trainable_counts": validate_trainable_selection(model, state),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "dataloader_offset": int(dataloader_offset),
        "rng_state": dict(rng_state or {}),
        "args": dict(args),
        "training_contract": training_contract(args.get("training_task", "history_ablation"),
                                               args.get("history_policy", "correct"),
                                               args.get("history_dropout", 0.0)),
    }
    tmp = Path(path).with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp)
    tmp.replace(path)


def load_temporal_checkpoint(path: str | Path, model: Any, expected_base: Mapping[str, Any],
                             optimizer: Any = None, scaler: Any = None, strict: bool = True) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported temporal checkpoint version")
    assert_same_base(payload.get("base_checkpoint", {}), expected_base)
    reader = getattr(raw_unet(model), "temporal_history", None)
    settings = payload.get("args", {})
    if reader is not None and hasattr(reader, "mode"):
        if settings.get("temporal_mode", reader.mode) != reader.mode:
            raise ValueError("history mode differs from checkpoint")
        if int(settings.get("temporal_hidden_dim", reader.hidden_dim)) != reader.hidden_dim:
            raise ValueError("history width differs from checkpoint")
    load_temporal_state(model, payload["trainable_state"], strict=strict)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload


def prune_checkpoints(directory: str | Path, keep: int) -> None:
    if keep < 1:
        raise ValueError("keep must be at least one")
    files = sorted(Path(directory).glob("temporal_pair_step_*.pt"))
    for old in files[:-int(keep)]:
        old.unlink()
