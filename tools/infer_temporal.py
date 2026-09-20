"""Persistent-history temporal inference for a prepared KITTI payload."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ldm.modules.temporal_pair_training import (
    assert_same_base,
    base_identity,
    build_history,
    load_temporal_checkpoint,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="single-frame base checkpoint")
    weights = p.add_mutually_exclusive_group()
    weights.add_argument("--temporal-checkpoint", default="", help="history and decoder checkpoint tied to the exact single-frame base")
    weights.add_argument("--static-checkpoint", default="", help="static RGB/dense adapter checkpoint")
    p.add_argument("--input", required=True, help="torch payload with kwargs, shape, optional history/history_geometry")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:4")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--temporal-mode", choices=["geometry", "content", "static", "static_dense", "static_adaptive", "static_centered"], default="geometry")
    p.add_argument("--temporal-hidden-dim", type=int, default=64)
    p.add_argument("--input-variant", choices=["rgb", "valid", "types"], default="types")
    p.add_argument("--history-mode", choices=["correct", "off", "wrong_history", "wrong_geometry"], default="correct")
    return p.parse_args(argv)


def evaluation_device(requested):
    """CLI GPU numbers are physical, even with CUDA_VISIBLE_DEVICES remapping."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in ("cuda:4", "cuda:5"):
        raise ValueError("only cpu or physical GPU4/5 are allowed")
    physical = requested.split(":")[1]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return torch.device(requested)
    ids = [part.strip() for part in visible.split(",")]
    if physical not in ids:
        raise ValueError("requested physical GPU is absent from CUDA_VISIBLE_DEVICES")
    return torch.device("cuda", ids.index(physical))


def tree_to(value, device):
    if torch.is_tensor(value): return value.detach().to(device)
    if isinstance(value, dict): return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list): return [tree_to(v, device) for v in value]
    if isinstance(value, tuple): return tuple(tree_to(v, device) for v in value)
    return value


def tensor_hash(value):
    value = value.detach().float().cpu().contiguous()
    h = hashlib.sha256(); h.update(str(tuple(value.shape)).encode()); h.update(str(value.dtype).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_static_adapter(path, reader, base):
    """Load an adapter only after checking its base and complete tensor schema."""
    payload = torch.load(path, map_location="cpu")
    version = payload.get("version")
    if version == "temporal_static_adapter_v1":
        if reader.mode != "static":
            raise ValueError("static v1 checkpoint requires static mode")
    elif version in {"temporal_static_adapter_v2", "temporal_static_adapter_v3", "temporal_static_adapter_v4"}:
        expected_mode = {
            "temporal_static_adapter_v2": "static_dense",
            "temporal_static_adapter_v3": "static_adaptive",
            "temporal_static_adapter_v4": "static_centered",
        }[version]
        if reader.mode != expected_mode or payload.get("model_mode") != expected_mode:
            raise ValueError("static checkpoint requires " + expected_mode + " mode")
        if expected_mode in {"static_adaptive", "static_centered"}:
            if payload.get("fusion_dim") != reader.fusion_dim or payload.get("time_embed_dim") != reader.time_embed_dim:
                raise ValueError("adaptive checkpoint fusion/time width differs")
        if payload.get("input_variant") != getattr(reader, "input_variant", None):
            raise ValueError("static v2 checkpoint input_variant differs")
        if payload.get("reference_kind") not in {"sparse", "dense"}:
            raise ValueError("static v2 checkpoint reference_kind is invalid")
    else:
        raise ValueError("unsupported static checkpoint version")
    assert_same_base(payload.get("base_checkpoint", {}), base)
    if payload.get("hidden_dim") != reader.hidden_dim:
        raise ValueError("static checkpoint width differs")
    state = payload.get("state_dict", {})
    expected = reader.state_dict()
    if set(state) != set(expected):
        raise ValueError("static adapter checkpoint keys differ")
    for key, value in state.items():
        if not torch.is_tensor(value) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError("static checkpoint shape/dtype differs: " + key)
        if not torch.isfinite(value).all():
            raise ValueError("static checkpoint has non-finite weights: " + key)
    reader.load_state_dict(state, strict=True)
    return payload


def load_model(config, checkpoint, temporal_checkpoint, device, mode="geometry", hidden_dim=64,
               static_checkpoint="", input_variant="types"):
    if static_checkpoint and (mode not in {"static", "static_dense", "static_adaptive", "static_centered"} or temporal_checkpoint):
        raise ValueError("static checkpoint requires static/static_dense/static_adaptive/static_centered mode and cannot mix temporal checkpoints")
    from omegaconf import OmegaConf
    from utils.util import instantiate_from_config
    cfg = OmegaConf.load(config); cfg.model.params.pre_ldm_model_path = None
    model = instantiate_from_config(cfg.model)
    payload = torch.load(checkpoint, map_location="cpu")
    model.DDPM.denoise_model.load_state_dict(payload["denoise_model"], strict=True)
    model.condition_model_sat.load_state_dict(payload["condition_model_sat"], strict=True)
    model.lidar_context_model.load_state_dict(payload["lidar_context_model"], strict=True)
    base = base_identity(checkpoint, payload)
    del payload; gc.collect()
    unet = model.DDPM.denoise_model
    if not hasattr(unet, "configure_temporal_history"):
        raise ValueError("UNet lacks configure_temporal_history")
    unet.configure_temporal_history(enabled=True, mode=mode, hidden_dim=int(hidden_dim), input_variant=input_variant)
    model = model.to(device).eval().requires_grad_(False)
    if temporal_checkpoint:
        load_temporal_checkpoint(temporal_checkpoint, model, base, strict=True)
    if static_checkpoint:
        load_static_adapter(static_checkpoint, model.DDPM.denoise_model.temporal_history, base)
    return model, base


def prepare_history_from_payload(data: Mapping[str, Any], device: torch.device, mode: str):
    if mode == "off":
        return None, None
    history = data.get("history")
    if history is None:
        return None, None
    history = tree_to(history, device)
    if "static_rgb" in history:
        if "history_geometry" in data or mode not in ("correct", "off"):
            raise ValueError("static payload requires explicit aligned RGB and correct/off mode; rebuild alignment for interventions")
    if "dense_rgb" in history:
        if "history_geometry" in data or mode not in ("correct", "off"):
            raise ValueError("dense static payload requires explicit aligned RGB and correct/off mode")
    metrics = None
    if "history_geometry" in data:
        from tools.temporal_ray_geometry import build_pair_ray_geometry
        geo_cfg = data["history_geometry"]
        latent = history.get("latent")
        if not torch.is_tensor(latent) or latent.ndim != 4:
            raise ValueError("history_geometry requires history['latent'] [B,C,H,W]")
        prev_rows = geo_cfg["prev_rows"]; cur_rows = geo_cfg["cur_rows"]
        if isinstance(prev_rows, dict): prev_rows = [prev_rows]
        if isinstance(cur_rows, dict): cur_rows = [cur_rows]
        if len(prev_rows) != latent.shape[0] or len(cur_rows) != latent.shape[0]:
            raise ValueError("one complete sensor-row pair is required per sample")
        geos = []
        metrics = []
        for prev, cur in zip(prev_rows, cur_rows):
            geo = build_pair_ray_geometry(prev, cur, kitti_root=geo_cfg.get("kitti_root"), grid=tuple(latent.shape[-2:]),
                                          num_depth_candidates=int(geo_cfg.get("num_depth_candidates", 16)))
            geos.append(geo); metrics.append(dict(geo.get("metrics", {})))
        history = build_history(latent, geos)
    # Correspondence-only intervention: keep current physical/satellite keys
    # fixed and corrupt which previous appearance they address.
    if mode == "wrong_geometry":
        if "history_grid" not in history:
            raise ValueError("wrong_geometry requires explicit geometric candidates")
        history = dict(history)
        history["history_grid"] = history["history_grid"].clone()
        history["history_grid"][..., 0] *= -1
    if mode == "wrong_history":
        latent = history["latent"]
        history = dict(history)
        if "wrong_history_latent" in data:
            wrong = data["wrong_history_latent"].detach().to(device)
            if wrong.shape != latent.shape or torch.equal(wrong, latent):
                raise ValueError("wrong history must be a different latent with the same shape")
            history["latent"] = wrong
        elif latent.shape[0] > 1:
            history["latent"] = torch.roll(latent, shifts=1, dims=0).detach()
            if torch.equal(history["latent"], latent):
                raise ValueError("batch rotation did not create a wrong-history control")
        else:
            raise ValueError("batch1 wrong_history needs an explicit unrelated wrong_history_latent")
    history["enabled"] = torch.ones(history["latent"].shape[0], dtype=torch.bool, device=device)
    return history, metrics


def sample_frame(model, kwargs, shape, seed, device, history=None, steps=50, guidance=7.5):
    from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(int(seed)); noise = torch.randn(shape, device=device)
    sampler = KITTI_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor)
    kwargs = tree_to(kwargs, device)
    params = {"conditioning": kwargs.get("context")}
    for key in ["left_camera_k", "gt_shift_x", "gt_shift_y", "theta", "range_img", "range_mask", "camera_to_lidar",
                "lidar_context", "lidar_evidence", "lidar_geometry_mask"]:
        params[key] = kwargs.get(key)
    with torch.no_grad(), (torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()):
        z, info = sampler.sample(S=steps, batch_size=shape[0], shape=list(shape[1:]), x_T=noise, eta=0.0,
                                 verbose=False, unconditional_guidance_scale=guidance, history=tree_to(history, device), **params)
    return z.float(), info, tensor_hash(noise)


def decode(model, z):
    with torch.no_grad(), (torch.cuda.amp.autocast() if z.device.type == "cuda" else nullcontext()):
        return ((model.pre_AE_model.decode(z * (1.0 / model.scale_factor)) + 1) / 2).clamp(0, 1)


def main(argv=None):
    args = parse_args(argv)
    device = evaluation_device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    out = Path(args.out)
    if out.exists(): raise ValueError("refusing overwrite")
    torch.set_num_threads(1)
    data = torch.load(args.input, map_location="cpu")
    model, base = load_model(args.config, args.checkpoint, args.temporal_checkpoint, device,
                            args.temporal_mode, args.temporal_hidden_dim, args.static_checkpoint,
                            input_variant=args.input_variant)
    history, geo_metrics = prepare_history_from_payload(data, device, args.history_mode)
    z, info, noise_hash = sample_frame(model, data["kwargs"], tuple(data["shape"]), args.seed, device, history, args.ddim_steps, args.guidance)
    rgb = decode(model, z)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".pt", ".pth"):
        torch.save({"latent": z.cpu(), "rgb": rgb.cpu()}, out)
    else:
        if rgb.shape[0] != 1: raise ValueError("image output requires batch1; use .pt for batches")
        from PIL import Image
        import numpy as np
        Image.fromarray((rgb[0].float().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(out)
    out.with_suffix(".json").write_text(json.dumps({"base_checkpoint": base, "temporal_checkpoint": args.temporal_checkpoint,
        "static_checkpoint": args.static_checkpoint, "temporal_mode": args.temporal_mode, "input_variant": args.input_variant,
        "history_mode": args.history_mode, "sampler_history": info.get("history") if isinstance(info, dict) else None,
        "history_geometry": geo_metrics, "noise_hash": noise_hash, "latent_hash": tensor_hash(z), "rgb_hash": tensor_hash(rgb)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
