"""Shared single-frame loading and geometric pair data utilities (no temporal plugins)."""
from __future__ import annotations
import hashlib
import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
import torch
from PIL import Image
from ldm.modules.temporal_pair_training import base_identity
IMAGE_SIZE = (512, 128)

def load_settings_args(path: str | Path) -> Namespace:
    payload = json.loads(Path(path).read_text())
    args = payload.get("args", payload.get("train_args", payload))
    required = [
        "config", "checkpoint", "train_manifest", "val_manifest", "kitti_root",
        "sd_base_ckpt", "lidar_pixel_feature_cache_root", "image_semantic_cache_root",
    ]
    missing = [key for key in required if key not in args]
    if missing:
        raise ValueError("settings JSON is missing required args: " + ", ".join(missing))
    defaults = {
        "batch_size": 1,
        "num_workers": 0,
        "temporal_lr": 1e-4,
        "decoder_lr": 0.0,
        "min_free_out_gb": 0.0,
        "local_rank": -1,
    }
    merged = {**defaults, **args}
    return Namespace(**merged)


def evaluation_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if not requested.startswith("cuda:"):
        raise ValueError("--device must be cpu or cuda:<physical_id>")
    physical = requested.split(":", 1)[1]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [part.strip() for part in visible.split(",")]
        if physical not in ids:
            raise ValueError(f"requested physical GPU {physical} is absent from CUDA_VISIBLE_DEVICES={visible}")
        return torch.device("cuda", ids.index(physical))
    return torch.device(requested)


def parse_sample_id(sample_id: str):
    parts = str(sample_id).split("/")
    if len(parts) < 3:
        raise ValueError("sample id must include date/drive/frame: " + str(sample_id))
    try:
        frame = int(parts[-1])
    except ValueError as exc:
        raise ValueError("sample id frame must be numeric: " + str(sample_id)) from exc
    return "/".join(parts[:-1]), frame


def drive_id(sample_id: str) -> str:
    drive, _frame = parse_sample_id(sample_id)
    return drive


def validate_pair_splits(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    train_ids = {sid for e in entries if e["split"] == "train" for sid in (e["previous"], e["current"])}
    heldout_ids = {sid for e in entries if e["split"] == "heldout" for sid in (e["previous"], e["current"])}
    overlap = sorted(train_ids & heldout_ids)
    train_drives = {drive_id(sid) for sid in train_ids}
    heldout_drives = {drive_id(sid) for sid in heldout_ids}
    if overlap:
        raise ValueError("train/heldout sample id overlap: " + ", ".join(overlap[:8]))
    for entry in entries:
        prev_drive, prev_frame = parse_sample_id(entry["previous"])
        cur_drive, cur_frame = parse_sample_id(entry["current"])
        if prev_drive != cur_drive:
            raise ValueError(f"pair {entry.get('name')} crosses drives")
        if cur_frame != prev_frame + 1:
            raise ValueError(f"pair {entry.get('name')} is not consecutive")
    counts = {"train": sum(e["split"] == "train" for e in entries), "heldout": sum(e["split"] == "heldout" for e in entries)}
    if counts["train"] < 1 or counts["heldout"] < 1:
        raise ValueError("both train and heldout pairs are required")
    return {"counts": counts, "train_drives": sorted(train_drives), "heldout_drives": sorted(heldout_drives)}


def index_dataset(dataset) -> Dict[str, int]:
    return {row["sample_id"]: idx for idx, row in enumerate(dataset.records)}


def manifest_key_for_split(split: str) -> str:
    if split == "train":
        return "train_manifest"
    if split == "heldout":
        return "val_manifest"
    raise ValueError("unknown pair split: " + str(split))


def sample_from_entry_dataset(indexes: Mapping[str, Mapping[str, Any]], entry: Mapping[str, Any], sample_id: str) -> dict:
    manifest_key = manifest_key_for_split(entry["split"])
    item = indexes[manifest_key]
    idx = item["index"].get(sample_id)
    if idx is None:
        raise KeyError(f"{sample_id} not found in declared {manifest_key} for pair {entry.get('name')}")
    return item["dataset"][idx]


def previous_rgb_sample(dataset, row: Mapping[str, Any]) -> dict:
    with Image.open(row["image_02_path"]) as image:
        return {"grd_left_imgs": dataset.grd_transform(image.convert("RGB"))}


def training_ns(args):
    return Namespace(
        lr=args.temporal_lr, sd_base_ckpt=args.sd_base_ckpt, lidar_support_loss_weight=1.0,
        lidar_support_dilation=8, lidar_depth_loss_weight=0.1, lidar_depth_log_eps=1e-3,
        lidar_semantic_alignment_weight=0.2, lidar_evidence_dilation=4,
        lidar_evidence_free_space_dilation=14, lidar_token_structure_target_ratio=0.08,
        lidar_reference_window=3, batch_size=args.batch_size, num_workers=args.num_workers,
        train_manifest=args.train_manifest, val_manifest=args.val_manifest, kitti_root=args.kitti_root,
        lidar_ray_feature_cache_root="", image_semantic_cache_root=args.image_semantic_cache_root,
        lidar_pixel_feature_cache_root=args.lidar_pixel_feature_cache_root,
    )


def stack_samples(samples):
    out = {}
    for key in samples[0].keys():
        first = samples[0][key]
        out[key] = torch.stack([x[key] for x in samples], 0) if torch.is_tensor(first) else [x[key] for x in samples]
    return out


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def encode_conditions(model, batch, device):
    batch = _to_device(batch, device)
    sat = model.get_input(batch, "sat_map") * 2 - 1
    rgb = model.get_input(batch, "grd_left_imgs").clamp(0, 1)
    lidar_cond = model.get_input(batch, model.lidar_condition_key)
    camera_to_lidar = model.get_input(batch, "camera_to_lidar").squeeze(-1)
    left_camera_k = model.get_input(batch, "left_camera_k").squeeze(-1)
    pix = model.get_optional_lidar_batch_tensor(batch, "lidar_pixel_features", device)
    pix_mask = model.get_optional_lidar_batch_tensor(batch, "lidar_pixel_features_mask", device)
    pix_avail = model.get_optional_lidar_batch_tensor(batch, "lidar_pixel_features_available", device)
    with torch.no_grad():
        context = model.make_condition(sat, batch)
        lidar_context = model.make_lidar_context(lidar_cond, camera_to_lidar=camera_to_lidar, left_camera_k=left_camera_k,
                                                 lidar_pixel_features=pix, lidar_pixel_features_mask=pix_mask,
                                                 lidar_pixel_features_available=pix_avail)
        lidar_evidence = model.make_lidar_evidence(lidar_cond)
        lidar_geometry_mask = model.make_lidar_geometry_mask(lidar_evidence)
    cond = {"context": context, "left_camera_k": left_camera_k, "gt_shift_x": batch["gt_shift_x"].to(device),
            "gt_shift_y": batch["gt_shift_y"].to(device), "theta": batch["theta"].to(device),
            "camera_to_lidar": camera_to_lidar, "lidar_context": lidar_context, "lidar_evidence": lidar_evidence,
            "lidar_geometry_mask": lidar_geometry_mask}
    return cond, rgb


def encode_latent(model, rgb):
    with torch.no_grad():
        posterior = model.pre_AE_model.encode(rgb * 2 - 1)
        latent = posterior.mode() if hasattr(posterior, "mode") else posterior.sample()
        return latent.detach() * float(model.scale_factor)


def load_base(args, device):
    from omegaconf import OmegaConf
    from utils.util import instantiate_from_config
    from tools.train_kitti_raea import configure_cfg
    cfg = configure_cfg(OmegaConf.load(args.config), training_ns(args))
    if not str(cfg.model.params.Lidar_context_config.target).endswith("LidarPixelConditionEncoder"):
        raise ValueError("persistent temporal entrypoint currently requires the V2.2 pixel LiDAR encoder")
    for split in ("train", "test"):
        params = cfg.data.params[split].params
        if not params.get("align_satellite_to_camera", True) or int(params.get("sat_size", 256)) != 256:
            raise ValueError("history satellite geometry requires camera-aligned 256px crops")
    if hasattr(cfg.model.params, "pre_ldm_model_path"):
        cfg.model.params.pre_ldm_model_path = None
    model = instantiate_from_config(cfg.model)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.DDPM.denoise_model.load_state_dict(payload["denoise_model"], strict=True)
    model.condition_model_sat.load_state_dict(payload["condition_model_sat"], strict=True)
    model.lidar_context_model.load_state_dict(payload["lidar_context_model"], strict=True)
    identity = base_identity(args.checkpoint, payload)
    del payload
    return model.to(device).eval(), identity, cfg
