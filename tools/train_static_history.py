#!/usr/bin/env python3
"""Bounded static-history adapter probe on original KITTI GPS30m splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.cuda.amp import GradScaler, autocast

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ldm.modules.temporal_amp import optimizer_step_with_retry
from ldm.modules.temporal_pair_training import (
    anchor_trainable_loss,
    base_identity,
    epsilon_prediction_loss,
    seed_training_step,
    state_dict_sha256,
)
from tools.infer_temporal import decode, sample_frame, tensor_hash, tree_to
from tools.temporal_static_geometry import build_static_pair
from tools.train_temporal_pairs import encode_conditions, encode_latent, load_base, stack_samples

CHECKPOINT_VERSION = "temporal_static_adapter_v1"
IMAGE_SIZE = (512, 128)
MONITOR_TIMESTEP = 500
CONDITION_POLICY = "ego_motion_depth_verified"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", required=True, help="JSON with existing train args under args or train_args")
    p.add_argument("--pairs-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:4")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--depth-tol-m", type=float, default=0.75)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args(argv)


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


def load_pair_entries(path: str | Path) -> List[dict]:
    payload = json.loads(Path(path).read_text())
    entries = payload.get("pairs") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("pairs JSON must contain a non-empty pairs list")
    names = [entry.get("name") for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("pair names must be unique")
    for entry in entries:
        if entry.get("split") not in {"train", "heldout"}:
            raise ValueError(f"pair {entry.get('name')} split must be train or heldout")
        if not entry.get("previous") or not entry.get("current"):
            raise ValueError(f"pair {entry.get('name')} is missing previous/current sample ids")
    return entries


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


def normalize_static_regions(regions: Mapping[str, Any] | None):
    if regions is None:
        return None
    prev = regions.get("prev_regions", regions.get("prev_rects", regions.get("prev", regions.get("reference"))))
    target = regions.get(
        "target_regions",
        regions.get("cur_regions", regions.get("target_rects", regions.get("cur_rects", regions.get("target", regions.get("current"))))),
    )
    if prev is None or target is None:
        raise ValueError("manual static regions require previous and target rectangles")
    return {"prev_regions": prev, "target_regions": target}


def stable_pair_seed(base_seed: int, name: str, stream: str) -> int:
    payload = f"{int(base_seed)}:{stream}:{name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 31)


def ddpm_num_timesteps(ddpm: Any) -> int:
    if hasattr(ddpm, "num_timesteps"):
        return int(ddpm.num_timesteps)
    if hasattr(ddpm, "timesteps"):
        return int(ddpm.timesteps)
    raise ValueError("DDPM object must expose num_timesteps or timesteps")


def index_dataset(dataset) -> Dict[str, int]:
    return {row["sample_id"]: idx for idx, row in enumerate(dataset.records)}


def manifest_key_for_split(split: str) -> str:
    if split == "train":
        return "train_manifest"
    if split == "heldout":
        return "val_manifest"
    raise ValueError("unknown pair split: " + str(split))


def row_for_entry_sample(indexes: Mapping[str, Mapping[str, Any]], entry: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    manifest_key = manifest_key_for_split(entry["split"])
    item = indexes[manifest_key]
    idx = item["index"].get(sample_id)
    if idx is None:
        raise KeyError(f"{sample_id} not found in declared {manifest_key} for pair {entry.get('name')}")
    return item["dataset"].records[idx]


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


def configure_static_adapter(model, hidden_dim: int):
    model.eval().requires_grad_(False)
    unet = model.DDPM.denoise_model
    if not hasattr(unet, "configure_temporal_history"):
        raise ValueError("UNet lacks configure_temporal_history")
    unet.configure_temporal_history(enabled=True, mode="static", hidden_dim=int(hidden_dim))
    params = []
    for name, parameter in unet.named_parameters():
        parameter.requires_grad_(name.startswith("temporal_history."))
        if parameter.requires_grad:
            params.append(parameter)
    if not params:
        raise ValueError("static adapter has no trainable parameters")
    return params


def frozen_unet_state(model) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.DDPM.denoise_model.state_dict().items()
        if not k.startswith("temporal_history.")
    }


def frozen_model_versions(model) -> Dict[str, str]:
    return {
        "denoise_without_adapter": state_dict_sha256(frozen_unet_state(model)),
        "condition_model_sat": state_dict_sha256(model.condition_model_sat.state_dict()),
        "lidar_context_model": state_dict_sha256(model.lidar_context_model.state_dict()),
        "pre_AE_model": state_dict_sha256(model.pre_AE_model.state_dict()),
    }


def static_history_to_tensors(geometry: Mapping[str, Any], previous_latent: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    strict = torch.as_tensor(geometry["strict_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    history = {
        "latent": previous_latent.detach().to(device),
        "static_rgb": torch.as_tensor(geometry["warped_rgb"], dtype=torch.float32, device=device).unsqueeze(0),
        "static_mask": strict,
        "static_confidence": strict.float(),
        "enabled": torch.ones((1,), dtype=torch.bool, device=device),
    }
    return history


def cat_history(items: Sequence[Mapping[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {key: torch.cat([item[key].to(device) for item in items], dim=0) for key in keys}


def tensor_to_pil(rgb: torch.Tensor) -> Image.Image:
    arr = rgb.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def numpy_rgb_to_pil(rgb_chw: np.ndarray) -> Image.Image:
    arr = np.clip(np.asarray(rgb_chw).transpose(1, 2, 0), 0, 1)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def gray_to_pil(values: np.ndarray) -> Image.Image:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size and arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    return Image.fromarray((np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8), mode="L").convert("RGB")


def make_panel(labels: Sequence[str], images: Sequence[Image.Image], path: Path) -> None:
    tile = IMAGE_SIZE
    label_h = 16
    panel = Image.new("RGB", (tile[0] * len(images), tile[1] + label_h), "white")
    draw = ImageDraw.Draw(panel)
    for idx, (label, image) in enumerate(zip(labels, images)):
        panel.paste(image.resize(tile, Image.NEAREST if image.mode == "L" else Image.BILINEAR), (idx * tile[0], label_h))
        draw.text((idx * tile[0] + 4, 2), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def roi_mask_from_geometry(geometry: Mapping[str, Any], prefer_strict: bool = True) -> torch.Tensor:
    key = "strict_mask" if prefer_strict and np.asarray(geometry["strict_mask"]).any() else "support_mask"
    return torch.as_tensor(geometry[key], dtype=torch.bool)


def masked_rgb_error(output: torch.Tensor, target: torch.Tensor, mask_1hw: torch.Tensor) -> float:
    mask = mask_1hw.to(output.device).bool()
    if int(mask.sum()) == 0:
        return float("nan")
    return float((output - target).abs().mean(0, keepdim=True)[mask].mean().detach().cpu())


def edge_proxy(output: torch.Tensor, mask_1hw: torch.Tensor) -> float:
    gray = output.detach().float().mean(0, keepdim=True)
    gx = (gray[:, :, 1:] - gray[:, :, :-1]).abs()
    gy = (gray[:, 1:, :] - gray[:, :-1, :]).abs()
    mask = mask_1hw.to(output.device).bool()
    mx = mask[:, :, 1:] & mask[:, :, :-1]
    my = mask[:, 1:, :] & mask[:, :-1, :]
    vals = []
    if int(mx.sum()):
        vals.append(gx[mx].mean())
    if int(my.sum()):
        vals.append(gy[my].mean())
    if not vals:
        return float("nan")
    return float(torch.stack(vals).mean().cpu())


def fixed_epsilon_monitor(model, item: Mapping[str, Any], seed: int, device: torch.device, use_history: bool,
                          timestep: int = MONITOR_TIMESTEP) -> Dict[str, float]:
    cond = tree_to(item["cond"], device)
    z_cur = item["z_cur"].to(device)
    history = tree_to(item["history"], device) if use_history else None
    gen = torch.Generator(device=device).manual_seed(int(seed))
    max_t = ddpm_num_timesteps(model.DDPM) - 1
    fixed_t = min(int(timestep), max_t)
    t = torch.full((z_cur.shape[0],), fixed_t, device=device, dtype=torch.long)
    noise = torch.randn(z_cur.shape, generator=gen, device=device, dtype=z_cur.dtype)
    x_noisy = model.DDPM.q_sample(x_start=z_cur, t=t, noise=noise)
    allowed = {
        "context", "lidar_context", "lidar_evidence", "lidar_geometry_mask",
        "control_grd", "left_camera_k", "gt_shift_x", "gt_shift_y", "theta",
    }
    denoise_cond = {key: value for key, value in cond.items() if key in allowed}
    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        pred = model.DDPM.denoise_model(x_noisy, t, history=history, **denoise_cond)
    return {"loss": float(F.mse_loss(pred.float(), noise.float()).cpu()), "pred_hash": tensor_hash(pred), "seed": int(seed), "timestep": fixed_t}


def single_epsilon_output(model, item: Mapping[str, Any], seed: int, device: torch.device, history_mode: str) -> torch.Tensor:
    cond = tree_to(item["cond"], device)
    z_cur = item["z_cur"].to(device)
    history = tree_to(item["history"], device) if history_mode == "correct" else None
    gen = torch.Generator(device=device).manual_seed(int(seed))
    t = torch.randint(0, ddpm_num_timesteps(model.DDPM), (z_cur.shape[0],), generator=gen, device=device).long()
    noise = torch.randn(z_cur.shape, generator=gen, device=device, dtype=z_cur.dtype)
    x_noisy = model.DDPM.q_sample(x_start=z_cur, t=t, noise=noise)
    allowed = {"context", "lidar_context", "lidar_evidence", "lidar_geometry_mask", "control_grd", "left_camera_k", "gt_shift_x", "gt_shift_y", "theta"}
    denoise_cond = {key: value for key, value in cond.items() if key in allowed}
    with torch.no_grad(), autocast(enabled=device.type == "cuda"):
        return model.DDPM.denoise_model(x_noisy, t, history=history, **denoise_cond).detach().float().cpu()


def trainable_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in params:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return total ** 0.5


def named_adapter_grad_norms(model) -> Dict[str, float]:
    reader = model.DDPM.denoise_model.temporal_history
    out = {"encoder_grad_norm": 0.0, "output_grad_norm": 0.0}
    for name, parameter in reader.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().square().sum().cpu())
        if name.startswith("encoder."):
            out["encoder_grad_norm"] += value
        elif name.startswith("output."):
            out["output_grad_norm"] += value
    return {key: value ** 0.5 for key, value in out.items()}


def prepare_cache(model, cfg, settings: Namespace, entries: Sequence[Mapping[str, Any]], device: torch.device, depth_tol_m: float) -> List[dict]:
    from utils.util import instantiate_from_config

    datasets = {
        "train_manifest": instantiate_from_config(cfg.data.params.train),
        "val_manifest": instantiate_from_config(cfg.data.params.test),
    }
    indexes = {name: {"dataset": dataset, "index": index_dataset(dataset)} for name, dataset in datasets.items()}
    cache = []
    model.eval()
    for entry in entries:
        split_dataset = indexes[manifest_key_for_split(entry["split"])]["dataset"]
        prev_row = dict(row_for_entry_sample(indexes, entry, entry["previous"]))
        cur_row = dict(row_for_entry_sample(indexes, entry, entry["current"]))
        geometry = build_static_pair(
            prev_row,
            cur_row,
            kitti_root=settings.kitti_root,
            image_size=IMAGE_SIZE,
            depth_tol_m=depth_tol_m,
            static_regions=normalize_static_regions(entry.get("static_regions")),
        )
        diagnostics = dict(geometry["diagnostics"])
        if int(diagnostics.get("support_count", 0)) <= 0:
            raise RuntimeError(f"{entry['name']} has empty static support")
        if int(diagnostics.get("strict_count", 0)) <= 0:
            raise RuntimeError(f"{entry['name']} has empty strong strict support")
        cur_sample = sample_from_entry_dataset(indexes, entry, entry["current"])
        prev_sample = previous_rgb_sample(split_dataset, prev_row)
        cur_batch = stack_samples([cur_sample])
        prev_batch = stack_samples([prev_sample])
        with torch.no_grad():
            cond, cur_rgb = encode_conditions(model, cur_batch, device)
            z_cur = encode_latent(model, cur_rgb)
            prev_rgb = model.get_input({k: v.to(device) if torch.is_tensor(v) else v for k, v in prev_batch.items()}, "grd_left_imgs").clamp(0, 1)
            z_prev = encode_latent(model, prev_rgb)
        history = static_history_to_tensors(geometry, z_prev, device)
        cache.append(
            {
                "name": entry["name"],
                "split": entry["split"],
                "previous": entry["previous"],
                "current": entry["current"],
                "annotation_source": entry.get("annotation_source", ""),
                "cond": tree_to(cond, torch.device("cpu")),
                "z_cur": z_cur.detach().cpu(),
                "z_prev": z_prev.detach().cpu(),
                "history": tree_to(history, torch.device("cpu")),
                "cur_rgb": cur_rgb.detach().cpu(),
                "prev_rgb": prev_rgb.detach().cpu(),
                "geometry": geometry,
                "diagnostics": diagnostics,
            }
        )
    return cache


def batch_from_cache(items: Sequence[Mapping[str, Any]], device: torch.device):
    cond = recursive_batch_to_device([item["cond"] for item in items], device)
    z_cur = torch.cat([item["z_cur"] for item in items], dim=0).to(device)
    history = cat_history([item["history"] for item in items], device)
    return cond, z_cur, history


def recursive_batch_to_device(values: Sequence[Any], device: torch.device):
    if not values:
        raise ValueError("cannot batch an empty value list")
    first = values[0]
    if torch.is_tensor(first):
        if not all(torch.is_tensor(value) for value in values):
            raise ValueError("batch structure mismatch: tensor mixed with non-tensor")
        return torch.cat([value.to(device) for value in values], dim=0)
    if first is None:
        if any(value is not None for value in values):
            raise ValueError("batch structure mismatch: None mixed with value")
        return None
    if isinstance(first, dict):
        keys = set(first)
        for value in values:
            if not isinstance(value, dict) or set(value) != keys:
                raise ValueError("batch structure mismatch: dict keys differ")
        return {key: recursive_batch_to_device([value[key] for value in values], device) for key in first}
    if isinstance(first, list):
        length = len(first)
        for value in values:
            if not isinstance(value, list) or len(value) != length:
                raise ValueError("batch structure mismatch: list lengths differ")
        return [recursive_batch_to_device([value[index] for value in values], device) for index in range(length)]
    if isinstance(first, tuple):
        length = len(first)
        for value in values:
            if not isinstance(value, tuple) or len(value) != length:
                raise ValueError("batch structure mismatch: tuple lengths differ")
        return tuple(recursive_batch_to_device([value[index] for value in values], device) for index in range(length))
    if all(value == first for value in values):
        return first
    raise ValueError("batch structure mismatch for non-tensor leaf")


def evaluate_monitor(model, cache: Sequence[Mapping[str, Any]], seed: int, device: torch.device, tag: str, out_dir: Path) -> List[dict]:
    rows = []
    selected = [item for item in cache if item["split"] == "train"][:2] + [item for item in cache if item["split"] == "heldout"][:2]
    for item in selected:
        pair_seed = stable_pair_seed(seed, item["name"], "monitor")
        for mode, use_history in (("off", False), ("correct", True)):
            rec = fixed_epsilon_monitor(model, item, pair_seed, device, use_history)
            rows.append({"phase": tag, "name": item["name"], "split": item["split"], "mode": mode, **rec})
    path = out_dir / "monitor.jsonl"
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def render_samples(model, cache: Sequence[Mapping[str, Any]], seed: int, device: torch.device, tag: str, out_dir: Path, steps: int, guidance: float) -> List[dict]:
    rows = []
    selected = [item for item in cache if item["split"] == "train"][:2] + [item for item in cache if item["split"] == "heldout"][:2]
    for item in selected:
        pair_seed = stable_pair_seed(seed, item["name"], "sample")
        item_dir = out_dir / "samples" / tag / item["name"]
        item_dir.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(item["prev_rgb"][0]).save(item_dir / "prev_gt.png")
        tensor_to_pil(item["cur_rgb"][0]).save(item_dir / "current_gt.png")
        numpy_rgb_to_pil(item["geometry"]["warped_rgb"]).save(item_dir / "warped_rgb.png")
        gray_to_pil(item["geometry"]["support_mask"][0].astype(np.float32)).save(item_dir / "support_mask.png")
        outputs = {}
        for mode, history in (("off", None), ("correct", item["history"])):
            z, _info, noise_hash = sample_frame(
                model,
                item["cond"],
                tuple(item["z_cur"].shape),
                pair_seed,
                device,
                history=history if history is None else tree_to(history, device),
                steps=steps,
                guidance=guidance,
            )
            rgb = decode(model, z).detach().float().cpu()[0]
            outputs[mode] = rgb
            tensor_to_pil(rgb).save(item_dir / f"{mode}.png")
            mask = roi_mask_from_geometry(item["geometry"])
            rows.append(
                {
                    "phase": tag,
                    "name": item["name"],
                    "split": item["split"],
                    "mode": mode,
                    "seed": pair_seed,
                    "noise_hash": noise_hash,
                    "roi_rgb_l1": masked_rgb_error(rgb, item["cur_rgb"][0], mask),
                    "roi_edge_proxy": edge_proxy(rgb, mask),
                }
            )
        make_panel(
            ["prevGT", "currentGT", "warped", "support", "off", "correct"],
            [
                tensor_to_pil(item["prev_rgb"][0]),
                tensor_to_pil(item["cur_rgb"][0]),
                numpy_rgb_to_pil(item["geometry"]["warped_rgb"]),
                gray_to_pil(item["geometry"]["support_mask"][0].astype(np.float32)),
                tensor_to_pil(outputs["off"]),
                tensor_to_pil(outputs["correct"]),
            ],
            item_dir / "comparison_panel.png",
        )
    with (out_dir / "sample_metrics.jsonl").open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def save_inference_payloads(cache: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    payload_dir = out_dir / "inference_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    selected = [item for item in cache if item["split"] == "train"][:2] + [item for item in cache if item["split"] == "heldout"][:2]
    for item in selected:
        path = payload_dir / f"{item['name']}.pt"
        history = dict(item["history"])
        payload = {
            "kwargs": item["cond"],
            "shape": tuple(item["z_prev"].shape),
            "history": history,
            "metadata": {
                "name": item["name"],
                "split": item["split"],
                "previous": item["previous"],
                "current": item["current"],
                "condition_policy": CONDITION_POLICY,
                "contains_target_gt": False,
                "contains_target_latent": False,
            },
        }
        torch.save(payload, path)
        manifest.append({"name": item["name"], "split": item["split"], "payload": str(path)})
    for split in ("train", "heldout"):
        batch_items = [item for item in cache if item["split"] == split][:2]
        if len(batch_items) < 2:
            continue
        cond, _z_cur, history = batch_from_cache(batch_items, torch.device("cpu"))
        z_prev = torch.cat([item["z_prev"] for item in batch_items], dim=0)
        path = payload_dir / f"{split}_batch2.pt"
        torch.save(
            {
                "kwargs": cond,
                "shape": tuple(z_prev.shape),
                "history": history,
                "metadata": {
                    "names": [item["name"] for item in batch_items],
                    "split": split,
                    "condition_policy": CONDITION_POLICY,
                    "contains_target_gt": False,
                    "contains_target_latent": False,
                },
            },
            path,
        )
        manifest.append({"name": f"{split}_batch2", "split": split, "payload": str(path)})
    (payload_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def save_static_checkpoint(path: Path, model, optimizer, step: int, base: Mapping[str, Any], args: Mapping[str, Any]) -> None:
    reader = model.DDPM.denoise_model.temporal_history
    payload = {
        "version": CHECKPOINT_VERSION,
        "artifact_kind": "temporal_static_adapter",
        "base_checkpoint": dict(base),
        "state_dict": {k: v.detach().cpu() for k, v in reader.state_dict().items()},
        "hidden_dim": int(reader.hidden_dim),
        "step": int(step),
        "args": dict(args),
        "geometry_config": {
            "image_size": list(IMAGE_SIZE),
            "depth_tol_m": float(args["depth_tol_m"]),
            "dependency": "sensor_ego_motion_lidar_depth",
            "condition_policy": CONDITION_POLICY,
            "vehicle_association_available": False,
        },
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main(argv=None):
    args = parse_args(argv)
    if not 1 <= int(args.steps) <= 500:
        raise ValueError("--steps must be in [1,500] for this bounded probe")
    if min(args.batch_size, args.log_every, args.ddim_steps) < 1:
        raise ValueError("batch/log/sample step counts must be positive")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "checkpoints").mkdir()
    device = evaluation_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    seed_training_step(args.seed, device)

    settings = load_settings_args(args.settings)
    settings.batch_size = args.batch_size
    entries = load_pair_entries(args.pairs_json)
    split_info = validate_pair_splits(entries)
    model, base, cfg = load_base(settings, device)
    config_text = Path(settings.config).read_text()
    from omegaconf import OmegaConf
    OmegaConf.save(cfg, out_dir / "cfg_resolved.yaml", resolve=True)
    (out_dir / "source_settings.json").write_text(json.dumps(json.loads(Path(args.settings).read_text()), indent=2, sort_keys=True))
    (out_dir / "source_pairs.json").write_text(json.dumps(json.loads(Path(args.pairs_json).read_text()), indent=2, sort_keys=True))
    (out_dir / "source_config.yaml").write_text(config_text)
    frozen_versions_before = frozen_model_versions(model)
    params = configure_static_adapter(model, args.hidden_dim)
    optimizer = torch.optim.AdamW(params, lr=float(args.lr))
    scaler = GradScaler(enabled=device.type == "cuda")
    cache = prepare_cache(model, cfg, settings, entries, device, args.depth_tol_m)
    model_args = {
        "settings": str(args.settings),
        "pairs_json": str(args.pairs_json),
        "out_dir": str(out_dir),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "ddim_steps": args.ddim_steps,
        "guidance": args.guidance,
        "depth_tol_m": args.depth_tol_m,
        "hidden_dim": args.hidden_dim,
        "monitor_seed": args.seed + 101,
        "sampling_seed": args.seed + 201,
        "monitor_timestep": MONITOR_TIMESTEP,
        "condition_policy": CONDITION_POLICY,
    }
    (out_dir / "args.json").write_text(json.dumps({"args": model_args, "base_checkpoint": base, "split_info": split_info, "frozen_versions_before": frozen_versions_before}, indent=2, sort_keys=True))
    geometry_rows = []
    for item in cache:
        row = {k: item[k] for k in ("name", "split", "previous", "current", "annotation_source")}
        row["diagnostics"] = item["diagnostics"]
        geometry_rows.append(row)
    (out_dir / "geometry_diagnostics.json").write_text(json.dumps(geometry_rows, indent=2, sort_keys=True))
    print(json.dumps({"cache_pairs": len(cache), "split_info": split_info, "trainable_params": sum(p.numel() for p in params)}, sort_keys=True), flush=True)

    first = cache[0]
    model.eval()
    no_adapter = model.DDPM.denoise_model.temporal_history
    model.DDPM.denoise_model.temporal_history = None
    baseline_eps = single_epsilon_output(model, first, args.seed + 11, device, "off")
    model.DDPM.denoise_model.temporal_history = no_adapter
    off_eps = single_epsilon_output(model, first, args.seed + 11, device, "off")
    correct_eps = single_epsilon_output(model, first, args.seed + 11, device, "correct")
    torch.testing.assert_close(baseline_eps, off_eps, rtol=0, atol=0)
    torch.testing.assert_close(off_eps, correct_eps, rtol=1e-5, atol=1e-6)
    before_off_z, _, before_off_noise = sample_frame(
        model, first["cond"], tuple(first["z_cur"].shape), args.seed + 23, device, None, args.ddim_steps, args.guidance
    )
    before_correct_z, _, before_correct_noise = sample_frame(
        model, first["cond"], tuple(first["z_cur"].shape), args.seed + 23, device, tree_to(first["history"], device), args.ddim_steps, args.guidance
    )
    if before_off_noise != before_correct_noise:
        raise RuntimeError("fixed-noise preflight sample did not use matching noise")
    torch.testing.assert_close(before_off_z, before_correct_z, rtol=1e-5, atol=1e-6)
    preflight = {
        "single_epsilon_baseline_off_close": True,
        "single_epsilon_off_correct_close": True,
        "full_sample_zero_init_close": True,
        "noise_hash": before_off_noise,
    }
    (out_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True))

    monitor_seed = args.seed + 101
    sampling_seed = args.seed + 201
    evaluate_monitor(model, cache, monitor_seed, device, "step_000000", out_dir)
    render_samples(model, cache, sampling_seed, device, "before", out_dir, args.ddim_steps, args.guidance)
    save_inference_payloads(cache, out_dir)

    train_items = [item for item in cache if item["split"] == "train"]
    metrics_path = out_dir / "metrics.jsonl"
    t0 = time.time()
    model.eval()
    model.DDPM.denoise_model.temporal_history.train()
    for step in range(1, args.steps + 1):
        rng = random.Random(args.seed + step * 7919)
        batch_items = [train_items[rng.randrange(len(train_items))] for _ in range(args.batch_size)]
        cond, z_cur, history = batch_from_cache(batch_items, device)
        step_seed = args.seed + step * 1009
        seed_training_step(step_seed, device)
        sat_probability = float(model.satellite_condition_dropout_prob)
        sat_drop = torch.rand((z_cur.shape[0], 1, 1), device=device) < sat_probability
        cond = dict(cond)
        cond["context"] = cond["context"] * (~sat_drop)

        def loss_closure():
            with autocast(enabled=device.type == "cuda"):
                loss, out = epsilon_prediction_loss(model.DDPM, z_cur, cond, history, step_seed)
                return anchor_trainable_loss(loss, params), out

        loss, out, grad_norm, amp_retries = optimizer_step_with_retry(loss_closure, params, optimizer, scaler)
        reader_metrics = {
            key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            for key, value in getattr(model.DDPM.denoise_model.temporal_history, "last_metrics", {}).items()
        }
        grad_parts = named_adapter_grad_norms(model)
        if step > 1 and (grad_parts["output_grad_norm"] <= 0.0 or grad_parts["encoder_grad_norm"] <= 0.0):
            raise RuntimeError("static adapter gradients are zero after first update")
        rec = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "static_grad_norm": trainable_grad_norm(params),
            "amp_scale": scaler.get_scale(),
            "amp_retries": amp_retries,
            "t_mean": float(out["t"].float().mean().cpu()),
            "satellite_dropout_fraction": float(sat_drop.float().mean().cpu()),
            "batch_names": [item["name"] for item in batch_items],
            "sec": round(time.time() - t0, 2),
            **grad_parts,
            **reader_metrics,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(rec, sort_keys=True), flush=True)
        if step == args.steps // 2:
            model.DDPM.denoise_model.temporal_history.eval()
            evaluate_monitor(model, cache, monitor_seed, device, f"step_{step:06d}", out_dir)
            model.DDPM.denoise_model.temporal_history.train()

    model.eval()
    evaluate_monitor(model, cache, monitor_seed, device, f"step_{args.steps:06d}", out_dir)
    render_samples(model, cache, sampling_seed, device, "after", out_dir, args.ddim_steps, args.guidance)
    after_off_z, _, after_off_noise = sample_frame(
        model, first["cond"], tuple(first["z_cur"].shape), args.seed + 23, device, None, args.ddim_steps, args.guidance
    )
    if after_off_noise != before_off_noise:
        raise RuntimeError("post-training OFF sample noise hash changed")
    torch.testing.assert_close(before_off_z, after_off_z, rtol=1e-5, atol=1e-6)
    frozen_versions_after = frozen_model_versions(model)
    if frozen_versions_before != frozen_versions_after:
        raise RuntimeError("frozen base parameters changed")
    save_static_checkpoint(out_dir / "checkpoints" / f"static_adapter_step_{args.steps:07d}.pt", model, optimizer, args.steps, base, model_args)
    (out_dir / "done.json").write_text(
        json.dumps(
            {
                "done": True,
                "steps": args.steps,
                "off_sample_unchanged_after_train": True,
                "frozen_versions_after": frozen_versions_after,
                "geometry_dependency": "sensor_ego_motion_lidar_depth",
                "condition_policy": CONDITION_POLICY,
                "vehicle_association_available": False,
                "vehicle_association_note": "No persistent vehicle track IDs or instance associations are used by this static probe.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps({"done": True, "steps": args.steps, "out_dir": str(out_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
