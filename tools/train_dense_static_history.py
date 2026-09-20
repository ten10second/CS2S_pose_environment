#!/usr/bin/env python3
"""Stage-2 bounded dense/static history adapter probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ldm.modules.temporal_amp import optimizer_step_with_retry
from ldm.modules.temporal_pair_training import anchor_trainable_loss, epsilon_prediction_loss, seed_training_step, state_dict_sha256
from tools.infer_temporal import decode, sample_frame, tensor_hash, tree_to
from tools.train_static_history import (
    IMAGE_SIZE,
    batch_from_cache,
    ddpm_num_timesteps,
    evaluation_device,
    fixed_epsilon_monitor,
    frozen_model_versions,
    load_pair_entries,
    load_settings_args,
    prepare_cache,
    recursive_batch_to_device,
    stable_pair_seed,
    trainable_grad_norm,
    validate_pair_splits,
)
from tools.train_temporal_pairs import load_base

CHECKPOINT_VERSION = "temporal_static_adapter_v2"
MODEL_MODE = "static_dense"
EVAL_TIMESTEPS = (100, 500, 900)
EVAL_STEPS = (0, 250, 500, 1000)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", required=True)
    p.add_argument("--pairs-json", required=True)
    p.add_argument("--reference-root", required=True, help="Directory containing <pair-name>/reference.npz")
    p.add_argument("--reference-kind", choices=["sparse", "dense"], required=True)
    p.add_argument("--input-variant", choices=["rgb", "valid", "types"], required=True)
    p.add_argument("--model-mode", choices=["static_dense", "static_adaptive"], default="static_dense")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:4")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--depth-tol-m", type=float, default=0.75)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args(argv)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_scalar_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.tolist())


def mask_tensor(array: np.ndarray, name: str) -> torch.Tensor:
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be [H,W] or [H,W,1]")
    if arr.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0]) or arr.dtype != np.bool_:
        raise ValueError(f"{name} must be bool at the reference image resolution")
    return torch.as_tensor(arr.astype(bool), dtype=torch.bool).unsqueeze(0).unsqueeze(0)


def rgb_tensor(array: np.ndarray, valid: torch.Tensor, name: str) -> torch.Tensor:
    arr = np.asarray(array, dtype=np.float32)
    if arr.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError(f"{name} must be HWC RGB with shape {(IMAGE_SIZE[1], IMAGE_SIZE[0], 3)}")
    if not np.isfinite(arr).all() or arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError(f"{name} must be finite RGB in [0,1]")
    rgb = torch.as_tensor(arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    if torch.any(rgb.masked_select(~valid.expand_as(rgb)) != 0):
        raise ValueError(f"{name} must be zero outside valid/support mask")
    return rgb


def dense_history_from_arrays(z_prev: torch.Tensor, rgb: torch.Tensor, valid: torch.Tensor,
                              measured: torch.Tensor, estimated: torch.Tensor) -> dict:
    if valid.shape != measured.shape or valid.shape != estimated.shape:
        raise ValueError("dense masks must share shape")
    if torch.any(measured & estimated):
        raise ValueError("dense measured and estimated masks must be disjoint")
    if not torch.equal(valid, measured | estimated):
        raise ValueError("dense valid must equal measured|estimated")
    return {
        "latent": z_prev.detach().cpu(),
        "dense_rgb": rgb.cpu(),
        "dense_valid": valid.cpu(),
        "dense_measured": measured.cpu(),
        "dense_estimated": estimated.cpu(),
        "enabled": torch.ones((z_prev.shape[0],), dtype=torch.bool),
    }


def sparse_dense_history(item: Mapping[str, Any]) -> dict:
    strict = mask_tensor(item["geometry"]["strict_mask"][0], "strict_mask")
    rgb = torch.as_tensor(item["geometry"]["warped_rgb"], dtype=torch.float32).unsqueeze(0)
    rgb = rgb * strict.float()
    empty = torch.zeros_like(strict)
    return dense_history_from_arrays(item["z_prev"], rgb, strict, strict, empty)


def load_reference_npz(path: Path, entry: Mapping[str, Any], z_prev: torch.Tensor) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    required = ("warped_rgb", "support_mask", "measured_mask", "estimated_mask")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    prev_keys = [key for key in ("previous", "prev", "previous_sample_id") if key in data]
    cur_keys = [key for key in ("current", "cur", "current_sample_id", "target_sample_id") if key in data]
    if not prev_keys or not cur_keys:
        raise ValueError(f"{path} must record previous/current identity")
    if npz_scalar_text(data[prev_keys[0]]) != entry["previous"]:
        raise ValueError(f"{path} previous identity mismatch")
    if npz_scalar_text(data[cur_keys[0]]) != entry["current"]:
        raise ValueError(f"{path} current identity mismatch")
    valid = mask_tensor(data["support_mask"], "support_mask")
    measured = mask_tensor(data["measured_mask"], "measured_mask")
    estimated = mask_tensor(data["estimated_mask"], "estimated_mask")
    rgb = rgb_tensor(data["warped_rgb"], valid, "warped_rgb")
    history = dense_history_from_arrays(z_prev, rgb, valid, measured, estimated)
    meta = {
        "path": str(path),
        "sha256": file_sha256(path),
        "valid_pixels": int(valid.sum()),
        "measured_pixels": int(measured.sum()),
        "estimated_pixels": int(estimated.sum()),
    }
    return history, meta


def replace_histories(cache: Sequence[dict], entries: Sequence[Mapping[str, Any]], reference_root: str | Path, reference_kind: str) -> list[dict]:
    entries_by_name = {entry["name"]: entry for entry in entries}
    out = []
    for item in cache:
        copy = dict(item)
        if reference_kind == "sparse":
            copy["history"] = sparse_dense_history(item)
            copy["reference_meta"] = {"reference_kind": "sparse", "source": "geometry.strict_mask"}
        else:
            entry = entries_by_name[item["name"]]
            history, meta = load_reference_npz(Path(reference_root) / item["name"] / "reference.npz", entry, item["z_prev"])
            copy["history"] = history
            copy["reference_meta"] = {"reference_kind": "dense", **meta}
        out.append(copy)
    return out


def configure_dense_adapter(model, hidden_dim: int, input_variant: str, model_mode: str = MODEL_MODE):
    model.eval().requires_grad_(False)
    unet = model.DDPM.denoise_model
    unet.configure_temporal_history(enabled=True, mode=model_mode, hidden_dim=int(hidden_dim), input_variant=input_variant)
    params = []
    for name, parameter in unet.named_parameters():
        parameter.requires_grad_(name.startswith("temporal_history."))
        if parameter.requires_grad:
            params.append(parameter)
    if not params:
        raise ValueError("static_dense adapter has no trainable parameters")
    return params


def named_adapter_grad_norms(model):
    reader = model.DDPM.denoise_model.temporal_history
    sums = {"encoder_grad_norm": 0.0, "output_grad_norm": 0.0, "fusion_grad_norm": 0.0}
    for name, parameter in reader.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().square().sum().cpu())
        if name.startswith("encoder."):
            sums["encoder_grad_norm"] += value
        elif name.startswith(("output.", "fusion.output.")):
            sums["output_grad_norm"] += value
        elif name.startswith("fusion."):
            sums["fusion_grad_norm"] += value
    return {key: value ** 0.5 for key, value in sums.items()}


def adapter_state_hash(model) -> str:
    return state_dict_sha256(model.DDPM.denoise_model.temporal_history.state_dict())


def save_dense_checkpoint(path: Path, model, optimizer, step: int, base: Mapping[str, Any], args: Mapping[str, Any]) -> None:
    reader = model.DDPM.denoise_model.temporal_history
    payload = {
        "version": "temporal_static_adapter_v3" if reader.mode == "static_adaptive" else CHECKPOINT_VERSION,
        "artifact_kind": "temporal_static_dense_adapter",
        "base_checkpoint": dict(base),
        "state_dict": {k: v.detach().cpu() for k, v in reader.state_dict().items()},
        "hidden_dim": int(reader.hidden_dim),
        "input_variant": reader.input_variant,
        "model_mode": reader.mode,
        "reference_kind": args["reference_kind"],
        "adapter_state_sha256": adapter_state_hash(model),
        "step": int(step),
        "args": dict(args),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    if reader.mode == "static_adaptive":
        payload.update(fusion_dim=reader.fusion_dim, time_embed_dim=reader.time_embed_dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def monitor_all_pairs(model, cache: Sequence[Mapping[str, Any]], base_seed: int, device: torch.device,
                      phase: str, out_dir: Path) -> None:
    path = out_dir / "monitor.jsonl"
    with path.open("a") as handle:
        for item in cache:
            for timestep in EVAL_TIMESTEPS:
                seed = stable_pair_seed(base_seed, item["name"], f"monitor_t{timestep}")
                for mode, use_history in (("off", False), ("correct", True)):
                    rec = fixed_epsilon_monitor(model, item, seed, device, use_history, timestep=timestep)
                    rec.update(phase=phase, name=item["name"], split=item["split"], mode=mode)
                    handle.write(json.dumps(rec, sort_keys=True) + "\n")


def save_dense_inference_payloads(cache: Sequence[Mapping[str, Any]], out_dir: Path, metadata: Mapping[str, Any]) -> None:
    payload_dir = out_dir / "inference_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in cache:
        path = payload_dir / f"{item['name']}.pt"
        torch.save({
            "kwargs": item["cond"],
            "shape": tuple(item["z_prev"].shape),
            "history": item["history"],
            "metadata": {
                "name": item["name"],
                "split": item["split"],
                "previous": item["previous"],
                "current": item["current"],
                "contains_target_gt": False,
                "contains_target_latent": False,
                **metadata,
            },
        }, path)
        manifest.append({"name": item["name"], "split": item["split"], "payload": str(path)})
    (payload_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def save_rgb(path: Path, rgb: torch.Tensor) -> None:
    from PIL import Image
    arr = rgb.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)


def render_all_after(model, cache: Sequence[Mapping[str, Any]], seed: int, device: torch.device,
                     out_dir: Path, steps: int, guidance: float) -> None:
    sample_dir = out_dir / "samples" / "after"
    rows = []
    for item in cache:
        item_dir = sample_dir / item["name"]
        item_dir.mkdir(parents=True, exist_ok=True)
        for mode, history in (("off", None), ("correct", item["history"])):
            z, _info, noise_hash = sample_frame(
                model, item["cond"], tuple(item["z_prev"].shape), stable_pair_seed(seed, item["name"], "sample"),
                device, history=None if history is None else tree_to(history, device), steps=steps, guidance=guidance)
            rgb = decode(model, z).detach().float().cpu()[0]
            save_rgb(item_dir / f"{mode}.png", rgb)
            rows.append({"name": item["name"], "split": item["split"], "mode": mode, "noise_hash": noise_hash,
                         "seed": stable_pair_seed(seed, item["name"], "sample")})
    with (out_dir / "sample_metrics.jsonl").open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None):
    args = parse_args(argv)
    if not 1 <= int(args.steps) <= 2000:
        raise ValueError("--steps must be in [1,2000]")
    if min(args.batch_size, args.log_every, args.ddim_steps) < 1 or args.lr <= 0:
        raise ValueError("batch/log/sample counts and learning rate must be positive")
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
    from omegaconf import OmegaConf
    OmegaConf.save(cfg, out_dir / "cfg_resolved.yaml", resolve=True)
    frozen_before = frozen_model_versions(model)
    params = configure_dense_adapter(model, args.hidden_dim, args.input_variant, args.model_mode)
    init_hash = adapter_state_hash(model)
    cache = prepare_cache(model, cfg, settings, entries, device, args.depth_tol_m)
    cache = replace_histories(cache, entries, args.reference_root, args.reference_kind)
    metadata = {
        "reference_kind": args.reference_kind,
        "input_variant": args.input_variant,
        "model_mode": args.model_mode,
        "adapter_init_sha256": init_hash,
    }
    model_args = vars(args).copy()
    model_args.update(metadata)
    (out_dir / "args.json").write_text(json.dumps({"args": model_args, "base_checkpoint": base, "split_info": split_info,
                                                   "frozen_versions_before": frozen_before}, indent=2, sort_keys=True))
    (out_dir / "source_pairs.json").write_text(json.dumps(json.loads(Path(args.pairs_json).read_text()), indent=2, sort_keys=True))
    (out_dir / "source_settings.json").write_text(json.dumps(json.loads(Path(args.settings).read_text()), indent=2, sort_keys=True))
    reference_rows = []
    for item in cache:
        row = {k: item[k] for k in ("name", "split", "previous", "current")}
        row["reference"] = item["reference_meta"]
        reference_rows.append(row)
    (out_dir / "reference_manifest.json").write_text(json.dumps(reference_rows, indent=2, sort_keys=True))
    save_dense_inference_payloads(cache, out_dir, metadata)

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = GradScaler(enabled=device.type == "cuda")
    model.eval()
    first = cache[0]
    before_off_z, _info, before_off_noise = sample_frame(
        model, first["cond"], tuple(first["z_prev"].shape), args.seed + 23, device, None, args.ddim_steps, args.guidance)
    before_on_z, _, before_on_noise = sample_frame(
        model, first["cond"], tuple(first["z_prev"].shape), args.seed + 23, device,
        first["history"], args.ddim_steps, args.guidance)
    if before_on_noise != before_off_noise:
        raise RuntimeError("zero-init sample noise differs")
    torch.testing.assert_close(before_on_z, before_off_z, rtol=1e-5, atol=1e-6)
    off = fixed_epsilon_monitor(model, first, args.seed + 17, device, False, timestep=500)
    correct = fixed_epsilon_monitor(model, first, args.seed + 17, device, True, timestep=500)
    if off["pred_hash"] != correct["pred_hash"]:
        raise RuntimeError("zero-init dense adapter changed epsilon preflight")
    monitor_all_pairs(model, cache, args.seed + 101, device, "step_000000", out_dir)

    train_items = [item for item in cache if item["split"] == "train"]
    metrics_path = out_dir / "metrics.jsonl"
    eval_steps = {step for step in EVAL_STEPS if 0 < step <= args.steps}
    eval_steps.add(args.steps)
    t0 = time.time()
    model.DDPM.denoise_model.temporal_history.train()
    for step in range(1, args.steps + 1):
        rng = random.Random(args.seed + step * 7919)
        batch_items = [train_items[rng.randrange(len(train_items))] for _ in range(args.batch_size)]
        cond, z_cur, history = batch_from_cache(batch_items, device)
        step_seed = args.seed + step * 1009
        seed_training_step(step_seed, device)
        sat_drop = torch.rand((z_cur.shape[0], 1, 1), device=device) < float(model.satellite_condition_dropout_prob)
        cond = dict(cond)
        cond["context"] = cond["context"] * (~sat_drop)

        def loss_closure():
            with autocast(enabled=device.type == "cuda"):
                loss, out = epsilon_prediction_loss(model.DDPM, z_cur, cond, history, step_seed)
                return anchor_trainable_loss(loss, params), out

        loss, out, grad_norm, amp_retries = optimizer_step_with_retry(loss_closure, params, optimizer, scaler)
        grad_parts = named_adapter_grad_norms(model)
        if step > 1 and (grad_parts["output_grad_norm"] <= 0.0 or grad_parts["encoder_grad_norm"] <= 0.0):
            raise RuntimeError("history adapter gradients are zero after first update")
        if step > 1 and args.model_mode == "static_adaptive" and grad_parts["fusion_grad_norm"] <= 0:
            raise RuntimeError("adaptive fusion gradients are zero after first update")
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
            **{k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
               for k, v in getattr(model.DDPM.denoise_model.temporal_history, "last_metrics", {}).items()},
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(rec, sort_keys=True), flush=True)
        if step in eval_steps:
            model.DDPM.denoise_model.temporal_history.eval()
            monitor_all_pairs(model, cache, args.seed + 101, device, f"step_{step:06d}", out_dir)
            model.DDPM.denoise_model.temporal_history.train()

    model.eval()
    after_off_z, _info, after_off_noise = sample_frame(
        model, first["cond"], tuple(first["z_prev"].shape), args.seed + 23, device, None, args.ddim_steps, args.guidance)
    if before_off_noise != after_off_noise:
        raise RuntimeError("post-training OFF sample noise hash changed")
    torch.testing.assert_close(before_off_z, after_off_z, rtol=1e-5, atol=1e-6)
    render_all_after(model, cache, args.seed, device, out_dir, args.ddim_steps, args.guidance)
    frozen_after = frozen_model_versions(model)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen base parameters changed")
    ckpt = out_dir / "checkpoints" / f"static_dense_adapter_step_{args.steps:07d}.pt"
    save_dense_checkpoint(ckpt, model, optimizer, args.steps, base, model_args)
    (out_dir / "done.json").write_text(json.dumps({"done": True, "steps": args.steps, "checkpoint": str(ckpt),
                                                   "frozen_versions_after": frozen_after, **metadata}, indent=2, sort_keys=True))
    print(json.dumps({"done": True, "steps": args.steps, "out_dir": str(out_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
