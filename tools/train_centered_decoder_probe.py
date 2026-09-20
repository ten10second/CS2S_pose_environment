#!/usr/bin/env python3
"""ABC small-sample centered-history decoder probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.cuda.amp import GradScaler, autocast

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ldm.modules.paired_color import sample_pair_color, apply_pair_color, warp_augmented_rgb
from ldm.modules.temporal_amp import optimizer_step_with_retry
from ldm.modules.temporal_pair_training import assert_same_base, epsilon_prediction_loss, seed_training_step, state_dict_sha256
from tools.infer_temporal import decode, sample_frame, tensor_hash
from tools.train_centered_static_history import atomic_json, item_history, load_json, plain_unet, unwrapped
from tools.train_dense_static_history import save_rgb
from tools.train_static_history import fixed_epsilon_monitor, load_settings_args, recursive_batch_to_device, stable_pair_seed
from tools.train_temporal_pairs import encode_latent, load_base


EVAL_STEPS = (100, 250, 500)
EVAL_TIMESTEPS = (100, 500, 900)
VALID_GROUPS = {"A", "B", "C"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=sorted(VALID_GROUPS), required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--settings", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--color-probability", type=float, default=0.7)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--keep-checkpoints", type=int, default=2)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    return p.parse_args(argv)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_selection(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    required = {"train": 8, "heldout": 8, "observation": 10}
    for key, count in required.items():
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != count or len(set(values)) != count:
            raise ValueError("selection.%s must contain %d unique names" % (key, count))
    return payload


def load_cache_items(cache_root: Path, selection: Mapping[str, Sequence[str]]) -> dict:
    manifest = load_json(cache_root / "manifest.json")
    rows = manifest["items"] if isinstance(manifest, dict) else manifest
    by_name = {row["name"]: row for row in rows}
    out = {}
    for split, names in selection.items():
        out[split] = []
        for name in names:
            if name not in by_name:
                raise KeyError("selection item missing from cache manifest: " + name)
            row = by_name[name]
            candidate = row.get("path") or row.get("cache")
            if candidate is None:
                raise KeyError("cache manifest row lacks path/cache for: " + name)
            path = Path(candidate)
            if not path.is_absolute():
                path = cache_root / path
            item = torch.load(path, map_location="cpu")
            if item["name"] != name:
                raise ValueError("cache item name mismatch: " + name)
            if item.get("split") != split:
                raise ValueError("cache item split mismatch for %s: expected %s got %s" % (name, split, item.get("split")))
            out[split].append(item)
    return out


def cache_identity(cache_root: Path) -> dict:
    done_path = cache_root / "cache_done.json"
    done = load_json(done_path)
    if not done.get("done"):
        raise ValueError("condition cache is incomplete")
    return {
        "cache_done": done,
        "manifest_sha256": file_sha256(cache_root / "manifest.json"),
        "cache_done_sha256": file_sha256(done_path),
    }


def inactive_decoder_branch_name(name: str) -> bool:
    # SpatialTransformer keeps both vanilla and deformable attention parameters.
    # In the current Sat/LiDAR generation path only one implementation is used
    # per attention slot; these alternate projections are not reached by forward.
    inactive_markers = (
        ".attn1.sampling_offsets.",
        ".attn1.attention_weights.",
        ".attn1.value_proj.",
        ".attn1.sample_to_out.",
        ".attn2.to_q.",
        ".attn2.to_k.",
        ".attn2.to_v.",
        ".attn2.to_out.0.",
    )
    return any(marker in name for marker in inactive_markers)


def decoder_trainable_name(name: str) -> bool:
    if name.startswith("out."):
        return True
    if not name.startswith("output_blocks."):
        return False
    parts = name.split(".")
    return len(parts) > 1 and parts[1].isdigit() and int(parts[1]) >= 9 and not inactive_decoder_branch_name(name)


def configure_trainables(model, group: str):
    raw = plain_unet(model)
    raw.configure_temporal_history(mode="static_centered", hidden_dim=64, input_variant="types")
    adapter, decoder = [], []
    trainable_names = []
    for name, parameter in raw.named_parameters():
        is_adapter = name.startswith("temporal_history.")
        is_decoder = decoder_trainable_name(name)
        train = (group in {"A", "B"} and is_adapter) or (group in {"B", "C"} and is_decoder)
        parameter.requires_grad_(train)
        if train:
            trainable_names.append(name)
            (adapter if is_adapter else decoder).append(parameter)
    if group in {"A", "B"} and not adapter:
        raise ValueError("group %s has no trainable adapter parameters" % group)
    if group in {"B", "C"} and not decoder:
        raise ValueError("group %s has no trainable decoder parameters" % group)
    if group == "A":
        param_groups = [{"name": "adapter", "params": adapter, "lr": 1e-4}]
    elif group == "B":
        param_groups = [
            {"name": "adapter", "params": adapter, "lr": 1e-4},
            {"name": "decoder", "params": decoder, "lr": 1e-5},
        ]
    else:
        param_groups = [{"name": "decoder", "params": decoder, "lr": 1e-5}]
    return param_groups, trainable_names


def filtered_state_hash(model, trainable_names: set[str]) -> str:
    state = plain_unet(model).state_dict()
    filtered = {k: v.detach().cpu() for k, v in state.items()
                if not k.startswith("temporal_history.") and k not in trainable_names}
    return state_dict_sha256(filtered)


def state_subset_hash(model, predicate) -> str:
    state = plain_unet(model).state_dict()
    subset = {k: v.detach().cpu() for k, v in state.items() if predicate(k)}
    return state_dict_sha256(subset)


def adapter_state_hash(model) -> str:
    return state_subset_hash(model, lambda name: name.startswith("temporal_history."))


def decoder_state_hash(model) -> str:
    return state_subset_hash(model, decoder_trainable_name)


def trainable_state(model, trainable_names: Sequence[str]) -> dict:
    state = plain_unet(model).state_dict()
    return {name: state[name].detach().cpu() for name in trainable_names}


def trainable_grad_summary(model, trainable_names: Sequence[str]) -> dict:
    named = dict(plain_unet(model).named_parameters())
    out = {"adapter_grad_norm": 0.0, "decoder_grad_norm": 0.0, "missing_grad": []}
    for name in trainable_names:
        grad = named[name].grad
        if grad is None:
            out["missing_grad"].append(name)
            continue
        value = float(grad.detach().float().square().sum().cpu())
        if name.startswith("temporal_history."):
            out["adapter_grad_norm"] += value
        else:
            out["decoder_grad_norm"] += value
    out["adapter_grad_norm"] **= 0.5
    out["decoder_grad_norm"] **= 0.5
    return out


def batch_items(train_items: Sequence[Mapping[str, Any]], step: int, batch_size: int) -> list[Mapping[str, Any]]:
    start = ((step - 1) * batch_size) % len(train_items)
    return [train_items[(start + i) % len(train_items)] for i in range(batch_size)]


def make_history_from_rgb(item: Mapping[str, Any], rgb: torch.Tensor, device: torch.device) -> dict:
    valid = item["valid"].unsqueeze(0).to(device)
    return {
        "latent": torch.zeros_like(item["z_cur"]).to(device),
        "dense_rgb": rgb.to(device),
        "dense_valid": valid,
        "dense_measured": item["measured"].unsqueeze(0).to(device),
        "dense_estimated": item["estimated"].unsqueeze(0).to(device),
        "enabled": torch.ones(1, dtype=torch.bool, device=device),
    }


def wrong_history(recipient: Mapping[str, Any], donor: Mapping[str, Any], device: torch.device) -> dict:
    donor_rgb = warp_augmented_rgb(donor["previous_rgb"][None].to(device),
                                   donor["source_flat_index"][None].to(device),
                                   donor["valid"].unsqueeze(0).to(device))
    recipient_valid = recipient["valid"].unsqueeze(0).to(device)
    donor_rgb = donor_rgb * recipient_valid.to(dtype=donor_rgb.dtype)
    return make_history_from_rgb(recipient, donor_rgb, device)


def eval_record(model, item: Mapping[str, Any], mode: str, seed: int, device: torch.device, timestep: int):
    if mode == "off":
        example, use_history = item, False
    elif mode == "correct":
        example, use_history = dict(item, history=item_history(item, device)), True
    elif mode == "wrong":
        example = dict(item, history=wrong_history(item, item["_donor"], device))
        use_history = True
    else:
        raise ValueError("unknown eval mode " + mode)
    rec = fixed_epsilon_monitor(model, example, seed, device, use_history, timestep)
    rec.update(name=item["name"], split=item["split"], mode=mode, timestep=timestep)
    return rec


def evaluate_fixed(model, groups: Mapping[str, Sequence[Mapping[str, Any]]], step: int, out: Path,
                   device: torch.device, seed: int, group_name: str, timesteps: Sequence[int] = EVAL_TIMESTEPS) -> dict:
    records = []
    modes = ("off",) if group_name == "C" else ("off", "correct", "wrong")
    with unwrapped(model):
        for split in ("train", "heldout", "observation"):
            items = list(groups[split])
            for i, item in enumerate(items):
                item = dict(item)
                item["_donor"] = items[(i + 1) % len(items)]
                for timestep in timesteps:
                    fixed_seed = stable_pair_seed(seed, item["name"], "abc_eval_t%d" % timestep)
                    for mode in modes:
                        rec = eval_record(model, item, mode, fixed_seed, device, timestep)
                        rec["step"] = step
                        records.append(rec)
    with (out / "eval_metrics.jsonl").open("a") as handle:
        for rec in records:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
    summary = {}
    for split in ("train", "heldout", "observation"):
        summary[split] = {}
        for mode in modes:
            vals = [r["loss"] for r in records if r["split"] == split and r["mode"] == mode]
            summary[split][mode] = sum(vals) / len(vals)
    return summary


def render_samples(model, groups: Mapping[str, Sequence[Mapping[str, Any]]], step: int, out: Path,
                   device: torch.device, seed: int, group_name: str, final: bool, ddim_steps: int, guidance: float):
    modes = ("off",) if group_name == "C" else ("off", "correct", "wrong")
    selected_splits = ["train", "heldout"] + (["observation"] if final else [])
    with unwrapped(model):
        for split in selected_splits:
            split_items = list(groups[split])
            for i, item in enumerate(split_items):
                donor = split_items[(i + 1) % len(split_items)]
                mode_noise_hashes = []
                folder = out / "samples" / ("step_%07d" % step) / item["name"]
                folder.mkdir(parents=True, exist_ok=True)
                histories = {
                    "off": None,
                    "correct": item_history(item, device),
                    "wrong": wrong_history(item, donor, device),
                }
                sample_seed = stable_pair_seed(seed, item["name"], "abc_sample")
                for mode in modes:
                    z, _info, noise_hash = sample_frame(
                        model, item["cond"], tuple(item["z_cur"].shape),
                        sample_seed, device, histories[mode], ddim_steps, guidance)
                    rgb = decode(model, z)[0].detach().float().cpu()
                    save_rgb(folder / ("%s.png" % mode), rgb)
                    mode_noise_hashes.append(noise_hash)
                    torch.save({"latent": z.detach().cpu(), "rgb": rgb, "noise_hash": noise_hash,
                                "sample_seed": sample_seed, "donor_name": donor["name"]},
                               folder / ("%s.pt" % mode))
                    metric = rgb_sample_metrics(item, rgb, histories[mode])
                    metric.update({"step": step, "name": item["name"], "split": item["split"],
                                   "mode": mode, "noise_hash": noise_hash, "sample_seed": sample_seed,
                                   "donor_name": donor["name"]})
                    with (out / "sample_metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(metric, sort_keys=True) + "\n")
                if len(set(mode_noise_hashes)) != 1:
                    raise RuntimeError("sample modes used different initial noise for " + item["name"])


def rgb_sample_metrics(item: Mapping[str, Any], rgb: torch.Tensor, history: Mapping[str, Any] | None) -> dict:
    current = item["current_rgb"].detach().float().cpu()
    pred = rgb.detach().float().cpu().clamp(0, 1)
    rec = {"rgb_l1_to_current": float((pred - current).abs().mean())}
    if history is not None:
        ref = history["dense_rgb"][0].detach().float().cpu().clamp(0, 1)
        valid = history["dense_valid"][0].detach().cpu().bool()
        rec["reference_valid_fraction"] = float(valid.float().mean())
        if bool(valid.any()):
            rec["rgb_l1_to_reference_valid"] = float(((pred - ref).abs() * valid.float()).sum() / (valid.float().sum() * pred.shape[0]).clamp_min(1))
        else:
            rec["rgb_l1_to_reference_valid"] = None
    return rec


def save_probe_checkpoint(model, step: int, out: Path, base: Mapping[str, Any], args, trainable_names: Sequence[str],
                          cache_id: Mapping[str, Any], frozen_hash_before: str,
                          initial_adapter_sha256: str, initial_decoder_sha256: str):
    payload = {
        "version": "centered_decoder_probe_v1",
        "group": args.group,
        "step": int(step),
        "base_checkpoint": dict(base),
        "cache_identity": dict(cache_id),
        "trainable_names": list(trainable_names),
        "trainable_state": trainable_state(model, trainable_names),
        "trainable_state_sha256": state_dict_sha256(trainable_state(model, trainable_names)),
        "frozen_hash_before": frozen_hash_before,
        "initial_adapter_sha256": initial_adapter_sha256,
        "initial_decoder_sha256": initial_decoder_sha256,
        "requires_grad_true": trainable_names,
        "frozen_hash_after": filtered_state_hash(model, set(trainable_names)),
        "args": vars(args),
    }
    path = out / "checkpoints" / ("probe_%s_step_%07d.pt" % (args.group, step))
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    checkpoints = sorted((out / "checkpoints").glob("probe_%s_step_*.pt" % args.group))
    for stale in checkpoints[:-int(args.keep_checkpoints)]:
        stale.unlink()
    return str(path)


def main(argv=None):
    args = parse_args(argv)
    if args.steps < 1 or args.batch_size < 1 or args.keep_checkpoints < 1:
        raise ValueError("steps, batch size, and checkpoint retention must be positive")
    total_steps = min(args.steps, args.smoke_steps) if args.smoke_steps else args.steps
    eval_steps = {s for s in EVAL_STEPS if s <= total_steps}
    eval_steps.add(total_steps)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    seed_training_step(args.seed, device)
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError("out-dir already exists")
    (out / "checkpoints").mkdir(parents=True)
    selection = load_selection(args.selection)
    cache_root = Path(args.cache_root)
    cache_id = cache_identity(cache_root)
    groups = load_cache_items(cache_root, selection)
    settings = load_settings_args(args.settings)
    settings.batch_size = args.batch_size
    model, base, cfg = load_base(settings, device)
    assert_same_base(cache_id["cache_done"]["fingerprint"]["base_checkpoint"], base)
    if cache_id["cache_done"]["fingerprint"]["settings_sha256"] != file_sha256(args.settings):
        raise ValueError("settings differ from cached conditions")
    model.eval().requires_grad_(False)
    param_groups, trainable_names = configure_trainables(model, args.group)
    frozen_hash_before = filtered_state_hash(model, set(trainable_names))
    initial_adapter_sha256 = adapter_state_hash(model)
    initial_decoder_sha256 = decoder_state_hash(model)
    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler(enabled=device.type == "cuda")
    atomic_json(out / "args.json", {
        "args": vars(args),
        "base_checkpoint": base,
        "cache_identity": cache_id,
        "selection": selection,
        "trainable_names": trainable_names,
        "trainable_count": len(trainable_names),
        "trainable_parameter_count": sum(int(p.numel()) for group in param_groups for p in group["params"]),
        "fresh_base": True,
        "fresh_centered_adapter": True,
        "frozen_hash_before": frozen_hash_before,
        "initial_adapter_sha256": initial_adapter_sha256,
        "initial_decoder_sha256": initial_decoder_sha256,
        "requires_grad_true": trainable_names,
    })
    plain_unet(model).temporal_history.train()
    started = time.time()
    for step in range(1, total_steps + 1):
        items = batch_items(groups["train"], step, args.batch_size)
        step_seed = args.seed + step * 1009
        seed_training_step(step_seed, device)
        color = sample_pair_color(len(items), torch.Generator().manual_seed(step_seed + 17), args.color_probability)
        previous = torch.stack([x["previous_rgb"] for x in items]).to(device)
        current = torch.stack([x["current_rgb"] for x in items]).to(device)
        previous_aug = apply_pair_color(previous, color)
        current_aug = apply_pair_color(current, color)
        valid = torch.stack([x["valid"] for x in items]).to(device)
        warped = warp_augmented_rgb(previous_aug, torch.stack([x["source_flat_index"] for x in items]).to(device), valid)
        z_cur = encode_latent(model, current_aug).float()
        history = None if args.group == "C" else {
            "latent": torch.zeros_like(z_cur),
            "dense_rgb": warped,
            "dense_valid": valid,
            "dense_measured": torch.stack([x["measured"] for x in items]).to(device),
            "dense_estimated": torch.stack([x["estimated"] for x in items]).to(device),
            "enabled": torch.ones(len(items), dtype=torch.bool, device=device),
        }
        cond = recursive_batch_to_device([x["cond"] for x in items], device)
        sat_drop = torch.rand((len(items), 1, 1), device=device) < float(model.satellite_condition_dropout_prob)
        cond = dict(cond)
        cond["context"] = cond["context"] * (~sat_drop)

        def closure():
            with autocast(enabled=device.type == "cuda"):
                loss, aux = epsilon_prediction_loss(model.DDPM, z_cur, cond, history, step_seed)
                return loss, aux

        loss, aux, grad_norm, retries = optimizer_step_with_retry(closure, [p for g in param_groups for p in g["params"]], optimizer, scaler)
        grad = trainable_grad_summary(model, trainable_names)
        if grad["missing_grad"]:
            preview = grad["missing_grad"][:40]
            suffix = "" if len(grad["missing_grad"]) <= 40 else " ... (+%d more)" % (len(grad["missing_grad"]) - 40)
            raise RuntimeError("missing gradients for trainables: " + ", ".join(preview) + suffix)
        if args.group in {"A", "B"} and grad["adapter_grad_norm"] <= 0:
            raise RuntimeError("adapter gradients are zero")
        if args.group in {"B", "C"} and grad["decoder_grad_norm"] <= 0:
            raise RuntimeError("decoder gradients are zero")
        rec = {
            "step": step,
            "group": args.group,
            "loss": float(loss.detach().cpu()),
            "grad_norm": grad_norm,
            "amp_retries": retries,
            "adapter_grad_norm": grad["adapter_grad_norm"],
            "decoder_grad_norm": grad["decoder_grad_norm"],
            "batch_names": [x["name"] for x in items],
            "noise_hash": tensor_hash(aux["noise"]),
            "target_latent_hash": tensor_hash(z_cur),
            "color_hash": hashlib.sha256(json.dumps({k: v.tolist() for k, v in color.items()}, sort_keys=True).encode()).hexdigest(),
            "t_mean": float(aux["t"].float().mean().cpu()),
            "satellite_dropout_fraction": float(sat_drop.float().mean().cpu()),
            "seconds": round(time.time() - started, 2),
        }
        with (out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
        if step in eval_steps:
            plain_unet(model).temporal_history.eval()
            final = step == total_steps
            eval_timesteps = (500,) if args.smoke_steps else EVAL_TIMESTEPS
            summary = evaluate_fixed(model, groups, step, out, device, args.seed + 101, args.group, eval_timesteps)
            if not args.smoke_steps:
                render_samples(model, groups, step, out, device, args.seed + 202, args.group, final, args.ddim_steps, args.guidance)
            checkpoint = save_probe_checkpoint(model, step, out, base, args, trainable_names, cache_id, frozen_hash_before,
                                               initial_adapter_sha256, initial_decoder_sha256)
            frozen_hash_after = filtered_state_hash(model, set(trainable_names))
            frozen_hash_unchanged = frozen_hash_after == frozen_hash_before
            if not frozen_hash_unchanged:
                raise RuntimeError("non-trainable frozen hash changed")
            atomic_json(out / "status.json", {"step": step, "summary": summary, "checkpoint": checkpoint,
                                              "frozen_hash_unchanged": frozen_hash_unchanged})
            plain_unet(model).temporal_history.train()
        if step <= 3 or step % 25 == 0:
            print(json.dumps(rec, sort_keys=True), flush=True)
    final_frozen_hash = filtered_state_hash(model, set(trainable_names))
    atomic_json(out / "done.json", {"done": True, "group": args.group, "steps": total_steps,
                                    "frozen_hash_after": final_frozen_hash,
                                    "frozen_hash_unchanged": final_frozen_hash == frozen_hash_before})


if __name__ == "__main__":
    main()
