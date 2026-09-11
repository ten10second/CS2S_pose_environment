"""Geometry-local history adapter (temporal v3, live design), with drive-held-out
fixed probes.

The frozen CFG backbone and the paused v1/v2 temporal experiments are not
modified. Each rank uses one frame pair; only the history encoder and the
selected history attention are trained, plus optionally the host feed-forward
tail.

Current design (Stage F, see docs/temporal_design_map.md):
  - history is the previous frame's RGB latent, read as temporal K/V by
    GeometryHistoryAttention, gated by the LiDAR/pose correspondence;
  - injection is the finest 640-d decoder fusion block (after the bottleneck
    depth head), so the residual is not a depth-feature channel;
  - the history encoder output is used in every step, including the no-history
    condition, which routes it through a learned null token and multiplies the
    residual by an exact zero (first frames stay identical to single-frame);
  - --unfreeze-host trains the injected block's ff/norm3 so generation can act
    on the retrieved appearance, and --appearance-x0-weight adds a masked
    reconstruction term (current-frame x0 on correspondence cells, not a
    previous-frame colour identity loss).
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

TOOLS = Path(__file__).resolve().parent
for path in (str(TOOLS), str(TOOLS.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from omegaconf import OmegaConf

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset
from utils.util import instantiate_from_config
from generate_kitti_raea_samples import load_checkpoint_into_model
from train_kitti_raea import synchronized_loss_finite
from temporal_history import (
    AFTER_BOTTLENECK,
    enable_history_attention,
    history_host_state_dict,
    history_trainable_parameters,
    load_history_host_state_dict,
    parse_history_block_indices,
)
from temporal_history_geometry import build_pair_geometry

MODE_GENERATOR = "geometry_history_v1_generator"


def build_stream_plan(rows):
    """Ordered (prev_idx, cur_idx, cache_valid) triples.

    Each drive is split into maximal strictly-consecutive frame runs. The
    geometry history trainer consumes only prev/cur; cache_valid is carried for
    compatibility with the streaming contract it was introduced for.
    """
    by_drive = defaultdict(list)
    for i, r in enumerate(rows):
        try:
            fi = int(r["frame_index"])
        except (KeyError, ValueError):
            continue
        by_drive[r.get("drive")].append((fi, i))

    def emit_run(run, out):
        for k in range(1, len(run)):
            out.append((run[k - 1], run[k], k > 1))

    plan = []
    for drive in sorted(by_drive):
        idxs = [i for _, i in sorted(by_drive[drive])]
        run = []
        for k, i in enumerate(idxs):
            if run and int(rows[i]["frame_index"]) - int(rows[run[-1]]["frame_index"]) != 1:
                emit_run(run, plan)
                run = []
            run.append(i)
        emit_run(run, plan)
    return plan


def move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(v, device) for v in batch)
    return batch


def grad_l2(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().float().square().sum())
    return total ** 0.5


def split_drive_pairs(rows, val_drives=None):
    """Never share a drive (or an adjacent endpoint) across train and probe."""
    plan = build_stream_plan(rows)
    drives = sorted({rows[p[0]]["drive"] for p in plan})
    if len(drives) < 2:
        raise ValueError("need at least two drives for independent validation")
    held = set(val_drives or drives[::max(2, len(drives) // 4)])
    if not held.issubset(drives):
        raise ValueError("requested validation drive is absent from manifest")
    train = [p for p in plan if rows[p[0]]["drive"] not in held]
    val = [p for p in plan if rows[p[0]]["drive"] in held]
    if not train or not val:
        raise ValueError("empty training or validation partition")
    return train, val, sorted(held)


class GeometryPairs(Dataset):
    def __init__(self, dataset, rows, pairs, root):
        self.dataset, self.rows, self.pairs, self.root = dataset, rows, pairs, root

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pi, ci, _ = self.pairs[index]
        geom = build_pair_geometry(self.rows[pi], self.rows[ci], self.root)
        # History needs only RGB. Do not construct unused satellite/LiDAR/DINO
        # conditions for the previous frame a second time on the shared host.
        with Image.open(self.dataset.records[pi]["image_02_path"]) as image:
            previous_rgb = self.dataset.grd_transform(image.convert("RGB"))
        return {"cur": self.dataset[ci], "prev": {"grd_left_imgs": previous_rgb},
                "history_grid": geom["history_grid"],
                "history_valid": geom["history_valid"],
                "pair_id": self.rows[ci]["sample_id"]}


class GeometryTrainingStep(torch.nn.Module):
    def __init__(self, model, encoder, hub, appearance_x0_weight=0.0):
        super().__init__()
        self.model, self.history_encoder, self.hub = model, encoder, hub
        self.appearance_x0_weight = float(appearance_x0_weight)

    def forward(self, batch, latent, geometry, has_history=True, drop_satellite=False):
        # Full encoder path even when history is absent: no DDP-unused params.
        if latent is None:
            ref = batch["grd_left_imgs"]
            latent = ref.new_zeros((ref.shape[0], 4, *self.history_encoder.grid))
        tokens = self.history_encoder(latent.detach())
        tokens = tokens + self.history_encoder.null_tokens(tokens.shape[0]) * 0.0
        self.hub.set({"history_tokens": tokens, "has_history": bool(has_history),
                      "history_hw": self.history_encoder.grid,
                      "history_grid": geometry["history_grid"],
                      "history_valid": geometry["history_valid"]})
        # Backbone remains eval(). Explicitly exercise the satellite-unconditional
        # branch; LiDAR, history and geometric inputs are identical in both.
        original = self.model.apply_satellite_condition_dropout
        self.model.apply_satellite_condition_dropout = (
            lambda cond: torch.zeros_like(cond) if drop_satellite else cond
        )
        valid = geometry["history_valid"].float()
        if valid.dim() == 3:
            valid = valid[:, None]
        ddpm = self.model.DDPM
        previous_appearance = getattr(ddpm, "_history_appearance_x0", None)
        if self.appearance_x0_weight > 0:
            ddpm._history_appearance_x0 = (valid, self.appearance_x0_weight)
        else:
            ddpm._history_appearance_x0 = None
        try:
            return self.model.training_step(batch, 0)
        finally:
            self.model.apply_satellite_condition_dropout = original
            ddpm._history_appearance_x0 = previous_appearance
            self.hub.clear()


@torch.no_grad()
def encode_history(model, batch):
    return model.pre_AE_model.encode(batch["grd_left_imgs"] * 2 - 1).sample() * model.scale_factor


@contextmanager
def pinned_denoising(ddpm, timestep, seed):
    """Pin exactly the diffusion inputs, not the first arbitrary RNG call."""
    original = ddpm.p_losses
    calls = []

    def pinned(x_start, t, *args, **kwargs):
        generator = torch.Generator(device=x_start.device).manual_seed(seed)
        kwargs["noise"] = torch.randn(x_start.shape, device=x_start.device,
                                     dtype=x_start.dtype, generator=generator)
        calls.append(True)
        return original(x_start, torch.full_like(t, timestep), *args, **kwargs)

    ddpm.p_losses = pinned
    try:
        yield
        if len(calls) != 1:
            raise RuntimeError(f"fixed probe intercepted {len(calls)} diffusion losses, expected one")
    finally:
        ddpm.p_losses = original


def fixed_probe(module, batch, latent, geom, wrong_latent, timesteps, seed, amp,
                satellite_arms=(False,)):
    """Paired history conditions at pinned timesteps and pinned diffusion noise.

    ``satellite_arms`` adds the condition-necessity axis: with the satellite
    conditioning zeroed, appearance is no longer available from the current
    conditions, so history becomes the only colour source. Training keeps the
    default single arm (satellite conditioned); the necessity probe runs both.
    """
    records = []
    shuffled = dict(geom)
    # Preserve target validity and the set of valid source coordinates exactly.
    coords = geom["history_grid"].clone()
    mask = geom["history_valid"].bool()
    coords[mask] = coords[mask].flip(0)
    shuffled["history_grid"] = coords
    conditions = (("disabled", latent, geom, False),
                  ("correct", latent, geom, True),
                  ("wrong_geometry", latent, shuffled, True),
                  ("wrong_history", wrong_latent, geom, True))
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    for t in timesteps:
        for satellite_blind in satellite_arms:
            for name, history, geometry, enabled in conditions:
                # Pins current VAE sampling as well as preserving the training RNG.
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    with torch.no_grad(), autocast(enabled=amp), pinned_denoising(module.model.DDPM, t, seed + 1):
                        loss = module(batch, history, geometry, enabled, bool(satellite_blind))
                metrics = {k: float(v) for k, v in module.model.DDPM.last_loss_metrics.items()
                           if isinstance(v, (int, float)) or (torch.is_tensor(v) and v.numel() == 1)}
                records.append({**metrics, "t": t, "condition": name,
                                "satellite_blind": bool(satellite_blind),
                                "loss_total": float(loss)})
    return records


def make_dataset(args, cfg):
    params = cfg.data.params.train.params
    return SatLidarRawDataset(
        manifest=args.manifest, kitti_root=args.kitti_root,
        condition_mode=str(getattr(params, "condition_mode", "raw_lidar_pointmap")),
        image_height=128, image_width=512, sat_size=256, max_depth=80.,
        align_satellite_to_camera=True, include_range_image=False,
        include_raw_lidar_points=False, include_tracklets=False,
        lidar_ray_feature_cache_root=args.lidar_ray_feature_cache_root,
        lidar_ray_feature_cache_suffix=".npz", lidar_ray_feature_dim=576,
        lidar_ray_depth_bins=4, lidar_ray_height=8, lidar_ray_width=32,
        image_semantic_cache_root=args.image_semantic_cache_root,
        image_semantic_cache_suffix=".npz", image_semantic_feature_key="dino_feat",
        image_semantic_feature_dim=384, image_semantic_height=8, image_semantic_width=32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sd-base-ckpt", "ckpt", "manifest", "kitti-root", "out-dir",
                 "lidar-ray-feature-cache-root", "image-semantic-cache-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea_cfgdrop10.yaml")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--history-dim", type=int, default=64)
    parser.add_argument("--block-indices", default=AFTER_BOTTLENECK)
    parser.add_argument("--unfreeze-host", action="store_true",
                        help="Train the feed-forward tail of injected decoder blocks")
    parser.add_argument("--appearance-x0-weight", type=float, default=0.0,
                        help="Masked x0 appearance loss on correspondence cells")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--timesteps", default="250,750")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--history-dropout", type=float, default=.1)
    parser.add_argument("--val-drives", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    MODE = MODE_GENERATOR
    index_spec = parse_history_block_indices(args.block_indices)
    timesteps = [int(x) for x in args.timesteps.split(",")]
    if not 0 <= args.history_dropout < 1 or args.steps < 1:
        raise ValueError("invalid dropout or step budget")
    rank, world, local = (int(os.environ.get(k, d)) for k, d in
                          (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")))
    torch.cuda.set_device(local)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    device = torch.device("cuda", local)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    out = Path(args.out_dir)
    if rank == 0:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"use a fresh output directory: {out}")
        out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        torch.distributed.barrier()
    cfg = OmegaConf.load(args.config)
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    train, val, held = split_drive_pairs(rows, args.val_drives.split(",") if args.val_drives else None)
    dataset = make_dataset(args, cfg)
    pairs = GeometryPairs(dataset, rows, train, args.kitti_root)
    sampler = DistributedSampler(pairs, world, rank, shuffle=True, seed=args.seed, drop_last=False)
    loader = DataLoader(pairs, batch_size=1, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    model = instantiate_from_config(cfg.model).cuda().eval()
    load_checkpoint_into_model(model, args.ckpt)
    hub, encoder, blocks = enable_history_attention(
        model, block_indices=index_spec,
        history_dim=args.history_dim, heads=4, dim_head=32)
    indices = tuple(block.history_block_index for block in blocks)
    model.cuda().eval()
    encoder.cuda().train()
    for param in model.parameters():
        param.requires_grad_(False)
    trainable = history_trainable_parameters(encoder, blocks, unfreeze_host=args.unfreeze_host)
    for param in trainable:
        param.requires_grad_(True)
    for block in blocks:
        block.history_attn.train()
        if args.unfreeze_host:
            if hasattr(block, "ff"):
                block.ff.train()
            if hasattr(block, "norm3"):
                block.norm3.train()
    module = GeometryTrainingStep(
        model, encoder, hub, appearance_x0_weight=args.appearance_x0_weight
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
    # A 65536 initial scale can overflow upstream frozen-UNet derivatives even
    # on a zero-output/no-history branch (inf * 0 becomes NaN). Start modestly.
    scaler = GradScaler(enabled=not args.no_amp, init_scale=1024.0)
    start = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        if (payload.get("mode") != MODE or Path(payload["base_ckpt"]).resolve() != Path(args.ckpt).resolve()
                or payload["args"]["block_indices"] != args.block_indices
                or list(payload.get("block_indices", [])) != list(indices)
                or payload["args"]["history_dim"] != args.history_dim
                or bool(payload["args"].get("unfreeze_host")) != bool(args.unfreeze_host)):
            raise ValueError("resume architecture/base checkpoint mismatch")
        if set(payload["history_attn"]) != {str(i) for i in range(len(blocks))}:
            raise ValueError("resume attention module set mismatch")
        if payload.get("manifest_sha256") != hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest():
            raise ValueError("resume manifest mismatch")
        encoder.load_state_dict(payload["history_encoder"], strict=True)
        for i, block in enumerate(blocks):
            block.history_attn.load_state_dict(payload["history_attn"][str(i)], strict=True)
        if args.unfreeze_host:
            load_history_host_state_dict(blocks, payload.get("history_host"))
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        start = int(payload["step"])
    wrapped = DDP(module, device_ids=[local], find_unused_parameters=False) if world > 1 else module
    # Each rank owns one fixed train probe and one fixed held-out probe, evenly
    # distributed over the full pair list (not merely four adjacent frames).
    probe_items = []
    for split, plan in (("train", train), ("val", val)):
        index = min(len(plan) - 1, len(plan) * rank // world)
        item = default_collate([GeometryPairs(dataset, rows, [plan[index]], args.kitti_root)[0]])
        item = move_batch_to_device(item, device)
        with torch.random.fork_rng(devices=[local]):
            torch.manual_seed(args.seed + index)
            latent = encode_history(model, item["prev"])
        probe_items.append((split, item, latent))
    metadata = {"mode": MODE, "args": vars(args), "world_size": world,
                "start_step": start,
                "block_indices": list(indices),
                "appearance_memory": True,
                "unfreeze_host": bool(args.unfreeze_host),
                "history_placement": AFTER_BOTTLENECK if index_spec == AFTER_BOTTLENECK else "explicit",
                "history_dim": args.history_dim,
                "train_pairs": len(train), "val_pairs": len(val), "val_drives": held,
                "base_ckpt": str(Path(args.ckpt).resolve()),
                "base_ckpt_bytes": Path(args.ckpt).stat().st_size,
                "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                "trainable_parameters": sum(p.numel() for p in trainable),
                "selected_blocks": [
                    {
                        "index": block.history_block_index,
                        "dim": block.ray_posterior_fusion.dim,
                        "stage": getattr(block, "history_block_stage", "unknown"),
                    }
                    for block in blocks
                ]}
    if rank == 0:
        (out / "run.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)
    metrics = (out / f"metrics_rank{rank}.jsonl").open("w")
    probes = (out / f"probes_rank{rank}.jsonl").open("w")
    disabled_baselines = {}

    def run_probes(step):
        for j, (split, item, latent) in enumerate(probe_items):
            wrong = probe_items[1 - j][2]  # guaranteed other partition/drive
            geom = {k: item[k] for k in ("history_grid", "history_valid")}
            results = fixed_probe(module, item["cur"], latent, geom, wrong, timesteps,
                                  args.seed + j, not args.no_amp)
            for result in results:
                if not np.isfinite(result["loss_total"]):
                    raise FloatingPointError("non-finite fixed probe")
                if result["condition"] == "disabled" and not args.unfreeze_host:
                    key = (split, result["t"])
                    baseline = disabled_baselines.setdefault(key, result["loss_total"])
                    if result["loss_total"] != baseline:
                        raise RuntimeError(f"frozen disabled probe changed: {key}: {baseline} -> {result['loss_total']}")
                record = {"step": step, "split": split, "pair_id": item["pair_id"][0],
                          "rank": rank, **result}
                probes.write(json.dumps(record) + "\n")
            if step == 0:
                for t in timesteps:
                    by_cond = {r["condition"]: r["loss_total"] for r in results if r["t"] == t}
                    if by_cond.get("correct") == by_cond.get("disabled"):
                        raise RuntimeError(
                            f"appearance skip did not change the correct-history probe at t={t}: {by_cond}"
                        )
            if rank == 0:
                by_t = {t: {r["condition"]: r["loss_total"] for r in results if r["t"] == t}
                        for t in timesteps}
                print(json.dumps({"event": "fixed_probe", "step": step, "split": split, "loss": by_t}), flush=True)
        probes.flush()

    run_probes(start)
    epoch = start // len(loader)
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    satellite_drop = float(getattr(cfg.model.params, "satellite_condition_dropout_prob", 0.))
    tick = time.time()
    for step in range(start + 1, args.steps + 1):
        step_start = time.time()
        try:
            item = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            item = next(iterator)
        item = move_batch_to_device(item, device)
        data_ready = time.time()
        latent = encode_history(model, item["prev"])
        # Force both edge paths in the first smoke steps, then random dropout.
        history = step != 2 and random.random() >= args.history_dropout
        drop_sat = step == 3 or random.random() < satellite_drop
        geom = {k: item[k] for k in ("history_grid", "history_valid")}
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=not args.no_amp):
            loss = wrapped(item["cur"], latent, geom, history, drop_sat)
        if not synchronized_loss_finite(loss, world > 1):
            raise FloatingPointError(f"non-finite loss at step {step}; synchronized abort")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        missing = [name for name, p in module.named_parameters() if p.requires_grad and p.grad is None]
        if missing:
            raise RuntimeError(f"unused trainable parameters: {missing}")
        gradients = {"encoder_grad_l2": grad_l2(encoder.parameters()),
                     "cond_query_grad_l2": grad_l2(p for b in blocks for p in b.history_attn.to_q_cond.parameters()),
                     "out_grad_l2": grad_l2(p for b in blocks for p in b.history_attn.to_out.parameters())}
        finite = torch.stack([torch.isfinite(p.grad).all() for p in trainable]).all().int()
        finite *= int(all(np.isfinite(v) for v in gradients.values()))
        if world > 1:
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not finite.item():
            print(json.dumps({"event": "nonfinite_gradients", "step": step,
                              "rank": rank, "scale": scaler.get_scale(),
                              "history": history, "loss": float(loss.detach()),
                              **gradients}), flush=True)
            raise FloatingPointError("non-finite unscaled gradients; synchronized abort")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        record = {"step": step, "rank": rank, "loss": float(loss.detach()), **gradients,
                  "history": history, "satellite_dropped": drop_sat,
                  "valid_fraction": float(geom["history_valid"].float().mean()),
                  "history_ratio": [b.history_attn.last_ratio for b in blocks],
                  "history_memory_ratio": [b.history_attn.last_memory_ratio for b in blocks],
                  "data_wait_seconds": data_ready - step_start,
                  "compute_seconds": time.time() - data_ready,
                  "seconds": time.time() - tick}
        metrics.write(json.dumps(record) + "\n")
        metrics.flush()
        if rank == 0 and (step <= 3 or step % args.log_every == 0):
            print(json.dumps(record), flush=True)
        if step % args.probe_every == 0 or step == args.steps:
            run_probes(step)
        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                payload = {**metadata, "step": step, "history_encoder": encoder.state_dict(),
                           "history_attn": {str(i): b.history_attn.state_dict() for i, b in enumerate(blocks)},
                           "history_host": history_host_state_dict(blocks) if args.unfreeze_host else {},
                           "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}
                temporary = out / f"geometry_history_step_{step}.pt.tmp"
                torch.save(payload, temporary)
                temporary.replace(out / f"geometry_history_step_{step}.pt")
            if world > 1:
                torch.distributed.barrier()
    metrics.close()
    probes.close()
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
