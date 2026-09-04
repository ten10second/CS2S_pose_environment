"""Temporal-evidence training on top of the frozen single-frame checkpoint.

Phase-1 design (see docs/temporal_preliminary_findings.md, resume point):
  - freeze the converged single-frame ray-posterior backbone entirely;
  - inject zero-initialised temporal gates into every ray-posterior fusion
    module (third evidence stream: previous frame's fused posterior,
    transported by the ground homography, validity-masked to reliable static
    history);
  - train only the gates with the unchanged single-frame diffusion objective.

Infra: pairs are walked in drive order (streaming), so the previous step's
current frame IS this step's previous frame. The per-module caches from that
forward are reused directly and the redundant teacher forward is skipped
(cache_valid=True items), and the dataset's single-slot sample reuse avoids
reloading the shared frame. num_workers must be 1 so consecutive items land
in the same worker.

Run with torchrun, e.g. 2 GPUs:
  torchrun --standalone --nproc_per_node 2 tools/train_kitti_temporal.py ...
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for p in (str(TOOLS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from omegaconf import OmegaConf  # noqa: E402

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402

from generate_kitti_raea_samples import load_checkpoint_into_model  # noqa: E402
from temporal_evidence import (  # noqa: E402
    TemporalTransportBuilder,
    enable_temporal_evidence,
    temporal_gate_parameters,
)

CONSTANTS = {
    "image_height": 128,
    "image_width": 512,
    "sat_size": 256,
    "max_depth": 80.0,
    "lidar_ray_feature_dim": 576,
    "lidar_ray_depth_bins": 4,
    "lidar_ray_height": 8,
    "lidar_ray_width": 32,
    "image_semantic_feature_dim": 384,
    "image_semantic_height": 8,
    "image_semantic_width": 32,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml")
    p.add_argument("--sd-base-ckpt", required=True)
    p.add_argument("--ckpt", required=True, help="frozen single-frame checkpoint (step_500000.pt)")
    p.add_argument("--manifest", required=True, help="train manifest")
    p.add_argument("--kitti-root", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lidar-ray-feature-cache-root", required=True)
    p.add_argument("--image-semantic-cache-root", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--batch-per-gpu", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=1,
                   help="must stay 1: streaming sample reuse is single-worker")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--temporal-strength-min", type=float, default=0.3)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--gate-bias", type=float, default=-6.0)
    p.add_argument("--resume", default="")
    return p.parse_args()


def build_stream_plan(rows):
    """Ordered (prev_idx, cur_idx, cache_valid) triples.

    Each drive is split into maximal strictly-consecutive frame runs; within a
    run the walk is sequential, so every pair except a run's first reuses the
    cache left by the previous step's forward (cache_valid=True).
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


class StreamDataset(Dataset):
    """Streams a plan in order. Single-slot sample reuse assumes num_workers=1
    (consecutive items share a worker); with more workers it only loses the
    reuse optimization, never correctness."""

    def __init__(self, dataset, rows, plan, kitti_root):
        self.dataset = dataset
        self.rows = rows
        self.plan = plan
        self.kitti_root = kitti_root
        self._last_idx = None
        self._last_sample = None

    def __len__(self):
        return len(self.plan)

    def _load(self, idx):
        if self._last_idx == idx and self._last_sample is not None:
            return self._last_sample
        sample = self.dataset[idx]
        self._last_idx, self._last_sample = idx, sample
        return sample

    def __getitem__(self, i):
        pi, ci, cache_valid = self.plan[i]
        # A rank's first item always loads prev: the module caches are empty
        # in a fresh process regardless of what the plan says.
        cache_valid = bool(cache_valid) and i > 0
        cur = self._load(ci)
        prev = None if cache_valid else self._load(pi)
        transport = TemporalTransportBuilder(
            self.rows[pi], self.rows[ci], kitti_root=self.kitti_root
        )
        return {"cur": cur, "prev": prev, "cache_valid": bool(cache_valid), "transport": transport}


class TrainingStepModule(torch.nn.Module):
    """Expose the diffusion training_step through a normal DDP forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.training_step(batch, 0)


def pair_collate(batch):
    """batch_size is always 1. Each sub-sample must be default-collated into a
    B=1 batch exactly like the original training loader produced; non-tensor
    fields are passed through wrapped."""
    from torch.utils.data._utils.collate import default_collate

    assert len(batch) == 1
    item = batch[0]
    item["cur"] = default_collate([item["cur"]])
    item["prev"] = None if item["prev"] is None else default_collate([item["prev"]])
    item["transport"] = [item["transport"]]
    return item


def move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(v, device) for v in batch)
    return batch


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt
    train_params = cfg.data.params.train.params

    rows = [json.loads(line) for line in open(args.manifest)]
    plan = build_stream_plan(rows)
    shard_plan = plan[rank::world]
    n_reuse = sum(1 for _, _, v in shard_plan if v)
    if rank == 0:
        print(f"[temporal-train] stream pairs: {len(plan)} (cache reuse {100.0 * n_reuse / max(len(plan), 1):.1f}%)")

    dataset = SatLidarRawDataset(
        manifest=args.manifest,
        kitti_root=args.kitti_root,
        condition_mode=str(getattr(train_params, "condition_mode", "raw_lidar_pointmap")),
        image_height=CONSTANTS["image_height"],
        image_width=CONSTANTS["image_width"],
        sat_size=CONSTANTS["sat_size"],
        max_depth=CONSTANTS["max_depth"],
        align_satellite_to_camera=True,
        include_range_image=bool(getattr(train_params, "include_range_image", False)),
        include_raw_lidar_points=False,
        lidar_ray_feature_cache_root=args.lidar_ray_feature_cache_root,
        lidar_ray_feature_cache_suffix=".npz",
        lidar_ray_feature_dim=CONSTANTS["lidar_ray_feature_dim"],
        lidar_ray_depth_bins=CONSTANTS["lidar_ray_depth_bins"],
        lidar_ray_height=CONSTANTS["lidar_ray_height"],
        lidar_ray_width=CONSTANTS["lidar_ray_width"],
        image_semantic_cache_root=args.image_semantic_cache_root,
        image_semantic_cache_suffix=".npz",
        image_semantic_feature_key=str(getattr(train_params, "image_semantic_feature_key", "dino_feat")),
        image_semantic_feature_dim=CONSTANTS["image_semantic_feature_dim"],
        image_semantic_height=CONSTANTS["image_semantic_height"],
        image_semantic_width=CONSTANTS["image_semantic_width"],
        include_tracklets=False,
    )
    stream_dataset = StreamDataset(dataset, rows, shard_plan, args.kitti_root)
    loader = DataLoader(
        stream_dataset,
        batch_size=args.batch_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=6 if args.num_workers > 0 else None,
        collate_fn=pair_collate,
    )

    model = instantiate_from_config(cfg.model).cuda().eval()
    load_checkpoint_into_model(model, args.ckpt)

    hub, blocks = enable_temporal_evidence(model, gate_bias=args.gate_bias)
    model.to(device)  # gates were created after the initial .cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    gate_params = temporal_gate_parameters(blocks)
    for param in gate_params:
        param.requires_grad_(True)
    if rank == 0:
        n_gate = sum(p.numel() for p in gate_params)
        print(f"[temporal-train] temporal blocks: {len(blocks)}, gate params: {n_gate}")

    model.train()
    training_model = TrainingStepModule(model)
    if distributed:
        training_model = DDP(training_model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(gate_params, lr=args.lr, weight_decay=0.0)
    scaler = GradScaler(enabled=args.amp)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        payload = torch.load(args.resume, map_location="cpu")
        gates_state = payload.get("temporal_gates", {})
        for name, block in enumerate(blocks):
            key = str(name)
            if key in gates_state:
                block.ray_posterior_fusion.temporal_gate.load_state_dict(gates_state[key])
        start_step = int(payload.get("step", 0))
        optimizer.load_state_dict(payload.get("optimizer", optimizer.state_dict()))
        for group in optimizer.param_groups:
            group["lr"] = args.lr  # saved state carries the old lr; CLI wins
        if rank == 0:
            print(f"[temporal-train] resumed from {args.resume} at step {start_step}, lr={args.lr}")

    def gate_stats():
        confs, maxs = [], []
        for block in blocks:
            f = block.ray_posterior_fusion
            if hasattr(f, "last_temporal_confidence"):
                confs.append(float(f.last_temporal_confidence))
            g = f.temporal_gate
            maxs.append(float(g.weight.abs().max().detach()))
        return float(np.mean(confs)) if confs else 0.0, float(np.max(maxs)) if maxs else 0.0

    log_every = max(1, args.log_every)
    metrics_file = (out_dir / "temporal_metrics.jsonl").open("a") if rank == 0 else None
    iterator = iter(loader)
    data_epoch = 0
    cache_warm = False  # module caches populated by a previous cur forward?
    running = []
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        try:
            item = next(iterator)
        except StopIteration:
            data_epoch += 1
            iterator = iter(loader)
            item = next(iterator)
        cur_batch = move_batch_to_device(item["cur"], device)
        builder = item["transport"][0] if isinstance(item["transport"], list) else item["transport"]

        # Sequential streaming: when the previous step's forward produced this
        # frame's predecessor (cache_valid AND caches warm), the per-module
        # caches are already in place and the teacher forward is skipped.
        # Otherwise run it once under no_grad with the hub cleared.
        hub.clear()
        if item["prev"] is not None and not cache_warm:
            prev_batch = move_batch_to_device(item["prev"], device)
            with torch.no_grad(), autocast(enabled=args.amp):
                _ = training_model(prev_batch)
        cache_warm = True

        strength = float(np.random.uniform(args.temporal_strength_min, 1.0))
        payload = builder.payload(strength=strength)
        if payload is None:
            hub.clear()
        else:
            hub.set(payload)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            loss = training_model(cur_batch)
        if not torch.isfinite(loss):
            if rank == 0:
                print(f"[temporal-train] non-finite loss at step {step}, skipping")
            optimizer.zero_grad(set_to_none=True)
            hub.clear()
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        hub.clear()
        running.append(float(loss.detach()))

        if step % log_every == 0 and rank == 0:
            conf, wmax = gate_stats()
            record = {
                "step": step,
                "loss": float(np.mean(running)),
                "temporal_conf_mean": conf,
                "gate_w_absmax": wmax,
                "strength": strength,
                "data_epoch": data_epoch,
                "sec_per_step": round((time.time() - t0) / log_every, 3),
            }
            running = []
            t0 = time.time()
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(json.dumps(record), flush=True)

        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                ckpt_payload = {
                    "step": step,
                    "temporal_gates": {
                        str(i): b.ray_posterior_fusion.temporal_gate.state_dict()
                        for i, b in enumerate(blocks)
                    },
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
                torch.save(ckpt_payload, out_dir / "temporal_latest.pt")
                torch.save(ckpt_payload, out_dir / f"temporal_step_{step}.pt")
            if distributed:
                torch.distributed.barrier()

    if metrics_file is not None:
        metrics_file.close()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
