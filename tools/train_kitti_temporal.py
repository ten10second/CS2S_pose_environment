"""Temporal-evidence training on top of the frozen single-frame checkpoint.

Phase-1 design (see docs/temporal_preliminary_findings.md, resume point):
  - freeze the converged single-frame ray-posterior backbone entirely;
  - inject zero-initialised temporal gates into every ray-posterior fusion
    module (third evidence stream: previous frame's fused posterior,
    transported by the ground homography, validity-masked to reliable static
    history);
  - train only the gates with the unchanged single-frame diffusion objective
    on adjacent same-drive frame pairs (teacher-forced previous frame).

Run with torchrun, e.g. 4 GPUs:
  torchrun --standalone --nproc_per_node 4 tools/train_kitti_temporal.py ...
"""
import argparse
import json
import os
import sys
import time
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
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--temporal-strength-min", type=float, default=0.3,
                   help="per-step lower bound of the temporal feature strength "
                        "multiplier (regularises against over-trusting history)")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--gate-bias", type=float, default=-6.0)
    p.add_argument("--resume", default="")
    return p.parse_args()


def build_pair_index(manifest_path):
    rows = [json.loads(line) for line in open(manifest_path)]
    pairs = []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if a.get("drive") != b.get("drive"):
            continue
        try:
            gap = int(b["frame_index"]) - int(a["frame_index"])
        except (KeyError, ValueError):
            continue
        if gap == 1:
            pairs.append((i, i + 1))
    return rows, pairs


class PairDataset(Dataset):
    """Returns {cur: sample, prev: sample, transport: TemporalTransportBuilder}."""

    def __init__(self, dataset, rows, pairs, kitti_root):
        self.dataset = dataset
        self.rows = rows
        self.pairs = pairs
        self.kitti_root = kitti_root

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pi, ci = self.pairs[i]
        prev_sample = self.dataset[pi]
        cur_sample = self.dataset[ci]
        transport = TemporalTransportBuilder(
            self.rows[pi], self.rows[ci], kitti_root=self.kitti_root
        )
        return {"cur": cur_sample, "prev": prev_sample, "transport": transport}


class TrainingStepModule(torch.nn.Module):
    """Expose the diffusion training_step through a normal DDP forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.training_step(batch, 0)


def pair_collate(batch):
    """batch_size is always 1. Each sub-sample must be default-collated into a
    B=1 batch exactly like the original training loader produced; the
    non-tensor transport builder is wrapped in a list instead."""
    from torch.utils.data._utils.collate import default_collate

    assert len(batch) == 1
    item = batch[0]
    item["cur"] = default_collate([item["cur"]])
    item["prev"] = default_collate([item["prev"]])
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

    rows, pairs = build_pair_index(args.manifest)
    if rank == 0:
        print(f"[temporal-train] adjacent same-drive pairs: {len(pairs)}")
    # rank-shard the pair order (deterministic; no epoch shuffle needed for
    # this scale, the dataset itself is much larger than one pass needs)
    shard_pairs = pairs[rank::world]

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
    pair_dataset = PairDataset(dataset, rows, shard_pairs, args.kitti_root)
    loader = DataLoader(
        pair_dataset,
        batch_size=args.batch_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
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
        if rank == 0:
            print(f"[temporal-train] resumed from {args.resume} at step {start_step}")

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
        prev_batch = move_batch_to_device(item["prev"], device)
        builder = item["transport"][0] if isinstance(item["transport"], list) else item["transport"]

        # 1) teacher-forced previous frame: populate per-module fused-posterior
        #    caches under no_grad, with the hub cleared (pure single-frame pass)
        hub.clear()
        with torch.no_grad(), autocast(enabled=args.amp):
            _ = training_model(prev_batch)

        # 2) current frame: consume the transported caches through the gates
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
                payload_ckpt = {
                    "step": step,
                    "temporal_gates": {
                        str(i): b.ray_posterior_fusion.temporal_gate.state_dict()
                        for i, b in enumerate(blocks)
                    },
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
                torch.save(payload_ckpt, out_dir / "temporal_latest.pt")
                torch.save(payload_ckpt, out_dir / f"temporal_step_{step}.pt")
            if distributed:
                torch.distributed.barrier()

    if metrics_file is not None:
        metrics_file.close()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
