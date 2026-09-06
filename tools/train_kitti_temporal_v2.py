"""Temporal-history v2 training (route-history, phase A).

Difference from the v1/route-C trainer, per docs TODO v2:
  - history = the PREVIOUS frame's final latent (teacher-forced: VAE-encoded
    GT), not the previous frame's satellite tokens and not the previous
    generation at train time;
  - the history attention's query explicitly receives the current conditions'
    fused summary, so it can learn to read history only where the conditions
    under-determine appearance;
  - objective is the unchanged single-frame denoising loss (P4-02);
  - no UNet teacher forward, no latent-transport caches, no streaming-skip
    optimisation to get wrong: every pair item carries its own previous frame;
  - no-history samples (run starts) route a learned null token through the
    K/V weights and multiply the output by an exact 0.0 flag, so DDP sees
    every parameter and first frames stay bit-identical to the frozen model
    (P4-04).

Run:
  torchrun --standalone --nproc_per_node 2 tools/train_kitti_temporal_v2.py ...
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
from train_kitti_temporal import (  # noqa: E402  reuse the reviewed plan builder
    CONSTANTS,
    TrainingStepModule,
    build_stream_plan,
    move_batch_to_device,
)
from temporal_history import (  # noqa: E402
    build_payload,
    enable_history_attention,
    history_tokens_from_gt,
    history_trainable_parameters,
)


def pair_collate_v2(batch):
    """B=1; each sub-sample default-collated like the original training loader;
    prev may be None (run starts)."""
    from torch.utils.data._utils.collate import default_collate

    assert len(batch) == 1
    item = batch[0]
    item["cur"] = default_collate([item["cur"]])
    item["prev"] = None if item["prev"] is None else default_collate([item["prev"]])
    return item


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml")
    p.add_argument("--sd-base-ckpt", required=True)
    p.add_argument("--ckpt", required=True, help="frozen single-frame checkpoint (step_500000.pt)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--kitti-root", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lidar-ray-feature-cache-root", required=True)
    p.add_argument("--image-semantic-cache-root", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-per-gpu", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--history-dim", type=int, default=256)
    p.add_argument("--resume", default="")
    return p.parse_args()


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
    if rank == 0:
        print(f"[hist-v2] stream pairs: {len(plan)} (per rank {len(shard_plan)})")

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

    class PairDatasetV2(Dataset):
        """Every item carries its own previous frame; run starts carry None."""

        def __init__(self, ds, rows_, plan_, kitti_root):
            self.ds, self.rows, self.plan, self.kitti_root = ds, rows_, plan_, kitti_root

        def __len__(self):
            return len(self.plan)

        def __getitem__(self, i):
            pi, ci, _ = self.plan[i]
            cur = self.ds[ci]
            prev = self.ds[pi] if i > 0 else None
            return {"cur": cur, "prev": prev}

    pair_ds = PairDatasetV2(dataset, rows, shard_plan, args.kitti_root)
    loader = DataLoader(
        pair_ds,
        batch_size=args.batch_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else 2,
        collate_fn=pair_collate_v2,
    )

    model = instantiate_from_config(cfg.model).cuda().eval()
    load_checkpoint_into_model(model, args.ckpt)

    hub, encoder, blocks = enable_history_attention(model, history_dim=args.history_dim)
    model.to(device)  # new modules were created after the initial .cuda()
    encoder.to(device)  # the shared encoder lives OUTSIDE the model tree
    assert next(encoder.parameters()).device.type == "cuda"
    n_backbone = 0
    for param in model.parameters():
        if param.requires_grad:
            n_backbone += 1
        param.requires_grad_(False)
    trainable = history_trainable_parameters(encoder, blocks)
    for param in trainable:
        param.requires_grad_(True)
    if rank == 0:
        print(f"[hist-v2] blocks: {len(blocks)}, trainable params: {sum(p.numel() for p in trainable):,}")

    model.train()
    training_model = TrainingStepModule(model)
    if distributed:
        training_model = DDP(training_model, device_ids=[local_rank], find_unused_parameters=False)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = GradScaler(enabled=args.amp)

    start_step = 0
    if args.resume and Path(args.resume).exists():
        payload = torch.load(args.resume, map_location="cpu")
        encoder.load_state_dict(payload.get("history_encoder", {}))
        attn_state = payload.get("history_attn", {})
        for i, block in enumerate(blocks):
            if str(i) in attn_state:
                block.history_attn.load_state_dict(attn_state[str(i)])
        start_step = int(payload.get("step", 0))
        optimizer.load_state_dict(payload.get("optimizer", optimizer.state_dict()))
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if rank == 0:
            print(f"[hist-v2] resumed from {args.resume} at step {start_step}, lr={args.lr}")

    log_every = max(1, args.log_every)
    metrics_file = (out_dir / "metrics.jsonl").open("a") if rank == 0 else None
    iterator = iter(loader)
    data_epoch = 0
    running, ratios = [], []
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        try:
            item = next(iterator)
        except StopIteration:
            data_epoch += 1
            iterator = iter(loader)
            item = next(iterator)
        cur_batch = move_batch_to_device(item["cur"], device)

        # history payload from the previous GT frame (teacher-forced, P4-01)
        hub.clear()
        prev = item["prev"]
        has_history = prev is not None
        if has_history:
            prev_batch = move_batch_to_device(prev, device)
            tokens = history_tokens_from_gt(model, encoder, prev_batch)
        else:
            tokens = None
        hub.set(build_payload(encoder, tokens, has_history))

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            loss = training_model(cur_batch)
        if not torch.isfinite(loss):
            if rank == 0:
                print(f"[hist-v2] non-finite loss at step {step}, skipping")
            optimizer.zero_grad(set_to_none=True)
            hub.clear()
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        hub.clear()
        running.append(float(loss.detach()))
        ratios.extend([float(b.history_ratio) for b in blocks if b.history_ratio is not None])

        if step % log_every == 0 and rank == 0:
            record = {
                "step": step,
                "loss": float(np.mean(running)),
                "hist_ratio_mean": float(np.mean(ratios)) if ratios else None,
                "hist_ratio_max": float(np.max(ratios)) if ratios else None,
                "has_history_frac": float(np.mean([has_history])),
                "data_epoch": data_epoch,
                "sec_per_step": round((time.time() - t0) / log_every, 3),
            }
            running, ratios = [], []
            t0 = time.time()
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(json.dumps(record), flush=True)

        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                ckpt = {
                    "step": step,
                    "history_encoder": encoder.state_dict(),
                    "history_attn": {str(i): b.history_attn.state_dict() for i, b in enumerate(blocks)},
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "base_ckpt": args.ckpt,
                    "mode": "temporal_v2_history_phaseA",
                }
                torch.save(ckpt, out_dir / "hist_v2_latest.pt")
                torch.save(ckpt, out_dir / f"hist_v2_step_{step}.pt")
            if distributed:
                torch.distributed.barrier()

    if metrics_file is not None:
        metrics_file.close()
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
