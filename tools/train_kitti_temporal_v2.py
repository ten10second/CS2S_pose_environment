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
    build_stream_plan,
    move_batch_to_device,
)
from temporal_history import (  # noqa: E402
    build_payload,
    enable_history_attention,
    history_latent_from_gt,
    history_trainable_parameters,
)
from train_kitti_raea import (  # noqa: E402
    ensure_fresh_run_directory_distributed,
    synchronized_loss_finite,
)


class TemporalHistoryTrainingStepModule(torch.nn.Module):
    """Own all trainable temporal modules under one DDP forward boundary."""

    def __init__(self, model, history_encoder, hub):
        super().__init__()
        self.model = model
        self.history_encoder = history_encoder
        self.hub = hub

    def forward(self, cur_batch, history_latent=None, has_history=True):
        batch = cur_batch["grd_left_imgs"].shape[0]
        if has_history:
            if history_latent is None:
                raise ValueError("history_latent is required when has_history=True")
            tokens = self.history_encoder(history_latent.detach())
            # The no-history token is a trainable parameter, so retain a zero
            # dependency on normal history steps to keep DDP usage consistent.
            tokens = tokens + self.history_encoder.null_tokens(batch) * 0.0
        else:
            # Exercise the encoder body and its null token even on reset frames;
            # HistoryCrossAttention masks the resulting residual by exact zero.
            p = next(self.history_encoder.parameters())
            dummy = torch.zeros(
                (batch, 4, *self.history_encoder.grid), device=p.device, dtype=p.dtype
            )
            dummy_tokens = self.history_encoder(dummy)
            tokens = self.history_encoder.null_tokens(batch)
            tokens = tokens + dummy_tokens.mean(dim=1, keepdim=True) * 0.0

        self.hub.set(build_payload(self.history_encoder, tokens, has_history))
        try:
            return self.model.training_step(cur_batch, 0)
        finally:
            self.hub.clear()


def pair_collate_v2(batch):
    """B=1; each sub-sample default-collated like the original training loader;
    prev may be None (run starts)."""
    from torch.utils.data._utils.collate import default_collate

    assert len(batch) == 1
    item = batch[0]
    item["cur"] = default_collate([item["cur"]])
    item["prev"] = None if item["prev"] is None else default_collate([item["prev"]])
    return item


def require_resume_payload(payload, args, blocks):
    required = {"step", "history_encoder", "history_attn", "optimizer", "mode", "base_ckpt"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"resume checkpoint missing keys: {missing}")
    if payload["mode"] != "temporal_v2_history_phaseA":
        raise ValueError(f"unexpected resume mode: {payload['mode']!r}")
    if Path(payload["base_ckpt"]).resolve() != Path(args.ckpt).resolve():
        raise ValueError("resume checkpoint was trained from a different base checkpoint")
    expected = {str(i) for i in range(len(blocks))}
    actual = set(payload["history_attn"])
    if actual != expected:
        raise ValueError(
            f"resume history block keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )


def probe_history_effect(
    training_model, cur_batch, history_latent, seed, amp=True, wrong_history_latent=None
):
    """Compare correct, disabled, and optional wrong history under identical noise."""
    if history_latent is None:
        return None
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), autocast(enabled=amp):
            loss_history = training_model(cur_batch, history_latent, True).detach()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), autocast(enabled=amp):
            loss_disabled = training_model(cur_batch, None, False).detach()
        loss_wrong = None
        if wrong_history_latent is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad(), autocast(enabled=amp):
                loss_wrong = training_model(
                    cur_batch, wrong_history_latent, True
                ).detach()
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return loss_history, loss_disabled, loss_wrong


def grad_l2(parameters):
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().float().square().sum())
    return total ** 0.5


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
    p.add_argument("--probe-every", type=int, default=0,
                   help="compare correct vs disabled history loss every N steps; 0 disables")
    p.add_argument("--probe-seed", type=int, default=20260907)
    p.add_argument("--overfit-pairs", type=int, default=0,
                   help="repeat only the first N consecutive pairs; 0 uses the full plan")
    p.add_argument("--resume", default="")
    return p.parse_args()


def main():
    args = parse_args()
    if args.batch_per_gpu != 1:
        raise ValueError("phase-A pair_collate_v2 currently requires --batch-per-gpu 1")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    out_dir = Path(args.out_dir)
    ensure_fresh_run_directory_distributed(
        out_dir, args.resume, distributed=distributed, is_main=rank == 0
    )
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt
    train_params = cfg.data.params.train.params

    rows = [json.loads(line) for line in open(args.manifest)]
    full_plan = build_stream_plan(rows)
    plan = full_plan
    if args.overfit_pairs > 0:
        plan = plan[:args.overfit_pairs]
    if not plan:
        raise RuntimeError("manifest produced no consecutive-frame training pairs")
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
        """Every item is an explicit consecutive-frame pair."""

        def __init__(self, ds, rows_, plan_, kitti_root):
            self.ds, self.rows, self.plan, self.kitti_root = ds, rows_, plan_, kitti_root

        def __len__(self):
            return len(self.plan)

        def __getitem__(self, i):
            pi, ci, _ = self.plan[i]
            cur = self.ds[ci]
            # build_stream_plan only emits valid consecutive pairs, including
            # the first item of each rank-local shard.
            prev = self.ds[pi]
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
    encoder.to(device)  # registered below by the DDP-wrapped training module
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

    # Keep the frozen single-frame backbone deterministic. Only the newly
    # introduced history modules are in training mode.
    model.eval()
    encoder.train()
    for block in blocks:
        block.history_attn.train()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scaler = GradScaler(enabled=args.amp)

    start_step = 0
    if args.resume:
        if not Path(args.resume).is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")
        payload = torch.load(args.resume, map_location="cpu")
        require_resume_payload(payload, args, blocks)
        encoder.load_state_dict(payload["history_encoder"], strict=True)
        attn_state = payload["history_attn"]
        for i, block in enumerate(blocks):
            block.history_attn.load_state_dict(attn_state[str(i)], strict=True)
        start_step = int(payload["step"])
        optimizer.load_state_dict(payload["optimizer"])
        if "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        if rank == 0:
            print(f"[hist-v2] resumed from {args.resume} at step {start_step}, lr={args.lr}")

    # Wrap only after an optional strict resume so DDP initialization
    # broadcasts the restored encoder and attention parameters.
    training_model = TemporalHistoryTrainingStepModule(model, encoder, hub)
    if distributed:
        training_model = DDP(training_model, device_ids=[local_rank], find_unused_parameters=False)

    wrong_history_latent = None
    if args.probe_every > 0 and shard_plan:
        reference_pi = shard_plan[0][0]
        reference_drive = rows[reference_pi].get("drive")
        candidates = [
            pi for pi, _ci, _valid in full_plan
            if pi != reference_pi and rows[pi].get("drive") != reference_drive
        ]
        if not candidates:
            candidates = [pi for pi, _ci, _valid in full_plan if pi != reference_pi]
        if candidates:
            from torch.utils.data._utils.collate import default_collate

            wrong_prev = move_batch_to_device(default_collate([dataset[candidates[0]]]), device)
            wrong_history_latent = history_latent_from_gt(model, wrong_prev)
            if rank == 0:
                print(
                    f"[hist-v2] wrong-history probe source: "
                    f"{rows[candidates[0]].get('drive')}:{rows[candidates[0]].get('frame_index')}"
                )

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

        # Frozen-VAE history latent; the trainable encoder runs inside DDP.
        prev = item["prev"]
        has_history = prev is not None
        if has_history:
            prev_batch = move_batch_to_device(prev, device)
            history_latent = history_latent_from_gt(model, prev_batch)
        else:
            history_latent = None

        if start_step == 0 and step == 1 and args.probe_every > 0:
            probe0 = probe_history_effect(
                training_model,
                cur_batch,
                history_latent,
                args.probe_seed,
                args.amp,
                wrong_history_latent,
            )
            if probe0 is not None and distributed:
                probe0_tensor = torch.stack([value for value in probe0 if value is not None])
                torch.distributed.all_reduce(probe0_tensor)
                probe0_tensor /= world
                values = list(probe0_tensor.unbind())
                probe0 = (values[0], values[1], values[2] if len(values) == 3 else None)
            if rank == 0 and probe0 is not None:
                record0 = {
                    "step": 0,
                    "probe_loss_history": float(probe0[0]),
                    "probe_loss_disabled": float(probe0[1]),
                    "probe_loss_wrong_history": (
                        float(probe0[2]) if probe0[2] is not None else None
                    ),
                    "probe_history_benefit": float(probe0[1] - probe0[0]),
                    "probe_correct_vs_wrong_benefit": (
                        float(probe0[2] - probe0[0]) if probe0[2] is not None else None
                    ),
                    "event": "fresh_run_baseline_before_first_update",
                }
                metrics_file.write(json.dumps(record0) + "\n")
                metrics_file.flush()
                print(json.dumps(record0), flush=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            loss = training_model(cur_batch, history_latent, has_history)
        if not synchronized_loss_finite(loss, distributed):
            if rank == 0:
                print(f"[hist-v2] non-finite loss at step {step}; all ranks skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        encoder_grad_l2 = grad_l2(encoder.parameters())
        cond_query_grad_l2 = grad_l2(
            p for b in blocks for p in b.history_attn.to_q_cond.parameters()
        )
        history_out_grad_l2 = grad_l2(
            p for b in blocks for p in b.history_attn.to_out.parameters()
        )
        scaler.step(optimizer)
        scaler.update()
        running.append(float(loss.detach()))
        ratios.extend([float(b.history_ratio) for b in blocks if b.history_ratio is not None])

        probe = None
        if args.probe_every > 0 and step % args.probe_every == 0:
            probe = probe_history_effect(
                training_model,
                cur_batch,
                history_latent,
                args.probe_seed,
                args.amp,
                wrong_history_latent,
            )
            if probe is not None and distributed:
                probe_tensor = torch.stack([value for value in probe if value is not None])
                torch.distributed.all_reduce(probe_tensor)
                probe_tensor /= world
                values = list(probe_tensor.unbind())
                probe = (values[0], values[1], values[2] if len(values) == 3 else None)

        if step % log_every == 0 and rank == 0:
            record = {
                "step": step,
                "loss": float(np.mean(running)),
                "hist_ratio_mean": float(np.mean(ratios)) if ratios else None,
                "hist_ratio_max": float(np.max(ratios)) if ratios else None,
                "has_history_frac": float(np.mean([has_history])),
                "probe_loss_history": float(probe[0]) if probe is not None else None,
                "probe_loss_disabled": float(probe[1]) if probe is not None else None,
                "probe_loss_wrong_history": (
                    float(probe[2]) if probe is not None and probe[2] is not None else None
                ),
                "probe_history_benefit": (
                    float(probe[1] - probe[0]) if probe is not None else None
                ),
                "probe_correct_vs_wrong_benefit": (
                    float(probe[2] - probe[0])
                    if probe is not None and probe[2] is not None else None
                ),
                "encoder_grad_l2": encoder_grad_l2,
                "cond_query_grad_l2": cond_query_grad_l2,
                "history_out_grad_l2": history_out_grad_l2,
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
                    "scaler": scaler.state_dict(),
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
