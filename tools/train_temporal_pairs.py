"""Train next-frame generation from a mandatory clean GT previous frame."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace
from itertools import islice
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ldm.modules.temporal_amp import optimizer_step_with_retry
from ldm.modules.temporal_pair_training import (
    anchor_trainable_loss,
    apply_history_dropout,
    base_identity,
    build_history,
    configure_temporal_pair_trainables,
    epsilon_prediction_loss,
    flatten_trainable_groups,
    make_optimizer_param_groups,
    load_temporal_checkpoint,
    prune_checkpoints,
    save_temporal_checkpoint,
    seed_training_step,
    training_contract,
    validate_resume_settings,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="Pretrained initialization; the decoder and history reader are trained")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--kitti-root", required=True)
    p.add_argument("--sd-base-ckpt", required=True)
    p.add_argument("--lidar-pixel-feature-cache-root", required=True)
    p.add_argument("--image-semantic-cache-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=None, help="Total full-data epochs; derives --steps from the distributed loader")
    p.add_argument("--fixed-eval-settings", default="", help="Recorded subset probe settings for identical fixed evaluation")
    p.add_argument("--eval-every", type=int, default=0, help="Fixed evaluation interval in updates; also evaluates start and end")
    p.add_argument("--pair-eval-settings", default="", help="Optional recorded heldout pairs; otherwise select deterministic heldout pairs")
    p.add_argument("--pair-eval-every", type=int, default=500, help="Independent GT-pair generation interval, including start/end; 0 disables")
    p.add_argument("--pair-eval-count", type=int, default=8)
    p.add_argument("--pair-eval-ddim-steps", type=int, default=50)
    p.add_argument("--pair-eval-guidance", type=float, default=7.5)
    p.add_argument("--pair-eval-off", action="store_true", help="Also render an optional OFF diagnostic alongside GT-pair generation")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--temporal-lr", type=float, default=1e-4)
    p.add_argument("--decoder-lr", type=float, default=2e-6)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--training-task", choices=["gt_next_frame", "history_ablation"], default="gt_next_frame")
    p.add_argument("--history-policy", choices=["correct", "off"], default="correct")
    p.add_argument("--history-dropout", type=float, default=0.0)
    p.add_argument("--temporal-mode", choices=["geometry", "content"], default="geometry")
    p.add_argument("--temporal-hidden-dim", type=int, default=64)
    p.add_argument("--geometry-depth-candidates", type=int, default=16)
    p.add_argument("--latent-grid-height", type=int, default=16)
    p.add_argument("--latent-grid-width", type=int, default=64)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--keep-checkpoints", type=int, default=2)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--resume-ckpt", default="", help="Continue the same task, optimizer and data order")
    initialization.add_argument("--init-temporal-ckpt", default="", help="Initialize history/decoder weights, starting a new optimizer and step counter")
    p.add_argument("--min-free-out-gb", type=float, default=8.0)
    p.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    return p.parse_args(argv)


def rank_info(args):
    visible = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5")
    if any(gpu.strip() not in {"4", "5"} for gpu in visible.split(",")):
        raise ValueError("temporal training is restricted to physical GPUs 4 and 5")
    local = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank(); world = torch.distributed.get_world_size(); local = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return {"rank": rank, "world": world, "local": local, "device": device, "distributed": distributed, "is_main": rank == 0}


def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def shutdown_training_loader(iterator):
    # PyTorch's multiprocess iterator owns its pin-memory thread and workers.
    # Close them while Python is still running, before NCCL/interpreter teardown.
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


class TrainingBatchSampler(BatchSampler):
    """Dispatch only batches this run will consume, including partial resumes."""
    def __init__(self, sampler, batch_size):
        super().__init__(sampler, batch_size, drop_last=True)
        self.start = 0
        self.stop = super().__len__()

    def set_window(self, start, stop):
        if not 0 <= start < stop <= super().__len__():
            raise ValueError("invalid training batch window")
        self.start, self.stop = start, stop

    def __iter__(self):
        return islice(super().__iter__(), self.start, self.stop)

    def __len__(self):
        return self.stop - self.start


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


class ConsecutivePairDataset(Dataset):
    def __init__(self, base, kitti_root=None, grid=(16, 64), depth_candidates=16, build_geometry=True):
        from tools.temporal_history_geometry import consecutive_rows
        self.base = base
        self.kitti_root = kitti_root
        self.grid = tuple(grid)
        self.depth_candidates = int(depth_candidates)
        self.build_geometry = bool(build_geometry)
        self.pairs = [(i - 1, i) for i in range(1, len(base.records)) if consecutive_rows(base.records[i - 1], base.records[i])]
        if not self.pairs:
            raise ValueError("no consecutive pairs in manifest")
    def __len__(self): return len(self.pairs)
    def __getitem__(self, index):
        pi, ci = self.pairs[index]
        # History consumes only RGB. Avoid rasterizing a second 576-channel
        # LiDAR tensor and loading unused satellite/semantic features per pair.
        from PIL import Image
        with Image.open(self.base.records[pi]["image_02_path"]) as image:
            previous_rgb = self.base.grd_transform(image.convert("RGB"))
        item = {"prev": {"grd_left_imgs": previous_rgb}, "cur": self.base[ci],
                "prev_row": self.base.records[pi], "cur_row": self.base.records[ci]}
        if self.build_geometry:
            from tools.temporal_ray_geometry import build_pair_ray_geometry
            geo = build_pair_ray_geometry(self.base.records[pi], self.base.records[ci], kitti_root=self.kitti_root,
                                          grid=self.grid, num_depth_candidates=self.depth_candidates)
            item["geometry"] = geo
            item["geometry_metrics"] = dict(geo.get("metrics", {}))
        return item


def collate_pairs(items):
    out = {"prev": stack_samples([x["prev"] for x in items]), "cur": stack_samples([x["cur"] for x in items]),
           "prev_rows": [x["prev_row"] for x in items], "cur_rows": [x["cur_row"] for x in items]}
    if "geometry" in items[0]:
        out["geometries"] = [x["geometry"] for x in items]
        out["geometry_metrics"] = [x.get("geometry_metrics", {}) for x in items]
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


def prepare_training_pair(model, batch, args, device, step_seed):
    """Encode previous GT as reference and current GT only as diffusion target."""
    contract = training_contract(args.training_task, args.history_policy, args.history_dropout)
    cur_cond, cur_rgb = encode_conditions(model, batch["cur"], device)
    sat_probability = float(model.satellite_condition_dropout_prob)
    sat_drop = torch.rand((cur_rgb.shape[0], 1, 1), device=device) < sat_probability
    cur_cond["context"] = cur_cond["context"] * (~sat_drop)
    z_cur = encode_latent(model, cur_rgb)
    history = None
    geom_metrics = []
    if args.history_policy == "correct":
        prev_batch = _to_device(batch["prev"], device)
        previous_rgb = model.get_input(prev_batch, "grd_left_imgs").clamp(0, 1)
        z_previous = encode_latent(model, previous_rgb)
        if z_previous.shape != z_cur.shape:
            raise ValueError("previous reference and current target latent shapes differ")
        history = build_history(z_previous, batch["geometries"])
        if contract["history_required"]:
            history["enabled"] = torch.ones(z_previous.shape[0], dtype=torch.bool, device=device)
        else:
            history = apply_history_dropout(history, args.history_dropout, step_seed + 17, device=device)
        geom_metrics = batch.get("geometry_metrics", [])
    return cur_cond, z_cur, history, geom_metrics, float(sat_drop.float().mean().cpu())


def build_pair_geometries(prev_rows, cur_rows, kitti_root, grid, depth_candidates):
    from tools.temporal_ray_geometry import build_pair_ray_geometry
    out = []
    metrics = []
    for prev, cur in zip(prev_rows, cur_rows):
        geo = build_pair_ray_geometry(prev, cur, kitti_root=kitti_root, grid=grid, num_depth_candidates=depth_candidates)
        metrics.append(dict(geo.get("metrics", {})))
        out.append(geo)
    return out, metrics


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


def save_args(out_dir, args, base):
    contract = training_contract(args.training_task, args.history_policy, args.history_dropout)
    (out_dir / "args.json").write_text(json.dumps({"args": vars(args), "base_checkpoint": base,
                                                  "training_contract": contract}, indent=2, sort_keys=True))


def main(argv=None):
    args = parse_args(argv)
    contract = training_contract(args.training_task, args.history_policy, args.history_dropout)
    if min(args.steps, args.batch_size, args.save_every, args.keep_checkpoints, args.log_every) < 1:
        raise ValueError("steps, batch, save interval and checkpoint retention must be positive")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.eval_every < 0 or bool(args.fixed_eval_settings) != (args.eval_every > 0):
        raise ValueError("fixed evaluation requires both --fixed-eval-settings and positive --eval-every")
    if args.pair_eval_every < 0 or min(args.pair_eval_count, args.pair_eval_ddim_steps) < 1:
        raise ValueError("pair evaluation count/steps must be positive and interval nonnegative")
    if not 0.0 < args.pair_eval_guidance < float("inf"):
        raise ValueError("pair evaluation guidance must be finite and positive")
    info = rank_info(args); device = info["device"]
    torch.set_num_threads(1)
    torch.manual_seed(args.seed + info["rank"] * 100000)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed + info["rank"] * 100000)
    out_dir = Path(args.out_dir)
    if info["is_main"]:
        out_dir.mkdir(parents=True, exist_ok=False if not args.resume_ckpt else True)
        (out_dir / "checkpoints").mkdir(exist_ok=True); (out_dir / "metrics").mkdir(exist_ok=True)
    if info["distributed"]: torch.distributed.barrier()
    if free_gb(out_dir) < args.min_free_out_gb:
        raise RuntimeError("out dir has %.1fGB free, below %.1f" % (free_gb(out_dir), args.min_free_out_gb))

    from utils.util import instantiate_from_config
    model, base, cfg = load_base(args, device)
    groups = configure_temporal_pair_trainables(model, mode=args.temporal_mode, hidden_dim=args.temporal_hidden_dim)
    if args.init_temporal_ckpt:
        payload = load_temporal_checkpoint(args.init_temporal_ckpt, model, base)
        args.initialized_from_step = int(payload["step"])
        args.initialization_source = {"checkpoint": args.init_temporal_ckpt, "step": args.initialized_from_step}
        del payload
    params = flatten_trainable_groups(groups)
    args.world_size = info["world"]
    if info["distributed"]:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model.DDPM.denoise_model = DDP(model.DDPM.denoise_model, device_ids=[info["local"]], output_device=info["local"],
                                       find_unused_parameters=False, broadcast_buffers=False)
        params = [p for p in model.DDPM.denoise_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(make_optimizer_param_groups(groups, args.temporal_lr, args.decoder_lr))
    scaler = GradScaler(enabled=device.type == "cuda")
    start_step = 0; data_epoch = 0; dataloader_offset = 0
    if args.resume_ckpt:
        payload = load_temporal_checkpoint(args.resume_ckpt, model, base)
        validate_resume_settings(payload["args"], vars(args))
        if payload.get("optimizer") is not None: optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler") is not None: scaler.load_state_dict(payload["scaler"])
        previous_args = payload["args"]
        if "initialization_source" in previous_args:
            args.initialization_source = previous_args["initialization_source"]
        elif previous_args.get("init_temporal_ckpt"):
            args.initialization_source = {"checkpoint": previous_args["init_temporal_ckpt"],
                                          "step": previous_args.get("initialized_from_step")}
        start_step = int(payload["step"]); data_epoch = int(payload.get("epoch", 0)); dataloader_offset = int(payload.get("dataloader_offset", 0))
        del payload

    train_dataset = ConsecutivePairDataset(instantiate_from_config(cfg.data.params.train), kitti_root=args.kitti_root,
                                            grid=(args.latent_grid_height, args.latent_grid_width),
                                            depth_candidates=args.geometry_depth_candidates,
                                            build_geometry=args.history_policy == "correct")
    train_sampler = DistributedSampler(train_dataset, num_replicas=info["world"], rank=info["rank"], shuffle=True, seed=args.seed)
    batch_sampler = TrainingBatchSampler(train_sampler, args.batch_size)
    steps_per_epoch = len(batch_sampler)
    loader = DataLoader(train_dataset, batch_sampler=batch_sampler,
                        num_workers=args.num_workers, collate_fn=collate_pairs, pin_memory=device.type == "cuda",
                        timeout=180 if args.num_workers else 0,
                        **({"prefetch_factor": 1, "persistent_workers": True, "multiprocessing_context": "spawn"} if args.num_workers else {}))
    if steps_per_epoch == 0:
        raise ValueError("batch size leaves no complete training batch")
    if args.epochs is not None:
        args.steps = args.epochs * steps_per_epoch
    if args.steps <= start_step:
        raise ValueError("target training duration must exceed the saved step")
    # Epoch-end checkpoints can resume without re-reading a completed epoch.
    data_epoch += dataloader_offset // steps_per_epoch
    dataloader_offset %= steps_per_epoch
    if info["is_main"]: save_args(out_dir, args, base)
    evaluator = None
    if args.fixed_eval_settings:
        from tools.temporal_fixed_eval import FixedTemporalEvaluator
        evaluator = FixedTemporalEvaluator(model, cfg, args, info, out_dir, base=base)
        evaluator.evaluate(start_step, start_step / steps_per_epoch)
    pair_evaluator = None
    if args.pair_eval_every:
        from tools.eval_temporal_gt_pairs import GTPairEvaluator
        pair_evaluator = GTPairEvaluator(model, cfg, args, info, out_dir, base=base)
        pair_evaluator.evaluate(start_step, start_step / steps_per_epoch)
    train_sampler.set_epoch(data_epoch)
    batch_sampler.set_window(dataloader_offset, min(steps_per_epoch, dataloader_offset + args.steps - start_step))
    iterator = iter(loader)
    log_path = out_dir / "metrics" / ("rank%d.jsonl" % info["rank"])
    t0 = time.time()
    if info["is_main"]:
        print(json.dumps({"pairs": len(train_dataset), "world": info["world"], "trainable_params": sum(p.numel() for p in params), "temporal_params": sum(p.numel() for p in groups["temporal"]), "decoder_params": sum(p.numel() for p in groups["decoder"]), "training_contract": contract, "steps_per_epoch": steps_per_epoch, "total_steps": args.steps}, sort_keys=True), flush=True)

    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator); dataloader_offset += 1
        except StopIteration:
            data_epoch += 1; dataloader_offset = 0
            train_sampler.set_epoch(data_epoch)
            batch_sampler.set_window(0, min(steps_per_epoch, args.steps - step + 1))
            iterator = iter(loader); batch = next(iterator); dataloader_offset = 1
        step_seed = args.seed + step * 1009 + info["rank"] * 1000003
        seed_training_step(step_seed, device)
        cur_cond, z_cur, history, geom_metrics, sat_dropout_fraction = prepare_training_pair(
            model, batch, args, device, step_seed)
        def loss_closure():
            with autocast(enabled=device.type == "cuda"):
                loss, out = epsilon_prediction_loss(model.DDPM, z_cur, cur_cond, history, step_seed)
                return anchor_trainable_loss(loss, params), out

        def record_amp_retry(event):
            event.update(step=step, rank=info["rank"], step_seed=step_seed)
            retry_path = out_dir / "metrics" / ("amp_rank%d.jsonl" % info["rank"])
            with retry_path.open("a") as handle: handle.write(json.dumps(event, sort_keys=True) + "\n")
            if info["is_main"]: print(json.dumps(event, sort_keys=True), flush=True)

        loss, out, grad_norm, amp_retries = optimizer_step_with_retry(
            loss_closure, params, optimizer, scaler, on_retry=record_amp_retry)
        rec = {"step": step, "rank": info["rank"], "loss": float(loss.detach().cpu()), "grad_norm": grad_norm,
               "amp_scale": scaler.get_scale(), "amp_retries": amp_retries,
               "t_mean": float(out["t"].float().mean().cpu()), "history_enabled": int(history["enabled"].sum().detach().cpu()) if history is not None else 0,
               "satellite_dropout_fraction": sat_dropout_fraction, "training_task": args.training_task,
               "history_source": contract["history_source"], "target_source": contract["target_source"],
               "batch": int(z_cur.shape[0]), "epoch": data_epoch, "epoch_fraction": step / steps_per_epoch, "dataloader_offset": dataloader_offset,
               "base_sha256": base.get("sha256"), "geo0": geom_metrics[0] if geom_metrics else {}, "sec": round(time.time() - t0, 2)}
        with log_path.open("a") as handle: handle.write(json.dumps(rec, sort_keys=True) + "\n")
        if info["is_main"] and (step == 1 or step % args.log_every == 0): print(json.dumps(rec, sort_keys=True), flush=True)
        if evaluator is not None and (step % args.eval_every == 0 or step == args.steps):
            optimizer.zero_grad(set_to_none=True)
            evaluator.evaluate(step, step / steps_per_epoch)
        if pair_evaluator is not None and (step % args.pair_eval_every == 0 or step == args.steps):
            optimizer.zero_grad(set_to_none=True)
            pair_evaluator.evaluate(step, step / steps_per_epoch)
        if step % args.save_every == 0:
            if info["is_main"]:
                ckpt_path = out_dir / "checkpoints" / ("temporal_pair_step_%07d.pt" % step)
                save_temporal_checkpoint(ckpt_path, model, optimizer, scaler, step, base, vars(args), data_epoch, dataloader_offset)
                prune_checkpoints(out_dir / "checkpoints", args.keep_checkpoints)
                latest = out_dir / "checkpoints" / "latest.pt"
                if latest.exists() or latest.is_symlink(): latest.unlink()
                latest.symlink_to(ckpt_path.name)
            if info["distributed"]:
                torch.distributed.barrier()
    shutdown_training_loader(iterator)
    if info["is_main"]:
        final_path = out_dir / "checkpoints" / ("temporal_pair_step_%07d.pt" % args.steps)
        if not final_path.exists():
            save_temporal_checkpoint(final_path, model, optimizer, scaler, args.steps, base, vars(args), data_epoch, dataloader_offset)
        prune_checkpoints(out_dir / "checkpoints", args.keep_checkpoints)
        latest = out_dir / "checkpoints" / "latest.pt"
        if latest.exists() or latest.is_symlink(): latest.unlink()
        latest.symlink_to(final_path.name)
        print(json.dumps({"done": True, "steps": args.steps}, sort_keys=True), flush=True)
    if info["distributed"]:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
