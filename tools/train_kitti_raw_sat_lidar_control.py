import argparse
import gc
import json
import sys
from itertools import cycle
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.util import instantiate_from_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight KITTI raw sat-lidar training loop.")
    parser.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dynamic.yaml")
    parser.add_argument("--condition-mode", default="dynamic_full", choices=["none", "bbox_dynamic", "dynamic_points", "raw_lidar", "dynamic_full"])
    parser.add_argument("--run-name", default="")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--base-model-ckpt", default="", help="Optional full-model checkpoint used as the frozen base for control modes.")
    parser.add_argument("--resume-ckpt", default="", help="Resume model/optimizer state and continue from checkpoint step.")
    parser.add_argument("--resume-weights-only", action="store_true", help="Load checkpoint weights and step but start a fresh optimizer.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the train manifest instead of iterating drives in manifest order.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pinned memory.")
    parser.add_argument("--dynamic-loss-weight", type=float, default=0.0, help="Extra latent diffusion MSE weight on dynamic_mask pixels.")
    parser.add_argument("--dynamic-x0-loss-weight", type=float, default=0.0, help="Extra dynamic-mask latent x0 reconstruction loss weight.")
    parser.add_argument("--dynamic-image-loss-weight", type=float, default=0.0, help="Extra dynamic-mask image-space x0 L1 loss weight.")
    parser.add_argument("--dynamic-crop-image-loss-weight", type=float, default=0.0, help="Extra image-space x0 L1 loss on a padded dynamic crop mask.")
    parser.add_argument("--dynamic-crop-padding", type=int, default=8, help="Pixel padding used to dilate dynamic boxes for --dynamic-crop-image-loss-weight.")
    parser.add_argument("--dynamic-point-loss-weight", type=float, default=0.0, help="Extra diffusion MSE on dilated dynamic LiDAR point pixels.")
    parser.add_argument("--dynamic-point-x0-loss-weight", type=float, default=0.0, help="Extra latent x0 loss on dilated dynamic LiDAR point pixels.")
    parser.add_argument("--dynamic-point-image-loss-weight", type=float, default=0.0, help="Extra image-space x0 L1 loss on dilated dynamic LiDAR point pixels.")
    parser.add_argument("--dynamic-point-dilation", type=int, default=4, help="Image-pixel dilation radius for dynamic point loss masks.")
    parser.add_argument("--dynamic-object-lpips-weight", type=float, default=0.0, help="Extra LPIPS loss on projected dynamic object crops from predicted x0.")
    parser.add_argument("--dynamic-object-lpips-padding", type=int, default=6, help="Pixel padding around projected object boxes for crop LPIPS.")
    parser.add_argument("--dynamic-object-lpips-size", type=int, default=64, help="Resize object crops to this square size before LPIPS; <=0 keeps native crop size.")
    parser.add_argument("--dynamic-object-lpips-max-boxes", type=int, default=4, help="Maximum dynamic object crops per image used by LPIPS loss.")
    parser.add_argument("--dynamic-oversample-factor", type=float, default=1.0, help="Sample frames with manifest num_dynamic_boxes > 0 this many times more often.")
    parser.add_argument("--min-dynamic-points", type=int, default=0, help="When >0, redraw batches until this many projected dynamic points are present.")
    parser.add_argument("--min-dynamic-mask-coverage", type=float, default=0.0, help="When >0, redraw batches until dynamic_mask coverage reaches this value.")
    parser.add_argument("--max-sample-attempts", type=int, default=1, help="Maximum draws per optimizer step when dynamic batch filters are enabled.")
    parser.add_argument("--control-hidden-channels", type=int, default=128, help="Hidden channels for multi-scale LiDAR control.")
    parser.add_argument("--control-scale", type=float, default=1.0, help="Multiplier applied to LiDAR control residuals.")
    parser.add_argument("--control-semantic-class-count", type=int, default=0, help="When >0, add trainable per-class embeddings to multiscale LiDAR control.")
    parser.add_argument("--control-semantic-class-scale", type=float, default=1.0, help="Scale for per-class LiDAR control embeddings.")
    parser.add_argument("--dynamic-class-token-weight", type=float, default=0.0, help="Append a trainable dynamic class summary token to cross-attention when > 0.")
    parser.add_argument("--lidar-unfreeze-output-blocks", type=int, default=0, help="Also train this many final UNet output blocks in LiDAR control mode.")
    parser.add_argument("--lidar-unfreeze-out", action="store_true", help="Also train the final UNet output layer in LiDAR control mode.")
    parser.add_argument("--lidar-unfreeze-transformers", default="none", choices=["none", "input", "middle", "output", "output_middle", "all"], help="Also train selected UNet SpatialTransformer modules in LiDAR control mode.")
    parser.add_argument("--lidar-unet-lr-scale", type=float, default=0.25, help="LR multiplier for locally unfrozen UNet params.")
    parser.add_argument("--out-root", default="results/sat_lidar_dynamic")
    return parser.parse_args()


def move_batch_to_cuda(batch):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.cuda(non_blocking=True) if torch.is_tensor(value) else value
    return moved


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def configure_for_mode(cfg, mode, batch_size, num_workers):
    cfg.data.params.batch_size = batch_size
    cfg.data.params.num_workers = num_workers
    cfg.data.params.train.params.condition_mode = mode
    cfg.data.params.test.params.condition_mode = mode
    if mode == "none":
        cfg.model.params.use_lidar_cond = False
        cfg.model.params.freeze_for_lidar_control = False
    else:
        cfg.model.params.use_lidar_cond = True
        cfg.model.params.freeze_for_lidar_control = True
    return cfg


def configure_multiscale_control(cfg, hidden_channels, control_scale, semantic_class_count=0, semantic_class_scale=1.0):
    control = cfg.model.params.DDPM_config.params.control_grd
    unet = cfg.model.params.DDPM_config.params.unet_config.params
    control.target = "models.KITTI_geo_ldm.lidar_condition_model.LidarMultiScaleControl"
    control.params.in_channels = 4
    control.params.model_channels = unet.model_channels
    control.params.channel_mult = list(unet.channel_mult)
    control.params.num_res_blocks = unet.num_res_blocks
    control.params.hidden_channels = hidden_channels
    control.params.middle_channels = unet.model_channels * list(unet.channel_mult)[-1]
    control.params.control_scale = control_scale
    control.params.semantic_class_count = int(semantic_class_count)
    control.params.semantic_class_scale = float(semantic_class_scale)
    return cfg


def load_full_model_checkpoint(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu")
    if "model" in payload:
        model.load_state_dict(payload["model"], strict=False)
        del payload
        gc.collect()
        return
    if "state_dict" in payload:
        model.load_state_dict(payload["state_dict"], strict=False)
        del payload
        gc.collect()
        return
    raise ValueError(f"Unsupported full-model checkpoint format: {ckpt_path}")


def checkpoint_base_model_path(ckpt_path):
    if not ckpt_path:
        return ""
    payload = torch.load(ckpt_path, map_location="cpu")
    base_model_ckpt = payload.get("base_model_ckpt", "")
    del payload
    gc.collect()
    return base_model_ckpt


def load_training_checkpoint(model, optimizer, ckpt_path, mode, load_optimizer=True):
    payload = torch.load(ckpt_path, map_location="cpu")
    if mode == "none" and "model" in payload:
        model.load_state_dict(payload["model"], strict=False)
    elif "control_grd" in payload:
        model.DDPM.control_grd.load_state_dict(payload["control_grd"], strict=False)
    else:
        raise ValueError(f"Unsupported training checkpoint format: {ckpt_path}")
    if load_optimizer and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if "dynamic_class_tokens" in payload and getattr(model, "dynamic_class_tokens", None) is not None:
        model.dynamic_class_tokens.data.copy_(payload["dynamic_class_tokens"])
    if "denoise_model_trainable" in payload:
        model.DDPM.denoise_model.load_state_dict(payload["denoise_model_trainable"], strict=False)
    step = int(payload.get("step", 0))
    base_model_ckpt = payload.get("base_model_ckpt", "")
    del payload
    gc.collect()
    return step, base_model_ckpt


def save_checkpoint(path, model, optimizer, step, mode, base_model_ckpt=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "condition_mode": mode,
        "base_model_ckpt": base_model_ckpt,
        "control_grd": model.DDPM.control_grd.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if mode == "none":
        payload["model"] = model.state_dict()
    if getattr(model, "dynamic_class_tokens", None) is not None:
        payload["dynamic_class_tokens"] = model.dynamic_class_tokens.detach().cpu()
    denoise_trainable = {
        name: param.detach().cpu()
        for name, param in model.DDPM.denoise_model.named_parameters()
        if param.requires_grad
    }
    if denoise_trainable:
        payload["denoise_model_trainable"] = denoise_trainable
    torch.save(payload, path)


def unwrap_records(dataset):
    base = dataset
    while hasattr(base, "data") and not hasattr(base, "records"):
        base = base.data
    return getattr(base, "records", None)


def build_sample_order(dataset, shuffle, seed, start_step, batch_size, dynamic_oversample_factor):
    if len(dataset) <= 0:
        raise ValueError("Training dataset is empty.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    if dynamic_oversample_factor > 1.0:
        records = unwrap_records(dataset)
        if records is None:
            raise ValueError("--dynamic-oversample-factor requires a dataset with manifest records.")
        weights = torch.tensor(
            [dynamic_oversample_factor if int(record.get("num_dynamic_boxes", 0)) > 0 else 1.0 for record in records],
            dtype=torch.float,
        )
        order = torch.multinomial(weights, num_samples=len(dataset), replacement=True, generator=generator).tolist()
    elif shuffle:
        order = torch.randperm(len(dataset), generator=generator).tolist()
    else:
        order = list(range(len(dataset)))
    start_offset = (max(start_step, 0) * batch_size) % len(order)
    return order[start_offset:] + order[:start_offset]


def build_train_loader(dataset, batch_size, num_workers, shuffle, seed, start_step, pin_memory, dynamic_oversample_factor):
    if shuffle:
        dataset = Subset(dataset, build_sample_order(dataset, shuffle, seed, start_step, batch_size, dynamic_oversample_factor))
    elif dynamic_oversample_factor > 1.0:
        dataset = Subset(dataset, build_sample_order(dataset, True, seed, start_step, batch_size, dynamic_oversample_factor))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )


def trim_metrics_for_resume(metrics_path, start_step):
    if not metrics_path.exists():
        return
    kept = []
    for line in metrics_path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(record.get("step", -1)) <= start_step:
            kept.append(line)
    metrics_path.write_text(("\n".join(kept) + "\n") if kept else "")


def main():
    args = parse_args()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(args.config)
    cfg = configure_for_mode(cfg, args.condition_mode, args.batch_size, args.num_workers)
    cfg = configure_multiscale_control(
        cfg,
        args.control_hidden_channels,
        args.control_scale,
        args.control_semantic_class_count,
        args.control_semantic_class_scale,
    )
    cfg.model.params.dynamic_loss_weight = args.dynamic_loss_weight
    cfg.model.params.dynamic_x0_loss_weight = args.dynamic_x0_loss_weight
    cfg.model.params.dynamic_image_loss_weight = args.dynamic_image_loss_weight
    cfg.model.params.dynamic_crop_image_loss_weight = args.dynamic_crop_image_loss_weight
    cfg.model.params.dynamic_crop_padding = args.dynamic_crop_padding
    cfg.model.params.dynamic_point_loss_weight = args.dynamic_point_loss_weight
    cfg.model.params.dynamic_point_x0_loss_weight = args.dynamic_point_x0_loss_weight
    cfg.model.params.dynamic_point_image_loss_weight = args.dynamic_point_image_loss_weight
    cfg.model.params.dynamic_point_dilation = args.dynamic_point_dilation
    cfg.model.params.dynamic_object_lpips_weight = args.dynamic_object_lpips_weight
    cfg.model.params.dynamic_object_lpips_padding = args.dynamic_object_lpips_padding
    cfg.model.params.dynamic_object_lpips_size = args.dynamic_object_lpips_size
    cfg.model.params.dynamic_object_lpips_max_boxes = args.dynamic_object_lpips_max_boxes
    cfg.model.params.dynamic_class_token_weight = args.dynamic_class_token_weight
    cfg.model.params.lidar_unfreeze_output_blocks = args.lidar_unfreeze_output_blocks
    cfg.model.params.lidar_unfreeze_out = args.lidar_unfreeze_out
    cfg.model.params.lidar_unfreeze_transformers = args.lidar_unfreeze_transformers
    cfg.model.params.lidar_unet_lr_scale = args.lidar_unet_lr_scale
    run_name = args.run_name or f"{args.condition_mode}_smoke_{args.steps}step"
    out_dir = Path(args.out_root) / run_name
    metrics_dir = out_dir / "metrics"
    ckpt_dir = out_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    if args.resume_ckpt and not args.base_model_ckpt:
        args.base_model_ckpt = checkpoint_base_model_path(args.resume_ckpt)

    model = instantiate_from_config(cfg.model).cuda()
    if args.base_model_ckpt:
        load_full_model_checkpoint(model, args.base_model_ckpt)
    model.learning_rate = args.lr
    optimizer = model.configure_optimizers()[0]
    start_step = 0
    resume_base_model_ckpt = ""
    if args.resume_ckpt:
        start_step, resume_base_model_ckpt = load_training_checkpoint(
            model,
            optimizer,
            args.resume_ckpt,
            args.condition_mode,
            load_optimizer=not args.resume_weights_only,
        )
        if resume_base_model_ckpt and args.base_model_ckpt and resume_base_model_ckpt != args.base_model_ckpt:
            raise ValueError(
                f"Resume checkpoint base_model_ckpt differs from --base-model-ckpt: {resume_base_model_ckpt} vs {args.base_model_ckpt}"
            )
        if resume_base_model_ckpt and not args.base_model_ckpt:
            args.base_model_ckpt = resume_base_model_ckpt
        print(f"resumed {args.resume_ckpt} from step {start_step}")
    if start_step >= args.steps:
        raise SystemExit(f"resume step {start_step} already >= target steps {args.steps}")
    model.train()

    data = instantiate_from_config(cfg.data)
    data.setup()
    loader = build_train_loader(
        data.datasets["train"],
        args.batch_size,
        args.num_workers,
        args.shuffle,
        args.seed,
        start_step,
        args.pin_memory,
        args.dynamic_oversample_factor,
    )
    iterator = cycle(loader)

    metrics_path = metrics_dir / "train_metrics.jsonl"
    if args.resume_ckpt:
        trim_metrics_for_resume(metrics_path, start_step)
    with metrics_path.open("a" if args.resume_ckpt else "w") as metrics_file:
        for step in range(start_step + 1, args.steps + 1):
            sample_attempts = 0
            while True:
                batch = move_batch_to_cuda(next(iterator))
                sample_attempts += 1
                num_dynamic_points = int(batch["num_projected_dynamic_points"].sum().detach().cpu()) if "num_projected_dynamic_points" in batch else 0
                dynamic_mask_coverage = float(batch["dynamic_mask"].float().mean().detach().cpu()) if "dynamic_mask" in batch else 0.0
                enough_points = num_dynamic_points >= args.min_dynamic_points
                enough_mask = dynamic_mask_coverage >= args.min_dynamic_mask_coverage
                if (enough_points and enough_mask) or sample_attempts >= max(1, args.max_sample_attempts):
                    break
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_step(batch, step)
            loss.backward()
            optimizer.step()

            record = {
                "step": step,
                "condition_mode": args.condition_mode,
                "loss": scalar(loss),
                "num_dynamic_boxes": int(batch["num_dynamic_boxes"].sum().detach().cpu()) if "num_dynamic_boxes" in batch else 0,
                "num_projected_dynamic_points": num_dynamic_points,
                "dynamic_mask_coverage": dynamic_mask_coverage,
                "dynamic_loss_weight": args.dynamic_loss_weight,
                "dynamic_x0_loss_weight": args.dynamic_x0_loss_weight,
                "dynamic_image_loss_weight": args.dynamic_image_loss_weight,
                "dynamic_crop_image_loss_weight": args.dynamic_crop_image_loss_weight,
                "dynamic_crop_padding": args.dynamic_crop_padding,
                "dynamic_point_loss_weight": args.dynamic_point_loss_weight,
                "dynamic_point_x0_loss_weight": args.dynamic_point_x0_loss_weight,
                "dynamic_point_image_loss_weight": args.dynamic_point_image_loss_weight,
                "dynamic_point_dilation": args.dynamic_point_dilation,
                "dynamic_object_lpips_weight": args.dynamic_object_lpips_weight,
                "dynamic_object_lpips_padding": args.dynamic_object_lpips_padding,
                "dynamic_object_lpips_size": args.dynamic_object_lpips_size,
                "dynamic_object_lpips_max_boxes": args.dynamic_object_lpips_max_boxes,
                "dynamic_oversample_factor": args.dynamic_oversample_factor,
                "dynamic_class_token_weight": args.dynamic_class_token_weight,
                "control_semantic_class_count": args.control_semantic_class_count,
                "control_semantic_class_scale": args.control_semantic_class_scale,
                "lidar_unfreeze_output_blocks": args.lidar_unfreeze_output_blocks,
                "lidar_unfreeze_out": args.lidar_unfreeze_out,
                "lidar_unfreeze_transformers": args.lidar_unfreeze_transformers,
                "lidar_unet_lr_scale": args.lidar_unet_lr_scale,
                "min_dynamic_points": args.min_dynamic_points,
                "min_dynamic_mask_coverage": args.min_dynamic_mask_coverage,
                "sample_attempts": sample_attempts,
            }
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_file.flush()
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == args.steps):
                print(json.dumps(record, sort_keys=True))

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step, args.condition_mode, args.base_model_ckpt)

    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, args.steps, args.condition_mode, args.base_model_ckpt)
    print(f"saved {metrics_path}")
    print(f"saved {ckpt_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
