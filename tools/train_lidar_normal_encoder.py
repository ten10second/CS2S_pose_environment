import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_lidar_normal import KittiLidarNormalDataset, collate_lidar_normal  # noqa: E402
from models.KITTI_geo_ldm.lidar_normal_encoder import (  # noqa: E402
    LidarNormalEncoder,
    sign_invariant_angular_error,
    sign_invariant_normal_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LiDAR encoder with per-point surface-normal distillation.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--label-root", default="dataset/kitti_lidar_normal_labels")
    parser.add_argument("--run-name", default="normal_encoder")
    parser.add_argument("--out-root", default="results/lidar_normal_distill")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--feature-channels", type=int, default=128)
    parser.add_argument("--max-points-per-sample", type=int, default=12000)
    parser.add_argument("--min-points", type=int, default=64)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--vis-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def move_to_cuda(batch):
    out = {}
    for key, value in batch.items():
        out[key] = value.cuda(non_blocking=True) if torch.is_tensor(value) else value
    return out


def train_step(model, batch, optimizer):
    model.train()
    output = model(
        batch["points_rect"],
        batch["intensity"],
        batch["batch_index"],
        batch_size=len(batch["sample_id"]),
    )
    valid = output["valid_mask"] & (batch["label_weight"] > 0.0)
    loss = sign_invariant_normal_loss(
        output["pred_normal_rect"],
        batch["normal_rect"],
        batch["label_weight"],
        valid,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    with torch.no_grad():
        angle = sign_invariant_angular_error(output["pred_normal_rect"], batch["normal_rect"], valid)
    return loss.detach(), angle.detach(), int(valid.sum().detach().cpu())


@torch.no_grad()
def evaluate(model, loader, max_batches):
    model.eval()
    losses = []
    angles = []
    valid_points = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = move_to_cuda(batch)
        output = model(
            batch["points_rect"],
            batch["intensity"],
            batch["batch_index"],
            batch_size=len(batch["sample_id"]),
        )
        valid = output["valid_mask"] & (batch["label_weight"] > 0.0)
        loss = sign_invariant_normal_loss(
            output["pred_normal_rect"],
            batch["normal_rect"],
            batch["label_weight"],
            valid,
        )
        angle = sign_invariant_angular_error(output["pred_normal_rect"], batch["normal_rect"], valid)
        losses.append(loss.detach().cpu())
        angles.append(angle.detach().cpu())
        valid_points += int(valid.sum().detach().cpu())
    if not losses:
        return {"val_loss": float("nan"), "val_angle_mean": float("nan"), "val_angle_median": float("nan"), "val_points": 0}
    angle_all = torch.cat(angles, dim=0) if angles else torch.empty(0)
    return {
        "val_loss": float(torch.stack(losses).mean()),
        "val_angle_mean": float(angle_all.mean()) if angle_all.numel() else float("nan"),
        "val_angle_median": float(angle_all.median()) if angle_all.numel() else float("nan"),
        "val_angle_p90": float(torch.quantile(angle_all, 0.9)) if angle_all.numel() else float("nan"),
        "val_points": valid_points,
    }


def _normal_rgb(normals):
    return np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _scatter_canvas(uv, values, image_shape, radius=1):
    h, w = int(image_shape[0]), int(image_shape[1])
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            xx = np.clip(x + dx, 0, w - 1)
            yy = np.clip(y + dy, 0, h - 1)
            canvas[yy, xx] = values
    return canvas


@torch.no_grad()
def save_visualization(model, loader, out_dir, step):
    model.eval()
    batch = next(iter(loader))
    batch_cuda = move_to_cuda(batch)
    output = model(
        batch_cuda["points_rect"],
        batch_cuda["intensity"],
        batch_cuda["batch_index"],
        batch_size=len(batch_cuda["sample_id"]),
    )
    offsets = batch["sample_offsets"].cpu().numpy()
    idx0, idx1 = int(offsets[0]), int(offsets[1])
    gt = batch["normal_rect"][idx0:idx1].cpu().numpy()
    pred = output["pred_normal_rect"][idx0:idx1].detach().cpu().numpy()
    uv = batch["uv_orig"][idx0:idx1].cpu().numpy()
    image_shape = batch["image_shape"][0].cpu().numpy()
    try:
        image = Image.open(batch["image_02_path"][0]).convert("RGB")
    except FileNotFoundError:
        image = Image.new("RGB", (int(image_shape[1]), int(image_shape[0])), (0, 0, 0))
    gt_canvas = Image.fromarray(_scatter_canvas(uv, _normal_rgb(gt), image_shape))
    pred_canvas = Image.fromarray(_scatter_canvas(uv, _normal_rgb(pred), image_shape))
    dot = np.abs((gt * pred).sum(axis=1)).clip(0.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    err_color = np.zeros((angle.shape[0], 3), dtype=np.uint8)
    err_color[:, 0] = np.clip(angle / 90.0 * 255.0, 0, 255).astype(np.uint8)
    err_color[:, 1] = np.clip((1.0 - angle / 90.0) * 255.0, 0, 255).astype(np.uint8)
    err_canvas = Image.fromarray(_scatter_canvas(uv, err_color, image_shape))

    target_w = 512
    scale = target_w / image.width
    target_h = max(1, int(image.height * scale))
    panels = [
        image.resize((target_w, target_h), Image.BILINEAR),
        gt_canvas.resize((target_w, target_h), Image.NEAREST),
        pred_canvas.resize((target_w, target_h), Image.NEAREST),
        err_canvas.resize((target_w, target_h), Image.NEAREST),
    ]
    panel = Image.new("RGB", (target_w * len(panels), target_h))
    for i, item in enumerate(panels):
        panel.paste(item, (i * target_w, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = batch["sample_id"][0].replace("/", "_")
    panel.save(out_dir / f"step_{step:06d}_{safe_id}.png")


def save_checkpoint(path, model, optimizer, step, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_root) / args.run_name
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "visualizations"
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    train_data = KittiLidarNormalDataset(
        args.train_manifest,
        args.label_root,
        min_points=args.min_points,
        max_points_per_sample=args.max_points_per_sample,
    )
    val_data = KittiLidarNormalDataset(
        args.val_manifest,
        args.label_root,
        min_points=args.min_points,
        max_points_per_sample=args.max_points_per_sample,
    )
    if args.overfit_samples > 0:
        count = min(args.overfit_samples, len(train_data))
        train_data = Subset(train_data, list(range(count)))
        val_data = Subset(train_data, list(range(count)))
    if len(train_data) == 0 or len(val_data) == 0:
        raise SystemExit(f"Empty train/val normal dataset: train={len(train_data)} val={len(val_data)}")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_lidar_normal,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        collate_fn=collate_lidar_normal,
        pin_memory=True,
        drop_last=False,
    )

    model = LidarNormalEncoder(
        hidden_channels=args.hidden_channels,
        feature_channels=args.feature_channels,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))

    metrics_path = metrics_dir / "train_metrics.jsonl"
    data_iter = iter(train_loader)
    with metrics_path.open("a" if args.resume else "w") as metrics_file:
        for step in range(start_step + 1, args.steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = move_to_cuda(batch)
            loss, angle, valid_points = train_step(model, batch, optimizer)
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "angle_mean": float(angle.mean().detach().cpu()) if angle.numel() else float("nan"),
                "angle_median": float(angle.median().detach().cpu()) if angle.numel() else float("nan"),
                "valid_points": valid_points,
                "batch_size": args.batch_size,
            }
            if step % args.eval_every == 0 or step == 1:
                record.update(evaluate(model, val_loader, args.val_batches))
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_file.flush()
            if step % args.log_every == 0 or step == 1:
                print(json.dumps(record, sort_keys=True), flush=True)
            if args.vis_every > 0 and (step % args.vis_every == 0 or step == 1):
                save_visualization(model, val_loader, vis_dir, step)
            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step, args)
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, step, args)
    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, args.steps, args)


if __name__ == "__main__":
    main()
