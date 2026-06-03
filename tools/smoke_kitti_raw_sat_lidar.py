import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from models.eval.dynamic_metrics import dynamic_masked_metrics  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test KITTI raw sat-lidar data and optional model backward.")
    parser.add_argument("--manifest", default="dataset/kitti_raw_sat_lidar/train_manifest.jsonl")
    parser.add_argument("--condition-mode", default="dynamic_full")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dynamic.yaml")
    parser.add_argument("--model-backward", action="store_true")
    return parser.parse_args()


def move_batch_to_cuda(batch):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.cuda() if torch.is_tensor(value) else value
    return moved


def main():
    args = parse_args()
    ds = SatLidarRawDataset(args.manifest, condition_mode=args.condition_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    fake = torch.rand_like(batch["grd_left_imgs"])
    metrics = dynamic_masked_metrics(fake, batch["grd_left_imgs"], batch["dynamic_mask"])
    report = {
        "manifest": args.manifest,
        "condition_mode": args.condition_mode,
        "batch_size": args.batch_size,
        "sat_map_shape": list(batch["sat_map"].shape),
        "grd_left_imgs_shape": list(batch["grd_left_imgs"].shape),
        "lidar_cond_shape": list(batch["lidar_cond"].shape),
        "dynamic_mask_shape": list(batch["dynamic_mask"].shape),
        "sample_ids": list(batch["sample_id"]),
        "num_dynamic_boxes": [int(x) for x in batch["num_dynamic_boxes"]],
        "num_projected_dynamic_points": [int(x) for x in batch["num_projected_dynamic_points"]],
        "dynamic_mask_coverage": float(batch["dynamic_mask"].float().mean()),
        "metric_smoke": {
            key: (float(value) if value.dtype.is_floating_point else int(value))
            for key, value in metrics.items()
        },
    }

    if args.model_backward:
        cfg = OmegaConf.load(args.config)
        cfg.data.params.batch_size = 1
        cfg.data.params.num_workers = 0
        model = instantiate_from_config(cfg.model).cuda()
        model.learning_rate = 7e-5
        model.configure_optimizers()
        one = {key: (value[:1] if torch.is_tensor(value) else value[:1]) for key, value in batch.items()}
        one = move_batch_to_cuda(one)
        loss = model.training_step(one, 0)
        loss.backward()
        control_grad = 0.0
        for param in model.DDPM.control_grd.parameters():
            if param.grad is not None:
                control_grad += float(param.grad.detach().abs().sum())
        report["model_backward"] = {
            "loss": float(loss.detach()),
            "control_grad_sum": control_grad,
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
