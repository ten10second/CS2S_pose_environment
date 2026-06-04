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
from dataloader.kitti_raw_lidar_utils import lidar_condition_channels, lidar_condition_gate_channel  # noqa: E402
from models.eval.dynamic_metrics import dynamic_masked_metrics  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402


CONDITION_MODES = (
    "none",
    "bbox_dynamic",
    "dynamic_points",
    "raw_lidar",
    "dynamic_full",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test KITTI raw sat-lidar data and optional model backward.")
    parser.add_argument("--manifest", default="dataset/kitti_raw_sat_lidar/train_manifest.jsonl")
    parser.add_argument("--condition-mode", default="raw_lidar", choices=CONDITION_MODES)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dynamic.yaml")
    parser.add_argument("--model-backward", action="store_true")
    return parser.parse_args()


def move_batch_to_cuda(batch):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.cuda() if torch.is_tensor(value) else value
    return moved


def configure_smoke_model(cfg, mode):
    cfg.data.params.batch_size = 1
    cfg.data.params.num_workers = 0
    cfg.data.params.train.params.condition_mode = mode
    cfg.data.params.test.params.condition_mode = mode
    cfg.model.params.use_lidar_cond = mode != "none"
    cfg.model.params.freeze_for_lidar_control = mode != "none"
    raw_geometry_modes = set()
    semantic_free_modes = {"none", "dynamic_points"}
    geometry_gate_channel = lidar_condition_gate_channel(mode)
    cfg.model.params.dynamic_class_token_weight = 0.0 if mode in semantic_free_modes else cfg.model.params.get("dynamic_class_token_weight", 0.0)
    cfg.model.params.static_teacher_consistency_weight = 0.25 if mode in raw_geometry_modes else 0.0
    cfg.model.params.static_teacher_gate_channel = geometry_gate_channel
    if mode in raw_geometry_modes:
        cfg.model.params.dynamic_loss_weight = 0.0
        cfg.model.params.dynamic_x0_loss_weight = 0.0
        cfg.model.params.dynamic_point_loss_weight = 0.0
        cfg.model.params.dynamic_point_x0_loss_weight = 0.0
    control = cfg.model.params.DDPM_config.params.control_grd
    unet = cfg.model.params.DDPM_config.params.unet_config.params
    control.target = "models.KITTI_geo_ldm.lidar_condition_model.LidarMultiScaleControl"
    control.params.in_channels = lidar_condition_channels(mode)
    control.params.model_channels = unet.model_channels
    control.params.channel_mult = list(unet.channel_mult)
    control.params.num_res_blocks = unet.num_res_blocks
    control.params.middle_channels = unet.model_channels * list(unet.channel_mult)[-1]
    control.params.semantic_class_count = 0 if mode in semantic_free_modes else int(control.params.get("semantic_class_count", 0))
    control.params.gate_channel = geometry_gate_channel
    control.params.gate_residuals = geometry_gate_channel >= 0
    return cfg


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
        "camera_imu_forward_right": batch["camera_imu_forward_right"].tolist() if "camera_imu_forward_right" in batch else [],
        "dynamic_mask_coverage": float(batch["dynamic_mask"].float().mean()),
        "lidar_cond_nonzero": int((batch["lidar_cond"] != 0).sum()),
        "metric_smoke": {
            key: (float(value) if value.dtype.is_floating_point else int(value))
            for key, value in metrics.items()
        },
    }
    if args.model_backward:
        cfg = configure_smoke_model(OmegaConf.load(args.config), args.condition_mode)
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
