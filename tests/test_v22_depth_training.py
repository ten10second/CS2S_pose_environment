import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from tools.train_kitti_raea import configure_cfg, load_training_checkpoint, parse_args


ROOT = Path(__file__).resolve().parents[1]


class V22DepthTrainingTests(unittest.TestCase):
    def test_training_configuration_selects_native_head_and_preserves_v21(self):
        with patch("sys.argv", ["train"]):
            args = parse_args()
        for filename, mode, output, bottleneck in (
            ("KITTI_raw_sat_lidar_pixel_cfgdrop10.yaml", "masked_area", 0.0, 1.0),
            ("KITTI_raw_sat_lidar_pixel_v22_cfgdrop10.yaml", "native", 1.0, 0.0),
        ):
            cfg = OmegaConf.load(ROOT / "configs/Boost_Sat2Den/train" / filename)
            effective = configure_cfg(cfg, args).model.params
            self.assertEqual(effective.lidar_depth_resample_mode, mode)
            self.assertEqual(effective.lidar_depth_output_scale, output)
            self.assertEqual(effective.lidar_depth_bottleneck_scale, bottleneck)
            self.assertEqual(effective.satellite_condition_dropout_prob, 0.1)

    @staticmethod
    def fake_model():
        denoiser = torch.nn.Module()
        denoiser.body = torch.nn.Linear(2, 2)
        denoiser.lidar_pixel_depth_head = torch.nn.Linear(2, 1)
        denoiser.lidar_depth_head_mode = "pixel"
        return SimpleNamespace(
            DDPM=SimpleNamespace(denoise_model=denoiser),
            condition_model_sat=torch.nn.Linear(2, 2),
            lidar_context_model=torch.nn.Linear(2, 2),
        )

    def test_init_ckpt_is_not_a_v22_training_entrypoint(self):
        with patch("sys.argv", ["train", "--init-ckpt", "/tmp/v21.pt"]):
            with self.assertRaises(SystemExit):
                parse_args()

    def test_strict_v22_resume_restores_same_version_training_state_only(self):
        model = self.fake_model()
        optimizer = torch.optim.Adam(model.DDPM.denoise_model.parameters())
        payload = {
            "metadata": {
                "architecture": "satellite_lidar_pixel_v22_ray_posterior",
            },
            "step": 2,
            "denoise_model": model.DDPM.denoise_model.state_dict(),
            "condition_model_sat": model.condition_model_sat.state_dict(),
            "lidar_context_model": model.lidar_context_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grad_scaler": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v22.pt"
            torch.save(payload, path)
            step = load_training_checkpoint(model, optimizer, path, expected_architecture="satellite_lidar_pixel_v22_ray_posterior")
            self.assertEqual(step, 2)
            self.assertFalse(hasattr(model, "_checkpoint_initialization_origin"))


if __name__ == "__main__":
    unittest.main()
