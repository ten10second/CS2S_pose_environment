import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from tools.train_kitti_raea import configure_cfg, initialize_v22_from_v21, load_training_checkpoint, parse_args


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

    def test_weights_initialization_keeps_fresh_head_and_rejects_unrelated_missing_keys(self):
        model = self.fake_model()
        fresh = copy.deepcopy(model.DDPM.denoise_model.lidar_pixel_depth_head.state_dict())
        payload = {
            "metadata": {"architecture": "satellite_lidar_pixel_v21_ray_posterior"},
            "step": 20000,
            "denoise_model": {
                key: torch.full_like(value, 0.75)
                for key, value in model.DDPM.denoise_model.state_dict().items()
                if not key.startswith("lidar_pixel_depth_head.")
            },
            "condition_model_sat": model.condition_model_sat.state_dict(),
            "lidar_context_model": model.lidar_context_model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v21.pt"
            torch.save(payload, path)
            info = initialize_v22_from_v21(model, path)
            self.assertEqual(info["source_step"], 20000)
            self.assertTrue(info["optimizer_reset"])
            self.assertTrue(torch.equal(model.DDPM.denoise_model.body.weight, torch.full((2, 2), 0.75)))
            for key, value in fresh.items():
                self.assertTrue(torch.equal(value, model.DDPM.denoise_model.lidar_pixel_depth_head.state_dict()[key]))
            del payload["denoise_model"]["body.weight"]
            torch.save(payload, path)
            with self.assertRaisesRegex(RuntimeError, "key mismatch"):
                initialize_v22_from_v21(model, path)

    def test_strict_v22_resume_preserves_v21_initialization_provenance(self):
        model = self.fake_model()
        optimizer = torch.optim.Adam(model.DDPM.denoise_model.parameters())
        origin = {"source_step": 20000, "optimizer_reset": True}
        payload = {
            "metadata": {
                "architecture": "satellite_lidar_pixel_v22_ray_posterior",
                "init_ckpt": "/v21/step_020000.pt",
                "initialization": origin,
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
            self.assertEqual(model._checkpoint_initialization_origin["initialization"], origin)
            self.assertEqual(model._checkpoint_initialization_origin["init_ckpt"], "/v21/step_020000.pt")


if __name__ == "__main__":
    unittest.main()
