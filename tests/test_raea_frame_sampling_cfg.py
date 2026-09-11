"""Check CFG wiring of the shared frame sampler without a checkpoint or CUDA."""
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import torch

from tools.raea_frame_sampling import sample_frame


class FrameSamplingCFGTest(unittest.TestCase):
    def setUp(self):
        self.latent = torch.zeros(1, 4, 2, 2)
        self.sampler = SimpleNamespace(ddim_sampling=Mock(return_value=(self.latent, {})))
        self.model = SimpleNamespace(
            pre_AE_model=SimpleNamespace(decode=lambda value: value), scale_factor=1.0,
        )
        self.pack = {name: torch.ones(1, 2, 3) for name in (
            "cond_label", "left_camera_k", "gt_shift_x", "gt_shift_y", "theta",
            "range_img", "range_mask", "camera_to_lidar", "lidar_context",
            "lidar_evidence", "lidar_geometry_mask",
        )}

    def sample(self, scale):
        sample_frame(self.model, self.sampler, self.pack, self.latent, 16, 7.5, 1.0,
                     uncond_cfg=scale)
        return self.sampler.ddim_sampling.call_args.kwargs

    def test_default_preserves_disabled_cfg(self):
        kwargs = self.sample(0.0)
        self.assertIsNone(kwargs["unconditional_conditioning"])
        self.assertEqual(kwargs["unconditional_guidance_scale"], 7.5)

    def test_cfg_zeros_only_satellite_and_overrides_scale(self):
        kwargs = self.sample(3.0)
        self.assertEqual(kwargs["unconditional_guidance_scale"], 3.0)
        self.assertTrue(torch.equal(kwargs["unconditional_conditioning"],
                                    torch.zeros_like(self.pack["cond_label"])))
        self.assertIs(self.sampler.ddim_sampling.call_args.args[0], self.pack["cond_label"])
        self.assertTrue(torch.all(self.pack["cond_label"] == 1))
        for name in ("lidar_context", "lidar_evidence", "lidar_geometry_mask", "left_camera_k"):
            self.assertIs(kwargs[name], self.pack[name])

    def test_scale_one_is_passed_without_amplification(self):
        kwargs = self.sample(1.0)
        self.assertEqual(kwargs["unconditional_guidance_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
