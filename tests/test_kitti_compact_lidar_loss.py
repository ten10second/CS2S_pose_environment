import math
import types
import unittest

import torch
from torch import nn

from models.KITTI_geo_ldm_diffusion.latent_diffusion import DDPM


class _DepthPredDenoiser(nn.Module):
    def __init__(self, output_depth, bottleneck_depth):
        super().__init__()
        self.last_lidar_depth_pred = output_depth
        self.last_lidar_bottleneck_depth_pred = bottleneck_depth

    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)


class CompactLidarLossTest(unittest.TestCase):
    @staticmethod
    def _ddpm(output_depth, bottleneck_depth):
        model = DDPM.__new__(DDPM)
        nn.Module.__init__(model)
        model.control_grd = None
        model.denoise_model = _DepthPredDenoiser(output_depth, bottleneck_depth)
        model.last_loss_metrics = {}
        model.q_sample = types.MethodType(
            lambda self, x_start, t, noise=None: x_start,
            model,
        )
        return model

    def test_depth_head_scales_select_only_requested_supervision(self):
        x_start = torch.zeros(1, 4, 2, 2)
        timestep = torch.zeros(1, dtype=torch.long)
        noise = torch.zeros_like(x_start)
        target = torch.full((1, 1, 2, 2), 0.5)
        mask = torch.ones_like(target)
        output_depth = torch.full_like(target, 0.25)
        bottleneck_depth = target.clone()

        model = self._ddpm(output_depth, bottleneck_depth)
        loss = model.p_losses(
            x_start,
            timestep,
            noise=noise,
            lidar_depth_target=target,
            lidar_depth_mask=mask,
            lidar_depth_loss_weight=1.0,
            lidar_depth_output_scale=0.0,
            lidar_depth_bottleneck_scale=1.0,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertNotIn("loss_lidar_depth_log_l1", model.last_loss_metrics)
        self.assertIn("loss_lidar_bottleneck_depth_log_l1", model.last_loss_metrics)

        model = self._ddpm(output_depth, bottleneck_depth)
        loss = model.p_losses(
            x_start,
            timestep,
            noise=noise,
            lidar_depth_target=target,
            lidar_depth_mask=mask,
            lidar_depth_loss_weight=1.0,
            lidar_depth_output_scale=1.0,
            lidar_depth_bottleneck_scale=0.0,
        )
        self.assertAlmostEqual(loss.item(), math.log(2.0), places=5)
        self.assertIn("loss_lidar_depth_log_l1", model.last_loss_metrics)
        self.assertNotIn("loss_lidar_bottleneck_depth_log_l1", model.last_loss_metrics)


if __name__ == "__main__":
    unittest.main()
