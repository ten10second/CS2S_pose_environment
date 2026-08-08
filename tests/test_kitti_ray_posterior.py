from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ldm.modules.KITTI_attention import (
    BasicTransformerBlock,
    CrossAttention,
    RayPosteriorEvidenceFusion,
)
from tools.build_kitti_utonia_ray_cache import pool_ray_depth_features


def _posterior_attention():
    return CrossAttention(
        query_dim=16,
        context_dim=16,
        heads=2,
        dim_head=8,
        use_lidar_ray_posterior=True,
        lidar_posterior_log_depth_sigma=0.2,
        lidar_posterior_strength=2.0,
    )


class RayPosteriorTest(unittest.TestCase):
    def test_posterior_mode_does_not_instantiate_legacy_raea_router(self):
        block = BasicTransformerBlock(
            dim=16,
            n_heads=2,
            d_head=8,
            context_dim=16,
            use_lidar_cross_attention=True,
            lidar_context_dim=16,
            ray_fusion_mode="ray_posterior",
            use_lidar_ray_posterior=True,
        )

        self.assertIsNone(block.ray_evidence_attn)
        self.assertIsNotNone(block.ray_posterior_fusion)

    def test_zero_lidar_is_exact_satellite_fallback(self):
        module = _posterior_attention()
        logits = torch.randn(1, 2, 3, 8)
        candidate_depth = torch.tensor(
            [[[5.0, 10.0, 20.0, 40.0, 5.0, 10.0, 20.0, 40.0]]] * 3
        )
        candidate_valid = torch.ones_like(candidate_depth, dtype=torch.bool)
        zero_evidence = torch.zeros(1, 2, 1, 3)

        actual = module._apply_lidar_ray_posterior(
            logits,
            candidate_depth,
            candidate_valid,
            zero_evidence,
            query_hw=(1, 3),
        )

        torch.testing.assert_close(actual, logits.softmax(dim=-1), rtol=0.0, atol=0.0)
        self.assertEqual(module.last_ray_posterior_hit_coverage.item(), 0.0)

    def test_lidar_depth_concentrates_satellite_candidates(self):
        module = _posterior_attention()
        logits = torch.zeros(1, 2, 1, 8)
        candidate_depth = torch.tensor([[[5.0, 10.0, 20.0, 40.0, 6.0, 12.0, 24.0, 48.0]]])
        candidate_valid = torch.ones_like(candidate_depth, dtype=torch.bool)
        evidence = torch.tensor([[[[20.0 / 80.0]], [[1.0]]]])

        posterior = module._apply_lidar_ray_posterior(
            logits,
            candidate_depth,
            candidate_valid,
            evidence,
            query_hw=(1, 1),
        )

        self.assertEqual(posterior[0, 0, 0].argmax().item(), 2)
        self.assertLess(module.last_ray_posterior_entropy, module.last_ray_posterior_prior_entropy)
        self.assertGreater(module.last_ray_posterior_weight_shift, 0.0)

    def test_posterior_fusion_only_adds_lidar_on_supported_rays(self):
        module = RayPosteriorEvidenceFusion(dim=8, lidar_gate_bias=-2.0)
        x = torch.randn(1, 4, 8)
        sat = torch.randn_like(x)
        lidar = torch.randn_like(x)

        no_hit = torch.zeros(1, 1, 1, 4)
        torch.testing.assert_close(
            module(x, sat, lidar, query_hw=(1, 4), lidar_geometry_mask=no_hit),
            sat,
            rtol=0.0,
            atol=0.0,
        )

        one_hit = no_hit.clone()
        one_hit[..., 2] = 1.0
        fused = module(x, sat, lidar, query_hw=(1, 4), lidar_geometry_mask=one_hit)
        torch.testing.assert_close(fused[:, :2], sat[:, :2], rtol=0.0, atol=0.0)
        torch.testing.assert_close(fused[:, 3:], sat[:, 3:], rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(fused[:, 2], sat[:, 2]))

    def test_ray_depth_pool_uses_every_input_point(self):
        point_count = 5003
        feature_dim = 4
        features = np.arange(point_count * feature_dim, dtype=np.float32).reshape(point_count, feature_dim)
        uv = np.stack(
            [
                np.linspace(0.0, 511.0, point_count, dtype=np.float32),
                np.linspace(0.0, 127.0, point_count, dtype=np.float32),
            ],
            axis=1,
        )
        depth = np.linspace(1.0, 79.0, point_count, dtype=np.float32)
        args = SimpleNamespace(
            ray_depth_bins=4,
            ray_height=8,
            ray_width=32,
            image_height=128,
            image_width=512,
            max_depth=80.0,
        )

        pooled, mask, counts = pool_ray_depth_features(features, uv, depth, args)

        self.assertEqual(pooled.shape, (feature_dim, 4, 8, 32))
        self.assertEqual(mask.shape, (1, 4, 8, 32))
        self.assertEqual(int(counts.sum()), point_count)


if __name__ == "__main__":
    unittest.main()
