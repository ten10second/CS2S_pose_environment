import unittest

import torch

from ldm.modules.KITTI_attention import RayAlignedEvidenceAttention


class RayEvidenceAttentionTest(unittest.TestCase):
    def test_router_learns_from_bias_only_initialization(self):
        torch.manual_seed(7)
        module = RayAlignedEvidenceAttention(
            dim=8,
            heads=2,
            dim_head=4,
            sat_bias=2.0,
            lidar_bias=-2.0,
            null_bias=-6.0,
        )
        x = torch.randn(2, 6, 8)
        sat_ref = torch.randn_like(x)
        lidar_ref = torch.randn_like(x)

        self.assertGreater(module.to_q.weight.norm().item(), 0.0)
        self.assertEqual(module.to_k.weight.norm().item(), 0.0)
        self.assertEqual(module.ray_proj.weight.norm().item(), 0.0)

        output = module(x, sat_ref, lidar_ref, query_hw=(2, 3))
        bias_weights = module.evidence_bias.softmax(dim=0)
        expected = (
            bias_weights[0] * sat_ref
            + bias_weights[1] * lidar_ref
            + bias_weights[2] * module.null_token.expand_as(x)
        )
        torch.testing.assert_close(output, expected)

        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        output.square().mean().backward()
        self.assertIsNotNone(module.to_k.weight.grad)
        self.assertGreater(module.to_k.weight.grad.abs().sum().item(), 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        self.assertGreater(module.to_k.weight.norm().item(), 0.0)
        output = module(x, sat_ref, lidar_ref, query_hw=(2, 3))
        self.assertGreater(module.last_evidence_lidar_weight_std.item(), 0.0)
        output.square().mean().backward()
        self.assertIsNotNone(module.to_q.weight.grad)
        self.assertGreater(module.to_q.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(module.ray_proj.weight.grad)
        self.assertGreater(module.ray_proj.weight.grad.abs().sum().item(), 0.0)

    def test_router_reset_preserves_bias_only_start_and_revives_query(self):
        torch.manual_seed(11)
        module = RayAlignedEvidenceAttention(dim=8, heads=2, dim_head=4)
        with torch.no_grad():
            module.to_q.weight.zero_()
            module.to_k.weight.normal_()
            module.ray_proj.weight.normal_()

        module.reset_router_parameters()

        self.assertGreater(module.to_q.weight.norm().item(), 0.0)
        self.assertEqual(module.to_k.weight.norm().item(), 0.0)
        self.assertEqual(module.ray_proj.weight.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
