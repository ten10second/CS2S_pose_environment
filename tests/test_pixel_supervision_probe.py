import unittest
from types import SimpleNamespace

import torch

from tools.probe_kitti_pixel_supervision import (
    active_depth_head_from_cfg,
    aggregate,
    depth_loss_stats,
    gradient_summary,
)


class PixelSupervisionProbeTest(unittest.TestCase):
    def test_invalid_zero_targets_count_before_clamp(self):
        pred = torch.full((1, 1, 1, 1), 0.25, requires_grad=True)
        target = torch.zeros(1, 1, 4, 4)
        mask = torch.zeros_like(target)
        mask[..., 1, 1] = 1.0

        stats = depth_loss_stats(pred, target, mask, mode="masked_area", eps=1e-3, weight=0.1)

        self.assertEqual(stats["invalid_zero_target_count_before_clamp"], 1)
        self.assertEqual(stats["support_count"], 1)
        self.assertTrue(torch.isfinite(stats["loss"]))
        stats["loss"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_gradient_summary_uses_total_minus_depth_other_component(self):
        g_depth = torch.tensor([1.0, 0.0, 0.0])
        g_total = torch.tensor([1.0, 2.0, 0.0])

        summary = gradient_summary(g_depth, g_total)

        self.assertAlmostEqual(summary["bottleneck_h_grad_depth_norm"], 1.0)
        self.assertAlmostEqual(summary["bottleneck_h_grad_other_norm"], 2.0)
        self.assertAlmostEqual(summary["bottleneck_h_grad_depth_to_other_ratio"], 0.5)
        self.assertAlmostEqual(
            summary["bottleneck_h_grad_depth_total_cosine"],
            1.0 / (5.0 ** 0.5),
            places=6,
        )
        self.assertAlmostEqual(summary["bottleneck_h_grad_depth_other_cosine"], 0.0)
        self.assertTrue(summary["bottleneck_h_grad_connected"])
        self.assertTrue(summary["bottleneck_h_grad_finite"])

    def test_gradient_summary_marks_disconnected_without_json_infinity(self):
        summary = gradient_summary(None, torch.tensor([1.0]))

        self.assertFalse(summary["bottleneck_h_grad_connected"])
        self.assertFalse(summary["bottleneck_h_grad_finite"])
        self.assertEqual(summary["bottleneck_h_grad_depth_to_other_ratio"], 0.0)

    def test_gradient_summary_can_label_output_scope(self):
        summary = gradient_summary(torch.tensor([3.0]), torch.tensor([5.0]), prefix="output_h_grad")

        self.assertIn("output_h_grad_depth_norm", summary)
        self.assertNotIn("bottleneck_h_grad_depth_norm", summary)
        self.assertAlmostEqual(summary["output_h_grad_depth_norm"], 3.0)
        self.assertTrue(summary["output_h_grad_connected"])

    def test_native_depth_loss_uses_valid_hits_without_resampling(self):
        pred = torch.full((1, 1, 128, 512), 0.5, requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.zeros_like(pred)
        target[..., 10, 20] = 0.25
        target[..., 11, 21] = 0.75
        mask[..., 10, 20] = 1.0
        mask[..., 11, 21] = 1.0

        stats = depth_loss_stats(pred, target, mask, mode="native", eps=1e-3, weight=0.1)

        self.assertEqual(stats["target_shape"], [1, 1, 128, 512])
        self.assertEqual(stats["mask_shape"], [1, 1, 128, 512])
        self.assertEqual(stats["support_count"], 2)
        self.assertEqual(stats["invalid_zero_target_count_before_clamp"], 0)
        expected = (
            abs(torch.log(torch.tensor(0.5)) - torch.log(torch.tensor(0.25)))
            + abs(torch.log(torch.tensor(0.5)) - torch.log(torch.tensor(0.75)))
        ) / 2.0
        self.assertAlmostEqual(float(stats["log_l1"]), float(expected), places=6)
        stats["loss"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_native_depth_loss_rejects_shape_mismatch(self):
        pred = torch.full((1, 1, 64, 256), 0.5)
        target = torch.full((1, 1, 128, 512), 0.25)
        mask = torch.ones_like(target)

        with self.assertRaises(ValueError):
            depth_loss_stats(pred, target, mask, mode="native", eps=1e-3, weight=0.1)

    def test_active_depth_head_selects_unique_enabled_head(self):
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                params=SimpleNamespace(lidar_depth_output_scale=1.0, lidar_depth_bottleneck_scale=0.0)
            )
        )

        spec = active_depth_head_from_cfg(cfg)

        self.assertEqual(spec["name"], "output")
        self.assertEqual(spec["attr"], "last_lidar_depth_pred")
        self.assertEqual(spec["gradient_scope"], "output_decoder_h")

    def test_active_depth_head_rejects_ambiguous_two_head_config(self):
        cfg = SimpleNamespace(
            model=SimpleNamespace(
                params=SimpleNamespace(lidar_depth_output_scale=1.0, lidar_depth_bottleneck_scale=1.0)
            )
        )

        with self.assertRaises(ValueError):
            active_depth_head_from_cfg(cfg)

    def test_aggregate_fails_on_invalid_targets_or_missing_gradients(self):
        base = {
            "finite_checks": {
                "total_loss_finite": True,
                "head_depth_finite": True,
                "weighted_depth_loss_finite": True,
            },
            "gradient_scope": "bottleneck_feature_h",
            "head_scope": "bottleneck",
            "bottleneck_h_grad_finite": True,
            "bottleneck_h_grad_connected": True,
            "invalid_zero_target_count_before_clamp": 0,
            "reported_bottleneck_depth_log_l1_agreement": {"matches": True},
            "reported_bottleneck_depth_contrib_agreement": {"matches": True},
        }
        self.assertTrue(aggregate([base])["probe_passed"])

        invalid = dict(base, invalid_zero_target_count_before_clamp=1)
        self.assertFalse(aggregate([invalid])["probe_passed"])

        disconnected = dict(base, bottleneck_h_grad_connected=False)
        self.assertFalse(aggregate([disconnected])["probe_passed"])

        mismatch = dict(base, reported_bottleneck_depth_contrib_agreement={"matches": False})
        self.assertFalse(aggregate([mismatch])["probe_passed"])


if __name__ == "__main__":
    unittest.main()
