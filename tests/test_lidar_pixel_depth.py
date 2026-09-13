import unittest

import torch

from models.KITTI_geo_ldm.lidar_pixel_depth import LidarPixelDepthHead
from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel


class LidarPixelDepthHeadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407)
        torch.set_num_threads(1)

    def test_head_predicts_full_resolution_from_decoder_features(self):
        head = LidarPixelDepthHead(320)
        x = torch.randn(2, 320, 16, 64, requires_grad=True)

        y = head(x)

        self.assertEqual(tuple(y.shape), (2, 1, 128, 512))
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_zero_features_are_finite_and_trainable(self):
        head = LidarPixelDepthHead(320)
        x = torch.zeros(1, 320, 16, 64, requires_grad=True)

        y = head(x)
        loss = y.mean()
        loss.backward()

        self.assertTrue(torch.isfinite(y).all())
        self.assertTrue(all(p.requires_grad for p in head.parameters()))
        self.assertGreater(sum(p.numel() for p in head.parameters()), 0)
        self.assertTrue(any(p.grad is not None for p in head.parameters()))

    def test_sparse_hit_loss_reaches_decoder_features_without_supervising_holes(self):
        head = LidarPixelDepthHead(320)
        x = torch.randn(1, 320, 16, 64, requires_grad=True)
        pred = head(x).sigmoid()
        pred.retain_grad()
        target = torch.zeros_like(pred)
        mask = torch.zeros_like(pred)
        target[..., 40, 100:102] = torch.tensor([0.125, 0.375])
        mask[..., 40, 100:102] = 1.0
        loss = ((pred.log() - target.clamp_min(1e-3).log()).abs() * mask).sum() / mask.sum()
        loss.backward()
        self.assertEqual(torch.count_nonzero(pred.grad * (1 - mask)).item(), 0)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    @staticmethod
    def _unet(**kwargs):
        params = dict(
            image_size=16,
            in_channels=4,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(),
            channel_mult=(1, 2, 4, 4),
            num_heads=4,
        )
        params.update(kwargs)
        return UNetModel(**params)

    def test_unet_default_keeps_latent_depth_head_behavior(self):
        model = self._unet()

        self.assertEqual(model.lidar_depth_head_mode, "latent")
        self.assertTrue(hasattr(model, "lidar_depth_head"))
        self.assertTrue(hasattr(model, "lidar_bottleneck_depth_head"))
        self.assertFalse(hasattr(model, "lidar_pixel_depth_head"))

        out = model(torch.randn(1, 4, 16, 64), torch.tensor([1]))
        self.assertEqual(tuple(out.shape), (1, 4, 16, 64))
        self.assertEqual(tuple(model.last_lidar_depth_pred.shape), (1, 1, 16, 64))

    def test_unet_pixel_mode_predicts_full_resolution_depth(self):
        model = self._unet(lidar_depth_head_mode="pixel")

        self.assertFalse(hasattr(model, "lidar_depth_head"))
        self.assertFalse(hasattr(model, "lidar_bottleneck_depth_head"))
        self.assertTrue(hasattr(model, "lidar_pixel_depth_head"))

        out = model(torch.randn(1, 4, 16, 64), torch.tensor([1]))
        self.assertEqual(tuple(out.shape), (1, 4, 16, 64))
        self.assertEqual(tuple(model.last_lidar_depth_pred.shape), (1, 1, 128, 512))
        self.assertIsNone(model.last_lidar_bottleneck_depth_pred)

    def test_unet_rejects_unknown_depth_head_mode(self):
        with self.assertRaisesRegex(ValueError, "lidar_depth_head_mode"):
            self._unet(lidar_depth_head_mode="nearest")


if __name__ == "__main__":
    unittest.main(verbosity=2)
