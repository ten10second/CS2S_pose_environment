import unittest

import torch
from torch import nn

from models.KITTI_geo_ldm.lidar_pixel_condition import (
    LidarPixelConditionEncoder, LidarSpatialResidual,
)
from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel
from ldm.modules.KITTI_attention import BasicTransformerBlock, CrossAttention


class PixelConditionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3407)
        torch.set_num_threads(1)

    def encoder(self):
        return LidarPixelConditionEncoder(
            point_feature_dim=4, hidden_channels=8, pyramid_channels=(8, 16, 32, 32),
            image_size=(64, 64), token_grid=(4, 4), semantic_feature_dim=6,
        )

    def inputs(self):
        raw = torch.zeros(1, 10, 64, 64)
        raw[:, 1, 2, 2] = raw[:, 1, 2, 3] = 1
        raw[:, 2, 2, 2] = 10.0 / 80
        raw[:, 2, 2, 3] = 30.0 / 80
        return raw, torch.randn(1, 4, 64, 64), raw[:, 1:2].clone()

    def test_joint_encoding_preserves_separate_pixels_before_reduction(self):
        encoder = self.encoder()
        raw, features, hit = self.inputs()
        initial = encoder.encode_pixels(features, raw[:, 2:3], hit)
        changed = features.clone()
        changed[:, 0, 2, 2] += 10
        output = encoder.encode_pixels(changed, raw[:, 2:3], hit)
        self.assertFalse(torch.allclose(initial[:, :, 2, 2], output[:, :, 2, 2]))
        torch.testing.assert_close(initial[:, :, 2, 3], output[:, :, 2, 3], rtol=0, atol=0)
        depth = raw[:, 2:3].clone()
        depth[:, :, 2, 2] = 0.9
        deeper = encoder.encode_pixels(features, depth, hit)
        self.assertFalse(torch.allclose(initial[:, :, 2, 2], deeper[:, :, 2, 2]))

    def test_pyramid_shapes_gradients_and_auxiliary_semantic_head(self):
        encoder = self.encoder()
        raw, features, hit = self.inputs()
        features.requires_grad_(True)
        result = encoder(raw, lidar_pixel_features=features, lidar_pixel_features_mask=hit)
        for f, mask, c, side in zip(result['features'], result['masks'], (8, 16, 32, 32), (8, 4, 2, 1)):
            self.assertEqual(tuple(f.shape), (1, c, side, side))
            self.assertEqual(tuple(mask.shape), (1, 1, side, side))
            self.assertTrue(torch.isfinite(f).all())
        self.assertEqual(tuple(encoder.last_semantic_pred_tokens.shape), (1, 16, 6))
        sum(f.square().mean() for f in result['features']).backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad[:, :, 2, 2].abs().sum().item(), 0)
        self.assertEqual(features.grad[:, :, 0, 0].abs().sum().item(), 0)

    def test_unknown_pixels_and_zero_lidar_cannot_inject_content(self):
        encoder = self.encoder()
        raw, features, hit = self.inputs()
        changed = features.clone()
        changed[:, :, hit[0, 0] == 0] = 10000
        a = encoder(raw, lidar_pixel_features=features, lidar_pixel_features_mask=hit)
        b = encoder(raw, lidar_pixel_features=changed, lidar_pixel_features_mask=hit)
        for left, right in zip(a['features'], b['features']):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        empty = encoder(torch.zeros_like(raw), lidar_pixel_features=features, lidar_pixel_features_mask=hit)
        for value in (*empty['features'], *empty['masks']):
            self.assertEqual(value.count_nonzero().item(), 0)

    def test_legacy_patch_cache_is_rejected(self):
        encoder = self.encoder()
        raw, features, hit = self.inputs()
        with self.assertRaisesRegex(ValueError, 'V2.1 requires'):
            encoder(raw, lidar_ray_features=torch.zeros(1, 4, 1, 8, 32))
        with self.assertRaisesRegex(ValueError, 'expected'):
            encoder(raw, lidar_pixel_features=features[:, :, ::2, ::2], lidar_pixel_features_mask=hit)

    def test_spatial_residual_initializes_to_identity_and_obeys_support(self):
        residual = LidarSpatialResidual(8, 32)
        x = torch.randn(1, 32, 8, 8)
        features = torch.randn(1, 8, 8, 8, requires_grad=True)
        support = torch.ones(1, 1, 8, 8)
        torch.testing.assert_close(residual(x, features, support), x, rtol=0, atol=0)
        nn.init.normal_(residual.projection.weight, std=0.1)
        torch.testing.assert_close(residual(x, features, support * 0), x, rtol=0, atol=0)
        result = residual(x, features, support)
        self.assertFalse(torch.allclose(result, x))
        result.square().mean().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_unet_consumes_every_spatial_scale(self):
        model = UNetModel(
            image_size=8, in_channels=4, model_channels=32, out_channels=4,
            num_res_blocks=1, attention_resolutions=(), channel_mult=(1, 2, 4, 4),
            num_heads=4, lidar_spatial_channels=(8, 16, 32, 32),
        )
        encoder = self.encoder()
        raw, features, hit = self.inputs()
        context = encoder(raw, lidar_pixel_features=features, lidar_pixel_features_mask=hit)
        called = []
        hooks = [module.register_forward_hook(lambda m, a, y: called.append(tuple(y.shape)))
                 for module in model.lidar_spatial_residuals.values()]
        result = model(torch.randn(1, 4, 8, 8), torch.tensor([1]), lidar_context=context)
        for hook in hooks:
            hook.remove()
        self.assertEqual(tuple(result.shape), (1, 4, 8, 8))
        self.assertEqual([shape[-1] for shape in called], [8, 4, 2, 1])
        loss = model.last_lidar_bottleneck_depth_pred.mean()
        loss.backward()
        for module in model.lidar_spatial_residuals.values():
            self.assertIsNotNone(module.projection.weight.grad)
            self.assertTrue(torch.isfinite(module.projection.weight.grad).all())

    def test_satellite_posterior_keeps_evidence_without_token_attention(self):
        class SatelliteSpy(nn.Module):
            def forward(self, x, **kwargs):
                self.evidence = kwargs.get('ray_depth_evidence')
                return torch.zeros_like(x)
        block = BasicTransformerBlock(
            dim=16, n_heads=2, d_head=8, context_dim=16,
            use_lidar_cross_attention=False, use_lidar_ray_posterior=True,
        )
        spy = SatelliteSpy()
        block.attn2 = spy
        evidence = torch.zeros(1, 8, 64, 64)
        block(torch.randn(1, 4, 16), lidar_evidence=evidence)
        self.assertIs(spy.evidence, evidence)
        attention = CrossAttention(query_dim=16, context_dim=16, heads=2, dim_head=8,
                                   use_lidar_ray_posterior=True)
        candidates = torch.tensor([[[10., 30.]]])
        observed = torch.tensor([[[[10. / 80]], [[1.]]]])
        weights = attention._apply_lidar_ray_posterior(
            torch.zeros(1, 2, 1, 2), candidates, torch.ones_like(candidates, dtype=torch.bool), observed, (1, 1),
        )
        self.assertTrue((weights[..., 0] > weights[..., 1]).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
