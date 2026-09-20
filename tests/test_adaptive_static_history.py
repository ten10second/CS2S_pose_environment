from __future__ import annotations

import unittest

import torch

from ldm.modules.static_history import (
    AdaptiveStaticHistoryAdapter,
    DenseStaticHistoryAdapter,
)
from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel


def make_dense_history(batch=2, latent_hw=(8, 8), image_hw=(64, 64), variant="types"):
    h, w = image_hw
    dense_valid = torch.zeros(batch, 1, h, w, dtype=torch.bool)
    dense_valid[:, :, : h // 2] = True
    dense_measured = torch.zeros_like(dense_valid)
    dense_estimated = torch.zeros_like(dense_valid)
    dense_measured[:, :, : h // 4] = True
    dense_estimated[:, :, h // 4 : h // 2] = True
    rgb = torch.rand(batch, 3, h, w) * dense_valid.float()
    if variant == "rgb":
        dense_measured.zero_()
        dense_estimated.copy_(dense_valid)
    return {
        "latent": torch.randn(batch, 4, *latent_hw),
        "dense_rgb": rgb,
        "dense_valid": dense_valid,
        "dense_measured": dense_measured,
        "dense_estimated": dense_estimated,
    }


class AdaptiveStaticHistoryAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.feature = torch.randn(2, 32, 8, 8)
        self.temb = torch.randn(2, 128)
        self.history = make_dense_history()
        self.adapter = AdaptiveStaticHistoryAdapter(32, 128, hidden_dim=64)

    def test_zero_off_empty_and_mixed_batch_identity(self):
        self.assertTrue(torch.equal(self.adapter(self.feature, history=self.history, temb=self.temb), self.feature))
        self.assertTrue(torch.equal(self.adapter(self.feature, temb=self.temb), self.feature))
        torch.nn.init.normal_(self.adapter.fusion.output.weight, std=0.05)
        torch.nn.init.normal_(self.adapter.fusion.output.bias, std=0.05)
        mixed = dict(self.history, enabled=torch.tensor([True, False]))
        out = self.adapter(self.feature, history=mixed, temb=self.temb)
        self.assertFalse(torch.equal(out[0], self.feature[0]))
        self.assertTrue(torch.equal(out[1], self.feature[1]))
        empty = make_dense_history()
        empty["dense_rgb"].zero_()
        empty["dense_valid"].zero_()
        empty["dense_measured"].zero_()
        empty["dense_estimated"].zero_()
        self.assertTrue(torch.equal(self.adapter(self.feature, history=empty, temb=self.temb), self.feature))

    def test_b2_mixed_present_rgb_only_uses_raw_valid_for_sample_gate(self):
        adapter = AdaptiveStaticHistoryAdapter(32, 128, hidden_dim=64, input_variant="rgb")
        torch.nn.init.normal_(adapter.fusion.output.weight, std=0.05)
        torch.nn.init.normal_(adapter.fusion.output.bias, std=0.05)
        history = make_dense_history(variant="rgb")
        history["dense_valid"][1].zero_()
        history["dense_rgb"][1].zero_()
        history["dense_measured"][1].zero_()
        history["dense_estimated"][1].zero_()
        out = adapter(self.feature, history=history, temb=self.temb)
        self.assertFalse(torch.equal(out[0], self.feature[0]))
        self.assertTrue(torch.equal(out[1], self.feature[1]))

    def test_current_and_time_sensitivity_after_update(self):
        optimizer = torch.optim.SGD(self.adapter.parameters(), lr=0.2)
        target = self.feature + 0.2
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            self.adapter(self.feature, history=self.history, temb=self.temb).sub(target).square().mean().backward()
            optimizer.step()
        base = self.adapter(self.feature, history=self.history, temb=self.temb)
        changed_feature = self.adapter(self.feature + 0.1, history=self.history, temb=self.temb)
        changed_time = self.adapter(self.feature, history=self.history, temb=self.temb + 0.1)
        self.assertGreater((base - changed_feature).abs().max().item(), 1e-6)
        self.assertGreater((base - changed_time).abs().max().item(), 1e-6)

    def test_cache_eval_parity_and_rejects_training_or_grad(self):
        torch.nn.init.normal_(self.adapter.fusion.output.weight, std=0.05)
        self.adapter.eval()
        with torch.no_grad():
            encoded = self.adapter.encode_history(self.history)
            cache = dict(self.history, dense_features=encoded)
            direct = self.adapter(self.feature, history=self.history, temb=self.temb)
            cached = self.adapter(self.feature, history=cache, temb=self.temb)
        torch.testing.assert_close(direct, cached, rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            self.adapter(self.feature, history=cache, temb=self.temb)
        self.adapter.train()
        with torch.no_grad(), self.assertRaises(ValueError):
            self.adapter(self.feature, history=cache, temb=self.temb)
        bad_cache = dict(self.history, dense_features=encoded[:, :, :4])
        self.adapter.eval()
        with torch.no_grad(), self.assertRaises(ValueError):
            self.adapter(self.feature, history=bad_cache, temb=self.temb)

    def test_explicit_loader_attrs(self):
        self.assertEqual(self.adapter.fusion_dim, 128)
        self.assertEqual(self.adapter.time_embed_dim, 128)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for half conv/autocast cache parity")
    def test_cuda_half_autocast_cache_parity(self):
        adapter = AdaptiveStaticHistoryAdapter(32, 128, hidden_dim=64).cuda().half().eval()
        torch.nn.init.normal_(adapter.fusion.output.weight, std=0.05)
        feature = self.feature.cuda().half()
        temb = self.temb.cuda().half()
        history = {key: value.cuda() if torch.is_tensor(value) else value
                   for key, value in self.history.items()}
        with torch.no_grad(), torch.cuda.amp.autocast():
            encoded = adapter.encode_history(history)
            self.assertEqual(encoded.dtype, torch.float16)
            cache = dict(history, dense_features=encoded)
            direct = adapter(feature, history=history, temb=temb)
            cached = adapter(feature, history=cache, temb=temb)
        self.assertEqual(direct.dtype, torch.float16)
        torch.testing.assert_close(direct.float(), cached.float(), rtol=1e-3, atol=1e-3)

    def test_static_dense_a0_parity_uses_public_encode_history(self):
        adapter = DenseStaticHistoryAdapter(32, hidden_dim=64)
        torch.nn.init.normal_(adapter.output.weight, std=0.05)
        out = adapter(self.feature, history=self.history)
        encoded = adapter.encode_history(self.history)
        if encoded.shape[-2:] != self.feature.shape[-2:]:
            encoded = torch.nn.functional.interpolate(
                encoded, size=self.feature.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.history["dense_valid"].flatten(1).any(dim=1).to(self.feature.dtype)[:, None, None, None]
        expected = self.feature + adapter.output(encoded.to(self.feature.dtype)) * gate
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for A0 half conv/autocast cache parity")
    def test_static_dense_a0_cuda_half_autocast_cache_parity(self):
        adapter = DenseStaticHistoryAdapter(32, hidden_dim=64).cuda().half().eval()
        torch.nn.init.normal_(adapter.output.weight, std=0.05)
        feature = self.feature.cuda().half()
        history = {key: value.cuda() if torch.is_tensor(value) else value
                   for key, value in self.history.items()}
        with torch.no_grad(), torch.cuda.amp.autocast():
            encoded = adapter.encode_history(history)
            self.assertEqual(encoded.dtype, torch.float16)
            cache = dict(history, dense_features=encoded)
            direct = adapter(feature, history=history)
            cached = adapter(feature, history=cache)
        self.assertEqual(direct.dtype, torch.float16)
        torch.testing.assert_close(direct.float(), cached.float(), rtol=1e-3, atol=1e-3)


class AdaptiveStaticUNetIntegrationTests(unittest.TestCase):
    def test_mini_unet_static_adaptive_gradients_after_two_updates(self):
        torch.set_num_threads(1)
        torch.manual_seed(321)
        model = UNetModel(image_size=8, in_channels=4, model_channels=32,
                          out_channels=4, num_res_blocks=1, attention_resolutions=(),
                          channel_mult=(1, 2), num_heads=1)
        torch.nn.init.normal_(model.out[-1].weight, std=0.01)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        x = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([5, 77])
        baseline = model(x, timesteps).detach()
        base_keys = set(model.state_dict())
        model.configure_temporal_history(mode="static_adaptive", hidden_dim=64, input_variant="types")
        self.assertIsInstance(model.temporal_history, AdaptiveStaticHistoryAdapter)
        self.assertEqual(set(k for k in model.state_dict() if not k.startswith("temporal_history.")), base_keys)
        history = make_dense_history(2, latent_hw=(8, 8), image_hw=(64, 64))
        self.assertTrue(torch.equal(model(x, timesteps, history=history).detach(), baseline))
        optimizer = torch.optim.SGD(model.temporal_history.parameters(), lr=0.3)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(x, timesteps, history=history).square().mean().backward()
            self.assertGreater(model.temporal_history.fusion.output.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(model.temporal_history.encoder[0].weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.temporal_history.fusion.current_proj.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.temporal_history.fusion.emb_proj.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                            if not n.startswith("temporal_history.")))
        self.assertTrue(torch.equal(model(x, timesteps).detach(), baseline))


if __name__ == "__main__":
    unittest.main()
