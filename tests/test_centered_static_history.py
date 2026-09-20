from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from ldm.modules.static_history import (
    AdaptiveStaticHistoryAdapter,
    CenteredStaticHistoryAdapter,
    DenseStaticHistoryAdapter,
)
from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel
from tools.infer_temporal import load_static_adapter
from test_persistent_sampling import Sampler


def make_dense_history(batch=2, latent_hw=(8, 8), image_hw=(64, 64)):
    h, w = image_hw
    valid = torch.zeros(batch, 1, h, w, dtype=torch.bool)
    valid[:, :, : h // 2] = True
    measured = torch.zeros_like(valid)
    estimated = torch.zeros_like(valid)
    measured[:, :, : h // 4] = True
    estimated[:, :, h // 4 : h // 2] = True
    return {
        "latent": torch.randn(batch, 4, *latent_hw),
        "dense_rgb": torch.rand(batch, 3, h, w) * valid.float(),
        "dense_valid": valid,
        "dense_measured": measured,
        "dense_estimated": estimated,
    }


def randomize(module):
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.05)


class CenteredStaticHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(818)
        self.feature = torch.randn(2, 32, 8, 8)
        self.temb = torch.randn(2, 128)
        self.history = make_dense_history()
        self.adapter = CenteredStaticHistoryAdapter(32, 128, hidden_dim=64)

    def test_zero_init_off_and_zero_history_exact_fallback(self):
        self.assertTrue(torch.equal(self.adapter(self.feature, history=self.history, temb=self.temb), self.feature))
        self.assertTrue(torch.equal(self.adapter(self.feature, temb=self.temb), self.feature))
        randomize(self.adapter.fusion)
        self.adapter.eval()
        with torch.no_grad():
            encoded = self.adapter.encode_history(self.history)
            zero_history = dict(self.history, dense_features=torch.zeros_like(encoded))
            out = self.adapter(self.feature, history=zero_history, temb=self.temb)
        self.assertTrue(torch.equal(out, self.feature))

    def test_both_fusion_calls_keep_gradients(self):
        randomize(self.adapter.fusion.output)
        seen = []

        def hook(_module, _args, output):
            output.retain_grad()
            seen.append(output)

        handle = self.adapter.fusion.register_forward_hook(hook)
        try:
            out = self.adapter(self.feature, history=self.history, temb=self.temb)
            (out - (self.feature + 0.1)).square().mean().backward()
        finally:
            handle.remove()
        self.assertEqual(len(seen), 2)
        self.assertGreater(seen[0].grad.abs().sum().item(), 0)
        self.assertGreater(seen[1].grad.abs().sum().item(), 0)

    def test_multiple_updates_reach_history_encoder(self):
        optimizer = torch.optim.SGD(self.adapter.parameters(), lr=0.2)
        target = self.feature + 0.25
        encoder_grad = 0.0
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            self.adapter(self.feature, history=self.history, temb=self.temb).sub(target).square().mean().backward()
            if self.adapter.encoder[0].weight.grad is not None:
                encoder_grad = max(encoder_grad, self.adapter.encoder[0].weight.grad.abs().sum().item())
            optimizer.step()
        self.assertGreater(encoder_grad, 0)

    def test_v4_loader_and_old_modes_remain_strict(self):
        reader = CenteredStaticHistoryAdapter(8, 16, hidden_dim=4, input_variant="types")
        payload = {
            "version": "temporal_static_adapter_v4",
            "model_mode": "static_centered",
            "state_dict": reader.state_dict(),
            "hidden_dim": reader.hidden_dim,
            "input_variant": reader.input_variant,
            "reference_kind": "dense",
            "fusion_dim": reader.fusion_dim,
            "time_embed_dim": reader.time_embed_dim,
            "base_checkpoint": {"sha256": "base"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "centered.pt"
            torch.save(payload, path)
            copy = CenteredStaticHistoryAdapter(8, 16, hidden_dim=4, input_variant="types")
            load_static_adapter(path, copy, {"sha256": "base"})
            for key, value in reader.state_dict().items():
                self.assertTrue(torch.equal(value, copy.state_dict()[key]))
            with self.assertRaisesRegex(ValueError, "static_centered"):
                load_static_adapter(path, AdaptiveStaticHistoryAdapter(8, 16, hidden_dim=4), {"sha256": "base"})
            with self.assertRaisesRegex(ValueError, "static_centered"):
                load_static_adapter(path, DenseStaticHistoryAdapter(8, hidden_dim=4), {"sha256": "base"})

    def test_sampler_caches_centered_history_per_call_and_cfg_order(self):
        sampler = Sampler()
        adapter = CenteredStaticHistoryAdapter(8, 16, hidden_dim=4).eval()
        sampler.model.denoise_model.temporal_history = adapter
        history = make_dense_history(batch=2, latent_hw=(4, 8), image_hw=(32, 64))
        history["enabled"] = torch.tensor([True, False])
        noise = torch.randn(2, 4, 4, 8)
        seen = []
        hook = adapter.encoder.register_forward_hook(lambda _m, _a, out: seen.append(out.detach().clone()))
        try:
            kwargs = dict(S=8, batch_size=2, shape=[4, 4, 8], x_T=noise,
                          conditioning=torch.ones(2, 16, 12), unconditional_guidance_scale=7.5)
            sampler.sample(history=history, **kwargs)
            self.assertEqual(len(seen), 1)
            self.assertNotIn("dense_features", history)
            for _, _, read in sampler.model.denoise_model.calls:
                self.assertTrue(torch.equal(read["dense_features"], torch.cat([seen[0]] * 2)))
                self.assertEqual(read["enabled"].tolist(), [True, False, True, False])
        finally:
            hook.remove()


class CenteredStaticUNetTests(unittest.TestCase):
    def test_mini_unet_static_centered_path(self):
        torch.set_num_threads(1)
        torch.manual_seed(919)
        model = UNetModel(image_size=8, in_channels=4, model_channels=32,
                          out_channels=4, num_res_blocks=1, attention_resolutions=(),
                          channel_mult=(1, 2), num_heads=1)
        torch.nn.init.normal_(model.out[-1].weight, std=0.01)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        x = torch.randn(2, 4, 8, 8)
        timesteps = torch.tensor([17, 71])
        baseline = model(x, timesteps).detach()
        base_keys = set(model.state_dict())
        model.configure_temporal_history(mode="static_centered", hidden_dim=64, input_variant="types")
        self.assertIsInstance(model.temporal_history, CenteredStaticHistoryAdapter)
        self.assertEqual(set(k for k in model.state_dict() if not k.startswith("temporal_history.")), base_keys)
        history = make_dense_history(2, latent_hw=(8, 8), image_hw=(64, 64))
        self.assertTrue(torch.equal(model(x, timesteps, history=history).detach(), baseline))
        optimizer = torch.optim.SGD(model.temporal_history.parameters(), lr=0.3)
        encoder_grad = 0.0
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(x, timesteps, history=history).square().mean().backward()
            if model.temporal_history.encoder[0].weight.grad is not None:
                encoder_grad = max(encoder_grad, model.temporal_history.encoder[0].weight.grad.abs().sum().item())
            optimizer.step()
        self.assertGreater(encoder_grad, 0)
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                            if not n.startswith("temporal_history.")))
        self.assertTrue(torch.equal(model(x, timesteps).detach(), baseline))


if __name__ == "__main__":
    unittest.main()
