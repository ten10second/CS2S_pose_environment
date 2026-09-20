from __future__ import annotations

import unittest

import torch

from ldm.modules.persistent_history import repeat_history, validate_history
from ldm.modules.static_history import DenseStaticHistoryAdapter, StaticHistoryAdapter

try:
    from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel
except ModuleNotFoundError:
    UNetModel = None


def make_dense_history(batch=2, latent_hw=(8, 8), image_hw=(64, 64)):
    h, w = image_hw
    dense_valid = torch.zeros(batch, 1, h, w, dtype=torch.bool)
    dense_valid[:, :, : h // 2] = True
    dense_measured = torch.zeros_like(dense_valid)
    dense_estimated = torch.zeros_like(dense_valid)
    dense_measured[:, :, : h // 4] = True
    dense_estimated[:, :, h // 4 : h // 2] = True
    rgb = torch.rand(batch, 3, h, w) * dense_valid.float()
    return {
        "latent": torch.randn(batch, 4, *latent_hw),
        "dense_rgb": rgb,
        "dense_valid": dense_valid,
        "dense_measured": dense_measured,
        "dense_estimated": dense_estimated,
    }


class DenseStaticHistoryAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.feature = torch.randn(2, 32, 8, 8)
        self.history = make_dense_history()
        self.adapter = DenseStaticHistoryAdapter(32, hidden_dim=16)

    def activate(self, module=None):
        module = module or self.adapter
        torch.nn.init.normal_(module.output.weight, std=0.05)

    def test_schema_accepts_dense_and_rejects_mixed_partial_bad_masks(self):
        validate_history(self.history)
        invalid = []
        partial = dict(self.history)
        partial.pop("dense_estimated")
        invalid.append(partial)
        invalid.append(dict(self.history, static_rgb=torch.zeros(2, 3, 16, 16)))
        invalid.append(dict(self.history, dense_valid=self.history["dense_valid"].float()))
        overlap = dict(self.history)
        overlap["dense_estimated"] = overlap["dense_estimated"].clone()
        overlap["dense_estimated"][:, :, 0, 0] = True
        invalid.append(overlap)
        not_union = dict(self.history)
        not_union["dense_valid"] = torch.zeros_like(self.history["dense_valid"])
        invalid.append(not_union)
        bad_rgb = dict(self.history)
        bad_rgb["dense_rgb"] = bad_rgb["dense_rgb"].clone()
        bad_rgb["dense_rgb"][:, :, -1, -1] = 0.5
        invalid.append(bad_rgb)
        for item in invalid:
            with self.subTest(fields=tuple(item)), self.assertRaises(ValueError):
                validate_history(item)

    def test_zero_off_and_empty_valid_are_identity(self):
        self.assertTrue(torch.equal(self.adapter(self.feature, history=self.history), self.feature))
        self.activate()
        self.assertTrue(torch.equal(self.adapter(self.feature), self.feature))
        off = dict(self.history, enabled=torch.tensor([True, False]))
        out = self.adapter(self.feature, history=off)
        self.assertFalse(torch.equal(out[0], self.feature[0]))
        self.assertTrue(torch.equal(out[1], self.feature[1]))
        empty = make_dense_history()
        empty["dense_rgb"].zero_()
        empty["dense_valid"].zero_()
        empty["dense_measured"].zero_()
        empty["dense_estimated"].zero_()
        self.adapter.zero_grad(set_to_none=True)
        result = self.adapter(self.feature, history=empty)
        self.assertTrue(torch.equal(result, self.feature))
        result.sum().backward()
        self.assertTrue(all(p.grad is not None for p in self.adapter.parameters()))

    def test_no_per_pixel_output_mask_after_activation(self):
        self.activate()
        out = self.adapter(self.feature, history=self.history)
        self.assertFalse(torch.equal(out, self.feature))
        self.assertGreater((out - self.feature).abs().sum().item(), 0)
        self.assertGreater((out[:, :, 4:] - self.feature[:, :, 4:]).abs().sum().item(), 0)

    def test_reference_encoder_gets_gradient_after_output_projection_update(self):
        optimizer = torch.optim.SGD(self.adapter.parameters(), lr=0.2)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = self.adapter(self.feature, history=self.history)
            (result - (self.feature + 0.25)).square().mean().backward()
            self.assertGreater(self.adapter.output.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(self.adapter.encoder[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()

    def test_variants_share_architecture_and_change_mask_channels_only(self):
        torch.manual_seed(7)
        typed = DenseStaticHistoryAdapter(32, hidden_dim=16, input_variant="types")
        valid = DenseStaticHistoryAdapter(32, hidden_dim=16, input_variant="valid")
        rgb = DenseStaticHistoryAdapter(32, hidden_dim=16, input_variant="rgb")
        valid.load_state_dict(typed.state_dict(), strict=True)
        rgb.load_state_dict(typed.state_dict(), strict=True)
        self.assertEqual({k: v.shape for k, v in typed.state_dict().items()},
                         {k: v.shape for k, v in valid.state_dict().items()})
        torch.nn.init.normal_(typed.output.weight, std=0.05)
        valid.load_state_dict(typed.state_dict(), strict=True)
        rgb.load_state_dict(typed.state_dict(), strict=True)
        out_types = typed(self.feature, history=self.history)
        out_valid = valid(self.feature, history=self.history)
        out_rgb = rgb(self.feature, history=self.history)
        self.assertGreater((out_types - out_valid).abs().max().item(), 1e-6)
        self.assertGreater((out_valid - out_rgb).abs().max().item(), 1e-6)

    def test_cfg_repeats_whole_batches(self):
        self.activate()
        repeated = repeat_history(self.history)
        for name, value in self.history.items():
            self.assertTrue(torch.equal(repeated[name][:2], value))
            self.assertTrue(torch.equal(repeated[name][2:], value))
        out = self.adapter(torch.cat([self.feature, self.feature]), history=repeated)
        expected = self.adapter(self.feature, history=self.history)
        torch.testing.assert_close(out, torch.cat([expected, expected]), rtol=1e-5, atol=1e-6)


class DenseStaticUNetIntegrationTests(unittest.TestCase):
    @unittest.skipIf(UNetModel is None, "UNet dependencies are not present in this staging subset")
    def test_unet_static_dense_identity_and_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
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
        model.configure_temporal_history(mode="static_dense", hidden_dim=16)
        self.assertIsInstance(model.temporal_history, DenseStaticHistoryAdapter)
        self.assertEqual(model.temporal_history.input_variant, "types")
        self.assertEqual(set(k for k in model.state_dict() if not k.startswith("temporal_history.")), base_keys)
        history = make_dense_history(2, latent_hw=(8, 8), image_hw=(64, 64))
        self.assertTrue(torch.equal(model(x, timesteps, history=history).detach(), baseline))
        optimizer = torch.optim.SGD(model.temporal_history.parameters(), lr=0.5)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(x, timesteps, history=history).square().mean().backward()
            self.assertGreater(model.temporal_history.output.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(model.temporal_history.encoder[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not n.startswith("temporal_history.")))
        self.assertTrue(torch.equal(model(x, timesteps).detach(), baseline))

    def test_configure_rejects_nondefault_variant_for_other_modes_and_preserves_static_v1(self):
        if UNetModel is None:
            self.skipTest("UNet dependencies are not present in this staging subset")
        model = UNetModel(image_size=8, in_channels=4, model_channels=32,
                          out_channels=4, num_res_blocks=1, attention_resolutions=(),
                          channel_mult=(1, 2), num_heads=1)
        with self.assertRaises(ValueError):
            model.configure_temporal_history(mode="static", hidden_dim=16, input_variant="rgb")
        model.configure_temporal_history(mode="static", hidden_dim=16)
        self.assertIsInstance(model.temporal_history, StaticHistoryAdapter)


if __name__ == "__main__":
    unittest.main()
