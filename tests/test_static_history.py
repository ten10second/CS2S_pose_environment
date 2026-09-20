import unittest

import torch

from ldm.modules.persistent_history import PersistentHistoryReader, repeat_history, validate_history
from ldm.modules.static_history import StaticHistoryAdapter, pool_static_reference
from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel


def make_static_history(batch=2, height=4, width=8):
    return {
        "latent": torch.randn(batch, 4, height, width),
        "static_rgb": torch.rand(batch, 3, height * 2, width * 2),
        "static_mask": torch.ones(batch, 1, height * 2, width * 2, dtype=torch.bool),
        "static_confidence": torch.full((batch, 1, height * 2, width * 2), .8),
    }


class StaticHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        self.adapter = StaticHistoryAdapter(32, hidden_dim=16)
        self.feature = torch.randn(2, 32, 4, 8)
        self.history = make_static_history()

    def activate(self):
        torch.nn.init.normal_(self.adapter.output.weight, std=.1)

    def test_zero_initialization_and_missing_history_are_exact_identity(self):
        self.assertTrue(torch.equal(self.adapter(self.feature, history=self.history), self.feature))
        self.activate()
        self.assertTrue(torch.equal(self.adapter(self.feature), self.feature))

    def test_sparse_pooling_preserves_observed_color(self):
        rgb = torch.zeros(1, 3, 4, 4)
        rgb[:, 0, 0, 0] = 1.0
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        mask[:, :, 0, 0] = True
        confidence = torch.full_like(mask, .6, dtype=torch.float32)
        colors, support, certainty = pool_static_reference(rgb, mask, confidence, (2, 2))
        torch.testing.assert_close(colors[0, :, 0, 0], torch.tensor([1., 0., 0.]))
        self.assertEqual(support[0, 0, 0, 0].item(), .25)
        self.assertAlmostEqual(certainty[0, 0, 0, 0].item(), .6)
        self.assertTrue(torch.equal(colors[:, :, 1, 1], torch.zeros(1, 3)))
        self.assertEqual(certainty[0, 0, 1, 1].item(), 0.)
        # Colors at unobserved pixels cannot contaminate a supported bin.
        rgb[:, :, 0, 1] = .9
        second = pool_static_reference(rgb, mask, confidence, (2, 2))
        for old, new in zip((colors, support, certainty), second):
            self.assertTrue(torch.equal(old, new))

    def test_activated_adapter_is_exactly_off_outside_support_or_enabled(self):
        self.activate()
        self.history["static_mask"][:, :, :, 8:] = False
        self.history["enabled"] = torch.tensor([True, False])
        result = self.adapter(self.feature, history=self.history)
        self.assertTrue(torch.equal(result[1], self.feature[1]))
        self.assertTrue(torch.equal(result[0, :, :, 4:], self.feature[0, :, :, 4:]))
        self.assertFalse(torch.equal(result[0, :, :, :4], self.feature[0, :, :, :4]))
        self.history["static_mask"].zero_()
        self.adapter.zero_grad(set_to_none=True)
        result = self.adapter(self.feature, history=self.history)
        self.assertTrue(torch.equal(result, self.feature))
        result.sum().backward()
        self.assertTrue(all(p.grad is not None for p in self.adapter.parameters()))
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in self.adapter.parameters()))

    def test_zero_confidence_and_no_history_keep_parameter_graph(self):
        self.activate()
        self.history["static_confidence"].zero_()
        self.assertTrue(torch.equal(self.adapter(self.feature, history=self.history), self.feature))
        self.adapter(self.feature).sum().backward()
        self.assertTrue(all(p.grad is not None for p in self.adapter.parameters()))

    def test_reference_encoder_learns_after_output_projection_first_update(self):
        self.history["static_rgb"].requires_grad_(True)
        self.history["latent"].requires_grad_(True)
        optimizer = torch.optim.SGD(self.adapter.parameters(), lr=.1)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            result = self.adapter(self.feature, history=self.history)
            (result - (self.feature + .5)).square().mean().backward()
            self.assertGreater(self.adapter.output.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(self.adapter.encoder[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertIsNone(self.history["static_rgb"].grad)
        self.assertIsNone(self.history["latent"].grad)

    def test_batch_and_cfg_order_do_not_mix_samples(self):
        self.activate()
        self.adapter.eval()
        self.history["enabled"] = torch.tensor([True, False])
        together = self.adapter(self.feature, history=self.history)
        apart = torch.cat([
            self.adapter(self.feature[i:i + 1], history={k: v[i:i + 1] for k, v in self.history.items()})
            for i in range(2)])
        torch.testing.assert_close(together, apart, rtol=1e-5, atol=1e-6)
        repeated = repeat_history(self.history)
        for name, value in self.history.items():
            self.assertTrue(torch.equal(repeated[name], torch.cat([value, value])))
        cfg = self.adapter(torch.cat([self.feature, self.feature]), history=repeated)
        torch.testing.assert_close(cfg, torch.cat([together, together]), rtol=1e-5, atol=1e-6)

    def test_schema_rejects_partial_mixed_nonfinite_range_batch_and_device(self):
        invalid = []
        partial = dict(self.history)
        partial.pop("static_confidence")
        invalid.append(partial)
        invalid.append(dict(self.history, history_grid=torch.zeros(2, 4, 8, 1, 2)))
        invalid.append(dict(self.history, static_rgb=self.history["static_rgb"][:1]))
        invalid.append(dict(self.history, static_mask=self.history["static_mask"].float()))
        invalid.append(dict(self.history, static_confidence=torch.full_like(self.history["static_confidence"], 1.1)))
        invalid.append(dict(self.history, static_rgb=torch.full_like(self.history["static_rgb"], float("nan"))))
        invalid.append(dict(self.history, latent=torch.full_like(self.history["latent"], float("inf"))))
        invalid.append(dict(self.history, enabled=torch.empty(2, dtype=torch.bool, device="meta")))
        for history in invalid:
            with self.subTest(fields=tuple(history)), self.assertRaises(ValueError):
                validate_history(history)
        with self.assertRaises(ValueError):
            self.adapter(self.feature, history={"latent": self.history["latent"]})
        with self.assertRaises(ValueError):
            self.adapter(self.feature[:1], history=self.history)
        with self.assertRaises(ValueError):
            self.adapter(self.feature.to("meta"), history=self.history)

    def test_legacy_content_schema_and_reader_state_are_unchanged(self):
        validate_history({"latent": self.history["latent"]})
        reader = PersistentHistoryReader(32, 12, hidden_dim=16, mode="content")
        state = reader.state_dict()
        copy = PersistentHistoryReader(32, 12, hidden_dim=16, mode="content")
        copy.load_state_dict(state, strict=True)
        torch.testing.assert_close(reader(self.feature, None, {"latent": self.history["latent"]}),
                                   copy(self.feature, None, {"latent": self.history["latent"]}), rtol=0, atol=0)


class StaticUNetIntegrationTests(unittest.TestCase):
    def test_frozen_unet_identity_and_gradients_through_decoder(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        model = UNetModel(image_size=8, in_channels=4, model_channels=32,
                          out_channels=4, num_res_blocks=1, attention_resolutions=(),
                          channel_mult=(1, 2), num_heads=1)
        torch.nn.init.normal_(model.out[-1].weight, std=.01)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        x = torch.randn(2, 4, 8, 8)
        times = torch.tensor([12, 88])
        baseline = model(x, times).detach()
        base_keys = set(model.state_dict())
        model.configure_temporal_history(mode="static", hidden_dim=16)
        history = make_static_history(2, 8, 8)
        self.assertIsInstance(model.temporal_history, StaticHistoryAdapter)
        self.assertEqual(set(k for k in model.state_dict() if not k.startswith("temporal_history.")), base_keys)
        self.assertTrue(torch.equal(model(x, times, history=history).detach(), baseline))
        optimizer = torch.optim.SGD(model.temporal_history.parameters(), lr=.5)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(x, times, history=history).square().mean().backward()
            self.assertGreater(model.temporal_history.output.weight.grad.abs().sum().item(), 0)
            if step == 1:
                self.assertGreater(model.temporal_history.encoder[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if not n.startswith("temporal_history.")))
        self.assertTrue(torch.equal(model(x, times).detach(), baseline))
        model.configure_temporal_history(mode="static", hidden_dim=16)
        with self.assertRaises(ValueError):
            model.configure_temporal_history(mode="geometry", hidden_dim=16)


if __name__ == "__main__":
    unittest.main()
