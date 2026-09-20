import unittest

import torch
from torch.nn import functional as F

from ldm.modules.persistent_history import PersistentHistoryReader, repeat_history
from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel


def make_history(b=2, h=4, w=8):
    latent = torch.randn(b, 4, h, w)
    grid = F.affine_grid(torch.eye(2, 3)[None].repeat(b, 1, 1), latent.shape, align_corners=False)
    grid = grid.unsqueeze(3).repeat(1, 1, 1, 3, 1)
    grid[..., 1, 0] = (grid[..., 1, 0] + 0.15).clamp(-.9, .9)
    grid[..., 2, 0] = (grid[..., 2, 0] - 0.15).clamp(-.9, .9)
    return {"latent": latent, "history_grid": grid, "sat_grid": grid.clone(),
            "valid": torch.ones(b, h, w, 3, dtype=torch.bool),
            "sat_valid": torch.ones(b, h, w, 3, dtype=torch.bool),
            "positions": torch.randn(b, h, w, 3, 4)}


class HistoryReaderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.reader = PersistentHistoryReader(32, 12, hidden_dim=16)
        self.x = torch.randn(2, 32, 4, 8)
        self.context = torch.randn(2, 16, 12)
        self.history = make_history()

    def activate(self):
        torch.nn.init.normal_(self.reader.output.weight, std=.05)

    def test_zero_init_and_off_preserve_base(self):
        self.assertTrue(torch.equal(self.reader(self.x, self.context, self.history), self.x))
        self.activate()
        self.assertTrue(torch.equal(self.reader(self.x, self.context), self.x))

    def test_gradients_reach_encoder_and_sat_matching_but_detach_history(self):
        self.activate()
        self.history["latent"].requires_grad_(True)
        self.reader(self.x, self.context, self.history).square().mean().backward()
        for name in ("encoder.0.weight", "query.weight", "sat_key.weight", "position_key.weight", "output.weight"):
            gradient = dict(self.reader.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
        self.assertIsNone(self.history["latent"].grad)

    def test_each_sample_is_independent(self):
        self.activate()
        self.reader.eval()
        together = self.reader(self.x, self.context, self.history)
        apart = torch.cat([self.reader(self.x[i:i+1], self.context[i:i+1],
                                      {k: v[i:i+1] for k, v in self.history.items()}) for i in range(2)])
        torch.testing.assert_close(together, apart, rtol=1e-5, atol=1e-6)

    def test_per_sample_dropout_is_exactly_off(self):
        self.activate()
        self.history["enabled"] = torch.tensor([True, False])
        out = self.reader(self.x, self.context, self.history)
        self.assertTrue(torch.equal(out[1], self.x[1]))
        self.assertFalse(torch.equal(out[0], self.x[0]))

    def test_empty_history_has_ddp_graph_for_every_parameter(self):
        self.reader(self.x, self.context, None).sum().backward()
        self.assertTrue(all(p.grad is not None for p in self.reader.parameters()))
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in self.reader.parameters()))

    def test_missing_geometry_uses_finite_content_fallback(self):
        self.activate()
        self.history["valid"].zero_()
        self.history["sat_valid"].zero_()
        out = self.reader(self.x, self.context, self.history)
        self.assertTrue(torch.isfinite(out).all())
        self.assertFalse(torch.equal(out, self.x))

    def test_sat_and_history_both_affect_candidate_readout(self):
        self.activate()
        expected = self.reader(self.x, self.context, self.history)
        changed_sat = self.reader(self.x, self.context.flip(1), self.history)
        changed_history = self.reader(self.x, self.context, dict(self.history, latent=self.history["latent"].flip(0)))
        self.assertGreater((changed_sat - expected).abs().max().item(), 1e-5)
        self.assertGreater((changed_history - expected).abs().max().item(), 1e-5)

    def test_cfg_repeats_whole_batches_not_interleaved_samples(self):
        repeated = repeat_history(self.history)
        for key, tensor in self.history.items():
            self.assertTrue(torch.equal(repeated[key][:2], tensor))
            self.assertTrue(torch.equal(repeated[key][2:], tensor))

    def test_invalid_batch_or_legacy_B_fields_raise(self):
        with self.assertRaises(ValueError):
            self.reader(self.x, self.context, dict(self.history, latent=self.history["latent"][:1]))
        with self.assertRaises(ValueError):
            self.reader(self.x, self.context, dict(self.history, timestep=500))

    def test_content_control_has_gradients_without_geometry(self):
        reader = PersistentHistoryReader(32, 12, hidden_dim=16, mode="content")
        torch.nn.init.normal_(reader.output.weight, std=.05)
        output = reader(self.x, self.context, {"latent": self.history["latent"]})
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(all(p.grad is not None for p in reader.parameters()))


class UNetIntegrationTests(unittest.TestCase):
    def test_real_unet_base_equivalence_and_history_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        model = UNetModel(image_size=8, in_channels=4, model_channels=32,
                          out_channels=4, num_res_blocks=1, attention_resolutions=(),
                          channel_mult=(1, 2), num_heads=1)
        torch.nn.init.normal_(model.out[-1].weight, std=.01)
        x = torch.randn(2, 4, 8, 8)
        times = torch.tensor([12, 88])
        base_keys = set(model.state_dict())
        reference = model(x, times).detach()
        model.configure_temporal_history(hidden_dim=16)
        self.assertEqual(set(k for k in model.state_dict() if not k.startswith("temporal_history.")), base_keys)
        history = make_history(2, 8, 8)
        self.assertTrue(torch.equal(model(x, times, history=history).detach(), reference))
        self.assertTrue(torch.equal(model(x, times).detach(), reference))
        model(x, times, history=history).square().mean().backward()
        self.assertGreater(model.temporal_history.output.weight.grad.abs().sum().item(), 0)
        torch.nn.init.normal_(model.temporal_history.output.weight, std=.03)
        model.zero_grad(set_to_none=True)
        model(x, times, history=history).square().mean().backward()
        self.assertGreater(model.temporal_history.encoder[0].weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
