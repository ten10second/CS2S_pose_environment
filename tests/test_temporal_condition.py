import unittest
from pathlib import Path

import torch

from ldm.modules.temporal_condition import repeat_history, validate_history
from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler
from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel


FIXTURE = Path("/mnt/shizhm/CS2S_run_control/temporal_a1_retirement_20260920/single_frame_before.pt")


def make_unet():
    return UNetModel(image_size=8, in_channels=4, model_channels=32,
                     out_channels=4, num_res_blocks=1, attention_resolutions=(),
                     channel_mult=(1, 2), num_heads=1)


def make_history(batch=2, height=8, width=8):
    latent = torch.randn(batch, 4, height, width)
    measured = torch.rand(batch, 1, height, width) * 0.6
    estimated = torch.rand(batch, 1, height, width) * (1.0 - measured)
    valid = measured + estimated
    masks = torch.cat([valid, measured, estimated], dim=1)
    return {"latent": latent, "masks": masks}


class TemporalConditionContractTests(unittest.TestCase):
    def test_validate_and_repeat_history_whole_batches(self):
        history = make_history(batch=2)
        validate_history(history, batch_size=2)
        repeated = repeat_history(history, repeats=2)
        for key, value in history.items():
            self.assertTrue(torch.equal(repeated[key][:2], value))
            self.assertTrue(torch.equal(repeated[key][2:], value))

    def test_invalid_contracts_raise(self):
        history = make_history(batch=2)
        bad = dict(history, dense_features=torch.zeros(2, 1, 8, 8))
        with self.assertRaises(ValueError):
            validate_history(bad, batch_size=2)
        with self.assertRaises(ValueError):
            validate_history({"latent": history["latent"]}, batch_size=2)
        bad_range = {"latent": history["latent"], "masks": history["masks"].clone()}
        bad_range["masks"][:, 0].add_(1.1)
        with self.assertRaises(ValueError):
            validate_history(bad_range, batch_size=2)
        bad_sum = {"latent": history["latent"], "masks": history["masks"].clone()}
        bad_sum["masks"][:, 2].zero_()
        with self.assertRaises(ValueError):
            validate_history(bad_sum, batch_size=2)
        bad_spatial = {"latent": history["latent"], "masks": history["masks"][:, :, :4, :4]}
        with self.assertRaises(ValueError):
            validate_history(bad_spatial, batch_size=2)


class UNetHistoryInputTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def test_fixture_strict_load_and_default_forward_unchanged(self):
        fixture = torch.load(FIXTURE, map_location="cpu")
        model = make_unet()
        model.load_state_dict(fixture["state"], strict=True)
        out = model(fixture["x"], fixture["t"])
        torch.testing.assert_close(out, fixture["out"], rtol=0, atol=0)

    def test_configure_history_input_is_zero_init_idempotent_and_off_equivalent(self):
        model = make_unet()
        x = torch.randn(2, 4, 8, 8)
        t = torch.tensor([3, 7])
        reference = model(x, t).detach()
        old_weight = model.input_blocks[0][0].weight.detach().clone()
        model.configure_history_input()
        model.configure_history_input()
        conv = model.input_blocks[0][0]
        self.assertEqual(conv.in_channels, 11)
        torch.testing.assert_close(conv.weight[:, :4], old_weight, rtol=0, atol=0)
        self.assertEqual(float(conv.weight[:, 4:].abs().sum()), 0.0)
        torch.testing.assert_close(model(x, t).detach(), reference, rtol=0, atol=0)
        torch.testing.assert_close(model(x, t, history=make_history()).detach(), reference, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            model.configure_temporal_history()

    def test_unconfigured_history_is_rejected(self):
        model = make_unet()
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4, 8, 8), torch.tensor([1, 2]), history=make_history())

    def test_all_zero_masks_neutralize_latent_after_history_weights_change(self):
        model = make_unet()
        model.configure_history_input()
        with torch.no_grad():
            model.input_blocks[0][0].weight[:, 4:].normal_(std=0.05)
        x = torch.randn(2, 4, 8, 8)
        t = torch.tensor([1, 2])
        history = {"latent": torch.randn(2, 4, 8, 8), "masks": torch.zeros(2, 3, 8, 8)}
        torch.testing.assert_close(model(x, t, history=history), model(x, t), rtol=0, atol=0)

    def test_fractional_valid_mask_does_not_attenuate_latent_channels(self):
        model = make_unet()
        model.configure_history_input()
        x = torch.randn(1, 4, 8, 8)
        latent = torch.full((1, 4, 8, 8), 2.0)
        masks = torch.zeros(1, 3, 8, 8)
        masks[:, 0] = 0.25
        masks[:, 1] = 0.10
        masks[:, 2] = 0.15
        augmented = model._history_augmented_input(x, {"latent": latent, "masks": masks})
        torch.testing.assert_close(augmented[:, 4:8], latent, rtol=0, atol=0)
        torch.testing.assert_close(augmented[:, 8:], masks, rtol=0, atol=0)

    def test_history_gradients_two_passes(self):
        model = make_unet()
        torch.nn.init.normal_(model.out[-1].weight, std=0.05)
        model.configure_history_input()
        x = torch.randn(2, 4, 8, 8)
        t = torch.tensor([4, 5])
        history = make_history()
        model(x, t, history=history).square().mean().backward()
        conv = model.input_blocks[0][0]
        self.assertGreater(float(conv.weight.grad[:, 4:].abs().sum()), 0.0)
        self.assertEqual(float(conv.weight.grad[:, :4].abs().sum()) > 0.0, True)

        model.zero_grad(set_to_none=True)
        history = make_history()
        history["latent"].requires_grad_(True)
        with torch.no_grad():
            conv.weight[:, 4:8].normal_(std=0.05)
        model(x, t, history=history).square().mean().backward()
        self.assertIsNotNone(history["latent"].grad)
        self.assertGreater(float(history["latent"].grad.abs().sum()), 0.0)


class DDIMHistoryInputTests(unittest.TestCase):
    def test_cfg_repeats_history_as_whole_batches(self):
        class FakeDenoiseModel:
            def __init__(self):
                self.calls = []

            def __call__(self, x, t, context=None, history=None, **kwargs):
                self.calls.append({"x": x.detach().clone(), "t": t.detach().clone(),
                                   "context": context.detach().clone(), "history": history})
                values = torch.where(context.flatten(1).abs().sum(dim=1) > 0,
                                     torch.full((x.shape[0],), 3.0), torch.full((x.shape[0],), 1.0))
                return values.to(x.device).view(-1, 1, 1, 1).expand_as(x)

        class FakeModel:
            def __init__(self):
                self.denoise_model = FakeDenoiseModel()

        sampler = KITTI_DDIMSampler.__new__(KITTI_DDIMSampler)
        sampler.model = FakeModel()
        sampler.ddim_alphas = torch.tensor([1.0])
        sampler.ddim_alphas_prev = torch.tensor([1.0])
        sampler.ddim_sqrt_one_minus_alphas = torch.tensor([0.0])
        sampler.ddim_sigmas = torch.tensor([0.0])

        x = torch.zeros(2, 4, 16, 64)
        t = torch.tensor([9, 10])
        conditioning = torch.ones(2, 2, 3)
        unconditional = torch.zeros_like(conditioning)
        history = make_history(batch=2, height=16, width=64)
        e_t, _, _ = sampler.p_sample_ddim(
            x, conditioning, t, index=0, unconditional_guidance_scale=2.5,
            unconditional_conditioning=unconditional, history=history,
        )
        call_history = sampler.model.denoise_model.calls[0]["history"]
        for key, value in history.items():
            self.assertTrue(torch.equal(call_history[key][:2], value))
            self.assertTrue(torch.equal(call_history[key][2:], value))
        self.assertTrue(torch.allclose(e_t, torch.full_like(e_t, 6.0)))


if __name__ == "__main__":
    unittest.main()
