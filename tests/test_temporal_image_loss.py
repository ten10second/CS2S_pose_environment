import unittest

import torch

from ldm.modules.temporal_image_loss import temporal_image_loss


class TensorVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decode_calls = 0

    def decode(self, z):
        self.decode_calls += 1
        return z[:, :3]


class FrozenScaleVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.25), requires_grad=False)

    def decode(self, z):
        return z[:, :3] * self.scale


class RaisingVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decode_calls = 0

    def decode(self, z):
        self.decode_calls += 1
        raise AssertionError("zero image weights must bypass decode")


class RecordingVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decode_dtype = None

    def decode(self, z):
        self.decode_dtype = z.dtype
        return z[:, :3]


def _base_latents(batch=1, height=2, width=2):
    return torch.zeros(batch, 4, height, width, dtype=torch.float32)


class TemporalImageLossTests(unittest.TestCase):
    def test_exact_epsilon_and_rgb_charbonnier_with_simple_decoder(self):
        prediction = torch.full((1, 4, 2, 2), 0.25)
        noise = torch.zeros_like(prediction)
        noisy_latent = torch.zeros_like(prediction)
        noisy_latent[:, 0] = 0.2
        noisy_latent[:, 1] = -0.4
        noisy_latent[:, 2] = 0.0
        target = torch.full((1, 3, 2, 2), 0.25)
        eps = 1e-3
        vae = TensorVAE()

        total, metrics = temporal_image_loss(
            prediction, noise, noisy_latent, torch.tensor([0]), torch.tensor([1.0]), target,
            vae, 1.0, rgb_weight=2.0, edge_weight=0.0, charbonnier_eps=eps,
        )

        decoded_rgb = (noisy_latent[:, :3] + 1.0) * 0.5
        expected_epsilon = torch.full((), 0.25 ** 2)
        expected_rgb = (torch.sqrt((decoded_rgb - target).pow(2) + eps ** 2) - eps).mean() * 2.0
        self.assertEqual(vae.decode_calls, 1)
        torch.testing.assert_close(total, expected_epsilon + expected_rgb)
        torch.testing.assert_close(metrics["epsilon"], expected_epsilon)
        torch.testing.assert_close(metrics["rgb"], expected_rgb)
        torch.testing.assert_close(metrics["rgb_raw"], expected_rgb / 2.0)
        torch.testing.assert_close(metrics["edge"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["edge_raw"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["time_weight"], torch.tensor(1.0))
        self.assertFalse(metrics["total"].requires_grad)

    def test_zero_image_weights_bypass_decode_and_match_epsilon_only(self):
        prediction = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]])
        noise = torch.zeros_like(prediction)
        vae = RaisingVAE()

        total, metrics = temporal_image_loss(
            prediction, noise, prediction.clone(), torch.tensor([0]), torch.tensor([0.5]),
            torch.zeros(1, 3, 1, 1), vae, 1.0, rgb_weight=0.0, edge_weight=0.0,
        )

        expected = prediction.pow(2).mean()
        self.assertEqual(vae.decode_calls, 0)
        torch.testing.assert_close(total, expected)
        torch.testing.assert_close(metrics["total"], expected)
        torch.testing.assert_close(metrics["rgb"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["edge"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["rgb_raw"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["edge_raw"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["time_weight"], torch.sqrt(torch.tensor(0.5)))

    def test_per_sample_timestep_weight_is_applied_before_batch_mean(self):
        alpha = torch.tensor([0.25, 1.0])
        desired_z0 = torch.zeros(2, 4, 1, 1)
        desired_z0[:, 0:3] = 0.0
        noisy = desired_z0 * torch.sqrt(alpha).view(2, 1, 1, 1)
        target = torch.zeros(2, 3, 1, 1)
        eps = 1e-3

        total, metrics = temporal_image_loss(
            torch.zeros_like(noisy), torch.zeros_like(noisy), noisy, torch.tensor([0, 1]), alpha,
            target, TensorVAE(), 1.0, rgb_weight=1.0, edge_weight=0.0, charbonnier_eps=eps,
        )

        per_sample = torch.sqrt((torch.full((2,), 0.5) - 0.0).pow(2) + eps ** 2) - eps
        expected_rgb = (per_sample * torch.sqrt(alpha)).mean()
        torch.testing.assert_close(total, expected_rgb)
        torch.testing.assert_close(metrics["rgb"], expected_rgb)
        torch.testing.assert_close(metrics["rgb_raw"], per_sample.mean())
        torch.testing.assert_close(metrics["time_weight"], torch.sqrt(alpha).mean())

    def test_rgb_and_edge_terms_backpropagate_to_epsilon_but_not_frozen_vae_params(self):
        prediction = torch.zeros(1, 4, 4, 4, requires_grad=True)
        noise = torch.zeros_like(prediction)
        noisy = torch.zeros_like(prediction)
        target = torch.zeros(1, 3, 4, 4)
        target[:, :, 1:3, 1:3] = 1.0
        vae = FrozenScaleVAE()

        total, metrics = temporal_image_loss(
            prediction, noise, noisy, torch.tensor([0]), torch.tensor([0.5]), target,
            vae, 1.0, rgb_weight=0.7, edge_weight=0.3,
        )
        total.backward()

        self.assertGreater(float(metrics["rgb"]), 0.0)
        self.assertGreater(float(metrics["edge"]), 0.0)
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        self.assertIsNone(vae.scale.grad)

    def test_decode_receives_fp32_when_prediction_is_lower_precision(self):
        prediction = torch.zeros(1, 4, 2, 2, dtype=torch.float16)
        noise = torch.zeros_like(prediction)
        noisy = torch.zeros_like(prediction)
        target = torch.zeros(1, 3, 2, 2)
        vae = RecordingVAE()

        total, _ = temporal_image_loss(
            prediction, noise, noisy, torch.tensor([0]), torch.tensor([0.5]), target,
            vae, 1.0, rgb_weight=1.0, edge_weight=0.0,
        )

        self.assertEqual(vae.decode_dtype, torch.float32)
        self.assertEqual(total.dtype, torch.float32)

    def test_sobel_edge_is_sum_of_x_and_y_axis_means(self):
        prediction = torch.zeros(1, 4, 4, 4)
        noise = torch.zeros_like(prediction)
        noisy = torch.zeros_like(prediction)
        target = torch.zeros(1, 3, 4, 4)
        target[:, :, :, 2:] = 1.0

        total, metrics = temporal_image_loss(
            prediction, noise, noisy, torch.tensor([0]), torch.tensor([1.0]), target,
            TensorVAE(), 1.0, rgb_weight=0.0, edge_weight=1.0,
        )

        predicted_rgb = torch.full_like(target, 0.5)
        kernel_x = target.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        kernel_y = target.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
        kernel = torch.stack((kernel_x, kernel_y), dim=0).view(2, 1, 3, 3).repeat(3, 1, 1, 1)
        pred_edges = torch.nn.functional.conv2d(torch.nn.functional.pad(predicted_rgb, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)
        target_edges = torch.nn.functional.conv2d(torch.nn.functional.pad(target, (1, 1, 1, 1), mode="replicate"), kernel, groups=3)
        edge_axes = (pred_edges - target_edges).abs().view(1, 3, 2, 4, 4).mean(dim=(1, 3, 4))
        expected_edge = edge_axes.sum(dim=1).mean()

        torch.testing.assert_close(total, expected_edge)
        torch.testing.assert_close(metrics["edge"], expected_edge)
        torch.testing.assert_close(metrics["edge_raw"], expected_edge)

    def test_decode_must_return_tensor(self):
        class BadVAE(torch.nn.Module):
            def decode(self, z):
                return (z[:, :3],)

        with self.assertRaises(TypeError):
            temporal_image_loss(
                torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 2, 2),
                torch.tensor([0]), torch.tensor([1.0]), torch.zeros(1, 3, 2, 2),
                BadVAE(), 1.0, rgb_weight=1.0, edge_weight=0.0,
            )

    def test_edge_only_term_produces_epsilon_prediction_gradient(self):
        prediction = torch.zeros(1, 4, 4, 4, requires_grad=True)
        noise = torch.zeros_like(prediction)
        noisy = torch.zeros_like(prediction)
        target = torch.zeros(1, 3, 4, 4)
        target[:, :, :, 2:] = 1.0

        total, metrics = temporal_image_loss(
            prediction, noise, noisy, torch.tensor([0]), torch.tensor([0.5]), target,
            TensorVAE(), 1.0, rgb_weight=0.0, edge_weight=1.0,
        )
        total.backward()

        self.assertGreater(float(metrics["edge"]), 0.0)
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

    def test_rejects_invalid_coefficients_shapes_targets_and_scale(self):
        prediction = _base_latents()
        target = torch.zeros(1, 3, 2, 2)
        valid_args = (prediction, torch.zeros_like(prediction), torch.zeros_like(prediction), torch.tensor([0]), torch.tensor([1.0]), target, TensorVAE(), 1.0)
        for rgb_weight, edge_weight in ((-0.1, 0.0), (0.0, float("nan"))):
            with self.assertRaises(ValueError):
                temporal_image_loss(*valid_args, rgb_weight=rgb_weight, edge_weight=edge_weight)
        with self.assertRaises(ValueError):
            temporal_image_loss(*valid_args[:-1], 0.0, rgb_weight=0.0, edge_weight=0.0)
        with self.assertRaises(ValueError):
            temporal_image_loss(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction), torch.tensor([0]), torch.tensor([0.0]), target, TensorVAE(), 1.0, rgb_weight=0.0, edge_weight=0.0)
        with self.assertRaises(ValueError):
            temporal_image_loss(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction), torch.tensor([2]), torch.tensor([1.0]), target, TensorVAE(), 1.0, rgb_weight=0.0, edge_weight=0.0)
        with self.assertRaises(ValueError):
            temporal_image_loss(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction), torch.tensor([0]), torch.tensor([1.0]), torch.zeros(1, 1, 2, 2), TensorVAE(), 1.0, rgb_weight=1.0, edge_weight=0.0)
        with self.assertRaises(ValueError):
            temporal_image_loss(prediction, torch.zeros_like(prediction), torch.zeros_like(prediction), torch.tensor([0]), torch.tensor([1.0]), torch.full((1, 3, 2, 2), 1.5), TensorVAE(), 1.0, rgb_weight=1.0, edge_weight=0.0)


if __name__ == "__main__":
    unittest.main()
