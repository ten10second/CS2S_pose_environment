"""Bounded image-space losses for temporal two-frame diffusion training."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _as_nonnegative_scalar(name: str, value: float) -> float:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.numel() != 1 or not bool(torch.isfinite(tensor).all()) or float(tensor.item()) < 0.0:
        raise ValueError(name + " must be a finite non-negative scalar")
    return float(tensor.item())


def _validate_diffusion_inputs(
    prediction: torch.Tensor,
    noise: torch.Tensor,
    noisy_latent: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    if prediction.shape != noise.shape or prediction.shape != noisy_latent.shape:
        raise ValueError("prediction, noise, and noisy_latent must have identical shapes")
    if prediction.ndim != 4 or not prediction.is_floating_point():
        raise ValueError("prediction must be a floating point latent tensor [B,C,H,W]")
    if not noise.is_floating_point() or not noisy_latent.is_floating_point():
        raise ValueError("noise and noisy_latent must be floating point tensors")
    if timesteps.ndim != 1 or timesteps.shape[0] != prediction.shape[0]:
        raise ValueError("timesteps must be a vector with one entry per sample")
    if alphas_cumprod.ndim != 1 or alphas_cumprod.numel() == 0:
        raise ValueError("alphas_cumprod must be a non-empty vector")
    if not torch.is_floating_point(alphas_cumprod):
        raise ValueError("alphas_cumprod must be floating point")
    if (not bool(torch.isfinite(alphas_cumprod).all()) or bool(torch.any(alphas_cumprod <= 0))
            or bool(torch.any(alphas_cumprod > 1))):
        raise ValueError("alphas_cumprod values must be finite and in (0,1]")
    if timesteps.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("timesteps must contain integer indices")
    if bool(torch.any(timesteps < 0)) or bool(torch.any(timesteps >= alphas_cumprod.numel())):
        raise ValueError("timesteps index outside alphas_cumprod")
    scale = torch.as_tensor(scale_factor, dtype=torch.float32)
    if scale.numel() != 1 or not bool(torch.isfinite(scale).all()) or float(scale.item()) <= 0.0:
        raise ValueError("scale_factor must be finite and positive")
    return alphas_cumprod.to(device=prediction.device, dtype=torch.float32)[timesteps.to(device=prediction.device, dtype=torch.long)]


def _sobel_xy(rgb: torch.Tensor) -> torch.Tensor:
    channels = rgb.shape[1]
    kernel_x = rgb.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    kernel_y = rgb.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
    kernel = torch.stack((kernel_x, kernel_y), dim=0).view(2, 1, 3, 3).repeat(channels, 1, 1, 1)
    padded = F.pad(rgb, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel, groups=channels)


def temporal_image_loss(
    prediction: torch.Tensor,
    noise: torch.Tensor,
    noisy_latent: torch.Tensor,
    timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    target_rgb: torch.Tensor,
    vae,
    scale_factor: float,
    rgb_weight: float,
    edge_weight: float,
    charbonnier_eps: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return epsilon MSE plus optional bounded decoded-RGB and Sobel losses.

    ``prediction`` is the model epsilon prediction for ``noisy_latent``. The
    image terms reconstruct the predicted clean latent, decode it through the
    supplied frozen VAE without detaching, map decoder output from [-1,1] to RGB,
    and compare against the current ground-truth RGB in [0,1].
    """
    rgb_weight_value = _as_nonnegative_scalar("rgb_weight", rgb_weight)
    edge_weight_value = _as_nonnegative_scalar("edge_weight", edge_weight)
    charbonnier_eps_value = _as_nonnegative_scalar("charbonnier_eps", charbonnier_eps)
    if charbonnier_eps_value <= 0.0:
        raise ValueError("charbonnier_eps must be positive")

    alpha = _validate_diffusion_inputs(prediction, noise, noisy_latent, timesteps, alphas_cumprod, scale_factor)
    if target_rgb.ndim != 4 or target_rgb.shape[0] != prediction.shape[0] or target_rgb.shape[1] != 3:
        raise ValueError("target_rgb must be [B,3,H,W] and match the latent batch")
    if not target_rgb.is_floating_point():
        raise ValueError("target_rgb must be floating point")
    target_f = target_rgb.to(device=prediction.device, dtype=torch.float32)
    if (not bool(torch.isfinite(target_f).all()) or bool(torch.any(target_f < 0.0))
            or bool(torch.any(target_f > 1.0))):
        raise ValueError("target_rgb must be finite and in [0,1]")

    prediction_f = prediction.float()
    noise_f = noise.float()
    epsilon_per_sample = (prediction_f - noise_f).pow(2).flatten(1).mean(dim=1)
    epsilon_loss = epsilon_per_sample.mean()

    rgb_loss = epsilon_loss.new_zeros(())
    edge_loss = epsilon_loss.new_zeros(())
    rgb_raw = epsilon_loss.new_zeros(())
    edge_raw = epsilon_loss.new_zeros(())
    sample_weight = torch.sqrt(alpha)
    if rgb_weight_value > 0.0 or edge_weight_value > 0.0:
        alpha_view = alpha.view(-1, 1, 1, 1)
        noisy_f = noisy_latent.float()
        z0_hat = (noisy_f - torch.sqrt(1.0 - alpha_view) * prediction_f) / torch.sqrt(alpha_view)
        with torch.cuda.amp.autocast(enabled=False):
            decoded = vae.decode(z0_hat / float(scale_factor))
            if not torch.is_tensor(decoded):
                raise TypeError("vae.decode must return a tensor")
            predicted_rgb = (decoded.float() + 1.0) * 0.5
            if predicted_rgb.shape != target_f.shape:
                raise ValueError("decoded prediction shape must match target_rgb")
            if rgb_weight_value > 0.0:
                charbonnier = torch.sqrt((predicted_rgb - target_f).pow(2) + charbonnier_eps_value ** 2) - charbonnier_eps_value
                rgb_per_sample = charbonnier.flatten(1).mean(dim=1)
                rgb_raw = rgb_per_sample.mean()
                rgb_loss = (rgb_per_sample * sample_weight).mean() * rgb_weight_value
            if edge_weight_value > 0.0:
                edge_delta = (_sobel_xy(predicted_rgb) - _sobel_xy(target_f)).abs()
                b, _, h, w = edge_delta.shape
                edge_per_axis = edge_delta.view(b, target_f.shape[1], 2, h, w).mean(dim=(1, 3, 4))
                edge_per_sample = edge_per_axis.sum(dim=1)
                edge_raw = edge_per_sample.mean()
                edge_loss = (edge_per_sample * sample_weight).mean() * edge_weight_value

    total = epsilon_loss + rgb_loss + edge_loss
    metrics = {
        "epsilon": epsilon_loss.detach(),
        "rgb": rgb_loss.detach(),
        "edge": edge_loss.detach(),
        "rgb_raw": rgb_raw.detach(),
        "edge_raw": edge_raw.detach(),
        "time_weight": sample_weight.mean().detach(),
        "total": total.detach(),
    }
    return total, metrics
