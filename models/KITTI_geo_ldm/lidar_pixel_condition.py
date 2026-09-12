"""Pixel-aligned LiDAR content conditioning; no offline patch averaging."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from models.KITTI_geo_ldm.lidar_condition_model import build_lidar_evidence_maps


class MaskedStrideConv(nn.Module):
    """Learn spatial reduction while excluding missing observations from support."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.register_buffer("count_kernel", torch.ones(1, 1, 3, 3), persistent=False)

    def forward(self, features, mask):
        count = F.conv2d(mask.float(), self.count_kernel.float(), stride=2, padding=1)
        valid = (count > 0).to(features.dtype)
        output = self.conv(features * mask.to(features.dtype))
        output = output * (9.0 / count.clamp_min(1.0)).to(output.dtype)
        return F.silu(output) * valid, valid


class LidarPixelConditionEncoder(nn.Module):
    """Jointly encode feature/depth/hit at each pixel, then build a spatial pyramid.

    Inputs are rendered from the same z-buffer-selected points. Each valid pixel
    reaches a nonlinear pointwise encoder before any spatial reduction. Missing
    pixels remain distinct from measured zero values through explicit masks.
    The 8x32 semantic head is auxiliary only; it is not the content bottleneck.
    """

    uses_pixel_features = True

    def __init__(
        self,
        point_feature_dim=576,
        hidden_channels=64,
        pyramid_channels=(64, 128, 256, 256),
        image_size=(128, 512),
        token_grid=(8, 32),
        semantic_feature_dim=384,
        evidence_dilation=4,
        evidence_free_space_dilation=14,
    ):
        super().__init__()
        self.point_feature_dim = int(point_feature_dim)
        self.image_size = tuple(image_size)
        self.token_grid = tuple(token_grid)
        self.pyramid_channels = tuple(int(c) for c in pyramid_channels)
        if len(self.pyramid_channels) != 4 or min(self.pyramid_channels) < 1:
            raise ValueError("pixel conditioning requires four positive pyramid channel counts")
        if any(int(s) % 64 for s in self.image_size):
            raise ValueError("pixel condition image dimensions must be multiples of 64")
        self.evidence_dilation = int(evidence_dilation)
        self.evidence_free_space_dilation = int(evidence_free_space_dilation)
        self.pixel_encoder = nn.Sequential(
            nn.Conv2d(self.point_feature_dim + 2, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
        )
        channels = (hidden_channels, hidden_channels, *self.pyramid_channels)
        self.down_blocks = nn.ModuleList()
        in_channels = hidden_channels
        for out_channels in channels:
            self.down_blocks.append(MaskedStrideConv(in_channels, out_channels))
            in_channels = out_channels
        self.semantic_head = (
            nn.Conv2d(self.pyramid_channels[1], int(semantic_feature_dim), 1)
            if semantic_feature_dim > 0 else None
        )
        self.last_semantic_pred_tokens = None
        self.last_token_structure_loss = None
        for name in (
            "last_hit_coverage", "last_empty_coverage", "last_point_feature_valid_ratio",
            "last_valid_sample_ratio", "last_point_count_mean", "last_pointmap_coverage",
            "last_ray_token_coverage",
        ):
            self.register_buffer(name, torch.tensor(0.0), persistent=False)

    def make_evidence_maps(self, lidar_cond):
        return build_lidar_evidence_maps(
            lidar_cond, dilation=self.evidence_dilation,
            free_space_dilation=self.evidence_free_space_dilation,
        )

    def encode_pixels(self, features, depth, hit):
        # Normalize each Utonia vector independently, never over spatial pixels.
        features = F.layer_norm(
            features.permute(0, 2, 3, 1).float(), (self.point_feature_dim,)
        ).permute(0, 3, 1, 2)
        log_depth = torch.log1p(depth.float().clamp(0.0, 1.0) * 80.0) / math.log(81.0)
        joint = torch.cat([features, log_depth, hit.float()], dim=1)
        encoded = self.pixel_encoder(joint)
        return encoded * hit.to(encoded.dtype)

    def forward(
        self, lidar_cond, raw_lidar_cond=None, lidar_pixel_features=None,
        lidar_pixel_features_mask=None, **unused,
    ):
        if lidar_pixel_features is None or lidar_pixel_features_mask is None:
            raise ValueError("V2.1 requires pixel features and their validity mask; V2 ray caches are incompatible")
        raw = raw_lidar_cond if raw_lidar_cond is not None else lidar_cond
        features = lidar_pixel_features
        mask = lidar_pixel_features_mask
        expected = (raw.shape[0], self.point_feature_dim, *self.image_size)
        if tuple(features.shape) != expected:
            raise ValueError(f"pixel features have shape {tuple(features.shape)}, expected {expected}")
        if tuple(mask.shape) != (raw.shape[0], 1, *self.image_size):
            raise ValueError("pixel feature mask must be [B,1,H,W] at the configured image size")
        if raw.ndim != 4 or raw.shape[1] < 3 or tuple(raw.shape[-2:]) != self.image_size:
            raise ValueError("raw projected depth/hit must match the pixel condition image")
        features = features.to(device=raw.device)
        mask = (mask.to(device=raw.device) > 0).to(raw.dtype)
        # Raw zero-LiDAR probes deliberately suppress cached content as well.
        hit = mask * (raw[:, 1:2] > 0).to(mask.dtype)
        h = self.encode_pixels(features, raw[:, 2:3], hit)
        outputs, supports = [], []
        support = hit
        for index, block in enumerate(self.down_blocks):
            h, support = block(h, support)
            if index >= 2:
                outputs.append(h)
                supports.append(support)
        if self.semantic_head is not None:
            semantic = self.semantic_head(outputs[1])
            if semantic.shape[-2:] != self.token_grid:
                semantic = F.interpolate(semantic, size=self.token_grid, mode="bilinear", align_corners=False)
            self.last_semantic_pred_tokens = semantic.flatten(2).transpose(1, 2)
        else:
            self.last_semantic_pred_tokens = None
        self.last_token_structure_loss = h.new_zeros(())
        with torch.no_grad():
            self.last_hit_coverage.copy_(hit.float().mean())
            self.last_empty_coverage.zero_()
            self.last_point_feature_valid_ratio.copy_(mask.float().mean())
            self.last_valid_sample_ratio.copy_(hit.flatten(1).any(dim=1).float().mean())
            self.last_point_count_mean.copy_(hit.flatten(1).sum(dim=1).float().mean())
            self.last_pointmap_coverage.copy_(hit.float().mean())
            self.last_ray_token_coverage.copy_(supports[1].float().mean())
        return {"features": tuple(outputs), "masks": tuple(supports)}


class LidarSpatialResidual(nn.Module):
    """Support-gated residual at one U-Net scale, initialized to preserve SD."""

    def __init__(self, condition_channels, unet_channels, gate_bias=-2.0):
        super().__init__()
        self.projection = nn.Conv2d(condition_channels, unet_channels, 1, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.lidar_gate = nn.Conv2d(unet_channels * 2, 1, 1)
        nn.init.zeros_(self.lidar_gate.weight)
        nn.init.constant_(self.lidar_gate.bias, gate_bias)
        for name in ("last_lidar_confidence", "last_lidar_mask_mean", "last_lidar_message_ratio"):
            self.register_buffer(name, torch.tensor(0.0), persistent=False)

    def forward(self, h, condition, mask):
        if condition.shape[0] != h.shape[0] or condition.shape[-2:] != h.shape[-2:]:
            raise ValueError("LiDAR pyramid features must match the U-Net batch and spatial dimensions")
        if tuple(mask.shape) != (h.shape[0], 1, *h.shape[-2:]):
            raise ValueError("LiDAR pyramid support must be [B,1,H,W]")
        delta = self.projection(condition.to(h.dtype))
        confidence = mask.to(h.dtype).clamp(0.0, 1.0) * torch.sigmoid(
            self.lidar_gate(torch.cat([h, delta], dim=1))
        )
        message = confidence * delta
        with torch.no_grad():
            self.last_lidar_confidence.copy_(confidence.float().mean())
            self.last_lidar_mask_mean.copy_(mask.float().mean())
            ratio = message.float().square().mean().sqrt() / h.float().square().mean().sqrt().clamp_min(1e-6)
            self.last_lidar_message_ratio.copy_(ratio)
        return h + message
