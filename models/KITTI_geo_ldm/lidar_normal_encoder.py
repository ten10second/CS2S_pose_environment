import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LidarNormalEncoder(nn.Module):
    """Learned BEV LiDAR encoder with a per-point surface-normal head."""

    def __init__(
        self,
        point_in_channels: int = 6,
        hidden_channels: int = 96,
        feature_channels: int = 128,
        pillar_size: float = 0.5,
        x_range: Tuple[float, float] = (-40.0, 40.0),
        z_range: Tuple[float, float] = (0.0, 80.0),
    ):
        super().__init__()
        self.pillar_size = float(pillar_size)
        self.x_range = tuple(float(v) for v in x_range)
        self.z_range = tuple(float(v) for v in z_range)
        self.x_bins = int(math.ceil((self.x_range[1] - self.x_range[0]) / self.pillar_size))
        self.z_bins = int(math.ceil((self.z_range[1] - self.z_range[0]) / self.pillar_size))
        self.feature_channels = int(feature_channels)

        self.point_mlp = nn.Sequential(
            nn.Linear(point_in_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, feature_channels),
            nn.SiLU(),
        )
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
        )
        self.normal_head = nn.Sequential(
            nn.Linear(feature_channels * 2 + point_in_channels, feature_channels),
            nn.SiLU(),
            nn.Linear(feature_channels, feature_channels // 2),
            nn.SiLU(),
            nn.Linear(feature_channels // 2, 3),
        )

    def _pillar_coords(self, points_rect: torch.Tensor):
        px = torch.floor((points_rect[:, 0] - self.x_range[0]) / self.pillar_size).long()
        pz = torch.floor((points_rect[:, 2] - self.z_range[0]) / self.pillar_size).long()
        valid = (
            (px >= 0)
            & (px < self.x_bins)
            & (pz >= 0)
            & (pz < self.z_bins)
            & torch.isfinite(points_rect).all(dim=1)
        )
        return px, pz, valid

    def _point_features(self, points_rect: torch.Tensor, intensity: torch.Tensor, px: torch.Tensor, pz: torch.Tensor):
        center_x = self.x_range[0] + (px.to(points_rect.dtype) + 0.5) * self.pillar_size
        center_z = self.z_range[0] + (pz.to(points_rect.dtype) + 0.5) * self.pillar_size
        x_norm = torch.clamp(points_rect[:, 0] / 40.0, -1.0, 1.0)
        y_norm = torch.clamp(points_rect[:, 1] / 5.0, -1.0, 1.0)
        z_norm = torch.clamp(points_rect[:, 2] / 80.0, 0.0, 1.0)
        dx = torch.clamp((points_rect[:, 0] - center_x) / self.pillar_size, -1.0, 1.0)
        dz = torch.clamp((points_rect[:, 2] - center_z) / self.pillar_size, -1.0, 1.0)
        refl = torch.clamp(intensity[:, 0], 0.0, 1.0)
        return torch.stack([x_norm, y_norm, z_norm, refl, dx, dz], dim=1)

    def forward(
        self,
        points_rect: torch.Tensor,
        intensity: torch.Tensor,
        batch_index: torch.Tensor,
        batch_size: int = None,
    ) -> Dict[str, torch.Tensor]:
        if batch_size is None:
            batch_size = int(batch_index.max().detach().cpu()) + 1 if batch_index.numel() else 0
        px, pz, valid = self._pillar_coords(points_rect)
        safe_px = px.clamp(0, self.x_bins - 1)
        safe_pz = pz.clamp(0, self.z_bins - 1)
        point_input = self._point_features(points_rect, intensity, safe_px, safe_pz)
        point_feat = self.point_mlp(point_input)

        flat_index = batch_index * (self.z_bins * self.x_bins) + safe_pz * self.x_bins + safe_px
        flat_size = max(batch_size, 1) * self.z_bins * self.x_bins
        bev_flat = point_feat.new_full((flat_size, self.feature_channels), -1e4)
        if point_feat.numel() > 0:
            scatter_index = flat_index[:, None].expand(-1, self.feature_channels)
            bev_flat.scatter_reduce_(0, scatter_index, point_feat, reduce="amax", include_self=True)
        bev_flat = torch.where(bev_flat < -9999.0, torch.zeros_like(bev_flat), bev_flat)
        bev = bev_flat.view(max(batch_size, 1), self.z_bins, self.x_bins, self.feature_channels).permute(0, 3, 1, 2)
        bev_feature = self.bev_encoder(bev)

        gathered = bev_feature[batch_index, :, safe_pz, safe_px]
        fused = torch.cat([point_feat, gathered, point_input], dim=1)
        pred_normal = F.normalize(self.normal_head(fused), dim=1, eps=1e-6)
        return {
            "pred_normal_rect": pred_normal,
            "point_feature": gathered,
            "bev_feature": bev_feature,
            "valid_mask": valid,
        }


def sign_invariant_normal_loss(
    pred_normal: torch.Tensor,
    target_normal: torch.Tensor,
    weight: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    target_normal = F.normalize(target_normal, dim=1, eps=1e-6)
    dot = torch.sum(pred_normal * target_normal, dim=1).abs().clamp(0.0, 1.0)
    loss = 1.0 - dot
    weight = weight.float()
    if valid_mask is not None:
        weight = weight * valid_mask.float()
    denom = weight.sum().clamp_min(1e-6)
    return (loss * weight).sum() / denom


def sign_invariant_angular_error(
    pred_normal: torch.Tensor,
    target_normal: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    target_normal = F.normalize(target_normal, dim=1, eps=1e-6)
    dot = torch.sum(pred_normal * target_normal, dim=1).abs().clamp(0.0, 1.0)
    angle = torch.rad2deg(torch.acos(dot))
    if valid_mask is not None:
        angle = angle[valid_mask.bool()]
    return angle
