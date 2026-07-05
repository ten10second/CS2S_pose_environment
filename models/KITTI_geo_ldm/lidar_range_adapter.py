import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


class KittiRangeImageEncoder(nn.Module):
    """Small encoder for KITTI Velodyne range images.

    Expected input channels are normalized range, reflectance, and valid mask.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64, out_channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, range_img: torch.Tensor) -> torch.Tensor:
        return self.net(range_img.float())


class KittiRangeImagePyramidEncoder(nn.Module):
    """Encode a KITTI range image into a small same-channel feature pyramid."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 128,
        num_levels: int = 4,
    ):
        super().__init__()
        self.num_levels = max(1, int(num_levels))
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        blocks = []
        outs = []
        for level in range(self.num_levels):
            stride = 1 if level == 0 else 2
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
            )
            outs.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1))
        self.blocks = nn.ModuleList(blocks)
        self.outs = nn.ModuleList(outs)

    def forward(self, range_img: torch.Tensor):
        h = self.stem(range_img.float())
        features = []
        for block, out in zip(self.blocks, self.outs):
            h = block(h)
            features.append(out(h))
        return features


class KittiRangeToCameraAdapter(nn.Module):
    """Route KITTI range-view features into one front-view camera latent.

    For each camera latent pixel, the module samples multiple depths along the
    camera ray, transforms those samples to the LiDAR frame, projects them to
    range-view coordinates, bilinearly samples range features, and attends over
    the sampled local range features.
    """

    def __init__(
        self,
        image_channels: int,
        range_channels: int,
        hidden_channels: Optional[int] = None,
        depth_samples: int = 16,
        min_depth: float = 1.0,
        max_depth: float = 80.0,
        image_size: Tuple[int, int] = (128, 512),
        fov_up_deg: float = 2.0,
        fov_down_deg: float = -24.9,
        use_depth_embed: bool = True,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        hidden = int(hidden_channels or image_channels)
        self.image_channels = int(image_channels)
        self.range_channels = int(range_channels)
        self.hidden_channels = hidden
        self.depth_samples = int(depth_samples)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.image_size = tuple(image_size)
        self.fov_up = math.radians(float(fov_up_deg))
        self.fov_down = math.radians(float(fov_down_deg))
        self.use_depth_embed = bool(use_depth_embed)
        self.residual_scale = float(residual_scale)

        self.query_proj = nn.Conv2d(self.image_channels, hidden, kernel_size=1)
        self.key_proj = nn.Linear(self.range_channels, hidden)
        self.value_proj = nn.Linear(self.range_channels, hidden)
        if self.use_depth_embed:
            self.depth_proj = nn.Sequential(
                nn.Linear(1, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
        else:
            self.depth_proj = None
        self.out_proj = zero_module(nn.Conv2d(hidden, self.image_channels, kernel_size=1))

    def _depth_values(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.depth_samples <= 1:
            return torch.tensor([(self.min_depth + self.max_depth) * 0.5], device=device, dtype=dtype)
        index = torch.arange(self.depth_samples, device=device, dtype=dtype)
        bin_size = (self.max_depth - self.min_depth) / (self.depth_samples * (1 + self.depth_samples))
        return self.min_depth + bin_size * index * (index + 1.0)

    @staticmethod
    def _expand_batch_matrix(matrix: torch.Tensor, batch_size: int) -> torch.Tensor:
        if matrix.dim() == 2:
            matrix = matrix.unsqueeze(0)
        if matrix.shape[0] == 1 and batch_size > 1:
            matrix = matrix.expand(batch_size, -1, -1)
        if matrix.shape[0] != batch_size:
            raise ValueError(f"Expected batch size {batch_size}, got matrix batch {matrix.shape[0]}")
        return matrix

    def _range_grid(
        self,
        camera_k: torch.Tensor,
        camera_to_lidar: torch.Tensor,
        latent_h: int,
        latent_w: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_h, image_w = tuple(image_size or self.image_size)
        camera_k = self._expand_batch_matrix(camera_k.to(device=device, dtype=dtype), batch_size)
        camera_to_lidar = self._expand_batch_matrix(camera_to_lidar.to(device=device, dtype=dtype), batch_size)

        xs = (torch.arange(latent_w, device=device, dtype=dtype) + 0.5) * (float(image_w) / float(latent_w))
        ys = (torch.arange(latent_h, device=device, dtype=dtype) + 0.5) * (float(image_h) / float(latent_h))
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        u = xx.reshape(1, latent_h * latent_w, 1)
        v = yy.reshape(1, latent_h * latent_w, 1)
        depths = self._depth_values(device, dtype).reshape(1, 1, -1)

        fx = camera_k[:, 0, 0].reshape(batch_size, 1, 1)
        fy = camera_k[:, 1, 1].reshape(batch_size, 1, 1)
        cx = camera_k[:, 0, 2].reshape(batch_size, 1, 1)
        cy = camera_k[:, 1, 2].reshape(batch_size, 1, 1)

        x_cam = (u - cx) / fx.clamp_min(1e-6) * depths
        y_cam = (v - cy) / fy.clamp_min(1e-6) * depths
        z_cam = depths.expand(batch_size, latent_h * latent_w, -1)
        ones = torch.ones_like(z_cam)
        cam_points = torch.stack([x_cam, y_cam, z_cam, ones], dim=-1)

        lidar_points = torch.matmul(camera_to_lidar[:, None, None], cam_points[..., None]).squeeze(-1)[..., :3]
        radius = torch.linalg.norm(lidar_points, dim=-1).clamp_min(1e-6)
        yaw = torch.atan2(lidar_points[..., 1], lidar_points[..., 0])
        pitch = torch.asin((lidar_points[..., 2] / radius).clamp(-1.0, 1.0))

        grid_x = yaw / math.pi
        row_fraction = (self.fov_up - pitch) / (self.fov_up - self.fov_down)
        grid_y = row_fraction * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)

        valid = (
            (radius >= self.min_depth)
            & (radius <= self.max_depth)
            & (grid_x > -1.0)
            & (grid_x < 1.0)
            & (grid_y > -1.0)
            & (grid_y < 1.0)
        )
        return grid, valid, depths.reshape(-1)

    def forward(
        self,
        image_latent: torch.Tensor,
        range_features: torch.Tensor,
        camera_k: torch.Tensor,
        camera_to_lidar: torch.Tensor,
        range_mask: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
        return_attention: bool = False,
    ):
        batch_size, _, latent_h, latent_w = image_latent.shape
        grid, valid, depths = self._range_grid(
            camera_k=camera_k,
            camera_to_lidar=camera_to_lidar,
            latent_h=latent_h,
            latent_w=latent_w,
            batch_size=batch_size,
            device=image_latent.device,
            dtype=image_latent.dtype,
            image_size=image_size,
        )

        flat_grid = grid.view(batch_size, latent_h * latent_w, self.depth_samples, 2)
        sampled = F.grid_sample(
            range_features.to(device=image_latent.device, dtype=image_latent.dtype),
            flat_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.permute(0, 2, 3, 1)

        if range_mask is not None:
            if range_mask.dim() == 3:
                range_mask = range_mask.unsqueeze(1)
            sampled_mask = F.grid_sample(
                range_mask.to(device=image_latent.device, dtype=image_latent.dtype),
                flat_grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            )
            valid = valid & (sampled_mask[:, 0] > 0.5)

        query = self.query_proj(image_latent).flatten(2).transpose(1, 2)
        keys = self.key_proj(sampled)
        values = self.value_proj(sampled)
        if self.depth_proj is not None:
            depth_embed = self.depth_proj((depths / self.max_depth).reshape(1, self.depth_samples, 1))
            keys = keys + depth_embed[:, None]
            values = values + depth_embed[:, None]

        scores = (query[:, :, None] * keys).sum(dim=-1) / math.sqrt(float(self.hidden_channels))
        scores = scores.masked_fill(~valid, -1e4)
        attention = torch.softmax(scores, dim=-1)
        has_valid = valid.any(dim=-1, keepdim=True)
        attention = attention * has_valid.to(dtype=attention.dtype)

        routed = (attention[..., None] * values).sum(dim=2)
        routed = routed.transpose(1, 2).reshape(batch_size, self.hidden_channels, latent_h, latent_w)
        residual = self.out_proj(routed) * self.residual_scale
        output = image_latent + residual

        if return_attention:
            return output, {
                "attention": attention.view(batch_size, latent_h, latent_w, self.depth_samples),
                "valid": valid.view(batch_size, latent_h, latent_w, self.depth_samples),
            }
        return output
