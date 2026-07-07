import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.KITTI_geo_ldm.lidar_range_adapter import (
    KittiRangeImagePyramidEncoder,
    KittiRangeToCameraAdapter,
)


def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


def build_lidar_evidence_maps(
    lidar_cond: torch.Tensor,
    dilation: int = 4,
    free_space_dilation: int = 14,
    depth_edge_threshold: float = 0.03,
) -> torch.Tensor:
    """Build deterministic camera-view evidence from projected KITTI LiDAR.

    Channels are depth, hit, inverse-depth, depth-edge, dilated-surface,
    approximate-empty-ray confidence, unknown, and bbox/class prior.
    The empty/free-space channel is deliberately conservative: it marks areas
    without nearby projected hits as weak negative evidence rather than a hard
    physical occupancy proof.
    """
    cond = lidar_cond.float()
    b, _, h, w = cond.shape
    device = cond.device
    dtype = cond.dtype
    zero = torch.zeros((b, 1, h, w), device=device, dtype=dtype)

    bbox_or_class = cond[:, 0:1].clamp(0.0, 1.0) if cond.shape[1] > 0 else zero
    hit = cond[:, 1:2].clamp(0.0, 1.0) if cond.shape[1] > 1 else zero
    depth = cond[:, 2:3].clamp(0.0, 1.0) * hit if cond.shape[1] > 2 else zero
    inv_depth = (1.0 - depth).clamp(0.0, 1.0) * hit

    dil_kernel = max(1, int(dilation) * 2 + 1)
    dilated_hit = F.max_pool2d(hit, kernel_size=dil_kernel, stride=1, padding=int(dilation)).clamp(0.0, 1.0)

    edge_kernel = 3
    local_max = F.max_pool2d(depth, kernel_size=edge_kernel, stride=1, padding=1)
    masked_depth = torch.where(hit > 0.5, depth, torch.ones_like(depth))
    local_min = -F.max_pool2d(-masked_depth, kernel_size=edge_kernel, stride=1, padding=1)
    depth_edge = ((local_max - local_min) > float(depth_edge_threshold)).to(dtype) * F.max_pool2d(
        hit, kernel_size=edge_kernel, stride=1, padding=1
    )

    free_kernel = max(1, int(free_space_dilation) * 2 + 1)
    nearby_hit = F.max_pool2d(hit, kernel_size=free_kernel, stride=1, padding=int(free_space_dilation)).clamp(0.0, 1.0)
    empty_confidence = (1.0 - nearby_hit).clamp(0.0, 1.0)
    unknown = (1.0 - dilated_hit).clamp(0.0, 1.0)

    return torch.cat(
        [
            depth,
            hit,
            inv_depth,
            depth_edge,
            dilated_hit,
            empty_confidence,
            unknown,
            bbox_or_class,
        ],
        dim=1,
    )


class LidarRangeTokenEncoder(nn.Module):
    """Camera-aligned LiDAR/range tokens for main UNet cross-attention."""

    def __init__(
        self,
        front_in_channels: int = 4,
        range_in_channels: int = 3,
        hidden_channels: int = 128,
        range_feature_channels: int = 128,
        token_dim: int = 768,
        token_grid: Tuple[int, int] = (8, 32),
        range_depth_samples: int = 8,
        image_size: Tuple[int, int] = (128, 512),
        use_evidence_maps: bool = True,
        evidence_dilation: int = 4,
        evidence_free_space_dilation: int = 14,
        token_output_norm: str = "none",
        token_structure_target_ratio: float = 0.0,
        fixed_coord_pos_scale: float = 0.25,
        use_pointmap_pe: bool = False,
        pointmap_pe_scale: float = 1.0,
    ):
        super().__init__()
        self.token_grid = tuple(token_grid)
        self.image_size = tuple(image_size)
        self.use_evidence_maps = bool(use_evidence_maps)
        self.evidence_dilation = int(evidence_dilation)
        self.evidence_free_space_dilation = int(evidence_free_space_dilation)
        self.token_output_norm_mode = str(token_output_norm or "none")
        self.fixed_coord_pos_scale = float(fixed_coord_pos_scale)
        if self.token_output_norm_mode not in {"none", "layernorm", "center_layernorm"}:
            raise ValueError(f"Unsupported token_output_norm: {self.token_output_norm_mode}")
        self.token_structure_target_ratio = float(token_structure_target_ratio)
        self.use_pointmap_pe = bool(use_pointmap_pe)
        self.pointmap_pe_scale = float(pointmap_pe_scale)
        self.evidence_channels = 8 if self.use_evidence_maps else 0
        self.pointmap_pe_channels = 24 if self.use_pointmap_pe else 0
        self.front_stem = nn.Sequential(
            nn.Conv2d(
                front_in_channels + self.evidence_channels + self.pointmap_pe_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.range_pyramid_encoder = KittiRangeImagePyramidEncoder(
            in_channels=range_in_channels,
            hidden_channels=max(32, int(range_feature_channels) // 2),
            out_channels=range_feature_channels,
            num_levels=1,
        )
        self.range_to_camera = KittiRangeToCameraAdapter(
            image_channels=hidden_channels,
            range_channels=range_feature_channels,
            hidden_channels=hidden_channels,
            depth_samples=range_depth_samples,
            image_size=self.image_size,
            residual_scale=1.0,
        )
        self.token_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, token_dim, kernel_size=1),
        )
        self.token_output_norm = (
            nn.LayerNorm(token_dim) if self.token_output_norm_mode in {"layernorm", "center_layernorm"} else None
        )
        token_count = int(self.token_grid[0]) * int(self.token_grid[1])
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, token_dim))
        self.register_buffer("last_valid_sample_ratio", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_hit_coverage", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_empty_coverage", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_pointmap_coverage", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_token_mean_norm", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_token_centered_norm", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_token_centered_to_mean_ratio", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_token_var_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_token_proj_bias_norm", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_token_output_mean_norm", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_token_output_centered_norm", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_token_output_centered_to_mean_ratio", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_token_output_var_mean", torch.tensor(0.0), persistent=False)
        self.last_token_structure_loss = None

    @staticmethod
    def coord_encoding_2d(height: int, width: int, dim: int, device, dtype) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, int(height), device=device, dtype=dtype).view(int(height), 1)
        x = torch.linspace(-1.0, 1.0, int(width), device=device, dtype=dtype).view(1, int(width))
        x = x.expand(int(height), int(width))
        y = y.expand(int(height), int(width))
        features = [x, y, x * y, x.square(), y.square()]
        for freq in (1.0, 2.0, 4.0, 8.0):
            features.extend(
                [
                    torch.sin(torch.pi * freq * x),
                    torch.cos(torch.pi * freq * x),
                    torch.sin(torch.pi * freq * y),
                    torch.cos(torch.pi * freq * y),
                ]
            )
        base = torch.stack(features, dim=-1).reshape(1, int(height) * int(width), -1)
        base = base - base.mean(dim=1, keepdim=True)
        base = base / base.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        repeat = int((int(dim) + int(base.shape[-1]) - 1) // int(base.shape[-1]))
        return base.repeat(1, 1, repeat)[..., : int(dim)]

    def make_evidence_maps(self, lidar_cond: torch.Tensor) -> torch.Tensor:
        return build_lidar_evidence_maps(
            lidar_cond,
            dilation=self.evidence_dilation,
            free_space_dilation=self.evidence_free_space_dilation,
        )

    def make_pointmap_pe(self, lidar_cond: torch.Tensor) -> torch.Tensor:
        if not self.use_pointmap_pe:
            b, _, h, w = lidar_cond.shape
            return lidar_cond.new_zeros((b, 0, h, w))
        b, _, h, w = lidar_cond.shape
        if lidar_cond.shape[1] < 7:
            self.last_pointmap_coverage.zero_()
            return lidar_cond.new_zeros((b, self.pointmap_pe_channels, h, w))
        coords = lidar_cond[:, 4:7].float()
        hit = (lidar_cond[:, 1:2].float() > 0.0).to(dtype=coords.dtype)
        features = []
        for freq in (1.0, 2.0, 4.0, 8.0):
            phase = 2.0 * torch.pi * float(freq) * coords
            features.append(torch.sin(phase) * hit)
            features.append(torch.cos(phase) * hit)
        self.last_pointmap_coverage.copy_(hit.mean().detach().to(self.last_pointmap_coverage.device))
        return float(self.pointmap_pe_scale) * torch.cat(features, dim=1).to(dtype=lidar_cond.dtype)

    def forward(
        self,
        lidar_cond: torch.Tensor,
        raw_lidar_cond: Optional[torch.Tensor] = None,
        range_img: Optional[torch.Tensor] = None,
        range_mask: Optional[torch.Tensor] = None,
        camera_k: Optional[torch.Tensor] = None,
        camera_to_lidar: Optional[torch.Tensor] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        evidence_source = raw_lidar_cond if raw_lidar_cond is not None else lidar_cond
        front_input = lidar_cond.float()
        if self.use_evidence_maps:
            evidence = self.make_evidence_maps(evidence_source)
            front_input = torch.cat([front_input, evidence], dim=1)
            self.last_hit_coverage.copy_(evidence[:, 1:2].mean().detach().to(self.last_hit_coverage.device))
            self.last_empty_coverage.copy_(evidence[:, 5:6].mean().detach().to(self.last_empty_coverage.device))
        if self.use_pointmap_pe:
            front_input = torch.cat([front_input, self.make_pointmap_pe(evidence_source)], dim=1)
        else:
            self.last_pointmap_coverage.zero_()
        front = self.front_stem(front_input)
        front = F.adaptive_avg_pool2d(front, self.token_grid)

        if range_img is not None and camera_k is not None and camera_to_lidar is not None:
            range_features = self.range_pyramid_encoder(range_img.float())[0]
            front, aux = self.range_to_camera(
                front,
                range_features,
                camera_k=camera_k,
                camera_to_lidar=camera_to_lidar,
                range_mask=range_mask,
                image_size=tuple(image_size or self.image_size),
                return_attention=True,
            )
            self.last_valid_sample_ratio.copy_(aux["valid"].float().mean().detach().to(self.last_valid_sample_ratio.device))
        else:
            self.last_valid_sample_ratio.zero_()

        tokens = self.token_proj(front).flatten(2).transpose(1, 2)
        token_mean = tokens.mean(dim=1, keepdim=True)
        token_centered = tokens - token_mean
        mean_norm_per_sample = token_mean.norm(dim=-1).squeeze(1)
        centered_norm_per_sample = token_centered.norm(dim=-1).mean(dim=1)
        token_ratio = centered_norm_per_sample / mean_norm_per_sample.clamp_min(1e-8)
        if self.token_structure_target_ratio > 0.0:
            target = torch.as_tensor(
                self.token_structure_target_ratio,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            self.last_token_structure_loss = F.relu(target - token_ratio).mean()
        else:
            self.last_token_structure_loss = tokens.new_zeros(())

        output_tokens = tokens
        if self.token_output_norm_mode == "center_layernorm":
            output_tokens = output_tokens - output_tokens.mean(dim=1, keepdim=True)
        if self.token_output_norm is not None:
            output_tokens = self.token_output_norm(output_tokens)

        with torch.no_grad():
            mean_norm = mean_norm_per_sample.mean()
            centered_norm = centered_norm_per_sample.mean()
            self.last_token_mean_norm.copy_(mean_norm.detach().to(self.last_token_mean_norm.device))
            self.last_token_centered_norm.copy_(
                centered_norm.detach().to(self.last_token_centered_norm.device)
            )
            self.last_token_centered_to_mean_ratio.copy_(
                token_ratio.mean().detach().to(self.last_token_centered_to_mean_ratio.device)
            )
            self.last_token_var_mean.copy_(
                tokens.float().var(dim=1, unbiased=False).mean().detach().to(self.last_token_var_mean.device)
            )
            output_mean = output_tokens.mean(dim=1, keepdim=True)
            output_centered = output_tokens - output_mean
            output_mean_norm = output_mean.norm(dim=-1).mean()
            output_centered_norm = output_centered.norm(dim=-1).mean()
            self.last_token_output_mean_norm.copy_(
                output_mean_norm.detach().to(self.last_token_output_mean_norm.device)
            )
            self.last_token_output_centered_norm.copy_(
                output_centered_norm.detach().to(self.last_token_output_centered_norm.device)
            )
            self.last_token_output_centered_to_mean_ratio.copy_(
                (output_centered_norm / output_mean_norm.clamp_min(1e-8))
                .detach()
                .to(self.last_token_output_centered_to_mean_ratio.device)
            )
            self.last_token_output_var_mean.copy_(
                output_tokens.float()
                .var(dim=1, unbiased=False)
                .mean()
                .detach()
                .to(self.last_token_output_var_mean.device)
            )
            final_proj = self.token_proj[-1]
            if getattr(final_proj, "bias", None) is not None:
                self.last_token_proj_bias_norm.copy_(
                    final_proj.bias.detach().float().norm().to(self.last_token_proj_bias_norm.device)
                )
            else:
                self.last_token_proj_bias_norm.zero_()
        fixed_pos = self.coord_encoding_2d(
            self.token_grid[0],
            self.token_grid[1],
            output_tokens.shape[-1],
            device=output_tokens.device,
            dtype=output_tokens.dtype,
        )
        return (
            output_tokens
            + self.pos_embed.to(device=output_tokens.device, dtype=output_tokens.dtype)
            + float(self.fixed_coord_pos_scale) * fixed_pos
        )


def _input_block_layout(model_channels: int, channel_mult, num_res_blocks: int):
    channels = [model_channels]
    downsamples = [1]
    ch = model_channels
    ds = 1
    for level, mult in enumerate(channel_mult):
        for _ in range(num_res_blocks):
            ch = model_channels * mult
            channels.append(ch)
            downsamples.append(ds)
        if level != len(channel_mult) - 1:
            downsamples.append(ds * 2)
            channels.append(ch)
            ds *= 2
    return channels, downsamples


class LidarMultiScaleControl(nn.Module):
    """Projected front-view LiDAR control residuals for KITTI latent UNet."""

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 320,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        hidden_channels: int = 128,
        middle_channels: int = 1280,
        control_scale: float = 1.0,
        semantic_class_count: int = 0,
        semantic_class_scale: float = 1.0,
        gate_channel: int = -1,
        gate_residuals: bool = False,
        gate_dilation: int = 0,
    ):
        super().__init__()
        self.control_scale = float(control_scale)
        self.semantic_class_count = int(semantic_class_count)
        self.gate_channel = int(gate_channel)
        self.gate_residuals = bool(gate_residuals)
        self.gate_dilation = int(gate_dilation)
        self.register_buffer(
            "semantic_class_scale_state",
            torch.tensor(float(semantic_class_scale), dtype=torch.float32),
            persistent=True,
        )
        self.input_channels, self.input_downsamples = _input_block_layout(
            model_channels, tuple(channel_mult), num_res_blocks
        )
        self.max_downsample = max(self.input_downsamples)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.down_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
                for _ in range(int(self.max_downsample).bit_length() - 1)
            ]
        )
        if self.semantic_class_count > 0:
            self.class_embedding = nn.Embedding(self.semantic_class_count, hidden_channels)
            nn.init.zeros_(self.class_embedding.weight)
        else:
            self.class_embedding = None
        self.skip_outs = nn.ModuleList(
            [zero_module(nn.Conv2d(hidden_channels, channels, kernel_size=1)) for channels in self.input_channels]
        )
        self.middle_out = zero_module(nn.Conv2d(hidden_channels, middle_channels, kernel_size=1))

    def _control_gate_source(self, cond_init_grd):
        if not self.gate_residuals or self.gate_channel < 0 or self.gate_channel >= cond_init_grd.shape[1]:
            return None
        gate = cond_init_grd[:, self.gate_channel : self.gate_channel + 1].float().clamp(0.0, 1.0)
        if self.gate_dilation > 0:
            kernel = 2 * self.gate_dilation + 1
            gate = F.max_pool2d(gate, kernel_size=kernel, stride=1, padding=self.gate_dilation)
        return gate.clamp(0.0, 1.0)

    def _front_features(self, x, cond_init_grd):
        cond = F.interpolate(cond_init_grd.float(), size=x.shape[-2:], mode="nearest")
        h = self.stem(cond)
        if self.class_embedding is not None and cond.shape[1] >= 4:
            class_mask = (cond[:, 3:4] > 0.0).to(dtype=h.dtype)
            class_ids = torch.round(cond[:, 3] * float(self.semantic_class_count)).long()
            class_ids = class_ids.clamp_(0, self.semantic_class_count - 1)
            class_features = self.class_embedding(class_ids).permute(0, 3, 1, 2).to(dtype=h.dtype)
            class_scale = self.semantic_class_scale_state.to(device=h.device, dtype=h.dtype)
            h = h + class_scale * class_features * class_mask

        features = {1: h}
        ds = 1
        for block in self.down_blocks:
            ds *= 2
            h = block(h)
            features[ds] = h
        return features

    def _residuals_from_features(self, x, features, gate_source):
        outs = []
        for target_ds, zero_conv in zip(self.input_downsamples, self.skip_outs):
            feature = features.get(target_ds)
            if feature is None:
                size = (max(1, x.shape[-2] // target_ds), max(1, x.shape[-1] // target_ds))
                feature = F.interpolate(features[self.max_downsample], size=size, mode="bilinear", align_corners=False)
            residual = zero_conv(feature).type_as(x) * self.control_scale
            if gate_source is not None:
                gate = F.interpolate(gate_source, size=residual.shape[-2:], mode="bilinear", align_corners=False)
                residual = residual * gate.to(device=residual.device, dtype=residual.dtype)
            outs.append(residual)

        middle_feature = features[self.max_downsample]
        middle_residual = self.middle_out(middle_feature).type_as(x) * self.control_scale
        if gate_source is not None:
            gate = F.interpolate(gate_source, size=middle_residual.shape[-2:], mode="bilinear", align_corners=False)
            middle_residual = middle_residual * gate.to(device=middle_residual.device, dtype=middle_residual.dtype)
        outs.append(middle_residual)
        return outs

    def forward(self, x, timesteps=None, cond_init_grd=None, cond_sat=None, cond_txt=None, **kwargs):
        if cond_init_grd is None:
            return None
        gate_source = self._control_gate_source(cond_init_grd)
        features = self._front_features(x, cond_init_grd)
        return self._residuals_from_features(x, features, gate_source)


class LidarGeometricCrossAttentionControlNet(LidarMultiScaleControl):
    """ControlNet shell plus X-Drive-style range-image-to-camera routing."""

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 320,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        hidden_channels: int = 128,
        middle_channels: int = 1280,
        control_scale: float = 1.0,
        semantic_class_count: int = 0,
        semantic_class_scale: float = 1.0,
        gate_channel: int = -1,
        gate_residuals: bool = False,
        gate_dilation: int = 0,
        range_in_channels: int = 3,
        range_feature_channels: int = 128,
        range_hidden_channels: int = 128,
        range_depth_samples: int = 16,
        range_residual_scale: float = 1.0,
        range_image_size=(128, 512),
        geo_xattn_scales=(1, 2, 4, 8),
    ):
        super().__init__(
            in_channels=in_channels,
            model_channels=model_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            hidden_channels=hidden_channels,
            middle_channels=middle_channels,
            control_scale=control_scale,
            semantic_class_count=semantic_class_count,
            semantic_class_scale=semantic_class_scale,
            gate_channel=gate_channel,
            gate_residuals=gate_residuals,
            gate_dilation=gate_dilation,
        )
        if isinstance(geo_xattn_scales, str):
            geo_xattn_scales = [int(item) for item in geo_xattn_scales.split(",") if item.strip()]
        self.geo_xattn_scales = tuple(sorted({int(scale) for scale in geo_xattn_scales if int(scale) >= 1}))
        if not self.geo_xattn_scales:
            self.geo_xattn_scales = (1, 2, 4, 8)
        self.range_image_size = tuple(range_image_size)
        self.range_pyramid_encoder = KittiRangeImagePyramidEncoder(
            in_channels=range_in_channels,
            hidden_channels=max(32, int(range_feature_channels) // 2),
            out_channels=range_feature_channels,
            num_levels=max(1, len(self.geo_xattn_scales)),
        )
        self.geo_xattn_adapters = nn.ModuleDict(
            {
                str(scale): KittiRangeToCameraAdapter(
                    image_channels=hidden_channels,
                    range_channels=range_feature_channels,
                    hidden_channels=range_hidden_channels,
                    depth_samples=range_depth_samples,
                    image_size=self.range_image_size,
                    residual_scale=range_residual_scale,
                )
                for scale in self.geo_xattn_scales
            }
        )
        for adapter in self.geo_xattn_adapters.values():
            nn.init.normal_(adapter.out_proj.weight, mean=0.0, std=1e-3)
            if adapter.out_proj.bias is not None:
                nn.init.zeros_(adapter.out_proj.bias)
        self.register_buffer("last_geo_xattn_valid_sample_ratio", torch.tensor(0.0), persistent=False)

    def _range_feature_for_scale(self, range_features, scale: int):
        try:
            index = self.geo_xattn_scales.index(int(scale))
        except ValueError:
            index = len(range_features) - 1
        return range_features[min(index, len(range_features) - 1)]

    def _apply_geo_xattn(self, features, range_img, range_mask, camera_k, camera_to_lidar, image_size):
        if range_img is None or camera_k is None or camera_to_lidar is None:
            self.last_geo_xattn_valid_sample_ratio.zero_()
            return features
        range_features = self.range_pyramid_encoder(range_img)
        valid_ratios = []
        for scale in self.geo_xattn_scales:
            if scale not in features or str(scale) not in self.geo_xattn_adapters:
                continue
            adapter = self.geo_xattn_adapters[str(scale)]
            routed, aux = adapter(
                features[scale],
                self._range_feature_for_scale(range_features, scale),
                camera_k=camera_k,
                camera_to_lidar=camera_to_lidar,
                range_mask=range_mask,
                image_size=image_size,
                return_attention=True,
            )
            features[scale] = routed
            valid_ratios.append(aux["valid"].float().mean().detach())
        if valid_ratios:
            self.last_geo_xattn_valid_sample_ratio.copy_(
                torch.stack(valid_ratios).mean().to(self.last_geo_xattn_valid_sample_ratio.device)
            )
        else:
            self.last_geo_xattn_valid_sample_ratio.zero_()
        return features

    def forward(
        self,
        x,
        timesteps=None,
        cond_init_grd=None,
        cond_sat=None,
        cond_txt=None,
        range_img=None,
        range_mask=None,
        camera_to_lidar=None,
        camera_k=None,
        left_camera_k=None,
        image_size=None,
        **kwargs,
    ):
        if cond_init_grd is None:
            return None
        gate_source = self._control_gate_source(cond_init_grd)
        features = self._front_features(x, cond_init_grd)
        camera_k = camera_k if camera_k is not None else left_camera_k
        routed_size = tuple(image_size) if image_size is not None else tuple(cond_init_grd.shape[-2:])
        features = self._apply_geo_xattn(
            features,
            range_img=range_img,
            range_mask=range_mask,
            camera_k=camera_k,
            camera_to_lidar=camera_to_lidar,
            image_size=routed_size,
        )
        return self._residuals_from_features(x, features, gate_source)
