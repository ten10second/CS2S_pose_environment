import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


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
    """Lightweight multi-scale LiDAR control for UNet middle and skip features."""

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
    ):
        super().__init__()
        self.control_scale = control_scale
        self.semantic_class_count = int(semantic_class_count)
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

    def forward(self, x, timesteps=None, cond_init_grd=None, cond_sat=None, cond_txt=None, **kwargs):
        if cond_init_grd is None:
            return None

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

        outs = []
        for target_ds, zero_conv in zip(self.input_downsamples, self.skip_outs):
            feature = features.get(target_ds)
            if feature is None:
                size = (max(1, x.shape[-2] // target_ds), max(1, x.shape[-1] // target_ds))
                feature = F.interpolate(features[self.max_downsample], size=size, mode="bilinear", align_corners=False)
            outs.append(zero_conv(feature).type_as(x) * self.control_scale)

        middle_feature = features[self.max_downsample]
        outs.append(self.middle_out(middle_feature).type_as(x) * self.control_scale)
        return outs
