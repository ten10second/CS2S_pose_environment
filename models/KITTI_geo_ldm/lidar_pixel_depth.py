import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import conv_nd, normalization


class LidarPixelDepthHead(nn.Module):
    """Predict a sparse-LiDAR supervision map from decoder features.

    The head reads the final U-Net decoder feature map and learns the whole
    8x upsampling path. It never consumes LiDAR targets or LiDAR condition
    features, so the auxiliary loss remains a prediction constraint on the
    generated-side representation.
    """

    def __init__(self, in_channels, dims=2):
        super().__init__()
        if dims != 2:
            raise ValueError("LidarPixelDepthHead only supports 2D U-Net features")
        in_channels = int(in_channels)
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        self.in_channels = in_channels
        self.stem = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(dims, in_channels, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.up_blocks = nn.ModuleList([
            nn.Sequential(conv_nd(dims, 128, 64, 3, padding=1), nn.SiLU()),
            nn.Sequential(conv_nd(dims, 64, 32, 3, padding=1), nn.SiLU()),
            nn.Sequential(conv_nd(dims, 32, 16, 3, padding=1), nn.SiLU()),
        ])
        self.out = conv_nd(dims, 16, 1, 3, padding=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("LidarPixelDepthHead expects NCHW decoder features")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} decoder channels, got {x.shape[1]}"
            )
        h = self.stem(x)
        for block in self.up_blocks:
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            h = block(h)
        return self.out(h)
