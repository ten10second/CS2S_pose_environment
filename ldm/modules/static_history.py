"""Inject geometrically aligned static RGB without another correspondence search."""

import torch
from torch import nn
from torch.nn import functional as F

from ldm.modules.persistent_history import DENSE_HISTORY_FIELDS, STATIC_HISTORY_FIELDS, validate_history


def pool_static_reference(rgb, mask, confidence, size):
    """Average only observed colors; sparse black holes are not RGB evidence.

    Returns RGB, observed area fraction, and mean confidence on observed pixels.
    Accumulate in float32 so sparse masks remain usable under mixed precision.
    """
    if rgb.shape[-2] < size[0] or rgb.shape[-1] < size[1]:
        raise ValueError("static RGB resolution must be at least the decoder resolution")
    support = F.adaptive_avg_pool2d(mask.float(), size)
    divisor = support.clamp_min(torch.finfo(torch.float32).tiny)
    colors = F.adaptive_avg_pool2d(rgb.float() * mask, size) / divisor
    certainty = F.adaptive_avg_pool2d(confidence.float() * mask, size) / divisor
    return colors, support, certainty


class StaticHistoryAdapter(nn.Module):
    """One local residual from aligned RGB, support, and confidence.

    Previous latent remains in the payload for the shared history/CFG contract;
    appearance comes exclusively from static_rgb. No pooling across samples or
    learned spatial lookup is used. The caller supplies geometric alignment.
    """

    def __init__(self, channels, hidden_dim=64):
        super().__init__()
        self.mode = "static"
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, hidden_dim, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.SiLU())
        self.output = nn.Conv2d(hidden_dim, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.last_metrics = {}

    def _anchor(self):
        return sum(p.reshape(-1)[0] * 0.0 for p in self.parameters())

    def forward(self, feature, context=None, history=None, temb=None):
        anchor = self._anchor() if self.training else 0.0
        if history is None:
            self.last_metrics = {}
            return feature + anchor
        b, _, h, w = feature.shape
        validate_history(history, b)
        if not STATIC_HISTORY_FIELDS <= set(history):
            raise ValueError("static mode requires aligned RGB, mask, and confidence")
        if history["latent"].shape[2:] != (h, w):
            raise ValueError("history must use the decoder's latent resolution")
        if history["latent"].device != feature.device:
            raise ValueError("history and current features must be on the same device")
        colors, support, certainty = pool_static_reference(
            history["static_rgb"].detach(), history["static_mask"],
            history["static_confidence"].detach(), (h, w))
        encoded = self.encoder(torch.cat([colors, support, certainty], dim=1).to(feature.dtype))
        enabled = history.get("enabled", torch.ones(b, device=feature.device, dtype=torch.bool))
        gate = (support > 0) * certainty * enabled[:, None, None, None]
        residual = self.output(encoded) * gate.to(feature.dtype)
        self.last_metrics = {
            "history_residual_rms": residual.detach().float().square().mean().sqrt(),
            "static_support": (support > 0).detach().float().mean(),
            "static_coverage": support.detach().mean(),
            "static_confidence": certainty.detach().sum() / (support > 0).sum().clamp_min(1),
        }
        return feature + residual + anchor


class DenseStaticHistoryAdapter(nn.Module):
    """Dense RGB history adapter for measured/estimated static references.

    The encoder always sees six channels at the supplied reference resolution:
    RGB plus valid/measured/estimated mask channels.  `input_variant` changes
    which mask channels are populated while preserving the architecture:

    - rgb: RGB only, mask channels zero.
    - valid: RGB plus valid mask.
    - types: RGB plus valid, measured, estimated masks.

    The residual is not multiplied by a per-pixel valid mask.  Only two
    sample-level gates are applied: `enabled` and an empty-valid guard.  This is
    deliberate so RGB/valid/types are meaningful A/B/C conditions rather than
    different output masks.
    """

    def __init__(self, channels, hidden_dim=64, input_variant="types"):
        super().__init__()
        if input_variant not in {"rgb", "valid", "types"}:
            raise ValueError("static_dense input_variant must be rgb, valid, or types")
        self.mode = "static_dense"
        self.hidden_dim = int(hidden_dim)
        self.input_variant = input_variant
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, hidden_dim, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.output = nn.Conv2d(hidden_dim, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.last_metrics = {}

    def _anchor(self):
        return sum(p.reshape(-1)[0] * 0.0 for p in self.parameters())

    def _dense_history(self, history):
        if history is not None and "dense_features" in history:
            return {key: value for key, value in history.items() if key != "dense_features"}
        return history

    def _make_input(self, history, dtype):
        rgb = history["dense_rgb"].detach()
        valid = history["dense_valid"].detach().to(rgb.dtype)
        measured = history["dense_measured"].detach().to(rgb.dtype)
        estimated = history["dense_estimated"].detach().to(rgb.dtype)
        if self.input_variant == "rgb":
            valid = torch.zeros_like(valid)
            measured = torch.zeros_like(measured)
            estimated = torch.zeros_like(estimated)
        elif self.input_variant == "valid":
            measured = torch.zeros_like(measured)
            estimated = torch.zeros_like(estimated)
        return torch.cat([rgb, valid, measured, estimated], dim=1).to(dtype)

    def _validate_dense_cache(self, cache, feature):
        if self.training or torch.is_grad_enabled():
            raise ValueError("dense history feature cache is only allowed in eval no_grad inference")
        expected = (feature.shape[0], self.hidden_dim, feature.shape[2], feature.shape[3])
        if (not torch.is_tensor(cache) or cache.shape != expected or cache.device != feature.device
                or cache.dtype != feature.dtype or not torch.isfinite(cache).all()):
            raise ValueError("dense_features cache must be finite [B,hidden_dim,H,W] on the feature device/dtype")

    def encode_history(self, history):
        return self.encoder(self._make_input(history, next(self.parameters()).dtype))

    def _encoded_history(self, history, feature):
        if "dense_features" in history:
            encoded = history["dense_features"]
            self._validate_dense_cache(encoded, feature)
        else:
            encoded = self.encode_history(history)
        if encoded.shape[-2:] != feature.shape[-2:]:
            encoded = F.interpolate(encoded, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return encoded.to(feature.dtype)

    def _sample_gate(self, history, feature):
        enabled = history.get("enabled", torch.ones(feature.shape[0], device=feature.device, dtype=torch.bool))
        nonempty = history["dense_valid"].flatten(1).any(dim=1)
        return (enabled & nonempty).to(feature.dtype)[:, None, None, None]

    def forward(self, feature, context=None, history=None, temb=None):
        anchor = self._anchor() if self.training else 0.0
        if history is None:
            self.last_metrics = {}
            return feature + anchor
        b, _, h, w = feature.shape
        validate_history(self._dense_history(history), b)
        if not DENSE_HISTORY_FIELDS <= set(history):
            raise ValueError("static_dense mode requires dense RGB and masks")
        if history["latent"].shape[2:] != (h, w):
            raise ValueError("history must use the decoder's latent resolution")
        if history["latent"].device != feature.device:
            raise ValueError("history and current features must be on the same device")
        encoded = self._encoded_history(history, feature)
        sample_gate = self._sample_gate(history, feature)
        residual = self.output(encoded) * sample_gate
        self.last_metrics = {
            "history_residual_rms": residual.detach().float().square().mean().sqrt(),
            "dense_valid_coverage": history["dense_valid"].detach().float().mean(),
            "dense_measured_coverage": history["dense_measured"].detach().float().mean(),
            "dense_estimated_coverage": history["dense_estimated"].detach().float().mean(),
            "dense_enabled_fraction": sample_gate.detach().float().mean(),
        }
        return feature + residual + anchor


def _group_count(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class AdaptiveHistoryFusion(nn.Module):
    """Fuse dense history with the current decoder state and UNet timestep emb."""

    def __init__(self, channels, emb_channels, hidden_dim=64, fusion_dim=128):
        super().__init__()
        self.current_norm = nn.GroupNorm(_group_count(channels), channels)
        self.current_proj = nn.Conv2d(channels, fusion_dim, 1)
        self.history_proj = nn.Conv2d(hidden_dim, fusion_dim, 1)
        self.joint = nn.Conv2d(fusion_dim * 2, fusion_dim, 3, padding=1)
        self.emb_proj = nn.Linear(emb_channels, fusion_dim)
        self.refine = nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1)
        self.output = nn.Conv2d(fusion_dim, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature, history_features, temb):
        if temb is None:
            raise ValueError("static_adaptive history requires the UNet timestep embedding")
        if temb.ndim != 2 or temb.shape[0] != feature.shape[0]:
            raise ValueError("timestep embedding must be [B,C]")
        current = self.current_proj(self.current_norm(feature.to(self.current_norm.weight.dtype)))
        history = self.history_proj(history_features)
        fused = self.joint(torch.cat([current, history], dim=1))
        emb = self.emb_proj(temb.to(self.emb_proj.weight.dtype)).to(fused.dtype)
        while emb.ndim < fused.ndim:
            emb = emb[..., None]
        fused = F.silu(fused + emb)
        fused = F.silu(self.refine(fused))
        return self.output(fused)


class AdaptiveStaticHistoryAdapter(DenseStaticHistoryAdapter):
    """State-dependent dense static history adapter for A1 experiments."""

    def __init__(self, channels, emb_channels, hidden_dim=64, input_variant="types"):
        super().__init__(channels, hidden_dim=hidden_dim, input_variant=input_variant)
        self.mode = "static_adaptive"
        self.fusion_dim = 128
        self.time_embed_dim = int(emb_channels)
        del self.output
        self.fusion = AdaptiveHistoryFusion(
            channels, self.time_embed_dim, hidden_dim=hidden_dim, fusion_dim=self.fusion_dim)

    def forward(self, feature, context=None, history=None, temb=None):
        anchor = self._anchor() if self.training else 0.0
        if history is None:
            self.last_metrics = {}
            return feature + anchor
        b, _, h, w = feature.shape
        validate_history(self._dense_history(history), b)
        if not DENSE_HISTORY_FIELDS <= set(history):
            raise ValueError("static_adaptive mode requires dense RGB and masks")
        if history["latent"].shape[2:] != (h, w):
            raise ValueError("history must use the decoder's latent resolution")
        if history["latent"].device != feature.device:
            raise ValueError("history and current features must be on the same device")
        encoded = self._encoded_history(history, feature)
        sample_gate = self._sample_gate(history, feature)
        residual = self.fusion(feature, encoded, temb) * sample_gate
        self.last_metrics = {
            "history_residual_rms": residual.detach().float().square().mean().sqrt(),
            "dense_valid_coverage": history["dense_valid"].detach().float().mean(),
            "dense_measured_coverage": history["dense_measured"].detach().float().mean(),
            "dense_estimated_coverage": history["dense_estimated"].detach().float().mean(),
            "dense_enabled_fraction": sample_gate.detach().float().mean(),
        }
        return feature + residual + anchor


class CenteredStaticHistoryAdapter(AdaptiveStaticHistoryAdapter):
    """Adaptive dense history adapter with current/time-only bias cancellation."""

    def __init__(self, channels, emb_channels, hidden_dim=64, input_variant="types"):
        super().__init__(channels, emb_channels, hidden_dim=hidden_dim, input_variant=input_variant)
        self.mode = "static_centered"

    def forward(self, feature, context=None, history=None, temb=None):
        anchor = self._anchor() if self.training else 0.0
        if history is None:
            self.last_metrics = {}
            return feature + anchor
        b, _, h, w = feature.shape
        validate_history(self._dense_history(history), b)
        if not DENSE_HISTORY_FIELDS <= set(history):
            raise ValueError("static_centered mode requires dense RGB and masks")
        if history["latent"].shape[2:] != (h, w):
            raise ValueError("history must use the decoder's latent resolution")
        if history["latent"].device != feature.device:
            raise ValueError("history and current features must be on the same device")
        encoded = self._encoded_history(history, feature)
        centered = self.fusion(feature, encoded, temb) - self.fusion(feature, torch.zeros_like(encoded), temb)
        sample_gate = self._sample_gate(history, feature)
        residual = centered * sample_gate
        self.last_metrics = {
            "history_residual_rms": residual.detach().float().square().mean().sqrt(),
            "centered_raw_rms": centered.detach().float().square().mean().sqrt(),
            "dense_valid_coverage": history["dense_valid"].detach().float().mean(),
            "dense_measured_coverage": history["dense_measured"].detach().float().mean(),
            "dense_estimated_coverage": history["dense_estimated"].detach().float().mean(),
            "dense_enabled_fraction": sample_gate.detach().float().mean(),
        }
        return feature + residual + anchor
