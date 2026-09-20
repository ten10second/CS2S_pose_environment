"""One persistent appearance reader; geometry restricts candidates, not RGB values.

Satellite features and 3-D position describe candidate keys. Values always come
from the previous frame. Coarse unaligned appearance tokens provide a fallback
for unknown object motion; they are not claimed to establish correspondence.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


STATIC_HISTORY_FIELDS = {"static_rgb", "static_mask", "static_confidence"}
DENSE_HISTORY_FIELDS = {"dense_rgb", "dense_valid", "dense_measured", "dense_estimated"}
GEOMETRY_HISTORY_FIELDS = {"history_grid", "sat_grid", "valid", "sat_valid", "positions"}
HISTORY_FIELDS = {"latent", "enabled", "dense_features"} | GEOMETRY_HISTORY_FIELDS | STATIC_HISTORY_FIELDS | DENSE_HISTORY_FIELDS


def validate_history(history, batch_size=None):
    if history is None:
        return
    if not isinstance(history, dict) or "latent" not in history:
        raise ValueError("history must contain a previous-frame latent")
    unknown = set(history) - HISTORY_FIELDS
    if unknown:
        raise ValueError("unsupported history fields: " + ", ".join(sorted(unknown)))
    z = history["latent"]
    if not torch.is_tensor(z) or z.ndim != 4 or min(z.shape) < 1:
        raise ValueError("history latent must be [B,C,H,W]")
    if batch_size is not None and z.shape[0] != batch_size:
        raise ValueError("history requires a separate latent for each batch sample")
    if "enabled" in history:
        flag = history["enabled"]
        if not torch.is_tensor(flag) or flag.shape != (z.shape[0],) or flag.dtype != torch.bool:
            raise ValueError("history enabled must be a bool [B] tensor")
    if STATIC_HISTORY_FIELDS & set(history):
        if not STATIC_HISTORY_FIELDS <= set(history):
            raise ValueError("static history requires RGB, mask, and confidence together")
        if (GEOMETRY_HISTORY_FIELDS | DENSE_HISTORY_FIELDS) & set(history):
            raise ValueError("static history cannot mix geometry or dense fields")
        if z.shape[1] != 4 or not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("static history latent must be finite floating point [B,4,h,w]")
        rgb = history["static_rgb"]
        mask = history["static_mask"]
        confidence = history["static_confidence"]
        if (not torch.is_tensor(rgb) or rgb.ndim != 4 or rgb.shape[:2] != (z.shape[0], 3)
                or min(rgb.shape[2:]) < 1 or not rgb.is_floating_point()):
            raise ValueError("static_rgb must be floating point [B,3,H,W]")
        shape = (z.shape[0], 1, *rgb.shape[2:])
        if not torch.is_tensor(mask) or mask.shape != shape or mask.dtype != torch.bool:
            raise ValueError("static_mask must be bool [B,1,H,W]")
        if (not torch.is_tensor(confidence) or confidence.shape != shape
                or not confidence.is_floating_point()):
            raise ValueError("static_confidence must be floating point [B,1,H,W]")
        for name in STATIC_HISTORY_FIELDS | {"enabled"}:
            if name in history and history[name].device != z.device:
                raise ValueError("static history tensors must share a device")
        for name, value in (("static_rgb", rgb), ("static_confidence", confidence)):
            if not torch.isfinite(value).all() or torch.any(value < 0) or torch.any(value > 1):
                raise ValueError(name + " must be finite and within [0,1]")
    if DENSE_HISTORY_FIELDS & set(history):
        if not DENSE_HISTORY_FIELDS <= set(history):
            raise ValueError("dense static history requires RGB, valid, measured, and estimated together")
        if (GEOMETRY_HISTORY_FIELDS | STATIC_HISTORY_FIELDS) & set(history):
            raise ValueError("dense static history cannot mix geometry or legacy static fields")
        if z.shape[1] != 4 or not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError("dense static history latent must be finite floating point [B,4,h,w]")
        rgb = history["dense_rgb"]
        if (not torch.is_tensor(rgb) or rgb.ndim != 4 or rgb.shape[:2] != (z.shape[0], 3)
                or min(rgb.shape[2:]) < 1 or not rgb.is_floating_point()):
            raise ValueError("dense_rgb must be floating point [B,3,H,W]")
        shape = (z.shape[0], 1, *rgb.shape[2:])
        for name in ("dense_valid", "dense_measured", "dense_estimated"):
            value = history[name]
            if not torch.is_tensor(value) or value.shape != shape or value.dtype != torch.bool:
                raise ValueError(name + " must be bool [B,1,H,W]")
        for name in DENSE_HISTORY_FIELDS | {"enabled"}:
            if name in history and history[name].device != z.device:
                raise ValueError("dense static history tensors must share a device")
        if not torch.isfinite(rgb).all() or torch.any(rgb < 0) or torch.any(rgb > 1):
            raise ValueError("dense_rgb must be finite and within [0,1]")
        valid = history["dense_valid"]
        measured = history["dense_measured"]
        estimated = history["dense_estimated"]
        if torch.any(measured & estimated):
            raise ValueError("dense measured and estimated masks must be disjoint")
        if not torch.equal(valid, measured | estimated):
            raise ValueError("dense valid mask must equal measured|estimated")
        if torch.any(rgb.masked_select(~valid.expand_as(rgb)) != 0):
            raise ValueError("dense_rgb must be zero outside dense_valid")

    if "dense_features" in history:
        if not DENSE_HISTORY_FIELDS <= set(history):
            raise ValueError("dense_features requires dense RGB and masks")
        features = history["dense_features"]
        if (not torch.is_tensor(features) or features.ndim != 4
                or features.shape[0] != z.shape[0] or min(features.shape) < 1
                or not features.is_floating_point() or not torch.isfinite(features).all()
                or features.device != z.device or features.requires_grad):
            raise ValueError("dense_features must be detached finite floating point [B,C,h,w] on history device")


def repeat_history(history, repeats=2):
    """CFG order is [entire unconditional batch, entire conditional batch]."""
    validate_history(history)
    if history is None:
        return None
    return {key: torch.cat([value] * repeats, dim=0) for key, value in history.items()}


class PersistentHistoryReader(nn.Module):
    def __init__(self, channels, context_dim, latent_channels=4, hidden_dim=64, mode="geometry"):
        super().__init__()
        if mode not in {"geometry", "content"}:
            raise ValueError("history mode must be geometry or content")
        self.mode = mode
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.encoder = nn.Sequential(nn.Conv2d(latent_channels, hidden_dim, 3, padding=1),
                                     nn.SiLU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))
        self.query = nn.Conv2d(channels, hidden_dim, 1)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.sat_key = nn.Linear(context_dim, hidden_dim, bias=False)
        self.position_key = nn.Linear(4, hidden_dim, bias=False)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Conv2d(hidden_dim, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.last_metrics = {}

    def _anchor(self):
        # Register every parameter even on an empty-history rank. Zero output
        # initialization makes Q/K/V gradients zero initially, not absent.
        return sum(p.reshape(-1)[0] * 0.0 for p in self.parameters())

    @staticmethod
    def _sample(feature, grid):
        b, h, w, k, _ = grid.shape
        sampled = F.grid_sample(feature, grid.reshape(b, h, w * k, 2).to(feature.dtype),
                                mode="bilinear", padding_mode="zeros", align_corners=False)
        return sampled.reshape(b, feature.shape[1], h * w, k).permute(0, 2, 3, 1)

    def forward(self, feature, context, history=None):
        anchor = self._anchor() if self.training else 0.0
        if history is None:
            self.last_metrics = {}
            return feature + anchor
        b, _, h, w = feature.shape
        validate_history(history, b)
        previous = history["latent"]
        if previous.shape[2:] != (h, w):
            raise ValueError("history must use the decoder's latent resolution")
        if previous.device != feature.device:
            raise ValueError("history and current features must be on the same device")
        previous_features = self.encoder(previous.detach().to(feature.dtype))
        query = self.query_norm(self.query(feature).flatten(2).transpose(1, 2)).float()

        if self.mode == "content":
            tokens = previous_features.flatten(2).transpose(1, 2)
            keys = self.key_norm(self.key(tokens)).float()
            values = self.value(tokens).float()
            logits = torch.matmul(query, keys.transpose(1, 2)) / math.sqrt(self.hidden_dim)
            logits = logits - math.log(tokens.shape[1])
            # A fixed zero-value option permits the reader to abstain.
            logits = torch.cat([logits, torch.zeros_like(logits[..., :1])], -1)
            weights = logits.softmax(-1)
            read = torch.matmul(weights[..., :-1], values)
        else:
            required = {"history_grid", "sat_grid", "valid", "sat_valid", "positions"}
            if not required <= set(history):
                raise ValueError("geometry history is missing candidate fields")
            grid = history["history_grid"]
            if grid.ndim != 5 or grid.shape[:3] != (b, h, w) or grid.shape[-1] != 2 or grid.shape[3] < 1:
                raise ValueError("history_grid must be [B,H,W,K,2]")
            k = grid.shape[3]
            if history["sat_grid"].shape != grid.shape:
                raise ValueError("sat and history grids must have identical candidate shapes")
            for name in ("valid", "sat_valid"):
                if history[name].shape != (b, h, w, k) or history[name].dtype != torch.bool:
                    raise ValueError(name + " must be bool [B,H,W,K]")
            if history["positions"].shape != (b, h, w, k, 4):
                raise ValueError("positions must be [B,H,W,K,4]")
            tokens = self._sample(previous_features, grid)
            if isinstance(context, (tuple, list)) and len(context) == 1:
                context = context[0]
            if context is None:
                satellite = tokens.new_zeros(b, h * w, k, self.context_dim)
            else:
                if context.ndim != 3 or context.shape[0] != b or context.shape[2] != self.context_dim:
                    raise ValueError("sat context must be [B,N,context_dim]")
                side = int(math.sqrt(context.shape[1]))
                if side * side != context.shape[1]:
                    raise ValueError("sat context must be a square token grid without a CLS token")
                sat_map = context.transpose(1, 2).reshape(b, self.context_dim, side, side)
                satellite = self._sample(sat_map, history["sat_grid"])
                satellite = satellite * history["sat_valid"].reshape(b, h * w, k, 1)
            keys = (self.key(tokens) + self.sat_key(satellite.to(tokens.dtype))
                    + self.position_key(history["positions"].reshape(b, h * w, k, 4).to(tokens.dtype)))
            keys = self.key_norm(keys).float()
            values = self.value(tokens).float()
            local_logits = (query.unsqueeze(2) * keys).sum(-1) / math.sqrt(self.hidden_dim)
            local_logits = local_logits.masked_fill(~history["valid"].reshape(b, h * w, k), float("-inf"))

            pooled = F.adaptive_avg_pool2d(previous_features, (min(4, h), min(8, w)))
            fallback = pooled.flatten(2).transpose(1, 2)
            fallback_keys = self.key_norm(self.key(fallback)).float()
            fallback_values = self.value(fallback).float()
            fallback_logits = torch.matmul(query, fallback_keys.transpose(1, 2)) / math.sqrt(self.hidden_dim)
            # Equal prior mass for geometry vs. content fallback; increasing
            # candidate count must not silently increase the branch's prior.
            local_count = history["valid"].reshape(b, h * w, k).sum(-1, keepdim=True).clamp_min(1)
            local_logits = local_logits - local_count.float().log()
            fallback_logits = fallback_logits - math.log(fallback.shape[1])
            logits = torch.cat([local_logits, fallback_logits, torch.zeros_like(local_logits[..., :1])], -1)
            weights = logits.softmax(-1)
            read = (weights[..., :k].unsqueeze(-1) * values).sum(2)
            read = read + torch.matmul(weights[..., k:-1], fallback_values)
        enabled = history.get("enabled", torch.ones(b, device=feature.device, dtype=torch.bool))
        residual = self.output(read.transpose(1, 2).reshape(b, self.hidden_dim, h, w).to(feature.dtype))
        residual = residual * enabled[:, None, None, None]
        self.last_metrics = {"history_residual_rms": residual.detach().float().square().mean().sqrt(),
                             "history_abstain_mass": weights[..., -1].detach().mean()}
        return feature + residual + anchor
