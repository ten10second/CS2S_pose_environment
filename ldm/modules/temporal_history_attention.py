"""Route-history: condition-aware history readout for temporal consistency v2.

Design contract (docs TODO v2, user's partition principle):
  content already explained by the CURRENT satellite/LiDAR conditions is the
  conditions' job; persistent appearance NOT determined by the conditions
  (vehicle colours, texture choices) is what history contributes.

Components:
  HistoryLatentEncoder — shared, maps the previous frame's final latent
      ẑ[t-1] (b,4,16,64) to history tokens (b, T, C_h) on a fixed grid.
  HistoryCrossAttention — one per fusion block. The QUERY explicitly receives
      the current conditions' fused summary (sat+LiDAR deltas, detached) so
      the attention can learn to read history only for what the conditions do
      not already explain. K/V come from the shared history tokens. A learned
      null key with a fixed zero value lets every query reject history; the
      out projection is zero-initialised, so enabling the stream is exactly
      inert at step 0. has_history=False routes a learned null token through
      the K/V weights (keeps every parameter in the DDP graph) and the output
      is multiplied by an exact 0.0 flag — first frames stay bit-identical to
      the frozen single-frame behaviour.
"""
import torch
from torch import nn

from ldm.modules.diffusionmodules.util import zero_module


class HistoryLatentEncoder(nn.Module):
    """Shared history encoder: ẑ[t-1] (b,4,H,W) → tokens (b, H*W, C_h)."""

    def __init__(self, latent_channels=4, hidden=256, out_dim=256, grid=(16, 64)):
        super().__init__()
        self.grid = tuple(grid)
        self.out_dim = out_dim
        self.in_proj = nn.Conv2d(latent_channels, hidden, 1)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1)
        )
        self.out = nn.Conv2d(hidden, out_dim, 1)
        # learned token used when a sample has no history; keeps K/V weights in
        # the autograd graph on first frames / run starts
        self.null_token = nn.Parameter(torch.randn(1, 1, out_dim) * 0.02)

    def forward(self, z):
        h = self.in_proj(z)
        h = h + self.block1(h)
        h = h + self.block2(h)
        b, c, hh, ww = h.shape
        assert (hh, ww) == self.grid, f"history grid {hh}x{ww} != expected {self.grid}"
        return self.out(h).flatten(2).transpose(1, 2)  # (b, T, C_h)

    def null_tokens(self, batch):
        return self.null_token.expand(batch, 1, self.out_dim)


class HistoryCrossAttention(nn.Module):
    """Per-block history readout with condition-aware query and rejectable null."""

    def __init__(self, dim, history_dim=256, heads=8, dim_head=64):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.inner = inner
        self.scale = dim_head ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_cond = nn.LayerNorm(dim)
        self.norm_hist = nn.LayerNorm(history_dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_q_cond = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(history_dim, inner, bias=False)
        self.to_v = nn.Linear(history_dim, inner, bias=False)
        # reject option: learned key, FIXED zero value — mass on null contributes exactly 0
        self.null_key = nn.Parameter(torch.randn(inner) * 0.02)
        self.register_buffer("null_value", torch.zeros(inner))
        # Bias-free is part of the reject-history contract: if all attention
        # mass is assigned to the fixed zero null value, the projected
        # residual must remain exactly zero after training as well as at init.
        self.to_out = zero_module(nn.Linear(inner, dim, bias=False))
        self.last_null_frac = None
        self.last_ratio = None  # ||hist_delta|| / ||cond_summary||, filled under no_grad

    def forward(self, x, cond_summary, history_tokens, has_history=True):
        """x: (b, N, dim) current latent features; cond_summary: (b, N, dim)
        current sat+LiDAR fused deltas (detached by the caller); history_tokens:
        (b, T, C_h) — may be the encoder's null token when has_history=False."""
        b, n, dim = x.shape
        q = self.to_q(self.norm_q(x)) + self.to_q_cond(self.norm_cond(cond_summary))
        k = self.to_k(self.norm_hist(history_tokens))
        v = self.to_v(self.norm_hist(history_tokens))

        null_key = self.null_key.to(k.dtype)
        k = torch.cat([k, null_key.view(1, 1, -1).expand(b, 1, -1)], dim=1)  # (b, T+1, inner)
        v = torch.cat([v, self.null_value.view(1, 1, -1).expand(b, 1, -1).to(v.dtype)], dim=1)

        qh = q.view(b, n, self.heads, self.dim_head).permute(0, 2, 1, 3)      # (b,h,N,dh)
        kh = k.view(b, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)     # (b,h,T+1,dh)
        vh = v.view(b, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)
        attn = torch.softmax(qh @ kh.transpose(-1, -2) * self.scale, dim=-1)  # (b,h,N,T+1)
        out = (attn @ vh).permute(0, 2, 1, 3).reshape(b, n, self.inner)
        out = self.to_out(out.to(x.dtype))

        if not has_history:
            # first frames stay bit-identical to the frozen single-frame model;
            # multiplying by 0.0 keeps the graph alive so DDP sees every param
            out = out * torch.zeros((), device=out.device, dtype=out.dtype)

        with torch.no_grad():
            self.last_null_frac = float(attn[..., -1].mean())
            denom = float(cond_summary.detach().float().norm(dim=-1).mean())
            num = float(out.detach().float().norm(dim=-1).mean())
            self.last_ratio = num / max(denom, 1e-6)
        return out
