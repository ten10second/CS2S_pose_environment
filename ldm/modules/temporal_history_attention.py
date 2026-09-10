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
import torch.nn.functional as F
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


class GeometryHistoryAttention(nn.Module):
    """Appearance memory: previous RGB latent as K/V, current features as Q.

    Geometry only gates and biases readout. Invalid cells and has_history=False
    stay exact zero. Local 3x3 copy and bilinear skip were removed after they
    failed to inherit appearance on generated video.
    """

    uses_geometry = True

    def __init__(
        self,
        dim,
        history_dim=64,
        heads=4,
        dim_head=32,
        memory_sigma=1.0,
        **_unused,
    ):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.inner = inner
        self.scale = dim_head ** -0.5
        self.memory_sigma = float(memory_sigma)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_cond = nn.LayerNorm(dim)
        self.norm_hist = nn.LayerNorm(history_dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_q_cond = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(history_dim, inner, bias=False)
        self.to_v = nn.Linear(history_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        nn.init.normal_(self.to_out.weight, std=0.02)
        self.last_null_frac = None
        self.last_valid_frac = None
        self.last_ratio = None
        self.last_memory_ratio = None

    @staticmethod
    def _repeat_to_batch(tensor, batch, name):
        if tensor.shape[0] == batch:
            return tensor
        if batch % tensor.shape[0] != 0:
            raise ValueError(f"{name} batch {tensor.shape[0]} cannot repeat to query batch {batch}")
        repeat = batch // tensor.shape[0]
        return tensor.repeat((repeat,) + (1,) * (tensor.dim() - 1))

    @staticmethod
    def _identity_grid(batch, height, width, device, dtype):
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    @classmethod
    def _resize_grid(cls, history_grid, history_valid, query_hw):
        qh, qw = int(query_hw[0]), int(query_hw[1])
        if history_grid.shape[1:3] == (qh, qw):
            grid = history_grid
            valid = history_valid
        else:
            b, hh, hw, _ = history_grid.shape
            src_centers = cls._identity_grid(
                b, hh, hw, history_grid.device, history_grid.dtype
            )
            dst_centers = cls._identity_grid(
                b, qh, qw, history_grid.device, history_grid.dtype
            )
            displacement = history_grid - src_centers
            displacement = F.interpolate(
                displacement.permute(0, 3, 1, 2).float(),
                size=(qh, qw),
                mode="nearest",
            ).permute(0, 2, 3, 1).to(history_grid.dtype)
            grid = dst_centers + displacement
            valid_coverage = F.interpolate(
                history_valid[:, None].float(),
                size=(qh, qw),
                mode="area",
            )[:, 0]
            valid = valid_coverage >= 1.0
        return grid, valid

    def _null_only(self, x, cond_summary, history_tokens, has_history):
        b, n, _dim = x.shape
        q = self.to_q(self.norm_q(x)) + self.to_q_cond(self.norm_cond(cond_summary))
        k = self.to_k(self.norm_hist(history_tokens))
        v = self.to_v(self.norm_hist(history_tokens))
        zero_dep = (q.mean() + k.mean() + v.mean() + self.to_out.weight.mean()) * 0.0
        out = self.to_out(x.new_zeros((b, n, self.inner)) + zero_dep)
        if not has_history:
            out = out * torch.zeros((), device=out.device, dtype=out.dtype)
        with torch.no_grad():
            self.last_null_frac = 1.0
            self.last_valid_frac = 0.0
            self.last_ratio = 0.0
            self.last_memory_ratio = 0.0
        return out

    def _memory_readout(self, x, cond_summary, history_tokens, grid, valid, height, width):
        batch, queries, _dim = x.shape
        hist = self.norm_hist(history_tokens)
        q = self.to_q(self.norm_q(x)) + self.to_q_cond(self.norm_cond(cond_summary))
        k = self.to_k(hist)
        v = self.to_v(hist)
        qh = q.view(batch, queries, self.heads, self.dim_head).permute(0, 2, 1, 3)
        kh = k.view(batch, height * width, self.heads, self.dim_head).permute(0, 2, 1, 3)
        vh = v.view(batch, height * width, self.heads, self.dim_head).permute(0, 2, 1, 3)
        scores = (qh @ kh.transpose(-1, -2)) * self.scale
        key_pos = self._identity_grid(
            batch, height, width, grid.device, grid.dtype
        ).reshape(batch, 1, height * width, 2)
        query_xy = grid.reshape(batch, queries, 2)
        valid_flat = valid.reshape(valid.shape[0], -1)
        if valid_flat.shape[0] != batch or valid_flat.shape[1] != queries:
            raise ValueError(
                f"history valid shape {tuple(valid.shape)} does not match query batch {batch}x{queries}"
            )
        finite = torch.isfinite(query_xy).all(dim=-1)
        cell_valid = valid_flat.bool() & finite
        query_xy = torch.where(cell_valid.unsqueeze(-1), query_xy, torch.zeros_like(query_xy))
        dist2 = ((query_xy.unsqueeze(2) - key_pos) ** 2).sum(dim=-1).clamp_min(0.0)
        sigma = max(self.memory_sigma, 1e-3)
        scores = scores + (-0.5 * dist2 / (sigma * sigma)).unsqueeze(1)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        attn = torch.softmax(scores, dim=-1)
        mem = (attn @ vh).permute(0, 2, 1, 3).reshape(batch, queries, self.inner)
        mem = self.to_out(mem.to(x.dtype))
        mem = torch.where(cell_valid.unsqueeze(-1), mem, torch.zeros_like(mem))
        with torch.no_grad():
            self.last_null_frac = float((~cell_valid).float().mean())
            self.last_valid_frac = float(cell_valid.float().mean())
        return mem

    def forward(
        self,
        x,
        cond_summary,
        history_tokens,
        has_history=True,
        history_grid=None,
        history_valid=None,
        query_hw=None,
        history_hw=None,
    ):
        b, n, _dim = x.shape
        history_tokens = self._repeat_to_batch(history_tokens, b, "history_tokens")
        if not has_history:
            return self._null_only(x, cond_summary, history_tokens, has_history)
        if history_grid is None or history_valid is None:
            raise ValueError(
                "GeometryHistoryAttention requires history_grid and history_valid when has_history=True"
            )
        history_grid = self._repeat_to_batch(history_grid, b, "history_grid")
        history_valid = self._repeat_to_batch(history_valid, b, "history_valid").bool()

        if history_hw is None:
            side = int(history_tokens.shape[1] ** 0.5)
            if side * side != history_tokens.shape[1]:
                raise ValueError("history_hw is required when history tokens are not square")
            history_hw = (side, side)
        hh, hw = int(history_hw[0]), int(history_hw[1])
        if history_tokens.shape[1] != hh * hw:
            raise ValueError(
                f"history token count {history_tokens.shape[1]} != history_hw {hh}x{hw}"
            )
        if query_hw is None:
            query_hw = history_grid.shape[1:3]
        qh, qw = int(query_hw[0]), int(query_hw[1])
        if n != qh * qw:
            raise ValueError(f"query tokens {n} != query_hw {qh}x{qw}")

        grid, valid = self._resize_grid(history_grid, history_valid, (qh, qw))
        out = self._memory_readout(x, cond_summary, history_tokens, grid, valid, hh, hw)
        with torch.no_grad():
            denom = float(cond_summary.detach().float().norm(dim=-1).mean())
            num = float(out.detach().float().norm(dim=-1).mean())
            self.last_ratio = num / max(denom, 1e-6)
            self.last_memory_ratio = self.last_ratio
        return out
