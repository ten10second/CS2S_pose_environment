"""Temporal-history v2 plumbing: HistoryState, payload building, injection.

Frame-boundary contract (P1-03): one frame's ENTIRE DDIM trajectory reads the
same immutable history payload; the payload is built from the PREVIOUS frame's
final latent only after that frame finished generating. First frames /
discontinuous frames / new sequences get has_history=False and fall back to
exact single-frame behaviour.
"""
import sys
from pathlib import Path

import torch

TOOLS_DIR = Path(__file__).resolve().parent
for p in (str(TOOLS_DIR), str(TOOLS_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


class HistoryState:
    """Per-sequence history holder (P1-01)."""

    def __init__(self, sequence_id, frame_index, latent):
        self.sequence_id = sequence_id
        self.frame_index = int(frame_index)
        self.latent = latent  # final z0, already multiplied by scale_factor


def should_use_history(prev_state, cur_sequence_id, cur_frame_index):
    """P1-04: history only for strictly consecutive frames in the same
    sequence; anything else resets to single-frame behaviour."""
    if prev_state is None:
        return False
    if prev_state.sequence_id != cur_sequence_id:
        return False
    return int(cur_frame_index) == prev_state.frame_index + 1


def enable_history_attention(
    model,
    heads=None,
    dim_head=None,
    history_dim=None,
    geometry=False,
    block_indices=None,
):
    """Inject condition-aware history attention into every ray-posterior fusion
    block. Deliberately does NOT touch the v1 temporal gate or the route-C
    satellite attention (P3-02): those stay disabled/uninitialised.

    Returns (hub, encoder, blocks). Trainable params: encoder (incl. null
    token) + per-block attention; freezing the backbone is the caller's job.
    """
    from ldm.modules.KITTI_attention import TemporalEvidenceHub
    from ldm.modules.temporal_history_attention import (
        GeometryHistoryAttention,
        HistoryCrossAttention,
        HistoryLatentEncoder,
    )

    hub = TemporalEvidenceHub()
    if geometry:
        history_dim = 64 if history_dim is None else history_dim
        heads = 4 if heads is None else heads
        dim_head = 32 if dim_head is None else dim_head
    else:
        history_dim = 256 if history_dim is None else history_dim
        heads = 8 if heads is None else heads
        dim_head = 64 if dim_head is None else dim_head
    encoder = HistoryLatentEncoder(hidden=64 if geometry else 256, out_dim=history_dim)
    candidates = []
    for module in model.modules():
        if (
            getattr(module, "ray_fusion_mode", None) == "ray_posterior"
            and getattr(module, "ray_posterior_fusion", None) is not None
        ):
            candidates.append(module)
    if not candidates:
        raise RuntimeError("no ray_posterior fusion blocks found")
    if block_indices is None:
        selected_indices = list(range(len(candidates)))
    else:
        selected_indices = [int(i) for i in block_indices]
        if not selected_indices:
            raise ValueError("history block_indices must not be empty")
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError(f"history block_indices contain duplicates: {selected_indices}")
        bad = [i for i in selected_indices if i < 0 or i >= len(candidates)]
        if bad:
            raise ValueError(
                f"history block_indices out of range: {bad}; available 0..{len(candidates) - 1}"
            )

    blocks = []
    for history_index, module_index in enumerate(selected_indices):
        module = candidates[module_index]
        expected_geometry = bool(getattr(module.history_attn, "uses_geometry", False))
        if module.history_attn is not None and expected_geometry != bool(geometry):
            raise RuntimeError(
                f"history block {module_index} already has incompatible history attention"
            )
        if geometry:
            if module.history_attn is None:
                module.history_attn = GeometryHistoryAttention(
                    dim=module.ray_posterior_fusion.dim,
                    history_dim=history_dim,
                    heads=heads,
                    dim_head=dim_head,
                )
        else:
            if module.history_attn is None:
                module.history_attn = HistoryCrossAttention(
                    dim=module.ray_posterior_fusion.dim,
                    history_dim=history_dim,
                    heads=heads,
                    dim_head=dim_head,
                )
        module.temporal_hub = hub
        module.history_attn_index = history_index
        module.history_block_index = module_index
        blocks.append(module)
    return hub, encoder, blocks


def history_trainable_parameters(encoder, blocks):
    params = list(encoder.parameters())
    for block in blocks:
        params.extend(block.history_attn.parameters())
    return params


@torch.no_grad()
def history_latent_from_gt(model, prev_batch):
    """Encode previous GT with the frozen VAE and return its scaled latent.

    The trainable history encoder deliberately does not run here: it must be
    called inside the DDP-wrapped training module so its gradients are reduced
    across ranks.
    """
    outputs = prev_batch["grd_left_imgs"] * 2 - 1
    z = model.pre_AE_model.encode(outputs).sample() * model.scale_factor
    return z


def build_payload(encoder, tokens, has_history, history_grid=None, history_valid=None):
    """P1-02: the payload consumed by every history attention block. With no
    history, routes the learned null token through the K/V weights so all
    parameters stay in the graph (P4-04)."""
    if tokens is None:
        tokens = encoder.null_tokens(1)
    payload = {
        "history_tokens": tokens,
        "history_hw": tuple(encoder.grid),
        "has_history": bool(has_history),
    }
    if history_grid is not None or history_valid is not None:
        if history_grid is None or history_valid is None:
            raise ValueError("history_grid and history_valid must be provided together")
        if history_grid.dim() != 4 or history_grid.shape[-1] != 2:
            raise ValueError("history_grid must have shape (B,H,W,2)")
        if history_valid.shape != history_grid.shape[:3]:
            raise ValueError("history_valid must have shape (B,H,W)")
        payload["history_grid"] = history_grid
        payload["history_valid"] = history_valid.bool()
    return payload
