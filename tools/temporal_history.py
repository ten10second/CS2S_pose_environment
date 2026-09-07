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


def enable_history_attention(model, heads=8, dim_head=64, history_dim=256):
    """Inject condition-aware history attention into every ray-posterior fusion
    block. Deliberately does NOT touch the v1 temporal gate or the route-C
    satellite attention (P3-02): those stay disabled/uninitialised.

    Returns (hub, encoder, blocks). Trainable params: encoder (incl. null
    token) + per-block attention; freezing the backbone is the caller's job.
    """
    from ldm.modules.KITTI_attention import TemporalEvidenceHub
    from ldm.modules.temporal_history_attention import HistoryCrossAttention, HistoryLatentEncoder

    hub = TemporalEvidenceHub()
    encoder = HistoryLatentEncoder(out_dim=history_dim)
    blocks = []
    for module in model.modules():
        if (
            getattr(module, "ray_fusion_mode", None) == "ray_posterior"
            and getattr(module, "ray_posterior_fusion", None) is not None
        ):
            if module.history_attn is None:
                module.history_attn = HistoryCrossAttention(
                    dim=module.ray_posterior_fusion.dim,
                    history_dim=history_dim,
                    heads=heads,
                    dim_head=dim_head,
                )
            module.temporal_hub = hub
            blocks.append(module)
    if not blocks:
        raise RuntimeError("no ray_posterior fusion blocks found")
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


def build_payload(encoder, tokens, has_history):
    """P1-02: the payload consumed by every history attention block. With no
    history, routes the learned null token through the K/V weights so all
    parameters stay in the graph (P4-04)."""
    if tokens is None:
        tokens = encoder.null_tokens(1)
    return {"history_tokens": tokens, "has_history": bool(has_history)}
