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


AFTER_BOTTLENECK = "after_bottleneck"


def _is_ray_posterior_block(module):
    return (
        getattr(module, "ray_fusion_mode", None) == "ray_posterior"
        and getattr(module, "ray_posterior_fusion", None) is not None
    )


def collect_ray_posterior_blocks(module):
    return [item for item in module.modules() if _is_ray_posterior_block(item)]


def find_unet_with_bottleneck(root):
    for module in root.modules():
        if (
            hasattr(module, "input_blocks")
            and hasattr(module, "middle_block")
            and hasattr(module, "output_blocks")
            and hasattr(module, "lidar_bottleneck_depth_head")
        ):
            return module
    return None


def fusion_block_stages(root):
    """Ordered ray-posterior blocks plus encoder/middle/decoder/unknown stages.

    Bottleneck depth is predicted from middle_block output. Encoder and middle
    fusion therefore inject before that head; decoder fusion injects after it.
    """
    unet = find_unet_with_bottleneck(root)
    all_blocks = collect_ray_posterior_blocks(root)
    if unet is None:
        return all_blocks, ["unknown"] * len(all_blocks)
    encoder = collect_ray_posterior_blocks(unet.input_blocks)
    middle = collect_ray_posterior_blocks(unet.middle_block)
    decoder = collect_ray_posterior_blocks(unet.output_blocks)
    partitioned = encoder + middle + decoder
    if partitioned != all_blocks:
        raise RuntimeError(
            "ray-posterior fusion blocks are not confined to UNet "
            "input_blocks/middle_block/output_blocks"
        )
    stages = (
        ["encoder"] * len(encoder)
        + ["middle"] * len(middle)
        + ["decoder"] * len(decoder)
    )
    return all_blocks, stages


def parse_history_block_indices(text):
    if text is None:
        return None
    raw = str(text).strip()
    if raw in {AFTER_BOTTLENECK, "post_bottleneck"}:
        return AFTER_BOTTLENECK
    parts = [part for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("history block_indices must not be empty")
    selected = [int(part) for part in parts]
    if len(set(selected)) != len(selected):
        raise ValueError(f"history block_indices contain duplicates: {selected}")
    return tuple(selected)


def resolve_history_block_indices(
    root,
    block_indices=None,
    geometry=False,
):
    """Resolve fusion indices. Geometry defaults to the finest 640-d decoder block."""
    candidates, stages = fusion_block_stages(root)
    if not candidates:
        raise RuntimeError("no ray_posterior fusion blocks found")
    spec = block_indices
    if spec is None:
        spec = AFTER_BOTTLENECK if geometry else list(range(len(candidates)))
    if spec == AFTER_BOTTLENECK:
        decoder_ids = [index for index, stage in enumerate(stages) if stage == "decoder"]
        if not decoder_ids:
            raise ValueError(
                "after_bottleneck history requires decoder ray-posterior fusion blocks"
            )
        dim640 = [
            index
            for index in decoder_ids
            if int(candidates[index].ray_posterior_fusion.dim) == 640
        ]
        selected = [dim640[-1] if dim640 else decoder_ids[-1]]
        return candidates, selected, stages, AFTER_BOTTLENECK
    selected = [int(index) for index in spec]
    if not selected:
        raise ValueError("history block_indices must not be empty")
    if len(set(selected)) != len(selected):
        raise ValueError(f"history block_indices contain duplicates: {selected}")
    bad = [index for index in selected if index < 0 or index >= len(candidates)]
    if bad:
        raise ValueError(
            f"history block_indices out of range: {bad}; available 0..{len(candidates) - 1}"
        )
    if geometry:
        pre = [index for index in selected if stages[index] in {"encoder", "middle"}]
        if pre:
            raise ValueError(
                "geometry history refuses pre-bottleneck fusion blocks "
                f"{pre} (stages {[stages[index] for index in pre]}). These inject "
                "before the bottleneck depth head."
            )
    return candidates, selected, stages, tuple(selected)


def enable_history_attention(
    model,
    heads=None,
    dim_head=None,
    history_dim=None,
    geometry=False,
    block_indices=None,
):
    """Inject condition-aware history attention into selected fusion blocks.

    Deliberately does NOT touch the v1 temporal gate or the route-C satellite
    attention (P3-02). Geometry history defaults to after-bottleneck decoder
    fusion so the shared encoder is not a depth-feature channel.

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
    candidates, selected_indices, stages, _spec = resolve_history_block_indices(
        model,
        block_indices=block_indices,
        geometry=geometry,
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
        module.history_block_stage = stages[module_index]
        blocks.append(module)
    return hub, encoder, blocks


def history_host_parameters(blocks):
    """Unfreeze the feed-forward tail after history is added so generation can use it."""
    params = []
    for block in blocks:
        if hasattr(block, "ff"):
            params.extend(block.ff.parameters())
        if hasattr(block, "norm3"):
            params.extend(block.norm3.parameters())
    return params


def history_trainable_parameters(encoder, blocks, unfreeze_host=False):
    params = list(encoder.parameters())
    for block in blocks:
        params.extend(block.history_attn.parameters())
    if unfreeze_host:
        params.extend(history_host_parameters(blocks))
    return params


def history_host_state_dict(blocks):
    payload = {}
    for index, block in enumerate(blocks):
        if not hasattr(block, "ff") or not hasattr(block, "norm3"):
            raise AttributeError(
                f"history host block {index} is missing ff/norm3; "
                "refusing to save an incomplete generator tail"
            )
        payload[str(index)] = {
            "ff": {key: value.detach().clone() for key, value in block.ff.state_dict().items()},
            "norm3": {key: value.detach().clone() for key, value in block.norm3.state_dict().items()},
        }
    return payload


def load_history_host_state_dict(blocks, host):
    if not host:
        raise KeyError("history checkpoint missing history_host")
    expected = {str(index) for index in range(len(blocks))}
    actual = set(host)
    if actual != expected:
        raise ValueError(
            f"history_host keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    for index, block in enumerate(blocks):
        state = host[str(index)]
        if "ff" not in state or "norm3" not in state:
            raise KeyError(f"history_host[{index}] missing ff or norm3")
        if not hasattr(block, "ff") or not hasattr(block, "norm3"):
            raise AttributeError(
                f"history host block {index} is missing ff/norm3; cannot restore generator tail"
            )
        block.ff.load_state_dict(state["ff"], strict=True)
        block.norm3.load_state_dict(state["norm3"], strict=True)


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
