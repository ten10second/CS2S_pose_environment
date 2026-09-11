"""Temporal-history plumbing: injection of the geometry history attention.

Frame-boundary contract: one frame's ENTIRE DDIM trajectory reads the same
immutable history payload; the payload is built from the PREVIOUS frame's final
latent only after that frame finished generating. First frames / discontinuous
frames / new sequences get has_history=False and fall back to exact single-frame
behaviour.

See docs/temporal_design_map.md for the current temporal design.
"""
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
for p in (str(TOOLS_DIR), str(TOOLS_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


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
):
    """Resolve fusion indices. Defaults to the finest 640-d decoder block."""
    candidates, stages = fusion_block_stages(root)
    if not candidates:
        raise RuntimeError("no ray_posterior fusion blocks found")
    spec = block_indices
    if spec is None:
        spec = AFTER_BOTTLENECK
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
    pre = [index for index in selected if stages[index] in {"encoder", "middle"}]
    if pre:
        raise ValueError(
            "history refuses pre-bottleneck fusion blocks "
            f"{pre} (stages {[stages[index] for index in pre]}). These inject "
            "before the bottleneck depth head."
        )
    return candidates, selected, stages, tuple(selected)


def enable_history_attention(
    model,
    heads=None,
    dim_head=None,
    history_dim=None,
    block_indices=None,
):
    """Inject geometry history attention into the selected fusion blocks.

    History defaults to the after-bottleneck decoder fusion block so the shared
    encoder is not a depth-feature channel.

    Returns (hub, encoder, blocks). Trainable params: encoder (incl. null
    token) + per-block attention; freezing the backbone is the caller's job.
    """
    from ldm.modules.KITTI_attention import HistoryEvidenceHub
    from ldm.modules.temporal_history_attention import (
        GeometryHistoryAttention,
        HistoryLatentEncoder,
    )

    hub = HistoryEvidenceHub()
    history_dim = 64 if history_dim is None else history_dim
    heads = 4 if heads is None else heads
    dim_head = 32 if dim_head is None else dim_head
    encoder = HistoryLatentEncoder(hidden=64, out_dim=history_dim)
    candidates, selected_indices, stages, _spec = resolve_history_block_indices(
        model,
        block_indices=block_indices,
    )

    blocks = []
    for history_index, module_index in enumerate(selected_indices):
        module = candidates[module_index]
        if module.history_attn is not None:
            raise RuntimeError(
                f"history block {module_index} already has history attention"
            )
        module.history_attn = GeometryHistoryAttention(
            dim=module.ray_posterior_fusion.dim,
            history_dim=history_dim,
            heads=heads,
            dim_head=dim_head,
        )
        module.history_hub = hub
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

