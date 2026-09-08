"""Pure latent composition for AR with sensor-guided refresh (no model state)."""
import numpy as np
import torch
import torch.nn.functional as F


def consecutive_rows(previous, current):
    """Raw manifest order alone is not a temporal-continuity guarantee."""
    if previous is None or current is None:
        return False
    if previous.get("drive") is None or previous.get("drive") != current.get("drive"):
        return False
    if previous.get("date") != current.get("date"):
        return False
    try:
        return int(current["frame_index"]) == int(previous["frame_index"]) + 1
    except (KeyError, TypeError, ValueError):
        return False


def seed_step_noise(noise_bank, seed):
    """Make the existing DDIM noise bank reproducible across separate processes."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for index, noise in enumerate(noise_bank):
        noise_bank[index] = torch.randn(noise.shape, generator=generator, dtype=noise.dtype)


def compose_history(previous, objects, refresh_mask, fresh_noise, transport=True):
    """Keep raw background, move supported object content, refresh old footprints.

    All masks index the image/latent grid, NOT an ego-warped background grid.
    A transported destination wins over refresh only when its source is in
    bounds and inside the previous object's footprint. Nearer objects win
    overlapping destination cells. Rejected destinations are refreshed.
    """
    if previous.ndim != 4 or previous.shape[0] != 1:
        raise ValueError("AR composition expects a single BCHW latent")
    if fresh_noise.shape != previous.shape:
        raise ValueError("fresh noise must match the previous latent")
    _, _, height, width = previous.shape
    device = previous.device

    def as_mask(value):
        mask = torch.as_tensor(value, device=device, dtype=torch.bool)
        if mask.numel() != height * width:
            raise ValueError("object/refresh mask has the wrong spatial shape")
        return mask.reshape(1, 1, height, width)

    refresh = as_mask(refresh_mask).clone()
    carried = torch.zeros_like(refresh)
    claimed = torch.zeros_like(refresh)
    old_footprints = torch.zeros_like(refresh)
    candidates = torch.zeros_like(refresh)
    result = previous.clone()
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    transported_objects = 0
    # Sort front-to-back; do not allow a farther object to overwrite a nearer one.
    for obj in sorted(objects, key=lambda item: float(item["depth"])):
        destination = as_mask(obj["cell_mask"])
        source_mask = as_mask(obj["prev_cell_mask"])
        candidates |= destination
        old_footprints |= source_mask
        visible_destination = destination & ~claimed
        if np.isfinite(float(obj["depth"])) and float(obj["depth"]) > 1.0:
            # Even an unavailable near object's history occludes farther history.
            claimed |= destination
        displacement = np.asarray(obj["d_lat"], dtype=np.float32)
        if displacement.shape != (2,) or not np.isfinite(displacement).all():
            continue
        if not np.isfinite(float(obj["depth"])) or float(obj["depth"]) <= 1.0:
            continue
        sx, sy = xx - float(displacement[0]), yy - float(displacement[1])
        in_bounds = ((sx >= 0.5) & (sx <= width - 0.5)
                     & (sy >= 0.5) & (sy <= height - 0.5))[None, None]
        grid = torch.stack((2.0 * sx / width - 1.0, 2.0 * sy / height - 1.0), dim=-1)[None]
        coverage = F.grid_sample(source_mask.float(), grid, align_corners=False)
        valid = visible_destination & in_bounds & (coverage >= 1.0 - 1e-6)
        if transport and bool(valid.any()):
            sampled = F.grid_sample(previous.float(), grid, align_corners=False).to(previous.dtype)
            result = torch.where(valid, sampled, result)
            carried |= valid
            transported_objects += 1

    # This final exclusion is essential: current candidate masks also contain
    # the matched objects, so clearing only their initial repaint flag is not enough.
    pre_refresh = refresh | candidates | old_footprints
    fill = pre_refresh & ~carried
    result = torch.where(fill, fresh_noise, result)
    stats = {
        "obj_matched": len(objects),
        "obj_transported": transported_objects,
        "transport_frac": float(carried.float().mean()),
        "fill_frac": float(fill.float().mean()),
        "old_footprint_refresh_frac": float((old_footprints & fill).float().mean()),
        "transport_protected_cells": int((carried & pre_refresh).sum()),
    }
    return result, stats
