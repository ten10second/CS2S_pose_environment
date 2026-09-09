"""Shared sequence helpers (kept for the geometry-history sampler)."""
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
