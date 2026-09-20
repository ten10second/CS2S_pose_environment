import torch


HISTORY_KEYS = {"latent", "masks"}
MASK_SUM_TOLERANCE = 1e-4


def validate_history(history, batch_size=None):
    if history is None:
        return
    if not isinstance(history, dict):
        raise ValueError("history must be a dict with latent and masks")
    unknown = set(history) - HISTORY_KEYS
    if unknown:
        raise ValueError("unsupported history fields: " + ", ".join(sorted(unknown)))
    if set(history) != HISTORY_KEYS:
        raise ValueError("history must contain exactly latent and masks")

    latent = history["latent"]
    masks = history["masks"]
    if not torch.is_tensor(latent) or latent.ndim != 4 or latent.shape[1] != 4:
        raise ValueError("history latent must be floating point [B,4,H,W]")
    if not latent.is_floating_point() or not torch.isfinite(latent).all():
        raise ValueError("history latent must be finite floating point [B,4,H,W]")
    if not torch.is_tensor(masks) or masks.ndim != 4 or masks.shape[1] != 3:
        raise ValueError("history masks must be floating point [B,3,H,W]")
    if not masks.is_floating_point() or not torch.isfinite(masks).all():
        raise ValueError("history masks must be finite floating point [B,3,H,W]")
    if latent.shape[0] < 1 or min(latent.shape[2:]) < 1:
        raise ValueError("history latent must have non-empty batch and spatial dimensions")
    if masks.shape[0] != latent.shape[0] or min(masks.shape[2:]) < 1:
        raise ValueError("history masks must have non-empty batch and match latent batch")
    if masks.shape[2:] != latent.shape[2:]:
        raise ValueError("history masks must match latent spatial dimensions")
    if batch_size is not None and latent.shape[0] != batch_size:
        raise ValueError("history requires one latent and mask set per batch sample")
    if masks.device != latent.device:
        raise ValueError("history latent and masks must share a device")
    if torch.any(masks < 0) or torch.any(masks > 1):
        raise ValueError("history masks must be finite fractions within [0,1]")

    valid = masks[:, 0]
    measured = masks[:, 1]
    estimated = masks[:, 2]
    if not torch.allclose(measured + estimated, valid, atol=MASK_SUM_TOLERANCE, rtol=0.0):
        raise ValueError("history masks must be ordered valid,measured,estimated with measured+estimated=valid")


def repeat_history(history, repeats=2):
    """CFG order is [entire unconditional batch, entire conditional batch]."""
    validate_history(history)
    if history is None:
        return None
    return {key: torch.cat([value] * repeats, dim=0) for key, value in history.items()}
