import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def _as_float_mask(mask: torch.Tensor, size):
    if mask.dim() == 3:
        mask = mask[:, None]
    mask = mask.float()
    if mask.shape[-2:] != size:
        mask = F.interpolate(mask, size=size, mode="nearest")
    return (mask > 0.5).float()


def masked_mse(fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    mask = _as_float_mask(mask, fake.shape[-2:])
    expanded = mask.expand_as(fake)
    denom = expanded.sum().clamp_min(eps)
    return (((fake - real) ** 2) * expanded).sum() / denom


def masked_psnr(fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8):
    mse = masked_mse(fake, real, mask, eps=eps)
    return -10.0 * torch.log10(mse.clamp_min(eps))


def _crop_to_mask(image: torch.Tensor, mask: torch.Tensor, pad: int = 4):
    ys, xs = torch.where(mask[0] > 0.5)
    if ys.numel() == 0:
        return None, None
    h, w = mask.shape[-2:]
    y0 = max(0, int(ys.min().item()) - pad)
    y1 = min(h, int(ys.max().item()) + pad + 1)
    x0 = max(0, int(xs.min().item()) - pad)
    x1 = min(w, int(xs.max().item()) + pad + 1)
    return image[:, y0:y1, x0:x1], mask[:, y0:y1, x0:x1]


def masked_ssim(fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor):
    mask = _as_float_mask(mask, fake.shape[-2:])
    scores = []
    for idx in range(fake.shape[0]):
        fake_crop, mask_crop = _crop_to_mask(fake[idx], mask[idx])
        if fake_crop is None:
            continue
        real_crop, _ = _crop_to_mask(real[idx], mask[idx])
        if fake_crop.shape[-2] < 11 or fake_crop.shape[-1] < 11:
            fake_crop = F.interpolate(fake_crop[None], size=(max(11, fake_crop.shape[-2]), max(11, fake_crop.shape[-1])), mode="bilinear", align_corners=False)[0]
            real_crop = F.interpolate(real_crop[None], size=fake_crop.shape[-2:], mode="bilinear", align_corners=False)[0]
            mask_crop = F.interpolate(mask_crop[None], size=fake_crop.shape[-2:], mode="nearest")[0]
        scores.append(ssim((fake_crop * mask_crop)[None], (real_crop * mask_crop)[None], data_range=1.0, size_average=True))
    if not scores:
        return torch.tensor(float("nan"), device=fake.device)
    return torch.stack(scores).mean()


def masked_lpips(fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor, lpips_model):
    if lpips_model is None:
        return torch.tensor(float("nan"), device=fake.device)
    mask = _as_float_mask(mask, fake.shape[-2:])
    scores = []
    for idx in range(fake.shape[0]):
        fake_crop, mask_crop = _crop_to_mask(fake[idx], mask[idx])
        if fake_crop is None:
            continue
        real_crop, _ = _crop_to_mask(real[idx], mask[idx])
        if fake_crop.shape[-2] < 64 or fake_crop.shape[-1] < 64:
            size = (max(64, fake_crop.shape[-2]), max(64, fake_crop.shape[-1]))
            fake_crop = F.interpolate(fake_crop[None], size=size, mode="bilinear", align_corners=False)[0]
            real_crop = F.interpolate(real_crop[None], size=size, mode="bilinear", align_corners=False)[0]
            mask_crop = F.interpolate(mask_crop[None], size=size, mode="nearest")[0]
        fake_input = (fake_crop * mask_crop)[None] * 2.0 - 1.0
        real_input = (real_crop * mask_crop)[None] * 2.0 - 1.0
        scores.append(lpips_model(real_input, fake_input).mean())
    if not scores:
        return torch.tensor(float("nan"), device=fake.device)
    return torch.stack(scores).mean()


def dynamic_masked_metrics(fake: torch.Tensor, real: torch.Tensor, mask: torch.Tensor, lpips_model=None):
    mask = _as_float_mask(mask, fake.shape[-2:])
    coverage = mask.flatten(1).mean(dim=1)
    valid = coverage > 0
    return {
        "dynamic_psnr": masked_psnr(fake, real, mask),
        "dynamic_ssim": masked_ssim(fake, real, mask),
        "dynamic_lpips": masked_lpips(fake, real, mask, lpips_model),
        "dynamic_mask_coverage": coverage.mean(),
        "dynamic_valid_images": valid.sum(),
    }
