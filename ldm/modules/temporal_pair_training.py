"""Shared identities, deterministic diffusion noise and minimal temporal checkpoints."""
from __future__ import annotations
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import numpy as np
import torch
import torch.nn.functional as F

def raw_unet(model: Any) -> torch.nn.Module:
    unet = model.DDPM.denoise_model if hasattr(model, "DDPM") else model
    return unet.module if hasattr(unet, "module") else unet


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(value.shape)).encode())
    h.update(str(value.dtype).encode())
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def base_identity(checkpoint: str | Path, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    identity = {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}
    if payload is not None:
        if "step" in payload:
            identity["step"] = int(payload["step"])
        if "denoise_model" in payload and isinstance(payload["denoise_model"], Mapping):
            identity["denoise_model_sha256"] = state_dict_sha256(payload["denoise_model"])
    return identity


def assert_same_base(expected: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    for key in ("sha256", "denoise_model_sha256", "step"):
        if key in expected or key in observed:
            if expected.get(key) != observed.get(key):
                raise ValueError("base checkpoint identity mismatch for %s" % key)


def seed_training_step(seed: int, device: Optional[torch.device] = None) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def epsilon_prediction_loss(ddpm: Any, x_start: torch.Tensor, cond: Mapping[str, Any], history: Optional[Mapping[str, Any]], seed: int):
    if x_start.ndim != 4 or not torch.isfinite(x_start).all():
        raise ValueError("x_start must be finite NCHW")
    gen = torch.Generator(device=x_start.device).manual_seed(int(seed))
    if hasattr(ddpm, "num_timesteps"):
        num_timesteps = int(ddpm.num_timesteps)
    elif hasattr(ddpm, "timesteps"):
        num_timesteps = int(ddpm.timesteps)
    else:
        raise ValueError("DDPM object must expose num_timesteps or timesteps")
    t = torch.randint(0, num_timesteps, (x_start.shape[0],), generator=gen, device=x_start.device).long()
    noise = torch.randn(x_start.shape, generator=gen, device=x_start.device, dtype=x_start.dtype)
    x_noisy = ddpm.q_sample(x_start=x_start, t=t, noise=noise)
    allowed = {
        "context", "lidar_context", "lidar_evidence", "lidar_geometry_mask",
        "control_grd", "left_camera_k", "gt_shift_x", "gt_shift_y", "theta",
    }
    denoise_cond = {key: value for key, value in dict(cond).items() if key in allowed}
    model_out = ddpm.denoise_model(x_noisy, t, history=history, **denoise_cond)
    loss = F.mse_loss(model_out.float(), noise.float())
    return loss, {"t": t.detach(), "noise": noise.detach(), "model_out": model_out, "x_noisy": x_noisy}


CHECKPOINT_VERSION = 'temporal_concat_rgb_v1'
MASK_ORDER = ['valid', 'measured', 'estimated']


def make_history_condition(model, warp_rgb, valid, measured, estimated):
    """Encode aligned RGB once; only the masks are area averaged."""
    from ldm.modules.temporal_condition import validate_history
    if warp_rgb.ndim != 4 or warp_rgb.shape[1] != 3:
        raise ValueError('warp_rgb must be B3HW')
    if not torch.isfinite(warp_rgb).all() or torch.any((warp_rgb < 0) | (warp_rgb > 1)):
        raise ValueError('warp_rgb must be finite RGB in [0,1]')
    expected = (warp_rgb.shape[0], 1, *warp_rgb.shape[-2:])
    for mask in (valid, measured, estimated):
        if mask.shape != expected or mask.device != warp_rgb.device or mask.dtype != torch.bool:
            raise ValueError('image-space masks must be bool B1HW on RGB device')
    if torch.any(measured & estimated) or not torch.equal(valid, measured | estimated):
        raise ValueError('measured and estimated must partition valid')
    with torch.no_grad():
        rgb = warp_rgb * valid.to(warp_rgb.dtype)
        posterior = model.pre_AE_model.encode(rgb * 2 - 1)
        latent = posterior.mode() if hasattr(posterior, 'mode') else posterior
        latent = latent.detach().float() * float(model.scale_factor)
        masks = F.adaptive_avg_pool2d(torch.cat([valid, measured, estimated], 1).float(), latent.shape[-2:])
        # VAE biases/receptive fields at empty cells are not history evidence.
        latent = latent * (masks[:, :1] > 0).to(latent.dtype)
    history = {'latent': latent, 'masks': masks}
    validate_history(history, warp_rgb.shape[0])
    return history


def configure_trainables(model):
    """Jointly adapt the existing UNet; VAE and sat/LiDAR encoders stay frozen."""
    model.eval().requires_grad_(False)
    unet = raw_unet(model)
    unet.configure_history_input()
    unet.requires_grad_(True).train()
    return [p for p in unet.parameters() if p.requires_grad]


def save_history_checkpoint(path, model, base, step, args):
    """Weights-only artifact; deliberately not advertised as optimizer resume."""
    path = Path(path)
    unet = raw_unet(model)
    if unet.input_blocks[0][0].in_channels != 11:
        raise ValueError('checkpoint requires 4 noisy + 4 history + 3 mask channels')
    payload = dict(version=CHECKPOINT_VERSION, artifact_kind='unet_weights_only',
                   base_checkpoint=dict(base), mask_order=MASK_ORDER, step=int(step), args=dict(args),
                   denoise_model={k:v.detach().cpu() for k,v in unet.state_dict().items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def load_history_checkpoint(path, model, base):
    payload = torch.load(path, map_location='cpu')
    if payload.get('version') != CHECKPOINT_VERSION or payload.get('mask_order') != MASK_ORDER:
        raise ValueError('incompatible history checkpoint: expected temporal_concat_rgb_v1')
    assert_same_base(payload.get('base_checkpoint', {}), base)
    unet = raw_unet(model)
    unet.configure_history_input()
    state = payload.get('denoise_model', {})
    expected = unet.state_dict()
    if set(state) != set(expected):
        raise ValueError('history checkpoint keys differ')
    for key, value in state.items():
        if not torch.is_tensor(value) or value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError('history checkpoint shape/dtype differs: ' + key)
        if not torch.isfinite(value).all():
            raise ValueError('history checkpoint contains nonfinite values: ' + key)
    unet.load_state_dict(state, strict=True)
    return {'step': payload['step'], 'args': payload.get('args', {})}
