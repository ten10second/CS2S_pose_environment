"""Generate the next frame with a fixed aligned history input; no legacy plugins."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
from tools.temporal_data import evaluation_device, load_settings_args, load_base
from ldm.modules.temporal_pair_training import load_history_checkpoint, make_history_condition

def tree_to(value, device):
    if torch.is_tensor(value): return value.detach().to(device)
    if isinstance(value, dict): return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, list): return [tree_to(v, device) for v in value]
    if isinstance(value, tuple): return tuple(tree_to(v, device) for v in value)
    return value


def tensor_hash(value):
    value = value.detach().float().cpu().contiguous()
    h = hashlib.sha256(); h.update(str(tuple(value.shape)).encode()); h.update(str(value.dtype).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def sample_frame(model, kwargs, shape, seed, device, history=None, steps=50, guidance=7.5):
    from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(int(seed)); noise = torch.randn(shape, device=device)
    sampler = KITTI_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor)
    kwargs = tree_to(kwargs, device)
    params = {"conditioning": kwargs.get("context")}
    for key in ["left_camera_k", "gt_shift_x", "gt_shift_y", "theta", "range_img", "range_mask", "camera_to_lidar",
                "lidar_context", "lidar_evidence", "lidar_geometry_mask"]:
        params[key] = kwargs.get(key)
    with torch.no_grad(), (torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()):
        z, info = sampler.sample(S=steps, batch_size=shape[0], shape=list(shape[1:]), x_T=noise, eta=0.0,
                                 verbose=False, unconditional_guidance_scale=guidance, history=tree_to(history, device), **params)
    return z.float(), info, tensor_hash(noise)


def decode(model, z):
    with torch.no_grad(), (torch.cuda.amp.autocast() if z.device.type == "cuda" else nullcontext()):
        return ((model.pre_AE_model.decode(z * (1.0 / model.scale_factor)) + 1) / 2).clamp(0, 1).float()



def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--temporal-checkpoint', default='')
    parser.add_argument('--input', required=True, help='PT: kwargs, warp_rgb, valid, measured, estimated; no target RGB needed')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:4')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--ddim-steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--history-off', action='store_true', help='Disable history for a matched control')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if out.exists(): raise ValueError('refusing overwrite: ' + str(out))
    device = evaluation_device(args.device)
    if not args.temporal_checkpoint and not args.history_off:
        raise ValueError('history generation requires a temporal checkpoint; use --history-off for original base')
    data = torch.load(args.input, map_location='cpu')
    model, base, _ = load_base(load_settings_args(args.settings), device)
    if args.temporal_checkpoint:
        load_history_checkpoint(args.temporal_checkpoint, model, base)
    model.eval().requires_grad_(False)
    history = None
    if not args.history_off:
        history = make_history_condition(model, *(data[k].to(device) for k in ('warp_rgb','valid','measured','estimated')))
        shape = history['latent'].shape
    else:
        shape = tuple(data['shape'])
    z, info, noise_hash = sample_frame(model, data['kwargs'], shape, args.seed, device, history, args.ddim_steps, args.guidance)
    result = {'rgb':decode(model,z).cpu(), 'latent':z.cpu(), 'base_checkpoint':base,
              'args':vars(args), 'noise_hash':noise_hash}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(json.dumps({'output':str(out),'shape':list(z.shape),'history':history is not None}))


if __name__ == '__main__':
    main()
