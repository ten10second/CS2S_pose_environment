"""Frame preparation and single-frame DDIM sampling shared by the live rollouts.

These two helpers were extracted from the removed noise-mode sampler
(`generate_kitti_raea_noise_modes.py`), which carried the temporal v0
correlated-noise / latent-transport strategies and the v1 learned-gate modes.
Those designs are deleted; only the conditioning pack and one DDIM frame remain
here, so the geometry-history rollout (temporal v3) does not depend on a file
named after a removed design.

See docs/temporal_design_map.md for the current temporal design.
"""
import sys
from pathlib import Path

import torch

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for p in (str(TOOLS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


@torch.no_grad()
def prepare_frame_inputs(model, batch):
    # Imported lazily: the sampler below is pure torch, and its CFG contract is
    # unit-tested without the heavy generation dependencies.
    from generate_kitti_raea_samples import make_lidar_geometry_mask_for_sampling

    inputs = model.get_input(batch, "sat_map").cuda()
    outputs = model.get_input(batch, "grd_left_imgs").cuda()
    lidar_cond = model.get_input(batch, model.lidar_condition_key).cuda()
    range_img = model.get_input(batch, "range_img").cuda() if "range_img" in batch else None
    range_mask = model.get_input(batch, "range_mask").cuda() if "range_mask" in batch else None
    lidar_ray_features = batch["lidar_ray_features"].cuda().float() if "lidar_ray_features" in batch else None
    lidar_ray_features_mask = (
        batch["lidar_ray_features_mask"].cuda().float() if "lidar_ray_features_mask" in batch else None
    )
    camera_to_lidar = model.get_input(batch, "camera_to_lidar").squeeze(-1).cuda()
    left_camera_k = model.get_input(batch, "left_camera_k").squeeze(-1).cuda()
    gt_shift_x = batch["gt_shift_x"].cuda()
    gt_shift_y = batch["gt_shift_y"].cuda()
    theta = batch["theta"].cuda()

    inputs = inputs * 2 - 1
    outputs = outputs * 2 - 1
    cond_label = model.make_condition(inputs, batch).detach()
    lidar_context = model.make_lidar_context(
        lidar_cond,
        range_img=range_img,
        range_mask=range_mask,
        camera_to_lidar=camera_to_lidar,
        left_camera_k=left_camera_k,
        lidar_ray_features=lidar_ray_features,
        lidar_ray_features_mask=lidar_ray_features_mask,
    )
    lidar_evidence = model.make_lidar_evidence(lidar_cond)
    lidar_geometry_mask = make_lidar_geometry_mask_for_sampling(model, lidar_evidence)
    target = torch.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)
    return {
        "cond_label": cond_label,
        "lidar_context": lidar_context,
        "lidar_evidence": lidar_evidence,
        "lidar_geometry_mask": lidar_geometry_mask,
        "left_camera_k": left_camera_k,
        "gt_shift_x": gt_shift_x,
        "gt_shift_y": gt_shift_y,
        "theta": theta,
        "range_img": range_img,
        "range_mask": range_mask,
        "camera_to_lidar": camera_to_lidar,
        "target": target,
    }


@torch.no_grad()
def sample_frame(model, sampler, pack, x_T, timesteps, guidance_scale, temperature, uncond_cfg=0.0):
    unconditional_conditioning = None
    if uncond_cfg > 0:
        guidance_scale = uncond_cfg
        unconditional_conditioning = torch.zeros_like(pack["cond_label"])
    samples, _ = sampler.ddim_sampling(
        pack["cond_label"],
        None,
        None,
        None,
        [1, 4, 16, 64],
        x_T=x_T,
        timesteps=timesteps,
        temperature=temperature,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=unconditional_conditioning,
        left_camera_k=pack["left_camera_k"],
        gt_shift_x=pack["gt_shift_x"],
        gt_shift_y=pack["gt_shift_y"],
        theta=pack["theta"],
        range_img=pack["range_img"],
        range_mask=pack["range_mask"],
        camera_to_lidar=pack["camera_to_lidar"],
        lidar_context=pack["lidar_context"],
        lidar_evidence=pack["lidar_evidence"],
        lidar_geometry_mask=pack["lidar_geometry_mask"],
    )
    pred = model.pre_AE_model.decode(samples * (1 / model.scale_factor))
    pred = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0)
    return pred, samples
