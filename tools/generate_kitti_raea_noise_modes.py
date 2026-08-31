"""Contiguous-sequence sampling with correlated initial-noise strategies.

Quick validation for temporal flicker caused by independent per-frame sampling
in regions unconstrained by LiDAR/satellite conditions (upper facades, tree
crowns). Modes:

  per_frame      fresh x_T per frame (baseline, matches standard inference)
  shared         one x_T shared by the whole sequence; sampling noise fully
                 correlated across frames (per-step noise in ddim_KITTI is
                 already a fixed module-level tensor, so sharing x_T is enough)
  autoregressive first frame from shared x_T at full strength; later frames
                 start from the previous frame's latent noised to an
                 intermediate timestep (SDEdit-style img2img chain)

  instance      segmented transport on top of the warp2 machinery: static
                 cells keep the agreement-gated homography transport, while
                 moving objects associated across scans by LiDAR cluster
                 matching (label-free) inherit content through their OWN
                 image-space displacement instead of being reset; unmatched
                 new objects and disappeared cells still reset to noise

The autoregressive chain calls ddim_sampling directly with the k smallest
ddim timesteps; index alignment with ddim_alphas/ddim_alphas_prev holds
because both are built over the same ascending timestep list.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for p in (str(TOOLS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402

import pose_warp_utils as pwu  # noqa: E402
import lidar_object_association as loa  # noqa: E402

from generate_kitti_raea_samples import (  # noqa: E402
    lidar_key_structure_stats,
    load_checkpoint_into_model,
    make_condition_rgb,
    make_lidar_geometry_mask_for_sampling,
    make_lidar_overlay,
    make_panel,
    sample_to_batch,
    save_tensor_image,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Contiguous-sequence sampling with correlated noise strategies.")
    parser.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml")
    parser.add_argument("--sd-base-ckpt", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--kitti-root", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--lidar-ray-feature-cache-root", required=True)
    parser.add_argument("--image-semantic-cache-root", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--noise-mode",
        choices=["per_frame", "shared", "autoregressive", "warp", "warp2", "instance"],
        default="shared",
    )
    parser.add_argument(
        "--ar-strength",
        type=float,
        default=0.7,
        help="Autoregressive noising strength: fraction of the DDIM schedule skipped; "
        "each frame after the first runs the remaining (1 - strength) smallest timesteps.",
    )
    parser.add_argument("--key-stats-max-tokens", type=int, default=256)
    parser.add_argument(
        "--warp-debug",
        type=int,
        default=0,
        help="Save side-by-side warp alignment debug images for the first N frames.",
    )
    return parser.parse_args()


def rebase_kitti_path(path, kitti_root):
    path = str(path)
    marker = "KITTI_RAW/"
    idx = path.find(marker)
    if idx >= 0 and kitti_root:
        return str(Path(kitti_root) / path[idx + len(marker):])
    return path


def dilate_mask(mask, iterations=1):
    for _ in range(iterations):
        mask = torch.nn.functional.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return mask


def resolve_calib_dir(calib_dir):
    """External-drive manifests nest a date folder inside the calib dir that the
    local copy flattens; find calib_cam_to_cam.txt here or one level up."""
    p = Path(calib_dir)
    if (p / "calib_cam_to_cam.txt").exists():
        return str(p)
    parent = p.parent
    if (parent / "calib_cam_to_cam.txt").exists():
        return str(parent)
    raise FileNotFoundError(f"calib_cam_to_cam.txt not found under {p} or {parent}")


def build_dynamic_mask_latent(geom, points_prev_velo, points_cur_velo, q_prev_in_cur, img_w, img_h, lat_h, lat_w, device):
    """Latent-space drop mask (1 = do not inherit warped content)."""
    status = pwu.consistency_status(q_prev_in_cur, points_cur_velo)
    u, v, depth = geom.project_velo_to_rect_img(q_prev_in_cur)
    u_lat = u * (lat_w / img_w)
    v_lat = v * (lat_h / img_h)
    in_img = (depth > 1.0) & (u_lat >= 0) & (u_lat < lat_w) & (v_lat >= 0) & (v_lat < lat_h)
    mask = torch.zeros((1, 1, lat_h, lat_w), device=device)
    drop = in_img & (status == 2)
    if drop.any():
        ia = np.clip(u_lat[drop].astype(int), 0, lat_w - 1)
        ie = np.clip(v_lat[drop].astype(int), 0, lat_h - 1)
        m = mask.squeeze().cpu().numpy()
        m[ie, ia] = 1.0
        mask = torch.from_numpy(m).to(device).reshape(1, 1, lat_h, lat_w)
        mask = dilate_mask(mask, iterations=1)
    stats = {
        "n_points": int(len(status)),
        "static_frac": float((status == 1).mean()),
        "dynamic_frac": float((status == 2).mean()),
        "mask_frac": float(mask.mean().item()),
    }
    return mask, stats


def ground_plane_rect(geom, points_prev_velo):
    """Ground plane of the previous frame, expressed in prev rectified-cam coords."""
    n_v, d_v = pwu.fit_ground_plane_velo(points_prev_velo)
    if n_v is None:
        return None, None
    A = geom.R0[:3, :3] @ geom.T_cam_velo[:3, :3]
    b = geom.R0[:3, :3] @ geom.T_cam_velo[:3, 3]
    n_r = A @ n_v
    d_r = d_v - float(n_r @ b)
    return n_r, d_r


def save_warp_debug(path, prev_gt, cur_gt, warped_prev_gt):
    """prev_gt/cur_gt/warped: (3,H,W) tensors in [0,1]."""
    from PIL import Image

    def to_img(t):
        arr = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    w, h = to_img(cur_gt).size
    label_h = 20
    canvas = Image.new("RGB", (w * 3, h + label_h), (255, 255, 255))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(
        [("GT prev", to_img(prev_gt)), ("GT cur", to_img(cur_gt)), ("warp(GT prev)", to_img(warped_prev_gt))]
    ):
        canvas.paste(img, (i * w, label_h))
        draw.text((i * w + 4, 4), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def prepare_frame_inputs(model, batch):
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
def sample_frame(model, sampler, pack, x_T, timesteps, guidance_scale, temperature):
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


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt
    cfg.data.params.test.params.manifest = args.manifest
    test_params = cfg.data.params.test.params

    def cfg_value(name, default):
        return getattr(test_params, name, default)

    dataset = SatLidarRawDataset(
        manifest=args.manifest,
        kitti_root=args.kitti_root,
        condition_mode=str(cfg_value("condition_mode", "raw_lidar_pointmap")),
        image_height=128,
        image_width=512,
        sat_size=256,
        max_depth=80.0,
        align_satellite_to_camera=True,
        include_range_image=bool(cfg_value("include_range_image", False)),
        include_raw_lidar_points=False,
        lidar_ray_feature_cache_root=args.lidar_ray_feature_cache_root,
        lidar_ray_feature_cache_suffix=str(cfg_value("lidar_ray_feature_cache_suffix", ".npz")),
        lidar_ray_feature_dim=int(cfg_value("lidar_ray_feature_dim", 576)),
        lidar_ray_depth_bins=int(cfg_value("lidar_ray_depth_bins", 4)),
        lidar_ray_height=int(cfg_value("lidar_ray_height", 8)),
        lidar_ray_width=int(cfg_value("lidar_ray_width", 32)),
        image_semantic_cache_root=args.image_semantic_cache_root,
        image_semantic_cache_suffix=str(cfg_value("image_semantic_cache_suffix", ".npz")),
        image_semantic_feature_key=str(cfg_value("image_semantic_feature_key", "dino_feat")),
        image_semantic_feature_dim=int(cfg_value("image_semantic_feature_dim", 384)),
        image_semantic_height=int(cfg_value("image_semantic_height", 8)),
        image_semantic_width=int(cfg_value("image_semantic_width", 32)),
        include_tracklets=False,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_index = max(0, int(args.start_index))
    end_index = min(start_index + int(args.num_samples), len(dataset))
    samples_list = [dataset[idx] for idx in range(start_index, end_index)]
    manifest_rows = [json.loads(line) for line in open(args.manifest)]
    manifest_rows = manifest_rows[start_index : start_index + len(samples_list)]
    for row in manifest_rows:
        for key in ("velodyne_path", "oxts_path", "calib_dir"):
            row[key] = rebase_kitti_path(row[key], args.kitti_root)
    print(f"[noise-modes] {len(samples_list)} frames, mode={args.noise_mode}")

    model = instantiate_from_config(cfg.model).cuda().eval()
    load_checkpoint_into_model(model, args.ckpt)
    sampler = KITTI_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor)
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.eta, verbose=False)

    torch.manual_seed(args.seed)
    shared_x_T = torch.randn((1, 4, 16, 64), device="cuda")
    shared_eps = torch.randn((1, 4, 16, 64), device="cuda")

    # ddim_sampling treats an int timesteps n as "keep the n-1 smallest ddim
    # timesteps", so request ar_keep + 1 to run ar_keep denoising steps.
    ar_keep = max(1, int(round(args.ddim_steps * (1.0 - args.ar_strength))))
    tau_start = int(sampler.ddim_timesteps[ar_keep - 1])
    ar_abar = float(sampler.alphas_cumprod[tau_start])
    ar_steps_arg = ar_keep + 1
    print(
        f"[noise-modes] autoregressive: keep {ar_keep}/{args.ddim_steps} steps, "
        f"tau_start={tau_start}, sqrt(abar)={ar_abar ** 0.5:.4f}"
    )

    records = []
    prev_latent = None
    prev_target = None
    prev_row = None
    geoms = {}
    object_tracker = loa.ObjectTracker()
    warp_debug_left = int(args.warp_debug)
    for idx, sample in enumerate(samples_list):
        sample_id = sample["sample_id"]
        row = manifest_rows[idx]
        batch = sample_to_batch(sample)
        pack = prepare_frame_inputs(model, batch)
        key_stats = lidar_key_structure_stats(model, pack["lidar_context"], max_tokens=args.key_stats_max_tokens)

        if args.noise_mode == "per_frame":
            torch.manual_seed(args.seed + idx)
            x_T = torch.randn((1, 4, 16, 64), device="cuda")
            timesteps = None
            noise_info = {"mode": "per_frame", "seed": args.seed + idx}
        elif args.noise_mode == "shared":
            x_T = shared_x_T.clone()
            timesteps = None
            noise_info = {"mode": "shared", "seed": args.seed}
        elif args.noise_mode in ("warp", "warp2", "instance"):
            if prev_latent is None or prev_row is None:
                x_T = shared_x_T.clone()
                timesteps = None
                noise_info = {"mode": args.noise_mode, "frame": idx, "init": "shared_x_T"}
            else:
                geom = geoms.setdefault(
                    row["calib_dir"],
                    pwu.SequenceGeometry(resolve_calib_dir(row["calib_dir"])),
                )
                p1 = pwu.load_velodyne(prev_row["velodyne_path"])
                p2 = pwu.load_velodyne(row["velodyne_path"])
                T_v = geom.relative_velo_pose(prev_row["oxts_path"], row["oxts_path"])
                q = (T_v[:3, :3] @ p1.T).T + T_v[:3, 3]
                img_w, img_h = geom.img_size
                status = pwu.consistency_status(q, p2)
                mask, mask_stats = build_dynamic_mask_latent(
                    geom, p1, p2, q, img_w, img_h, 16, 64, "cuda"
                )
                n_v, d_v = pwu.fit_ground_plane_velo(p1)
                if n_v is None:
                    x_T = (ar_abar ** 0.5) * prev_latent + ((1.0 - ar_abar) ** 0.5) * shared_eps
                    timesteps = ar_steps_arg
                    noise_info = {"mode": args.noise_mode, "frame": idx, "init": "ar_fallback_no_ground"}
                else:
                    A = geom.R0[:3, :3] @ geom.T_cam_velo[:3, :3]
                    b = geom.R0[:3, :3] @ geom.T_cam_velo[:3, 3]
                    n_r = A @ n_v
                    d_r = d_v - float(n_r @ b)
                    T_c = geom.relative_cam_pose(prev_row["oxts_path"], row["oxts_path"])
                    H = pwu.ground_homography(T_c, n_r, d_r, geom.K_rect)
                    warpedH, validH = pwu.warp_latent_with_H(
                        prev_latent, H, img_w, img_h, 16, 64, "cuda"
                    )
                    if args.noise_mode == "warp":
                        keep = (validH * (1.0 - mask)).reshape(1, 1, 16, 64)
                        warped = warpedH
                        extra = {}
                    else:
                        flowA, fvalid, _, nfitted = pwu.depth_flow_affine(
                            p1[status == 1], q[status == 1], geom, img_w, img_h, 16, 64
                        )
                        flowH = pwu.homography_flow_cells(H, img_w, img_h, 16, 64)
                        agree_np = (
                            (fvalid > 0)
                            & (validH.squeeze().cpu().numpy() > 0)
                            & (np.linalg.norm(flowA - flowH, axis=-1) < 1.0)
                        )
                        agree = torch.from_numpy(agree_np.astype(np.float32)).to("cuda").reshape(1, 1, 16, 64)
                        # per-cell backward grid: H source on agreed cells, identity elsewhere
                        ys, xs = torch.meshgrid(
                            torch.arange(16, device="cuda", dtype=torch.float32) + 0.5,
                            torch.arange(64, device="cuda", dtype=torch.float32) + 0.5,
                            indexing="ij",
                        )
                        flowH_t = torch.from_numpy(flowH.astype(np.float32)).to("cuda").permute(2, 0, 1).unsqueeze(0)
                        grid_src = torch.where(agree > 0.5, flowH_t, torch.stack([xs, ys]).unsqueeze(0))
                        gx = 2.0 * grid_src[:, 0] / 64 - 1.0
                        gy = 2.0 * grid_src[:, 1] / 16 - 1.0
                        grid_t = torch.stack([gx, gy], dim=-1)
                        warped = torch.nn.functional.grid_sample(
                            prev_latent, grid_t, mode="bilinear", padding_mode="zeros", align_corners=False
                        )
                        keep = (1.0 - mask).reshape(1, 1, 16, 64)
                        extra = {
                            "agree_frac": float(agree.mean().item()),
                            "affine_cells": int(nfitted),
                        }
                    if args.noise_mode == "instance":
                        # plane tuple (a, b, c) with ground z = a*x + b*y - c,
                        # recovered from the normalized normal form (n, d).
                        plane = (-n_v[0] / n_v[2], -n_v[1] / n_v[2], d_v / n_v[2])
                        matched, assoc_stats = loa.associate_objects(
                            p1, p2, T_v, geom, img_w, img_h, 16, 64, plane=plane
                        )
                        obj_ids = object_tracker.update(matched)
                        obj_cell_np = np.zeros((16, 64), bool)
                        obj_records = []
                        for oid, obj in zip(obj_ids, matched):
                            ys_np, xs_np = np.meshgrid(
                                np.arange(16, dtype=np.float32) + 0.5,
                                np.arange(64, dtype=np.float32) + 0.5,
                                indexing="ij",
                            )
                            # backward transport: current cell center minus the
                            # object's latent displacement reaches its previous
                            # content.
                            src_np = np.stack(
                                [
                                    xs_np - obj["d_lat"][0],
                                    ys_np - obj["d_lat"][1],
                                ],
                                axis=0,
                            )
                            m_t = torch.from_numpy(obj["cell_mask"]).to("cuda")
                            grid_src = torch.where(
                                m_t.unsqueeze(0),
                                torch.from_numpy(src_np).to("cuda"),
                                grid_src[0],
                            ).unsqueeze(0)
                            obj_cell_np |= obj["cell_mask"]
                            obj_records.append(
                                {
                                    "oid": int(oid),
                                    "bbox": obj["bbox"],
                                    "d_lat": [float(x) for x in obj["d_lat"]],
                                    "d_velo_norm": float(np.linalg.norm(obj["d_velo"])),
                                    "count": obj["count"],
                                    "depth": obj["depth"],
                                }
                            )
                        obj_t = torch.from_numpy(obj_cell_np.astype(np.float32)).to("cuda").reshape(1, 1, 16, 64)
                        # matched object cells are exempt from the dynamic reset:
                        # they inherit via their own motion instead.
                        keep = keep * (1.0 - obj_t) + obj_t
                        extra = {
                            **extra,
                            **assoc_stats,
                            "n_objects_inherited": len(matched),
                            "objects": obj_records,
                        }
                    fill = torch.randn_like(warped)
                    L = warped * keep + fill * (1.0 - keep)
                    x_T = (ar_abar ** 0.5) * L + ((1.0 - ar_abar) ** 0.5) * shared_eps
                    timesteps = ar_steps_arg
                    noise_info = {
                        "mode": args.noise_mode,
                        "frame": idx,
                        "init": "pose_warp",
                        "steps": ar_keep,
                        "tau_start": tau_start,
                        "ar_strength": args.ar_strength,
                        "keep_frac": float(keep.mean().item()),
                        **mask_stats,
                        **extra,
                    }
                    if warp_debug_left > 0:
                        wimg = pwu.warp_latent_with_H(
                            prev_target.unsqueeze(0), H, img_w, img_h, 128, 512, "cuda"
                        )[0][0]
                        save_warp_debug(
                            out_dir / "warp_debug" / f"{idx:03d}.png",
                            prev_target,
                            pack["target"][0],
                            wimg,
                        )
                        warp_debug_left -= 1
        else:
            if prev_latent is None:
                x_T = shared_x_T.clone()
                timesteps = None
                noise_info = {"mode": "autoregressive", "frame": idx, "init": "shared_x_T"}
            else:
                x_T = (ar_abar ** 0.5) * prev_latent + ((1.0 - ar_abar) ** 0.5) * shared_eps
                timesteps = ar_steps_arg
                noise_info = {
                    "mode": "autoregressive",
                    "frame": idx,
                    "init": "prev_latent",
                    "tau_start": tau_start,
                    "steps": ar_keep,
                    "ar_strength": args.ar_strength,
                }

        pred, latent = sample_frame(model, sampler, pack, x_T, timesteps, args.guidance_scale, args.temperature)
        prev_latent = latent.detach()
        prev_target = pack["target"][0].detach()
        prev_row = row

        safe_id = str(sample_id).replace("/", "__")
        gt_path = out_dir / "images" / "gt" / f"{safe_id}.png"
        sat_path = out_dir / "images" / "satellite" / f"{safe_id}.png"
        overlay_path = out_dir / "images" / "lidar_overlay" / f"{safe_id}.png"
        cond_path = out_dir / "images" / "lidar_cond" / f"{safe_id}.png"
        pred_path = out_dir / "images" / "normal" / f"{safe_id}.png"
        save_tensor_image(pack["target"][0], gt_path)
        save_tensor_image(sample["sat_map"], sat_path)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        make_lidar_overlay(pack["target"][0], sample["lidar_cond"]).save(overlay_path)
        save_tensor_image(make_condition_rgb(sample["lidar_cond"]), cond_path)
        save_tensor_image(pred[0], pred_path)

        image_paths = {
            "Satellite input": sat_path,
            "LiDAR condition": cond_path,
            "LiDAR input (on GT)": overlay_path,
            "GT": gt_path,
            f"trained:normal({args.noise_mode})": pred_path,
        }
        panel_path = make_panel(out_dir, sample_id, image_paths)
        records.append(
            {
                "sample_id": sample_id,
                "panel_path": str(panel_path),
                "noise": noise_info,
                **{k: str(v) for k, v in image_paths.items()},
                "key_structure": key_stats,
            }
        )
        print(f"[noise-modes] {idx + 1}/{len(samples_list)} {sample_id} done")
        del batch, pack, pred, latent, x_T
        torch.cuda.empty_cache()

    (out_dir / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True))
    summary = {
        "out_dir": str(out_dir),
        "num_samples": len(records),
        "noise_mode": args.noise_mode,
        "ar_strength": args.ar_strength,
        "seed": args.seed,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
