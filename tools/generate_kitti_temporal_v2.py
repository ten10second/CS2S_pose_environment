"""Sequential rollout with GENERATED history for temporal-history v2.

Unlike the training script (teacher-forced GT previous frame), this samples
frames one after another and feeds each frame's ACTUALLY GENERATED final
latent as the next frame's history — the error-accumulation regime the
training phase A cannot measure.

Frame 0 has no history (exact single-frame behaviour); frames 1..N consume
the previous frame's generated latent through the same history encoder and
attention used in training.

Output layout matches the other sampling modes (images/{gt,normal,...},
panels, records.json) so the shared evaluation scripts work unchanged.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for p in (str(TOOLS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from omegaconf import OmegaConf  # noqa: E402

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402

from generate_kitti_raea_samples import (  # noqa: E402
    load_checkpoint_into_model,
    make_condition_rgb,
    make_lidar_overlay,
    save_tensor_image,
)
from generate_kitti_raea_noise_modes import (  # noqa: E402
    make_panel,
    prepare_frame_inputs,
    sample_frame,
    sample_to_batch,
)
from temporal_history import enable_history_attention  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml")
    p.add_argument("--sd-base-ckpt", required=True)
    p.add_argument("--ckpt", required=True, help="frozen single-frame checkpoint")
    p.add_argument("--hist-ckpt", required=True, help="hist_v2 checkpoint (encoder+attention)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--kitti-root", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--lidar-ray-feature-cache-root", required=True)
    p.add_argument("--image-semantic-cache-root", required=True)
    p.add_argument("--start-index", type=int, required=True)
    p.add_argument("--num-samples", type=int, default=120)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--history-dim", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt
    test_params = cfg.data.params.test.params

    dataset = SatLidarRawDataset(
        manifest=args.manifest,
        kitti_root=args.kitti_root,
        condition_mode=str(getattr(test_params, "condition_mode", "raw_lidar_pointmap")),
        image_height=128,
        image_width=512,
        sat_size=256,
        max_depth=80.0,
        align_satellite_to_camera=True,
        include_range_image=False,
        include_raw_lidar_points=False,
        lidar_ray_feature_cache_root=args.lidar_ray_feature_cache_root,
        lidar_ray_feature_cache_suffix=".npz",
        lidar_ray_feature_dim=576,
        lidar_ray_depth_bins=4,
        lidar_ray_height=8,
        lidar_ray_width=32,
        image_semantic_cache_root=args.image_semantic_cache_root,
        image_semantic_cache_suffix=".npz",
        image_semantic_feature_key="dino_feat",
        image_semantic_feature_dim=384,
        image_semantic_height=8,
        image_semantic_width=32,
        include_tracklets=False,
    )

    model = instantiate_from_config(cfg.model)
    load_checkpoint_into_model(model, args.ckpt)
    hub, encoder, blocks = enable_history_attention(model, history_dim=args.history_dim)
    model = model.cuda().eval()
    encoder = encoder.cuda().eval()

    hist_ckpt = torch.load(args.hist_ckpt, map_location="cpu")
    if hist_ckpt.get("mode") != "temporal_v2_history_phaseA":
        raise ValueError(f"unexpected hist checkpoint mode: {hist_ckpt.get('mode')!r}")
    encoder.load_state_dict(hist_ckpt["history_encoder"], strict=True)
    for i, block in enumerate(blocks):
        block.history_attn.load_state_dict(hist_ckpt["history_attn"][str(i)], strict=True)
    devs = {str(b.history_attn.to_out.weight.device) for b in blocks}
    assert devs == {"cuda:0"}, f"history attention on wrong devices: {devs}"
    print(f"[rollout] hist weights loaded from {args.hist_ckpt} (step {hist_ckpt.get('step', '?')})")

    sampler = KITTI_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor)
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=1.0, verbose=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    z_prev = None  # previous frame's GENERATED final latent
    for idx in range(args.start_index, args.start_index + args.num_samples):
        sample = dataset[idx]
        sample_id = sample["sample_id"]
        batch = sample_to_batch(sample)
        pack = prepare_frame_inputs(model, batch)

        if z_prev is None:
            hub.clear()
            has_history = False
        else:
            tokens = encoder(z_prev)
            hub.set({"history_tokens": tokens, "has_history": True})
            has_history = True

        torch.manual_seed(args.seed + idx)
        x_T = torch.randn((1, 4, 16, 64), device="cuda")
        pred, latent = sample_frame(
            model, sampler, pack, x_T, None, args.guidance_scale, 1.0
        )
        z_prev = latent.detach()  # becomes the next frame's history

        safe_id = str(sample_id).replace("/", "__")
        image_paths = {
            "Satellite input": out_dir / "images" / "satellite" / f"{safe_id}.png",
            "LiDAR condition": out_dir / "images" / "lidar_cond" / f"{safe_id}.png",
            "LiDAR input (on GT)": out_dir / "images" / "lidar_overlay" / f"{safe_id}.png",
            "GT": out_dir / "images" / "gt" / f"{safe_id}.png",
        }
        for sub, path in (
            ("satellite", image_paths["Satellite input"]),
            ("lidar_cond", image_paths["LiDAR condition"]),
            ("lidar_overlay", image_paths["LiDAR input (on GT)"]),
            ("gt", image_paths["GT"]),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        save_tensor_image(pack["target"][0], image_paths["GT"])
        save_tensor_image(sample["sat_map"], image_paths["Satellite input"])
        make_lidar_overlay(pack["target"][0], sample["lidar_cond"]).save(
            image_paths["LiDAR input (on GT)"]
        )
        save_tensor_image(make_condition_rgb(sample["lidar_cond"]), image_paths["LiDAR condition"])
        pred_path = out_dir / "images" / "normal" / f"{safe_id}.png"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        save_tensor_image(pred[0], pred_path)

        panel_path = make_panel(
            out_dir, sample_id, {**image_paths, "trained:normal(rollout)": pred_path}
        )
        lat_norm = float(latent.detach().float().pow(2).mean().sqrt())
        records.append(
            {
                "sample_id": sample_id,
                "frame_index": int(sample.get("frame_index", idx)),
                "has_history": has_history,
                "latent_rms": lat_norm,
                "panel_path": str(panel_path),
                "normal_path": str(pred_path),
            }
        )
        print(f"[rollout] {idx - args.start_index + 1}/{args.num_samples} {sample_id} "
              f"hist={has_history} latent_rms={lat_norm:.3f}", flush=True)
        del batch, pack, pred, latent
        torch.cuda.empty_cache()

    (out_dir / "records.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"out_dir": str(out_dir), "num_samples": len(records)}))


if __name__ == "__main__":
    main()
