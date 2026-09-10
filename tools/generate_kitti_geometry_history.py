"""Geometry-aware generated-history rollout for KITTI temporal experiments.

This sampler keeps the single-frame CFG/LiDAR conditioning path intact and
adds only a geometry-local history payload:

  current satellite + current LiDAR + generated previous latent
  + prev->current geometry correspondence -> current frame

Frame history is always the previous GENERATED latent, except for the optional
``--first-rgb`` bootstrap where only frame 0 may initialize history from the
real first RGB observation. Gaps and drive changes clear history and should be
bit-identical to the frozen single-frame model when the history adapter is
zero-initialized.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for p in (str(TOOLS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


HISTORY_MODE = "geometry_history_v1_transport"
HISTORY_MODES = {
    "geometry_history_v1",
    "geometry_history_v1_transport",
    "geometry_history_v1_generator",
}
DEFAULT_GRID = (16, 64)


def parse_block_indices(text):
    if text is None or str(text).strip() == "":
        raise ValueError("--block-indices is required for geometry-history rollout")
    from temporal_history import AFTER_BOTTLENECK, parse_history_block_indices

    parsed = parse_history_block_indices(text)
    if parsed == AFTER_BOTTLENECK:
        return AFTER_BOTTLENECK
    if not parsed:
        raise ValueError("--block-indices did not contain any indices")
    if any(value < 0 for value in parsed):
        raise ValueError(f"block index must be non-negative, got {parsed}")
    return tuple(parsed)


def cfg_batch_factor(uncond_cfg):
    """The DDIM sampler duplicates the batch only for active CFG scale != 1."""
    return 2 if float(uncond_cfg) > 0.0 and float(uncond_cfg) != 1.0 else 1


def ensure_fresh_out_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"--out-dir must be fresh/empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical_path(value):
    p = Path(str(value)).expanduser()
    if p.exists():
        return str(p.resolve())
    return str(p)


def same_base_checkpoint(saved, requested):
    return _canonical_path(saved) == _canonical_path(requested)


def require_geometry_history_payload(payload, args, blocks):
    required = {
        "step",
        "history_encoder",
        "history_attn",
        "mode",
        "base_ckpt",
        "block_indices",
        "history_dim",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"history checkpoint missing required keys: {missing}")
    if payload["mode"] not in HISTORY_MODES:
        raise ValueError(f"unexpected history checkpoint mode: {payload['mode']!r}")
    if not same_base_checkpoint(payload["base_ckpt"], args.ckpt):
        raise ValueError(
            "history checkpoint base_ckpt mismatch: "
            f"{payload['base_ckpt']!r} != {args.ckpt!r}"
        )
    if tuple(payload["block_indices"]) != tuple(args.block_indices):
        raise ValueError(
            "history checkpoint block_indices mismatch: "
            f"{tuple(payload['block_indices'])} != {tuple(args.block_indices)}"
        )
    if int(payload["history_dim"]) != int(args.history_dim):
        raise ValueError(
            f"history_dim mismatch: {payload['history_dim']} != {args.history_dim}"
        )
    expected = {str(i) for i in range(len(blocks))}
    actual = set(payload["history_attn"])
    if actual != expected:
        raise ValueError(
            f"history_attn keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )


def load_geometry_history_checkpoint(path, args, encoder, blocks, map_location="cpu"):
    from temporal_history import load_history_host_state_dict

    payload = torch.load(path, map_location=map_location)
    require_geometry_history_payload(payload, args, blocks)
    encoder.load_state_dict(payload["history_encoder"], strict=True)
    for i, block in enumerate(blocks):
        block.history_attn.load_state_dict(payload["history_attn"][str(i)], strict=True)
    if payload.get("unfreeze_host"):
        load_history_host_state_dict(blocks, payload.get("history_host"))
    return payload


def row_sequence_id(row):
    if row is None:
        return None
    for key in ("drive", "sequence_id", "seq_id"):
        if key in row:
            return str(row[key])
    sample_id = str(row.get("sample_id", ""))
    parts = sample_id.split("/")
    return parts[1] if len(parts) > 1 else parts[0]


def row_frame_index(row, fallback):
    if row is not None and "frame_index" in row:
        return int(row["frame_index"])
    return int(fallback)


def consecutive_rows(prev_row, cur_row, prev_fallback=None, cur_fallback=None):
    if prev_row is None or cur_row is None:
        return False
    if row_sequence_id(prev_row) != row_sequence_id(cur_row):
        return False
    prev_idx = row_frame_index(prev_row, prev_fallback if prev_fallback is not None else -10**9)
    cur_idx = row_frame_index(cur_row, cur_fallback if cur_fallback is not None else 10**9)
    return cur_idx == prev_idx + 1


def rebase_kitti_path(path, kitti_root):
    path = str(path)
    marker = "KITTI_RAW/"
    idx = path.find(marker)
    if idx >= 0 and kitti_root:
        return str(Path(kitti_root) / path[idx + len(marker):])
    return path


def rebase_manifest_row(row, kitti_root):
    row = dict(row)
    for key in ("velodyne_path", "oxts_path", "calib_dir"):
        if key in row:
            row[key] = rebase_kitti_path(row[key], kitti_root)
    return row


def _geometry_value(geometry, name, default=None):
    if isinstance(geometry, dict):
        return geometry.get(name, default)
    return getattr(geometry, name, default)


def _as_batched_tensor(value, *, device, dtype, batch_factor):
    t = torch.as_tensor(value, device=device, dtype=dtype).unsqueeze(0)
    return t.repeat((batch_factor,) + (1,) * (t.dim() - 1))


@torch.no_grad()
def build_geometry_history_payload(encoder, history_latent, geometry, has_history, batch_factor=1):
    """Build the hub payload consumed by geometry-aware history attention."""
    device = next(encoder.parameters()).device
    if has_history:
        if history_latent is None:
            raise ValueError("history_latent is required when has_history=True")
        tokens = encoder(history_latent.detach())
        if batch_factor > 1:
            tokens = tokens.repeat(batch_factor, 1, 1)
    else:
        tokens = encoder.null_tokens(batch_factor)

    grid_h, grid_w = tuple(getattr(encoder, "grid", DEFAULT_GRID))
    if geometry is None:
        history_grid = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
        history_valid = np.zeros((grid_h, grid_w), dtype=bool)
    else:
        history_grid = _geometry_value(geometry, "history_grid")
        history_valid = _geometry_value(geometry, "history_valid")
        if history_grid is None or history_valid is None:
            raise KeyError("geometry must provide history_grid and history_valid")

    payload = {
        "history_tokens": tokens,
        "has_history": bool(has_history),
        "history_hw": (grid_h, grid_w),
        "history_grid": _as_batched_tensor(
            history_grid, device=device, dtype=torch.float32, batch_factor=batch_factor
        ),
        "history_valid": _as_batched_tensor(
            history_valid, device=device, dtype=torch.bool, batch_factor=batch_factor
        ),
    }
    # Preserve optional support/visibility maps if the geometry builder exposes
    # them; attention modules that do not use these keys can ignore them.
    for name in ("history_support", "history_visibility"):
        value = _geometry_value(geometry, name) if geometry is not None else None
        if value is not None:
            payload[name] = _as_batched_tensor(
                value, device=device, dtype=torch.float32, batch_factor=batch_factor
            )
    return payload


def next_history_latent_source(frame_offset, first_rgb, history_source="generated"):
    source = str(history_source or "generated")
    if source not in {"generated", "gt"}:
        raise ValueError(f"unsupported history_source: {history_source!r}")
    if bool(first_rgb) and int(frame_offset) == 0:
        return "first_rgb"
    if source == "gt":
        return "gt_rgb"
    return "generated"


def use_observed_initial_frame(frame_offset, first_rgb):
    return bool(first_rgb) and int(frame_offset) == 0


def require_contiguous_sequence(rows, expected_count):
    if len(rows) != expected_count or not rows:
        raise ValueError('requested complete sequence is not available')
    if any(not consecutive_rows(a, b) for a, b in zip(rows, rows[1:])):
        raise ValueError('diagnostic sequence must be consecutive within one drive')


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def assert_sampling_inputs_equal(first, second):
    """GT may change the visualization target, never the denoiser conditions."""
    if set(first) != set(second):
        raise RuntimeError('sampling input keys changed when future RGB was replaced')
    for key in first:
        if key == 'target':
            continue
        a, b = first[key], second[key]
        equal = torch.equal(a, b) if torch.is_tensor(a) and torch.is_tensor(b) else a is b
        if not equal:
            raise RuntimeError(f'future RGB changed sampling condition: {key}')


@torch.no_grad()
def encode_rgb_history_latent(model, target):
    """Encode an RGB tensor in [0,1] as a scaled latent for history bootstrap."""
    z = model.pre_AE_model.encode(target * 2.0 - 1.0).sample()
    return z * model.scale_factor


def enable_geometry_history_attention(model, args, payload=None):
    from temporal_history import AFTER_BOTTLENECK, enable_history_attention

    try:
        hub, encoder, blocks = enable_history_attention(
            model,
            geometry=True,
            block_indices=args.block_indices,
            history_dim=args.history_dim,
            heads=args.heads,
            dim_head=args.dim_head,
        )
    except TypeError as exc:
        raise RuntimeError(
            "temporal_history.enable_history_attention must support "
            "geometry=True and block_indices for geometry-history rollout"
        ) from exc
    args.block_indices = tuple(block.history_block_index for block in blocks)
    return hub, encoder, blocks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml")
    p.add_argument("--sd-base-ckpt", required=True)
    p.add_argument("--ckpt", required=True, help="frozen CFG single-frame checkpoint")
    p.add_argument("--hist-ckpt", required=True, help="geometry_history_v1 checkpoint")
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
    p.add_argument("--uncond-cfg", type=float, default=0.0)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--history-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim-head", type=int, default=32)
    p.add_argument(
        "--block-indices",
        type=parse_block_indices,
        required=True,
        help="Fusion indices, or 'after_bottleneck' to match the post-depth decoder layer",
    )
    p.add_argument("--first-rgb", action="store_true",
                   help="bootstrap frame 1 from frame 0 RGB; later history is generated only")
    p.add_argument(
        "--history-source",
        choices=("generated", "gt"),
        default="generated",
        help="generated: closed-loop latents; gt: encode each previous GT RGB as history",
    )
    p.add_argument("--disable-history", action="store_true",
                   help="same loaded checkpoint, but bypass history for the paired baseline")
    p.add_argument("--require-contiguous", action="store_true",
                   help="reject truncated, nonconsecutive or cross-drive diagnostic clips")
    return p.parse_args()


def main():
    from omegaconf import OmegaConf

    from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset
    from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler, default_noise
    from ar_dyn_utils import seed_step_noise
    from utils.util import instantiate_from_config

    from generate_kitti_raea_samples import (
        load_checkpoint_into_model,
        make_condition_rgb,
        make_lidar_overlay,
        make_panel,
        save_tensor_image,
    )
    from generate_kitti_raea_noise_modes import prepare_frame_inputs, sample_frame, sample_to_batch
    from temporal_history_geometry import build_pair_geometry

    args = parse_args()
    out_dir = ensure_fresh_out_dir(args.out_dir)
    batch_factor = cfg_batch_factor(args.uncond_cfg)

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
    start_index = max(0, int(args.start_index))
    end_index = min(start_index + int(args.num_samples), len(dataset))
    samples_list = [dataset[idx] for idx in range(start_index, end_index)]
    manifest_rows = [json.loads(line) for line in open(args.manifest)]
    manifest_rows = [
        rebase_manifest_row(row, args.kitti_root)
        for row in manifest_rows[start_index : start_index + len(samples_list)]
    ]
    if args.require_contiguous:
        require_contiguous_sequence(manifest_rows, int(args.num_samples))

    model = instantiate_from_config(cfg.model)
    load_checkpoint_into_model(model, args.ckpt)
    hist_payload = torch.load(args.hist_ckpt, map_location="cpu")
    hub, encoder, blocks = enable_geometry_history_attention(model, args, hist_payload)
    model = model.cuda().eval()
    encoder = encoder.cuda().eval()

    hist_payload = load_geometry_history_checkpoint(args.hist_ckpt, args, encoder, blocks)
    print(
        f"[geometry-history] weights loaded from {args.hist_ckpt} "
        f"(step {hist_payload.get('step', '?')}, blocks={tuple(args.block_indices)})"
    )

    sampler = KITTI_DDIMSampler(model.DDPM, model.pre_AE_model, model.scale_factor)
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.eta, verbose=False)
    # The sampler creates this bank at import time, before CLI seeding. Pin it
    # explicitly for paired eta>0 runs in separate processes.
    seed_step_noise(default_noise, args.seed)
    step_noise_sha256 = tensor_sha256(torch.stack(default_noise))
    torch.manual_seed(args.seed + start_index)

    # Read-only runtime evidence that real CFG and history execute at each step.
    step_trace = []
    denoiser_batches = []
    def observe_denoiser(_module, inputs):
        denoiser_batches.append(int(inputs[0].shape[0]))
    def history_observer(block_index):
        def observe(attention, inputs, output):
            with torch.no_grad():
                denominator = float(inputs[0].detach().float().norm(dim=-1).mean())
                numerator = float(output.detach().float().norm(dim=-1).mean())
            step_trace.append(dict(block_index=block_index,
                null_all=attention.last_null_frac,
                valid_neighbor_fraction=attention.last_valid_frac,
                residual_to_condition=attention.last_ratio,
                memory_to_condition=attention.last_memory_ratio,
                residual_to_x=numerator / max(denominator, 1e-6)))
        return observe
    hooks = [model.DDPM.denoise_model.register_forward_pre_hook(observe_denoiser)]
    hooks.extend(b.history_attn.register_forward_hook(history_observer(b.history_block_index)) for b in blocks)

    records = []
    prev_latent = None
    prev_row = None
    prev_frame_index = None
    future_rgb_condition_check_passed = None
    for offset, (sample, row) in enumerate(zip(samples_list, manifest_rows)):
        sample_id = sample["sample_id"]
        frame_index = row_frame_index(row, start_index + offset)
        batch = sample_to_batch(sample)
        pack = prepare_frame_inputs(model, batch)
        if offset == 1 and args.first_rgb:
            poisoned = dict(batch)
            for key in ('grd_left_imgs', 'image_semantic_feat'):
                if key in poisoned:
                    poisoned[key] = torch.zeros_like(poisoned[key])
            poisoned_pack = prepare_frame_inputs(model, poisoned)
            assert_sampling_inputs_equal(pack, poisoned_pack)
            future_rgb_condition_check_passed = True
            del poisoned, poisoned_pack

        observed_initial = use_observed_initial_frame(offset, args.first_rgb)
        has_history = (
            not args.disable_history and prev_latent is not None
            and consecutive_rows(prev_row, row, prev_frame_index, frame_index)
        )
        input_history_hash = tensor_sha256(prev_latent) if has_history else None
        geometry = None
        if args.disable_history:
            hub.clear()
            valid_frac = 0.0
        elif has_history:
            geometry = build_pair_geometry(prev_row, row, args.kitti_root, grid=DEFAULT_GRID)
            hub.set(
                build_geometry_history_payload(
                    encoder, prev_latent, geometry, True, batch_factor=batch_factor
                )
            )
            valid_frac = float(np.asarray(_geometry_value(geometry, "history_valid")).mean())
        else:
            hub.set(
                build_geometry_history_payload(
                    encoder, None, None, False, batch_factor=batch_factor
                )
            )
            valid_frac = 0.0

        step_trace.clear()
        denoiser_batches.clear()
        x_T_hash = None
        source = next_history_latent_source(offset, args.first_rgb, args.history_source)
        if observed_initial:
            pred = pack["target"].detach()
            latent = None
            torch.manual_seed(args.seed + start_index)
            prev_latent = encode_rgb_history_latent(model, pack["target"]).detach()
            hub.clear()
        else:
            torch.manual_seed(args.seed + start_index + offset)
            x_T = torch.randn((1, 4, 16, 64), device="cuda")
            x_T_hash = tensor_sha256(x_T)
            pred, latent = sample_frame(
                model,
                sampler,
                pack,
                x_T,
                None,
                args.guidance_scale,
                args.temperature,
                uncond_cfg=args.uncond_cfg,
            )
            hub.clear()
            if source == "gt_rgb":
                prev_latent = encode_rgb_history_latent(model, pack["target"]).detach()
            else:
                prev_latent = latent.detach()
            if not denoiser_batches or any(b != batch_factor for b in denoiser_batches):
                raise RuntimeError('actual denoiser batch does not match requested CFG')
            if args.disable_history and step_trace:
                raise RuntimeError('disabled baseline unexpectedly read history')
            if has_history and len(step_trace) != len(denoiser_batches) * len(blocks):
                raise RuntimeError('history did not execute at every denoising step')
        prev_row = row
        prev_frame_index = frame_index

        safe_id = str(sample_id).replace("/", "__")
        image_paths = {
            "Satellite input": out_dir / "images" / "satellite" / f"{safe_id}.png",
            "LiDAR condition": out_dir / "images" / "lidar_cond" / f"{safe_id}.png",
            "LiDAR input (on GT)": out_dir / "images" / "lidar_overlay" / f"{safe_id}.png",
            "GT": out_dir / "images" / "gt" / f"{safe_id}.png",
        }
        for path in image_paths.values():
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
            out_dir, sample_id, {**image_paths, "geometry-history": pred_path}
        )
        lat_source = prev_latent if latent is None else latent
        lat_norm = float(lat_source.detach().float().pow(2).mean().sqrt())
        records.append(
            {
                "sample_id": sample_id,
                "frame_index": frame_index,
                "sequence_id": row_sequence_id(row),
                "has_history": bool(has_history),
                "history_disabled": bool(args.disable_history),
                "history_input_sha256": input_history_hash,
                "history_output_sha256": tensor_sha256(prev_latent),
                "initial_noise_sha256": x_T_hash,
                "denoiser_batch_sizes": list(denoiser_batches),
                "history_attention_steps": list(step_trace),
                "is_observed_initial_frame": bool(observed_initial),
                "history_valid_frac": valid_frac,
                "history_source_for_next": source,
                "latent_rms": lat_norm,
                "panel_path": str(panel_path),
                "normal_path": str(pred_path),
            }
        )
        print(
            f"[geometry-history] {offset + 1}/{len(samples_list)} {sample_id} "
            f"hist={has_history} valid={valid_frac:.3f} next={source}",
            flush=True,
        )
        # Keep completed-frame provenance even if a later frame fails.
        (out_dir / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True))
        if observed_initial:
            del batch, pack, pred
        else:
            del batch, pack, pred, latent, x_T
        torch.cuda.empty_cache()

    (out_dir / "records.json").write_text(json.dumps(records, indent=2, sort_keys=True))
    summary = {
        "out_dir": str(out_dir),
        "num_samples": len(records),
        "mode": HISTORY_MODE,
        "seed": args.seed,
        "cfg_batch_factor": batch_factor,
        "uncond_cfg": args.uncond_cfg,
        "base_ckpt": args.ckpt,
        "hist_ckpt": args.hist_ckpt,
        "hist_step": hist_payload.get("step"),
        "block_indices": list(args.block_indices),
        "history_dim": args.history_dim,
        "heads": args.heads,
        "dim_head": args.dim_head,
        "first_rgb": bool(args.first_rgb),
        "history_source": args.history_source,
        "history_disabled": bool(args.disable_history),
        "step_noise_sha256": step_noise_sha256,
        "future_rgb_condition_check_passed": future_rgb_condition_check_passed,
        "args": vars(args),
    }
    for hook in hooks:
        hook.remove()
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
