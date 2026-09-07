"""Post-hoc re-evaluation of temporal-v2 checkpoints with a trustworthy probe.

Fixes the four probe limitations flagged in review (P2 follow-up):
  - FIXED history latent per pair (VAE sample drawn once from a per-pair
    seeded generator, cached, reused across every condition/timestep);
  - FIXED diffusion timestep per measurement (torch.randint intercepted for
    the duration of the forward), sweeping a fixed timestep list;
  - FIXED noise per (pair, t, draw) — identical across all history
    conditions, so condition differences are pure history effects;
  - per-pair, per-timestep, per-condition JSONL rows with the FULL loss
    component breakdown from DDPM.last_loss_metrics (denoising base, point
    region terms, LiDAR depth terms, total), not just aggregates.

Also reports the step-0 (fresh, untrained history modules) reference where
all conditions must be identical.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for p in (str(TOOLS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from omegaconf import OmegaConf  # noqa: E402

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset  # noqa: E402
from torch.utils.data._utils.collate import default_collate  # noqa: E402
from utils.util import instantiate_from_config  # noqa: E402

from generate_kitti_raea_samples import load_checkpoint_into_model  # noqa: E402
from train_kitti_temporal import (  # noqa: E402
    CONSTANTS,
    build_stream_plan,
    move_batch_to_device,
)
from temporal_history import (  # noqa: E402
    enable_history_attention,
    history_trainable_parameters,
)


class fixed_timestep:
    """Force torch.randint results to a constant during the wrapped forward.
    Only intercepts calls whose (low, high) match the DDPM range; everything
    else passes through untouched."""

    def __init__(self, t_value, low=0, high=1000):
        self.t_value = t_value
        self.low = low
        self.high = high
        self._orig = None

    def __enter__(self):
        target = self

        def patched(*args, **kwargs):
            if (
                len(args) >= 2
                and args[0] == target.low
                and isinstance(args[1], int)
                and args[1] == target.high
            ):
                size = args[2] if len(args) > 2 else kwargs.get("size", ())
                dev = kwargs.get("device", "cpu")
                dtype = kwargs.get("dtype", torch.long)
                return torch.full(size, target.t_value, device=dev, dtype=dtype)
            return target._orig(*args, **kwargs)

        self._orig = torch.randint
        torch.randint = patched
        return self

    def __exit__(self, *exc):
        torch.randint = self._orig
        return False


class fixed_noise:
    """First torch.randn_like call inside the context returns a tensor drawn
    from a seeded generator; later calls pass through. Used to pin the diffusion
    noise of one measurement."""

    def __init__(self, seed, ref_device, ref_shape):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.tensor = torch.randn(ref_shape, generator=g).to(ref_device)
        self._orig = None

    def __enter__(self):
        target = self

        def patched(inp, *a, **kw):
            if target.tensor is not None:
                out = target.tensor.to(device=inp.device, dtype=inp.dtype)
                target.tensor = None
                return out
            return target._orig(inp, *a, **kw)

        import torch as _torch
        target._orig = _torch.randn_like
        _torch.randn_like = patched
        return self

    def __exit__(self, *exc):
        import torch as _torch
        _torch.randn_like = self._orig
        return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml")
    p.add_argument("--sd-base-ckpt", required=True)
    p.add_argument("--ckpt", required=True, help="frozen single-frame checkpoint")
    p.add_argument("--hist-checkpoints", required=True,
                   help="comma-separated hist_v2 checkpoints; the token 'fresh' "
                        "evaluates the untrained zero-init modules")
    p.add_argument("--manifest", required=True)
    p.add_argument("--kitti-root", default="")
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--lidar-ray-feature-cache-root", required=True)
    p.add_argument("--image-semantic-cache-root", required=True)
    p.add_argument("--train-pairs", type=int, default=16)
    p.add_argument("--val-offset", type=int, default=48)
    p.add_argument("--val-pairs", type=int, default=16)
    p.add_argument("--timesteps", default="250,750")
    p.add_argument("--draws", type=int, default=2)
    p.add_argument("--history-dim", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    timesteps = [int(t) for t in args.timesteps.split(",")]

    cfg = OmegaConf.load(args.config)
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt
    train_params = cfg.data.params.test.params

    rows = [json.loads(line) for line in open(args.manifest)]
    full_plan = build_stream_plan(rows)
    train_pairs = full_plan[: args.train_pairs]
    val_pairs = full_plan[args.val_offset : args.val_offset + args.val_pairs]
    print(f"[re-eval] train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}, "
          f"timesteps: {timesteps}, draws: {args.draws}")

    dataset = SatLidarRawDataset(
        manifest=args.manifest,
        kitti_root=args.kitti_root,
        condition_mode=str(getattr(train_params, "condition_mode", "raw_lidar_pointmap")),
        image_height=CONSTANTS["image_height"],
        image_width=CONSTANTS["image_width"],
        sat_size=CONSTANTS["sat_size"],
        max_depth=CONSTANTS["max_depth"],
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

    model = instantiate_from_config(cfg.model).cuda().eval()
    load_checkpoint_into_model(model, args.ckpt)
    hub, encoder, blocks = enable_history_attention(model, history_dim=args.history_dim)
    model.temporal_hub = hub
    model.DDPM.temporal_hub = hub  # both roots; blocks and EvalModule share it
    model.to(device)
    encoder.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    def gt_latent(pair, seed):
        """Deterministic per-pair history latent: the repo's distribution
        sample() uses the global RNG, so seed it immediately before the call;
        computed once and cached (review: history latent must not vary between
        probe conditions)."""
        pi, _ci, _v = pair
        prev_batch = move_batch_to_device(default_collate([dataset[pi]]), device)
        outputs = prev_batch["grd_left_imgs"] * 2 - 1
        with torch.no_grad():
            torch.manual_seed(7000 + pi + seed)
            dist = model.pre_AE_model.encode(outputs)
            z = dist.sample() * model.scale_factor
        return z.detach()

    # fixed wrong-history sources, computed once
    ref_drive = rows[train_pairs[0][0]].get("drive")
    same_far = next(
        (p for p in full_plan
         if rows[p[0]].get("drive") == ref_drive
         and abs(int(rows[p[0]].get("frame_index", 0)) - int(rows[train_pairs[0][0]].get("frame_index", 0))) >= 500),
        full_plan[-1],
    )
    other = next(p for p in full_plan if rows[p[0]].get("drive") != ref_drive)
    wrong_latents = {
        "same_drive_far": gt_latent(same_far, seed=555),
        "other_drive": gt_latent(other, seed=666),
    }
    print(f"[re-eval] wrong sources ready: {list(wrong_latents)}")

    hist_ckpts = []
    for token in args.hist_checkpoints.split(","):
        token = token.strip()
        if token == "fresh":
            hist_ckpts.append(("fresh", None))
        else:
            p = Path(token)
            if not p.exists():
                raise FileNotFoundError(p)
            hist_ckpts.append((p.stem, p))

    class EvalModule(torch.nn.Module):
        """Calls the OUTER pl module's training_step (which assembles all
        loss_kwargs/aux terms); the hub is attached to both roots so payload
        setting works regardless of which root the blocks hang under."""

        def __init__(self, pl_model, history_encoder):
            super().__init__()
            self.pl_model = pl_model
            self.history_encoder = history_encoder

        def forward(self, cur_batch, history_latent=None, has_history=True):
            if has_history:
                if history_latent is None:
                    raise ValueError("history_latent required when has_history=True")
                tokens = self.history_encoder(history_latent.detach())
                tokens = tokens + self.history_encoder.null_tokens(
                    cur_batch["grd_left_imgs"].shape[0]
                ) * 0.0
            else:
                ref = cur_batch["grd_left_imgs"]
                z_ph = torch.zeros(ref.shape[0], 4, 16, 64, device=ref.device, dtype=ref.dtype)
                tokens = self.history_encoder(z_ph)
            self.pl_model.temporal_hub.set(
                {"history_tokens": tokens, "has_history": has_history}
            )
            loss = self.pl_model.training_step(cur_batch, 0)
            assert loss is not None, "training_step returned None"
            return loss

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows_out = out_jsonl.open("w")

    for ckpt_name, ckpt_path in hist_ckpts:
        if ckpt_path is not None:
            payload = torch.load(ckpt_path, map_location="cpu")
            encoder.load_state_dict(payload["history_encoder"])
            for i, block in enumerate(blocks):
                block.history_attn.load_state_dict(payload["history_attn"][str(i)])
        else:
            # re-initialise to the exact step-0 state for the fresh reference
            hub2, encoder2, blocks2 = enable_history_attention(
                instantiate_from_config(cfg.model), history_dim=args.history_dim
            )
            encoder.load_state_dict(encoder2.state_dict())
            for i, block in enumerate(blocks):
                block.history_attn.load_state_dict(blocks2[i].history_attn.state_dict())
        model.to(device)
        model.eval()
        eval_module = EvalModule(model, encoder).to(device).eval()

        for split, pairs in (("train", train_pairs), ("val", val_pairs)):
            for pair_no, pair in enumerate(pairs):
                pi, ci, _v = pair
                cur_batch = move_batch_to_device(default_collate([dataset[ci]]), device)
                z_hist = gt_latent(pair, seed=7000 + pi)
                pair_id = f"{rows[pi].get('drive')}:{rows[pi].get('frame_index')}->{rows[ci].get('drive')}:{rows[ci].get('frame_index')}"
                ref_img = cur_batch["grd_left_imgs"]

                for t_fixed in timesteps:
                    for draw in range(args.draws):
                        for cond_name, (latent, has_hist) in {
                            "correct": (z_hist, True),
                            "disabled": (None, False),
                            "wrong:other_drive": (wrong_latents["other_drive"], True),
                            "wrong:same_drive_far": (wrong_latents["same_drive_far"], True),
                        }.items():
                            torch.manual_seed(9000 + draw)
                            with fixed_timestep(t_fixed), fixed_noise(
                                9000 + draw, ref_img.device, (ref_img.shape[0], 4, 16, 64)
                            ):
                                with torch.no_grad():
                                    loss_total = eval_module(cur_batch, latent, has_hist)
                            metrics = dict(model.DDPM.last_loss_metrics)
                            row = {
                                "checkpoint": ckpt_name,
                                "split": split,
                                "pair_no": pair_no,
                                "pair_id": pair_id,
                                "t": t_fixed,
                                "draw": draw,
                                "condition": cond_name,
                                "loss_total": float(loss_total),
                                **{k: float(v) for k, v in metrics.items()
                                   if isinstance(v, (int, float))},
                            }
                            rows_out.write(json.dumps(row) + "\n")
                rows_out.flush()
                print(f"[re-eval] {ckpt_name} {split} pair {pair_no} done", flush=True)

    rows_out.close()
    print(f"wrote {out_jsonl}")


if __name__ == "__main__":
    main()
