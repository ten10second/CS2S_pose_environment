"""Root-cause probe: is the history stream necessary, or merely redundant?

The Stage F result is that history-on and history-off score the same. Two very
different causes predict that, and they need different fixes:

  (a) REDUNDANT CONDITION — the current satellite condition already determines
      the appearance, so nothing in the training objective requires reading
      history. AdamW weight decay then drives the residual to zero. Fix: change
      the objective/mechanism so history is the only source of some information.
  (b) BROKEN READOUT — the correspondence mask is reduced to almost nothing at
      the injection resolution, or the residual is too small to matter. Fix:
      geometry resolution / injection point / readout capacity.

The decisive test is a 2x2: {history on, off} x {satellite conditioned,
satellite zeroed}. With the satellite zeroed the model has geometry only, which
is the Cyclops regime (their source is single-channel intensity): appearance can
only come from history. If history helps there, (a) is confirmed. If it helps
nowhere, (b) is confirmed.

Everything is paired: same pair, same timestep, same diffusion noise, same VAE
sampling for all eight cells. No training happens.

Usage (on the server, where the checkpoints live; single visible GPU):

  python tools/probe_history_necessity.py \
    --config <run>/base/cfg_run_config.yaml \
    --sd-base-ckpt /mnt/shizhm/BasicModel/checkpoints/sd-v1-4.ckpt \
    --ckpt <run>/base/cfg_step_250000.pt \
    --hist-ckpt <run>/stage_f_generator/geometry_history_step_1000.pt \
    --manifest <train_manifest.jsonl> --kitti-root /mnt/shizhm/DATA/KITTI/KITTI_RAW \
    --lidar-ray-feature-cache-root <...> --image-semantic-cache-root <...> \
    --block-indices after_bottleneck --history-dim 64 --heads 4 --dim-head 32 \
    --split train --pair-index 0 --timesteps 250,750 --out probe_necessity.json

Read `loss_eps_base` first (the denoising term every condition shares); the
total also carries the appearance x0 term, which is only meaningful when
--appearance-x0-weight matches the trained run.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate
from omegaconf import OmegaConf

TOOLS = Path(__file__).resolve().parent
for path in (str(TOOLS), str(TOOLS.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils.util import instantiate_from_config  # noqa: E402
from generate_kitti_raea_samples import load_checkpoint_into_model  # noqa: E402
from generate_kitti_geometry_history import (  # noqa: E402
    enable_geometry_history_attention,
    load_geometry_history_checkpoint,
    parse_block_indices,
)
from train_kitti_geometry_history import (  # noqa: E402
    GeometryPairs,
    GeometryTrainingStep,
    encode_history,
    fixed_probe,
    make_dataset,
    move_batch_to_device,
    split_drive_pairs,
)

CONDITIONS = ("disabled", "correct", "wrong_geometry", "wrong_history")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("sd-base-ckpt", "ckpt", "hist-ckpt", "manifest", "kitti-root",
                 "lidar-ray-feature-cache-root", "image-semantic-cache-root", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--config",
                   default="configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea_cfgdrop10.yaml")
    p.add_argument("--block-indices", type=parse_block_indices, default="after_bottleneck")
    p.add_argument("--history-dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dim-head", type=int, default=32)
    p.add_argument("--appearance-x0-weight", type=float, default=1.0,
                   help="must match the trained run's --appearance-x0-weight")
    p.add_argument("--split", choices=("train", "val"), default="train")
    p.add_argument("--pair-index", type=int, default=0,
                   help="index into the chosen partition; negative counts from the end")
    p.add_argument("--timesteps", default="250,750")
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--val-drives", default="")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def readout_stats(blocks, geometry):
    """Raw (16,64) coverage plus what the injection block actually read.

    The attention records both numbers on every has_history forward, so this is
    read after the probe and needs no extra pass.
    """
    valid = geometry["history_valid"].bool()
    raw = float(valid.float().mean())
    block = blocks[0].history_attn
    return {
        "raw_fraction_16x64": raw,
        "effective_fraction_at_block": block.last_valid_frac,
        "null_fraction_at_block": block.last_null_frac,
        "residual_to_condition": block.last_ratio,
    }


def main():
    args = parse_args()
    timesteps = [int(x) for x in args.timesteps.split(",")]
    cfg = OmegaConf.load(args.config)
    cfg.model.params.AE_ckpt_path = args.sd_base_ckpt
    cfg.model.params.pre_ldm_model_path = args.sd_base_ckpt

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip()]
    train, val, held = split_drive_pairs(
        rows, args.val_drives.split(",") if args.val_drives else None
    )
    dataset = make_dataset(args, cfg)
    partition = train if args.split == "train" else val
    index = int(args.pair_index)
    index = index if index >= 0 else len(partition) + index
    if not 0 <= index < len(partition):
        raise IndexError(f"pair index {index} outside {args.split} partition "
                         f"({len(partition)} pairs)")

    device = torch.device("cuda", 0)
    item = move_batch_to_device(
        default_collate([GeometryPairs(dataset, rows, [partition[index]], args.kitti_root)[0]]),
        device,
    )
    other = val[0] if args.split == "train" else train[0]
    wrong_item = move_batch_to_device(
        default_collate([GeometryPairs(dataset, rows, [other], args.kitti_root)[0]]), device
    )

    model = instantiate_from_config(cfg.model).cuda().eval()
    load_checkpoint_into_model(model, args.ckpt)
    hub, encoder, blocks = enable_geometry_history_attention(model, args)
    model = model.cuda().eval()
    encoder = encoder.cuda().eval()
    payload = load_geometry_history_checkpoint(args.hist_ckpt, args, encoder, blocks)
    module = GeometryTrainingStep(
        model, encoder, hub, appearance_x0_weight=args.appearance_x0_weight
    )
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)

    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(args.seed)
        latent = encode_history(model, item["prev"])
        wrong_latent = encode_history(model, wrong_item["prev"])

    geometry = {k: item[k] for k in ("history_grid", "history_valid")}
    results = fixed_probe(
        module, item["cur"], latent, geometry, wrong_latent, timesteps,
        args.seed, not args.no_amp, satellite_arms=(False, True),
    )
    coverage = readout_stats(blocks, geometry)

    by_cell = {(row["t"], row["satellite_blind"], row["condition"]): row for row in results}
    summary = []
    for t in timesteps:
        for blind in (False, True):
            cells = {name: by_cell.get((t, blind, name)) for name in CONDITIONS}
            if any(cell is None for cell in cells.values()):
                continue
            summary.append({
                "t": t,
                "satellite_blind": blind,
                "loss": {name: cells[name]["loss_total"] for name in CONDITIONS},
                "loss_eps_base": {name: cells[name].get("loss_eps_base") for name in CONDITIONS},
                "benefit_disabled_minus_correct":
                    cells["disabled"]["loss_total"] - cells["correct"]["loss_total"],
                "benefit_disabled_minus_correct_eps":
                    (cells["disabled"].get("loss_eps_base", 0.0)
                     - cells["correct"].get("loss_eps_base", 0.0)),
                "benefit_disabled_minus_wrong_geometry":
                    cells["disabled"]["loss_total"] - cells["wrong_geometry"]["loss_total"],
                "benefit_wrong_history_minus_correct":
                    cells["wrong_history"]["loss_total"] - cells["correct"]["loss_total"],
            })

    header = {"pair_id": item["pair_id"][0], "split": args.split, "pair_index": index,
              "history_step": payload.get("step"), "coverage": coverage,
              "val_drives": held}
    print(json.dumps(header, indent=2))
    print(f"\n{'t':>5} {'satellite':>10} {'disabled':>10} {'correct':>10} "
          f"{'wrongGeom':>10} {'wrongHist':>10} {'benefit':>10}")
    for entry in summary:
        loss = entry["loss"]
        print(f"{entry['t']:>5} {'zeroed' if entry['satellite_blind'] else 'on':>10} "
              f"{loss['disabled']:>10.4f} {loss['correct']:>10.4f} "
              f"{loss['wrong_geometry']:>10.4f} {loss['wrong_history']:>10.4f} "
              f"{entry['benefit_disabled_minus_correct']:>+10.4f}")
    print("\nbenefit > 0 means correct history beat the no-history baseline.")

    Path(args.out).write_text(json.dumps(
        {**header, "args": vars(args), "timesteps": summary, "rows": results},
        indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
