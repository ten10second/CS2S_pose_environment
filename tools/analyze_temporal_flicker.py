"""Temporal flicker comparison across noise modes.

For each run, compute mean consecutive-frame LPIPS over the generated sequence
and over the GT sequence of the same frames. The ratio gen/gt normalizes away
forward motion (GT itself changes between frames); ratio -> 1 means the
generated sequence evolves as smoothly as the real scene, higher = flicker.

Usage:
  python tools/analyze_temporal_flicker.py --device cuda:0
"""
import argparse
import json
from pathlib import Path

import torch
import lpips

R = Path("/mnt/shizhm/DATA/KITTI/CS2S_results/kitti_ray_posterior/ray_posterior_utonia_dino_sd14_4gpu_20260812/inference")

RUNS = {
    "drive0057_58f per_frame (baseline)": "contiguous_seq_drive0057_58f",
    "drive0057_58f shared": "contiguous_seq_drive0057_58f_sharednoise",
    "drive0057_58f autoreg_s07": "contiguous_seq_drive0057_58f_autoreg_s07",
    "drive0057_58f posewarp_s05": "contiguous_seq_drive0057_58f_posewarp_s05",
    "drive0020_300f per_frame (baseline)": "contiguous_seq_drive0020_300f",   # sharded
    "drive0020_300f autoreg_s07": "contiguous_seq_drive0020_300f_autoreg_s07",
    "drive0020_300f posewarp_s05": "contiguous_seq_drive0020_300f_posewarp_s05",
    "drive0020_300f posewarp2_s05": "contiguous_seq_drive0020_300f_posewarp2_s05",
    "drive0020_300f instance_s05": "contiguous_seq_drive0020_300f_instance_s05",
}


def collect_pairs(run_dir: Path):
    """Sorted consecutive (img_t, img_t+1) pairs for gt and gen."""
    if (run_dir / "images" / "gt").is_dir():
        gen_dir = run_dir / "images" / "normal"
        gt_dir = run_dir / "images" / "gt"
    else:  # sharded layout
        gen_imgs, gt_imgs = [], []
        for shard in sorted(run_dir.glob("shard*")):
            gen_imgs.extend((shard / "images" / "normal").glob("*.png"))
        gen_dir = None
        gt_dir = None
        gen_imgs = sorted(gen_imgs, key=lambda p: p.name)
        names = [p.name for p in gen_imgs]
        gt_imgs = []
        for shard in sorted(run_dir.glob("shard*")):
            gt_imgs.extend((shard / "images" / "gt").glob("*.png"))
        gt_by_name = {p.name: p for p in gt_imgs}
        gt_seq = [gt_by_name[n] for n in names if n in gt_by_name]
        return gen_imgs, gt_seq
    gen_seq = sorted(gen_dir.glob("*.png"), key=lambda p: p.name)
    gt_by_name = {p.name: p for p in gt_dir.glob("*.png")}
    gt_seq = [gt_by_name[p.name] for p in gen_seq if p.name in gt_by_name]
    return gen_seq, gt_seq


def to_tensor(p, device):
    from PIL import Image
    import numpy as np
    arr = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def consecutive_lpips(seq, lpips_fn, device, batch=8):
    vals = []
    for i in range(len(seq) - 1):
        a = to_tensor(seq[i], device)
        b = to_tensor(seq[i + 1], device)
        vals.append(float(lpips_fn(a * 2 - 1, b * 2 - 1).item()))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    results = {}
    for label, run in RUNS.items():
        run_dir = R / run
        gen_seq, gt_seq = collect_pairs(run_dir)
        if len(gen_seq) < 2:
            print(f"skip {label}: not enough frames")
            continue
        t_gen = consecutive_lpips(gen_seq, lpips_fn, device)
        t_gt = consecutive_lpips(gt_seq, lpips_fn, device)
        import numpy as np
        mg, mt = float(np.mean(t_gen)), float(np.mean(t_gt))
        results[label] = {
            "run": run,
            "frames": len(gen_seq),
            "tlpips_gen": mg,
            "tlpips_gt": mt,
            "flicker_ratio": mg / mt,
            "tlpips_gen_std": float(np.std(t_gen)),
        }
        print(f"{label:42s} n={len(gen_seq):4d}  tLPIPS_gen={mg:.4f}  tLPIPS_gt={mt:.4f}  ratio={mg/mt:.3f}")

    out = R / "temporal_flicker_analysis.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
