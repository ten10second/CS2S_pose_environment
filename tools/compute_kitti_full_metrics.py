"""Compute generation metrics over sharded full-test inference outputs.

Metrics:
  - FID / KID (torchvision ImageNet InceptionV3 pool-2048 features, 299px)
  - LPIPS (alexnet) per pair
  - CLIP image-image cosine (ViT-B/32) per pair
  - PSNR / SSIM per pair (reference only; diffusion samples are not pixel-aligned)
  - Ray-posterior geometry stats aggregated from records.json

Usage:
  python tools/compute_kitti_full_metrics.py \
      --run-root <run>/inference/full_test_500k --out-json <run>/metrics_full_test_500k.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionFeatures(torch.nn.Module):
    """2048-d pool features from torchvision's ImageNet InceptionV3."""

    def __init__(self, device):
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)

    @torch.no_grad()
    def forward(self, x):
        x = (x - 0.5) / 0.5
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        return self.model(x)


def frechet_distance(mu1, sig1, mu2, sig2, eps=1e-6):
    from scipy import linalg
    diff = mu2 - mu1
    covmean, _ = linalg.sqrtm(sig1.dot(sig2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sig1.shape[0]) * eps
        covmean = linalg.sqrtm((sig1 + offset).dot(sig2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2.0 * np.trace(covmean))


def polynomial_mmd(X, Y, gamma=None, degree=3, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K_XX = (X @ X.T * gamma + coef0) ** degree
    K_YY = (Y @ Y.T * gamma + coef0) ** degree
    K_XY = (X @ Y.T * gamma + coef0) ** degree
    m, n = X.shape[0], Y.shape[0]
    sum_XX = (K_XX.sum() - np.trace(K_XX)) / (m * (m - 1))
    sum_YY = (K_YY.sum() - np.trace(K_YY)) / (n * (n - 1))
    sum_XY = K_XY.mean()
    return sum_XX + sum_YY - 2.0 * sum_XY


def kid_score(real_feats, fake_feats, num_subsets=100, max_subset_size=1000, seed=0):
    rng = np.random.RandomState(seed)
    subset_size = min(max_subset_size, len(real_feats), len(fake_feats))
    vals = []
    for _ in range(num_subsets):
        r = real_feats[rng.choice(len(real_feats), subset_size, replace=False)]
        f = fake_feats[rng.choice(len(fake_feats), subset_size, replace=False)]
        vals.append(polynomial_mmd(r, f))
    return float(np.mean(vals)), float(np.std(vals))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--kid-subsets", type=int, default=100)
    return p.parse_args()


def collect_pairs(run_root: Path):
    pairs = []
    for shard in sorted(run_root.glob("shard*")):
        gt_dir = shard / "images" / "gt"
        gen_dir = shard / "images" / "normal"
        if not gt_dir.is_dir():
            continue
        for gt_path in sorted(gt_dir.glob("*.png")):
            gen_path = gen_dir / gt_path.name
            if gen_path.exists():
                pairs.append((str(gt_path), str(gen_path)))
    return pairs


def load_batch(paths, resize299, device):
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        out.append(resize299(img))
    return torch.stack(out).to(device)


def main():
    args = parse_args()
    run_root = Path(args.run_root)
    device = torch.device(args.device)

    pairs = collect_pairs(run_root)
    print(f"pairs: {len(pairs)}")
    assert pairs, "no gt/gen pairs found"

    resize299 = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # ---------- FID / KID via torchvision InceptionV3 features ----------
    extractor = InceptionFeatures(device)

    def extract(kind):
        feats = []
        with torch.no_grad():
            for i in range(0, len(pairs), args.batch):
                chunk = pairs[i : i + args.batch]
                paths = [c[0 if kind == "real" else 1] for c in chunk]
                feats.append(extractor(load_batch(paths, resize299, device)).detach().cpu().numpy())
                if (i // args.batch) % 20 == 0:
                    print(f"inception {kind}: {i + len(chunk)}/{len(pairs)}", flush=True)
        return np.concatenate(feats, axis=0)

    real_feats = extract("real")
    fake_feats = extract("fake")
    mu_r, mu_f = real_feats.mean(0), fake_feats.mean(0)
    sig_r = np.cov(real_feats, rowvar=False)
    sig_f = np.cov(fake_feats, rowvar=False)
    fid_score = frechet_distance(mu_r, sig_r, mu_f, sig_f)
    kid_mean, kid_std = kid_score(real_feats, fake_feats,
                                  num_subsets=args.kid_subsets, max_subset_size=1000)
    del extractor
    torch.cuda.empty_cache()

    # ---------- LPIPS / CLIP / PSNR / SSIM on pairs ----------
    import lpips as lpips_pkg
    import clip as clip_pkg
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
    from torchmetrics.image.psnr import PeakSignalNoiseRatio

    lpips_fn = lpips_pkg.LPIPS(net="alex").to(device).eval()
    clip_model, _ = clip_pkg.load("ViT-B/32", device=str(device))
    clip_model = clip_model.eval()
    clip_pre = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)

    lpips_vals, clip_vals, per_drive = [], [], {}
    with torch.no_grad():
        for i in range(0, len(pairs), args.batch):
            chunk = pairs[i : i + args.batch]
            gts = torch.stack([to_tensor(Image.open(c[0]).convert("RGB")) for c in chunk]).to(device)
            gens = torch.stack([to_tensor(Image.open(c[1]).convert("RGB")) for c in chunk]).to(device)

            lp = lpips_fn(gens * 2 - 1, gts * 2 - 1).flatten()
            lpips_vals.extend(lp.tolist())

            cg = clip_model.encode_image(clip_pre(gts))
            cf = clip_model.encode_image(clip_pre(gens))
            cg = F.normalize(cg, dim=-1); cf = F.normalize(cf, dim=-1)
            clip_cos = (cg * cf).sum(-1)
            clip_vals.extend(clip_cos.tolist())

            ssim_fn.update(gens, gts)
            psnr_fn.update(gens, gts)

            for sample_idx, (gt_path, gen_path) in enumerate(chunk):
                drive = Path(gen_path).name.split("__")[1] if "__" in Path(gen_path).name else "unknown"
                per_drive.setdefault(drive, []).append((float(lp[sample_idx]), float(clip_cos[sample_idx])))

            if (i // args.batch) % 20 == 0:
                print(f"pairwise: {i + len(chunk)}/{len(pairs)}", flush=True)

    ssim_score = float(ssim_fn.compute())
    psnr_score = float(psnr_fn.compute())

    # ---------- records.json aggregation ----------
    agg_keys = [
        "ray_posterior_depth_log_error_mean",
        "ray_posterior_hit_coverage_mean",
        "ray_posterior_lidar_confidence_mean",
        "ray_posterior_lidar_message_ratio_mean",
        "ray_posterior_prior_entropy_mean",
        "ray_posterior_entropy_mean",
        "ray_posterior_weight_shift_mean",
        "lidar_attn_entropy_norm_mean",
        "lidar_attn_max_mean",
    ]
    agg = {k: [] for k in agg_keys}
    for shard in sorted(run_root.glob("shard*")):
        rec_path = shard / "records.json"
        if not rec_path.exists():
            continue
        records = json.loads(rec_path.read_text())
        for rec in records:
            attn = (rec.get("lidar_attention_stats", {})
                        .get("trained:normal", {})
                        .get("attention", {}))
            for k in agg_keys:
                if k in attn:
                    agg[k].append(float(attn[k]))

    result = {
        "num_pairs": len(pairs),
        "fid": fid_score,
        "kid_mean": kid_mean,
        "kid_std": kid_std,
        "lpips_mean": float(np.mean(lpips_vals)),
        "lpips_std": float(np.std(lpips_vals)),
        "clip_i_cosine_mean": float(np.mean(clip_vals)),
        "clip_i_cosine_std": float(np.std(clip_vals)),
        "ssim": ssim_score,
        "psnr": psnr_score,
        "ray_posterior_agg": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                              for k, v in agg.items() if v},
        "per_drive_lpips_clip": {d: {"lpips": float(np.mean([x[0] for x in v])),
                                     "clip": float(np.mean([x[1] for x in v])),
                                     "n": len(v)} for d, v in per_drive.items()},
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_drive_lpips_clip"}, indent=2))


to_tensor = transforms.ToTensor()


if __name__ == "__main__":
    main()
