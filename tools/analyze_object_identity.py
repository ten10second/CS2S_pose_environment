"""Object appearance-identity metric across consecutive frames.

For object instances matched by the LiDAR associator (instance run's
records.json carries per-frame oid + bbox), crop the SAME box from each
sampling mode's generated frames and from GT, and measure CLIP cosine between
consecutive frames. Same box across modes isolates appearance identity from
box placement. "shifted" reference crops a displaced box: unrelated-content
similarity floor.

Usage:
  python tools/analyze_object_identity.py --device cuda:0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

R = Path("/mnt/shizhm/DATA/KITTI/CS2S_results/kitti_ray_posterior/ray_posterior_utonia_dino_sd14_4gpu_20260812/inference")
INSTANCE_RUN = R / "contiguous_seq_drive0020_300f_instance_s05"
MODES = {
    "GT": R / "contiguous_seq_drive0020_300f_instance_s05/images/gt",
    "instance": R / "contiguous_seq_drive0020_300f_instance_s05/images/normal",
    "warp2": R / "contiguous_seq_drive0020_300f_posewarp2_s05/images/normal",
    "autoreg": R / "contiguous_seq_drive0020_300f_autoreg_s07/images/normal",
}
SHARD_BASELINE = R / "contiguous_seq_drive0020_300f"  # sharded per-frame baseline


def baseline_path(name):
    for shard in sorted(SHARD_BASELINE.glob("shard*")):
        p = shard / "images" / "normal" / name
        if p.exists():
            return p
    return None


def load_pairs(max_gap=3):
    """[(name_t, name_t1, oid, bbox)] where oid appears within max_gap frames."""
    recs = json.loads((INSTANCE_RUN / "records.json").read_text())
    names = [Path(r["GT"]).name for r in recs]
    by_name = {Path(r["GT"]).name: r for r in recs}
    pairs = []
    for i, a in enumerate(names):
        ra = by_name[a]
        oa = {o["oid"]: o["bbox"] for o in ra["noise"].get("objects", [])}
        for j in range(i + 1, min(i + 1 + max_gap, len(names))):
            b = names[j]
            ob = {o["oid"]: o["bbox"] for o in by_name[b]["noise"].get("objects", [])}
            for oid in set(oa) & set(ob):
                pairs.append((a, b, oid, oa[oid]))
    return pairs


def crop_clip(img: Image.Image, bbox, shift=(0, 0), size=64):
    import torch.nn.functional as F
    # association boxes are in full-res KITTI coords (1242x375); saved frames
    # are the dataset resolution (512x128).
    sx, sy = img.width / 1242.0, img.height / 375.0
    x1, y1, x2, y2 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
    x1 += shift[0] * sx; x2 += shift[0] * sx; y1 += shift[1] * sy; y2 += shift[1] * sy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    c = img.crop((int(x1), int(y1), int(x2), int(y2))).resize((size, size))
    arr = torch.from_numpy(np.asarray(c).astype(np.float32) / 255.0).permute(2, 0, 1)
    return (arr - 0.5) / 0.5


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    import clip
    model, _ = clip.load("ViT-B/32", device=str(device))
    model = model.eval()

    pairs = load_pairs()
    print(f"object pairs (same oid within gap<=3 frames): {len(pairs)}")
    gt_dir = MODES["GT"]

    def encode(crop):
        x = crop.unsqueeze(0).to(device)
        f = model.encode_image(x)
        return f / f.norm(dim=-1, keepdim=True)

    sims = {m: [] for m in list(MODES) + ["baseline_per_frame", "shifted_ref"]}
    for name_a, name_b, oid, bbox in pairs:
        for mode, mdir in MODES.items():
            ia = Image.open(mdir / name_a).convert("RGB")
            ib = Image.open(mdir / name_b).convert("RGB")
            ca, cb = crop_clip(ia, bbox), crop_clip(ib, bbox)
            if ca is None or cb is None:
                continue
            sims[mode].append(float((encode(ca) * encode(cb)).sum()))
        bname = Path(name_a).name
        ba, bb = baseline_path(bname), baseline_path(Path(name_b).name)
        if ba and bb:
            ca, cb = crop_clip(Image.open(ba).convert("RGB"), bbox), crop_clip(Image.open(bb).convert("RGB"), bbox)
            if ca is not None and cb is not None:
                sims["baseline_per_frame"].append(float((encode(ca) * encode(cb)).sum()))
        ia = Image.open(gt_dir / name_a).convert("RGB")
        w = bbox[2] - bbox[0]
        ca = crop_clip(ia, bbox)
        cb = crop_clip(Image.open(gt_dir / name_b).convert("RGB"), bbox, shift=(int(w * 1.5), 0))
        if ca is not None and cb is not None:
            sims["shifted_ref"].append(float((encode(ca) * encode(cb)).sum()))

    result = {}
    for m, vals in sims.items():
        if vals:
            result[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
            print(f"{m:20s} mean={np.mean(vals):.4f} std={np.std(vals):.4f} n={len(vals)}")
    out = R / "object_identity_analysis.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
