"""Region-decomposed temporal flicker: where does frame-to-frame change live?

Hypothesis (satellite-anchoring proposal): the satellite condition already
stabilizes the static world, so generated flicker concentrates in dynamic /
disoccluded regions. Method: reuse the LiDAR cross-frame consistency signal
(pose_warp_utils) to split each consecutive frame pair's pixels into
  static    — previous-frame LiDAR returns confirmed by the current scan
  dynamic   — returns whose surface is gone (moved object / disocclusion)
  uncovered — pixels without any previous-frame LiDAR support
then compute mean LPIPS(gen_t, gen_t+1) and LPIPS(gt_t, gt_t+1) inside
half-resolution masks of each class, per pair, and report the gen/gt ratio
per region, averaged over the clip.

Usage:
  python tools/analyze_region_flicker.py \
      --run-a <gen dir with images/normal> --gt-dir <gt dir> \
      --manifest <test manifest> --start-index 2047 --num-frames 300 \
      --kitti-root ... --calib-dir ... --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parent
for p in (str(TOOLS_DIR), str(TOOLS_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pose_warp_utils as pwu  # noqa: E402


def rebase(path, kitti_root):
    path = str(path)
    marker = "KITTI_RAW/"
    i = path.find(marker)
    return str(Path(kitti_root) / path[i + len(marker):]) if i >= 0 and kitti_root else path


def region_masks(row_prev, row_cur, kitti_root, geom, H=128, W=512):
    p1 = pwu.load_velodyne(rebase(row_prev["velodyne_path"], kitti_root))
    p2 = pwu.load_velodyne(rebase(row_cur["velodyne_path"], kitti_root))
    T_v = geom.relative_velo_pose(
        rebase(row_prev["oxts_path"], kitti_root), rebase(row_cur["oxts_path"], kitti_root)
    )
    q = (T_v[:3, :3] @ p1.T).T + T_v[:3, 3]
    status = pwu.consistency_status(q, p2)
    u, v, d = geom.project_velo_to_rect_img(q)
    su = (u * W / geom.img_size[0]).astype(int)
    sv = (v * H / geom.img_size[1]).astype(int)
    inb = (d > 1.0) & (su >= 0) & (su < W) & (sv >= 0) & (sv < H)

    static = np.zeros((H, W), bool)
    dynamic = np.zeros((H, W), bool)
    static[sv[inb & (status == 1)], su[inb & (status == 1)]] = True
    dynamic[sv[inb & (status == 2)], su[inb & (status == 2)]] = True

    from scipy import ndimage as ndi
    struct = np.ones((5, 5), bool)
    static_d = ndi.binary_dilation(static, structure=struct)
    dynamic_d = ndi.binary_dilation(dynamic, structure=struct)
    dynamic_only = dynamic_d & ~static_d
    uncovered = ~(static_d | dynamic_d)
    return {
        "static": static_d,
        "dynamic": dynamic_only,
        "uncovered": uncovered,
    }


def load_tensor(path, device):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def masked_lpips(a, b, mask_bool, lpips_fn):
    """mask_bool: (H, W) numpy. Returns LPIPS restricted to the mask by
    evaluating full-image LPIPS on masked-out-zeroed pairs (spatial variant)."""
    m = torch.from_numpy(mask_bool.astype(np.float32))[None, None].to(a.device)
    am, bm = a * m, b * m
    with torch.no_grad():
        val = lpips_fn(am * 2 - 1, bm * 2 - 1)
    area = float(m.mean())
    return float(val.item()), area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="dir containing images/normal")
    ap.add_argument("--gt-dir", required=True, help="dir containing gt pngs")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--start-index", type=int, required=True)
    ap.add_argument("--num-frames", type=int, default=300)
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)][
        args.start_index : args.start_index + args.num_frames
    ]
    for r in rows:
        for k in ("velodyne_path", "oxts_path"):
            r[k] = rebase(r[k], args.kitti_root)

    import lpips
    device = torch.device(args.device)
    lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(device).eval()

    gen_dir = Path(args.run_a) / "images" / "normal"
    gt_dir = Path(args.gt_dir)
    geom = pwu.SequenceGeometry(args.calib_dir)

    acc = {r: {"gen": [], "gt": []} for r in ("static", "dynamic", "uncovered", "full")}
    pairs_used = 0
    for i in range(len(rows) - 1):
        name_cur = f"{rows[i+1]['date']}__{rows[i+1]['drive']}__{rows[i+1]['frame_id']}.png"
        g_cur, g_next = gen_dir / f"{rows[i]['date']}__{rows[i]['drive']}__{rows[i]['frame_id']}.png", gen_dir / name_cur
        t_cur, t_next = gt_dir / f"{rows[i]['date']}__{rows[i]['drive']}__{rows[i]['frame_id']}.png", gt_dir / name_cur
        if not (g_cur.exists() and g_next.exists() and t_cur.exists() and t_next.exists()):
            continue
        masks = region_masks(rows[i], rows[i + 1], args.kitti_root, geom)
        for tag, (pa, pb) in {
            "gen": (g_cur, g_next),
            "gt": (t_cur, t_next),
        }.items():
            a, b = load_tensor(pa, device), load_tensor(pb, device)
            with torch.no_grad():
                spatial = lpips_fn(a * 2 - 1, b * 2 - 1)  # (1,1,H,W)
            spatial = spatial[0, 0].cpu().numpy()
            for rname, m in masks.items():
                acc[rname][tag].append(float(spatial[m].mean()) if m.any() else np.nan)
            acc["full"][tag].append(float(spatial.mean()))
        pairs_used += 1
        if pairs_used % 40 == 0:
            print(f"pairs {pairs_used}", flush=True)

    result = {"pairs": pairs_used}
    print(f"\n{'region':<10} {'gen dLPIPS':>10} {'gt dLPIPS':>10} {'ratio':>7}")
    for rname in ("static", "dynamic", "uncovered", "full"):
        g = np.nanmean(acc[rname]["gen"])
        t = np.nanmean(acc[rname]["gt"])
        result[rname] = {"gen": float(g), "gt": float(t), "ratio": float(g / t)}
        print(f"{rname:<10} {g:>10.4f} {t:>10.4f} {g/t:>7.3f}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
