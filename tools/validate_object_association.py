"""Offline validation of LiDAR cross-frame object association.

Runs the association over consecutive frame pairs of one drive, prints
per-pair stats, and saves debug overlays (matched clusters projected onto the
GT image, colored by persistent object id).

Usage:
  python tools/validate_object_association.py \
      --manifest <test_manifest.jsonl> --kitti-root /mnt/.../KITTI_RAW \
      --start-index 2047 --num-pairs 30 [--save-overlays]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
for p in (str(TOOLS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pose_warp_utils as pwu  # noqa: E402
import lidar_object_association as loa  # noqa: E402

COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 190), (0, 128, 128), (220, 190, 255),
]


def rebase(path, kitti_root):
    path = str(path)
    marker = "KITTI_RAW/"
    i = path.find(marker)
    return str(Path(kitti_root) / path[i + len(marker):]) if i >= 0 and kitti_root else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--start-index", type=int, required=True)
    ap.add_argument("--num-pairs", type=int, default=30)
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--out-dir", default="assoc_validation")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.manifest)]
    rows = rows[args.start_index : args.start_index + args.num_pairs + 1]
    for r in rows:
        for key in ("velodyne_path", "oxts_path", "calib_dir", "image_02_path"):
            r[key] = rebase(r[key], args.kitti_root)

    geom = pwu.SequenceGeometry(args.calib_dir)
    tracker = loa.ObjectTracker()
    img_w, img_h = geom.img_size

    stats_all = []
    for i in range(len(rows) - 1):
        prev, cur = rows[i], rows[i + 1]
        p1 = pwu.load_velodyne(prev["velodyne_path"])
        p2 = pwu.load_velodyne(cur["velodyne_path"])
        T_v = geom.relative_velo_pose(prev["oxts_path"], cur["oxts_path"])
        n_v, d_v = pwu.fit_ground_plane_velo(p1)
        plane = None if n_v is None else (-n_v[0] / n_v[2], -n_v[1] / n_v[2], d_v / n_v[2])
        matched, stats = loa.associate_objects(
            p1, p2, T_v, geom, img_w, img_h, 16, 64, plane=plane
        )
        ids = tracker.update(matched)
        stats["frame"] = cur["frame_index"]
        stats["motion_norms"] = [round(float(np.linalg.norm(o["d_velo"])), 2) for o in matched]
        stats["oids"] = ids
        stats_all.append(stats)
        print(
            f"frame {cur['frame_index']}: cur_objs={stats['n_cur_obj']:3d} "
            f"matched={stats['n_matched']:2d} unexplained={stats['unexplained_frac']:.3f} "
            f"motions={stats['motion_norms']}"
        )

        if args.save_overlays and matched:
            img_path = Path(cur["image_02_path"]) / f"{cur['frame_id']}.png"
            if not img_path.exists():
                img_path = Path(cur["image_02_path"])
            base = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(base)
            for oid, obj in zip(ids, matched):
                color = COLORS[oid % len(COLORS)]
                x1, y1, x2, y2 = obj["bbox"]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                draw.text((x1 + 2, max(0, y1 - 12)), f"id{oid}", fill=color)
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            base.save(out / f"overlay_{cur['frame_index']}.png")

    n = len(stats_all)
    total_cur = sum(s["n_cur_obj"] for s in stats_all)
    total_match = sum(s["n_matched"] for s in stats_all)
    print(f"\nsummary: {n} pairs, {total_cur} cur clusters, {total_match} matched "
          f"({100.0 * total_match / max(total_cur, 1):.1f}%), "
          f"unexplained_frac mean={np.mean([s['unexplained_frac'] for s in stats_all]):.3f}, "
          f"clusters/pair mean={total_cur / n:.1f}")


if __name__ == "__main__":
    main()
