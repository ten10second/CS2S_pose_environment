#!/usr/bin/env python3
"""Plot all KITTI_location vehicle trajectories on one coordinate map."""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.KITTI_utils import gps2utm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", default="dataset/KITTI_location")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def split_frames(split_root):
    frames = set()
    for name in ("train_files.txt", "test1_files.txt", "test2_files.txt"):
        for line in (Path(split_root) / name).read_text().splitlines():
            if line.strip():
                frames.add(line.split()[0])
    return sorted(frames)


def read_trajectories(split_root, raw_root):
    trajectories = defaultdict(list)
    for relative_path in split_frames(split_root):
        date, drive, image_name = relative_path.split("/")
        frame_id = Path(image_name).stem
        oxts_path = Path(raw_root) / date / drive / "oxts" / "data" / f"{frame_id}.txt"
        if not oxts_path.exists():
            continue
        fields = oxts_path.read_text().split()
        if len(fields) < 2:
            continue
        utm_x, utm_y = gps2utm(float(fields[0]), float(fields[1]))
        trajectories[(date, drive)].append((int(frame_id), float(utm_x), float(utm_y)))
    return trajectories


def main():
    args = parse_args()
    trajectories = read_trajectories(args.split_root, args.raw_root)
    if not trajectories:
        raise SystemExit("No trajectories were loaded.")

    all_xy = np.asarray([(x, y) for points in trajectories.values() for _, x, y in points])
    origin = np.floor(all_xy.min(axis=0) / 1000.0) * 1000.0
    colors = plt.cm.turbo(np.linspace(0.02, 0.98, len(trajectories)))

    fig, ax = plt.subplots(figsize=(10, 12.5), constrained_layout=True)
    for color, (_, points) in zip(colors, sorted(trajectories.items())):
        points = sorted(points)
        xy = (np.asarray([(x, y) for _, x, y in points]) - origin) / 1000.0
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.25, alpha=0.9)

    span = (all_xy.max(axis=0) - all_xy.min(axis=0)) / 1000.0
    ax.set_title(
        f"KITTI Vehicle Trajectories ({len(trajectories)} Drives, {span[0]:.1f} x {span[1]:.1f} km)",
        fontsize=16,
        weight="bold",
        pad=12,
    )
    ax.set_xlabel("East-West distance (km)")
    ax.set_ylabel("North-South distance (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.6)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, facecolor="white")
    print(output.resolve())


if __name__ == "__main__":
    main()
