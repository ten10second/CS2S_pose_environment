"""Assemble inference panels into an mp4 video with cv2.

Usage:
  python tools/assemble_panel_video.py --panel-dir <dir>/panels --out-video <dir>/seq.mp4 --fps 10
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel-dir", required=True)
    p.add_argument("--out-video", required=True)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--width", type=int, default=1152, help="output video width")
    args = p.parse_args()

    panels = sorted(Path(args.panel_dir).glob("*.png"))
    assert panels, f"no panels in {args.panel_dir}"
    print(f"{len(panels)} panels, e.g. {panels[0].name}")

    first = cv2.imread(str(panels[0]))
    h, w = first.shape[:2]
    scale = args.width / w
    out_w, out_h = args.width, int(round(h * scale / 2) * 2)
    writer = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    for path in panels:
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip unreadable {path}")
            continue
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()
    print(f"wrote {args.out_video} ({len(panels)} frames @ {args.fps} fps = {len(panels)/args.fps:.1f}s)")


if __name__ == "__main__":
    main()
