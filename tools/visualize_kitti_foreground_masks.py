import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_raw_lidar_utils import read_jsonl  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize KITTI foreground mask cache overlays.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--suffix", default="_foreground.png")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--metadata", default="")
    return parser.parse_args()


def safe_sample_id(sample_id):
    return sample_id.replace("/", "__")


def overlay_mask(image, mask):
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    mask_arr = (np.asarray(mask.convert("L")) > 0)[..., None]
    color = np.zeros_like(rgb)
    color[..., 1] = 255.0
    out = np.where(mask_arr, rgb * 0.45 + color * 0.55, rgb)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def draw_instances(image, instances):
    draw = ImageDraw.Draw(image)
    for row in instances:
        xyxy = row.get("xyxy")
        if not xyxy:
            continue
        cls = row.get("class", "fg")
        score = row.get("sam2_score", row.get("score", 0.0))
        draw.rectangle(xyxy, outline=(255, 60, 40), width=2)
        draw.text((xyxy[0] + 2, max(0, xyxy[1] - 12)), f"{cls} {score:.2f}", fill=(255, 60, 40))
    return image


def main():
    args = parse_args()
    records = read_jsonl(args.manifest)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    mask_root = Path(args.mask_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if args.metadata:
        for line in Path(args.metadata).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            metadata[row["sample_id"]] = row
    for idx, record in enumerate(records):
        image = Image.open(record["image_02_path"]).convert("RGB")
        mask_path = mask_root / f"{safe_sample_id(record['sample_id'])}{args.suffix}"
        if not mask_path.is_file():
            continue
        mask = Image.open(mask_path).convert("L").resize(image.size, Image.NEAREST)
        over = overlay_mask(image, mask)
        over = draw_instances(over, metadata.get(record["sample_id"], {}).get("instances", []))
        canvas = Image.new("RGB", (image.width * 2, image.height + 24), "white")
        canvas.paste(image, (0, 24))
        canvas.paste(over, (image.width, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), "GT", fill=(0, 0, 0))
        draw.text((image.width + 6, 5), "foreground mask overlay", fill=(0, 0, 0))
        out_path = out_dir / f"{safe_sample_id(record['sample_id'])}.png"
        canvas.save(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
