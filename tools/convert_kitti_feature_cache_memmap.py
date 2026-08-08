import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np


POINT_FEATURE_KEY = "utonia_feat"
POINT_MASK_KEY = "lidar_point_features_mask"
IMAGE_FEATURE_KEY = "dino_feat"
IMAGE_MASK_KEY = "image_semantic_mask"
RAY_FEATURE_KEY = "utonia_ray_feat"
RAY_MASK_KEY = "utonia_ray_mask"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert KITTI feature NPZ files to float16 NPY memmaps.")
    parser.add_argument("--kind", choices=["point", "image", "ray"], required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--point-dim", type=int, default=576)
    parser.add_argument("--image-channels", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=32)
    parser.add_argument("--ray-depth-bins", type=int, default=4)
    parser.add_argument("--ray-height", type=int, default=8)
    parser.add_argument("--ray-width", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def required_array(payload, key):
    if key not in payload:
        raise KeyError(f"Required cache array is missing: {key}")
    return np.asarray(payload[key])


def normalize_point(payload, feature_shape, mask_shape):
    features = required_array(payload, POINT_FEATURE_KEY)
    mask = required_array(payload, POINT_MASK_KEY)
    if tuple(features.shape) != tuple(feature_shape):
        raise ValueError(f"Utonia feature shape {features.shape} != expected {feature_shape}")
    if tuple(mask.shape) != tuple(mask_shape):
        raise ValueError(f"Utonia mask shape {mask.shape} != expected {mask_shape}")
    return features.astype(np.float16), (mask > 0.5).astype(np.uint8)


def normalize_image(payload, feature_shape, mask_shape):
    channels, height, width = feature_shape
    features_hwc = required_array(payload, IMAGE_FEATURE_KEY)
    mask_hw = required_array(payload, IMAGE_MASK_KEY)
    expected_hwc = (height, width, channels)
    expected_hw = (height, width)
    if tuple(features_hwc.shape) != expected_hwc:
        raise ValueError(f"DINO feature shape {features_hwc.shape} != expected {expected_hwc}")
    if tuple(mask_hw.shape) != expected_hw:
        raise ValueError(f"DINO mask shape {mask_hw.shape} != expected {expected_hw}")
    features = features_hwc.transpose(2, 0, 1).astype(np.float16)
    mask = (mask_hw[None] > 0.5).astype(np.uint8)
    if tuple(mask.shape) != tuple(mask_shape):
        raise ValueError(f"DINO converted mask shape {mask.shape} != expected {mask_shape}")
    return features, mask


def normalize_ray(payload, feature_shape, mask_shape):
    features = required_array(payload, RAY_FEATURE_KEY)
    mask = required_array(payload, RAY_MASK_KEY)
    if tuple(features.shape) != tuple(feature_shape):
        raise ValueError(f"Utonia ray feature shape {features.shape} != expected {feature_shape}")
    if tuple(mask.shape) != tuple(mask_shape):
        raise ValueError(f"Utonia ray mask shape {mask.shape} != expected {mask_shape}")
    return features.astype(np.float16), (mask > 0.5).astype(np.uint8)


def load_and_normalize(path, normalize, feature_shape, mask_shape):
    with np.load(path, allow_pickle=False) as payload:
        return normalize(payload, feature_shape, mask_shape)


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    files = sorted(source_root.glob("*.npz"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {source_root}")

    if args.kind == "point":
        feature_shape = (int(args.point_count), int(args.point_dim))
        mask_shape = (int(args.point_count),)
        normalize = normalize_point
    elif args.kind == "image":
        feature_shape = (int(args.image_channels), int(args.image_height), int(args.image_width))
        mask_shape = (1, int(args.image_height), int(args.image_width))
        normalize = normalize_image
    else:
        feature_shape = (
            int(args.point_dim),
            int(args.ray_depth_bins),
            int(args.ray_height),
            int(args.ray_width),
        )
        mask_shape = (1, int(args.ray_depth_bins), int(args.ray_height), int(args.ray_width))
        normalize = normalize_ray

    features_path = output_root / "features.npy"
    masks_path = output_root / "masks.npy"
    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(files), *feature_shape),
    )
    masks = np.lib.format.open_memmap(
        masks_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(files), *mask_shape),
    )

    started = time.time()
    index = {}
    convert = partial(
        load_and_normalize,
        normalize=normalize,
        feature_shape=feature_shape,
        mask_shape=mask_shape,
    )
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for row, (path, converted) in enumerate(zip(files, executor.map(convert, files))):
            feature_row, mask_row = converted
            features[row] = feature_row
            masks[row] = mask_row
            index[path.stem] = row
            if row == 0 or (row + 1) % max(1, int(args.progress_every)) == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "kind": args.kind,
                            "rows": row + 1,
                            "total": len(files),
                            "rows_per_second": (row + 1) / elapsed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    features.flush()
    masks.flush()
    meta = {
        "format": "kitti_feature_memmap_v1",
        "kind": args.kind,
        "count": len(files),
        "feature_shape": list(feature_shape),
        "mask_shape": list(mask_shape),
        "feature_dtype": "float16",
        "mask_dtype": "uint8",
        "features_file": features_path.name,
        "masks_file": masks_path.name,
        "index": index,
        "source_root": str(source_root),
    }
    (output_root / "memmap_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps({"complete": True, "output_root": str(output_root), **meta}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
