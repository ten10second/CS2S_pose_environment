import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np


POINT_FEATURE_KEYS = ("lidar_point_features", "utonia_feat", "point_features", "features", "feat")
POINT_MASK_KEYS = ("lidar_point_features_mask", "point_feature_mask", "point_mask", "mask")
IMAGE_FEATURE_KEYS = ("image_semantic_feat", "dino_feat", "clip_feat", "features", "feat")
IMAGE_MASK_KEYS = ("image_semantic_mask", "semantic_mask", "valid_mask", "mask")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert KITTI feature NPZ files to float16 NPY memmaps.")
    parser.add_argument("--kind", choices=["point", "image"], required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--point-count", type=int, default=4096)
    parser.add_argument("--point-dim", type=int, default=576)
    parser.add_argument("--image-channels", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def first_array(payload, keys):
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def normalize_point(payload, feature_shape, mask_shape):
    features = np.zeros(feature_shape, dtype=np.float16)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cached = first_array(payload, POINT_FEATURE_KEYS)
    if cached is None:
        return features, mask
    cached = np.asarray(cached)
    if cached.ndim == 1:
        cached = cached[:, None]
    elif cached.ndim > 2:
        cached = cached.reshape(cached.shape[0], -1)
    point_count = min(feature_shape[0], cached.shape[0])
    channel_count = min(feature_shape[1], cached.shape[1])
    features[:point_count, :channel_count] = cached[:point_count, :channel_count].astype(np.float16)
    cached_mask = first_array(payload, POINT_MASK_KEYS)
    if cached_mask is None:
        mask[:point_count] = 1
    else:
        cached_mask = np.asarray(cached_mask).reshape(-1)
        mask_count = min(point_count, cached_mask.shape[0])
        mask[:mask_count] = cached_mask[:mask_count] > 0.5
    return features, mask


def normalize_image(payload, feature_shape, mask_shape):
    channels, height, width = feature_shape
    features = np.zeros(feature_shape, dtype=np.float16)
    mask = np.zeros(mask_shape, dtype=np.uint8)
    cached = first_array(payload, IMAGE_FEATURE_KEYS)
    if cached is None:
        return features, mask
    cached = np.asarray(cached)
    if cached.ndim == 2:
        if cached.shape[0] == height * width:
            cached = cached.reshape(height, width, cached.shape[1]).transpose(2, 0, 1)
        elif cached.shape[1] == height * width:
            cached = cached.reshape(cached.shape[0], height, width)
        else:
            cached = cached.reshape(cached.shape[0], -1, 1)
    elif cached.ndim == 3 and (cached.shape[-1] == channels or cached.shape[:2] == (height, width)):
        cached = cached.transpose(2, 0, 1)
    elif cached.ndim > 3:
        cached = cached.reshape(cached.shape[0], cached.shape[-2], cached.shape[-1])
    channel_count = min(channels, cached.shape[0])
    height_count = min(height, cached.shape[1])
    width_count = min(width, cached.shape[2])
    features[:channel_count, :height_count, :width_count] = cached[
        :channel_count, :height_count, :width_count
    ].astype(np.float16)
    cached_mask = first_array(payload, IMAGE_MASK_KEYS)
    if cached_mask is None:
        mask[:, :height_count, :width_count] = 1
    else:
        cached_mask = np.asarray(cached_mask)
        if cached_mask.ndim == 3:
            cached_mask = cached_mask[0] if cached_mask.shape[0] == 1 else cached_mask[..., 0]
        mask_height = min(height, cached_mask.shape[0])
        mask_width = min(width, cached_mask.shape[1])
        mask[:, :mask_height, :mask_width] = cached_mask[:mask_height, :mask_width] > 0.5
    return features, mask


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
    else:
        feature_shape = (int(args.image_channels), int(args.image_height), int(args.image_width))
        mask_shape = (1, int(args.image_height), int(args.image_width))
        normalize = normalize_image

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
