import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.kitti_pixel_feature_cache import (  # noqa: E402
    PIXEL_DEPTH_KEY,
    PIXEL_FEATURE_KEY,
    PIXEL_INDEX_KEY,
    PIXEL_MEMMAP_FORMAT,
    atomic_write_json,
    load_npz_pixel_cache,
    safe_sample_id,
    validate_pixel_ragged_memmap_arrays,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert KITTI pixel feature NPZ files to ragged NPY memmaps.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feature-dim", type=int, default=576)
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1, help="Reserved for bounded future use; conversion is sequential.")
    return parser.parse_args()


def require_empty_output_dir(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)


def iter_npz_files(source_root: Path, limit: int = 0):
    files = sorted(Path(source_root).glob("*.npz"))
    if int(limit) > 0:
        files = files[: int(limit)]
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {source_root}")
    return files


def scan_pixel_cache(files, output_size, feature_dim: int, progress_every: int = 100):
    offsets = [0]
    index = {}
    started = time.time()
    for row, path in enumerate(files):
        arrays = load_npz_pixel_cache(path, output_size=output_size, feature_dim=feature_dim)
        count = int(arrays[PIXEL_INDEX_KEY].shape[0])
        offsets.append(offsets[-1] + count)
        index[safe_sample_id(path.stem)] = row
        if row == 0 or (row + 1) % max(1, int(progress_every)) == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "phase": "scan",
                        "rows": row + 1,
                        "total": len(files),
                        "total_points": int(offsets[-1]),
                        "rows_per_second": (row + 1) / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return np.asarray(offsets, dtype=np.int64), index


def write_pixel_memmaps(files, output_root: Path, offsets, output_size, feature_dim: int, progress_every: int = 100):
    total_points = int(offsets[-1])
    features = np.lib.format.open_memmap(
        output_root / "features.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total_points, int(feature_dim)),
    )
    pixel_index = np.lib.format.open_memmap(
        output_root / "pixel_index.npy",
        mode="w+",
        dtype=np.int64,
        shape=(total_points,),
    )
    depth = np.lib.format.open_memmap(
        output_root / "depth.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_points,),
    )
    offset_map = np.lib.format.open_memmap(
        output_root / "offsets.npy",
        mode="w+",
        dtype=np.int64,
        shape=tuple(offsets.shape),
    )
    offset_map[...] = offsets
    started = time.time()
    for row, path in enumerate(files):
        arrays = load_npz_pixel_cache(path, output_size=output_size, feature_dim=feature_dim)
        start = int(offsets[row])
        end = int(offsets[row + 1])
        if arrays[PIXEL_INDEX_KEY].shape[0] != end - start:
            raise ValueError(f"Pixel count changed between scan and write: {path}")
        features[start:end] = arrays[PIXEL_FEATURE_KEY]
        pixel_index[start:end] = arrays[PIXEL_INDEX_KEY]
        depth[start:end] = arrays[PIXEL_DEPTH_KEY]
        if row == 0 or (row + 1) % max(1, int(progress_every)) == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "phase": "write",
                        "rows": row + 1,
                        "total": len(files),
                        "points_written": end,
                        "rows_per_second": (row + 1) / elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    features.flush()
    pixel_index.flush()
    depth.flush()
    offset_map.flush()
    validate_pixel_ragged_memmap_arrays(features, pixel_index, depth, offset_map, len(files), feature_dim)


def convert_pixel_cache(
    source_root: Path,
    output_root: Path,
    output_size,
    feature_dim: int,
    limit: int = 0,
    workers: int = 1,
    progress_every: int = 100,
):
    del workers
    source_root = Path(source_root)
    output_root = Path(output_root)
    files = iter_npz_files(source_root, limit=limit)
    require_empty_output_dir(output_root)
    offsets, index = scan_pixel_cache(files, output_size, int(feature_dim), progress_every=progress_every)
    write_pixel_memmaps(files, output_root, offsets, output_size, int(feature_dim), progress_every=progress_every)
    meta = {
        "format": PIXEL_MEMMAP_FORMAT,
        "count": len(files),
        "total_points": int(offsets[-1]),
        "feature_dim": int(feature_dim),
        "image_size": [int(output_size[0]), int(output_size[1])],
        "feature_dtype": "float16",
        "pixel_index_dtype": "int64",
        "depth_dtype": "float32",
        "features_file": "features.npy",
        "pixel_index_file": "pixel_index.npy",
        "depth_file": "depth.npy",
        "offsets_file": "offsets.npy",
        "index": index,
        "source_root": str(source_root),
    }
    atomic_write_json(output_root / "pixel_memmap_meta.json", meta)
    return meta


def main():
    args = parse_args()
    started = time.time()
    output_size = (int(args.image_height), int(args.image_width))
    meta = convert_pixel_cache(
        Path(args.source_root),
        Path(args.output_root),
        output_size,
        int(args.feature_dim),
        limit=int(args.limit),
        workers=int(args.workers),
        progress_every=int(args.progress_every),
    )
    elapsed = max(time.time() - started, 1e-6)
    print(
        json.dumps(
            {
                "complete": True,
                "output_root": str(args.output_root),
                "rows": int(meta["count"]),
                "total_points": int(meta["total_points"]),
                "rows_per_second": int(meta["count"]) / elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
