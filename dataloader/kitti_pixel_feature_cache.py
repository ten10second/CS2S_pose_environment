import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


PIXEL_CACHE_FORMAT = "kitti_utonia_pixel_cache_v1"
PIXEL_MEMMAP_FORMAT = "kitti_pixel_feature_ragged_memmap_v1"
PIXEL_FEATURE_KEY = "utonia_pixel_features"
PIXEL_INDEX_KEY = "pixel_index"
PIXEL_DEPTH_KEY = "depth"


def safe_sample_id(sample_id: str) -> str:
    return str(sample_id).replace("/", "__")


def pixel_index_from_uv(uv: np.ndarray, indices: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = output_size
    selected_uv = np.asarray(uv, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]
    x = np.rint(selected_uv[:, 0]).astype(np.int64)
    y = np.rint(selected_uv[:, 1]).astype(np.int64)
    x = np.clip(x, 0, out_w - 1)
    y = np.clip(y, 0, out_h - 1)
    return (y * out_w + x).astype(np.int64, copy=False)


def build_visible_pixel_payload(
    features: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    visible_indices: np.ndarray,
    output_size: Tuple[int, int],
    feature_dim: int = 576,
) -> Dict[str, np.ndarray]:
    features = np.asarray(features)
    depth = np.asarray(depth, dtype=np.float32)
    visible_indices = np.asarray(visible_indices, dtype=np.int64)
    if features.ndim != 2:
        raise ValueError(f"pixel features must be 2-D, got {features.shape}")
    if features.shape[1] != int(feature_dim):
        raise ValueError(f"pixel feature dim {features.shape[1]} != expected {int(feature_dim)}")
    if visible_indices.size and (visible_indices.min() < 0 or visible_indices.max() >= features.shape[0]):
        raise ValueError("visible indices are outside feature rows")

    selected_features = features[visible_indices].astype(np.float16, copy=False)
    selected_depth = depth[visible_indices].astype(np.float32, copy=False)
    pixel_index = pixel_index_from_uv(uv, visible_indices, output_size)
    return validate_pixel_arrays(selected_features, pixel_index, selected_depth, output_size, feature_dim)


def validate_pixel_arrays(
    features: np.ndarray,
    pixel_index: np.ndarray,
    depth: np.ndarray,
    output_size: Tuple[int, int],
    feature_dim: int = 576,
) -> Dict[str, np.ndarray]:
    out_h, out_w = output_size
    features = np.asarray(features)
    pixel_index = np.asarray(pixel_index)
    depth = np.asarray(depth)
    if features.ndim != 2:
        raise ValueError(f"pixel features must be 2-D, got {features.shape}")
    if features.shape[1] != int(feature_dim):
        raise ValueError(f"pixel feature dim {features.shape[1]} != expected {int(feature_dim)}")
    if pixel_index.shape != (features.shape[0],):
        raise ValueError(f"pixel_index shape {pixel_index.shape} != expected {(features.shape[0],)}")
    if depth.shape != (features.shape[0],):
        raise ValueError(f"depth shape {depth.shape} != expected {(features.shape[0],)}")
    if not np.issubdtype(pixel_index.dtype, np.integer):
        raise ValueError(f"pixel_index must use an integer dtype, got {pixel_index.dtype}")
    pixel_index = pixel_index.astype(np.int64, copy=False)
    depth = depth.astype(np.float32, copy=False)
    features = features.astype(np.float16, copy=False)
    if features.size and not np.isfinite(features.astype(np.float32, copy=False)).all():
        raise ValueError("pixel features contain non-finite values")
    if not np.isfinite(depth).all():
        raise ValueError("pixel depth contains non-finite values")
    if np.any(depth <= 0.0):
        raise ValueError("pixel depth must be positive")
    pixel_count = int(out_h) * int(out_w)
    if np.any(pixel_index < 0) or np.any(pixel_index >= pixel_count):
        raise ValueError("pixel_index contains out-of-bounds pixels")
    if np.unique(pixel_index).size != pixel_index.size:
        raise ValueError("pixel_index must contain unique z-buffer-visible pixels")
    return {
        PIXEL_FEATURE_KEY: features,
        PIXEL_INDEX_KEY: pixel_index,
        PIXEL_DEPTH_KEY: depth,
    }


def load_npz_pixel_cache(path: Path, output_size: Tuple[int, int], feature_dim: int = 576) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if "format" not in payload:
            raise KeyError(f"pixel cache missing format metadata: {path}")
        cache_format = str(np.asarray(payload["format"]).item())
        if cache_format != PIXEL_CACHE_FORMAT:
            raise ValueError(f"invalid pixel cache format {cache_format!r}: {path}")
        if "image_height" not in payload or "image_width" not in payload:
            raise KeyError(f"pixel cache missing image size metadata: {path}")
        if int(np.asarray(payload["image_height"]).item()) != int(output_size[0]):
            raise ValueError(f"pixel cache height mismatch: {path}")
        if int(np.asarray(payload["image_width"]).item()) != int(output_size[1]):
            raise ValueError(f"pixel cache width mismatch: {path}")
        features = None
        for key in (PIXEL_FEATURE_KEY, "lidar_pixel_features", "pixel_features", "features"):
            if key in payload:
                features = payload[key]
                break
        pixel_index = payload[PIXEL_INDEX_KEY] if PIXEL_INDEX_KEY in payload else None
        depth = payload[PIXEL_DEPTH_KEY] if PIXEL_DEPTH_KEY in payload else None
        if features is None or pixel_index is None or depth is None:
            raise KeyError(f"pixel cache missing required arrays: {path}")
        return validate_pixel_arrays(features, pixel_index, depth, output_size, feature_dim)


def rasterize_pixel_features(
    arrays: Dict[str, np.ndarray],
    output_size: Tuple[int, int],
    feature_dim: int = 576,
) -> Tuple[np.ndarray, np.ndarray]:
    arrays = validate_pixel_arrays(
        arrays[PIXEL_FEATURE_KEY],
        arrays[PIXEL_INDEX_KEY],
        arrays[PIXEL_DEPTH_KEY],
        output_size,
        feature_dim,
    )
    out_h, out_w = output_size
    features = np.zeros((int(feature_dim), int(out_h), int(out_w)), dtype=np.float16)
    mask = np.zeros((1, int(out_h), int(out_w)), dtype=np.float32)
    pixel_index = arrays[PIXEL_INDEX_KEY]
    if pixel_index.size:
        y = pixel_index // int(out_w)
        x = pixel_index % int(out_w)
        features[:, y, x] = arrays[PIXEL_FEATURE_KEY].T
        mask[0, y, x] = 1.0
    return features, mask


def validate_against_lidar_cond(
    arrays: Dict[str, np.ndarray],
    lidar_cond: np.ndarray,
    output_size: Tuple[int, int],
    max_depth: float,
    depth_atol: float = 2e-3,
) -> None:
    cond = np.asarray(lidar_cond)
    if cond.ndim != 3 or cond.shape[0] < 3:
        raise ValueError(f"lidar_cond must have at least 3 channels, got {cond.shape}")
    out_h, out_w = output_size
    if tuple(cond.shape[-2:]) != (int(out_h), int(out_w)):
        raise ValueError(f"lidar_cond spatial shape {cond.shape[-2:]} != expected {(out_h, out_w)}")
    arrays = validate_pixel_arrays(
        arrays[PIXEL_FEATURE_KEY],
        arrays[PIXEL_INDEX_KEY],
        arrays[PIXEL_DEPTH_KEY],
        output_size,
        arrays[PIXEL_FEATURE_KEY].shape[1],
    )
    hit_mask = cond[1].reshape(-1) > 0.5
    depth_norm = cond[2].reshape(-1).astype(np.float32)
    pixel_index = arrays[PIXEL_INDEX_KEY]
    if pixel_index.size == 0:
        return
    if not np.all(hit_mask[pixel_index]):
        raise ValueError("pixel cache contains pixels missing from current lidar_cond hits")
    cache_depth_norm = np.minimum(arrays[PIXEL_DEPTH_KEY], float(max_depth)) / float(max_depth)
    if not np.allclose(depth_norm[pixel_index], cache_depth_norm, rtol=0.0, atol=float(depth_atol)):
        max_err = float(np.max(np.abs(depth_norm[pixel_index] - cache_depth_norm)))
        raise ValueError(f"pixel cache depth does not match current lidar_cond depth (max error {max_err:.6g})")


class PixelFeatureRaggedMemmapCache:
    META_NAME = "pixel_memmap_meta.json"

    def __init__(self, root: Path, output_size: Tuple[int, int], feature_dim: int = 576):
        self.root = Path(root)
        self.output_size = tuple(output_size)
        self.feature_dim = int(feature_dim)
        meta_path = self.root / self.META_NAME
        self.enabled = meta_path.is_file()
        self.index = {}
        self._features = None
        self._pixel_index = None
        self._depth = None
        self._offsets = None
        if not self.enabled:
            return
        meta = json.loads(meta_path.read_text())
        if meta.get("format") != PIXEL_MEMMAP_FORMAT:
            raise ValueError(f"Invalid pixel memmap metadata: {meta_path}")
        if int(meta.get("feature_dim", -1)) != self.feature_dim:
            raise ValueError(f"pixel memmap feature dim {meta.get('feature_dim')} != expected {self.feature_dim}")
        if tuple(meta.get("image_size", ())) != self.output_size:
            raise ValueError(f"pixel memmap image size {meta.get('image_size')} != expected {self.output_size}")
        validate_pixel_ragged_memmap_meta(meta, self.root, self.feature_dim, self.output_size)
        self.features_path = self.root / meta["features_file"]
        self.pixel_index_path = self.root / meta["pixel_index_file"]
        self.depth_path = self.root / meta["depth_file"]
        self.offsets_path = self.root / meta["offsets_file"]
        self.index = {str(key): int(value) for key, value in meta["index"].items()}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        state["_pixel_index"] = None
        state["_depth"] = None
        state["_offsets"] = None
        return state

    def _open(self):
        if self._features is None:
            self._features = np.load(self.features_path, mmap_mode="r", allow_pickle=False)
            self._pixel_index = np.load(self.pixel_index_path, mmap_mode="r", allow_pickle=False)
            self._depth = np.load(self.depth_path, mmap_mode="r", allow_pickle=False)
            self._offsets = np.load(self.offsets_path, mmap_mode="r", allow_pickle=False)
            validate_pixel_ragged_memmap_arrays(
                self._features,
                self._pixel_index,
                self._depth,
                self._offsets,
                len(self.index),
                self.feature_dim,
            )

    def get(self, sample_id: str):
        row = self.index.get(safe_sample_id(sample_id))
        if row is None:
            return None
        self._open()
        start = int(self._offsets[row])
        end = int(self._offsets[row + 1])
        return validate_pixel_arrays(
            np.asarray(self._features[start:end]),
            np.asarray(self._pixel_index[start:end]),
            np.asarray(self._depth[start:end]),
            self.output_size,
            self.feature_dim,
        )


def validate_pixel_ragged_memmap_meta(
    meta: Dict[str, object],
    root: Path,
    feature_dim: int,
    image_size: Tuple[int, int],
) -> None:
    if meta.get("format") != PIXEL_MEMMAP_FORMAT:
        raise ValueError(f"Invalid pixel ragged memmap format: {meta.get('format')!r}")
    if int(meta.get("feature_dim", -1)) != int(feature_dim):
        raise ValueError(f"pixel ragged feature dim {meta.get('feature_dim')} != expected {int(feature_dim)}")
    if tuple(meta.get("image_size", ())) != tuple(image_size):
        raise ValueError(f"pixel ragged image size {tuple(meta.get('image_size', ()))} != expected {tuple(image_size)}")
    for key in ("features_file", "pixel_index_file", "depth_file", "offsets_file"):
        value = meta.get(key)
        if not value:
            raise KeyError(f"pixel ragged metadata missing {key}")
        path = Path(root) / str(value)
        if not path.is_file():
            raise FileNotFoundError(f"Required pixel ragged memmap file not found: {path}")
    index = meta.get("index")
    if not isinstance(index, dict):
        raise ValueError("pixel ragged metadata index must be an object")
    rows = sorted(int(value) for value in index.values())
    expected_rows = list(range(len(rows)))
    if rows != expected_rows:
        raise ValueError("pixel ragged metadata index rows must be a contiguous 0-based permutation")
    if int(meta.get("count", len(rows))) != len(rows):
        raise ValueError(f"pixel ragged count {meta.get('count')} != index rows {len(rows)}")


def validate_pixel_ragged_memmap_arrays(
    features: np.ndarray,
    pixel_index: np.ndarray,
    depth: np.ndarray,
    offsets: np.ndarray,
    row_count: int,
    feature_dim: int,
) -> None:
    if features.dtype != np.float16 or features.ndim != 2 or features.shape[1] != int(feature_dim):
        raise ValueError(f"pixel ragged features shape/dtype invalid: {features.shape} {features.dtype}")
    if not np.issubdtype(pixel_index.dtype, np.integer) or pixel_index.ndim != 1:
        raise ValueError(f"pixel ragged pixel_index shape/dtype invalid: {pixel_index.shape} {pixel_index.dtype}")
    if depth.dtype != np.float32 or depth.ndim != 1:
        raise ValueError(f"pixel ragged depth shape/dtype invalid: {depth.shape} {depth.dtype}")
    if not np.issubdtype(offsets.dtype, np.integer) or offsets.shape != (int(row_count) + 1,):
        raise ValueError(f"pixel ragged offsets shape/dtype invalid: {offsets.shape} {offsets.dtype}")
    if features.shape[0] != pixel_index.shape[0] or features.shape[0] != depth.shape[0]:
        raise ValueError("pixel ragged feature/pixel/depth row counts differ")
    offsets64 = np.asarray(offsets, dtype=np.int64)
    if offsets64[0] != 0:
        raise ValueError("pixel ragged offsets must start at zero")
    if np.any(offsets64[1:] < offsets64[:-1]):
        raise ValueError("pixel ragged offsets must be monotonic")
    if int(offsets64[-1]) != int(features.shape[0]):
        raise ValueError(f"pixel ragged final offset {int(offsets64[-1])} != total rows {int(features.shape[0])}")


def required_safe_ids_from_manifests(manifests: Iterable[object]) -> set:
    required_ids = set()
    for manifest in manifests:
        if not manifest:
            continue
        for line in Path(manifest).read_text().splitlines():
            if line.strip():
                required_ids.add(safe_sample_id(json.loads(line)["sample_id"]))
    return required_ids


def preflight_pixel_ragged_cache(
    root: Path,
    feature_dim: int,
    image_size: Tuple[int, int],
    manifests: Iterable[object] = (),
) -> Dict[str, object]:
    root = Path(root)
    meta_path = root / PixelFeatureRaggedMemmapCache.META_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Required pixel ragged memmap metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    validate_pixel_ragged_memmap_meta(meta, root, feature_dim, image_size)
    features = np.load(root / meta["features_file"], mmap_mode="r", allow_pickle=False)
    pixel_index = np.load(root / meta["pixel_index_file"], mmap_mode="r", allow_pickle=False)
    depth = np.load(root / meta["depth_file"], mmap_mode="r", allow_pickle=False)
    offsets = np.load(root / meta["offsets_file"], mmap_mode="r", allow_pickle=False)
    index = meta.get("index", {})
    validate_pixel_ragged_memmap_arrays(features, pixel_index, depth, offsets, len(index), feature_dim)
    total_points = int(features.shape[0])
    if int(meta.get("total_points", total_points)) != total_points:
        raise ValueError(f"pixel ragged total_points {meta.get('total_points')} != features rows {total_points}")
    required_ids = required_safe_ids_from_manifests(manifests)
    missing = sorted(required_ids.difference(index))
    if missing:
        examples = ", ".join(missing[:5])
        raise RuntimeError(
            f"pixel ragged memmap cache misses {len(missing)}/{len(required_ids)} required samples; "
            f"examples: {examples}"
        )
    return {
        "lidar_pixel_cache_rows": len(index),
        "lidar_pixel_cache_required_rows": len(required_ids),
        "lidar_pixel_cache_total_points": total_points,
        "lidar_pixel_cache_root": str(root),
    }


def atomic_write_json(path: Path, data: Dict[str, object]) -> None:
    path = Path(path)
    temp_path = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    temp_path.unlink(missing_ok=True)
    try:
        temp_path.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
