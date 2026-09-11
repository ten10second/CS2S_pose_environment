import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from dataloader.kitti_raw_lidar_utils import (  # noqa: E402
    _rasterize_points,
    zbuffer_visible_point_indices,
)
from tools.build_kitti_utonia_ray_cache import pool_visible_ray_features  # noqa: E402


def test_zbuffer_keeps_nearest_point_per_rounded_pixel():
    uv = np.asarray([[1.2, 1.2], [1.4, 1.4], [2.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    depth = np.asarray([10.0, 5.0, 7.0, 1.0], dtype=np.float32)
    valid = np.asarray([True, True, True, False])

    selected = zbuffer_visible_point_indices(uv, depth, valid, (4, 4))

    assert selected.tolist() == [1, 2]


def test_zbuffer_tie_keeps_lower_source_index():
    uv = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    depth = np.asarray([5.0, 5.0], dtype=np.float32)
    valid = np.ones((2,), dtype=bool)

    selected = zbuffer_visible_point_indices(uv, depth, valid, (4, 4))

    assert selected.tolist() == [0]


def test_zbuffer_rejects_nonfinite_and_nonpositive_depths():
    uv = np.asarray([[0.0, 0.0], [np.nan, 1.0], [2.0, 2.0]], dtype=np.float32)
    depth = np.asarray([0.0, 3.0, np.inf], dtype=np.float32)
    valid = np.ones((3,), dtype=bool)

    selected = zbuffer_visible_point_indices(uv, depth, valid, (4, 4))

    assert selected.size == 0


def test_existing_depth_rasterizer_uses_same_front_surface():
    uv = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    depth = np.asarray([40.0, 8.0], dtype=np.float32)
    valid = np.ones((2,), dtype=bool)

    mask, depth_map, projected_count = _rasterize_points(uv, depth, valid, (4, 4), 80.0)

    assert projected_count == 2
    assert mask.sum() == 1.0
    assert np.isclose(depth_map[1, 1], 0.1)


def test_refactored_depth_rasterizer_matches_v1_loop():
    rng = np.random.default_rng(3407)
    uv = rng.uniform(low=[0.0, 0.0], high=[15.9, 7.9], size=(500, 2)).astype(np.float32)
    depth = rng.uniform(0.1, 100.0, size=(500,)).astype(np.float32)
    valid = rng.random(500) > 0.2
    expected_mask = np.zeros((8, 16), dtype=np.float32)
    expected_depth = np.zeros((8, 16), dtype=np.float32)
    nearest = np.full((8, 16), np.inf, dtype=np.float32)
    for idx in np.nonzero(valid)[0]:
        x = int(np.clip(round(float(uv[idx, 0])), 0, 15))
        y = int(np.clip(round(float(uv[idx, 1])), 0, 7))
        if depth[idx] < nearest[y, x]:
            nearest[y, x] = depth[idx]
            expected_mask[y, x] = 1.0
            expected_depth[y, x] = min(float(depth[idx]), 80.0) / 80.0

    actual_mask, actual_depth, _ = _rasterize_points(uv, depth, valid, (8, 16), 80.0)

    assert np.array_equal(actual_mask, expected_mask)
    assert np.allclose(actual_depth, expected_depth)


def test_patch_pooling_averages_distinct_visible_pixels():
    args = SimpleNamespace(image_height=4, image_width=4, ray_height=2, ray_width=2)
    uv = np.asarray([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    features = np.asarray([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], dtype=np.float32)

    pooled, mask, counts = pool_visible_ray_features(features, uv, args)

    assert pooled.shape == (2, 1, 2, 2)
    assert mask.shape == (1, 1, 2, 2)
    assert np.allclose(pooled[:, 0, 0, 0], [4.0, 6.0])
    assert np.allclose(pooled[:, 0, 1, 1], [10.0, 12.0])
    assert counts.reshape(2, 2).tolist() == [[2.0, 0.0], [0.0, 1.0]]
    assert mask.sum() == 2


def test_zbuffer_then_pool_excludes_occluded_feature():
    args = SimpleNamespace(image_height=4, image_width=4, ray_height=2, ray_width=2)
    uv = np.asarray([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    depth = np.asarray([30.0, 5.0, 8.0], dtype=np.float32)
    features = np.asarray([[100.0], [4.0], [8.0]], dtype=np.float32)
    selected = zbuffer_visible_point_indices(uv, depth, np.ones((3,), dtype=bool), (4, 4))

    pooled, _, _ = pool_visible_ray_features(features[selected], uv[selected], args)

    assert np.allclose(pooled[:, 0, 0, 0], [6.0])


def load_tests(loader, tests, pattern):
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    return unittest.TestSuite(unittest.FunctionTestCase(function) for function in functions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
