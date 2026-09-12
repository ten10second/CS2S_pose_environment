import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset
from dataloader.kitti_pixel_feature_cache import (
    PIXEL_CACHE_FORMAT,
    PIXEL_DEPTH_KEY,
    PIXEL_FEATURE_KEY,
    PIXEL_INDEX_KEY,
    PixelFeatureRaggedMemmapCache,
    build_visible_pixel_payload,
    load_npz_pixel_cache,
    preflight_pixel_ragged_cache,
    rasterize_pixel_features,
    validate_against_lidar_cond,
)
from dataloader.kitti_raw_lidar_utils import zbuffer_visible_point_indices
from tools.convert_kitti_pixel_cache_memmap import convert_pixel_cache


class KittiPixelFeatureCacheTest(unittest.TestCase):
    def test_same_pixel_keeps_nearest_visible_point_feature(self):
        uv = np.asarray([[1.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
        depth = np.asarray([20.0, 5.0, 7.0], dtype=np.float32)
        features = np.asarray([[100.0, 101.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        selected = zbuffer_visible_point_indices(uv, depth, np.ones((3,), dtype=bool), (4, 4))

        payload = build_visible_pixel_payload(features, uv, depth, selected, (4, 4), feature_dim=2)

        self.assertEqual(payload[PIXEL_INDEX_KEY].tolist(), [5, 6])
        np.testing.assert_array_equal(payload[PIXEL_FEATURE_KEY], features[[1, 2]].astype(np.float16))
        np.testing.assert_array_equal(payload[PIXEL_DEPTH_KEY], depth[[1, 2]])

    def test_same_patch_different_pixels_remain_distinct(self):
        uv = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        depth = np.asarray([4.0, 6.0], dtype=np.float32)
        features = np.asarray([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        payload = build_visible_pixel_payload(features, uv, depth, np.asarray([0, 1]), (4, 4), feature_dim=2)

        raster, mask = rasterize_pixel_features(payload, (4, 4), feature_dim=2)

        self.assertEqual(mask.sum(), 2.0)
        np.testing.assert_array_equal(raster[:, 0, 0], features[0].astype(np.float16))
        np.testing.assert_array_equal(raster[:, 1, 1], features[1].astype(np.float16))

    def test_cache_depth_and_hits_must_match_current_lidar_cond_subset(self):
        arrays = {
            PIXEL_FEATURE_KEY: np.ones((2, 2), dtype=np.float16),
            PIXEL_INDEX_KEY: np.asarray([0, 5], dtype=np.int64),
            PIXEL_DEPTH_KEY: np.asarray([8.0, 16.0], dtype=np.float32),
        }
        lidar_cond = np.zeros((4, 4, 4), dtype=np.float32)
        lidar_cond[1].reshape(-1)[[0, 5, 7]] = 1.0
        lidar_cond[2].reshape(-1)[[0, 5, 7]] = np.asarray([0.1, 0.2, 1.0], dtype=np.float32)

        validate_against_lidar_cond(arrays, lidar_cond, (4, 4), max_depth=80.0)

        lidar_cond[2].reshape(-1)[5] = 0.25
        with self.assertRaisesRegex(ValueError, "depth does not match"):
            validate_against_lidar_cond(arrays, lidar_cond, (4, 4), max_depth=80.0)

    def test_cache_pixels_must_be_integer_and_fp16_finite(self):
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            rasterize_pixel_features(
                {
                    PIXEL_FEATURE_KEY: np.ones((1, 2), dtype=np.float32),
                    PIXEL_INDEX_KEY: np.asarray([1.5], dtype=np.float32),
                    PIXEL_DEPTH_KEY: np.asarray([4.0], dtype=np.float32),
                },
                (4, 4),
                feature_dim=2,
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rasterize_pixel_features(
                {
                    PIXEL_FEATURE_KEY: np.asarray([[1.0e10, 0.0]], dtype=np.float32),
                    PIXEL_INDEX_KEY: np.asarray([1], dtype=np.int64),
                    PIXEL_DEPTH_KEY: np.asarray([4.0], dtype=np.float32),
                },
                (4, 4),
                feature_dim=2,
            )

    def test_ragged_memmap_roundtrip_matches_multi_frame_npz_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "npz"
            output = Path(temp_dir) / "memmap"
            source.mkdir()
            first = {
                PIXEL_FEATURE_KEY: np.arange(4, dtype=np.float32).reshape(2, 2).astype(np.float16),
                PIXEL_INDEX_KEY: np.asarray([0, 5], dtype=np.int64),
                PIXEL_DEPTH_KEY: np.asarray([4.0, 6.0], dtype=np.float32),
            }
            second = {
                PIXEL_FEATURE_KEY: np.asarray([[8.0, 10.0]], dtype=np.float16),
                PIXEL_INDEX_KEY: np.asarray([7], dtype=np.int64),
                PIXEL_DEPTH_KEY: np.asarray([9.0], dtype=np.float32),
            }
            third = {
                PIXEL_FEATURE_KEY: np.zeros((0, 2), dtype=np.float16),
                PIXEL_INDEX_KEY: np.zeros((0,), dtype=np.int64),
                PIXEL_DEPTH_KEY: np.zeros((0,), dtype=np.float32),
            }
            np.savez(source / "drive__frame0.npz", format=np.asarray(PIXEL_CACHE_FORMAT), image_height=4, image_width=4, **first)
            np.savez(source / "drive__frame1.npz", format=np.asarray(PIXEL_CACHE_FORMAT), image_height=4, image_width=4, **second)
            np.savez(source / "drive__frame2.npz", format=np.asarray(PIXEL_CACHE_FORMAT), image_height=4, image_width=4, **third)

            meta = convert_pixel_cache(source, output, (4, 4), feature_dim=2, workers=1)
            cache = PixelFeatureRaggedMemmapCache(output, (4, 4), feature_dim=2)

            self.assertEqual(meta["total_points"], 3)
            self.assertEqual(meta["count"], 3)
            np.testing.assert_array_equal(cache.get("drive/frame0")[PIXEL_FEATURE_KEY], first[PIXEL_FEATURE_KEY])
            np.testing.assert_array_equal(cache.get("drive/frame1")[PIXEL_INDEX_KEY], second[PIXEL_INDEX_KEY])
            self.assertEqual(cache.get("drive/frame2")[PIXEL_FEATURE_KEY].shape, (0, 2))
            np.testing.assert_array_equal(
                load_npz_pixel_cache(source / "drive__frame1.npz", (4, 4), feature_dim=2)[PIXEL_DEPTH_KEY],
                cache.get("drive/frame1")[PIXEL_DEPTH_KEY],
            )
            stats = preflight_pixel_ragged_cache(output, feature_dim=2, image_size=(4, 4), manifests=())
            self.assertEqual(stats["lidar_pixel_cache_total_points"], 3)

    def test_ragged_converter_refuses_non_empty_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "npz"
            output = Path(temp_dir) / "memmap"
            source.mkdir()
            output.mkdir()
            (output / "stale").write_text("old")
            payload = {
                PIXEL_FEATURE_KEY: np.ones((1, 2), dtype=np.float16),
                PIXEL_INDEX_KEY: np.asarray([0], dtype=np.int64),
                PIXEL_DEPTH_KEY: np.asarray([4.0], dtype=np.float32),
            }
            np.savez(source / "drive__frame0.npz", format=np.asarray(PIXEL_CACHE_FORMAT), image_height=4, image_width=4, **payload)
            with self.assertRaises(FileExistsError):
                convert_pixel_cache(source, output, (4, 4), feature_dim=2, workers=1)

    def test_npz_preflight_streams_required_manifest_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "npz"
            root.mkdir()
            manifest = Path(temp_dir) / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps({"sample_id": sample_id})
                    for sample_id in ("drive/frame0", "drive/frame1")
                )
            )
            for frame, depth in (("frame0", 4.0), ("frame1", 6.0)):
                payload = {
                    PIXEL_FEATURE_KEY: np.ones((1, 2), dtype=np.float16),
                    PIXEL_INDEX_KEY: np.asarray([0], dtype=np.int64),
                    PIXEL_DEPTH_KEY: np.asarray([depth], dtype=np.float32),
                }
                np.savez(
                    root / f"drive__{frame}.npz",
                    format=np.asarray(PIXEL_CACHE_FORMAT),
                    image_height=4,
                    image_width=4,
                    **payload,
                )

            stats = preflight_pixel_ragged_cache(root, feature_dim=2, image_size=(4, 4), manifests=[manifest])

            self.assertEqual(stats["lidar_pixel_cache_format"], "npz")
            self.assertEqual(stats["lidar_pixel_cache_rows"], 2)
            self.assertEqual(stats["lidar_pixel_cache_required_rows"], 2)
            self.assertEqual(stats["lidar_pixel_cache_total_points"], 2)

    def test_npz_preflight_fails_on_missing_manifest_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "npz"
            root.mkdir()
            manifest = Path(temp_dir) / "manifest.jsonl"
            manifest.write_text(json.dumps({"sample_id": "drive/frame0"}))

            with self.assertRaisesRegex(RuntimeError, "pixel NPZ cache misses"):
                preflight_pixel_ragged_cache(root, feature_dim=2, image_size=(4, 4), manifests=[manifest])

    def test_ragged_memmap_rejects_corrupt_offsets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            np.save(root / "features.npy", np.ones((2, 2), dtype=np.float16))
            np.save(root / "pixel_index.npy", np.asarray([0, 1], dtype=np.int64))
            np.save(root / "depth.npy", np.asarray([4.0, 5.0], dtype=np.float32))
            np.save(root / "offsets.npy", np.asarray([0, 2, 1], dtype=np.int64))
            meta = {
                "format": "kitti_pixel_feature_ragged_memmap_v1",
                "count": 2,
                "total_points": 2,
                "feature_dim": 2,
                "image_size": [4, 4],
                "features_file": "features.npy",
                "pixel_index_file": "pixel_index.npy",
                "depth_file": "depth.npy",
                "offsets_file": "offsets.npy",
                "index": {"drive__frame0": 0, "drive__frame1": 1},
            }
            (root / "pixel_memmap_meta.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "monotonic"):
                PixelFeatureRaggedMemmapCache(root, (4, 4), feature_dim=2).get("drive/frame0")

    def test_dataset_pixel_cache_missing_sample_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = SatLidarRawDataset.__new__(SatLidarRawDataset)
            dataset.lidar_pixel_feature_cache_root = Path(temp_dir)
            dataset.lidar_pixel_feature_cache_suffix = ".npz"
            dataset.lidar_pixel_feature_dim = 2
            dataset._lidar_pixel_feature_memmap = None
            dataset.image_size = (4, 4)
            dataset.max_depth = 80.0
            dataset._cache_path = SatLidarRawDataset._cache_path.__get__(dataset, SatLidarRawDataset)
            dataset._safe_cache_id = SatLidarRawDataset._safe_cache_id
            lidar_cond = np.zeros((4, 4, 4), dtype=np.float32)

            with self.assertRaises(FileNotFoundError):
                dataset._lidar_pixel_feature_cache("drive/frame0", lidar_cond)


if __name__ == "__main__":
    unittest.main()
