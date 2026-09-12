import unittest

import numpy as np

from dataloader.kitti_pixel_feature_cache import PIXEL_DEPTH_KEY, PIXEL_INDEX_KEY
from tools.migrate_kitti_pixel_cache_projection import (
    MissingSourceFeaturesError,
    remap_features_by_source_index,
    verify_legacy_cache_exact,
)


class PixelProjectionMigrationTest(unittest.TestCase):
    def test_remap_features_by_source_index_reorders_and_subsets(self):
        features = np.asarray(
            [
                [10, 11],
                [20, 21],
                [30, 31],
                [40, 41],
            ],
            dtype=np.float16,
        )
        old_indices = np.asarray([100, 200, 300, 400], dtype=np.int64)
        new_indices = np.asarray([300, 100, 400], dtype=np.int64)

        remapped, rows = remap_features_by_source_index(features, old_indices, new_indices)

        np.testing.assert_array_equal(rows, np.asarray([2, 0, 3], dtype=np.int64))
        np.testing.assert_array_equal(remapped, features[[2, 0, 3]])
        self.assertEqual(remapped.dtype, np.float16)

    def test_remap_features_by_source_index_raises_for_missing_canonical_source(self):
        features = np.asarray([[1], [2]], dtype=np.float16)
        old_indices = np.asarray([5, 7], dtype=np.int64)
        new_indices = np.asarray([7, 9], dtype=np.int64)

        with self.assertRaises(MissingSourceFeaturesError) as ctx:
            remap_features_by_source_index(features, old_indices, new_indices)

        self.assertEqual(ctx.exception.missing_indices, [9])

    def test_verify_legacy_cache_exact_rejects_pixel_mismatch(self):
        arrays = {
            PIXEL_INDEX_KEY: np.asarray([10, 12], dtype=np.int64),
            PIXEL_DEPTH_KEY: np.asarray([1.0, 2.0], dtype=np.float32),
        }
        legacy_payload = {
            "pixel_index": np.asarray([10, 13], dtype=np.int64),
            "depth": np.asarray([1.0, 2.0], dtype=np.float32),
        }

        with self.assertRaisesRegex(ValueError, "legacy pixel_index exact check failed"):
            verify_legacy_cache_exact(arrays, legacy_payload, "sample")


if __name__ == "__main__":
    unittest.main()
