import unittest

import numpy as np

from tools.convert_kitti_feature_cache_memmap import normalize_image, normalize_point


class KittiFeatureCacheConversionTest(unittest.TestCase):
    def test_current_utonia_cache_format(self):
        features = np.arange(24, dtype=np.float32).reshape(4, 6)
        mask = np.asarray([1, 1, 0, 0], dtype=np.float32)
        converted_features, converted_mask = normalize_point(
            {
                "utonia_feat": features,
                "lidar_point_features_mask": mask,
            },
            feature_shape=(4, 6),
            mask_shape=(4,),
        )
        np.testing.assert_array_equal(converted_features, features.astype(np.float16))
        np.testing.assert_array_equal(converted_mask, mask.astype(np.uint8))

    def test_current_dino_cache_format(self):
        features_hwc = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = np.ones((2, 3), dtype=np.float32)
        converted_features, converted_mask = normalize_image(
            {
                "dino_feat": features_hwc,
                "image_semantic_mask": mask,
            },
            feature_shape=(4, 2, 3),
            mask_shape=(1, 2, 3),
        )
        np.testing.assert_array_equal(
            converted_features,
            features_hwc.transpose(2, 0, 1).astype(np.float16),
        )
        np.testing.assert_array_equal(converted_mask, mask[None].astype(np.uint8))

    def test_wrong_feature_shapes_fail_instead_of_being_padded(self):
        with self.assertRaisesRegex(ValueError, "Utonia feature shape"):
            normalize_point(
                {
                    "utonia_feat": np.zeros((3, 6), dtype=np.float32),
                    "lidar_point_features_mask": np.ones((4,), dtype=np.float32),
                },
                feature_shape=(4, 6),
                mask_shape=(4,),
            )

        with self.assertRaisesRegex(ValueError, "DINO feature shape"):
            normalize_image(
                {
                    "dino_feat": np.zeros((2, 2, 4), dtype=np.float32),
                    "image_semantic_mask": np.ones((2, 3), dtype=np.float32),
                },
                feature_shape=(4, 2, 3),
                mask_shape=(1, 2, 3),
            )


if __name__ == "__main__":
    unittest.main()
