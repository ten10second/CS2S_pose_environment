"""Regression for a real KITTI point that changed pixels across NumPy builds."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader.kitti_pixel_feature_cache import pixel_index_from_uv
from dataloader.kitti_raw_lidar_utils import project_velo_to_image, zbuffer_visible_point_indices


def kitti_2011_09_26_calibration():
    rect = np.eye(4, dtype=np.float32)
    rect[:3, :3] = np.asarray([
        [9.999239e-01, 9.837760e-03, -7.445048e-03],
        [-9.869795e-03, 9.999421e-01, -4.278459e-03],
        [7.402527e-03, 4.351614e-03, 9.999631e-01],
    ], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray([
        [7.533745e-03, -9.999714e-01, -6.166020e-04],
        [1.480249e-02, 7.280733e-04, -9.998902e-01],
        [9.998621e-01, 7.523790e-03, 1.480755e-02],
    ], dtype=np.float32)
    transform[:3, 3] = [-4.069766e-03, -7.631618e-02, -2.717806e-01]
    return {
        "R_rect_00_ext": rect,
        "Tr_velo_to_cam": transform,
        "P_rect_02": np.asarray([
            [7.215377e02, 0, 6.095593e02, 4.485728e01],
            [0, 7.215377e02, 1.728540e02, 2.163791e-01],
            [0, 0, 1, 2.745884e-03],
        ], dtype=np.float32),
        "S_rect_02": np.asarray([1242, 375], dtype=np.float32),
    }


class ProjectionPrecisionTests(unittest.TestCase):
    def test_real_half_pixel_boundary_keeps_correct_pixel(self):
        # drive_0095_sync / frame 155 / raw point 28554. Legacy FP32
        # produced x=38.5 in sd21 and x=38.500011 in ControlS2S.
        xyz = np.asarray([[10.726, 7.542, -0.455]], dtype=np.float32)
        uv, depth, valid = project_velo_to_image(xyz, kitti_2011_09_26_calibration())
        np.testing.assert_array_equal(uv, np.asarray([[38.50000567145805, 72.46110431915501]], np.float32))
        self.assertEqual(uv.dtype, np.float32)
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(valid[0])
        ids = zbuffer_visible_point_indices(uv, depth, valid, (128, 512))
        self.assertEqual(pixel_index_from_uv(uv, ids, (128, 512)).tolist(), [72 * 512 + 39])

    def test_projection_does_not_depend_on_batch_size_or_memory_layout(self):
        point = np.asarray([10.726, 7.542, -0.455, 0.5], dtype=np.float32)
        batch = np.tile(point, (1025, 1))
        calib = kitti_2011_09_26_calibration()
        expected = project_velo_to_image(batch[:1, :3], calib)
        for xyz in (batch[:, :3], batch[:, :3].copy(), np.asfortranarray(batch[:, :3])):
            actual = project_velo_to_image(xyz, calib)
            for one, many in zip(expected, actual):
                np.testing.assert_array_equal(many, np.repeat(one, len(xyz), axis=0))

    def test_empty_and_outside_camera(self):
        calib = kitti_2011_09_26_calibration()
        uv, depth, valid = project_velo_to_image(np.empty((0, 3), np.float32), calib)
        self.assertEqual(uv.shape, (0, 2))
        self.assertEqual(depth.shape, (0,))
        self.assertEqual(valid.shape, (0,))
        _, _, valid = project_velo_to_image(np.asarray([[-10, 0, 0]], np.float32), calib)
        self.assertFalse(valid[0])


if __name__ == "__main__":
    unittest.main()
