"""Depth-free history coverage without overwriting measured geometry rejection."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools import temporal_history_geometry as geo


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.grid = (3, 5)
        self.k = np.array([[4., 0., 2.], [0., 4., 1.], [0., 0., 1.]])
        self.calib = geo.RawKittiGeometry(
            np.column_stack([self.k, [-.4, 0., 0.]]),
            np.eye(4), np.eye(4), np.eye(4), (5, 3))
        self.empty = np.empty((0, 3), np.float32)

    def build(self, prev=None, cur=None, transform=None):
        return geo.build_history_geometry_from_arrays(
            self.empty if prev is None else prev,
            self.empty if cur is None else cur,
            np.eye(4) if transform is None else transform,
            self.calib, grid=self.grid)

    def test_no_lidar_identity_covers_image(self):
        result = self.build()
        self.assertTrue(result['history_valid'].all())
        self.assertTrue(result['rotation_valid'].all())
        self.assertFalse(result['lidar_valid'].any())
        yy, xx = np.indices(self.grid)
        expected = geo.align_corners_false_normalize(np.stack([xx, yy], -1), self.grid)
        np.testing.assert_allclose(result['history_grid'], expected, atol=1e-7)
        self.assertEqual(result['metrics']['coverage_rotation'], 1.)

    def test_rotation_has_correct_direction_and_bounds(self):
        transform = np.eye(4)
        angle = .12
        transform[:3, :3] = [[np.cos(angle), 0., np.sin(angle)],
                            [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
        result = self.build(transform=transform)
        ray = np.linalg.solve(self.k, [2., 1., 1.])
        p = self.k @ transform[:3, :3] @ ray
        expected_x = p[0] / p[2]
        self.assertAlmostEqual(float(result['history_grid_px'][1, 2, 0]), expected_x, places=6)
        self.assertGreater(expected_x, 2.)
        self.assertFalse(result['history_valid'][:, -1].any())
        self.assertTrue(result['history_valid'][1, 2])

    def test_depth_free_fallback_ignores_translation(self):
        translated = np.eye(4)
        translated[0, 3] = 2.
        np.testing.assert_array_equal(self.build()['history_grid'],
                                      self.build(transform=translated)['history_grid'])

    def test_resized_non_square_identity_and_camera_baseline(self):
        result = geo.build_history_geometry_from_arrays(
            self.empty, self.empty, np.eye(4), self.calib,
            grid=(6, 10), image_size=(20, 12))
        self.assertTrue(result['history_valid'].all())
        yy, xx = np.indices((6, 10))
        expected = geo.align_corners_false_normalize(np.stack([xx, yy], -1), (6, 10))
        np.testing.assert_allclose(result['history_grid'], expected, atol=1e-7)

    def test_resized_rotation_matches_raw_camera_projection(self):
        transform = np.eye(4)
        angle = .12
        transform[:3, :3] = [[np.cos(angle), 0., np.sin(angle)],
                            [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
        result = geo.build_history_geometry_from_arrays(
            self.empty, self.empty, transform, self.calib,
            grid=(6, 10), image_size=(20, 12))
        raw_current = np.array([2.125, 1.125, 1.])
        raw_prev = self.k @ transform[:3, :3] @ np.linalg.solve(self.k, raw_current)
        expected = 2 * raw_prev[:2] / raw_prev[2] - .25
        self.assertTrue(result['history_valid'][2, 4])
        np.testing.assert_allclose(result['history_grid_px'][2, 4], expected, atol=1e-6)

    def test_behind_camera_is_not_history(self):
        transform = np.diag([-1., 1., -1., 1.])
        result = self.build(transform=transform)
        self.assertFalse(result['history_valid'].any())
        self.assertTrue(np.isfinite(result['history_grid']).all())

    def test_row_discontinuity_is_rejected_before_io(self):
        prev = {'date': 'a', 'drive': '1', 'frame_index': 3}
        for cur in [{'date': 'a', 'drive': '1', 'frame_index': 5},
                    {'date': 'a', 'drive': '2', 'frame_index': 4}]:
            with self.assertRaises(ValueError):
                geo.build_pair_geometry(prev, cur)

    def test_lidar_anchors_remain_exact_and_priority(self):
        # Camera02 baseline is included by the existing full projection.
        cur = np.array([[.1, 0., 4.]], np.float32)
        transform = np.eye(4)
        transform[0, 3] = .5
        prev = geo.transform_points(cur, transform)
        sparse = geo.build_lidar_history_geometry_from_arrays(
            prev, cur, transform, self.calib, grid=self.grid)
        result = self.build(prev, cur, transform)
        mask = sparse['history_valid']
        self.assertTrue(mask.any())
        np.testing.assert_array_equal(result['history_grid'][mask], sparse['history_grid'][mask])
        np.testing.assert_array_equal(result['lidar_valid'], mask)
        self.assertFalse(result['rotation_valid'][mask].any())
        self.assertTrue(result['history_valid'][~result['current_covered']].all())

    def test_depth_conflict_does_not_fall_back(self):
        cur = np.array([[.1, 0., 4.]], np.float32)
        prev = np.array([[.1, 0., 1.]], np.float32)
        result = self.build(prev, cur)
        covered = result['current_covered']
        self.assertTrue(covered.any())
        self.assertFalse(result['history_valid'][covered].any())
        np.testing.assert_array_equal(result['known_rejected'], covered)
        self.assertTrue(result['history_valid'][~covered].all())

    def test_camera_extrinsic_rotation_is_used(self):
        self.calib.t_cam_velo[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        t = np.eye(4)
        angle = .12
        t[:3, :3] = [[np.cos(angle), 0., np.sin(angle)],
                     [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]]
        result = self.build(transform=t)
        self.assertAlmostEqual(float(result['history_grid_px'][1, 2, 0]), 2., places=6)
        self.assertGreater(float(result['history_grid_px'][1, 2, 1]), 1.)

    def test_invalid_pose_and_grid_fail(self):
        for transform in [np.full((4, 4), np.nan), np.zeros((4, 4))]:
            with self.assertRaises(ValueError):
                self.build(transform=transform)
        with self.assertRaises(ValueError):
            geo.build_history_geometry_from_arrays(self.empty, self.empty, np.eye(4),
                                                    self.calib, grid=(0, 5))



if __name__ == '__main__':
    unittest.main()
