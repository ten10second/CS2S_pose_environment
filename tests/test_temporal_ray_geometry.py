from __future__ import annotations

import unittest
import numpy as np

from tools.temporal_history_geometry import RawKittiGeometry, transform_points
from tools.temporal_ray_geometry import (
    SOURCE_CONTENT_FALLBACK,
    SOURCE_OBJECT_COMP,
    SOURCE_STATIC_MEASUREMENT,
    SOURCE_UNKNOWN_RAY,
    _camera_depths_to_velo_points,
    _camera_rays_for_feature_grid,
    _satellite_grid_from_velo_points,
    build_ray_geometry_from_arrays,
)


def synthetic_geometry(image_size=(8, 4)):
    w, h = image_size
    p = np.array([[20.0, 0.0, (w - 1) / 2.0, 0.0],
                  [0.0, 20.0, (h - 1) / 2.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    # KITTI-like axes: camera x=right=-velo_y, camera y=down=-velo_z, camera z=forward=velo_x.
    t_cam_velo = np.eye(4, dtype=np.float64)
    t_cam_velo[:3, :3] = np.array([[0.0, -1.0, 0.0],
                                    [0.0, 0.0, -1.0],
                                    [1.0, 0.0, 0.0]], dtype=np.float64)
    return RawKittiGeometry(
        p_rect_02=p,
        r_rect_00_ext=np.eye(4, dtype=np.float64),
        t_cam_velo=t_cam_velo,
        t_imu_velo=None,
        image_size=image_size,
    )


def cell_points(geometry, grid=(2, 4), depth=20.0):
    rays = _camera_rays_for_feature_grid(geometry, grid, geometry.image_size)
    return _camera_depths_to_velo_points(geometry, rays, np.array([depth], dtype=np.float32))[:, :, 0, :]


class TemporalRayGeometryTests(unittest.TestCase):
    def assert_contract(self, out, grid=(2, 4), k=5):
        h, w = grid
        self.assertEqual(out['history_grid'].shape, (h, w, k, 2))
        self.assertEqual(out['sat_grid'].shape, (h, w, k, 2))
        self.assertEqual(out['valid'].shape, (h, w, k))
        self.assertEqual(out['sat_valid'].shape, (h, w, k))
        self.assertEqual(out['positions'].shape, (h, w, k, 4))
        self.assertEqual(out['valid'].dtype, np.bool_)
        self.assertEqual(out['sat_valid'].dtype, np.bool_)
        self.assertTrue(np.isfinite(out['history_grid']).all())
        self.assertTrue(np.isfinite(out['sat_grid']).all())
        self.assertTrue(np.isfinite(out['positions']).all())
        self.assertTrue(((out['positions'][..., 2] >= 0.0) & (out['positions'][..., 2] <= 1.0)).all())

    def test_no_lidar_uses_positive_inverse_depth_candidates_full_grid(self):
        geom = synthetic_geometry()
        out = build_ray_geometry_from_arrays(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            np.eye(4), geom, grid=(2, 4), num_depth_candidates=5, depth_min_m=2.0, depth_max_m=50.0)
        self.assert_contract(out)
        self.assertTrue(out['valid'].all())
        self.assertTrue((out['source'] == SOURCE_UNKNOWN_RAY).all())
        self.assertTrue(np.all(np.diff(out['depth_candidates_m']) > 0.0))
        self.assertGreater(out['depth_candidates_m'][1] - out['depth_candidates_m'][0], 0.0)
        default = build_ray_geometry_from_arrays(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            np.eye(4), geom, grid=(1, 1))
        self.assertAlmostEqual(float(default['depth_candidates_m'][0]), 2.0)
        self.assertAlmostEqual(float(default['depth_candidates_m'][-1]), 120.0, places=4)
        tilted = synthetic_geometry(image_size=(8, 6))
        height_out = build_ray_geometry_from_arrays(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            np.eye(4), tilted, grid=(3, 4), num_depth_candidates=4,
            depth_min_m=2.0, depth_max_m=80.0)
        # Off-horizon rays have different velo z as depth changes; positions[1]
        # is normalized physical height, not image row index.
        self.assertGreater(np.ptp(height_out['positions'][0, 0, :, 1]), 0.0)

    def test_identity_keeps_ray_candidates_at_feature_cell_centers(self):
        geom = synthetic_geometry()
        out = build_ray_geometry_from_arrays(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            np.eye(4), geom, grid=(2, 4), num_depth_candidates=3, depth_min_m=4.0, depth_max_m=40.0)
        yy, xx = np.indices((2, 4), dtype=np.float32)
        expected_x = (2.0 * (xx + 0.5) / 4.0) - 1.0
        expected_y = (2.0 * (yy + 0.5) / 2.0) - 1.0
        self.assertTrue(np.allclose(out['history_grid'][..., 0], expected_x[:, :, None], atol=1e-5))
        self.assertTrue(np.allclose(out['history_grid'][..., 1], expected_y[:, :, None], atol=1e-5))

    def test_translation_changes_near_candidates_more_than_far_candidates(self):
        geom = synthetic_geometry()
        pose = np.eye(4, dtype=np.float64)
        pose[1, 3] = -1.0  # current point maps one meter to the right in previous camera image.
        out = build_ray_geometry_from_arrays(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            pose, geom, grid=(2, 4), num_depth_candidates=4, depth_min_m=4.0, depth_max_m=80.0)
        identity_x = (2.0 * (0.0 + 0.5) / 4.0) - 1.0
        near_shift = abs(float(out['history_grid'][0, 0, 0, 0] - identity_x))
        far_shift = abs(float(out['history_grid'][0, 0, -1, 0] - identity_x))
        self.assertGreater(near_shift, far_shift)


    def test_nonzero_projection_translation_backprojects_to_requested_rect_depth(self):
        geom = synthetic_geometry()
        geom.p_rect_02 = geom.p_rect_02.copy()
        geom.p_rect_02[:, 3] = np.asarray([-40.0, 6.0, 2.0], dtype=np.float64)
        rays = _camera_rays_for_feature_grid(geom, (2, 4), geom.image_size)
        points = _camera_depths_to_velo_points(geom, rays, np.asarray([7.0, 19.0], dtype=np.float32))
        uv, depth, valid = geom.project_velo_to_image(points.reshape(-1, 3), image_size=geom.image_size)
        self.assertTrue(valid.all())
        self.assertTrue(np.allclose(depth.reshape(2, 4, 2), np.asarray([7.0, 19.0]).reshape(1, 1, 2), atol=1e-5))
        feature = ((uv.reshape(2, 4, 2, 2) + 0.5) * np.asarray([4 / 8, 2 / 4], dtype=np.float32)) - 0.5
        yy, xx = np.indices((2, 4), dtype=np.float32)
        self.assertTrue(np.allclose(feature[..., 0], xx[:, :, None], atol=1e-5))
        self.assertTrue(np.allclose(feature[..., 1], yy[:, :, None], atol=1e-5))

    def test_satellite_uses_camera2_center_with_projection_and_imu_extrinsics(self):
        geom = synthetic_geometry()
        geom.p_rect_02 = geom.p_rect_02.copy()
        geom.p_rect_02[:, 3] = np.asarray([-8.0, 4.0, 0.0], dtype=np.float64)
        imu_to_velo = np.eye(4, dtype=np.float64)
        imu_to_velo[:3, 3] = np.asarray([3.0, -2.0, 0.5], dtype=np.float64)
        geom.t_imu_velo = imu_to_velo
        # Camera2 center in rect0 is -K^-1 P[:,3], then transformed back to velo.
        center = np.linalg.inv(geom.r_rect_00_ext @ geom.t_cam_velo) @ np.r_[
            -(np.linalg.inv(geom.p_rect_02[:, :3]) @ geom.p_rect_02[:, 3]), 1.0]
        sat, valid = _satellite_grid_from_velo_points(center[:3].reshape(1, 1, 1, 3), geom,
                                                      sat_size=256, meter_per_pixel=1.0)
        expected_center = np.asarray([0.0, 0.0], dtype=np.float32)
        self.assertTrue(valid.item())
        self.assertTrue(np.allclose(sat[0, 0, 0], expected_center, atol=1.0 / 256.0))

    def test_actual_point_centers_override_nearest_depth_candidate_and_sat_key(self):
        geom = synthetic_geometry()
        current_by_cell = cell_points(geom, depth=18.0)
        cur = current_by_cell.reshape(-1, 3)
        prev = cur.copy()
        out = build_ray_geometry_from_arrays(prev, cur, np.eye(4), geom, grid=(2, 4), num_depth_candidates=5,
                                             depth_min_m=4.0, depth_max_m=40.0,
                                             sat_size=128, sat_meter_per_pixel=1.0)
        self.assertTrue((out['source'] == SOURCE_STATIC_MEASUREMENT).any())
        yy, xx, kk = np.nonzero(out['source'] == SOURCE_STATIC_MEASUREMENT)
        log_min, log_max = np.log(4.0), np.log(40.0)
        for y, x, k in zip(yy.tolist(), xx.tolist(), kk.tolist()):
            self.assertEqual(out['positions'][y, x, k, 3], 1.0)
            depth = float(np.exp(out['positions'][y, x, k, 0] * (log_max - log_min) + log_min))
            self.assertAlmostEqual(depth, 18.0, places=4)
            self.assertAlmostEqual(float(out['positions'][y, x, k, 1]), float(current_by_cell[y, x, 2]) / 40.0, places=6)
            expected_sat = _satellite_grid_from_velo_points(
                current_by_cell[y:y + 1, x:x + 1, None, :], geom, sat_size=128, meter_per_pixel=1.0)[0][0, 0, 0]
            self.assertTrue(np.allclose(out['sat_grid'][y, x, k], expected_sat, atol=1e-6))
            bucket_point = cell_points(geom, depth=float(out['depth_candidates_m'][k]))[y:y + 1, x:x + 1, None, :]
            bucket_sat = _satellite_grid_from_velo_points(bucket_point, geom, sat_size=128, meter_per_pixel=1.0)[0][0, 0, 0]
            self.assertFalse(np.allclose(out['sat_grid'][y, x, k], bucket_sat, atol=1e-6))

    def test_occluded_measured_cell_becomes_content_fallback_not_static(self):
        geom = synthetic_geometry()
        cur_grid = cell_points(geom, depth=20.0)
        cur = cur_grid.reshape(-1, 3)
        prev = cur.copy()
        # Move one previous support point much closer in the same previous cell so depth support rejects it.
        prev[0] = cell_points(geom, depth=5.0).reshape(-1, 3)[0]
        out = build_ray_geometry_from_arrays(prev, cur, np.eye(4), geom, grid=(2, 4), num_depth_candidates=5,
                                             depth_min_m=4.0, depth_max_m=40.0, depth_tol_m=0.25)
        self.assertTrue((out['source'][0, 0] == SOURCE_CONTENT_FALLBACK).all())
        self.assertFalse((out['source'][0, 0] == SOURCE_STATIC_MEASUREMENT).any())
        self.assertFalse(out['valid'][0, 0].any())
        self.assertTrue(np.allclose(out['positions'][0, 0, :, 2], SOURCE_CONTENT_FALLBACK / 4.0))

    def test_dynamic_object_recovery_reuses_object_path(self):
        geom = synthetic_geometry(image_size=(64, 32))
        cluster = []
        for dx in np.linspace(-0.7, 0.7, 5):
            for dy in np.linspace(-0.7, 0.7, 5):
                for dz in np.linspace(-0.4, 0.4, 3):
                    cluster.append([18.0 + dy, dx, dz])
        cur_obj = np.asarray(cluster, dtype=np.float32)
        prev_obj = cur_obj.copy()
        prev_obj[:, 1] -= 2.0  # object moved right in previous after ego compensation.
        # Add enough ground points for the object recovery plane fit.
        xs = np.linspace(5.0, 28.0, 15)
        ys = np.linspace(-8.0, 8.0, 12)
        ground = np.asarray([[x, y, -1.7] for x in xs for y in ys], dtype=np.float32)
        cur = np.concatenate([cur_obj, ground], axis=0)
        prev = np.concatenate([prev_obj, ground], axis=0)
        out = build_ray_geometry_from_arrays(prev, cur, np.eye(4), geom, grid=(16, 32), num_depth_candidates=6,
                                             depth_min_m=4.0, depth_max_m=50.0, depth_tol_m=0.2)
        obj = out['source'] == SOURCE_OBJECT_COMP
        self.assertGreater(int(obj.sum()), 0)
        self.assertGreater(out['metrics']['object_cell_count'], 0)
        log_min, log_max = np.log(4.0), np.log(50.0)
        object_depths = np.exp(out['positions'][obj, 0] * (log_max - log_min) + log_min)
        self.assertTrue(((object_depths > 17.0) & (object_depths < 19.0)).all())
        self.assertTrue(np.all(np.abs(out['positions'][obj, 1]) < 0.03))
        for depth in object_depths:
            self.assertFalse(np.any(np.isclose(depth, out['depth_candidates_m'], atol=1e-5)))


    def test_unknown_ray_occlusion_rejects_only_known_behind_surface(self):
        geom = synthetic_geometry()
        near_surface = cell_points(geom, depth=5.0)[0:1, 0:1].reshape(1, 3)
        out = build_ray_geometry_from_arrays(
            near_surface, np.zeros((0, 3), dtype=np.float32), np.eye(4), geom,
            grid=(2, 4), num_depth_candidates=5, depth_min_m=2.0, depth_max_m=40.0,
            depth_tol_m=0.1)
        self.assertTrue(out['valid'][0, 0, 0])
        self.assertFalse(out['valid'][0, 0, -1])
        self.assertGreater(out['metrics']['occluded_unknown_ray_candidate_count'], 0)
        # A different cell has no previous z-buffer support, so uncertainty remains usable.
        self.assertTrue(out['valid'][1, 3].all())

    def test_satellite_axes_and_crop_alignment(self):
        geom = synthetic_geometry()
        pts = np.array([[[[10.0, 0.0, 0.0], [0.0, -10.0, 0.0], [0.0, 10.0, 0.0]]]], dtype=np.float32)
        grid, valid = _satellite_grid_from_velo_points(pts, geom, sat_size=256, meter_per_pixel=1.0)
        center = _satellite_grid_from_velo_points(np.zeros((1, 1, 1, 3), dtype=np.float32), geom, sat_size=256, meter_per_pixel=1.0)[0][0, 0, 0]
        forward, right, left = grid[0, 0]
        self.assertTrue(valid.all())
        self.assertGreater(forward[0], center[0])  # +velo x uses dataloader camera_forward/PIL-x.
        self.assertGreater(right[1], center[1])    # -velo y uses dataloader camera_right/PIL-y.
        self.assertLess(left[1], center[1])

    def test_satellite_coverage_does_not_clear_history_valid(self):
        geom = synthetic_geometry()
        out = build_ray_geometry_from_arrays(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            np.eye(4), geom, grid=(2, 4), num_depth_candidates=3, depth_min_m=20.0, depth_max_m=200.0,
            sat_size=8, sat_meter_per_pixel=0.1)
        self.assertTrue(out['valid'].all())
        self.assertFalse(out['sat_valid'].all())

    def test_invalid_inputs_rejected_and_outputs_finite(self):
        geom = synthetic_geometry()
        with self.assertRaises(ValueError):
            build_ray_geometry_from_arrays(np.zeros((0, 3)), np.zeros((0, 3)), np.eye(3), geom)
        with self.assertRaises(ValueError):
            build_ray_geometry_from_arrays(np.zeros((0, 3)), np.zeros((0, 3)), np.eye(4), geom,
                                           num_depth_candidates=0)
        out = build_ray_geometry_from_arrays(
            np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32),
            np.eye(4), geom, grid=(1, 1), num_depth_candidates=1, depth_min_m=2.0, depth_max_m=3.0)
        self.assertTrue(np.isfinite(out['history_grid']).all())
        self.assertTrue(np.isfinite(out['positions']).all())


if __name__ == '__main__':
    unittest.main()
