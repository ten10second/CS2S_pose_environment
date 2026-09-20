from __future__ import annotations

import unittest

import numpy as np

from tools.temporal_dense_reprojection import build_dense_reference, calibrate_depth
from tools.temporal_history_geometry import RawKittiGeometry


def synthetic_geometry(image_size=(8, 4), projection_translation=False):
    w, h = image_size
    p = np.array(
        [
            [20.0, 0.0, (w - 1) / 2.0, 0.0],
            [0.0, 20.0, (h - 1) / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    if projection_translation:
        p[:, 3] = np.asarray([-40.0, 6.0, 2.0], dtype=np.float64)
    t_cam_velo = np.eye(4, dtype=np.float64)
    t_cam_velo[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return RawKittiGeometry(
        p_rect_02=p,
        r_rect_00_ext=np.eye(4, dtype=np.float64),
        t_cam_velo=t_cam_velo,
        t_imu_velo=None,
        image_size=image_size,
    )


def rgb_pattern(width=8, height=4):
    yy, xx = np.indices((height, width), dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 0] = xx / max(width - 1, 1)
    rgb[..., 1] = yy / max(height - 1, 1)
    rgb[..., 2] = 0.5
    return rgb


def dense_points_from_depth(geometry, depth_map):
    h, w = depth_map.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    valid = np.isfinite(depth_map) & (depth_map > 0.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    p = geometry.p_rect_02
    inv_k = np.linalg.inv(p[:, :3])
    b = inv_k @ p[:, 3]
    uv1 = np.stack([xx[valid], yy[valid], np.ones(int(valid.sum()))], axis=0)
    rays = (inv_k @ uv1).T
    scale = (depth_map[valid].astype(np.float64) + b[2]) / rays[:, 2]
    rect = rays * scale[:, None] - b[None, :]
    rect_h = np.concatenate([rect, np.ones((len(rect), 1), dtype=np.float64)], axis=1).T
    velo = np.linalg.inv(geometry.r_rect_00_ext @ geometry.t_cam_velo) @ rect_h
    return velo.T[:, :3].astype(np.float32)


class DenseReprojectionTests(unittest.TestCase):
    def test_identity_reprojects_rgb_and_marks_measured(self):
        geom = synthetic_geometry()
        depth = np.full((4, 8), 20.0, dtype=np.float32)
        points = dense_points_from_depth(geom, depth)
        out = build_dense_reference(rgb_pattern(), depth, points, points.copy(), geom, np.eye(4), depth_tol_m=0.1)
        self.assertTrue(out["support_mask"].all())
        self.assertTrue(out["measured_mask"].all())
        self.assertFalse(out["estimated_mask"].any())
        self.assertTrue(np.allclose(out["warped_rgb"], rgb_pattern()))

    def test_nonzero_projection_translation_round_trips(self):
        geom = synthetic_geometry(projection_translation=True)
        depth = np.full((4, 8), 19.0, dtype=np.float32)
        points = dense_points_from_depth(geom, depth)
        out = build_dense_reference(rgb_pattern(), depth, points, points.copy(), geom, np.eye(4), depth_tol_m=1e-4)
        self.assertTrue(out["support_mask"].all())
        self.assertTrue(np.allclose(out["source_uv"][out["support_mask"]][:, 0] % 1.0, 0.0, atol=1e-5))
        self.assertTrue(np.allclose(out["warped_rgb"], rgb_pattern(), atol=1e-5))

    def test_translation_moves_source_coordinates(self):
        geom = synthetic_geometry()
        depth = np.full((4, 8), 20.0, dtype=np.float32)
        points = dense_points_from_depth(geom, depth)
        pose = np.eye(4, dtype=np.float64)
        pose[1, 3] = -1.0
        out = build_dense_reference(rgb_pattern(), depth, points, np.zeros((0, 3), dtype=np.float32), geom, pose)
        ys, xs = np.nonzero(out["support_mask"])
        self.assertGreater(len(xs), 0)
        self.assertTrue((out["source_uv"][ys, xs, 0] < xs).any())

    def test_target_occlusion_rejects_conflicted_projection(self):
        geom = synthetic_geometry()
        depth = np.full((4, 8), np.nan, dtype=np.float32)
        depth[0, 0] = 20.0
        prev_points = dense_points_from_depth(geom, depth)
        near_depth = np.full((4, 8), np.nan, dtype=np.float32)
        near_depth[0, 0] = 5.0
        cur_points = dense_points_from_depth(geom, near_depth)
        out = build_dense_reference(rgb_pattern(), depth, prev_points, cur_points, geom, np.eye(4), depth_tol_m=0.25)
        self.assertFalse(out["support_mask"][0, 0])
        self.assertTrue(out["target_conflict_mask"][0, 0])
        self.assertEqual(out["diagnostics"]["target_conflict_count"], 1)

    def test_measured_requires_source_and_target_lidar(self):
        geom = synthetic_geometry()
        depth = np.full((4, 8), 20.0, dtype=np.float32)
        all_points = dense_points_from_depth(geom, depth)
        source_one = all_points[0:1]
        out = build_dense_reference(rgb_pattern(), depth, source_one, all_points.copy(), geom, np.eye(4), depth_tol_m=0.1)
        self.assertTrue(out["measured_mask"][0, 0])
        self.assertFalse(out["measured_mask"][0, 1])
        self.assertTrue(out["estimated_mask"][0, 1])
        missing_target = build_dense_reference(rgb_pattern(), depth, source_one, np.zeros((0, 3), dtype=np.float32), geom, np.eye(4), depth_tol_m=0.1)
        self.assertFalse(missing_target["measured_mask"][0, 0])
        self.assertTrue(missing_target["estimated_mask"][0, 0])

    def test_nonfinite_empty_depth_is_safe(self):
        geom = synthetic_geometry()
        depth = np.full((4, 8), np.nan, dtype=np.float32)
        out = build_dense_reference(
            rgb_pattern(),
            depth,
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            geom,
            np.eye(4),
        )
        self.assertFalse(out["support_mask"].any())
        self.assertEqual(out["diagnostics"]["dense_source_count"], 0)

    def test_calibrate_depth_uses_source_sparse_depth_median_scale(self):
        pred = np.array([[10.0, 20.0], [30.0, np.nan]], dtype=np.float32)
        sparse = np.array([[20.0, 40.0], [60.0, np.inf]], dtype=np.float32)
        calibrated, diag = calibrate_depth(pred, sparse, np.ones((2, 2), dtype=bool))
        self.assertAlmostEqual(diag["scale"], 2.0)
        self.assertTrue(np.allclose(calibrated[:2, :1].reshape(-1), np.array([20.0, 60.0], dtype=np.float32)))
        self.assertEqual(diag["fit_count"], 3)


if __name__ == "__main__":
    unittest.main()
