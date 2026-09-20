from __future__ import annotations

import unittest

import numpy as np

from tools.temporal_dense_eval import (
    filter_known_pose_matches,
    known_pose_fundamental,
    masked_image_diagnostics,
    projected_match_errors,
    project_source_pixels_with_depth,
    sampson_epipolar_error,
)
from tools.temporal_history_geometry import RawKittiGeometry


def synthetic_geometry(image_size=(16, 8)):
    w, h = image_size
    p = np.array(
        [
            [30.0, 0.0, (w - 1) / 2.0, 0.0],
            [0.0, 30.0, (h - 1) / 2.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
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


class TemporalDenseEvalTests(unittest.TestCase):
    def test_known_pose_filter_keeps_exact_projection_and_rejects_outlier(self):
        geom = synthetic_geometry()
        depth = np.full((8, 16), 20.0, dtype=np.float32)
        source_xy = np.asarray([[5.0, 3.0], [9.0, 4.0]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float64)
        pose[1, 3] = -1.0
        projected = project_source_pixels_with_depth(source_xy, depth, geom, pose)
        target_xy = projected["target_xy"].copy()
        target_xy[1, 1] += 4.0

        out = filter_known_pose_matches(source_xy, target_xy, geom, pose, (16, 8), max_sampson_px=0.5)

        self.assertEqual(out["diagnostics"]["candidate_match_count"], 2)
        self.assertEqual(out["diagnostics"]["epipolar_inlier_count"], 1)
        self.assertTrue(np.allclose(out["source_xy"], source_xy[:1]))
        self.assertEqual(out["diagnostics"]["upper_image_candidate_count"], 1)
        self.assertEqual(out["diagnostics"]["upper_image_inlier_count"], 1)

    def test_known_pose_filter_reports_source_no_lidar_matches(self):
        geom = synthetic_geometry()
        depth = np.full((8, 16), 20.0, dtype=np.float32)
        source_xy = np.asarray([[5.0, 3.0], [9.0, 4.0]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float64)
        pose[1, 3] = -1.0
        target_xy = project_source_pixels_with_depth(source_xy, depth, geom, pose)["target_xy"]
        source_lidar_mask = np.zeros((8, 16), dtype=bool)
        source_lidar_mask[3, 5] = True

        out = filter_known_pose_matches(
            source_xy,
            target_xy,
            geom,
            pose,
            (16, 8),
            max_sampson_px=0.5,
            source_lidar_mask=source_lidar_mask,
        )

        self.assertEqual(out["diagnostics"]["source_lidar_inlier_count"], 1)
        self.assertEqual(out["diagnostics"]["source_no_lidar_inlier_count"], 1)

    def test_sampson_error_is_small_for_projected_source_depth(self):
        geom = synthetic_geometry()
        depth = np.full((8, 16), 18.0, dtype=np.float32)
        source_xy = np.asarray([[4.0, 2.0], [11.0, 5.0]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float64)
        pose[1, 3] = -0.75
        target_xy = project_source_pixels_with_depth(source_xy, depth, geom, pose)["target_xy"]
        fundamental = known_pose_fundamental(geom, pose, (16, 8))

        err = sampson_epipolar_error(source_xy, target_xy, fundamental)

        self.assertTrue(np.all(err < 1e-6), err)

    def test_projected_match_errors_accepts_direct_projection_or_depth(self):
        geom = synthetic_geometry()
        depth = np.full((8, 16), 20.0, dtype=np.float32)
        source_xy = np.asarray([[5.0, 3.0], [9.0, 4.0]], dtype=np.float32)
        pose = np.eye(4, dtype=np.float64)
        pose[1, 3] = -1.0
        target_xy = project_source_pixels_with_depth(source_xy, depth, geom, pose)["target_xy"]

        direct = projected_match_errors(source_xy, target_xy, projected_target_xy=target_xy)
        from_depth = projected_match_errors(source_xy, target_xy, pred_depth=depth, geometry=geom, prev_to_cur_velo=pose)

        self.assertEqual(direct["diagnostics"]["valid_count"], 2)
        self.assertAlmostEqual(direct["diagnostics"]["mean_error_px"], 0.0, places=6)
        self.assertAlmostEqual(from_depth["diagnostics"]["median_error_px"], 0.0, places=6)

    def test_masked_image_diagnostics_uses_fixed_common_support(self):
        pred = np.zeros((3, 4, 3), dtype=np.float32)
        target = np.zeros((3, 4, 3), dtype=np.float32)
        pred[1, 1] = 1.0
        target[1, 1] = 0.25
        target[0, 0] = 1.0
        pred_support = np.zeros((3, 4), dtype=bool)
        pred_support[1, 1:3] = True
        target_support = np.zeros((3, 4), dtype=bool)
        target_support[1, :2] = True

        out = masked_image_diagnostics(pred, target, pred_support, target_support)

        self.assertEqual(out["common_count"], 1.0)
        self.assertAlmostEqual(out["rgb_l1"], 0.75, places=6)
        self.assertTrue(np.isnan(out["gradient_l1"]))
        self.assertAlmostEqual(out["common_coverage"], 1.0 / 12.0, places=6)

    def test_masked_image_diagnostics_gradient_on_common_edges(self):
        pred = np.zeros((2, 3, 3), dtype=np.float32)
        target = np.zeros((2, 3, 3), dtype=np.float32)
        pred[:, 1] = 1.0
        target[:, 2] = 1.0
        support = np.ones((2, 3), dtype=bool)

        out = masked_image_diagnostics(pred, target, support)

        self.assertEqual(out["gradient_pair_count"], 7.0)
        self.assertGreater(out["gradient_l1"], 0.0)


if __name__ == "__main__":
    unittest.main()
