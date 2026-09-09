import os
import sys
import unittest

import numpy as np


ROOT = os.path.dirname(os.path.dirname(__file__))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import lidar_object_association as loa  # noqa: E402


class SimpleGeom:
    def project_velo_to_rect_img(self, points):
        return points[:, 0] * 10.0 + 100.0, points[:, 1] * 10.0 + 50.0, points[:, 2] + 10.0


def make_cluster(center, n=12, z=0.5):
    offsets = []
    for i in range(n):
        offsets.append(((i % 4) * 0.05, ((i // 4) % 4) * 0.05, 0.0))
    pts = np.asarray(offsets, dtype=np.float32)
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    pts[:, 2] += z
    return pts


class LidarObjectAssociationTest(unittest.TestCase):
    def setUp(self):
        self.geom = SimpleGeom()
        self.T = np.eye(4, dtype=np.float32)
        self.plane = (0.0, 0.0, 0.0)

    def associate(self, prev, cur, **kwargs):
        opts = {
            "img_w": 400,
            "img_h": 200,
            "lat_h": 20,
            "lat_w": 40,
            "plane": self.plane,
            "cluster_voxel": 0.01,
        }
        opts.update(kwargs)
        return loa.associate_objects(prev, cur, self.T, self.geom, **opts)

    def test_cluster_objects_keeps_isolated_points_separate_without_edges(self):
        pts = np.asarray(
            [[0.0, 0.0, 0.5], [10.0, 0.0, 0.5], [20.0, 0.0, 0.5]],
            dtype=np.float32,
        )
        clusters = loa.cluster_objects(pts, radius=0.8, min_points=1)
        self.assertEqual(len(clusters), 3)
        self.assertEqual([c["count"] for c in clusters], [1, 1, 1])

    def test_greedy_match_reserves_actual_best_index(self):
        prev = np.vstack([make_cluster((10.0, 0.0)), make_cluster((20.0, 0.0))])
        cur = np.vstack([make_cluster((10.5, 0.0)), make_cluster((20.5, 0.0))])

        matched, stats = self.associate(prev, cur, match_radius=12.0, motion_gate=2.0)

        self.assertEqual(stats["n_matched"], 2)
        self.assertEqual(len(matched), 2)
        self.assertTrue(all(np.linalg.norm(obj["d_velo"]) < 1.0 for obj in matched))

    def test_rejected_match_does_not_consume_previous_candidate(self):
        prev = make_cluster((0.0, 0.0), n=12)
        rejected = make_cluster((3.0, 0.0), n=16)
        accepted = make_cluster((0.6, 0.0), n=12)
        cur = np.vstack([rejected, accepted])

        matched, stats = self.associate(cur=cur, prev=prev, match_radius=5.0, motion_gate=2.0)

        self.assertEqual(stats["n_matched"], 1)
        self.assertEqual(len(matched), 1)
        self.assertLess(np.linalg.norm(matched[0]["d_velo"]), 1.0)

    def test_match_exposes_current_and_previous_latent_masks(self):
        prev = make_cluster((1.0, 0.0), n=12)
        cur = make_cluster((4.0, 0.0), n=12)

        matched, _ = self.associate(prev, cur, match_radius=4.0, motion_gate=4.0)

        self.assertEqual(len(matched), 1)
        obj = matched[0]
        self.assertIn("cell_mask", obj)
        self.assertIn("prev_cell_mask", obj)
        self.assertEqual(obj["cell_mask"].shape, (20, 40))
        self.assertEqual(obj["prev_cell_mask"].shape, (20, 40))
        self.assertTrue(obj["cell_mask"].any())
        self.assertTrue(obj["prev_cell_mask"].any())
        self.assertFalse(np.array_equal(obj["cell_mask"], obj["prev_cell_mask"]))

    def test_current_candidates_use_plane_transformed_from_previous_frame(self):
        T = np.eye(4, dtype=np.float32)
        T[2, 3] = 1.0
        cur_on_current_ground = np.asarray([[0.0, 0.0, 4.5]], dtype=np.float32)

        prev_plane_as_current = loa.above_ground_mask(cur_on_current_ground, self.plane)
        transformed_plane = loa.transform_ground_plane(self.plane, T)
        current_frame_mask = loa.above_ground_mask(cur_on_current_ground, transformed_plane)

        self.assertFalse(prev_plane_as_current[0])
        self.assertTrue(current_frame_mask[0])

    def test_transform_ground_plane_returns_none_for_near_vertical_plane(self):
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        self.assertIsNone(loa.transform_ground_plane(self.plane, T))
