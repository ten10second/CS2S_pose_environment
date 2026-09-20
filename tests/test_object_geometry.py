"""Moving-object recovery is ego-rejected cell fill, not a color-stat shortcut."""
from __future__ import annotations

import unittest

import numpy as np

from tools import temporal_history_geometry as geo
from tools.temporal_object_geometry import (
    associate_clusters,
    cluster_nonground_points,
    estimate_object_transform,
)


def _blob(center, n=24, scale=0.08, seed=0):
    rng = np.random.RandomState(seed)
    return (np.asarray(center, np.float32) + rng.randn(n, 3).astype(np.float32) * scale)


def _ground(n=80, seed=1):
    rng = np.random.RandomState(seed)
    ang = rng.rand(n) * 2 * np.pi
    rad = 4.0 + rng.rand(n) * 6.0
    pts = np.stack([rad * np.cos(ang), rad * np.sin(ang), np.full(n, -1.6)], axis=1)
    return pts.astype(np.float32)


class ObjectGeometryTests(unittest.TestCase):
    def test_cluster_separates_two_blobs_and_rejects_facade(self):
        a = _blob([2.0, 0.0, 0.5], seed=0)
        b = _blob([-2.0, 0.0, 0.5], seed=1)
        clusters = cluster_nonground_points(np.concatenate([a, b], 0), voxel_m=0.4, min_points=8)
        self.assertEqual(len(clusters), 2)
        facade = np.mgrid[-8:8:0.4, -1:1:0.4, 0:1:0.4]
        facade = np.stack(facade, axis=-1).reshape(-1, 3).astype(np.float32)
        self.assertEqual(cluster_nonground_points(facade, min_points=8, max_extent_m=12.0), [])

    def test_association_is_mutual_after_ego(self):
        cur = [_blob([1.0, 0.0, 0.2], seed=2), _blob([4.0, 1.0, 0.2], seed=3)]
        cur = [{"points": c, "centroid": c.mean(0), "indices": np.arange(len(c))} for c in cur]
        pose = np.eye(4)
        pose[0, 3] = 0.4
        prev = []
        for c in cur:
            pts = geo.transform_points(c["points"], pose)
            prev.append({"points": pts, "centroid": pts.mean(0), "indices": np.arange(len(pts))})
        matches = associate_clusters(cur, prev, pose, max_dist_m=1.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual({m["current_index"] for m in matches}, {0, 1})

    def test_stationary_after_ego_is_not_moving(self):
        cur = _blob([1.0, 0.0, 0.4], seed=4)
        pose = np.eye(4)
        pose[0, 3] = 0.5
        prev = geo.transform_points(cur, pose)
        out = estimate_object_transform(cur, prev, pose)
        self.assertFalse(out["accepted"])
        self.assertEqual(out["reason"], "stationary")
        self.assertLess(out["ego_centroid_shift_m"], 0.2)

    def test_object_translation_is_recovered(self):
        cur = _blob([1.0, 0.0, 0.4], seed=5)
        prev = cur.copy()
        prev[:, 0] += 0.7
        out = estimate_object_transform(cur, prev, np.eye(4))
        self.assertTrue(out["accepted"])
        mapped = geo.transform_points(cur, out["transform"])
        self.assertLess(np.linalg.norm(mapped.mean(0) - prev.mean(0)), 0.08)


class ObjectRecoveryInHistoryTests(unittest.TestCase):
    def setUp(self):
        self.grid = (3, 5)
        k = np.array([[4., 0., 2.], [0., 4., 1.], [0., 0., 1.]])
        self.calib = geo.RawKittiGeometry(
            np.column_stack([k, [-.4, 0., 0.]]),
            np.eye(4), np.eye(4), np.eye(4), (5, 3))

    def test_parked_blob_stays_on_lidar_path(self):
        car = _blob([0.1, 0.0, 4.0], n=20, scale=0.05, seed=6)
        pose = np.eye(4)
        pose[0, 3] = 0.4
        prev = np.concatenate([_ground(), geo.transform_points(car, pose)], 0)
        cur = np.concatenate([_ground(seed=2), car], 0)
        result = geo.build_history_geometry_from_arrays(prev, cur, pose, self.calib, grid=self.grid)
        self.assertTrue(result["lidar_valid"].any())
        self.assertFalse(result["object_valid"].any())
        self.assertEqual(result["metrics"]["object_moving_accepted"], 0)

    def test_moving_blob_recovers_ego_rejected_cells(self):
        car = _blob([0.1, 0.0, 4.0], n=24, scale=0.04, seed=7)
        moved = car.copy()
        moved[:, 0] += 0.55
        prev = np.concatenate([_ground(), moved], 0)
        cur = np.concatenate([_ground(seed=3), car], 0)
        result = geo.build_history_geometry_from_arrays(prev, cur, np.eye(4), self.calib, grid=self.grid)
        rejected = result["current_covered"] & ~result["lidar_valid"]
        self.assertTrue(rejected.any())
        self.assertTrue(result["object_valid"].any())
        self.assertFalse(result["lidar_valid"][result["object_valid"]].any())
        self.assertFalse(result["rotation_valid"][result["object_valid"]].any())
        self.assertTrue((result["history_source"][result["object_valid"]] == 3).all())
        self.assertGreater(result["metrics"]["object_moving_accepted"], 0)

    def test_unassociated_rejected_cells_stay_invalid(self):
        car = _blob([0.1, 0.0, 4.0], n=24, scale=0.04, seed=8)
        prev = _ground()
        cur = np.concatenate([_ground(seed=4), car], 0)
        result = geo.build_history_geometry_from_arrays(prev, cur, np.eye(4), self.calib, grid=self.grid)
        covered = result["current_covered"]
        self.assertTrue(covered.any())
        self.assertFalse(result["object_valid"].any())
        self.assertFalse(result["history_valid"][covered].any())
        self.assertTrue(result["known_rejected"][covered].all())


if __name__ == "__main__":
    unittest.main()
