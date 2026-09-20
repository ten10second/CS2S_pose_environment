from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image

from dataloader.kitti_raw_lidar_utils import TrackletBox
from tools import visualize_static_history as vsh
from tools.temporal_static_geometry import (
    _project_velo_to_image_resized,
    build_static_history_from_arrays,
    build_static_pair,
)
from tools.temporal_history_geometry import RawKittiGeometry


def synthetic_geometry(image_size=(8, 4)):
    w, h = image_size
    p = np.array(
        [
            [20.0, 0.0, (w - 1) / 2.0, 0.0],
            [0.0, 20.0, (h - 1) / 2.0, 0.0],
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


def pixel_center_points(geometry, image_size=(8, 4), depth=20.0):
    w, h = image_size
    yy, xx = np.indices((h, w), dtype=np.float64)
    k = geometry.p_rect_02[:, :3]
    uv1 = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(h * w)], axis=0)
    cam = (np.linalg.inv(k) @ uv1).T * float(depth)
    cam_h = np.concatenate([cam, np.ones((len(cam), 1), dtype=np.float64)], axis=1).T
    velo = np.linalg.inv(geometry.r_rect_00_ext @ geometry.t_cam_velo) @ cam_h
    return velo.T[:, :3].astype(np.float32)


def rgb_pattern(width=8, height=4):
    yy, xx = np.indices((height, width), dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 0] = xx / max(width - 1, 1)
    rgb[..., 1] = yy / max(height - 1, 1)
    rgb[..., 2] = 0.25
    return rgb


def box_around(point, size=4.0):
    return TrackletBox(
        object_type="Car",
        class_id=1,
        frame_id=0,
        h=size,
        w=size,
        l=size,
        tx=float(point[0]),
        ty=float(point[1]),
        tz=float(point[2] - size / 2.0),
        rx=0.0,
        ry=0.0,
        rz=0.0,
    )


class TemporalStaticGeometryTests(unittest.TestCase):
    def test_identity_warps_visible_static_points_and_strict_mask(self):
        geom = synthetic_geometry()
        image_size = (8, 4)
        points = pixel_center_points(geom, image_size, depth=20.0)
        out = build_static_history_from_arrays(
            rgb_pattern(*image_size),
            points,
            points.copy(),
            np.eye(4),
            geom,
            image_size=image_size,
            prev_boxes=[],
            cur_boxes=[],
        )
        self.assertEqual(out["warped_rgb"].shape, (3, 4, 8))
        self.assertTrue(out["support_mask"].all())
        self.assertTrue(out["strict_mask"].all())
        self.assertTrue(np.allclose(out["warped_rgb"].transpose(1, 2, 0), rgb_pattern(*image_size)))
        self.assertEqual(out["diagnostics"]["support_count"], 32)
        self.assertEqual(out["diagnostics"]["strict_count"], 32)

    def test_translation_moves_reference_color_to_new_pixel(self):
        geom = synthetic_geometry()
        image_size = (8, 4)
        points = pixel_center_points(geom, image_size, depth=20.0)
        pose = np.eye(4, dtype=np.float64)
        pose[1, 3] = -1.0
        out = build_static_history_from_arrays(
            rgb_pattern(*image_size),
            points,
            np.zeros((0, 3), dtype=np.float32),
            pose,
            geom,
            image_size=image_size,
            prev_boxes=[],
            cur_boxes=[],
        )
        support = out["support_mask"][0]
        self.assertGreater(int(support.sum()), 0)
        self.assertFalse(np.allclose(out["warped_rgb"].transpose(1, 2, 0), rgb_pattern(*image_size)))
        ys, xs = np.nonzero(support)
        self.assertTrue((out["debug"]["support_xy"][ys, xs, 0] < xs).any())

    def test_source_zbuffer_uses_nearest_source_color(self):
        geom = synthetic_geometry()
        image_size = (8, 4)
        near = pixel_center_points(geom, image_size, depth=10.0)[0:1]
        far = pixel_center_points(geom, image_size, depth=20.0)[0:1]
        prev_points = np.concatenate([far, near], axis=0)
        cur_points = near.copy()
        rgb = np.zeros((4, 8, 3), dtype=np.float32)
        rgb[0, 0] = np.asarray([0.8, 0.2, 0.1], dtype=np.float32)
        out = build_static_history_from_arrays(
            rgb,
            prev_points,
            cur_points,
            np.eye(4),
            geom,
            image_size=image_size,
            prev_boxes=[],
            cur_boxes=[],
        )
        self.assertTrue(out["support_mask"][0, 0, 0])
        self.assertTrue(np.allclose(out["warped_rgb"][:, 0, 0], rgb[0, 0]))
        self.assertAlmostEqual(float(out["debug"]["projected_depth"][0, 0]), 10.0, places=5)

    def test_source_visibility_uses_all_points_before_dynamic_filter(self):
        geom = synthetic_geometry()
        image_size = (8, 4)
        foreground_car = pixel_center_points(geom, image_size, depth=10.0)[0:1]
        background_wall = pixel_center_points(geom, image_size, depth=20.0)[0:1]
        prev_points = np.concatenate([background_wall, foreground_car], axis=0)
        cur_points = background_wall.copy()
        out = build_static_history_from_arrays(
            rgb_pattern(*image_size),
            prev_points,
            cur_points,
            np.eye(4),
            geom,
            image_size=image_size,
            prev_boxes=[box_around(foreground_car[0])],
            cur_boxes=[],
            dynamic_exclusion_available=True,
        )
        self.assertEqual(out["diagnostics"]["support_count"], 0)
        self.assertEqual(out["diagnostics"]["source_dynamic_visible_reject_count"], 1)

    def test_target_occlusion_rejects_history_behind_current_surface(self):
        geom = synthetic_geometry()
        image_size = (8, 4)
        prev = pixel_center_points(geom, image_size, depth=20.0)[0:1]
        cur_near = pixel_center_points(geom, image_size, depth=5.0)[0:1]
        out = build_static_history_from_arrays(
            rgb_pattern(*image_size),
            prev,
            cur_near,
            np.eye(4),
            geom,
            image_size=image_size,
            depth_tol_m=0.25,
            prev_boxes=[],
            cur_boxes=[],
        )
        self.assertFalse(out["support_mask"][0, 0, 0])
        self.assertEqual(out["diagnostics"]["support_count"], 0)

    def test_target_visibility_keeps_dynamic_occluder_in_depth_buffer(self):
        geom = synthetic_geometry()
        image_size = (8, 4)
        prev_wall = pixel_center_points(geom, image_size, depth=20.0)[0:1]
        current_car = pixel_center_points(geom, image_size, depth=5.0)[0:1]
        out = build_static_history_from_arrays(
            rgb_pattern(*image_size),
            prev_wall,
            current_car,
            np.eye(4),
            geom,
            image_size=image_size,
            depth_tol_m=0.25,
            prev_boxes=[],
            cur_boxes=[box_around(current_car[0])],
            dynamic_exclusion_available=True,
        )
        self.assertEqual(out["diagnostics"]["support_count"], 0)
        self.assertGreaterEqual(out["diagnostics"]["target_dynamic_visible_reject_count"], 1)

    def test_dynamic_boxes_exclude_source_and_target_points(self):
        geom = synthetic_geometry()
        image_size = (8, 4)
        point = pixel_center_points(geom, image_size, depth=20.0)[0:1]
        box = box_around(point[0])
        source_out = build_static_history_from_arrays(
            rgb_pattern(*image_size),
            point,
            point.copy(),
            np.eye(4),
            geom,
            image_size=image_size,
            prev_boxes=[box],
            cur_boxes=[],
            dynamic_exclusion_available=True,
        )
        target_out = build_static_history_from_arrays(
            rgb_pattern(*image_size),
            point,
            point.copy(),
            np.eye(4),
            geom,
            image_size=image_size,
            prev_boxes=[],
            cur_boxes=[box],
            dynamic_exclusion_available=True,
        )
        self.assertEqual(source_out["diagnostics"]["support_count"], 0)
        self.assertEqual(target_out["diagnostics"]["support_count"], 0)
        self.assertEqual(source_out["diagnostics"]["source_dynamic_point_count"], 1)

    def test_empty_inputs_without_dynamic_exclusion_return_empty_geometry_diagnostic(self):
        geom = synthetic_geometry()
        out = build_static_history_from_arrays(
            rgb_pattern(),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.eye(4),
            geom,
            image_size=(8, 4),
            dynamic_exclusion_available=False,
        )
        self.assertFalse(out["support_mask"].any())
        self.assertFalse(out["diagnostics"]["zero_safe"])
        self.assertEqual(out["diagnostics"]["dependency"], "ego_motion_depth_consistency_without_instance_exclusion")
        self.assertTrue(out["diagnostics"]["target_dynamic_unknown"])

    def test_manual_static_regions_gate_source_and_target_without_tracklets(self):
        geom = synthetic_geometry()
        image_size = (8, 4)
        points = pixel_center_points(geom, image_size, depth=20.0)
        out = build_static_history_from_arrays(
            rgb_pattern(*image_size),
            points,
            points.copy(),
            np.eye(4),
            geom,
            image_size=image_size,
            dynamic_exclusion_available=False,
            static_regions={
                "prev_regions": [[0.0, 0.0, 0.5, 1.0]],
                "target_regions": [[0.0, 0.0, 0.5, 1.0]],
            },
        )
        self.assertEqual(out["diagnostics"]["dependency"], "manual_static_regions_offline_diagnostic")
        self.assertGreater(out["diagnostics"]["support_count"], 0)
        self.assertLess(out["diagnostics"]["support_count"], 32)
        self.assertFalse(out["support_mask"][0, :, 4:].any())

    def test_manual_static_regions_accepts_target_alias(self):
        geom = synthetic_geometry()
        image_size = (8, 4)
        points = pixel_center_points(geom, image_size, depth=20.0)
        out = build_static_history_from_arrays(
            rgb_pattern(*image_size),
            points,
            points.copy(),
            np.eye(4),
            geom,
            image_size=image_size,
            dynamic_exclusion_available=False,
            static_regions={"prev": [[0.0, 0.0, 1.0, 1.0]], "target": [[0.0, 0.0, 1.0, 1.0]]},
        )
        self.assertEqual(out["diagnostics"]["dependency"], "manual_static_regions_offline_diagnostic")
        self.assertEqual(out["diagnostics"]["support_count"], 32)

    def test_resized_projection_uses_pil_half_pixel_convention(self):
        geom = synthetic_geometry(image_size=(8, 4))
        point = pixel_center_points(geom, (8, 4), depth=20.0)[1 * 8 + 1 : 1 * 8 + 2]
        uv, depth, valid = _project_velo_to_image_resized(point, geom, (4, 2))
        self.assertTrue(valid[0])
        self.assertAlmostEqual(float(depth[0]), 20.0, places=5)
        self.assertTrue(np.allclose(uv[0], np.asarray([0.25, 0.25], dtype=np.float32), atol=1e-5))

    def test_invalid_inputs_are_rejected(self):
        geom = synthetic_geometry()
        with self.assertRaises(ValueError):
            build_static_history_from_arrays(
                rgb_pattern(),
                np.zeros((3,), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.eye(4),
                geom,
            )
        with self.assertRaises(ValueError):
            build_static_history_from_arrays(
                rgb_pattern(),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.eye(3),
                geom,
            )

    def test_build_pair_runs_without_tracklets_or_manual_regions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prev_image = root / "prev.png"
            arr = (np.arange(8 * 4 * 3, dtype=np.uint8).reshape(4, 8, 3))
            Image.fromarray(arr, mode="RGB").save(prev_image)
            prev = {
                "date": "2011_09_30",
                "drive": "2011_09_30_drive_0001_sync",
                "frame_index": 0,
                "frame_id": "0000000000",
                "sample_id": "a/0",
                "image_02_path": str(prev_image),
                "velodyne_path": str(root / "missing_prev.bin"),
                "oxts_path": str(root / "missing_prev.txt"),
                "calib_dir": str(root / "missing_calib"),
            }
            cur = {
                "date": "2011_09_30",
                "drive": "2011_09_30_drive_0001_sync",
                "frame_index": 1,
                "frame_id": "0000000001",
                "sample_id": "a/1",
                "image_02_path": str(root / "target_does_not_exist.png"),
                "velodyne_path": str(root / "missing_cur.bin"),
                "oxts_path": str(root / "missing_cur.txt"),
                "calib_dir": str(root / "missing_calib"),
            }
            geom = synthetic_geometry(image_size=(4, 2))
            geom.relative_velo_pose = lambda _prev, _cur: np.eye(4)
            points = pixel_center_points(geom, (4, 2), depth=20.0)
            with mock.patch("tools.temporal_static_geometry.get_geometry", return_value=geom), mock.patch(
                "tools.temporal_static_geometry.load_velodyne", side_effect=[points, points.copy()]
            ):
                out = build_static_pair(prev, cur, kitti_root=None, image_size=(4, 2))
            self.assertEqual(out["diagnostics"]["dependency"], "ego_motion_depth_consistency_without_instance_exclusion")
            self.assertTrue(out["diagnostics"]["target_dynamic_unknown"])
            self.assertEqual(out["diagnostics"]["diagnostic_missing_field_count"], 2)
            self.assertGreater(out["diagnostics"]["strict_count"], 0)

    def test_build_pair_does_not_open_missing_target_rgb_when_geometry_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prev_image = root / "prev.png"
            Image.fromarray((rgb_pattern(8, 4) * 255).astype(np.uint8), mode="RGB").save(prev_image)
            prev = {
                "date": "2011_09_30",
                "drive": "2011_09_30_drive_0001_sync",
                "frame_index": 0,
                "frame_id": "0000000000",
                "sample_id": "a/0",
                "image_02_path": str(prev_image),
                "velodyne_path": str(root / "missing_prev.bin"),
                "oxts_path": str(root / "missing_prev.txt"),
                "calib_dir": str(root / "calib"),
            }
            cur = {
                "date": "2011_09_30",
                "drive": "2011_09_30_drive_0001_sync",
                "frame_index": 1,
                "frame_id": "0000000001",
                "sample_id": "a/1",
                "image_02_path": str(root / "target_does_not_exist.png"),
                "velodyne_path": str(root / "missing_cur.bin"),
                "oxts_path": str(root / "missing_cur.txt"),
                "calib_dir": str(root / "calib"),
            }
            geom = synthetic_geometry()
            geom.relative_velo_pose = lambda _prev, _cur: np.eye(4)
            points = pixel_center_points(geom, (8, 4), depth=20.0)
            with mock.patch("tools.temporal_static_geometry.get_geometry", return_value=geom), mock.patch(
                "tools.temporal_static_geometry.load_velodyne", side_effect=[points, points.copy()]
            ):
                out = build_static_pair(
                    prev,
                    cur,
                    kitti_root=None,
                    image_size=(8, 4),
                    static_regions={"prev": [[0.0, 0.0, 1.0, 1.0]], "target": [[0.0, 0.0, 1.0, 1.0]]},
                )
            self.assertEqual(out["diagnostics"]["support_count"], 32)

    def test_zero_mask_saves_black_not_white(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zero.png"
            vsh._save_unit_gray(path, np.zeros((2, 3), dtype=np.float32))
            arr = np.asarray(Image.open(path))
            self.assertEqual(int(arr.max()), 0)

    def test_pairs_json_entries_preserve_order_and_target_regions(self):
        entries = [
            {"name": "b", "previous": "p2", "current": "c2", "static_regions": {"prev": [[0, 0, 1, 1]], "target": [[0, 0, 0.5, 1]]}},
            {"name": "a", "previous": "p1", "current": "c1", "static_regions": {"prev": [[0, 0, 1, 1]], "target": [[0.5, 0, 1, 1]]}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.json"
            path.write_text('{"pairs": ' + __import__("json").dumps(entries) + "}")
            loaded = vsh._load_region_entries(str(path))
        self.assertEqual([item["name"] for item in loaded], ["b", "a"])
        regions = vsh._regions_from_entry(loaded[0])
        self.assertEqual(regions["target_regions"], [[0, 0, 0.5, 1]])
        self.assertIsNone(vsh._regions_from_entry({"previous": "p", "current": "c", "name": "no_roi"}))


if __name__ == "__main__":
    unittest.main()
