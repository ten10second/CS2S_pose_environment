import numpy as np
import unittest

from tools.temporal_history_geometry import (
    RawKittiGeometry,
    align_corners_false_normalize,
    build_history_geometry_from_arrays,
    ground_proxy_mask,
)


def make_geometry(p_tx=0.0):
    return RawKittiGeometry(
        p_rect_02=np.asarray(
            [
                [100.0, 0.0, 50.0, p_tx],
                [0.0, 100.0, 25.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        r_rect_00_ext=np.eye(4, dtype=np.float64),
        t_cam_velo=np.eye(4, dtype=np.float64),
        t_imu_velo=np.eye(4, dtype=np.float64),
        image_size=(100, 50),
    )


def point_for_pixel(u, v, z=10.0, p_tx=0.0):
    x = ((u - 50.0) * z - p_tx) / 100.0
    y = (v - 25.0) * z / 100.0
    return [x, y, z]


def test_full_projection_uses_p_rect_translation_column():
    geom = make_geometry(p_tx=20.0)
    uv, depth, valid = geom.project_velo_to_image(np.asarray([[0.0, 0.0, 10.0]], dtype=np.float32))
    assert valid.tolist() == [True]
    assert np.allclose(depth, [10.0])
    assert np.allclose(uv[0], [52.0, 25.0])


def test_identity_builds_supported_previous_lookup_grid():
    geom = make_geometry()
    point = np.asarray([point_for_pixel(50.0, 25.0, 10.0)], dtype=np.float32)
    out = build_history_geometry_from_arrays(point, point, np.eye(4), geom, grid=(5, 10), depth_tol_m=0.2)
    valid = out["history_valid"]
    assert valid.sum() == 1
    y, x = np.argwhere(valid)[0]
    assert (y, x) == (2, 5)
    assert np.allclose(out["history_grid_px"][y, x], [5.0, 2.0])
    expected = align_corners_false_normalize(out["history_grid_px"][y:y + 1, x:x + 1], (5, 10))[0, 0]
    assert np.allclose(out["history_grid"][y, x], expected)


def test_offcenter_identity_maps_query_center_not_point_position():
    geom = make_geometry()
    point = np.asarray([point_for_pixel(54.0, 27.0, 10.0)], dtype=np.float32)
    out = build_history_geometry_from_arrays(point, point, np.eye(4), geom, grid=(5, 10), depth_tol_m=0.2)
    valid = out["history_valid"]
    assert valid.sum() == 1
    y, x = np.argwhere(valid)[0]
    assert np.allclose(out["history_grid_px"][y, x], [float(x), float(y)])


def test_translation_maps_current_cell_to_previous_coordinate():
    geom = make_geometry()
    prev_point = np.asarray([point_for_pixel(40.0, 25.0, 10.0)], dtype=np.float32)
    cur_point = np.asarray([point_for_pixel(50.0, 25.0, 10.0)], dtype=np.float32)
    cur_to_prev = np.eye(4)
    cur_to_prev[0, 3] = -1.0
    out = build_history_geometry_from_arrays(prev_point, cur_point, cur_to_prev, geom, grid=(5, 10), depth_tol_m=0.2)
    valid = out["history_valid"]
    assert valid.sum() == 1
    y, x = np.argwhere(valid)[0]
    assert (y, x) == (2, 5)
    assert np.allclose(out["history_grid_px"][y, x], [4.0, 2.0], atol=1e-5)
    assert np.isclose(out["metrics"]["mean_reprojection_vs_identity_cells"], 1.0)


def test_rotation_maps_current_query_by_point_displacement():
    geom = make_geometry()
    cur_point = np.asarray([point_for_pixel(60.0, 25.0, 10.0)], dtype=np.float32)
    prev_point = np.asarray([point_for_pixel(50.0, 35.0, 10.0)], dtype=np.float32)
    cur_to_prev = np.eye(4)
    cur_to_prev[:3, :3] = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    out = build_history_geometry_from_arrays(prev_point, cur_point, cur_to_prev, geom, grid=(5, 10), depth_tol_m=0.2)
    valid = out["history_valid"]
    assert valid.sum() == 1
    y, x = np.argwhere(valid)[0]
    assert (y, x) == (2, 6)
    assert np.allclose(out["history_grid_px"][y, x], [5.0, 3.0], atol=1e-5)


def test_occluded_previous_support_is_invalid_not_identity():
    geom = make_geometry()
    current_surface = np.asarray([point_for_pixel(50.0, 25.0, 10.0)], dtype=np.float32)
    closer_prev_surface = np.asarray([point_for_pixel(50.0, 25.0, 8.0)], dtype=np.float32)
    out = build_history_geometry_from_arrays(
        closer_prev_surface,
        current_surface,
        np.eye(4),
        geom,
        grid=(5, 10),
        depth_tol_m=0.2,
    )
    assert out["history_valid"].sum() == 0
    assert np.allclose(out["history_grid"], 0.0)


def test_unknown_previous_support_is_invalid():
    geom = make_geometry()
    current_surface = np.asarray([point_for_pixel(50.0, 25.0, 10.0)], dtype=np.float32)
    far_away_prev = np.asarray([point_for_pixel(80.0, 25.0, 10.0)], dtype=np.float32)
    out = build_history_geometry_from_arrays(far_away_prev, current_surface, np.eye(4), geom, grid=(5, 10))
    assert out["current_covered"].sum() == 1
    assert out["history_valid"].sum() == 0


def test_ground_proxy_plane_sign_marks_ground_and_rejects_elevated_points():
    xs = np.linspace(4.0, 18.0, 9)
    ys = np.linspace(-4.0, 4.0, 9)
    ground = np.asarray([[x, y, -1.5] for x in xs for y in ys], dtype=np.float32)
    elevated = np.asarray([[x, y, 0.0] for x in xs[:2] for y in ys[:2]], dtype=np.float32)
    mask = ground_proxy_mask(np.concatenate([ground, elevated], axis=0), threshold_m=0.2)
    assert mask[: len(ground)].all()
    assert not mask[len(ground):].any()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name in sorted(n for n in globals() if n.startswith("test_")):
        suite.addTest(unittest.FunctionTestCase(globals()[name]))
    return suite


if __name__ == "__main__":
    unittest.main()
