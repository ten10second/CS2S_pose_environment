"""Multi-depth temporal ray geometry for persistent history conditioning.

The output is current-cell -> previous-history geometry.  Each current feature
cell receives K candidates.  LiDAR/object correspondences overwrite the nearest
candidate as measured evidence; cells without current LiDAR keep positive-depth
ray candidates across the whole image.  Ego-rejected measured cells that cannot
be object recovered are marked as content fallback instead of being promoted to a
strong static correspondence.  positions[..., 1] is normalized physical
height using Velodyne z / 40m, not image row position.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from dataloader import KITTI_utils as kitti_utils
from tools.temporal_history_geometry import (
    RawKittiGeometry,
    _nearest_depth_by_cell,
    align_corners_false_normalize,
    consecutive_rows,
    get_geometry,
    image_to_feature_xy,
    load_velodyne,
    rebase_kitti_path,
    transform_points,
)
from tools.temporal_object_geometry import recover_moving_object_history

SOURCE_INVALID = 0
SOURCE_UNKNOWN_RAY = 1
SOURCE_STATIC_MEASUREMENT = 2
SOURCE_OBJECT_COMP = 3
SOURCE_CONTENT_FALLBACK = 4


def _require_grid(grid: Tuple[int, int]) -> Tuple[int, int]:
    if (
        len(grid) != 2
        or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or int(v) <= 0 for v in grid)
    ):
        raise ValueError("grid must be a pair of positive integers")
    return int(grid[0]), int(grid[1])


def _require_pose(pose: np.ndarray) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        raise ValueError("cur_to_prev_velo must be a finite 4x4 matrix")
    if not np.allclose(arr[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("cur_to_prev_velo must be homogeneous")
    return arr


def _inverse_depth_candidates(num_depth_candidates: int, depth_min_m: float, depth_max_m: float) -> np.ndarray:
    if not isinstance(num_depth_candidates, (int, np.integer)) or isinstance(num_depth_candidates, bool) or num_depth_candidates < 1:
        raise ValueError("num_depth_candidates must be a positive integer")
    if not np.isfinite(depth_min_m) or not np.isfinite(depth_max_m) or depth_min_m <= 0 or depth_max_m <= depth_min_m:
        raise ValueError("depth range must be finite and satisfy 0 < min < max")
    inv = np.linspace(1.0 / float(depth_min_m), 1.0 / float(depth_max_m), int(num_depth_candidates), dtype=np.float64)
    return (1.0 / inv).astype(np.float32)


def _camera_rays_for_feature_grid(
    geometry: RawKittiGeometry,
    grid: Tuple[int, int],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Return rectified camera rays through current feature-cell centers."""
    h, w = grid
    img_w, img_h = image_size
    src_w, src_h = geometry.image_size
    yy, xx = np.indices((h, w), dtype=np.float64)
    # Inverse of image_to_feature_xy and project_velo_to_image resizing.
    u = ((xx + 0.5) * img_w / float(w) - 0.5) * src_w / float(img_w)
    v = ((yy + 0.5) * img_h / float(h) - 0.5) * src_h / float(img_h)
    k = np.asarray(geometry.p_rect_02[:, :3], dtype=np.float64)
    try:
        inv_k = np.linalg.inv(k)
    except np.linalg.LinAlgError as error:
        raise ValueError("camera calibration must be invertible") from error
    rays = (inv_k @ np.stack([u, v, np.ones_like(u)], axis=0).reshape(3, -1)).T.reshape(h, w, 3)
    if not np.isfinite(rays).all():
        raise ValueError("camera rays are not finite")
    return rays


def _camera_depths_to_velo_points(
    geometry: RawKittiGeometry,
    rays_cam: np.ndarray,
    depths: np.ndarray,
) -> np.ndarray:
    h, w, _ = rays_cam.shape
    k = len(depths)
    p = np.asarray(geometry.p_rect_02, dtype=np.float64)
    k_cam = p[:, :3]
    try:
        b = np.linalg.inv(k_cam) @ p[:, 3]
    except np.linalg.LinAlgError as error:
        raise ValueError("camera projection matrix must be invertible") from error
    rays = np.asarray(rays_cam, dtype=np.float64)
    rz = rays[..., 2]
    if not np.all(np.abs(rz) > 1e-9):
        raise ValueError("camera rays must have nonzero z")
    scale = (depths.reshape(1, 1, k) + float(b[2])) / rz[:, :, None]
    points_rect = rays[:, :, None, :] * scale[..., None] - b.reshape(1, 1, 1, 3)
    points_h = np.concatenate([points_rect.reshape(-1, 3), np.ones((h * w * k, 1), dtype=np.float64)], axis=1).T
    try:
        velo_h = np.linalg.inv(geometry.r_rect_00_ext @ geometry.t_cam_velo) @ points_h
    except np.linalg.LinAlgError as error:
        raise ValueError("camera-to-velo calibration must be invertible") from error
    return velo_h.T[:, :3].reshape(h, w, k, 3).astype(np.float32)


def _project_feature_candidates(
    points_velo: np.ndarray,
    geometry: RawKittiGeometry,
    grid: Tuple[int, int],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = points_velo.reshape(-1, 3)
    uv, depth, in_image = geometry.project_velo_to_image(flat, image_size=image_size)
    feature_xy = image_to_feature_xy(uv, image_size, grid)
    h, w, k = points_velo.shape[:3]
    feature_xy = feature_xy.reshape(h, w, k, 2)
    depth = depth.reshape(h, w, k)
    in_image = in_image.reshape(h, w, k)
    eps = 1e-5
    in_feature = (
        in_image
        & np.isfinite(feature_xy).all(axis=-1)
        & (feature_xy[..., 0] >= -eps)
        & (feature_xy[..., 0] <= grid[1] - 1.0 + eps)
        & (feature_xy[..., 1] >= -eps)
        & (feature_xy[..., 1] <= grid[0] - 1.0 + eps)
        & np.isfinite(depth)
        & (depth > 0.0)
    )
    clipped = feature_xy.copy()
    clipped[..., 0] = np.clip(clipped[..., 0], 0.0, grid[1] - 1.0)
    clipped[..., 1] = np.clip(clipped[..., 1], 0.0, grid[0] - 1.0)
    norm = align_corners_false_normalize(clipped, grid)
    norm[~in_feature] = 0.0
    feature_xy[~in_feature] = 0.0
    return norm.astype(np.float32), feature_xy.astype(np.float32), in_feature.astype(bool), depth.astype(np.float32)


def _previous_depth_zbuffer(
    prev_points_velo: np.ndarray,
    geometry: RawKittiGeometry,
    grid: Tuple[int, int],
    image_size: Tuple[int, int],
) -> np.ndarray:
    points = np.asarray(prev_points_velo, dtype=np.float32)
    h, w = grid
    if points.size == 0:
        return np.full((h, w), np.inf, dtype=np.float32)
    uv, depth, in_image = geometry.project_velo_to_image(points, image_size=image_size)
    z, _idx = _nearest_depth_by_cell(uv, depth, in_image, grid, image_size)
    return z.astype(np.float32)


def _occlusion_mask_from_previous_depth(
    history_px: np.ndarray,
    prev_depth: np.ndarray,
    prev_zbuffer: np.ndarray,
    valid: np.ndarray,
    depth_tol_m: float,
) -> np.ndarray:
    """Reject candidates behind a known previous-frame nearer LiDAR surface.

    Cells without previous LiDAR depth stay usable: absence of depth is unknown,
    not evidence for rejection.  The tolerance only handles z-buffer support;
    it is not a same-depth requirement.
    """
    if not np.isfinite(depth_tol_m) or depth_tol_m < 0:
        raise ValueError("depth_tol_m must be finite and nonnegative")
    h, w = prev_zbuffer.shape
    cell_x = np.floor(history_px[..., 0] + 0.5).astype(np.int64)
    cell_y = np.floor(history_px[..., 1] + 0.5).astype(np.int64)
    in_cell = valid & (cell_x >= 0) & (cell_x < w) & (cell_y >= 0) & (cell_y < h)
    nearest = np.full(valid.shape, np.inf, dtype=np.float32)
    nearest[in_cell] = prev_zbuffer[cell_y[in_cell], cell_x[in_cell]]
    known = np.isfinite(nearest)
    return known & np.isfinite(prev_depth) & (prev_depth > nearest + float(depth_tol_m))


def _camera2_center_in_velo(geometry: RawKittiGeometry) -> np.ndarray:
    p = np.asarray(geometry.p_rect_02, dtype=np.float64)
    try:
        center_rect0 = -(np.linalg.inv(p[:, :3]) @ p[:, 3])
        rect_to_velo = np.linalg.inv(geometry.r_rect_00_ext @ geometry.t_cam_velo)
    except np.linalg.LinAlgError as error:
        raise ValueError("camera calibration must be invertible") from error
    center_h = rect_to_velo @ np.asarray([center_rect0[0], center_rect0[1], center_rect0[2], 1.0], dtype=np.float64)
    return center_h[:3].astype(np.float64)


def _velo_to_imu_local(points_velo: np.ndarray, geometry: RawKittiGeometry) -> np.ndarray:
    pts = np.asarray(points_velo, dtype=np.float64)
    center = _camera2_center_in_velo(geometry)
    if geometry.t_imu_velo is None:
        return pts - center.reshape((1,) * (pts.ndim - 1) + (3,))
    try:
        velo_to_imu = np.linalg.inv(np.asarray(geometry.t_imu_velo, dtype=np.float64))
    except np.linalg.LinAlgError as error:
        raise ValueError("IMU-to-Velodyne calibration must be invertible") from error
    flat = pts.reshape(-1, 3)
    center_flat = center.reshape(1, 3)
    both = np.concatenate([flat, center_flat], axis=0)
    both_h = np.concatenate([both, np.ones((len(both), 1), dtype=np.float64)], axis=1).T
    imu = (velo_to_imu @ both_h).T[:, :3]
    point_imu = imu[:-1].reshape(pts.shape)
    center_imu = imu[-1]
    return point_imu - center_imu.reshape((1,) * (pts.ndim - 1) + (3,))


def _satellite_grid_from_velo_points(
    points_velo: np.ndarray,
    geometry: RawKittiGeometry,
    sat_size: int = 256,
    meter_per_pixel: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map Velodyne points to camera2-centered aligned satellite crop coords.

    KITTI_raw_sat_lidar shifts the rotated satellite crop by camera2's
    IMU-frame forward/right offset.  Therefore candidate keys must be expressed
    relative to the camera2 center, not the Velodyne origin.  With the crop
    aligned to the vehicle frame, local forward maps to PIL x and local right
    maps to PIL y.
    """
    if not isinstance(sat_size, (int, np.integer)) or isinstance(sat_size, bool) or int(sat_size) <= 0:
        raise ValueError("sat_size must be a positive integer")
    base_mpp = float(kitti_utils.get_meter_per_pixel()) if meter_per_pixel is None else float(meter_per_pixel)
    mpp = base_mpp * (float(kitti_utils.SatMap_end_sidelength) / float(sat_size))
    if not np.isfinite(mpp) or mpp <= 0:
        raise ValueError("meter_per_pixel must be positive and finite")
    local = _velo_to_imu_local(points_velo, geometry)
    forward_m = local[..., 0]
    right_m = -local[..., 1]
    cx = (float(sat_size) - 1.0) / 2.0
    cy = (float(sat_size) - 1.0) / 2.0
    px = cx + forward_m / mpp
    py = cy + right_m / mpp
    valid = np.isfinite(px) & np.isfinite(py) & (px >= 0.0) & (px <= sat_size - 1.0) & (py >= 0.0) & (py <= sat_size - 1.0)
    norm = np.zeros((*local.shape[:-1], 2), dtype=np.float32)
    norm[..., 0] = (2.0 * (np.clip(px, 0.0, sat_size - 1.0) + 0.5) / float(sat_size)) - 1.0
    norm[..., 1] = (2.0 * (np.clip(py, 0.0, sat_size - 1.0) + 0.5) / float(sat_size)) - 1.0
    norm[~valid] = 0.0
    return norm.astype(np.float32), valid.astype(bool)


def _current_measurements_by_cell(
    cur_points_velo: np.ndarray,
    geometry: RawKittiGeometry,
    grid: Tuple[int, int],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest current LiDAR point and rectified depth for each feature cell."""
    points = np.asarray(cur_points_velo, dtype=np.float32)
    h, w = grid
    point_by_cell = np.zeros((h, w, 3), dtype=np.float32)
    depth_by_cell = np.full((h, w), np.inf, dtype=np.float32)
    covered = np.zeros((h, w), dtype=bool)
    if points.size == 0:
        return point_by_cell, depth_by_cell, covered
    uv, depth, in_image = geometry.project_velo_to_image(points, image_size=image_size)
    _nearest, idx = _nearest_depth_by_cell(uv, depth, in_image, grid, image_size)
    covered = idx >= 0
    if np.any(covered):
        point_by_cell[covered] = points[idx[covered], :3]
        depth_by_cell[covered] = depth[idx[covered]]
    return point_by_cell, depth_by_cell, covered


def _write_measured_slot(
    y: int,
    x: int,
    source_code: int,
    kind_grid: np.ndarray,
    out_grid: np.ndarray,
    sat_grid: np.ndarray,
    sat_valid: np.ndarray,
    out_valid: np.ndarray,
    source: np.ndarray,
    measurement_flag: np.ndarray,
    positions_depth: np.ndarray,
    positions_height: np.ndarray,
    depths: np.ndarray,
    point_by_cell: np.ndarray,
    depth_by_cell: np.ndarray,
    sat_size: int,
    sat_meter_per_pixel: Optional[float],
    geometry: RawKittiGeometry,
) -> None:
    depth = float(depth_by_cell[y, x])
    if not np.isfinite(depth) or depth <= 0:
        return
    point = point_by_cell[y, x]
    if not np.isfinite(point).all():
        return
    idx = int(np.argmin(np.abs(depths - depth)))
    out_grid[y, x, idx] = kind_grid[y, x]
    exact_sat, exact_sat_valid = _satellite_grid_from_velo_points(
        point.reshape(1, 1, 1, 3), geometry=geometry, sat_size=sat_size, meter_per_pixel=sat_meter_per_pixel)
    sat_grid[y, x, idx] = exact_sat[0, 0, 0]
    sat_valid[y, x, idx] = bool(exact_sat_valid[0, 0, 0])
    out_valid[y, x, idx] = True
    source[y, x, idx] = source_code
    measurement_flag[y, x, idx] = 1.0
    positions_depth[y, x, idx] = depth
    positions_height[y, x, idx] = float(point[2]) / 40.0


def _fill_measured_static_and_object(
    out_grid: np.ndarray,
    sat_grid: np.ndarray,
    sat_valid: np.ndarray,
    out_valid: np.ndarray,
    source: np.ndarray,
    measurement_flag: np.ndarray,
    positions_depth: np.ndarray,
    positions_height: np.ndarray,
    depths: np.ndarray,
    base: Dict[str, np.ndarray],
    object_valid: np.ndarray,
    object_grid: np.ndarray,
    point_by_cell: np.ndarray,
    depth_by_cell: np.ndarray,
    sat_size: int,
    sat_meter_per_pixel: Optional[float],
    geometry: RawKittiGeometry,
) -> None:
    lidar_valid = np.asarray(base["lidar_valid"], dtype=bool)
    current_covered = np.asarray(base["current_covered"], dtype=bool)

    for kind_valid, kind_grid, kind_source in (
        (lidar_valid, np.asarray(base["history_grid"], dtype=np.float32), SOURCE_STATIC_MEASUREMENT),
        (object_valid, object_grid.astype(np.float32), SOURCE_OBJECT_COMP),
    ):
        yy, xx = np.nonzero(kind_valid)
        for y, x in zip(yy.tolist(), xx.tolist()):
            _write_measured_slot(
                y, x, kind_source, kind_grid, out_grid, sat_grid, sat_valid,
                out_valid, source, measurement_flag, positions_depth, positions_height, depths,
                point_by_cell, depth_by_cell, sat_size, sat_meter_per_pixel, geometry)

    # Local geometry must not reopen ego-rejected measured cells.  The global
    # persistent content reader handles these fallback cells separately.
    fallback = current_covered & ~(lidar_valid | object_valid)
    source[fallback, :] = SOURCE_CONTENT_FALLBACK
    out_valid[fallback, :] = False
    measurement_flag[fallback, :] = 1.0
    depth = depth_by_cell[fallback]
    if depth.size:
        positions_depth[fallback, :] = depth[:, None]
        positions_height[fallback, :] = point_by_cell[fallback, 2][:, None] / 40.0

def _metrics(valid: np.ndarray, sat_valid: np.ndarray, source: np.ndarray, current_covered: np.ndarray) -> Dict[str, float]:
    total = float(valid.size)
    out = {
        "grid_h": int(valid.shape[0]),
        "grid_w": int(valid.shape[1]),
        "num_depth_candidates": int(valid.shape[2]),
        "valid_candidate_count": int(valid.sum()),
        "sat_valid_candidate_count": int(sat_valid.sum()),
        "coverage_candidates": float(valid.sum()) / max(total, 1.0),
        "sat_coverage_candidates": float(sat_valid.sum()) / max(total, 1.0),
        "current_lidar_cell_count": int(current_covered.sum()),
    }
    for code, name in [
        (SOURCE_UNKNOWN_RAY, "unknown_ray"),
        (SOURCE_STATIC_MEASUREMENT, "static_measurement"),
        (SOURCE_OBJECT_COMP, "object_comp"),
        (SOURCE_CONTENT_FALLBACK, "content_fallback"),
    ]:
        mask = source == code
        if code != SOURCE_CONTENT_FALLBACK:
            mask = mask & valid
        count = int(mask.sum())
        out[f"{name}_candidate_count"] = count
        out[f"coverage_{name}"] = count / max(total, 1.0)
    return out


def build_ray_geometry_from_arrays(
    prev_points_velo: np.ndarray,
    cur_points_velo: np.ndarray,
    cur_to_prev_velo: np.ndarray,
    geometry: RawKittiGeometry,
    grid: Tuple[int, int] = (16, 64),
    num_depth_candidates: int = 16,
    depth_min_m: float = 2.0,
    depth_max_m: float = 120.0,
    image_size: Optional[Tuple[int, int]] = None,
    depth_tol_m: float = 0.75,
    sat_size: int = 256,
    sat_meter_per_pixel: Optional[float] = None,
) -> Dict[str, np.ndarray | Dict[str, float]]:
    """Pure array builder for persistent-history ray candidates."""
    h, w = _require_grid(grid)
    pose = _require_pose(cur_to_prev_velo)
    image_size = image_size or geometry.image_size
    depths = _inverse_depth_candidates(num_depth_candidates, depth_min_m, depth_max_m)

    rays = _camera_rays_for_feature_grid(geometry, (h, w), image_size)
    current_points = _camera_depths_to_velo_points(geometry, rays, depths)
    prev_points_for_ray = transform_points(current_points.reshape(-1, 3), pose).reshape(h, w, len(depths), 3)
    history_grid, history_px, ray_valid, prev_depth = _project_feature_candidates(
        prev_points_for_ray, geometry, (h, w), image_size)
    prev_zbuffer = _previous_depth_zbuffer(prev_points_velo, geometry, (h, w), image_size)
    occluded = _occlusion_mask_from_previous_depth(
        history_px, prev_depth, prev_zbuffer, ray_valid, depth_tol_m)
    ray_valid = ray_valid & ~occluded
    history_grid[~ray_valid] = 0.0
    history_px[~ray_valid] = 0.0

    source = np.zeros((h, w, len(depths)), dtype=np.uint8)
    source[ray_valid] = SOURCE_UNKNOWN_RAY
    valid = ray_valid.copy()
    positions_depth = np.broadcast_to(depths.reshape(1, 1, -1), (h, w, len(depths))).astype(np.float32).copy()
    positions_height = (current_points[..., 2] / 40.0).astype(np.float32).copy()
    measurement_flag = np.zeros((h, w, len(depths)), dtype=np.float32)

    # Direct LiDAR and object-compensated measured correspondences reuse the old
    # vetted static/object paths.  Rotation fallback from the old builder is not
    # used as strong evidence; unknown cells already receive explicit ray depths.
    from tools.temporal_history_geometry import build_lidar_history_geometry_from_arrays

    base = build_lidar_history_geometry_from_arrays(
        prev_points_velo, cur_points_velo, pose, geometry,
        grid=(h, w), image_size=image_size, depth_tol_m=depth_tol_m)
    recovered = recover_moving_object_history(
        prev_points_velo, cur_points_velo, pose, geometry,
        np.asarray(base["history_valid"], dtype=bool),
        np.asarray(base["current_covered"], dtype=bool),
        (h, w), image_size=image_size, depth_tol_m=depth_tol_m)
    base = dict(base)
    base["lidar_valid"] = np.asarray(base["history_valid"], dtype=bool)
    object_valid = np.asarray(recovered["object_valid"], dtype=bool) & ~base["lidar_valid"]
    object_grid = np.asarray(recovered["object_grid"], dtype=np.float32)
    point_by_cell, depth_by_cell, measured_covered = _current_measurements_by_cell(
        cur_points_velo, geometry, (h, w), image_size)
    object_valid = object_valid & measured_covered
    # Keep the old path's current_covered authoritative, but require actual
    # point/depth lookup before writing measured candidates.
    base["current_covered"] = np.asarray(base["current_covered"], dtype=bool) & measured_covered

    sat_grid, sat_valid = _satellite_grid_from_velo_points(
        current_points, geometry=geometry, sat_size=sat_size, meter_per_pixel=sat_meter_per_pixel)
    # Satellite coverage is an independent annotation; it must not invalidate
    # history candidates.
    sat_valid = sat_valid & np.isfinite(sat_grid).all(axis=-1)
    _fill_measured_static_and_object(
        history_grid, sat_grid, sat_valid, valid, source, measurement_flag,
        positions_depth, positions_height, depths, base, object_valid, object_grid,
        point_by_cell, depth_by_cell, sat_size, sat_meter_per_pixel, geometry)

    log_depth = np.log(np.maximum(positions_depth, 1e-6))
    log_min, log_max = np.log(float(depth_min_m)), np.log(float(depth_max_m))
    log_norm = ((log_depth - log_min) / max(log_max - log_min, 1e-9)).astype(np.float32)
    positions = np.stack([
        np.broadcast_to(log_norm, (h, w, len(depths))),
        positions_height,
        source.astype(np.float32) / 4.0,
        measurement_flag,
    ], axis=-1).astype(np.float32)
    # Keep source/4 visible even when local geometry is invalid, e.g. source4
    # content fallback cells routed to the global persistent reader.

    finite_arrays = [history_grid, sat_grid, positions]
    if not all(np.isfinite(a).all() for a in finite_arrays):
        raise ValueError("ray geometry contains non-finite values")

    metrics = _metrics(valid, sat_valid, source, np.asarray(base["current_covered"], dtype=bool))
    metrics.update({
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "source_invalid": SOURCE_INVALID,
        "source_unknown_ray": SOURCE_UNKNOWN_RAY,
        "source_static_measurement": SOURCE_STATIC_MEASUREMENT,
        "source_object_comp": SOURCE_OBJECT_COMP,
        "source_content_fallback": SOURCE_CONTENT_FALLBACK,
        "object_cell_count": int(object_valid.sum()),
        "occluded_unknown_ray_candidate_count": int(occluded.sum()),
        "history_mode": "persistent_ray_candidates_static_object_content_fallback",
    })
    if isinstance(recovered.get("diagnostics"), dict):
        for key, value in recovered["diagnostics"].items():
            if isinstance(value, (int, float, np.integer, np.floating, bool)):
                metrics[f"object_{key}"] = float(value) if isinstance(value, (float, np.floating)) else int(value)

    return {
        "history_grid": history_grid.astype(np.float32),
        "sat_grid": sat_grid.astype(np.float32),
        "valid": valid.astype(bool),
        "sat_valid": sat_valid.astype(bool),
        "positions": positions.astype(np.float32),
        "metrics": metrics,
        # Extra diagnostics are intentionally namespaced and may be ignored by
        # callers that only consume the contracted fields above.
        "history_grid_px": history_px.astype(np.float32),
        "source": source.astype(np.uint8),
        "depth_candidates_m": depths.astype(np.float32),
        "current_lidar_covered": np.asarray(base["current_covered"], dtype=bool),
    }


def build_pair_ray_geometry(
    prev_row: dict,
    cur_row: dict,
    kitti_root: str | Path | None = None,
    grid: Tuple[int, int] = (16, 64),
    num_depth_candidates: int = 16,
    depth_min_m: float = 2.0,
    depth_max_m: float = 120.0,
) -> Dict[str, np.ndarray | Dict[str, float]]:
    """Build persistent ray geometry from two consecutive KITTI manifest rows."""
    if not consecutive_rows(prev_row, cur_row):
        raise ValueError("history requires consecutive frames from the same date/drive")
    row_prev = dict(prev_row)
    row_cur = dict(cur_row)
    for row in (row_prev, row_cur):
        for key in (
            "velodyne_path", "oxts_path", "calib_dir", "calib_cam_to_cam_path",
            "calib_velo_to_cam_path", "image_02_path",
        ):
            if key in row:
                row[key] = rebase_kitti_path(row[key], kitti_root)
    geometry = get_geometry(row_cur["calib_dir"])
    prev_to_cur = geometry.relative_velo_pose(row_prev["oxts_path"], row_cur["oxts_path"])
    cur_to_prev = np.linalg.inv(prev_to_cur)
    prev_points = load_velodyne(row_prev["velodyne_path"])
    cur_points = load_velodyne(row_cur["velodyne_path"])
    return build_ray_geometry_from_arrays(
        prev_points, cur_points, cur_to_prev, geometry,
        grid=grid, num_depth_candidates=num_depth_candidates,
        depth_min_m=depth_min_m, depth_max_m=depth_max_m)


__all__ = [
    "SOURCE_INVALID",
    "SOURCE_UNKNOWN_RAY",
    "SOURCE_STATIC_MEASUREMENT",
    "SOURCE_OBJECT_COMP",
    "SOURCE_CONTENT_FALLBACK",
    "build_pair_ray_geometry",
    "build_ray_geometry_from_arrays",
]
