"""Dense-depth previous RGB reprojection diagnostics.

This module is intentionally independent from training.  It forward-projects
the previous RGB using a dense source depth map and verifies the projected
surface against sparse source/target LiDAR when available.  It never consumes
target RGB and does not fill holes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from dataloader.kitti_raw_lidar_utils import zbuffer_visible_point_indices
from tools.temporal_history_geometry import RawKittiGeometry, transform_points


def _project_velo_to_image_resized(
    points_velo: np.ndarray,
    geometry: RawKittiGeometry,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = image_size
    points = np.asarray(points_velo, dtype=np.float64)
    if points.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    pts_h = np.concatenate([points[:, :3], np.ones((len(points), 1), dtype=np.float64)], axis=1).T
    rect_h = geometry.r_rect_00_ext @ geometry.t_cam_velo @ pts_h
    pix = geometry.p_rect_02 @ rect_h
    depth = rect_h[2]
    safe = np.where(np.abs(pix[2]) < 1e-9, 1e-9, pix[2])
    raw_uv = (pix[:2] / safe).T
    src_w, src_h = geometry.image_size
    uv = np.empty_like(raw_uv, dtype=np.float64)
    uv[:, 0] = (raw_uv[:, 0] + 0.5) * (float(width) / float(src_w)) - 0.5
    uv[:, 1] = (raw_uv[:, 1] + 0.5) * (float(height) / float(src_h)) - 0.5
    eps = 1e-5
    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= -eps)
        & (uv[:, 0] <= width - 1.0 + eps)
        & (uv[:, 1] >= -eps)
        & (uv[:, 1] <= height - 1.0 + eps)
    )
    uv[:, 0] = np.clip(uv[:, 0], 0.0, width - 1.0)
    uv[:, 1] = np.clip(uv[:, 1], 0.0, height - 1.0)
    return uv.astype(np.float32), depth.astype(np.float32), valid.astype(bool)


def _require_hw(name: str, arr: np.ndarray, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"{name} must have shape [H,W]")
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} shape {out.shape} != expected {shape}")
    return out


def _require_rgb(prev_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(prev_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("prev_rgb must have shape [H,W,3]")
    if not np.isfinite(rgb).all():
        raise ValueError("prev_rgb contains non-finite values")
    return np.clip(rgb, 0.0, 1.0)


def _require_points(name: str, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"{name} must have shape [N,3+]")
    if not np.isfinite(pts[:, :3]).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return pts[:, :3]


def _require_pose(prev_to_cur_velo: np.ndarray) -> np.ndarray:
    pose = np.asarray(prev_to_cur_velo, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("prev_to_cur_velo must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("prev_to_cur_velo must be homogeneous")
    return pose


def _sparse_depth_map(
    points_velo: np.ndarray,
    geometry: RawKittiGeometry,
    image_size: Tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    if len(points_velo) == 0:
        return depth_map
    uv, depth, valid = _project_velo_to_image_resized(points_velo, geometry, image_size)
    visible = zbuffer_visible_point_indices(uv, depth, valid, (height, width))
    for idx in visible:
        x = int(np.clip(round(float(uv[idx, 0])), 0, width - 1))
        y = int(np.clip(round(float(uv[idx, 1])), 0, height - 1))
        depth_map[y, x] = float(depth[idx])
    return depth_map


def _dense_pixels_to_velo(
    prev_depth: np.ndarray,
    geometry: RawKittiGeometry,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project dense resized-image pixels to Velodyne coordinates.

    Uses the full `P_rect_02`, including its translation column.  The resized
    pixel centers are mapped back to calibrated-image coordinates using the
    inverse PIL half-pixel convention.
    """
    width, height = image_size
    yy, xx = np.indices((height, width), dtype=np.float64)
    valid = np.isfinite(prev_depth) & (prev_depth > 0.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    src_w, src_h = geometry.image_size
    raw_u = ((xx[valid] + 0.5) * float(src_w) / float(width)) - 0.5
    raw_v = ((yy[valid] + 0.5) * float(src_h) / float(height)) - 0.5
    p = np.asarray(geometry.p_rect_02, dtype=np.float64)
    k = p[:, :3]
    try:
        inv_k = np.linalg.inv(k)
    except np.linalg.LinAlgError as error:
        raise ValueError("camera projection matrix must be invertible") from error
    b = inv_k @ p[:, 3]
    rays = (inv_k @ np.stack([raw_u, raw_v, np.ones_like(raw_u)], axis=0)).T
    rz = rays[:, 2]
    if np.any(np.abs(rz) <= 1e-9):
        raise ValueError("camera rays must have nonzero z")
    depth = prev_depth[valid].astype(np.float64)
    scale = (depth + float(b[2])) / rz
    points_rect = rays * scale[:, None] - b[None, :]
    points_h = np.concatenate([points_rect, np.ones((len(points_rect), 1), dtype=np.float64)], axis=1).T
    try:
        velo_h = np.linalg.inv(geometry.r_rect_00_ext @ geometry.t_cam_velo) @ points_h
    except np.linalg.LinAlgError as error:
        raise ValueError("camera-to-velo transform must be invertible") from error
    source_uv = np.stack([xx[valid], yy[valid]], axis=1).astype(np.float32)
    return velo_h.T[:, :3].astype(np.float32), source_uv


def build_dense_reference(
    prev_rgb: np.ndarray,
    prev_depth: np.ndarray,
    prev_points: np.ndarray,
    cur_points: np.ndarray,
    geometry: RawKittiGeometry,
    prev_to_cur_velo: np.ndarray,
    depth_tol_m: float = 0.75,
) -> Dict[str, object]:
    """Forward-project previous RGB using dense previous rect-camera depth."""
    rgb = _require_rgb(prev_rgb)
    height, width = rgb.shape[:2]
    image_size = (width, height)
    depth = _require_hw("prev_depth", prev_depth, (height, width))
    prev_pts = _require_points("prev_points", prev_points)
    cur_pts = _require_points("cur_points", cur_points)
    pose = _require_pose(prev_to_cur_velo)
    if not np.isfinite(depth_tol_m) or depth_tol_m < 0.0:
        raise ValueError("depth_tol_m must be finite and nonnegative")

    warped = np.zeros((height, width, 3), dtype=np.float32)
    support = np.zeros((height, width), dtype=bool)
    measured = np.zeros((height, width), dtype=bool)
    estimated = np.zeros((height, width), dtype=bool)
    target_conflict = np.zeros((height, width), dtype=bool)
    projected_depth = np.zeros((height, width), dtype=np.float32)
    source_uv_image = np.zeros((height, width, 2), dtype=np.float32)

    source_sparse_depth = _sparse_depth_map(prev_pts, geometry, image_size)
    target_sparse_depth = _sparse_depth_map(cur_pts, geometry, image_size)
    dense_points, source_uv = _dense_pixels_to_velo(depth, geometry, image_size)
    if len(dense_points) == 0:
        diagnostics = {
            "dense_source_count": 0,
            "support_count": 0,
            "measured_count": 0,
            "estimated_count": 0,
            "target_conflict_count": 0,
            "source_measured_candidate_count": 0,
            "target_verified_candidate_count": 0,
            "target_missing_candidate_count": 0,
        }
        return {
            "warped_rgb": warped,
            "support_mask": support,
            "measured_mask": measured,
            "estimated_mask": estimated,
            "target_conflict_mask": target_conflict,
            "projected_depth": projected_depth,
            "source_uv": source_uv_image,
            "diagnostics": diagnostics,
        }

    dense_cur = transform_points(dense_points, pose)
    uv_cur, z_cur, in_cur = _project_velo_to_image_resized(dense_cur, geometry, image_size)
    visible = zbuffer_visible_point_indices(uv_cur, z_cur, in_cur, (height, width))

    source_x = np.rint(source_uv[:, 0]).astype(np.int64)
    source_y = np.rint(source_uv[:, 1]).astype(np.int64)
    source_depth_at_pixel = depth[source_y, source_x]
    sparse_source_at_pixel = source_sparse_depth[source_y, source_x]
    source_measured = np.isfinite(sparse_source_at_pixel) & (
        np.abs(source_depth_at_pixel - sparse_source_at_pixel) <= float(depth_tol_m)
    )

    conflict_count = 0
    target_verified_count = 0
    target_missing_count = 0
    for idx in visible:
        x = int(np.clip(round(float(uv_cur[idx, 0])), 0, width - 1))
        y = int(np.clip(round(float(uv_cur[idx, 1])), 0, height - 1))
        z_proj = float(z_cur[idx])
        z_target = float(target_sparse_depth[y, x])
        has_target = np.isfinite(z_target)
        if has_target and abs(z_proj - z_target) > float(depth_tol_m):
            target_conflict[y, x] = True
            conflict_count += 1
            continue
        warped[y, x] = rgb[source_y[idx], source_x[idx]]
        support[y, x] = True
        projected_depth[y, x] = z_proj
        source_uv_image[y, x] = source_uv[idx]
        if has_target and bool(source_measured[idx]):
            measured[y, x] = True
            target_verified_count += 1
        else:
            estimated[y, x] = True
            if has_target:
                target_verified_count += 1
            else:
                target_missing_count += 1

    diagnostics = {
        "dense_source_count": int(len(dense_points)),
        "visible_projected_count": int(len(visible)),
        "support_count": int(support.sum()),
        "measured_count": int(measured.sum()),
        "estimated_count": int(estimated.sum()),
        "target_conflict_count": int(target_conflict.sum()),
        "target_conflict_candidate_count": int(conflict_count),
        "source_measured_candidate_count": int(source_measured.sum()),
        "target_verified_candidate_count": int(target_verified_count),
        "target_missing_candidate_count": int(target_missing_count),
        "depth_tol_m": float(depth_tol_m),
    }
    return {
        "warped_rgb": warped,
        "support_mask": support,
        "measured_mask": measured,
        "estimated_mask": estimated,
        "target_conflict_mask": target_conflict,
        "projected_depth": projected_depth,
        "source_uv": source_uv_image,
        "diagnostics": diagnostics,
    }


def calibrate_depth(
    pred_depth: np.ndarray,
    source_sparse_depth: np.ndarray,
    fit_mask: Optional[np.ndarray] = None,
    min_depth_m: float = 1.0,
    max_depth_m: float = 80.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Fit a robust global multiplicative scale from source LiDAR only."""
    pred = _require_hw("pred_depth", pred_depth)
    sparse = _require_hw("source_sparse_depth", source_sparse_depth, pred.shape)
    if fit_mask is None:
        mask = np.ones(pred.shape, dtype=bool)
    else:
        mask = np.asarray(fit_mask, dtype=bool)
        if mask.shape != pred.shape:
            raise ValueError("fit_mask shape must match pred_depth")
    valid = (
        mask
        & np.isfinite(pred)
        & np.isfinite(sparse)
        & (pred >= float(min_depth_m))
        & (pred <= float(max_depth_m))
        & (sparse >= float(min_depth_m))
        & (sparse <= float(max_depth_m))
    )
    if not np.any(valid):
        diagnostics = {
            "fit_count": 0,
            "scale": 1.0,
            "reason": "no valid source sparse depth samples",
        }
        return pred.astype(np.float32, copy=True), diagnostics
    ratios = sparse[valid] / np.maximum(pred[valid], 1e-6)
    scale = float(np.median(ratios))
    calibrated = (pred * scale).astype(np.float32)
    before = np.abs(pred[valid] - sparse[valid])
    after = np.abs(calibrated[valid] - sparse[valid])
    diagnostics = {
        "fit_count": int(valid.sum()),
        "scale": scale,
        "ratio_median": scale,
        "ratio_mad": float(np.median(np.abs(ratios - scale))),
        "mean_abs_error_before_m": float(before.mean()),
        "mean_abs_error_after_m": float(after.mean()),
        "median_abs_error_before_m": float(np.median(before)),
        "median_abs_error_after_m": float(np.median(after)),
        "min_depth_m": float(min_depth_m),
        "max_depth_m": float(max_depth_m),
    }
    return calibrated, diagnostics
