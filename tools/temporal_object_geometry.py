"""Recover moving-object history cells rejected by ego-only LiDAR correspondence.

Parked objects stay on the ego `lidar_valid` path. A cluster is treated as
moving only after ego compensation still leaves a large centroid shift.
Failed association never falls back to rotation on measured cells.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.temporal_history_geometry import (
    RawKittiGeometry,
    _nearest_depth_by_cell,
    align_corners_false_normalize,
    fit_ground_plane_velo,
    ground_proxy_mask,
    image_to_feature_xy,
    transform_points,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def cluster_nonground_points(
    points: np.ndarray,
    voxel_m: float = 0.6,
    min_points: int = 8,
    max_extent_m: float = 12.0,
    max_height_m: float = 4.0,
) -> List[Dict[str, np.ndarray]]:
    """Voxel connected components on already non-ground points."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape [N,3+]")
    if not np.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError("voxel_m must be positive and finite")
    if not isinstance(min_points, int) or isinstance(min_points, bool) or min_points < 1:
        raise ValueError("min_points must be a positive int")
    if len(pts) < min_points:
        return []
    keys = np.floor(pts[:, :3] / float(voxel_m)).astype(np.int32)
    voxels: Dict[Tuple[int, int, int], List[int]] = {}
    for i, key in enumerate(keys):
        voxels.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(i)
    occupied = set(voxels)
    seen = set()
    clusters: List[Dict[str, np.ndarray]] = []
    for origin in occupied:
        if origin in seen:
            continue
        stack = [origin]
        seen.add(origin)
        members: List[int] = []
        while stack:
            x, y, z = stack.pop()
            members.extend(voxels[(x, y, z)])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nxt = (x + dx, y + dy, z + dz)
                        if nxt in occupied and nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
        if len(members) < min_points:
            continue
        idx = np.asarray(members, dtype=np.int64)
        cloud = pts[idx, :3]
        extent = cloud.max(axis=0) - cloud.min(axis=0)
        if float(extent[0]) > max_extent_m or float(extent[1]) > max_extent_m or float(extent[2]) > max_height_m:
            continue
        clusters.append({
            "indices": idx,
            "points": cloud,
            "centroid": cloud.mean(axis=0).astype(np.float32),
        })
    return clusters


def associate_clusters(
    current: Sequence[Dict[str, np.ndarray]],
    previous: Sequence[Dict[str, np.ndarray]],
    cur_to_prev: np.ndarray,
    max_dist_m: float = 4.0,
) -> List[Dict[str, object]]:
    """Mutual nearest-centroid matches after ego compensation."""
    pose = np.asarray(cur_to_prev, dtype=np.float64)
    _require(pose.shape == (4, 4) and np.isfinite(pose).all(), "cur_to_prev must be a finite 4x4")
    if not np.isfinite(max_dist_m) or max_dist_m <= 0:
        raise ValueError("max_dist_m must be positive and finite")
    if not current or not previous:
        return []
    cur_cent = np.stack([c["centroid"] for c in current], axis=0)
    prev_cent = np.stack([p["centroid"] for p in previous], axis=0)
    cur_in_prev = transform_points(cur_cent, pose)
    delta = cur_in_prev[:, None, :] - prev_cent[None, :, :]
    dist = np.linalg.norm(delta, axis=2)
    cur_best = dist.argmin(axis=1)
    prev_best = dist.argmin(axis=0)
    matches = []
    used_prev = set()
    for i, j in enumerate(cur_best):
        j = int(j)
        if int(prev_best[j]) != i or j in used_prev:
            continue
        d = float(dist[i, j])
        if d > max_dist_m:
            continue
        used_prev.add(j)
        matches.append({
            "current_index": i,
            "previous_index": j,
            "ego_centroid_shift_m": d,
            "current": current[i],
            "previous": previous[j],
        })
    return matches


def _translation_matrix(delta: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.asarray(delta, dtype=np.float64).reshape(3)
    return out


def _kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    centered_s = src - mu_s
    centered_d = dst - mu_d
    cov = centered_s.T @ centered_d
    u, _s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        r = vt.T @ u.T
    t = mu_d - r @ mu_s
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def estimate_object_transform(
    current_points: np.ndarray,
    previous_points: np.ndarray,
    cur_to_prev: np.ndarray,
    min_pairs: int = 8,
    nn_radius_m: float = 1.5,
    max_rmse_m: float = 0.75,
    min_moving_m: float = 0.35,
    max_moving_m: float = 5.0,
) -> Dict[str, object]:
    """Rigid map current object points (current velo) to previous velo.

    Residual motion is measured after ego compensation. Below min_moving_m the
    object is treated as parked and should keep the ego LiDAR path.
    """
    pose = np.asarray(cur_to_prev, dtype=np.float64)
    _require(pose.shape == (4, 4) and np.isfinite(pose).all(), "cur_to_prev must be a finite 4x4")
    cur = np.asarray(current_points, dtype=np.float32)
    prev = np.asarray(previous_points, dtype=np.float32)
    empty = {
        "accepted": False,
        "moving": False,
        "transform": pose.astype(np.float64),
        "ego_centroid_shift_m": 0.0,
        "rmse_m": float("inf"),
        "pair_count": 0,
        "reason": "empty",
    }
    if len(cur) < min_pairs or len(prev) < min_pairs:
        empty["reason"] = "too_few_points"
        return empty
    cur_in_prev = transform_points(cur, pose)
    shift = prev.mean(axis=0) - cur_in_prev.mean(axis=0)
    shift_n = float(np.linalg.norm(shift))
    t0 = _translation_matrix(shift)
    aligned = transform_points(cur_in_prev, t0)
    try:
        from scipy.spatial import cKDTree
        dist, nn = cKDTree(prev).query(aligned, k=1)
    except Exception:
        delta = aligned[:, None, :] - prev[None, :, :]
        d2 = np.sum(delta * delta, axis=2)
        nn = d2.argmin(axis=1)
        dist = np.sqrt(d2[np.arange(len(aligned)), nn])
    ok = np.isfinite(dist) & (dist <= nn_radius_m)
    pair_count = int(ok.sum())
    if pair_count < min_pairs:
        return {
            "accepted": False,
            "moving": shift_n >= min_moving_m,
            "transform": pose,
            "ego_centroid_shift_m": shift_n,
            "rmse_m": float("inf"),
            "pair_count": pair_count,
            "reason": "too_few_pairs",
        }
    t_delta = _kabsch(aligned[ok], prev[nn[ok]])
    t_full = t_delta @ t0 @ pose
    mapped = transform_points(cur, t_full)
    try:
        from scipy.spatial import cKDTree
        rmse_dist, _ = cKDTree(prev).query(mapped, k=1)
    except Exception:
        delta = mapped[:, None, :] - prev[None, :, :]
        rmse_dist = np.sqrt(np.sum(delta * delta, axis=2).min(axis=1))
    finite = np.isfinite(rmse_dist)
    rmse = float(np.sqrt(np.mean(np.square(rmse_dist[finite])))) if np.any(finite) else float("inf")
    moving = (shift_n >= min_moving_m) and (shift_n <= max_moving_m)
    accepted = moving and rmse <= max_rmse_m and pair_count >= min_pairs
    reason = "ok"
    if not moving:
        reason = "stationary" if shift_n < min_moving_m else "implausible_motion"
    elif rmse > max_rmse_m:
        reason = "rmse"
    return {
        "accepted": bool(accepted),
        "moving": bool(moving and shift_n >= min_moving_m),
        "transform": t_full.astype(np.float64),
        "ego_centroid_shift_m": shift_n,
        "rmse_m": rmse,
        "pair_count": pair_count,
        "reason": reason,
    }


def _fill_cells_with_transform(
    cur_points: np.ndarray,
    transform: np.ndarray,
    geometry: RawKittiGeometry,
    grid: Tuple[int, int],
    image_size: Tuple[int, int],
    depth_tol_m: float,
    prev_z: np.ndarray,
    cur_uv: np.ndarray,
    cur_idx: np.ndarray,
    eligible_cells: np.ndarray,
    eligible_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid
    history_grid = np.zeros((h, w, 2), dtype=np.float32)
    history_grid_px = np.zeros((h, w, 2), dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)
    if not np.any(eligible_cells):
        return history_grid, history_grid_px, valid
    mapped_uv, mapped_depth, mapped_in = geometry.project_velo_to_image(
        transform_points(cur_points, transform), image_size=image_size)
    img_w, img_h = image_size
    prev_cell_x = np.floor(mapped_uv[:, 0] * (float(w) / float(img_w))).astype(np.int64)
    prev_cell_y = np.floor(mapped_uv[:, 1] * (float(h) / float(img_h))).astype(np.int64)
    support_ok = np.zeros((len(cur_points),), dtype=bool)
    in_prev_cell = (
        mapped_in
        & (prev_cell_x >= 0) & (prev_cell_x < w)
        & (prev_cell_y >= 0) & (prev_cell_y < h)
        & eligible_points
    )
    if np.any(in_prev_cell):
        nearest_prev = np.full((len(cur_points),), np.inf, dtype=np.float32)
        nearest_prev[in_prev_cell] = prev_z[prev_cell_y[in_prev_cell], prev_cell_x[in_prev_cell]]
        support_ok = in_prev_cell & np.isfinite(nearest_prev) & (np.abs(nearest_prev - mapped_depth) <= depth_tol_m)
    yy, xx = np.nonzero(eligible_cells)
    if len(yy) == 0:
        return history_grid, history_grid_px, valid
    chosen = cur_idx[yy, xx]
    chosen_ok = np.zeros((len(yy),), dtype=bool)
    in_cloud = (chosen >= 0) & (chosen < len(support_ok))
    chosen_ok[in_cloud] = support_ok[chosen[in_cloud]]
    if not np.any(chosen_ok):
        return history_grid, history_grid_px, valid
    vy = yy[chosen_ok]
    vx = xx[chosen_ok]
    src = chosen[chosen_ok]
    prev_feature_px = image_to_feature_xy(mapped_uv[src], image_size, grid)
    cur_feature_px = image_to_feature_xy(cur_uv[src], image_size, grid)
    query_center_px = np.stack([vx.astype(np.float32), vy.astype(np.float32)], axis=1)
    history_feature_px = query_center_px + (prev_feature_px - cur_feature_px)
    in_feature = (
        (history_feature_px[:, 0] >= -0.5) & (history_feature_px[:, 0] <= w - 0.5)
        & (history_feature_px[:, 1] >= -0.5) & (history_feature_px[:, 1] <= h - 0.5)
    )
    if not np.any(in_feature):
        return history_grid, history_grid_px, valid
    vy, vx = vy[in_feature], vx[in_feature]
    history_feature_px = history_feature_px[in_feature]
    history_grid_px[vy, vx] = history_feature_px
    history_grid[vy, vx] = align_corners_false_normalize(history_feature_px, grid)
    valid[vy, vx] = True
    return history_grid, history_grid_px, valid


def recover_moving_object_history(
    prev_points_velo: np.ndarray,
    cur_points_velo: np.ndarray,
    cur_to_prev_velo: np.ndarray,
    geometry: RawKittiGeometry,
    lidar_valid: np.ndarray,
    current_covered: np.ndarray,
    grid: Tuple[int, int],
    image_size: Optional[Tuple[int, int]] = None,
    depth_tol_m: float = 0.75,
) -> Dict[str, object]:
    """Fill ego-rejected measured cells using object-compensated transforms."""
    image_size = image_size or geometry.image_size
    h, w = grid
    object_grid = np.zeros((h, w, 2), dtype=np.float32)
    object_grid_px = np.zeros((h, w, 2), dtype=np.float32)
    object_valid = np.zeros((h, w), dtype=bool)
    diagnostics = {
        "current_clusters": 0,
        "previous_clusters": 0,
        "associated": 0,
        "moving_accepted": 0,
        "stationary_skipped": 0,
        "rejected_align": 0,
        "object_cell_count": 0,
        "unrecovered_rejected_cells": int((current_covered & ~lidar_valid).sum()),
    }
    plane = fit_ground_plane_velo(cur_points_velo)
    if plane[0] is None:
        diagnostics["reason"] = "no_ground_plane"
        return {"object_valid": object_valid, "object_grid": object_grid,
                "object_grid_px": object_grid_px, "diagnostics": diagnostics}
    cur_ground = ground_proxy_mask(cur_points_velo)
    prev_ground = ground_proxy_mask(prev_points_velo)
    cur_fg = cur_points_velo[~cur_ground]
    prev_fg = prev_points_velo[~prev_ground]
    cur_clusters = cluster_nonground_points(cur_fg)
    prev_clusters = cluster_nonground_points(prev_fg)
    diagnostics["current_clusters"] = len(cur_clusters)
    diagnostics["previous_clusters"] = len(prev_clusters)
    matches = associate_clusters(cur_clusters, prev_clusters, cur_to_prev_velo)
    diagnostics["associated"] = len(matches)
    if not matches:
        return {"object_valid": object_valid, "object_grid": object_grid,
                "object_grid_px": object_grid_px, "diagnostics": diagnostics}

    cur_uv, cur_depth, cur_in = geometry.project_velo_to_image(cur_points_velo, image_size=image_size)
    cur_z, cur_idx = _nearest_depth_by_cell(cur_uv, cur_depth, cur_in, grid, image_size)
    prev_uv, prev_depth, prev_in = geometry.project_velo_to_image(prev_points_velo, image_size=image_size)
    prev_z, _ = _nearest_depth_by_cell(prev_uv, prev_depth, prev_in, grid, image_size)
    rejected = current_covered & ~lidar_valid

    # Map clustered foreground points back to original current-cloud indices.
    cur_fg_index = np.nonzero(~cur_ground)[0]

    for match in matches:
        estimate = estimate_object_transform(
            match["current"]["points"], match["previous"]["points"], cur_to_prev_velo)
        if estimate["reason"] == "stationary":
            diagnostics["stationary_skipped"] += 1
            continue
        if not estimate["accepted"]:
            diagnostics["rejected_align"] += 1
            continue
        diagnostics["moving_accepted"] += 1
        local_idx = np.asarray(match["current"]["indices"], dtype=np.int64)
        global_idx = cur_fg_index[local_idx]
        eligible_points = np.zeros((len(cur_points_velo),), dtype=bool)
        eligible_points[global_idx] = True
        chosen = cur_idx
        eligible_cells = rejected.copy()
        yy, xx = np.nonzero(eligible_cells)
        if len(yy):
            src = chosen[yy, xx]
            keep = (src >= 0) & eligible_points[src]
            mask = np.zeros_like(eligible_cells)
            mask[yy[keep], xx[keep]] = True
            eligible_cells = mask
        grid_t, px_t, valid_t = _fill_cells_with_transform(
            cur_points_velo, estimate["transform"], geometry, grid, image_size,
            depth_tol_m, prev_z, cur_uv, cur_idx, eligible_cells, eligible_points)
        write = valid_t & ~object_valid
        object_grid[write] = grid_t[write]
        object_grid_px[write] = px_t[write]
        object_valid[write] = True

    object_valid &= ~lidar_valid
    diagnostics["object_cell_count"] = int(object_valid.sum())
    diagnostics["unrecovered_rejected_cells"] = int((rejected & ~object_valid).sum())
    return {
        "object_valid": object_valid,
        "object_grid": object_grid,
        "object_grid_px": object_grid_px,
        "diagnostics": diagnostics,
    }
