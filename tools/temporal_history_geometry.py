"""Geometry for sparse current-query -> previous-history lookup.

This module intentionally does not fill holes.  A current latent cell receives a
previous-frame coordinate only when a current LiDAR surface projects into that
cell and the same world point is also supported by the previous LiDAR scan.
Unknown cells stay invalid instead of falling back to identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def parse_calib_file(path: str | Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            data[key.strip()] = np.asarray([float(x) for x in value.split()], dtype=np.float64)
        except ValueError:
            continue
    return data


def resolve_calib_dir(calib_dir: str | Path) -> Path:
    path = Path(calib_dir)
    if (path / "calib_cam_to_cam.txt").is_file():
        return path
    parent = path.parent
    if (parent / "calib_cam_to_cam.txt").is_file():
        return parent
    for candidate in path.rglob("calib_cam_to_cam.txt"):
        if (candidate.parent / "calib_velo_to_cam.txt").is_file():
            return candidate.parent
    raise FileNotFoundError(f"calib_cam_to_cam.txt not found under {path}")


def rebase_kitti_path(path: str | Path, kitti_root: str | Path | None) -> str:
    value = str(path)
    if not kitti_root:
        return value
    marker = "KITTI_RAW/"
    idx = value.find(marker)
    if idx >= 0:
        return str(Path(kitti_root) / value[idx + len(marker):])
    return value


def _transform_from_rt(calib: Dict[str, np.ndarray]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = calib["R"].reshape(3, 3)
    transform[:3, 3] = calib["T"].reshape(3)
    return transform


def _rot_from_oxts(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


@dataclass
class RawKittiGeometry:
    """Calibration and pose helpers for KITTI raw image_02 geometry."""

    p_rect_02: np.ndarray
    r_rect_00_ext: np.ndarray
    t_cam_velo: np.ndarray
    t_imu_velo: Optional[np.ndarray]
    image_size: Tuple[int, int]  # (width, height)

    @classmethod
    def from_calib_dir(cls, calib_dir: str | Path) -> "RawKittiGeometry":
        calib_dir = resolve_calib_dir(calib_dir)
        cam = parse_calib_file(calib_dir / "calib_cam_to_cam.txt")
        velo = parse_calib_file(calib_dir / "calib_velo_to_cam.txt")
        imu_path = calib_dir / "calib_imu_to_velo.txt"
        imu = parse_calib_file(imu_path) if imu_path.is_file() else {}

        r_rect = cam.get("R_rect_00", np.eye(3, dtype=np.float64).reshape(-1)).reshape(3, 3)
        r_rect_ext = np.eye(4, dtype=np.float64)
        r_rect_ext[:3, :3] = r_rect
        image_size = tuple(int(v) for v in cam.get("S_rect_02", np.asarray([1242.0, 375.0]))[:2])
        return cls(
            p_rect_02=cam["P_rect_02"].reshape(3, 4),
            r_rect_00_ext=r_rect_ext,
            t_cam_velo=_transform_from_rt(velo),
            t_imu_velo=_transform_from_rt(imu) if imu else None,
            image_size=(int(image_size[0]), int(image_size[1])),
        )

    def project_velo_to_image(
        self,
        points_velo: np.ndarray,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project Velodyne points to rectified image_02 pixels.

        Uses the full 3x4 P_rect_02, including its translation column.  Returns
        uv in the requested image size, rectified depth, and an in-image mask.
        """
        points = np.asarray(points_velo, dtype=np.float64)
        if points.size == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )
        if points.shape[1] > 3:
            points = points[:, :3]
        out_w, out_h = image_size or self.image_size
        src_w, src_h = self.image_size
        pts_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1).T
        rect_h = self.r_rect_00_ext @ self.t_cam_velo @ pts_h
        pix = self.p_rect_02 @ rect_h
        depth = rect_h[2]
        safe = np.where(np.abs(pix[2]) < 1e-9, 1e-9, pix[2])
        uv = (pix[:2] / safe).T
        uv[:, 0] *= float(out_w) / float(src_w)
        uv[:, 1] *= float(out_h) / float(src_h)
        valid = (
            np.isfinite(depth)
            & (depth > 0.0)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] < out_w)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < out_h)
        )
        return uv.astype(np.float32), depth.astype(np.float32), valid

    def oxts_pose(self, oxts_path: str | Path) -> np.ndarray:
        vals = np.loadtxt(oxts_path)
        lat, lon, alt, roll, pitch, yaw = vals[:6]
        earth_radius = 6378137.0
        scale = np.cos(48.9 * np.pi / 180.0)
        mx = scale * lon * np.pi * earth_radius / 180.0
        my = scale * earth_radius * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _rot_from_oxts(float(roll), float(pitch), float(yaw))
        pose[:3, 3] = [mx, my, alt]
        return pose

    def relative_velo_pose(self, oxts_prev: str | Path, oxts_cur: str | Path) -> np.ndarray:
        """Map previous Velodyne coordinates to current Velodyne coordinates."""
        if self.t_imu_velo is None:
            raise ValueError("calib_imu_to_velo.txt is required for OXTS relative poses")
        prev_world = self.oxts_pose(oxts_prev)
        cur_world = self.oxts_pose(oxts_cur)
        t_velo_imu = np.linalg.inv(self.t_imu_velo)
        return self.t_imu_velo @ np.linalg.inv(cur_world) @ prev_world @ t_velo_imu


def load_velodyne(path: str | Path, max_range: float = 80.0) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    dist_xy = np.linalg.norm(pts[:, :2], axis=1)
    keep = np.isfinite(pts).all(axis=1) & (dist_xy <= max_range) & (pts[:, 2] > -3.5)
    return pts[keep].astype(np.float32)


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pts_h = np.concatenate([points[:, :3], np.ones((len(points), 1), dtype=np.float64)], axis=1).T
    out = (np.asarray(transform, dtype=np.float64) @ pts_h).T[:, :3]
    return out.astype(np.float32)


def align_corners_false_normalize(pixel_xy: np.ndarray, grid: Tuple[int, int]) -> np.ndarray:
    """Convert feature-grid pixel centers/coords to grid_sample normalized coords."""
    h, w = grid
    out = np.empty_like(pixel_xy, dtype=np.float32)
    out[..., 0] = (2.0 * (pixel_xy[..., 0] + 0.5) / float(w)) - 1.0
    out[..., 1] = (2.0 * (pixel_xy[..., 1] + 0.5) / float(h)) - 1.0
    return out


def image_to_feature_xy(pixel_xy: np.ndarray, image_size: Tuple[int, int], grid: Tuple[int, int]) -> np.ndarray:
    """Map continuous image pixel coordinates to feature-index coordinates."""
    img_w, img_h = image_size
    h, w = grid
    out = np.empty_like(pixel_xy, dtype=np.float32)
    out[..., 0] = (pixel_xy[..., 0] + 0.5) * (float(w) / float(img_w)) - 0.5
    out[..., 1] = (pixel_xy[..., 1] + 0.5) * (float(h) / float(img_h)) - 0.5
    return out


def _nearest_depth_by_cell(
    uv: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    grid: Tuple[int, int],
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = grid
    img_w, img_h = image_size
    cell_x = np.floor(uv[:, 0] * (float(w) / float(img_w))).astype(np.int64)
    cell_y = np.floor(uv[:, 1] * (float(h) / float(img_h))).astype(np.int64)
    ok = valid & (cell_x >= 0) & (cell_x < w) & (cell_y >= 0) & (cell_y < h)
    z = np.full((h, w), np.inf, dtype=np.float32)
    idx = np.full((h, w), -1, dtype=np.int64)
    if not np.any(ok):
        return z, idx
    flat = cell_y[ok] * w + cell_x[ok]
    order = np.argsort(depth[ok])
    src_idx = np.nonzero(ok)[0][order]
    flat_sorted = flat[order]
    _, first = np.unique(flat_sorted, return_index=True)
    chosen = src_idx[first]
    yy = cell_y[chosen]
    xx = cell_x[chosen]
    z[yy, xx] = depth[chosen]
    idx[yy, xx] = chosen
    return z, idx


def fit_ground_plane_velo(points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 50:
        return None, None
    r = np.linalg.norm(pts[:, :2], axis=1)
    sel = (pts[:, 2] < -1.2) & (r > 3.0) & (r < 25.0)
    road = pts[sel]
    if len(road) < 50:
        return None, None
    a_mat = np.stack([road[:, 0], road[:, 1], np.ones(len(road))], axis=1)
    sol, *_ = np.linalg.lstsq(a_mat, road[:, 2], rcond=None)
    a, b, c = sol
    normal = np.asarray([-a, -b, 1.0], dtype=np.float64)
    norm = np.linalg.norm(normal)
    normal /= max(norm, 1e-9)
    d = -c / max(norm, 1e-9)
    return normal.astype(np.float32), float(d)


def ground_proxy_mask(points: np.ndarray, threshold_m: float = 0.35) -> np.ndarray:
    normal, d = fit_ground_plane_velo(points)
    if normal is None:
        return np.zeros((len(points),), dtype=bool)
    signed = np.asarray(points[:, :3], dtype=np.float64) @ normal.astype(np.float64) + float(d)
    return np.abs(signed) <= threshold_m


def build_history_geometry_from_arrays(
    prev_points_velo: np.ndarray,
    cur_points_velo: np.ndarray,
    cur_to_prev_velo: np.ndarray,
    geometry: RawKittiGeometry,
    grid: Tuple[int, int] = (16, 64),
    image_size: Optional[Tuple[int, int]] = None,
    depth_tol_m: float = 0.75,
) -> Dict[str, np.ndarray | Dict[str, float]]:
    """Build sparse previous-coordinate grid for current latent cells.

    Args:
        prev_points_velo: previous scan points in previous Velodyne frame.
        cur_points_velo: current scan points in current Velodyne frame.
        cur_to_prev_velo: 4x4 transform from current Velodyne to previous.
        geometry: calibration shared by this KITTI raw date/drive.
        grid: feature grid as (H, W).
        image_size: projection image size as (W, H); defaults to raw rect size.
        depth_tol_m: previous-scan z-buffer support tolerance.
    """
    image_size = image_size or geometry.image_size
    h, w = grid
    history_grid_px = np.zeros((h, w, 2), dtype=np.float32)
    history_grid = np.zeros((h, w, 2), dtype=np.float32)
    history_valid = np.zeros((h, w), dtype=bool)
    current_grid_px = np.zeros((h, w, 2), dtype=np.float32)
    current_point_grid_px = np.zeros((h, w, 2), dtype=np.float32)
    previous_point_grid_px = np.zeros((h, w, 2), dtype=np.float32)
    current_point_image_xy = np.zeros((h, w, 2), dtype=np.float32)
    previous_point_image_xy = np.zeros((h, w, 2), dtype=np.float32)
    source_depth = np.full((h, w), np.inf, dtype=np.float32)
    target_depth = np.full((h, w), np.inf, dtype=np.float32)
    ground_proxy = np.zeros((h, w), dtype=bool)
    non_ground_proxy = np.zeros((h, w), dtype=bool)

    cur_uv, cur_depth, cur_in = geometry.project_velo_to_image(cur_points_velo, image_size=image_size)
    cur_z, cur_idx = _nearest_depth_by_cell(cur_uv, cur_depth, cur_in, grid, image_size)
    cells_with_cur = cur_idx >= 0
    if not np.any(cells_with_cur):
        metrics = _metrics(history_valid, cells_with_cur, ground_proxy, non_ground_proxy, current_grid_px, history_grid_px, grid)
        return {
            "history_grid": history_grid,
            "history_grid_px": history_grid_px,
            "current_grid_px": current_grid_px,
            "current_point_grid_px": current_point_grid_px,
            "previous_point_grid_px": previous_point_grid_px,
            "current_point_image_xy": current_point_image_xy,
            "previous_point_image_xy": previous_point_image_xy,
            "history_valid": history_valid,
            "current_covered": cells_with_cur,
            "ground_proxy": ground_proxy,
            "non_ground_proxy": non_ground_proxy,
            "source_depth": source_depth,
            "target_depth": target_depth,
            "metrics": metrics,
        }

    prev_uv, prev_depth, prev_in = geometry.project_velo_to_image(prev_points_velo, image_size=image_size)
    prev_z, _prev_idx = _nearest_depth_by_cell(prev_uv, prev_depth, prev_in, grid, image_size)

    cur_points_prev = transform_points(cur_points_velo, cur_to_prev_velo)
    mapped_uv, mapped_depth, mapped_in = geometry.project_velo_to_image(cur_points_prev, image_size=image_size)
    cur_ground = ground_proxy_mask(cur_points_velo)

    img_w, img_h = image_size
    prev_cell_x = np.floor(mapped_uv[:, 0] * (float(w) / float(img_w))).astype(np.int64)
    prev_cell_y = np.floor(mapped_uv[:, 1] * (float(h) / float(img_h))).astype(np.int64)
    support_ok = np.zeros((len(cur_points_velo),), dtype=bool)
    in_prev_cell = (
        mapped_in
        & (prev_cell_x >= 0)
        & (prev_cell_x < w)
        & (prev_cell_y >= 0)
        & (prev_cell_y < h)
    )
    if np.any(in_prev_cell):
        nearest_prev = np.full((len(cur_points_velo),), np.inf, dtype=np.float32)
        nearest_prev[in_prev_cell] = prev_z[prev_cell_y[in_prev_cell], prev_cell_x[in_prev_cell]]
        support_ok = in_prev_cell & np.isfinite(nearest_prev) & (np.abs(nearest_prev - mapped_depth) <= depth_tol_m)

    yy, xx = np.nonzero(cells_with_cur)
    chosen = cur_idx[yy, xx]
    chosen_ok = support_ok[chosen]
    if np.any(chosen_ok):
        vy = yy[chosen_ok]
        vx = xx[chosen_ok]
        src = chosen[chosen_ok]
        prev_feature_px = image_to_feature_xy(mapped_uv[src], image_size, grid)
        cur_feature_px = image_to_feature_xy(cur_uv[src], image_size, grid)
        query_center_px = np.stack([vx.astype(np.float32), vy.astype(np.float32)], axis=1)
        history_feature_px = query_center_px + (prev_feature_px - cur_feature_px)
        in_feature = (
            (history_feature_px[:, 0] >= -0.5)
            & (history_feature_px[:, 0] <= w - 0.5)
            & (history_feature_px[:, 1] >= -0.5)
            & (history_feature_px[:, 1] <= h - 0.5)
        )
        vy = vy[in_feature]
        vx = vx[in_feature]
        src = src[in_feature]
        history_feature_px = history_feature_px[in_feature]
        prev_feature_px = prev_feature_px[in_feature]
        cur_feature_px = cur_feature_px[in_feature]
        if len(src) == 0:
            metrics = _metrics(history_valid, cells_with_cur, ground_proxy, non_ground_proxy, current_grid_px, history_grid_px, grid)
            return {
                "history_grid": history_grid,
                "history_grid_px": history_grid_px,
                "current_grid_px": current_grid_px,
                "current_point_grid_px": current_point_grid_px,
                "previous_point_grid_px": previous_point_grid_px,
                "current_point_image_xy": current_point_image_xy,
                "previous_point_image_xy": previous_point_image_xy,
                "history_valid": history_valid,
                "current_covered": cells_with_cur,
                "ground_proxy": ground_proxy,
                "non_ground_proxy": non_ground_proxy,
                "source_depth": source_depth,
                "target_depth": target_depth,
                "metrics": metrics,
            }
        history_grid_px[vy, vx] = history_feature_px
        current_grid_px[vy, vx] = np.stack(
            [vx.astype(np.float32), vy.astype(np.float32)],
            axis=1,
        )
        current_point_grid_px[vy, vx] = cur_feature_px
        previous_point_grid_px[vy, vx] = prev_feature_px
        current_point_image_xy[vy, vx] = cur_uv[src]
        previous_point_image_xy[vy, vx] = mapped_uv[src]
        history_grid[vy, vx] = align_corners_false_normalize(history_feature_px, grid)
        history_valid[vy, vx] = True
        source_depth[vy, vx] = mapped_depth[src]
        target_depth[vy, vx] = cur_z[vy, vx]
        ground_proxy[vy, vx] = cur_ground[src]
        non_ground_proxy[vy, vx] = ~cur_ground[src]

    metrics = _metrics(history_valid, cells_with_cur, ground_proxy, non_ground_proxy, current_grid_px, history_grid_px, grid)
    return {
        "history_grid": history_grid,
        "history_grid_px": history_grid_px,
        "current_grid_px": current_grid_px,
        "current_point_grid_px": current_point_grid_px,
        "previous_point_grid_px": previous_point_grid_px,
        "current_point_image_xy": current_point_image_xy,
        "previous_point_image_xy": previous_point_image_xy,
        "history_valid": history_valid,
        "current_covered": cells_with_cur,
        "ground_proxy": ground_proxy,
        "non_ground_proxy": non_ground_proxy,
        "source_depth": source_depth,
        "target_depth": target_depth,
        "metrics": metrics,
    }


def _metrics(
    history_valid: np.ndarray,
    current_covered: np.ndarray,
    ground_proxy: np.ndarray,
    non_ground_proxy: np.ndarray,
    current_grid_px: np.ndarray,
    history_grid_px: np.ndarray,
    grid: Tuple[int, int],
) -> Dict[str, float]:
    total = float(history_valid.size)
    current_count = int(current_covered.sum())
    valid_count = int(history_valid.sum())
    ground_count = int((history_valid & ground_proxy).sum())
    nonground_count = int((history_valid & non_ground_proxy).sum())
    if valid_count:
        err = np.linalg.norm(history_grid_px[history_valid] - current_grid_px[history_valid], axis=1)
        mean_reproj = float(err.mean())
        p95_reproj = float(np.percentile(err, 95))
    else:
        mean_reproj = 0.0
        p95_reproj = 0.0
    return {
        "grid_h": int(grid[0]),
        "grid_w": int(grid[1]),
        "num_cells": int(total),
        "current_lidar_cell_count": current_count,
        "valid_history_cell_count": valid_count,
        "coverage_all": valid_count / total,
        "coverage_of_current_lidar": valid_count / max(float(current_count), 1.0),
        "coverage_ground_proxy_all": ground_count / total,
        "coverage_non_ground_proxy_all": nonground_count / total,
        "coverage_ground_proxy": ground_count / max(float(valid_count), 1.0),
        "coverage_non_ground_proxy": nonground_count / max(float(valid_count), 1.0),
        "mean_reprojection_vs_identity_cells": mean_reproj,
        "p95_reprojection_vs_identity_cells": p95_reproj,
        "note": "ground/non-ground is a geometric height proxy, not semantic facade labels",
    }


_GEOMETRY_CACHE: Dict[str, RawKittiGeometry] = {}


def get_geometry(calib_dir: str | Path) -> RawKittiGeometry:
    resolved = str(resolve_calib_dir(calib_dir))
    if resolved not in _GEOMETRY_CACHE:
        _GEOMETRY_CACHE[resolved] = RawKittiGeometry.from_calib_dir(resolved)
    return _GEOMETRY_CACHE[resolved]


def consecutive_rows(prev_row: dict, cur_row: dict) -> bool:
    return (
        prev_row.get("date") == cur_row.get("date")
        and prev_row.get("drive") == cur_row.get("drive")
        and int(cur_row.get("frame_index", cur_row.get("frame_id"))) == int(prev_row.get("frame_index", prev_row.get("frame_id"))) + 1
    )


def build_pair_geometry(
    prev_row: dict,
    cur_row: dict,
    kitti_root: str | Path | None = None,
    grid: Tuple[int, int] = (16, 64),
    max_range: float = 80.0,
    depth_tol_m: float = 0.75,
) -> Dict[str, np.ndarray | Dict[str, float]]:
    row_prev = dict(prev_row)
    row_cur = dict(cur_row)
    for row in (row_prev, row_cur):
        for key in ("velodyne_path", "oxts_path", "calib_dir", "calib_cam_to_cam_path", "calib_velo_to_cam_path", "image_02_path"):
            if key in row:
                row[key] = rebase_kitti_path(row[key], kitti_root)

    geometry = get_geometry(row_cur["calib_dir"])
    prev_to_cur = geometry.relative_velo_pose(row_prev["oxts_path"], row_cur["oxts_path"])
    cur_to_prev = np.linalg.inv(prev_to_cur)
    prev_points = load_velodyne(row_prev["velodyne_path"], max_range=max_range)
    cur_points = load_velodyne(row_cur["velodyne_path"], max_range=max_range)
    result = build_history_geometry_from_arrays(
        prev_points,
        cur_points,
        cur_to_prev,
        geometry,
        grid=grid,
        image_size=geometry.image_size,
        depth_tol_m=depth_tol_m,
    )
    metrics = dict(result["metrics"])
    metrics.update(
        {
            "prev_sample_id": row_prev.get("sample_id", ""),
            "cur_sample_id": row_cur.get("sample_id", ""),
            "prev_frame_index": int(row_prev.get("frame_index", row_prev.get("frame_id", -1))),
            "cur_frame_index": int(row_cur.get("frame_index", row_cur.get("frame_id", -1))),
            "date": row_cur.get("date", ""),
            "drive": row_cur.get("drive", ""),
            "prev_to_cur_translation_m": float(np.linalg.norm(prev_to_cur[:3, 3])),
        }
    )
    result["metrics"] = metrics
    return result


def stream_consecutive_pairs(rows: Iterable[dict]) -> Iterable[Tuple[dict, dict]]:
    prev = None
    for row in rows:
        if prev is not None and consecutive_rows(prev, row):
            yield prev, row
        prev = row
