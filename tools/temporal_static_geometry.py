"""Static history RGB reprojection for two-frame KITTI checks.

This module builds a sparse, evidence-backed reference RGB condition from the
previous frame into the current camera view.  It intentionally does not fill
holes or read the target RGB image; target RGB is only for visualization in the
CLI wrapper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from dataloader.kitti_raw_lidar_utils import (
    TrackletBox,
    box_corners_velo,
    parse_tracklet_xml,
    points_in_boxes,
    zbuffer_visible_point_indices,
)
from tools.temporal_history_geometry import (
    RawKittiGeometry,
    consecutive_rows,
    get_geometry,
    load_velodyne,
    rebase_kitti_path,
    transform_points,
)


ArrayDict = Dict[str, object]


def _require_image_size(image_size: Tuple[int, int]) -> Tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size values must be positive")
    return width, height


def _require_points(name: str, points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{name} must have shape [N,3+]")
    if not np.isfinite(arr[:, :3]).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return arr[:, :3]


def _require_pose(prev_to_cur_velo: np.ndarray) -> np.ndarray:
    pose = np.asarray(prev_to_cur_velo, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("prev_to_cur_velo must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("prev_to_cur_velo must be homogeneous")
    return pose


def _project_velo_to_image_resized(
    points_velo: np.ndarray,
    geometry: RawKittiGeometry,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points with PIL resize-compatible half-pixel scaling.

    Raw KITTI projection produces coordinates in the calibrated image size.
    Resizing with PIL maps source pixel centers by `(u + 0.5) * scale - 0.5`;
    this helper keeps the static reprojection path on that convention without
    changing the legacy geometry helpers used elsewhere.
    """
    width, height = _require_image_size(image_size)
    points = np.asarray(points_velo, dtype=np.float64)
    if points.size == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_velo must have shape [N,3+]")
    src_w, src_h = geometry.image_size
    if int(src_w) <= 0 or int(src_h) <= 0:
        raise ValueError("geometry.image_size must contain positive calibrated dimensions")
    pts_h = np.concatenate([points[:, :3], np.ones((len(points), 1), dtype=np.float64)], axis=1).T
    rect_h = geometry.r_rect_00_ext @ geometry.t_cam_velo @ pts_h
    pix = geometry.p_rect_02 @ rect_h
    depth = rect_h[2]
    safe = np.where(np.abs(pix[2]) < 1e-9, 1e-9, pix[2])
    raw_uv = (pix[:2] / safe).T
    uv = np.empty_like(raw_uv, dtype=np.float64)
    uv[:, 0] = (raw_uv[:, 0] + 0.5) * (float(width) / float(src_w)) - 0.5
    uv[:, 1] = (raw_uv[:, 1] + 0.5) * (float(height) / float(src_h)) - 0.5
    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    return uv.astype(np.float32), depth.astype(np.float32), valid.astype(bool)


def _load_resized_rgb(path: str | Path, image_size: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    width, height = _require_image_size(image_size)
    with Image.open(path) as image:
        raw_size = image.size
        rgb = image.convert("RGB").resize((width, height), Image.BILINEAR)
    return (np.asarray(rgb, dtype=np.float32) / 255.0).astype(np.float32), (int(raw_size[0]), int(raw_size[1]))


def _mask_from_regions(
    regions: object,
    image_size: Tuple[int, int],
    name: str,
) -> Optional[np.ndarray]:
    if regions is None:
        return None
    width, height = _require_image_size(image_size)
    if isinstance(regions, np.ndarray):
        mask = np.asarray(regions)
        if mask.shape != (height, width):
            raise ValueError(f"{name} mask must have shape [height,width]")
        return mask.astype(bool)
    mask = np.zeros((height, width), dtype=bool)
    for rect in regions:
        vals = [float(v) for v in rect]
        if len(vals) != 4:
            raise ValueError(f"{name} regions must be [x0,y0,x1,y1]")
        x0, y0, x1, y1 = vals
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        ix0 = max(0, min(width, int(np.floor(min(x0, x1)))))
        iy0 = max(0, min(height, int(np.floor(min(y0, y1)))))
        ix1 = max(0, min(width, int(np.ceil(max(x0, x1)))))
        iy1 = max(0, min(height, int(np.ceil(max(y0, y1)))))
        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = True
    return mask


def _manual_region_masks(static_regions: Optional[dict], image_size: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if static_regions is None:
        return None, None
    prev_regions = static_regions.get("prev_regions", static_regions.get("prev", static_regions.get("prev_mask")))
    cur_regions = static_regions.get(
        "cur_regions",
        static_regions.get(
            "target_regions",
            static_regions.get("target", static_regions.get("cur", static_regions.get("target_mask"))),
        ),
    )
    prev_mask = _mask_from_regions(prev_regions, image_size, "previous static")
    cur_mask = _mask_from_regions(cur_regions, image_size, "current static")
    if prev_mask is None or cur_mask is None:
        raise ValueError("manual static regions require both previous and current masks/rectangles")
    return prev_mask, cur_mask


def _lookup_mask(mask: Optional[np.ndarray], uv: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.ones((len(uv),), dtype=bool)
    height, width = mask.shape
    x = np.rint(uv[:, 0]).astype(np.int64)
    y = np.rint(uv[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    keep = np.zeros((len(uv),), dtype=bool)
    keep[inside] = mask[y[inside], x[inside]]
    return keep


def _visible_depth_map(
    points_velo: np.ndarray,
    geometry: RawKittiGeometry,
    image_size: Tuple[int, int],
    eligible: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    width, height = _require_image_size(image_size)
    output_size = (height, width)
    uv, depth, valid = _project_velo_to_image_resized(points_velo, geometry, image_size)
    if eligible is not None:
        valid = valid & np.asarray(eligible, dtype=bool)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    index_map = np.full((height, width), -1, dtype=np.int64)
    visible = zbuffer_visible_point_indices(uv, depth, valid, output_size)
    for idx in visible:
        x = int(np.clip(round(float(uv[idx, 0])), 0, width - 1))
        y = int(np.clip(round(float(uv[idx, 1])), 0, height - 1))
        depth_map[y, x] = float(depth[idx])
        index_map[y, x] = int(idx)
    return depth_map, index_map


def _box_region_mask(
    boxes: Sequence[TrackletBox],
    geometry: RawKittiGeometry,
    image_size: Tuple[int, int],
) -> np.ndarray:
    width, height = _require_image_size(image_size)
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        uv, _depth, valid = _project_velo_to_image_resized(box_corners_velo(box), geometry, image_size)
        if not np.any(valid):
            continue
        valid_uv = uv[valid]
        x0 = max(0, int(np.floor(valid_uv[:, 0].min())))
        y0 = max(0, int(np.floor(valid_uv[:, 1].min())))
        x1 = min(width - 1, int(np.ceil(valid_uv[:, 0].max())))
        y1 = min(height - 1, int(np.ceil(valid_uv[:, 1].max())))
        if x1 >= x0 and y1 >= y0:
            mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _project_source_visible_points(
    prev_points_velo: np.ndarray,
    prev_rgb: np.ndarray,
    geometry: RawKittiGeometry,
    image_size: Tuple[int, int],
    source_static: np.ndarray,
    prev_static_mask: Optional[np.ndarray] = None,
    source_dynamic_region: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    width, height = _require_image_size(image_size)
    uv, depth, valid = _project_velo_to_image_resized(prev_points_velo, geometry, image_size)
    # Visibility belongs to the sensor observation, so decide it from all points
    # before excluding dynamic classes from the static-history condition.
    visible = zbuffer_visible_point_indices(uv, depth, valid, (height, width))
    if visible.size:
        roi_keep = _lookup_mask(prev_static_mask, uv[visible])
        point_static = np.asarray(source_static, dtype=bool)[visible]
        region_static = ~_lookup_mask(source_dynamic_region, uv[visible]) if source_dynamic_region is not None else np.ones_like(point_static)
        keep = point_static & roi_keep & region_static
    else:
        keep = np.zeros((0,), dtype=bool)
    stats = {
        "source_visible_all_count": int(visible.size),
        "source_dynamic_visible_reject_count": int((~np.asarray(source_static, dtype=bool)[visible]).sum()) if visible.size else 0,
        "source_roi_reject_count": int((~_lookup_mask(prev_static_mask, uv[visible])).sum()) if visible.size else 0,
        "source_dynamic_2d_reject_count": int(_lookup_mask(source_dynamic_region, uv[visible]).sum()) if visible.size and source_dynamic_region is not None else 0,
    }
    visible = visible[keep]
    if visible.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            stats,
        )
    px = np.rint(uv[visible, 0]).astype(np.int64)
    py = np.rint(uv[visible, 1]).astype(np.int64)
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    colors = prev_rgb[py, px].astype(np.float32)
    return prev_points_velo[visible].astype(np.float32), colors, uv[visible].astype(np.float32), depth[visible].astype(np.float32), stats


def _zero_result(
    image_size: Tuple[int, int],
    diagnostics: Dict[str, object],
    reference_roi_mask: Optional[np.ndarray] = None,
    target_roi_mask: Optional[np.ndarray] = None,
) -> ArrayDict:
    width, height = _require_image_size(image_size)
    zeros = np.zeros((height, width), dtype=np.float32)
    roi_zero = np.zeros((height, width), dtype=bool)
    result = {
        "warped_rgb": np.zeros((3, height, width), dtype=np.float32),
        "support_mask": np.zeros((1, height, width), dtype=bool),
        "confidence": np.zeros((1, height, width), dtype=np.float32),
        "strict_mask": np.zeros((1, height, width), dtype=bool),
        "diagnostics": diagnostics,
        "debug": {
            "projected_depth": zeros.copy(),
            "target_depth": np.full((height, width), np.inf, dtype=np.float32),
            "support_xy": np.zeros((height, width, 2), dtype=np.float32),
            "prev_rgb": np.zeros((height, width, 3), dtype=np.float32),
            "reference_roi_mask": reference_roi_mask.astype(bool) if reference_roi_mask is not None else roi_zero.copy(),
            "target_roi_mask": target_roi_mask.astype(bool) if target_roi_mask is not None else roi_zero.copy(),
        },
    }
    return result


def build_static_history_from_arrays(
    prev_rgb: np.ndarray,
    prev_points_velo: np.ndarray,
    cur_points_velo: np.ndarray,
    prev_to_cur_velo: np.ndarray,
    geometry: RawKittiGeometry,
    image_size: Tuple[int, int] = (512, 128),
    depth_tol_m: float = 0.75,
    prev_boxes: Optional[Sequence[TrackletBox]] = None,
    cur_boxes: Optional[Sequence[TrackletBox]] = None,
    dynamic_exclusion_available: bool = False,
    static_regions: Optional[dict] = None,
) -> ArrayDict:
    """Project previous-frame static RGB samples into the current image.

    `prev_rgb` must already be resized to `image_size` as HWC float RGB [0,1].
    The function never consumes current-frame RGB.  Without dynamic boxes this
    is an ego-motion/depth-consistency diagnostic, not semantic static
    segmentation; moving objects can still pass the strict mask.
    """
    width, height = _require_image_size(image_size)
    if not np.isfinite(depth_tol_m) or depth_tol_m < 0.0:
        raise ValueError("depth_tol_m must be finite and nonnegative")
    rgb = np.asarray(prev_rgb, dtype=np.float32)
    if rgb.shape != (height, width, 3):
        raise ValueError("prev_rgb must have shape [height,width,3] matching image_size")
    if not np.isfinite(rgb).all():
        raise ValueError("prev_rgb contains non-finite values")
    rgb = np.clip(rgb, 0.0, 1.0)
    prev_points = _require_points("prev_points_velo", prev_points_velo)
    cur_points = _require_points("cur_points_velo", cur_points_velo)
    prev_to_cur = _require_pose(prev_to_cur_velo)

    manual_prev_mask, manual_cur_mask = _manual_region_masks(static_regions, (width, height))
    manual_regions = manual_prev_mask is not None and manual_cur_mask is not None

    diagnostics: Dict[str, object] = {
        "image_width": width,
        "image_height": height,
        "depth_tol_m": float(depth_tol_m),
        "dynamic_exclusion_available": bool(dynamic_exclusion_available),
        "manual_static_regions": bool(manual_regions),
        "dependency": "manual_static_regions_offline_diagnostic"
        if manual_regions
        else (
            "ego_motion_depth_consistency_with_instance_exclusion"
            if dynamic_exclusion_available
            else "ego_motion_depth_consistency_without_instance_exclusion"
        ),
        "source_dynamic_unknown": not bool(dynamic_exclusion_available),
        "target_dynamic_unknown": not bool(dynamic_exclusion_available),
        "vehicle_track_ids_available": False,
        "vehicle_track_ids_note": "KITTI raw tracklet parser provides per-frame boxes but drops persistent track IDs.",
    }

    prev_boxes = list(prev_boxes or [])
    cur_boxes = list(cur_boxes or [])
    source_dynamic, _ = points_in_boxes(prev_points, prev_boxes) if dynamic_exclusion_available else (
        np.zeros((len(prev_points),), dtype=bool),
        np.zeros((len(prev_points),), dtype=np.int64),
    )
    target_dynamic_points, _ = points_in_boxes(cur_points, cur_boxes) if dynamic_exclusion_available else (
        np.zeros((len(cur_points),), dtype=bool),
        np.zeros((len(cur_points),), dtype=np.int64),
    )
    source_static = ~source_dynamic
    prev_dynamic_region = _box_region_mask(prev_boxes, geometry, image_size) if dynamic_exclusion_available else None
    cur_dynamic_region = _box_region_mask(cur_boxes, geometry, image_size) if dynamic_exclusion_available else np.zeros((height, width), dtype=bool)

    source_points, source_colors, source_uv, _source_depth, source_stats = _project_source_visible_points(
        prev_points, rgb, geometry, image_size, source_static, manual_prev_mask, prev_dynamic_region
    )
    cur_depth_map, cur_index = _visible_depth_map(cur_points, geometry, image_size)
    target_dynamic_visible = np.zeros((height, width), dtype=bool)
    current_visible = cur_index >= 0
    if np.any(current_visible):
        visible_indices = cur_index[current_visible]
        target_dynamic_visible[current_visible] = target_dynamic_points[visible_indices]
    target_dynamic_visible |= cur_dynamic_region

    warped = np.zeros((height, width, 3), dtype=np.float32)
    support = np.zeros((height, width), dtype=bool)
    confidence = np.zeros((height, width), dtype=np.float32)
    strict = np.zeros((height, width), dtype=bool)
    projected_depth = np.zeros((height, width), dtype=np.float32)
    support_xy = np.zeros((height, width, 2), dtype=np.float32)

    if source_points.size:
        projected_points = transform_points(source_points, prev_to_cur)
        target_dynamic_projected, _ = points_in_boxes(projected_points, cur_boxes)
        uv_cur, z_cur, in_cur = _project_velo_to_image_resized(projected_points, geometry, image_size)
        in_cur = in_cur & ~target_dynamic_projected & _lookup_mask(manual_cur_mask, uv_cur)
        visible = zbuffer_visible_point_indices(uv_cur, z_cur, in_cur, (height, width))
        source_projected_count = int(visible.size)
        for idx in visible:
            x = int(np.clip(round(float(uv_cur[idx, 0])), 0, width - 1))
            y = int(np.clip(round(float(uv_cur[idx, 1])), 0, height - 1))
            if target_dynamic_visible[y, x]:
                continue
            z_proj = float(z_cur[idx])
            z_target = float(cur_depth_map[y, x])
            has_target_depth = np.isfinite(z_target)
            behind_target = has_target_depth and (z_proj > z_target + float(depth_tol_m))
            if behind_target:
                continue
            warped[y, x] = source_colors[idx]
            support[y, x] = True
            projected_depth[y, x] = z_proj
            support_xy[y, x] = source_uv[idx]
            if has_target_depth and abs(z_proj - z_target) <= float(depth_tol_m):
                strict[y, x] = True
                confidence[y, x] = 1.0
            elif has_target_depth:
                confidence[y, x] = 0.0
            else:
                confidence[y, x] = 0.0
    else:
        source_projected_count = 0

    support_count = int(support.sum())
    strict_count = int(strict.sum())
    dynamic_visible_reject_count = 0
    target_occluded_count = 0
    target_depth_mismatch_count = 0
    target_no_depth_count = 0
    target_roi_reject_count = 0
    target_dynamic_projected_reject_count = 0
    if source_points.size:
        projected_points = transform_points(source_points, prev_to_cur)
        target_dynamic_projected, _ = points_in_boxes(projected_points, cur_boxes)
        uv_cur, z_cur, in_cur = _project_velo_to_image_resized(projected_points, geometry, image_size)
        roi_keep = _lookup_mask(manual_cur_mask, uv_cur)
        target_roi_reject_count = int((in_cur & ~roi_keep).sum())
        target_dynamic_projected_reject_count = int((in_cur & target_dynamic_projected).sum())
        eligible = in_cur & ~target_dynamic_projected & roi_keep
        visible = zbuffer_visible_point_indices(uv_cur, z_cur, eligible, (height, width))
        for idx in visible:
            x = int(np.clip(round(float(uv_cur[idx, 0])), 0, width - 1))
            y = int(np.clip(round(float(uv_cur[idx, 1])), 0, height - 1))
            z_target = float(cur_depth_map[y, x])
            if target_dynamic_visible[y, x]:
                dynamic_visible_reject_count += 1
            elif not np.isfinite(z_target):
                target_no_depth_count += 1
            elif float(z_cur[idx]) > z_target + float(depth_tol_m):
                target_occluded_count += 1
            elif abs(float(z_cur[idx]) - z_target) > float(depth_tol_m):
                target_depth_mismatch_count += 1
    diagnostics.update(
        {
            "zero_safe": False,
            "strict_mask_note": "Depth-verified ego-motion compatible surface; not semantic static segmentation.",
            "calibrated_image_width": int(geometry.image_size[0]),
            "calibrated_image_height": int(geometry.image_size[1]),
            "source_point_count": int(len(prev_points)),
            "target_point_count": int(len(cur_points)),
            "prev_box_count": int(len(prev_boxes)),
            "cur_box_count": int(len(cur_boxes)),
            "source_dynamic_point_count": int(source_dynamic.sum()),
            "target_dynamic_point_count": int(target_dynamic_points.sum()),
            "target_dynamic_visible_pixel_count": int(target_dynamic_visible.sum()),
            "source_visible_static_count": int(len(source_points)),
            "source_projected_count": source_projected_count,
            "support_count": support_count,
            "strict_count": strict_count,
            "support_coverage": support_count / float(max(width * height, 1)),
            "strict_coverage": strict_count / float(max(width * height, 1)),
            "unverified_support_count": int((support & ~np.isfinite(cur_depth_map)).sum()),
            "target_dynamic_visible_reject_count": dynamic_visible_reject_count,
            "target_dynamic_projected_reject_count": target_dynamic_projected_reject_count,
            "target_roi_reject_count": target_roi_reject_count,
            "target_occluded_reject_count": target_occluded_count,
            "target_depth_mismatch_count": target_depth_mismatch_count,
            "target_no_depth_candidate_count": target_no_depth_count,
            "manual_prev_region_pixels": int(manual_prev_mask.sum()) if manual_regions else 0,
            "manual_cur_region_pixels": int(manual_cur_mask.sum()) if manual_regions else 0,
            **source_stats,
        }
    )
    return {
        "warped_rgb": warped.transpose(2, 0, 1).astype(np.float32),
        "support_mask": support[None].astype(bool),
        "confidence": confidence[None].astype(np.float32),
        "strict_mask": strict[None].astype(bool),
        "diagnostics": diagnostics,
        "debug": {
            "projected_depth": projected_depth.astype(np.float32),
            "target_depth": cur_depth_map.astype(np.float32),
            "support_xy": support_xy.astype(np.float32),
            "prev_rgb": rgb.astype(np.float32),
            "reference_roi_mask": manual_prev_mask.astype(bool) if manual_regions else np.zeros((height, width), dtype=bool),
            "target_roi_mask": manual_cur_mask.astype(bool) if manual_regions else np.zeros((height, width), dtype=bool),
        },
    }


def _row_with_rebased_paths(row: dict, kitti_root: str | Path | None) -> dict:
    out = dict(row)
    for key in (
        "velodyne_path",
        "oxts_path",
        "calib_dir",
        "calib_cam_to_cam_path",
        "calib_velo_to_cam_path",
        "image_02_path",
        "tracklet_xml_path",
    ):
        if key in out:
            out[key] = rebase_kitti_path(out[key], kitti_root)
    return out


def _tracklet_boxes_for_row(row: dict) -> Tuple[Optional[List[TrackletBox]], Dict[str, object]]:
    xml_path = row.get("tracklet_xml_path")
    if not xml_path:
        return None, {"available": False, "reason": "missing tracklet_xml_path"}
    path = Path(xml_path)
    if not path.is_file():
        return None, {"available": False, "reason": f"missing tracklet xml: {path}"}
    frame_index = int(row.get("frame_index", row.get("frame_id")))
    boxes = parse_tracklet_xml(str(path)).get(frame_index, [])
    return list(boxes), {"available": True, "path": str(path), "frame_index": frame_index, "box_count": len(boxes)}


def build_static_pair(
    prev_row: dict,
    cur_row: dict,
    kitti_root: str | Path | None,
    image_size: Tuple[int, int] = (512, 128),
    depth_tol_m: float = 0.75,
    static_regions: Optional[dict] = None,
) -> ArrayDict:
    """Build static previous-RGB reprojection for a manifest frame pair.

    The formal builder reads previous RGB, both LiDAR scans, calibration, OXTS
    pose, and optional tracklet boxes.  It never opens the current RGB path.
    """
    width, height = _require_image_size(image_size)
    if not consecutive_rows(prev_row, cur_row):
        raise ValueError("static history requires consecutive frames from the same date/drive")
    prev = _row_with_rebased_paths(prev_row, kitti_root)
    cur = _row_with_rebased_paths(cur_row, kitti_root)
    for key in ("image_02_path", "velodyne_path", "oxts_path", "calib_dir"):
        if key not in prev:
            raise KeyError(f"previous row missing {key}")
    for key in ("velodyne_path", "oxts_path", "calib_dir"):
        if key not in cur:
            raise KeyError(f"current row missing {key}")

    prev_boxes, prev_box_info = _tracklet_boxes_for_row(prev)
    cur_boxes, cur_box_info = _tracklet_boxes_for_row(cur)
    dynamic_available = bool(prev_box_info["available"] and cur_box_info["available"])
    manual_prev_mask, manual_cur_mask = _manual_region_masks(static_regions, (width, height))
    manual_regions = manual_prev_mask is not None and manual_cur_mask is not None

    geometry = get_geometry(cur["calib_dir"])
    prev_rgb, prev_raw_size = _load_resized_rgb(prev["image_02_path"], (width, height))
    prev_to_cur = geometry.relative_velo_pose(prev["oxts_path"], cur["oxts_path"])
    prev_points = load_velodyne(prev["velodyne_path"])
    cur_points = load_velodyne(cur["velodyne_path"])
    out = build_static_history_from_arrays(
        prev_rgb=prev_rgb,
        prev_points_velo=prev_points,
        cur_points_velo=cur_points,
        prev_to_cur_velo=prev_to_cur,
        geometry=geometry,
        image_size=(width, height),
        depth_tol_m=depth_tol_m,
        prev_boxes=prev_boxes,
        cur_boxes=cur_boxes,
        dynamic_exclusion_available=dynamic_available,
        static_regions={"prev_mask": manual_prev_mask, "target_mask": manual_cur_mask} if manual_regions else None,
    )
    out["diagnostics"]["prev_tracklet"] = prev_box_info
    out["diagnostics"]["cur_tracklet"] = cur_box_info
    out["diagnostics"]["previous_rgb_raw_width"] = int(prev_raw_size[0])
    out["diagnostics"]["previous_rgb_raw_height"] = int(prev_raw_size[1])
    out["diagnostics"]["previous_rgb_matches_calibration_size"] = tuple(prev_raw_size) == tuple(geometry.image_size)
    out["diagnostics"]["diagnostic_missing_field_count"] = int(not prev_box_info["available"]) + int(not cur_box_info["available"])
    return out
