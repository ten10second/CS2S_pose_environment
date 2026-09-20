"""Diagnostics for dense previous-frame reprojection.

The helpers here are evaluation-only.  They may read the target RGB image after
a reprojection method has produced its output, but they do not create history
conditions or trainable targets.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from tools.temporal_history_geometry import RawKittiGeometry, transform_points
from tools.temporal_static_geometry import _project_velo_to_image_resized


def _require_rgb(name: str, image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{name} must have shape [H,W,3]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.clip(arr, 0.0, 1.0)


def _require_xy(name: str, xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape [N,2]")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _require_mask(name: str, mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != expected {shape}")
    return arr


def _require_pose(prev_to_cur_velo: np.ndarray) -> np.ndarray:
    pose = np.asarray(prev_to_cur_velo, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("prev_to_cur_velo must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("prev_to_cur_velo must be homogeneous")
    return pose


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in vec]
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _resize_to_raw_homography(geometry: RawKittiGeometry, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    raw_w, raw_h = geometry.image_size
    sx = float(raw_w) / float(width)
    sy = float(raw_h) / float(height)
    return np.asarray(
        [[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def known_pose_fundamental(
    geometry: RawKittiGeometry,
    prev_to_cur_velo: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Return F for resized previous/current image coordinates.

    The relative pose is supplied in Velodyne coordinates.  KITTI rectified
    projection can include a translation column in P, so the epipolar transform
    is computed in the effective projected-camera coordinates used by P.
    """
    pose = _require_pose(prev_to_cur_velo)
    cam_from_velo = np.asarray(geometry.r_rect_00_ext @ geometry.t_cam_velo, dtype=np.float64)
    cur_from_prev_cam = cam_from_velo @ pose @ np.linalg.inv(cam_from_velo)
    rotation = cur_from_prev_cam[:3, :3]
    translation = cur_from_prev_cam[:3, 3]
    p = np.asarray(geometry.p_rect_02, dtype=np.float64)
    k = p[:, :3]
    inv_k = np.linalg.inv(k)
    baseline_shift = inv_k @ p[:, 3]
    effective_t = translation + baseline_shift - rotation @ baseline_shift
    f_raw = inv_k.T @ _skew(effective_t) @ rotation @ inv_k
    h_resize_to_raw = _resize_to_raw_homography(geometry, image_size)
    f_resized = h_resize_to_raw.T @ f_raw @ h_resize_to_raw
    norm = np.linalg.norm(f_resized)
    if norm > 0.0:
        f_resized = f_resized / norm
    return f_resized.astype(np.float64)


def sampson_epipolar_error(source_xy: np.ndarray, target_xy: np.ndarray, fundamental: np.ndarray) -> np.ndarray:
    """Sampson distance in pixels for known-pose epipolar consistency."""
    src = _require_xy("source_xy", source_xy).astype(np.float64)
    tgt = _require_xy("target_xy", target_xy).astype(np.float64)
    if len(src) != len(tgt):
        raise ValueError("source_xy and target_xy must have the same length")
    f = np.asarray(fundamental, dtype=np.float64)
    if f.shape != (3, 3) or not np.isfinite(f).all():
        raise ValueError("fundamental must be a finite 3x3 matrix")
    if len(src) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.concatenate([src, np.ones((len(src), 1), dtype=np.float64)], axis=1)
    x2 = np.concatenate([tgt, np.ones((len(tgt), 1), dtype=np.float64)], axis=1)
    fx1 = (f @ x1.T).T
    ftx2 = (f.T @ x2.T).T
    residual = np.sum(x2 * fx1, axis=1)
    denom = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    denom = np.maximum(denom, 1e-12)
    return (residual * residual / denom).astype(np.float32)


def filter_known_pose_matches(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    geometry: RawKittiGeometry,
    prev_to_cur_velo: np.ndarray,
    image_size: Tuple[int, int],
    max_sampson_px: float = 2.0,
    source_lidar_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Reject feature matches that disagree with the known pose/calibration."""
    src = _require_xy("source_xy", source_xy)
    tgt = _require_xy("target_xy", target_xy)
    if len(src) != len(tgt):
        raise ValueError("source_xy and target_xy must have the same length")
    f = known_pose_fundamental(geometry, prev_to_cur_velo, image_size)
    errors = sampson_epipolar_error(src, tgt, f)
    threshold_sq = float(max_sampson_px) ** 2
    inlier = errors <= threshold_sq
    height = int(image_size[1])
    width = int(image_size[0])
    upper = tgt[:, 1] < (0.5 * float(height))
    diagnostics = {
        "candidate_match_count": int(len(src)),
        "epipolar_inlier_count": int(inlier.sum()),
        "epipolar_reject_count": int((~inlier).sum()),
        "upper_image_candidate_count": int(upper.sum()),
        "upper_image_inlier_count": int((upper & inlier).sum()),
        "max_sampson_px": float(max_sampson_px),
    }
    if source_lidar_mask is not None:
        mask = _require_mask("source_lidar_mask", source_lidar_mask, (height, width))
        x = np.rint(src[:, 0]).astype(np.int64)
        y = np.rint(src[:, 1]).astype(np.int64)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        has_lidar = np.zeros((len(src),), dtype=bool)
        has_lidar[inside] = mask[y[inside], x[inside]]
        diagnostics["source_lidar_candidate_count"] = int(has_lidar.sum())
        diagnostics["source_no_lidar_candidate_count"] = int((~has_lidar).sum())
        diagnostics["source_lidar_inlier_count"] = int((has_lidar & inlier).sum())
        diagnostics["source_no_lidar_inlier_count"] = int(((~has_lidar) & inlier).sum())
    return {
        "source_xy": src[inlier],
        "target_xy": tgt[inlier],
        "epipolar_error": errors[inlier],
        "inlier_mask": inlier,
        "diagnostics": diagnostics,
    }


def _image_to_u8_gray(image: np.ndarray) -> np.ndarray:
    rgb = _require_rgb("image", image)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.rint(np.clip(gray, 0.0, 1.0) * 255.0).astype(np.uint8)


def sift_known_pose_matches(
    prev_rgb: np.ndarray,
    target_rgb: np.ndarray,
    geometry: RawKittiGeometry,
    prev_to_cur_velo: np.ndarray,
    ratio: float = 0.75,
    max_sampson_px: float = 2.0,
    max_features: int = 4000,
    source_lidar_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run deterministic OpenCV SIFT matching plus known-pose epipolar filtering.

    Real-scene matches are a noisy diagnostic because windows repeat and moving
    objects violate the static pose model; use the counts and errors as a
    sanity check, not exact ground truth.
    """
    try:
        import cv2  # type: ignore
    except ImportError as error:
        raise RuntimeError("OpenCV with SIFT is required for sift_known_pose_matches") from error
    prev = _require_rgb("prev_rgb", prev_rgb)
    target = _require_rgb("target_rgb", target_rgb)
    if prev.shape != target.shape:
        raise ValueError("prev_rgb and target_rgb must have the same shape")
    height, width = prev.shape[:2]
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=int(max_features))
    kp1, desc1 = sift.detectAndCompute(_image_to_u8_gray(prev), None)
    kp2, desc2 = sift.detectAndCompute(_image_to_u8_gray(target), None)
    if desc1 is None or desc2 is None or not kp1 or not kp2:
        empty = np.zeros((0, 2), dtype=np.float32)
        return {
            "source_xy": empty,
            "target_xy": empty,
            "epipolar_error": np.zeros((0,), dtype=np.float32),
            "diagnostics": {"source_keypoint_count": len(kp1), "target_keypoint_count": len(kp2), "epipolar_inlier_count": 0},
        }
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(desc1, desc2, k=2)
    backward = matcher.knnMatch(desc2, desc1, k=2)

    def ratio_map(pairs):
        out = {}
        for pair in pairs:
            if len(pair) < 2:
                continue
            a, b = pair
            if a.distance < float(ratio) * b.distance:
                out[int(a.queryIdx)] = int(a.trainIdx)
        return out

    fwd = ratio_map(forward)
    bwd = ratio_map(backward)
    pairs = [(i, j) for i, j in fwd.items() if bwd.get(j) == i]
    source_xy = np.asarray([kp1[i].pt for i, _j in pairs], dtype=np.float32).reshape(-1, 2)
    target_xy = np.asarray([kp2[j].pt for _i, j in pairs], dtype=np.float32).reshape(-1, 2)
    filtered = filter_known_pose_matches(
        source_xy,
        target_xy,
        geometry,
        prev_to_cur_velo,
        (width, height),
        max_sampson_px=max_sampson_px,
        source_lidar_mask=source_lidar_mask,
    )
    diag = dict(filtered["diagnostics"])
    diag.update(
        {
            "source_keypoint_count": int(len(kp1)),
            "target_keypoint_count": int(len(kp2)),
            "ratio_match_count": int(len(fwd)),
            "mutual_ratio_match_count": int(len(pairs)),
            "ratio": float(ratio),
        }
    )
    filtered["diagnostics"] = diag
    return filtered


def project_source_pixels_with_depth(
    source_xy: np.ndarray,
    pred_depth: np.ndarray,
    geometry: RawKittiGeometry,
    prev_to_cur_velo: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Project arbitrary previous-image pixels using a predicted source depth."""
    src = _require_xy("source_xy", source_xy)
    depth = np.asarray(pred_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("pred_depth must have shape [H,W]")
    height, width = depth.shape
    pose = _require_pose(prev_to_cur_velo)
    x = np.rint(src[:, 0]).astype(np.int64)
    y = np.rint(src[:, 1]).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    valid = inside.copy()
    z = np.zeros((len(src),), dtype=np.float64)
    z[inside] = depth[y[inside], x[inside]]
    valid &= np.isfinite(z) & (z > 0.0)
    if not np.any(valid):
        return {"target_xy": np.zeros((len(src), 2), dtype=np.float32), "valid": valid}
    raw_h = _resize_to_raw_homography(geometry, (width, height))
    src_h = np.concatenate([src[valid].astype(np.float64), np.ones((int(valid.sum()), 1), dtype=np.float64)], axis=1)
    raw = (raw_h @ src_h.T).T
    p = np.asarray(geometry.p_rect_02, dtype=np.float64)
    inv_k = np.linalg.inv(p[:, :3])
    b = inv_k @ p[:, 3]
    rays = (inv_k @ raw.T).T
    scale = (z[valid] + float(b[2])) / np.maximum(rays[:, 2], 1e-9)
    rect = rays * scale[:, None] - b[None, :]
    rect_h = np.concatenate([rect, np.ones((len(rect), 1), dtype=np.float64)], axis=1).T
    velo = (np.linalg.inv(geometry.r_rect_00_ext @ geometry.t_cam_velo) @ rect_h).T[:, :3]
    cur_velo = transform_points(velo, pose)
    uv, _depth, in_image = _project_velo_to_image_resized(cur_velo, geometry, (width, height))
    out = np.zeros((len(src), 2), dtype=np.float32)
    out[valid] = uv
    valid_indices = np.nonzero(valid)[0]
    valid[valid_indices] &= in_image
    return {"target_xy": out, "valid": valid}


def projected_match_errors(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    projected_target_xy: Optional[np.ndarray] = None,
    pred_depth: Optional[np.ndarray] = None,
    geometry: Optional[RawKittiGeometry] = None,
    prev_to_cur_velo: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compare diagnostic feature matches with projected source positions."""
    src = _require_xy("source_xy", source_xy)
    tgt = _require_xy("target_xy", target_xy)
    if len(src) != len(tgt):
        raise ValueError("source_xy and target_xy must have the same length")
    if projected_target_xy is None:
        if pred_depth is None or geometry is None or prev_to_cur_velo is None:
            raise ValueError("provide projected_target_xy or pred_depth with geometry and pose")
        projected = project_source_pixels_with_depth(src, pred_depth, geometry, prev_to_cur_velo)
        pred_xy = projected["target_xy"]
        valid = projected["valid"]
    else:
        pred_xy = _require_xy("projected_target_xy", projected_target_xy)
        if len(pred_xy) != len(src):
            raise ValueError("projected_target_xy must have the same length as source_xy")
        valid = np.isfinite(pred_xy).all(axis=1)
    error = np.linalg.norm(pred_xy - tgt, axis=1).astype(np.float32)
    valid_error = error[valid]
    return {
        "error_px": error,
        "valid_mask": valid,
        "diagnostics": {
            "match_count": int(len(src)),
            "valid_count": int(valid.sum()),
            "mean_error_px": float(valid_error.mean()) if len(valid_error) else float("nan"),
            "median_error_px": float(np.median(valid_error)) if len(valid_error) else float("nan"),
            "p90_error_px": float(np.percentile(valid_error, 90)) if len(valid_error) else float("nan"),
        },
    }


def masked_image_diagnostics(
    pred_rgb: np.ndarray,
    target_rgb: np.ndarray,
    pred_support: np.ndarray,
    target_support: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute RGB L1 and finite-difference gradient L1 on fixed common support."""
    pred = _require_rgb("pred_rgb", pred_rgb)
    target = _require_rgb("target_rgb", target_rgb)
    if pred.shape != target.shape:
        raise ValueError("pred_rgb and target_rgb must have the same shape")
    height, width = pred.shape[:2]
    support = _require_mask("pred_support", pred_support, (height, width))
    if target_support is not None:
        support = support & _require_mask("target_support", target_support, (height, width))
    count = int(support.sum())
    rgb_l1 = float(np.abs(pred[support] - target[support]).mean()) if count else float("nan")
    hx = support[:, 1:] & support[:, :-1]
    hy = support[1:, :] & support[:-1, :]
    gx = np.abs((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1]))
    gy = np.abs((pred[1:, :] - pred[:-1, :]) - (target[1:, :] - target[:-1, :]))
    pieces = []
    if np.any(hx):
        pieces.append(gx[hx])
    if np.any(hy):
        pieces.append(gy[hy])
    grad_l1 = float(np.concatenate(pieces, axis=0).mean()) if pieces else float("nan")
    return {
        "common_count": float(count),
        "common_coverage": float(count) / float(height * width),
        "pred_support_coverage": float(np.asarray(pred_support, dtype=bool).sum()) / float(height * width),
        "rgb_l1": rgb_l1,
        "gradient_l1": grad_l1,
        "gradient_pair_count": float(int(hx.sum() + hy.sum())),
    }
