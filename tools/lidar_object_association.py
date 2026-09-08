"""Label-free cross-frame object association from raw velodyne scans.

Given two consecutive scans (prev, cur) and the ego-motion transform between
them, cluster "unsupported" current-frame returns (surfaces not explained by
the ego-motion-compensated previous scan) into object candidates, match them
against clusters of the previous scan, and return per-object latent-space
transport displacements.

This is pure sensor geometry: no learned components, no annotations. It is
the instance-correspondence signal for segmented temporal latent transport:
  - matched object   -> transport the previous latent content by the object's
                        own image-space displacement (identity carried over)
  - unmatched object -> reset to fresh noise (first appearance / occlusion)
  - static cells     -> ground-homography / static-flow transport (pose_warp2)
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def voxel_downsample(points, voxel=0.3):
    keys = np.floor(points[:, :3] / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def above_ground_mask(points, plane, h_min=0.4, h_max=4.0):
    """plane = (a, b, c) with ground z = a*x + b*y - c (fit_ground_plane_velo)."""
    if plane is None:
        return np.zeros(len(points), bool)
    a, b, c = plane
    h = points[:, 2] - (a * points[:, 0] + b * points[:, 1] - c)
    return (h > h_min) & (h < h_max)


def transform_ground_plane(plane, T_cur_prev):
    """Transform previous-frame ground z = a*x + b*y - c into current coords."""
    if plane is None:
        return None
    a, b, c = plane
    n_src = np.array([-a, -b, 1.0], dtype=np.float64)
    R = T_cur_prev[:3, :3]
    t = T_cur_prev[:3, 3]
    n_dst = R @ n_src
    if abs(n_dst[2]) < 1e-8:
        return None
    c_dst = float(c - n_dst @ t)
    return (-n_dst[0] / n_dst[2], -n_dst[1] / n_dst[2], c_dst / n_dst[2])


AZ_BIN = 0.2
EL_BIN = 0.4
AZ_MAX = 360.0
EL_MIN, EL_MAX = -26.0, 4.0
NA = int(AZ_MAX / AZ_BIN)
NE = int((EL_MAX - EL_MIN) / EL_BIN)


def _sph_coords(points):
    r = np.linalg.norm(points, axis=1)
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360
    el = np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))
    return az, el, r


def _range_image(points):
    az, el, r = _sph_coords(points)
    ri = np.full((NE, NA), np.inf, np.float32)
    valid = np.isfinite(r) & (r > 0.0) & (el >= EL_MIN) & (el < EL_MAX)
    if valid.any():
        ia = ((az[valid] / AZ_BIN).astype(int)) % NA
        ie = ((el[valid] - EL_MIN) / EL_BIN).astype(int)
        np.minimum.at(ri, (ie, ia), r[valid].astype(np.float32))
    return ri


def cluster_objects(points, radius=0.8, min_points=12):
    """Radius-graph connected components. Returns list of
    {idx, centroid, count} sorted by count desc."""
    n = len(points)
    if n == 0:
        return []
    tree = cKDTree(points[:, :3])
    pairs = tree.query_pairs(radius, output_type="ndarray")
    labels = np.full(n, -1, dtype=np.int64)
    if len(pairs):
        g = coo_matrix(
            (np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
        )
        _, lab = connected_components(g, directed=False)
        labels = lab
    else:
        labels = np.arange(n, dtype=np.int64)
    objects = []
    for lid in np.unique(labels):
        idx = np.where(labels == lid)[0]
        if len(idx) < min_points:
            continue
        objects.append(
            {
                "idx": idx,
                "centroid": points[idx, :3].mean(axis=0),
                "count": int(len(idx)),
            }
        )
    objects.sort(key=lambda o: -o["count"])
    return objects


def _latent_cells(u, v, img_w, img_h, lat_w, lat_h, dilate=1):
    cells = np.zeros((lat_h, lat_w), bool)
    lu = (u * lat_w / img_w).astype(int)
    lv = (v * lat_h / img_h).astype(int)
    ok = (lu >= 0) & (lu < lat_w) & (lv >= 0) & (lv < lat_h)
    cells[lv[ok], lu[ok]] = True
    if dilate > 0:
        cells = ndi.binary_dilation(cells, iterations=dilate)
    return cells


def _in_fov(points, geom, img_w, img_h, min_depth=1.0):
    """Rectified-camera FOV filter: only in-image surfaces can participate in
    latent transport (out-of-FOV content exists in neither image)."""
    u, v, d = geom.project_velo_to_rect_img(points[:, :3])
    return (d > min_depth) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)


def raw_previous_refresh_mask(
    points_prev,
    points_cur,
    T_cur_prev,
    geom,
    img_w,
    img_h,
    lat_h,
    lat_w,
    plane=None,
    tol=0.3,
):
    """Cells in the previous raw latent footprint that should be refreshed.

    Previous above-ground in-FOV returns are ego-transformed into the current
    LiDAR frame and compared against current-scan angular range neighborhoods.
    Any observed depth disagreement beyond tol, closer or farther, marks the
    original previous-frame projection. This is a conservative range-bin
    heuristic: unobserved angular bins and bins with any agreeing neighbor are
    left unchanged.
    """
    mask = np.zeros((lat_h, lat_w), bool)
    stats = {
        "raw_prev_candidates": 0,
        "raw_prev_observed": 0,
        "raw_prev_refresh_points": 0,
        "raw_prev_refresh_point_frac": 0.0,
        "raw_prev_refresh_mask_frac": 0.0,
    }
    if len(points_prev) == 0 or plane is None:
        return mask, stats

    prev_ok = above_ground_mask(points_prev, plane) & _in_fov(points_prev, geom, img_w, img_h)
    stats["raw_prev_candidates"] = int(prev_ok.sum())
    if not prev_ok.any() or len(points_cur) == 0:
        return mask, stats

    p1 = points_prev[prev_ok]
    q1 = (T_cur_prev[:3, :3] @ p1.T).T + T_cur_prev[:3, 3]
    ri = _range_image(points_cur)
    az, el, r = _sph_coords(q1[:, :3])
    valid_ang = np.isfinite(r) & (r > 0.0) & (el >= EL_MIN) & (el < EL_MAX)
    ia = ((az / AZ_BIN).astype(int)) % NA
    ie = ((el - EL_MIN) / EL_BIN).astype(int)

    observed = np.zeros(len(r), bool)
    agrees = np.zeros(len(r), bool)
    for da in (-1, 0, 1):
        for de in (-1, 0, 1):
            cur_r = ri[np.clip(ie + de, 0, NE - 1), (ia + da) % NA]
            finite = np.isfinite(cur_r)
            observed |= valid_ang & finite
            agrees |= valid_ang & finite & (np.abs(cur_r - r) <= tol)

    refresh = observed & ~agrees
    stats["raw_prev_observed"] = int(observed.sum())
    stats["raw_prev_refresh_points"] = int(refresh.sum())
    stats["raw_prev_refresh_point_frac"] = float(refresh.mean())
    if not refresh.any():
        return mask, stats

    u, v, dep = geom.project_velo_to_rect_img(p1[refresh, :3])
    ok = (dep > 1.0) & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    if ok.any():
        mask = _latent_cells(u[ok], v[ok], img_w, img_h, lat_w, lat_h)
        stats["raw_prev_refresh_mask_frac"] = float(mask.mean())
    return mask, stats


def associate_objects(
    points_prev,
    points_cur,
    T_velo_cur_prev,
    geom,
    img_w,
    img_h,
    lat_h,
    lat_w,
    plane=None,
    explain_tol=0.35,
    match_radius=2.5,
    motion_gate=2.0,
    size_ratio_gate=(0.25, 4.0),
    cluster_voxel=0.3,
):
    """Associate moving-object clusters across consecutive scans.

    Returns (objects, stats):
      objects: list of matched objects
        {cell_mask (lat_h,lat_w bool): latent cells the object occupies now,
         prev_cell_mask (lat_h,lat_w bool): latent cells occupied previously,
         d_lat (2,): backward displacement in latent coords (src = pos - d_lat),
         d_velo (3,): 3D motion in the current velo frame,
         count, count_prev, depth}
      stats: diagnostics {n_cur_obj, n_prev_obj, n_matched, unexplained_frac}
    """
    p1 = points_prev
    p2 = points_cur
    q1 = (T_velo_cur_prev[:3, :3] @ p1.T).T + T_velo_cur_prev[:3, 3]

    tree_q = cKDTree(q1[:, :3])
    dist_cur, _ = tree_q.query(p2[:, :3], k=1)
    unexplained = dist_cur > explain_tol
    stats = {"unexplained_frac": float(unexplained.mean())}
    plane_cur = transform_ground_plane(plane, T_velo_cur_prev)

    cur_pool = voxel_downsample(
        p2[unexplained & above_ground_mask(p2, plane_cur) & _in_fov(p2, geom, img_w, img_h)],
        cluster_voxel,
    )
    # Previous-side clusters cover ALL above-ground in-FOV returns: an object
    # visible at prev is trivially "explained" by itself, so the unexplained
    # filter must not apply here — the matching distance gate handles identity.
    prev_pool = voxel_downsample(
        p1[above_ground_mask(p1, plane) & _in_fov(p1, geom, img_w, img_h)],
        cluster_voxel,
    )

    cur_objs = cluster_objects(cur_pool)
    prev_objs = cluster_objects(prev_pool)
    stats["n_cur_obj"] = len(cur_objs)
    stats["n_prev_obj"] = len(prev_objs)

    T_inv = np.linalg.inv(T_velo_cur_prev)
    matched = []
    used_prev = set()
    for cand in cur_objs:
        c_cur = cand["centroid"]
        c_in_prev = (T_inv[:3, :3] @ c_cur) + T_inv[:3, 3]
        best, best_k, best_d = None, None, match_radius
        for k, po in enumerate(prev_objs):
            if k in used_prev:
                continue
            d3 = np.linalg.norm(po["centroid"] - c_in_prev)
            if d3 < best_d:
                ratio = cand["count"] / max(po["count"], 1)
                if size_ratio_gate[0] <= ratio <= size_ratio_gate[1]:
                    best, best_k, best_d = po, k, d3
        if best is None:
            continue
        a_in_cur = (T_velo_cur_prev[:3, :3] @ best["centroid"]) + T_velo_cur_prev[:3, 3]
        d_velo = c_cur - a_in_cur
        if np.linalg.norm(d_velo) > motion_gate:
            continue

        u_c, v_c, dep_c = geom.project_velo_to_rect_img(cur_pool[cand["idx"]][:, :3])
        ok_c = (
            (dep_c > 1.0)
            & (u_c >= 0) & (u_c < img_w) & (v_c >= 0) & (v_c < img_h)
        )
        if ok_c.sum() < 3:
            continue
        u_p, v_p, dep_p = geom.project_velo_to_rect_img(prev_pool[best["idx"]][:, :3])
        ok_p = (
            (dep_p > 1.0)
            & (u_p >= 0) & (u_p < img_w) & (v_p >= 0) & (v_p < img_h)
        )
        if ok_p.sum() < 3:
            continue
        used_prev.add(best_k)
        cu_c, cv_c = u_c[ok_c].mean(), v_c[ok_c].mean()
        cu_p, cv_p = u_p[ok_p].mean(), v_p[ok_p].mean()
        d_lat = np.array(
            [
                (cu_c - cu_p) * lat_w / img_w,
                (cv_c - cv_p) * lat_h / img_h,
            ],
            np.float32,
        )
        matched.append(
            {
                "cell_mask": _latent_cells(u_c[ok_c], v_c[ok_c], img_w, img_h, lat_w, lat_h),
                "prev_cell_mask": _latent_cells(u_p[ok_p], v_p[ok_p], img_w, img_h, lat_w, lat_h),
                "d_lat": d_lat,
                "d_velo": d_velo.astype(np.float32),
                "count": int(ok_c.sum()),
                "count_prev": best["count"],
                "depth": float(np.median(dep_c[ok_c])),
                "bbox": [
                    float(u_c[ok_c].min()), float(v_c[ok_c].min()),
                    float(u_c[ok_c].max()), float(v_c[ok_c].max()),
                ],
                "centroid_cur": c_cur.astype(np.float32),
            }
        )
    stats["n_matched"] = len(matched)
    return matched, stats


class ObjectTracker:
    """Greedy centroid chaining across frames to keep persistent object ids
    (for later appearance-consistency evaluation; not used by transport)."""

    def __init__(self, gate=2.0):
        self.gate = gate
        self.tracks = {}  # oid -> centroid (cur velo frame)

    def update(self, matched):
        assign = []
        used = set()
        for obj in matched:
            c = obj["centroid_cur"]
            best, best_d = None, self.gate
            for oid, tc in self.tracks.items():
                if oid in used:
                    continue
                d = float(np.linalg.norm(tc - c))
                if d < best_d:
                    best, best_d = oid, d
            if best is None:
                best = max(self.tracks.keys(), default=-1) + 1
            self.tracks[best] = c
            used.add(best)
            assign.append(best)
        # forget stale tracks (crude: keep the 64 most recent)
        if len(self.tracks) > 64:
            for oid in list(self.tracks.keys())[: len(self.tracks) - 64]:
                del self.tracks[oid]
        return assign
