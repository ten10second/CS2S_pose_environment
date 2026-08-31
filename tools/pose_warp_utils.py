"""Geometry utilities for pose-guided latent warping on KITTI raw sequences.

Everything here is deterministic sensor-side geometry: no learned components.
- oxts IMU poses (mercator + yaw/pitch/roll)
- calibration chain imu <-> velo <-> cam (rectified)
- ground-plane estimation from a single velodyne scan
- ground-plane homography between two frames
- cross-frame LiDAR consistency (range-image residual) for dynamic detection
"""

from pathlib import Path

import numpy as np


def parse_calib_file(path):
    data = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            try:
                data[key.strip()] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


class SequenceGeometry:
    """Per-drive calibration + pose access. Paths must already be rebased to a
    readable KITTI_RAW root."""

    def __init__(self, calib_dir):
        calib_dir = Path(calib_dir)
        cam = parse_calib_file(calib_dir / "calib_cam_to_cam.txt")
        vel = parse_calib_file(calib_dir / "calib_velo_to_cam.txt")
        imu = parse_calib_file(calib_dir / "calib_imu_to_velo.txt")

        self.R0 = np.eye(4)
        self.R0[:3, :3] = cam["R_rect_00"].reshape(3, 3)
        self.K_rect = cam["P_rect_02"].reshape(3, 4)[:3, :3].copy()
        self.img_size = tuple(int(v) for v in cam["S_rect_02"][:2])  # (w, h)

        self.T_cam_velo = np.eye(4)
        self.T_cam_velo[:3, :3] = vel["R"].reshape(3, 3)
        self.T_cam_velo[:3, 3] = vel["T"]
        self.T_velo_cam = np.linalg.inv(self.T_cam_velo)

        self.T_imu_velo = np.eye(4)
        self.T_imu_velo[:3, :3] = imu["R"].reshape(3, 3)
        self.T_imu_velo[:3, 3] = imu["T"]
        self.T_velo_imu = np.linalg.inv(self.T_imu_velo)

    @staticmethod
    def _rot(roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def oxts_pose(self, oxts_path):
        vals = np.loadtxt(oxts_path)
        lat, lon, alt, roll, pitch, yaw = vals[:6]
        er = 6378137.0
        scale = np.cos(48.9 * np.pi / 180)  # KITTI Karlsruhe region
        mx = scale * lon * np.pi * er / 180
        my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
        T = np.eye(4)
        T[:3, :3] = self._rot(roll, pitch, yaw)
        T[:3, 3] = [mx, my, alt]
        return T

    def relative_cam_pose(self, oxts_prev, oxts_cur):
        """4x4 transform mapping rectified-cam coords of the previous frame to
        the current frame: p_rect_cur = T @ p_rect_prev."""
        T1 = self.oxts_pose(oxts_prev)
        T2 = self.oxts_pose(oxts_cur)
        T_c2c1 = (
            self.T_cam_velo @ self.T_imu_velo @ np.linalg.inv(T2) @ T1 @ self.T_velo_imu @ self.T_velo_cam
        )
        return self.R0 @ T_c2c1 @ np.linalg.inv(self.R0)

    def relative_velo_pose(self, oxts_prev, oxts_cur):
        """4x4 transform mapping prev-frame velodyne coords to current-frame
        velodyne coords: p_velo_cur = T @ p_velo_prev."""
        T1 = self.oxts_pose(oxts_prev)
        T2 = self.oxts_pose(oxts_cur)
        return self.T_imu_velo @ np.linalg.inv(T2) @ T1 @ self.T_velo_imu

    def project_velo_to_rect_img(self, points_velo):
        """Project velodyne points (already in the target frame's velo coords)
        to rectified image_02 pixel coords of that frame."""
        cam = (self.T_cam_velo[:3, :3] @ points_velo.T).T + self.T_cam_velo[:3, 3]
        rect = (self.R0[:3, :3] @ cam.T).T
        K = self.K_rect
        depth = rect[:, 2]
        u = K[0, 0] * rect[:, 0] / depth + K[0, 2]
        v = K[1, 1] * rect[:, 1] / depth + K[1, 2]
        return u, v, depth


def load_velodyne(path, max_r=60.0):
    p = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    keep = (np.linalg.norm(p[:, :2], axis=1) < max_r) & (p[:, 2] > -3.0)
    return p[keep]


def fit_ground_plane_velo(points, z_below=-1.4, r_min=3.0, r_max=20.0):
    """Fit ground plane ax + by - z + c = 0 (velo frame, z up) from road returns."""
    r = np.linalg.norm(points[:, :2], axis=1)
    sel = (points[:, 2] < z_below) & (r > r_min) & (r < r_max)
    p = points[sel]
    if len(p) < 200:
        return None, None
    A = np.stack([p[:, 0], p[:, 1], -np.ones(len(p))], axis=1)
    sol, *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
    a, b, c = sol
    # plane: z = a*x + b*y - c  <=>  (-a, -b, 1) . p + c = 0
    n = np.array([-a, -b, 1.0])
    length = np.linalg.norm(n)
    n /= length
    d = c / length
    return n, d


def sph_coords(p):
    r = np.linalg.norm(p, axis=1)
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360
    el = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
    return az, el, r


AZ_BIN = 0.2
EL_BIN = 0.4
AZ_MAX = 360.0
EL_MIN, EL_MAX = -26.0, 4.0
NA = int(AZ_MAX / AZ_BIN)
NE = int((EL_MAX - EL_MIN) / EL_BIN)


def range_image(p):
    az, el, r = sph_coords(p)
    ri = np.full((NE, NA), np.inf, np.float32)
    ia = np.clip((az / AZ_BIN).astype(int), 0, NA - 1)
    ie = np.clip(((el - EL_MIN) / EL_BIN).astype(int), 0, NE - 1)
    np.minimum.at(ri, (ie, ia), r.astype(np.float32))
    return ri


def consistency_status(points_prev_in_cur, points_cur, tol=0.3):
    """Classify previous-frame points (transformed into the current velo frame)
    against the current scan. Returns per-point codes: 0 unobserved, 1 static,
    2 dynamic (beam blocked closer = the surface there is gone)."""
    ri = range_image(points_cur)
    az, el, r = sph_coords(points_prev_in_cur)
    ia = np.clip((az / AZ_BIN).astype(int), 0, NA - 1)
    ie = np.clip(((el - EL_MIN) / EL_BIN).astype(int), 0, NE - 1)
    best_abs = np.full(len(r), np.inf, np.float32)
    best_res = np.full(len(r), np.inf, np.float32)
    for da in (-1, 0, 1):
        for de in (-1, 0, 1):
            v = ri[np.clip(ie + de, 0, NE - 1), np.clip(ia + da, 0, NA - 1)]
            res = v - r
            best_abs = np.minimum(best_abs, np.abs(res))
            best_res = np.minimum(best_res, res)
    status = np.zeros(len(r), np.uint8)
    obs = np.isfinite(best_res)
    status[obs & (best_abs <= tol)] = 1
    status[obs & (best_abs > tol) & (best_res < -tol)] = 2
    return status


def ground_homography(T_cur_prev, n_prev_rect, d_prev_rect, K):
    """Homography H mapping previous-frame rectified pixels to current-frame
    pixels via the ground plane (n, d) expressed in the previous rectified-cam
    frame. T_cur_prev maps prev rect-cam coords to cur rect-cam coords."""
    R = T_cur_prev[:3, :3]
    t = T_cur_prev[:3, 3]
    M = R - np.outer(t, n_prev_rect) / d_prev_rect
    return K @ M @ np.linalg.inv(K)


def depth_flow_splat(points_prev, points_prev_in_cur, geom, img_w, img_h, lat_h, lat_w, min_depth=1.0,
                     kernel=2, sigma_s=1.0, sigma_z=0.35, w_thresh=0.35):
    """Depth-gated soft splatting: per-cell backward flow from LiDAR anchors.

    Each anchor spreads its source position over a (2k+1)^2 neighbourhood of
    target cells with a Gaussian spatial weight, gated by depth similarity to
    the cell's reference depth (hard nearest-depth splat, pass 1). This keeps
    depth edges sharp (anchors behind an edge are gated out) while averaging
    away per-anchor sampling noise.

    Returns flow (lat_h, lat_w, 2), valid (lat_h, lat_w), cover (lat_h, lat_w).
    """
    u_p, v_p, d_p = geom.project_velo_to_rect_img(points_prev)
    u_c, v_c, d_c = geom.project_velo_to_rect_img(points_prev_in_cur)
    su_all = u_p * (lat_w / img_w)
    sv_all = v_p * (lat_h / img_h)
    tu_all = u_c * (lat_w / img_w)
    tv_all = v_c * (lat_h / img_h)
    ok = (
        (d_p > min_depth)
        & (d_c > min_depth)
        & (su_all >= 0) & (su_all < lat_w) & (sv_all >= 0) & (sv_all < lat_h)
        & (tu_all >= -kernel) & (tu_all < lat_w + kernel) & (tv_all >= -kernel) & (tv_all < lat_h + kernel)
    )
    flow = np.zeros((lat_h, lat_w, 2), np.float32)
    valid = np.zeros((lat_h, lat_w), np.float32)
    cover = np.zeros((lat_h, lat_w), np.float32)
    if not ok.any():
        return flow, valid, cover

    ti = np.floor(tv_all[ok]).astype(int)
    tj = np.floor(tu_all[ok]).astype(int)
    ti = np.clip(ti, 0, lat_h - 1)
    tj = np.clip(tj, 0, lat_w - 1)
    su = su_all[ok]
    sv = sv_all[ok]
    zc = d_c[ok]

    # pass 1: hard nearest-depth reference per cell
    order = np.argsort(zc)
    ti_s, tj_s = ti[order], tj[order]
    z_s = zc[order]
    cell = ti_s.astype(np.int64) * lat_w + tj_s
    _, first = np.unique(cell, return_index=True)
    z_ref = np.full(lat_h * lat_w, np.inf, np.float32)
    z_ref[cell[first]] = z_s[first]
    z_ref = z_ref.reshape(lat_h, lat_w)
    cover[ti_s[first], tj_s[first]] = 1.0

    # pass 2: depth-gated Gaussian accumulation over a (2k+1)^2 footprint
    acc_w = np.zeros((lat_h, lat_w), np.float32)
    acc_u = np.zeros((lat_h, lat_w), np.float32)
    acc_v = np.zeros((lat_h, lat_w), np.float32)
    for dy in range(-kernel, kernel + 1):
        for dx in range(-kernel, kernel + 1):
            yy = ti + dy
            xx = tj + dx
            inb = (yy >= 0) & (yy < lat_h) & (xx >= 0) & (xx < lat_w)
            if not inb.any():
                continue
            zr = z_ref[np.clip(yy, 0, lat_h - 1), np.clip(xx, 0, lat_w - 1)]
            has_ref = np.isfinite(zr) & inb
            if not has_ref.any():
                continue
            dz = (zc - zr) / (np.maximum(zr, 1.0) * sigma_z)
            wz = np.exp(-0.5 * dz * dz)
            ws = np.exp(-0.5 * (dx * dx + dy * dy) / (sigma_s * sigma_s))
            w = ws * wz * has_ref
            acc_w += np.bincount(yy[has_ref] * lat_w + xx[has_ref], weights=w[has_ref], minlength=lat_h * lat_w).reshape(lat_h, lat_w)
            acc_u += np.bincount(yy[has_ref] * lat_w + xx[has_ref], weights=(w * su)[has_ref], minlength=lat_h * lat_w).reshape(lat_h, lat_w)
            acc_v += np.bincount(yy[has_ref] * lat_w + xx[has_ref], weights=(w * sv)[has_ref], minlength=lat_h * lat_w).reshape(lat_h, lat_w)

    good = acc_w >= w_thresh
    flow[..., 0][good] = acc_u[good] / acc_w[good]
    flow[..., 1][good] = acc_v[good] / acc_w[good]
    valid[good] = 1.0
    return flow, valid, cover


def depth_flow_affine(points_prev, points_prev_in_cur, geom, img_w, img_h, lat_h, lat_w, min_depth=1.0,
                      kernel=2, sigma_s=1.0, sigma_z=0.35, w_thresh=0.35, fit_r=2, min_anchors=8):
    """Soft depth-gated splat (see depth_flow_splat) + per-cell local affine
    flow fit. The anchor flow is only exact AT the anchor; evaluating it at a
    different position within the cell biases the flow on slanted surfaces.
    A weighted local affine fit f(x) = a + J (x - c) over neighbouring anchors
    recovers sub-cell accuracy while degrading gracefully to the smooth
    averaged flow where anchors are sparse.

    Returns flow, valid, cover at latent resolution.
    """
    flow, valid, cover = depth_flow_splat(
        points_prev, points_prev_in_cur, geom, img_w, img_h, lat_h, lat_w,
        min_depth=min_depth, kernel=kernel, sigma_s=sigma_s, sigma_z=sigma_z, w_thresh=w_thresh,
    )
    u_p, v_p, d_p = geom.project_velo_to_rect_img(points_prev)
    u_c, v_c, d_c = geom.project_velo_to_rect_img(points_prev_in_cur)
    su = u_p * (lat_w / img_w)
    sv = v_p * (lat_h / img_h)
    tu = u_c * (lat_w / img_w)
    tv = v_c * (lat_h / img_h)
    ok = (
        (d_p > min_depth) & (d_c > min_depth)
        & (su >= 0) & (su < lat_w) & (sv >= 0) & (sv < lat_h)
        & (tu >= -fit_r - 1) & (tu < lat_w + fit_r + 1)
        & (tv >= -fit_r - 1) & (tv < lat_h + fit_r + 1)
    )
    tu_o, tv_o = tu[ok], tv[ok]
    su_o, sv_o = su[ok], sv[ok]
    z_o = d_c[ok]
    cell_of_anchor = np.clip(tv_o, 0, lat_h - 1).astype(int) * lat_w + np.clip(tu_o, 0, lat_w - 1).astype(int)
    buckets = {}
    for ci in np.unique(cell_of_anchor):
        buckets[ci] = np.where(cell_of_anchor == ci)[0]

    z_ref = np.full(lat_h * lat_w, np.inf, np.float32)
    hard = np.argsort(z_o)
    c_s = cell_of_anchor[hard]
    _, first = np.unique(c_s, return_index=True)
    z_ref[c_s[first]] = z_o[hard][first]

    sigma_d2 = 2.0 * fit_r * fit_r
    fitted = 0
    for cy in range(lat_h):
        for cx in range(lat_w):
            if valid[cy, cx] == 0:
                continue
            ccy = cy + 0.5
            ccx = cx + 0.5
            zc = z_ref[cy * lat_w + cx]
            if not np.isfinite(zc):
                continue
            idxs = []
            for dy in range(-fit_r, fit_r + 1):
                for dx in range(-fit_r, fit_r + 1):
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < lat_h and 0 <= xx < lat_w:
                        b = buckets.get(yy * lat_w + xx)
                        if b is not None:
                            idxs.append(b)
            if not idxs:
                continue
            idxs = np.concatenate(idxs)
            ddx = tu_o[idxs] - ccx
            ddy = tv_o[idxs] - ccy
            near = (np.abs(ddx) <= fit_r) & (np.abs(ddy) <= fit_r)
            if near.sum() < min_anchors:
                continue
            ddx, ddy = ddx[near], ddy[near]
            z_n = z_o[idxs][near]
            dz = (z_n - zc) / (max(zc, 1.0) * sigma_z)
            w = np.exp(-0.5 * (ddx * ddx + ddy * ddy) / sigma_d2) * np.exp(-0.5 * dz * dz)
            A = np.stack([np.ones_like(ddx), ddx, ddy], axis=1) * np.sqrt(w)[:, None]
            bx = su_o[idxs][near] * np.sqrt(w)
            by = sv_o[idxs][near] * np.sqrt(w)
            solx, *_ = np.linalg.lstsq(A, bx, rcond=None)
            soly, *_ = np.linalg.lstsq(A, by, rcond=None)
            flow[cy, cx, 0] = solx[0]
            flow[cy, cx, 1] = soly[0]
            fitted += 1
    return flow, valid, cover, fitted


def flow_to_grid(flow, valid, lat_h, lat_w, device):
    """Convert a per-cell backward flow to a grid_sample grid (invalid cells
    point at 0,0; they are zeroed out by the keep mask downstream)."""
    import torch

    gx = 2.0 * flow[..., 0] / lat_w - 1.0
    gy = 2.0 * flow[..., 1] / lat_h - 1.0
    grid = np.stack([gx, gy], axis=-1)
    grid[valid == 0] = 0.0
    grid_t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).to(device)
    return grid_t


def warp_image_with_flow(image, flow, valid, img_h, img_w, lat_h, lat_w, device):
    """Debug helper: warp a full-resolution image (1,3,H,W) with a latent-res
    flow, upsampling the flow to pixel resolution (each cell covers an
    (img_h/lat_h, img_w/lat_w) block)."""
    import torch

    sy, sx = img_h / lat_h, img_w / lat_w
    flow_up = np.repeat(np.repeat(flow, int(round(sy)), axis=0), int(round(sx)), axis=1)
    valid_up = np.repeat(np.repeat(valid, int(round(sy)), axis=0), int(round(sx)), axis=1)
    cur_h, cur_w = valid_up.shape
    src_px = flow_up * np.array([sx, sy], np.float32)  # latent->pixel scale
    gx = 2.0 * src_px[..., 0] / cur_w - 1.0
    gy = 2.0 * src_px[..., 1] / cur_h - 1.0
    grid = np.stack([gx, gy], axis=-1)
    grid[valid_up == 0] = 0.0
    grid_t = torch.from_numpy(grid.astype(np.float32)).unsqueeze(0).to(device)
    out = torch.nn.functional.grid_sample(image, grid_t, mode="bilinear", padding_mode="zeros", align_corners=False)
    keep = torch.from_numpy(valid_up.astype(np.float32)).reshape(1, 1, cur_h, cur_w).to(device)
    return out * keep, keep


def homography_flow_cells(H_img, img_w, img_h, lat_h, lat_w):
    """Per-cell backward flow of an image-space homography, at latent
    resolution (source latent coords for every cell)."""
    sx, sy = img_w / lat_w, img_h / lat_h
    D = np.diag([sx, sy, 1.0])
    H_lat = np.linalg.inv(D) @ H_img @ D
    Hi = np.linalg.inv(H_lat)
    ys, xs = np.meshgrid(
        np.arange(lat_h, dtype=np.float64) + 0.5,
        np.arange(lat_w, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    pts = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3)
    src = pts @ Hi.T
    src = src[:, :2] / src[:, 2:3]
    return src.reshape(lat_h, lat_w, 2)


def warp_latent_with_H(latent, H_img, img_w, img_h, lat_h, lat_w, device):
    """Sample `latent` (1,C,lat_h,lat_w) at positions given by the inverse of
    the image-space homography H_img. Returns warped latent and valid mask."""
    import torch

    sx, sy = img_w / lat_w, img_h / lat_h
    D = np.diag([sx, sy, 1.0])
    # latent px u_l corresponds to image px D @ u_l, so the latent-space
    # forward homography is D^{-1} @ H @ D (NOT D @ H @ D^{-1}).
    H_lat = np.linalg.inv(D) @ H_img @ D
    H_inv = np.linalg.inv(H_lat)

    ys, xs = np.meshgrid(
        np.arange(lat_h, dtype=np.float64) + 0.5,
        np.arange(lat_w, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    ones = np.ones_like(xs)
    pts = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3)  # (N,3) pixel centers
    src = pts @ H_inv.T
    src = src[:, :2] / src[:, 2:3]  # (N,2) source latent pixel centers

    valid = (
        (src[:, 0] >= 0) & (src[:, 0] < lat_w) & (src[:, 1] >= 0) & (src[:, 1] < lat_h)
    )
    # align_corners=False: pixel-center coords -> normalized [-1, 1]
    gx = 2.0 * src[:, 0] / lat_w - 1.0
    gy = 2.0 * src[:, 1] / lat_h - 1.0
    grid = np.stack([gx, gy], axis=-1).reshape(1, lat_h, lat_w, 2)
    grid_t = torch.from_numpy(grid.astype(np.float32)).to(device)
    warped = torch.nn.functional.grid_sample(
        latent, grid_t, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    mask = torch.from_numpy(valid.astype(np.float32).reshape(1, 1, lat_h, lat_w)).to(device)
    return warped, mask
