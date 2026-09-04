"""Temporal-evidence plumbing: pose-transport payload + post-hoc injection.

TemporalTransportBuilder turns a (prev_frame, cur_frame) manifest pair into a
transport payload — per-resolution backward grids from the ground homography
plus validity masks excluding LiDAR-inconsistent cells — consumed by the
temporal evidence stream in RayPosteriorEvidenceFusion.

enable_temporal_evidence(model) injects zero-initialised temporal gates into
every ray-posterior fusion module and returns (hub, modules) without touching
any config or construction code, so the frozen single-frame checkpoint keeps
its exact behaviour until the gates are trained.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

TOOLS_DIR = Path(__file__).resolve().parent
for p in (str(TOOLS_DIR), str(TOOLS_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pose_warp_utils as pwu  # noqa: E402


def rebase_kitti_path(path, kitti_root):
    path = str(path)
    marker = "KITTI_RAW/"
    idx = path.find(marker)
    if idx >= 0 and kitti_root:
        return str(Path(kitti_root) / path[idx + len(marker):])
    return path


def resolve_calib_dir(calib_dir):
    """External-drive manifests nest a date folder inside the calib dir that
    the local copy flattens; find calib_cam_to_cam.txt here or one level up."""
    p = Path(calib_dir)
    if (p / "calib_cam_to_cam.txt").exists():
        return str(p)
    parent = p.parent
    if (parent / "calib_cam_to_cam.txt").exists():
        return str(parent)
    raise FileNotFoundError(f"calib_cam_to_cam.txt not found under {p} or {parent}")


class TemporalTransportBuilder:
    """Builds the transport payload for one (prev -> cur) frame pair.

    The payload's `transport(hw, device, dtype)` returns (grid, validity):
    grid is the normalized grid_sample grid taking current-frame latent cells
    to their source positions in the previous frame via the ground-plane
    homography; validity zeroes dynamic (LiDAR-inconsistent) and out-of-view
    cells so only reliable static history is transported.
    """

    def __init__(self, prev_row, cur_row, kitti_root=None, max_r=60.0):
        self.geom = pwu.SequenceGeometry(resolve_calib_dir(rebase_kitti_path(cur_row["calib_dir"], kitti_root)))
        self.img_w, self.img_h = self.geom.img_size
        p1 = pwu.load_velodyne(rebase_kitti_path(prev_row["velodyne_path"], kitti_root), max_r=max_r)
        p2 = pwu.load_velodyne(rebase_kitti_path(cur_row["velodyne_path"], kitti_root), max_r=max_r)
        self.T_v = self.geom.relative_velo_pose(
            rebase_kitti_path(prev_row["oxts_path"], kitti_root),
            rebase_kitti_path(cur_row["oxts_path"], kitti_root),
        )
        q1 = (self.T_v[:3, :3] @ p1.T).T + self.T_v[:3, 3]
        status = pwu.consistency_status(q1, p2)
        n_v, d_v = pwu.fit_ground_plane_velo(p1)
        if n_v is None:
            self.valid = False
            return
        self.valid = True
        A = self.geom.R0[:3, :3] @ self.geom.T_cam_velo[:3, :3]
        b = self.geom.R0[:3, :3] @ self.geom.T_cam_velo[:3, 3]
        n_r = A @ n_v
        d_r = d_v - float(n_r @ b)
        T_c = self.geom.relative_cam_pose(
            rebase_kitti_path(prev_row["oxts_path"], kitti_root),
            rebase_kitti_path(cur_row["oxts_path"], kitti_root),
        )
        self.H = pwu.ground_homography(T_c, n_r, d_r, self.geom.K_rect)
        # dynamic points: previous returns whose surface is gone in the
        # current scan (moved objects / disocclusions), in current velo coords
        dyn_pts = q1[status == 2]
        self.dyn_pts = dyn_pts
        self._cache = {}

    def _dynamic_mask(self, h, w):
        mask = np.zeros((h, w), np.float32)
        if len(self.dyn_pts):
            u, v, depth = self.geom.project_velo_to_rect_img(self.dyn_pts)
            lu = np.clip((u * w / self.img_w).astype(int), 0, w - 1)
            lv = np.clip((v * h / self.img_h).astype(int), 0, h - 1)
            ok = (depth > 1.0) & (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
            mask[lv[ok], lu[ok]] = 1.0
            if ok.any():
                mask = ndi.binary_dilation(mask, iterations=1).astype(np.float32)
        return mask

    def transport(self, hw, device, dtype):
        key = (int(hw[0]), int(hw[1]))
        if key not in self._cache:
            h, w = key
            flow = pwu.homography_flow_cells(self.H, self.img_w, self.img_h, h, w)
            inb = (flow[..., 0] >= 0) & (flow[..., 0] < w) & (flow[..., 1] >= 0) & (flow[..., 1] < h)
            valid = (inb.astype(np.float32) * (1.0 - self._dynamic_mask(h, w))).clip(0.0, 1.0)
            gx = 2.0 * flow[..., 0] / w - 1.0
            gy = 2.0 * flow[..., 1] / h - 1.0
            grid = np.stack([gx, gy], axis=-1).astype(np.float32)[None]
            self._cache[key] = (
                torch.from_numpy(grid).to(device=device, dtype=torch.float32),
                torch.from_numpy(valid[None, None]).to(device=device, dtype=dtype),
            )
        grid, valid = self._cache[key]
        return grid, valid

    def payload(self, strength=1.0):
        if not self.valid:
            return None
        return {"transport": self.transport, "strength": float(strength)}


def enable_temporal_evidence(model, gate_bias=-6.0):
    """Inject temporal gates post-hoc into every ray-posterior fusion module.

    Freezing is left to the caller. Returns (hub, blocks) where blocks are the
    BasicTransformerBlock modules wired to the hub (the gate parameters live
    in block.ray_posterior_fusion.temporal_gate).
    """
    from ldm.modules.KITTI_attention import TemporalEvidenceHub

    hub = TemporalEvidenceHub()
    blocks = []
    for module in model.modules():
        if getattr(module, "ray_fusion_mode", None) == "ray_posterior" and getattr(module, "ray_posterior_fusion", None) is not None:
            module.ray_posterior_fusion.enable_temporal(gate_bias=gate_bias)
            module.temporal_hub = hub
            blocks.append(module)
    if not blocks:
        raise RuntimeError("no ray_posterior fusion modules found; is this the ray-posterior checkpoint?")
    return hub, blocks


def temporal_gate_parameters(blocks):
    params = []
    for block in blocks:
        params.extend(block.ray_posterior_fusion.temporal_gate.parameters())
    return params


def freeze_temporal_snapshot(blocks):
    """Snapshot each block's current fused posterior as the frozen temporal
    reference for the NEXT frame's entire DDIM trajectory. Without this, the
    later denoising steps of frame t would consume frame t's own intermediate
    state instead of the previous frame's posterior."""
    for block in blocks:
        block.frozen_fused_delta = (
            None if block.last_fused_delta is None else block.last_fused_delta.clone()
        )
