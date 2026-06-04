from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from dataloader.kitti_raw_lidar_utils import read_jsonl


def normal_label_path(cache_root: str, sample_id: str) -> Path:
    return Path(cache_root) / f"{sample_id}.npz"


class KittiLidarNormalDataset(Dataset):
    """KITTI raw LiDAR points with per-point normal pseudo-label cache."""

    def __init__(
        self,
        manifest: str,
        label_root: str,
        min_points: int = 64,
        max_points_per_sample: int = 0,
        min_label_weight: float = 1e-4,
    ):
        self.manifest = manifest
        self.label_root = label_root
        self.min_points = int(min_points)
        self.max_points_per_sample = int(max_points_per_sample)
        self.min_label_weight = float(min_label_weight)
        records = read_jsonl(manifest)
        self.records: List[dict] = []
        for record in records:
            path = normal_label_path(label_root, record["sample_id"])
            if not path.exists():
                continue
            try:
                with np.load(path) as payload:
                    count = int(payload["points_rect"].shape[0])
            except Exception:
                continue
            if count >= self.min_points:
                cached = dict(record)
                cached["normal_label_path"] = str(path)
                cached["normal_label_count"] = count
                self.records.append(cached)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with np.load(record["normal_label_path"]) as payload:
            points_rect = payload["points_rect"].astype(np.float32)
            intensity = payload["intensity"].astype(np.float32)
            normals = payload["normal_rect"].astype(np.float32)
            weights = payload["label_weight"].astype(np.float32)
            uv_orig = payload["uv_orig"].astype(np.float32)
            depth_rect = payload["depth_rect"].astype(np.float32)
            image_shape = payload["image_shape"].astype(np.int64)

        valid = np.isfinite(points_rect).all(axis=1)
        valid &= np.isfinite(normals).all(axis=1)
        valid &= np.isfinite(weights)
        valid &= weights > self.min_label_weight
        if valid.any():
            points_rect = points_rect[valid]
            intensity = intensity[valid]
            normals = normals[valid]
            weights = weights[valid]
            uv_orig = uv_orig[valid]
            depth_rect = depth_rect[valid]

        if self.max_points_per_sample > 0 and points_rect.shape[0] > self.max_points_per_sample:
            choice = np.random.choice(points_rect.shape[0], self.max_points_per_sample, replace=False)
            points_rect = points_rect[choice]
            intensity = intensity[choice]
            normals = normals[choice]
            weights = weights[choice]
            uv_orig = uv_orig[choice]
            depth_rect = depth_rect[choice]

        normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(normal_norm, 1e-6)

        return {
            "points_rect": torch.from_numpy(points_rect).float(),
            "intensity": torch.from_numpy(intensity[:, None]).float(),
            "normal_rect": torch.from_numpy(normals).float(),
            "label_weight": torch.from_numpy(weights).float(),
            "uv_orig": torch.from_numpy(uv_orig).float(),
            "depth_rect": torch.from_numpy(depth_rect).float(),
            "image_shape": torch.from_numpy(image_shape).long(),
            "sample_id": record["sample_id"],
            "image_02_path": record["image_02_path"],
            "normal_label_path": record["normal_label_path"],
        }


def collate_lidar_normal(batch):
    points = []
    intensity = []
    normals = []
    weights = []
    uv = []
    depth = []
    batch_index = []
    sample_ids = []
    image_shapes = []
    image_paths = []
    offsets = [0]
    for idx, sample in enumerate(batch):
        count = sample["points_rect"].shape[0]
        points.append(sample["points_rect"])
        intensity.append(sample["intensity"])
        normals.append(sample["normal_rect"])
        weights.append(sample["label_weight"])
        uv.append(sample["uv_orig"])
        depth.append(sample["depth_rect"])
        batch_index.append(torch.full((count,), idx, dtype=torch.long))
        sample_ids.append(sample["sample_id"])
        image_shapes.append(sample["image_shape"])
        image_paths.append(sample["image_02_path"])
        offsets.append(offsets[-1] + count)

    return {
        "points_rect": torch.cat(points, dim=0),
        "intensity": torch.cat(intensity, dim=0),
        "normal_rect": torch.cat(normals, dim=0),
        "label_weight": torch.cat(weights, dim=0),
        "uv_orig": torch.cat(uv, dim=0),
        "depth_rect": torch.cat(depth, dim=0),
        "batch_index": torch.cat(batch_index, dim=0),
        "sample_offsets": torch.tensor(offsets, dtype=torch.long),
        "image_shape": torch.stack(image_shapes, dim=0),
        "sample_id": sample_ids,
        "image_02_path": image_paths,
    }
