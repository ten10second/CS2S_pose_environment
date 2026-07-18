import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from dataloader import KITTI_utils as kitti_utils
from dataloader.kitti_raw_lidar_utils import (
    build_kitti_range_image,
    build_raw_lidar_point_samples,
    camera2_to_lidar_matrix,
    generate_lidar_condition,
    lidar_to_camera2_matrix,
    scaled_lidar_to_image_matrix,
    load_raw_calibration,
    parse_tracklet_xml,
    read_jsonl,
    scaled_camera_k,
)


class _FeatureMemmapCache:
    META_NAME = "memmap_meta.json"

    def __init__(self, root: Path, kind: str, feature_shape: Tuple[int, ...], mask_shape: Tuple[int, ...]):
        meta_path = root / self.META_NAME
        self.enabled = meta_path.is_file()
        self.root = root
        self.kind = kind
        self.feature_shape = tuple(feature_shape)
        self.mask_shape = tuple(mask_shape)
        self._features = None
        self._masks = None
        self.index = {}
        if not self.enabled:
            return

        meta = json.loads(meta_path.read_text())
        if meta.get("format") != "kitti_feature_memmap_v1" or meta.get("kind") != kind:
            raise ValueError(f"Invalid {kind} memmap metadata: {meta_path}")
        if tuple(meta["feature_shape"]) != self.feature_shape:
            raise ValueError(
                f"{kind} memmap feature shape {tuple(meta['feature_shape'])} != expected {self.feature_shape}"
            )
        if tuple(meta["mask_shape"]) != self.mask_shape:
            raise ValueError(
                f"{kind} memmap mask shape {tuple(meta['mask_shape'])} != expected {self.mask_shape}"
            )
        self.features_path = root / meta["features_file"]
        self.masks_path = root / meta["masks_file"]
        self.index = {str(key): int(value) for key, value in meta["index"].items()}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        state["_masks"] = None
        return state

    def _open(self):
        if self._features is None:
            self._features = np.load(self.features_path, mmap_mode="r", allow_pickle=False)
            self._masks = np.load(self.masks_path, mmap_mode="r", allow_pickle=False)

    def get(self, safe_sample_id: str):
        row = self.index.get(safe_sample_id)
        if row is None:
            return None
        self._open()
        return np.array(self._features[row], copy=True), np.array(self._masks[row], copy=True)


class SatLidarRawDataset(Dataset):
    """KITTI raw satellite + image_02 + Velodyne dynamic condition dataset."""

    def __init__(
        self,
        manifest: str,
        condition_mode: str = "raw_lidar",
        image_height: int = 128,
        image_width: int = 512,
        sat_size: int = 256,
        max_depth: float = 80.0,
        max_dynamic_boxes: int = 32,
        align_satellite_to_camera: bool = True,
        include_range_image: bool = True,
        range_height: int = 64,
        range_width: int = 1024,
        include_raw_lidar_points: bool = False,
        raw_lidar_point_count: int = 8192,
        foreground_mask_root: str = "",
        foreground_mask_suffix: str = "_foreground.png",
        lidar_point_feature_cache_root: str = "",
        lidar_point_feature_cache_suffix: str = ".npz",
        lidar_point_feature_dim: int = 0,
        image_semantic_cache_root: str = "",
        image_semantic_cache_suffix: str = ".npz",
        image_semantic_feature_key: str = "image_semantic_feat",
        image_semantic_feature_dim: int = 0,
        image_semantic_height: int = 8,
        image_semantic_width: int = 32,
        include_tracklets: bool = True,
    ):
        self.manifest = manifest
        self.records = read_jsonl(manifest)
        self.condition_mode = condition_mode
        self.image_size: Tuple[int, int] = (image_height, image_width)
        self.sat_size = sat_size
        self.max_depth = max_depth
        self.max_dynamic_boxes = max_dynamic_boxes
        self.align_satellite_to_camera = align_satellite_to_camera
        self.include_range_image = include_range_image
        self.range_height = range_height
        self.range_width = range_width
        self.include_raw_lidar_points = bool(include_raw_lidar_points)
        self.raw_lidar_point_count = int(raw_lidar_point_count)
        self.foreground_mask_root = Path(foreground_mask_root) if foreground_mask_root else None
        self.foreground_mask_suffix = foreground_mask_suffix
        self.lidar_point_feature_cache_root = (
            Path(lidar_point_feature_cache_root) if lidar_point_feature_cache_root else None
        )
        self.lidar_point_feature_cache_suffix = lidar_point_feature_cache_suffix
        self.lidar_point_feature_dim = int(lidar_point_feature_dim)
        self.image_semantic_cache_root = Path(image_semantic_cache_root) if image_semantic_cache_root else None
        self.image_semantic_cache_suffix = image_semantic_cache_suffix
        self.image_semantic_feature_key = image_semantic_feature_key
        self.image_semantic_feature_dim = int(image_semantic_feature_dim)
        self.image_semantic_size = (int(image_semantic_height), int(image_semantic_width))
        self._lidar_feature_memmap = (
            _FeatureMemmapCache(
                self.lidar_point_feature_cache_root,
                kind="point",
                feature_shape=(self.raw_lidar_point_count, self.lidar_point_feature_dim),
                mask_shape=(self.raw_lidar_point_count,),
            )
            if self.lidar_point_feature_cache_root is not None and self.lidar_point_feature_dim > 0
            else None
        )
        self._image_semantic_memmap = (
            _FeatureMemmapCache(
                self.image_semantic_cache_root,
                kind="image",
                feature_shape=(self.image_semantic_feature_dim, *self.image_semantic_size),
                mask_shape=(1, *self.image_semantic_size),
            )
            if self.image_semantic_cache_root is not None and self.image_semantic_feature_dim > 0
            else None
        )
        self.include_tracklets = bool(include_tracklets)
        self.meter_per_pixel = kitti_utils.get_meter_per_pixel()

        self.sat_to_tensor = transforms.ToTensor()
        self.grd_transform = transforms.Compose(
            [
                transforms.Resize((image_height, image_width)),
                transforms.ToTensor(),
            ]
        )

        self._calib_cache: Dict[str, dict] = {}
        self._tracklet_cache: Dict[str, dict] = {}

    def _foreground_mask_path(self, sample_id: str) -> Path:
        safe_id = sample_id.replace("/", "__")
        return self.foreground_mask_root / f"{safe_id}{self.foreground_mask_suffix}"

    @staticmethod
    def _safe_cache_id(sample_id: str) -> str:
        return sample_id.replace("/", "__")

    def _cache_path(self, root: Path, sample_id: str, suffix: str) -> Path:
        return root / f"{self._safe_cache_id(sample_id)}{suffix}"

    def _foreground_mask(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros((1, *self.image_size), dtype=torch.float32)
        available = torch.tensor([0.0], dtype=torch.float32)
        if self.foreground_mask_root is None:
            return mask, available
        mask_path = self._foreground_mask_path(sample_id)
        if not mask_path.is_file():
            return mask, available
        with Image.open(mask_path) as mask_img:
            mask_pil = mask_img.convert("L").resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
        mask = (TF.to_tensor(mask_pil) > 0.5).float()
        available = torch.tensor([1.0], dtype=torch.float32)
        return mask, available

    @staticmethod
    def _npz_first_array(payload, keys):
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    def _lidar_point_feature_cache(self, sample_id: str) -> Dict[str, torch.Tensor]:
        if self.lidar_point_feature_cache_root is None or self.lidar_point_feature_dim <= 0:
            return {}
        if self._lidar_feature_memmap is not None and self._lidar_feature_memmap.enabled:
            cached = self._lidar_feature_memmap.get(self._safe_cache_id(sample_id))
            if cached is not None:
                features, mask = cached
                return {
                    "lidar_point_features": torch.from_numpy(features),
                    "lidar_point_features_mask": torch.from_numpy(mask),
                    "lidar_point_features_available": torch.ones(1, dtype=torch.float32),
                }
        features = np.zeros((self.raw_lidar_point_count, self.lidar_point_feature_dim), dtype=np.float32)
        mask = np.zeros((self.raw_lidar_point_count,), dtype=np.float32)
        available = np.array([0.0], dtype=np.float32)
        cache_path = self._cache_path(
            self.lidar_point_feature_cache_root,
            sample_id,
            self.lidar_point_feature_cache_suffix,
        )
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as payload:
                cached = self._npz_first_array(
                    payload,
                    (
                        "lidar_point_features",
                        "utonia_feat",
                        "point_features",
                        "features",
                        "feat",
                    ),
                )
                cached_mask = self._npz_first_array(
                    payload,
                    (
                        "lidar_point_features_mask",
                        "point_feature_mask",
                        "point_mask",
                        "mask",
                    ),
                )
                if cached is not None:
                    cached = np.asarray(cached, dtype=np.float32)
                    if cached.ndim == 1:
                        cached = cached[:, None]
                    elif cached.ndim > 2:
                        cached = cached.reshape(cached.shape[0], -1)
                    point_count = min(features.shape[0], cached.shape[0])
                    channel_count = min(features.shape[1], cached.shape[1])
                    features[:point_count, :channel_count] = cached[:point_count, :channel_count]
                    if cached_mask is not None:
                        cached_mask = np.asarray(cached_mask, dtype=np.float32).reshape(-1)
                        mask[: min(point_count, cached_mask.shape[0])] = cached_mask[
                            : min(point_count, cached_mask.shape[0])
                        ]
                    else:
                        mask[:point_count] = 1.0
                    available[...] = 1.0
        return {
            "lidar_point_features": torch.from_numpy(features).float(),
            "lidar_point_features_mask": torch.from_numpy(mask).float(),
            "lidar_point_features_available": torch.from_numpy(available).float(),
        }

    def _image_semantic_cache(self, sample_id: str) -> Dict[str, torch.Tensor]:
        if self.image_semantic_cache_root is None or self.image_semantic_feature_dim <= 0:
            return {}
        if self._image_semantic_memmap is not None and self._image_semantic_memmap.enabled:
            cached = self._image_semantic_memmap.get(self._safe_cache_id(sample_id))
            if cached is not None:
                features, mask = cached
                return {
                    "image_semantic_feat": torch.from_numpy(features),
                    "image_semantic_mask": torch.from_numpy(mask),
                    "image_semantic_available": torch.ones(1, dtype=torch.float32),
                }
        out_h, out_w = self.image_semantic_size
        features = np.zeros((self.image_semantic_feature_dim, out_h, out_w), dtype=np.float32)
        mask = np.zeros((1, out_h, out_w), dtype=np.float32)
        available = np.array([0.0], dtype=np.float32)
        cache_path = self._cache_path(
            self.image_semantic_cache_root,
            sample_id,
            self.image_semantic_cache_suffix,
        )
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as payload:
                cached = self._npz_first_array(
                    payload,
                    (
                        self.image_semantic_feature_key,
                        "image_semantic_feat",
                        "dino_feat",
                        "clip_feat",
                        "features",
                        "feat",
                    ),
                )
                cached_mask = self._npz_first_array(
                    payload,
                    (
                        "image_semantic_mask",
                        "semantic_mask",
                        "valid_mask",
                        "mask",
                    ),
                )
                if cached is not None:
                    cached = np.asarray(cached, dtype=np.float32)
                    if cached.ndim == 2:
                        if cached.shape[0] == out_h * out_w:
                            cached = cached.reshape(out_h, out_w, cached.shape[1]).transpose(2, 0, 1)
                        elif cached.shape[1] == out_h * out_w:
                            cached = cached.reshape(cached.shape[0], out_h, out_w)
                        else:
                            cached = cached.reshape(cached.shape[0], -1, 1)
                    elif cached.ndim == 3 and (
                        cached.shape[-1] == self.image_semantic_feature_dim
                        or cached.shape[:2] == (out_h, out_w)
                    ):
                        cached = cached.transpose(2, 0, 1)
                    elif cached.ndim > 3:
                        cached = cached.reshape(cached.shape[0], cached.shape[-2], cached.shape[-1])
                    channel_count = min(features.shape[0], cached.shape[0])
                    height_count = min(features.shape[1], cached.shape[1])
                    width_count = min(features.shape[2], cached.shape[2])
                    features[:channel_count, :height_count, :width_count] = cached[
                        :channel_count, :height_count, :width_count
                    ]
                    if cached_mask is not None:
                        cached_mask = np.asarray(cached_mask, dtype=np.float32)
                        if cached_mask.ndim == 3:
                            cached_mask = cached_mask[0] if cached_mask.shape[0] == 1 else cached_mask[..., 0]
                        height_mask = min(out_h, cached_mask.shape[0])
                        width_mask = min(out_w, cached_mask.shape[1])
                        mask[:, :height_mask, :width_mask] = cached_mask[:height_mask, :width_mask]
                    else:
                        mask[:, :height_count, :width_count] = 1.0
                    available[...] = 1.0
        return {
            "image_semantic_feat": torch.from_numpy(features).float(),
            "image_semantic_mask": torch.from_numpy(mask).float(),
            "image_semantic_available": torch.from_numpy(available).float(),
        }

    def __len__(self):
        return len(self.records)

    def _calib(self, calib_dir: str):
        if calib_dir not in self._calib_cache:
            self._calib_cache[calib_dir] = load_raw_calibration(calib_dir)
        return self._calib_cache[calib_dir]

    def _boxes_by_frame(self, xml_path: str):
        if not self.include_tracklets:
            return {}
        if not xml_path:
            return {}
        if xml_path not in self._tracklet_cache:
            self._tracklet_cache[xml_path] = parse_tracklet_xml(xml_path)
        return self._tracklet_cache[xml_path]

    @staticmethod
    def _read_heading(oxts_path: str) -> float:
        with Path(oxts_path).open("r") as handle:
            values = handle.readline().strip().split()
        if len(values) < 6:
            raise ValueError(f"Malformed OXTS packet: {oxts_path}")
        return float(values[5])

    @staticmethod
    def _camera_forward_right(calib: Dict[str, object]) -> Tuple[float, float]:
        if "cam2_imu_forward_right" in calib:
            offset = calib["cam2_imu_forward_right"]
            return float(offset[0]), float(offset[1])
        return float(kitti_utils.CameraGPS_shift_left[0]), float(kitti_utils.CameraGPS_shift_left[1])

    def _satellite_image(self, sat_img: Image.Image, oxts_path: str, calib: Dict[str, object]) -> Image.Image:
        sat_rgb = sat_img.convert("RGB")
        if not self.align_satellite_to_camera:
            return sat_rgb.resize((self.sat_size, self.sat_size), Image.BILINEAR)

        heading = self._read_heading(oxts_path)
        camera_forward, camera_right = self._camera_forward_right(calib)
        sat_map = sat_rgb.resize(
            (kitti_utils.SatMap_process_sidelength, kitti_utils.SatMap_process_sidelength),
            Image.BILINEAR,
        )
        sat_rot = sat_map.rotate(-heading / math.pi * 180.0, resample=Image.BILINEAR)
        sat_align_cam = sat_rot.transform(
            sat_rot.size,
            Image.AFFINE,
            (
                1,
                0,
                camera_forward / self.meter_per_pixel,
                0,
                1,
                camera_right / self.meter_per_pixel,
            ),
            resample=Image.BILINEAR,
        )
        sat_crop = TF.center_crop(sat_align_cam, kitti_utils.SatMap_end_sidelength)
        if self.sat_size != kitti_utils.SatMap_end_sidelength:
            sat_crop = sat_crop.resize((self.sat_size, self.sat_size), Image.BILINEAR)
        return sat_crop

    def __getitem__(self, idx):
        record = self.records[idx]
        calib = self._calib(record["calib_dir"])
        boxes_by_frame = self._boxes_by_frame(record.get("tracklet_xml_path", ""))
        frame_index = int(record["frame_index"])
        boxes = boxes_by_frame.get(frame_index, [])

        with Image.open(record["satellite_path"]) as sat_img:
            sat_map = self.sat_to_tensor(self._satellite_image(sat_img, record["oxts_path"], calib))
        with Image.open(record["image_02_path"]) as grd_img:
            grd_left_img = self.grd_transform(grd_img.convert("RGB"))
        foreground_mask, foreground_mask_available = self._foreground_mask(record["sample_id"])

        lidar = generate_lidar_condition(
            record["velodyne_path"],
            boxes,
            calib,
            output_size=self.image_size,
            mode=self.condition_mode,
            max_depth=self.max_depth,
            max_dynamic_boxes=self.max_dynamic_boxes,
        )
        range_lidar = None
        if self.include_range_image:
            range_lidar = build_kitti_range_image(
                record["velodyne_path"],
                height=self.range_height,
                width=self.range_width,
                max_range=self.max_depth,
            )
        raw_lidar_points = None
        if self.include_raw_lidar_points:
            raw_lidar_points = build_raw_lidar_point_samples(
                record["velodyne_path"],
                calib,
                output_size=self.image_size,
                max_depth=self.max_depth,
                max_points=self.raw_lidar_point_count,
            )

        sample = {
            "sat_map_gt": sat_map,
            "sat_map": sat_map,
            "left_camera_k": torch.from_numpy(scaled_camera_k(calib, self.image_size)),
            "lidar_to_image": torch.from_numpy(scaled_lidar_to_image_matrix(calib, self.image_size)),
            "lidar_to_camera": torch.from_numpy(lidar_to_camera2_matrix(calib)),
            "camera_to_lidar": torch.from_numpy(camera2_to_lidar_matrix(calib)),
            "grd_left_imgs": grd_left_img,
            "gt_shift_x": torch.tensor([0.0], dtype=torch.float32),
            "gt_shift_y": torch.tensor([0.0], dtype=torch.float32),
            "theta": torch.tensor([0.0], dtype=torch.float32),
            "camera_imu_forward_right": torch.tensor(self._camera_forward_right(calib), dtype=torch.float32),
            "file_name": f"{record['sample_id']}.png",
            "sample_id": record["sample_id"],
            "lidar_cond": torch.from_numpy(lidar["lidar_cond"]).float(),
            "dynamic_mask": torch.from_numpy(lidar["dynamic_mask"]).float(),
            "dynamic_boxes": torch.from_numpy(lidar["dynamic_boxes"]).float(),
            "dynamic_box_valid": torch.from_numpy(lidar["dynamic_box_valid"]).float(),
            "foreground_mask": foreground_mask,
            "foreground_mask_available": foreground_mask_available,
            "dynamic_class_hist": torch.from_numpy(lidar["dynamic_class_hist"]).float(),
            "lidar_valid_mask": torch.from_numpy(lidar["lidar_valid_mask"]).float(),
            "num_dynamic_boxes": torch.as_tensor(lidar["num_dynamic_boxes"], dtype=torch.long),
            "num_projected_lidar_points": torch.as_tensor(lidar["num_projected_lidar_points"], dtype=torch.long),
            "num_projected_dynamic_points": torch.as_tensor(lidar["num_projected_dynamic_points"], dtype=torch.long),
        }
        if range_lidar is not None:
            sample.update(
                {
                    "range_img": torch.from_numpy(range_lidar["range_img"]).float(),
                    "range_mask": torch.from_numpy(range_lidar["range_mask"]).float(),
                    "range_depth": torch.from_numpy(range_lidar["range_depth"]).float(),
                    "range_intensity": torch.from_numpy(range_lidar["range_intensity"]).float(),
                    "num_range_points": torch.as_tensor(range_lidar["num_range_points"], dtype=torch.long),
                }
            )
        if raw_lidar_points is not None:
            sample.update(
                {
                    "lidar_points": torch.from_numpy(raw_lidar_points["lidar_points"]).float(),
                    "lidar_points_mask": torch.from_numpy(raw_lidar_points["lidar_points_mask"]).float(),
                    "num_raw_lidar_points": torch.as_tensor(raw_lidar_points["num_raw_lidar_points"], dtype=torch.long),
                    "num_raw_lidar_projected_points": torch.as_tensor(
                        raw_lidar_points["num_raw_lidar_projected_points"], dtype=torch.long
                    ),
                }
            )
            sample.update(self._lidar_point_feature_cache(record["sample_id"]))
        sample.update(self._image_semantic_cache(record["sample_id"]))
        return sample


class SatLidarRawDatasetTest(SatLidarRawDataset):
    pass
