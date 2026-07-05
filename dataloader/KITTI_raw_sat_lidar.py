import math
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from dataloader import KITTI_utils as kitti_utils
from dataloader.kitti_raw_lidar_utils import (
    build_kitti_range_image,
    camera2_to_lidar_matrix,
    generate_lidar_condition,
    lidar_to_camera2_matrix,
    scaled_lidar_to_image_matrix,
    load_raw_calibration,
    parse_tracklet_xml,
    read_jsonl,
    scaled_camera_k,
)


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
        foreground_mask_root: str = "",
        foreground_mask_suffix: str = "_foreground.png",
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
        self.foreground_mask_root = Path(foreground_mask_root) if foreground_mask_root else None
        self.foreground_mask_suffix = foreground_mask_suffix
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
        return sample


class SatLidarRawDatasetTest(SatLidarRawDataset):
    pass
