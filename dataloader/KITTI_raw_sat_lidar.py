from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloader.kitti_raw_lidar_utils import (
    generate_lidar_condition,
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
        condition_mode: str = "dynamic_full",
        image_height: int = 128,
        image_width: int = 512,
        sat_size: int = 256,
        max_depth: float = 80.0,
        max_dynamic_boxes: int = 32,
    ):
        self.manifest = manifest
        self.records = read_jsonl(manifest)
        self.condition_mode = condition_mode
        self.image_size: Tuple[int, int] = (image_height, image_width)
        self.max_depth = max_depth
        self.max_dynamic_boxes = max_dynamic_boxes

        self.sat_transform = transforms.Compose(
            [
                transforms.Resize((sat_size, sat_size)),
                transforms.ToTensor(),
            ]
        )
        self.grd_transform = transforms.Compose(
            [
                transforms.Resize((image_height, image_width)),
                transforms.ToTensor(),
            ]
        )

        self._calib_cache: Dict[str, dict] = {}
        self._tracklet_cache: Dict[str, dict] = {}

    def __len__(self):
        return len(self.records)

    def _calib(self, calib_dir: str):
        if calib_dir not in self._calib_cache:
            self._calib_cache[calib_dir] = load_raw_calibration(calib_dir)
        return self._calib_cache[calib_dir]

    def _boxes_by_frame(self, xml_path: str):
        if not xml_path:
            return {}
        if xml_path not in self._tracklet_cache:
            self._tracklet_cache[xml_path] = parse_tracklet_xml(xml_path)
        return self._tracklet_cache[xml_path]

    def __getitem__(self, idx):
        record = self.records[idx]
        calib = self._calib(record["calib_dir"])
        boxes_by_frame = self._boxes_by_frame(record.get("tracklet_xml_path", ""))
        frame_index = int(record["frame_index"])
        boxes = boxes_by_frame.get(frame_index, [])

        with Image.open(record["satellite_path"]) as sat_img:
            sat_map = self.sat_transform(sat_img.convert("RGB"))
        with Image.open(record["image_02_path"]) as grd_img:
            grd_left_img = self.grd_transform(grd_img.convert("RGB"))

        lidar = generate_lidar_condition(
            record["velodyne_path"],
            boxes,
            calib,
            output_size=self.image_size,
            mode=self.condition_mode,
            max_depth=self.max_depth,
            max_dynamic_boxes=self.max_dynamic_boxes,
        )

        sample = {
            "sat_map_gt": sat_map,
            "sat_map": sat_map,
            "left_camera_k": torch.from_numpy(scaled_camera_k(calib, self.image_size)),
            "grd_left_imgs": grd_left_img,
            "gt_shift_x": torch.tensor([0.0], dtype=torch.float32),
            "gt_shift_y": torch.tensor([0.0], dtype=torch.float32),
            "theta": torch.tensor([0.0], dtype=torch.float32),
            "file_name": f"{record['sample_id']}.png",
            "sample_id": record["sample_id"],
            "lidar_cond": torch.from_numpy(lidar["lidar_cond"]).float(),
            "dynamic_mask": torch.from_numpy(lidar["dynamic_mask"]).float(),
            "dynamic_boxes": torch.from_numpy(lidar["dynamic_boxes"]).float(),
            "dynamic_box_valid": torch.from_numpy(lidar["dynamic_box_valid"]).float(),
            "dynamic_class_hist": torch.from_numpy(lidar["dynamic_class_hist"]).float(),
            "lidar_valid_mask": torch.from_numpy(lidar["lidar_valid_mask"]).float(),
            "num_dynamic_boxes": torch.as_tensor(lidar["num_dynamic_boxes"], dtype=torch.long),
            "num_projected_lidar_points": torch.as_tensor(lidar["num_projected_lidar_points"], dtype=torch.long),
            "num_projected_dynamic_points": torch.as_tensor(lidar["num_projected_dynamic_points"], dtype=torch.long),
        }
        return sample


class SatLidarRawDatasetTest(SatLidarRawDataset):
    pass
