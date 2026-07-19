import json
import tempfile
import unittest
from pathlib import Path

from dataloader.KITTI_raw_sat_lidar import SatLidarRawDataset


class KittiDatasetPathTest(unittest.TestCase):
    def test_kitti_root_rebases_manifest_paths_without_changing_sample_identity(self):
        record = {
            "date": "2011_09_26",
            "drive": "2011_09_26_drive_0002_sync",
            "frame_id": "0000000048",
            "frame_index": 48,
            "sample_id": "2011_09_26/2011_09_26_drive_0002_sync/0000000048",
            "calib_dir": "/old/KITTI_RAW/2011_09_26/2011_09_26_calib",
            "image_02_path": "/old/image.png",
            "oxts_path": "/old/oxts.txt",
            "satellite_path": "/old/satellite.png",
            "velodyne_path": "/old/velodyne.bin",
            "tracklet_xml_path": "/old/tracklet_labels.xml",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest = temp_root / "manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n")
            kitti_root = temp_root / "KITTI_RAW"
            calib_root = kitti_root / record["date"] / f"{record['date']}_calib" / record["date"]
            calib_root.mkdir(parents=True)
            (calib_root / "calib_cam_to_cam.txt").write_text("camera calibration")
            (calib_root / "calib_velo_to_cam.txt").write_text("velodyne calibration")
            dataset = SatLidarRawDataset(manifest=str(manifest), kitti_root=str(kitti_root))

            rebased = dataset.records[0]
            drive_root = kitti_root / record["date"] / record["drive"]
            self.assertEqual(rebased["sample_id"], record["sample_id"])
            self.assertEqual(rebased["calib_dir"], str(calib_root))
            self.assertEqual(
                rebased["image_02_path"],
                str(drive_root / "image_02" / "data" / "0000000048.png"),
            )
            self.assertEqual(
                rebased["velodyne_path"],
                str(drive_root / "velodyne_points" / "data" / "0000000048.bin"),
            )
            self.assertEqual(rebased["tracklet_xml_path"], str(drive_root / "tracklet_labels.xml"))


if __name__ == "__main__":
    unittest.main()
