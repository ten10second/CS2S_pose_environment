import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.build_kitti_raw_sat_lidar_manifest import collect_records


class KittiManifestBuilderTest(unittest.TestCase):
    def test_kitti_location_split_builds_current_satellite_lidar_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "KITTI_RAW"
            split_root = Path(temp_dir) / "KITTI_location"
            date = "2011_09_26"
            drive = f"{date}_drive_0001_sync"
            calib_dir = root / date / f"{date}_calib"
            calib_dir.mkdir(parents=True)
            (calib_dir / "calib_cam_to_cam.txt").write_text("calibration")
            (calib_dir / "calib_velo_to_cam.txt").write_text("calibration")
            drive_root = root / date / drive
            for relative, suffix in (
                ("image_02/data", ".png"),
                ("satellite", ".png"),
                ("velodyne_points/data", ".bin"),
                ("oxts/data", ".txt"),
            ):
                directory = drive_root / relative
                directory.mkdir(parents=True)
                for frame_id in ("0000000000", "0000000001"):
                    (directory / f"{frame_id}{suffix}").write_bytes(b"data")

            split_root.mkdir()
            (split_root / "train_files.txt").write_text(f"{date}/{drive}/0000000000.png\n")
            (split_root / "test1_files.txt").write_text(f"{date}/{drive}/0000000001.png\n")
            (split_root / "test2_files.txt").write_text("")
            args = SimpleNamespace(
                kitti_root=str(root),
                date=["all"],
                split_root=str(split_root),
                split_mode="kitti_location",
                require_tracklet=False,
                val_every=5,
                val_drives=None,
                frame_stride=1,
                max_samples=0,
                skip_lidar_counts=True,
            )

            train, val, test1, test2, stats = collect_records(args)

            self.assertEqual(len(train), 1)
            self.assertEqual(len(val), 0)
            self.assertEqual(len(test1), 1)
            self.assertEqual(len(test2), 0)
            self.assertEqual(train[0]["split"], "train")
            self.assertEqual(test1[0]["split"], "test1")
            self.assertEqual(stats["split_strategy"], "kitti_location_train_test1_test2")


if __name__ == "__main__":
    unittest.main()
