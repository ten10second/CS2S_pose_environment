import tempfile
import unittest
from pathlib import Path

from tools.cache_kitti_raw_manifest_files import PATH_KEYS, _cache_record


class KittiManifestCacheTest(unittest.TestCase):
    def test_current_manifest_files_are_copied_with_relative_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "source"
            cache_root = Path(temp_dir) / "cache"
            record = {}
            for index, key in enumerate(PATH_KEYS):
                path = source_root / "drive" / f"{index}.data"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(key.encode())
                record[key] = str(path)
            calib_dir = source_root / "calib"
            calib_dir.mkdir()
            (calib_dir / "calib_imu_to_velo.txt").write_text("calibration")
            record["calib_dir"] = str(calib_dir)

            cached, copied, missing = _cache_record(
                record,
                source_root=source_root,
                cache_root=cache_root,
                overwrite=False,
            )

            self.assertEqual(set(copied), set(PATH_KEYS))
            self.assertEqual(missing, [])
            for key in PATH_KEYS:
                self.assertTrue(Path(cached[key]).is_file())
                Path(cached[key]).resolve().relative_to(cache_root.resolve())


if __name__ == "__main__":
    unittest.main()
