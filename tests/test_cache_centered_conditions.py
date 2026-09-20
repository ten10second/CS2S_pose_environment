import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "cache_centered_conditions.py"
spec = importlib.util.spec_from_file_location("cache_centered_conditions", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CenteredConditionCacheTests(unittest.TestCase):
    def test_rgb_chw_accepts_uint8_and_normalizes(self):
        arr = np.zeros((128, 512, 3), dtype=np.uint8)
        arr[..., 0] = 255
        rgb = module.rgb_chw(arr, "rgb")
        self.assertEqual(rgb.shape, (3, 128, 512))
        self.assertEqual(rgb.dtype, torch.float32)
        self.assertTrue(torch.equal(rgb[0], torch.ones(128, 512)))
        self.assertTrue(torch.equal(rgb[1], torch.zeros(128, 512)))

    def test_rgb_chw_rejects_bad_float_range(self):
        arr = np.zeros((128, 512, 3), dtype=np.float32)
        arr[0, 0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "finite float RGB"):
            module.rgb_chw(arr, "rgb")


if __name__ == "__main__":
    unittest.main()
