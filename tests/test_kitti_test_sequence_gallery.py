import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from io import StringIO

from tools.build_kitti_test_sequence_gallery import build_gallery, main


def gallery_record(clip_index, frame_position, seed=1234):
    sample_id = f"clip{clip_index}/frame{frame_position:02d}"
    return {
        "sample_id": sample_id,
        "split": "test2",
        "drive": f"drive_{clip_index:04d}",
        "frame_index": 1000 + frame_position,
        "clip_index": clip_index,
        "seed": seed,
        "initial_noise_sha256": "a" * 64,
        "sources": {
            "satellite": f"images/satellite/{clip_index}_{frame_position}.png",
            "lidar_overlay": f"images/lidar_overlay/{clip_index}_{frame_position}.png",
            "gt": f"images/gt/{clip_index}_{frame_position}.png",
        },
        "outputs": {
            "3.0": f"images/cfg_3/{clip_index}_{frame_position}.png",
            "7.5": f"images/cfg_7.5/{clip_index}_{frame_position}.png",
        },
    }


def write_gallery_inputs(path):
    metadata = {
        "checkpoint_step": 45000,
        "cfg_scales": ["3.0", "7.5"],
        "frames_per_clip": 16,
        "num_clips": 2,
        "selection_report": {"clips": [{"clip_index": 0}, {"clip_index": 1}]},
    }
    records = [
        gallery_record(clip_index, frame_position, seed=700 + clip_index)
        for clip_index in range(2)
        for frame_position in range(16)
    ]
    (path / "metadata.json").write_text(json.dumps(metadata))
    (path / "records.json").write_text(json.dumps(records))


class KittiTestSequenceGalleryTest(unittest.TestCase):
    def test_build_gallery_writes_offline_comparison_html(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            write_gallery_inputs(temp)

            output_path = build_gallery(temp)
            html = output_path.read_text()

            self.assertEqual(output_path, temp / "comparison.html")
            self.assertIn("KITTI test2 连续帧对比", html)
            self.assertIn("CFG 3.0", html)
            self.assertIn("CFG 7.5", html)
            self.assertIn("逐帧独立生成", html)
            self.assertIn("images/cfg_7.5/1_15.png", html)
            self.assertNotIn("时序模型", html)

            match = re.search(
                r'<script id="gallery-data" type="application/json">(.*?)</script>',
                html,
                re.DOTALL,
            )
            self.assertIsNotNone(match)
            payload = json.loads(match.group(1))
            self.assertEqual(payload["checkpointLabel"], "45k")
            self.assertEqual(payload["framesPerClip"], 16)
            self.assertEqual(payload["numClips"], 2)
            self.assertEqual(payload["clips"][1][15]["outputs"]["7.5"], "images/cfg_7.5/1_15.png")

    def test_rejects_non_test2_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            write_gallery_inputs(temp)
            records = json.loads((temp / "records.json").read_text())
            records[0]["split"] = "train"
            (temp / "records.json").write_text(json.dumps(records))

            with self.assertRaisesRegex(ValueError, "split must be test2"):
                build_gallery(temp)

    def test_cli_accepts_input_dir(self):
        import sys

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            write_gallery_inputs(temp)
            old_argv = sys.argv
            try:
                sys.argv = ["build_kitti_test_sequence_gallery.py", "--input-dir", str(temp)]
                with redirect_stdout(StringIO()) as output:
                    main()
            finally:
                sys.argv = old_argv

            self.assertEqual(Path(output.getvalue().strip()), temp / "comparison.html")
            self.assertTrue((temp / "comparison.html").is_file())


if __name__ == "__main__":
    unittest.main()
