import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.assemble_geometry_history_comparison import (
    ComparisonError,
    choose_h264_encoder,
    compute_differences,
    validate_pair,
)


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((8, 16, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def safe_id(sample_id: str) -> str:
    return sample_id.replace("/", "__")


def make_summary(*, disabled: bool) -> dict:
    return {
        "args": {
            "disable_history": disabled,
            "ddim_steps": 3,
            "eta": 0.0,
            "temperature": 1.0,
            "seed": 123,
            "start_index": 100,
            "num_samples": 32,
        },
        "step_noise_sha256": "step-noise",
        "cfg_batch_factor": 2,
        "uncond_cfg": 3.0,
        "hist_ckpt": "/server/stage_d/geometry_history_step_1000.pt",
        "base_ckpt": "/server/base/cfg_step_250000.pt",
        "hist_step": 1000,
        "block_indices": [12],
        "first_rgb": True,
        "future_rgb_condition_check_passed": True,
    }


def make_records(run_dir: Path, *, disabled: bool, image_offset: int = 0) -> list:
    records = []
    prev_output = "bootstrap-latent"
    for index in range(32):
        sample_id = f"drive/foo/{index:010d}"
        sid = safe_id(sample_id)
        gt_value = 25 + index
        if index == 0:
            pred_value = gt_value
        else:
            pred_value = gt_value + image_offset
        write_image(run_dir / "images" / "gt" / f"{sid}.png", gt_value)
        write_image(run_dir / "images" / "normal" / f"{sid}.png", pred_value)
        output_hash = "bootstrap-latent" if index == 0 else f"{'off' if disabled else 'on'}-latent-{index}"
        record = {
            "sample_id": sample_id,
            "frame_index": 100 + index,
            "sequence_id": "drive/foo",
            "is_observed_initial_frame": index == 0,
            "has_history": (index > 0 and not disabled),
            "history_disabled": disabled,
            "history_input_sha256": None if index == 0 else prev_output,
            "history_output_sha256": output_hash,
            "initial_noise_sha256": None if index == 0 else f"xT-{index}",
            "denoiser_batch_sizes": [] if index == 0 else [2, 2, 2],
            "history_attention_steps": [] if disabled or index == 0 else ["b12", "b12", "b12"],
            "normal_path": f"/server/copied/elsewhere/{sid}.png",
        }
        records.append(record)
        prev_output = output_hash
    return records


def make_run(root: Path, name: str, *, disabled: bool, image_offset: int = 0) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "run_summary.json").write_text(json.dumps(make_summary(disabled=disabled)))
    records = make_records(run_dir, disabled=disabled, image_offset=image_offset)
    (run_dir / "records.json").write_text(json.dumps(records))
    return run_dir


class GeometryHistoryComparisonTests(unittest.TestCase):
    def test_available_h264_encoder_is_selected(self):
        self.assertEqual(choose_h264_encoder(' V..... libopenh264 OpenH264 encoder'), 'libopenh264')
        self.assertEqual(choose_h264_encoder(' V..... libopenh264 OpenH264\n V..... libx264 H264'), 'libx264')
        self.assertIsNone(choose_h264_encoder(' V..... mpeg4 MPEG4 encoder'))

    def make_pair(self, *, on_offset: int = 7, off_offset: int = 2):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        on_dir = make_run(root, "on", disabled=False, image_offset=on_offset)
        off_dir = make_run(root, "off", disabled=True, image_offset=off_offset)
        return tmp, on_dir, off_dir

    def test_valid_pair_checks_history_chain_and_pixel_differences(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)

        payload = validate_pair(on_dir, off_dir)
        metrics = compute_differences(on_dir, off_dir, payload["on_records"], payload["off_records"])

        self.assertEqual(payload["on_records"][1]["history_input_sha256"], "bootstrap-latent")
        self.assertEqual(metrics["on_minus_off_abs"]["per_frame"][0]["mean_abs_rgb_0_255"], 0.0)
        self.assertEqual(metrics["on_minus_off_abs"]["per_frame"][1]["mean_abs_rgb_0_255"], 5.0)
        self.assertEqual(
            metrics["on_minus_off_abs"]["overall_excluding_first_mean_abs_rgb_0_255"],
            5.0,
        )

    def test_missing_frame_fails_fast(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)
        records = json.loads((on_dir / "records.json").read_text())
        records.pop()
        (on_dir / "records.json").write_text(json.dumps(records))

        with self.assertRaisesRegex(ComparisonError, "exactly 32"):
            validate_pair(on_dir, off_dir)

    def test_unpaired_initial_noise_fails_fast(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)
        records = json.loads((off_dir / "records.json").read_text())
        records[9]["initial_noise_sha256"] = "different"
        (off_dir / "records.json").write_text(json.dumps(records))

        with self.assertRaisesRegex(ComparisonError, "initial_noise_sha256 mismatch"):
            validate_pair(on_dir, off_dir)

    def test_unpaired_sampler_args_fail_fast(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)
        summary = json.loads((off_dir / "run_summary.json").read_text())
        summary["args"]["eta"] = 0.5
        (off_dir / "run_summary.json").write_text(json.dumps(summary))

        with self.assertRaisesRegex(ComparisonError, "args.eta"):
            validate_pair(on_dir, off_dir)

    def test_unpaired_ddim_steps_fail_fast(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)
        summary = json.loads((off_dir / "run_summary.json").read_text())
        summary["args"]["ddim_steps"] = 4
        (off_dir / "run_summary.json").write_text(json.dumps(summary))

        with self.assertRaisesRegex(ComparisonError, "args.ddim_steps"):
            validate_pair(on_dir, off_dir)

    def test_matched_positive_non_stage_d_history_step_is_valid(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)
        for run_dir in (on_dir, off_dir):
            summary = json.loads((run_dir / "run_summary.json").read_text())
            summary["hist_step"] = 1064
            summary["hist_ckpt"] = "/server/stage_f/history_adapter.pt"
            (run_dir / "run_summary.json").write_text(json.dumps(summary))

        validate_pair(on_dir, off_dir)

    def test_nonpositive_history_step_fails_fast(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)
        summary = json.loads((on_dir / "run_summary.json").read_text())
        summary["hist_step"] = 0
        (on_dir / "run_summary.json").write_text(json.dumps(summary))

        with self.assertRaisesRegex(ComparisonError, "hist_step must be positive"):
            validate_pair(on_dir, off_dir)

    def test_denoiser_call_count_must_match_ddim_steps(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)
        records = json.loads((on_dir / "records.json").read_text())
        records[5]["denoiser_batch_sizes"] = [2, 2]
        records[5]["history_attention_steps"] = ["b12", "b12"]
        (on_dir / "records.json").write_text(json.dumps(records))

        with self.assertRaisesRegex(ComparisonError, "denoiser call count"):
            validate_pair(on_dir, off_dir)

    def test_first_frame_must_match_gt_pixels(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)
        first_id = safe_id("drive/foo/0000000000")
        write_image(on_dir / "images" / "normal" / f"{first_id}.png", 99)

        with self.assertRaisesRegex(ComparisonError, "first RGB frame"):
            validate_pair(on_dir, off_dir)

    def test_server_normal_path_is_resolved_to_local_copy_by_basename(self):
        tmp, on_dir, off_dir = self.make_pair()
        self.addCleanup(tmp.cleanup)
        payload = validate_pair(on_dir, off_dir)

        metrics = compute_differences(on_dir, off_dir, payload["on_records"], payload["off_records"])
        self.assertEqual(len(metrics["on_minus_off_abs"]["per_frame"]), 32)


if __name__ == "__main__":
    unittest.main()
