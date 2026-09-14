import json
import tempfile
import unittest
from pathlib import Path

from tools.select_kitti_test_sequences import (
    build_report,
    read_jsonl,
    select_test_clips,
    validate_fixed_test_sequence_manifest,
    validate_selected_clips,
    write_jsonl as selector_write_jsonl,
)


def record(date, drive, frame_index, split="test2"):
    frame = f"{frame_index:010d}"
    return {
        "sample_id": f"{date}/{drive}/{frame}",
        "date": date,
        "drive": drive,
        "frame_index": frame_index,
        "split": split,
        "image_02_path": f"/kitti/{date}/{drive}/image_02/data/{frame}.png",
    }


def write_jsonl(path, records):
    with Path(path).open("w") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")


class KittiTestSequenceSelectionTest(unittest.TestCase):
    def test_selects_middle_segments_from_first_two_eligible_test_drives(self):
        test_records = []
        test_records.extend(record("2011_09_26", "drive_0005", idx) for idx in range(20))
        test_records.extend(record("2011_09_26", "drive_0003", idx) for idx in range(30, 50))
        test_records.extend(record("2011_09_26", "drive_0007", idx) for idx in range(100, 120))
        train_records = [record("2011_09_26", "drive_0001", idx, split="train") for idx in range(5)]

        clips = select_test_clips(test_records, train_records, clips=2, frames_per_clip=16)

        self.assertEqual(len(clips), 2)
        self.assertEqual([item["drive"] for item in clips[0]], ["drive_0003"] * 16)
        self.assertEqual([item["frame_index"] for item in clips[0]], list(range(32, 48)))
        self.assertEqual([item["drive"] for item in clips[1]], ["drive_0005"] * 16)
        self.assertEqual([item["frame_index"] for item in clips[1]], list(range(2, 18)))

    def test_rejects_train_drive_overlap(self):
        test_records = [record("2011_09_26", "drive_0001", idx) for idx in range(20)]
        train_records = [record("2011_09_26", "drive_0001", 100, split="train")]

        with self.assertRaisesRegex(RuntimeError, "Found 0 eligible"):
            select_test_clips(test_records, train_records, clips=1, frames_per_clip=16)

    def test_rejects_non_test2_records(self):
        test_records = [record("2011_09_26", "drive_0001", idx, split="test1") for idx in range(20)]

        with self.assertRaisesRegex(ValueError, "Expected split=test2"):
            select_test_clips(test_records, [], clips=1, frames_per_clip=16)

    def test_fails_closed_when_insufficient_consecutive_frames(self):
        test_records = [record("2011_09_26", "drive_0001", idx) for idx in list(range(8)) + list(range(20, 28))]

        with self.assertRaisesRegex(RuntimeError, "need 1 clips x 16 frames"):
            select_test_clips(test_records, [], clips=1, frames_per_clip=16)

    def test_validation_rejects_boundary_break_inside_clip(self):
        clip = [record("2011_09_26", "drive_0001", idx) for idx in range(8)]
        clip.append(record("2011_09_26", "drive_0001", 99))

        with self.assertRaisesRegex(ValueError, "not strictly consecutive"):
            validate_selected_clips([clip], [], expected_clips=1, frames_per_clip=9)

    def test_cli_writes_original_records_and_report(self):
        from tools.select_kitti_test_sequences import main
        import sys
        from contextlib import redirect_stdout
        import io

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            test_manifest = temp / "test.jsonl"
            train_manifest = temp / "train.jsonl"
            output_manifest = temp / "fixed.jsonl"
            report_json = temp / "fixed.report.json"
            test_records = [record("2011_09_26", "drive_0001", idx) for idx in range(20)]
            train_records = [record("2011_09_26", "drive_9999", idx, split="train") for idx in range(3)]
            write_jsonl(test_manifest, test_records)
            write_jsonl(train_manifest, train_records)

            old_argv = sys.argv
            try:
                sys.argv = [
                    "select_kitti_test_sequences.py",
                    "--test-manifest",
                    str(test_manifest),
                    "--train-manifest",
                    str(train_manifest),
                    "--output-manifest",
                    str(output_manifest),
                    "--report-json",
                    str(report_json),
                    "--clips",
                    "1",
                    "--frames-per-clip",
                    "16",
                ]
                with redirect_stdout(io.StringIO()):
                    main()
            finally:
                sys.argv = old_argv

            selected = read_jsonl(output_manifest)
            report_payload = json.loads(report_json.read_text())
            self.assertEqual(selected, test_records[2:18])
            self.assertEqual(report_payload["selected_sample_train_overlap"], [])
            self.assertEqual(report_payload["selected_drive_train_overlap"], [])
            self.assertEqual(report_payload["test_manifest_sha256"], report_payload["test_manifest_sha256"].lower())

    def test_validate_fixed_manifest_uses_report_and_train_overlap_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            test_manifest = temp / "test.jsonl"
            train_manifest = temp / "train.jsonl"
            selected_manifest = temp / "fixed.jsonl"
            report_json = temp / "fixed.report.json"
            test_records = [record("2011_09_26", "drive_0001", idx) for idx in range(16)]
            train_records = [record("2011_09_26", "drive_9999", idx, split="train") for idx in range(2)]
            write_jsonl(test_manifest, test_records)
            write_jsonl(train_manifest, train_records)
            selector_write_jsonl(selected_manifest, test_records)
            report_json.write_text(
                json.dumps(
                    build_report([test_records], train_records, test_manifest, train_manifest, selected_manifest)
                )
            )

            clips = validate_fixed_test_sequence_manifest(
                selected_manifest,
                train_manifest,
                test_manifest,
                report_json,
                expected_clips=1,
                frames_per_clip=16,
            )

            self.assertEqual(clips, [test_records])

    def test_validate_fixed_manifest_rejects_report_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            test_manifest = temp / "test.jsonl"
            train_manifest = temp / "train.jsonl"
            selected_manifest = temp / "fixed.jsonl"
            report_json = temp / "fixed.report.json"
            selected = [record("2011_09_26", "drive_0001", idx) for idx in range(16)]
            write_jsonl(test_manifest, selected)
            write_jsonl(train_manifest, [])
            selector_write_jsonl(selected_manifest, selected)
            report = build_report([selected], [], test_manifest, train_manifest, selected_manifest)
            report["selected_sample_ids"] = ["wrong"]
            report_json.write_text(json.dumps(report))

            with self.assertRaisesRegex(ValueError, "selected_sample_ids"):
                validate_fixed_test_sequence_manifest(
                    selected_manifest,
                    train_manifest,
                    test_manifest,
                    report_json,
                    expected_clips=1,
                    frames_per_clip=16,
                )

    def test_validate_fixed_manifest_rejects_sample_that_only_claims_test2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            test_manifest = temp / "test.jsonl"
            train_manifest = temp / "train.jsonl"
            selected_manifest = temp / "fixed.jsonl"
            authoritative = [record("2011_09_26", "drive_0001", idx) for idx in range(16)]
            impostor = [record("2011_09_26", "drive_9999", idx) for idx in range(16)]
            write_jsonl(test_manifest, authoritative)
            write_jsonl(train_manifest, [])
            selector_write_jsonl(selected_manifest, impostor)

            with self.assertRaisesRegex(ValueError, "not present in authoritative test manifest"):
                validate_fixed_test_sequence_manifest(
                    selected_manifest,
                    train_manifest,
                    test_manifest,
                    expected_clips=1,
                    frames_per_clip=16,
                )

    def test_validate_fixed_manifest_rejects_path_tampering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            test_manifest = temp / "test.jsonl"
            train_manifest = temp / "train.jsonl"
            selected_manifest = temp / "fixed.jsonl"
            selected = [record("2011_09_26", "drive_0001", idx) for idx in range(16)]
            tampered = [dict(item) for item in selected]
            tampered[3]["image_02_path"] = "/tmp/poisoned.png"
            write_jsonl(test_manifest, selected)
            write_jsonl(train_manifest, [])
            selector_write_jsonl(selected_manifest, tampered)

            with self.assertRaisesRegex(ValueError, "does not match authoritative test manifest"):
                validate_fixed_test_sequence_manifest(
                    selected_manifest,
                    train_manifest,
                    test_manifest,
                    expected_clips=1,
                    frames_per_clip=16,
                )

    def test_validate_fixed_manifest_rejects_report_source_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            test_manifest = temp / "test.jsonl"
            train_manifest = temp / "train.jsonl"
            selected_manifest = temp / "fixed.jsonl"
            report_json = temp / "fixed.report.json"
            selected = [record("2011_09_26", "drive_0001", idx) for idx in range(16)]
            write_jsonl(test_manifest, selected)
            write_jsonl(train_manifest, [])
            selector_write_jsonl(selected_manifest, selected)
            report_json.write_text(
                json.dumps(
                    {
                        "num_clips": 1,
                        "frames_per_clip": 16,
                        "test_manifest_sha256": "bad",
                        "train_manifest_sha256": "bad",
                        "selected_sample_ids": [item["sample_id"] for item in selected],
                        "selected_drive_train_overlap": [],
                        "selected_sample_train_overlap": [],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "test_manifest_sha256"):
                validate_fixed_test_sequence_manifest(
                    selected_manifest,
                    train_manifest,
                    test_manifest,
                    report_json,
                    expected_clips=1,
                    frames_per_clip=16,
                )


if __name__ == "__main__":
    unittest.main()
