import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_centered_history_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_centered_history_data", MODULE_PATH)
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


def row(drive, frame):
    return {
        "sample_id": f"2011_09_26/{drive}/{frame:010d}",
        "date": "2011_09_26",
        "drive": drive,
        "frame_index": frame,
        "frame_id": f"{frame:010d}",
    }


def test_source_flat_index_reconstructs_warp_after_color_lookup():
    prev_rgb = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    support = np.zeros((3, 4), dtype=bool)
    support[0, 0] = True
    support[1, 2] = True
    support[2, 3] = True
    source_uv = np.zeros((3, 4, 2), dtype=np.float32)
    source_uv[0, 0] = [1.0, 0.0]
    source_uv[1, 2] = [3.0, 2.0]
    source_uv[2, 3] = [0.0, 1.0]
    index = prep.source_flat_index_from_uv(source_uv, support)
    warped = prep.reconstruct_warp_from_source_index(prev_rgb, index)
    assert index[0, 0] == 1
    assert index[1, 2] == 11
    assert index[2, 3] == 4
    assert np.allclose(warped[0, 0], prev_rgb[0, 1] / 255.0)
    assert np.allclose(warped[1, 2], prev_rgb[2, 3] / 255.0)
    assert np.allclose(warped[2, 3], prev_rgb[1, 0] / 255.0)
    assert np.all(warped[~support] == 0)
    stats = prep.validate_source_index(prev_rgb, warped, support, index)
    assert stats["source_index_reconstruction_max_abs"] == 0.0


def test_source_flat_index_rejects_out_of_bounds_supported_uv():
    support = np.ones((2, 2), dtype=bool)
    source_uv = np.zeros((2, 2, 2), dtype=np.float32)
    source_uv[1, 1] = [9.0, 0.0]
    try:
        prep.source_flat_index_from_uv(source_uv, support)
    except ValueError as exc:
        assert "out-of-bounds" in str(exc)
    else:
        raise AssertionError("expected out-of-bounds source_uv to fail")


def test_select_even_pairs_avoids_reused_frames_and_fixed_ids():
    rows = [row("drive_a", i) for i in range(10)] + [row("drive_b", i) for i in range(10)]
    used = {row("drive_a", 0)["sample_id"], row("drive_a", 1)["sample_id"]}
    pairs = prep.select_even_pairs(rows, 6, used)
    seen = set(used)
    drives = set()
    for prev, cur in pairs:
        assert prep.consecutive_rows(prev, cur)
        assert prev["sample_id"] not in seen
        assert cur["sample_id"] not in seen
        seen.add(prev["sample_id"])
        seen.add(cur["sample_id"])
        drives.add(prev["drive"])
    assert len(pairs) == 6
    assert drives == {"drive_a", "drive_b"}


def test_build_pair_plan_keeps_fixed_as_observation_and_excludes_from_training(tmp_path, monkeypatch):
    train_rows = [row("drive_train", i) for i in range(20)]
    held_rows = [row("drive_held", i) for i in range(10)]
    train_path = tmp_path / "train.jsonl"
    held_path = tmp_path / "held.jsonl"
    train_path.write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    held_path.write_text("\n".join(json.dumps(r) for r in held_rows) + "\n")
    fixed = [
        {
            "name": "train_01",
            "split": "train",
            "previous": train_rows[4]["sample_id"],
            "current": train_rows[5]["sample_id"],
        },
        {
            "name": "heldout_00",
            "split": "heldout",
            "previous": held_rows[1]["sample_id"],
            "current": held_rows[2]["sample_id"],
        },
    ]
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(json.dumps(fixed))
    entries, _, stats = prep.build_pair_plan(
        {
            "train_manifest": str(train_path),
            "val_manifest": str(held_path),
            "kitti_root": "/tmp/kitti",
        },
        fixed_path,
        train_count=3,
        heldout_count=2,
    )
    assert [e["name"] for e in entries[:2]] == ["obs_train_01", "obs_heldout_00"]
    train_selected = [e for e in entries if e["split"] == "train"]
    held_selected = [e for e in entries if e["split"] == "heldout"]
    fixed_ids = set(stats["fixed_sample_ids_excluded_from_train_heldout"])
    for entry in train_selected + held_selected:
        assert entry["previous"] not in fixed_ids
        assert entry["current"] not in fixed_ids
    assert len(train_selected) == 3
    assert len(held_selected) == 2


def test_validate_source_index_accepts_one_over_255_tolerance():
    prev_rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    prev_rgb[0, 0] = [255, 127, 0]
    support = np.zeros((2, 2), dtype=bool)
    support[1, 1] = True
    index = np.full((2, 2), -1, dtype=np.int64)
    index[1, 1] = 0
    warped = np.zeros((2, 2, 3), dtype=np.float32)
    warped[1, 1] = prev_rgb[0, 0].astype(np.float32) / 255.0
    assert prep.validate_source_index(prev_rgb, warped, support, index)["outside_support_max_abs"] == 0.0
