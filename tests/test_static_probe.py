from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tools.train_static_history import (
    CHECKPOINT_VERSION,
    CONDITION_POLICY,
    ddpm_num_timesteps,
    evaluate_monitor,
    load_pair_entries,
    normalize_static_regions,
    recursive_batch_to_device,
    row_for_entry_sample,
    save_static_checkpoint,
    sample_from_entry_dataset,
    stable_pair_seed,
    static_history_to_tensors,
    validate_pair_splits,
)
from tools.infer_temporal import load_static_adapter, parse_args, prepare_history_from_payload


def pair(name, split, drive, prev_frame, cur_frame, static_regions=True):
    out = {
        "name": name,
        "split": split,
        "previous": f"2011_09_30/{drive}/{prev_frame:010d}",
        "current": f"2011_09_30/{drive}/{cur_frame:010d}",
    }
    if static_regions:
        out["static_regions"] = {"prev": [[0, 0, 1, 1]], "target": [[0, 0, 1, 1]]}
    return out


class StaticProbeHelperTests(unittest.TestCase):
    def test_pair_split_validation_accepts_shared_drive_but_rejects_overlap_and_bad_pairs(self):
        clean = [
            pair("train", "train", "2011_09_30_drive_0018_sync", 1, 2),
            pair("heldout", "heldout", "2011_09_30_drive_0018_sync", 9, 10),
        ]
        info = validate_pair_splits(clean)
        self.assertEqual(info["counts"], {"train": 1, "heldout": 1})
        with self.assertRaisesRegex(ValueError, "sample id overlap"):
            validate_pair_splits([clean[0], dict(clean[0], name="heldout", split="heldout")])
        with self.assertRaisesRegex(ValueError, "crosses drives"):
            validate_pair_splits([
                clean[0],
                {
                    "name": "bad",
                    "split": "heldout",
                    "previous": "2011_09_30/drive_a/0000000001",
                    "current": "2011_09_30/drive_b/0000000002",
                },
            ])
        with self.assertRaisesRegex(ValueError, "not consecutive"):
            validate_pair_splits([clean[0], pair("skip", "heldout", "2011_09_30_drive_0018_sync", 20, 22)])

    def test_pairs_json_allows_missing_regions_and_requires_unique_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.json"
            path.write_text(json.dumps({"pairs": [pair("a", "train", "d1", 1, 2, static_regions=False), pair("b", "heldout", "d2", 1, 2)]}))
            self.assertEqual(len(load_pair_entries(path)), 2)
            path.write_text(json.dumps({"pairs": [pair("dup", "train", "d1", 1, 2), pair("dup", "heldout", "d2", 1, 2)]}))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_pair_entries(path)

    def test_region_aliases_normalize_to_geometry_contract(self):
        out = normalize_static_regions({"prev": [[0, 0, 1, 1]], "target": [[0.1, 0.1, 0.9, 0.9]]})
        self.assertEqual(set(out), {"prev_regions", "target_regions"})
        self.assertEqual(out["prev_regions"], [[0, 0, 1, 1]])
        self.assertEqual(out["target_regions"], [[0.1, 0.1, 0.9, 0.9]])
        self.assertIsNone(normalize_static_regions(None))
        with self.assertRaisesRegex(ValueError, "previous and target"):
            normalize_static_regions({"prev": [[0, 0, 1, 1]]})

    def test_declared_manifest_lookup_prevents_crosssplit_fallback(self):
        train_row = {"sample_id": "2011_09_30/d/0000000001", "value": "train"}
        val_row = {"sample_id": "2011_09_30/d/0000000009", "value": "val"}
        indexes = {
            "train_manifest": {"dataset": FakeDataset([train_row]), "index": {train_row["sample_id"]: 0}},
            "val_manifest": {"dataset": FakeDataset([val_row]), "index": {val_row["sample_id"]: 0}},
        }
        train_entry = {"name": "train", "split": "train"}
        heldout_entry = {"name": "heldout", "split": "heldout"}
        self.assertEqual(row_for_entry_sample(indexes, train_entry, train_row["sample_id"])["value"], "train")
        self.assertEqual(sample_from_entry_dataset(indexes, heldout_entry, val_row["sample_id"])["value"], "val")
        with self.assertRaisesRegex(KeyError, "declared train_manifest"):
            row_for_entry_sample(indexes, train_entry, val_row["sample_id"])
        with self.assertRaisesRegex(KeyError, "declared val_manifest"):
            sample_from_entry_dataset(indexes, heldout_entry, train_row["sample_id"])

    def test_monitor_seed_is_stable_across_phases(self):
        cache = [{"name": "pair_a", "split": "train"}, {"name": "pair_b", "split": "heldout"}]

        def fake_monitor(_model, item, seed, _device, use_history, timestep=500):
            return {
                "loss": 0.0,
                "pred_hash": f"{item['name']}-{use_history}",
                "seed": seed,
                "timestep": timestep,
            }

        with tempfile.TemporaryDirectory() as tmp, mock.patch("tools.train_static_history.fixed_epsilon_monitor", fake_monitor):
            before = evaluate_monitor(None, cache, 1234, torch.device("cpu"), "before", Path(tmp))
            after = evaluate_monitor(None, cache, 1234, torch.device("cpu"), "after", Path(tmp))
        self.assertEqual([row["seed"] for row in before], [row["seed"] for row in after])
        self.assertEqual(before[0]["seed"], stable_pair_seed(1234, "pair_a", "monitor"))
        self.assertTrue(all(row["timestep"] == 500 for row in before + after))

    def test_ddpm_timestep_helper_accepts_timesteps_fallback(self):
        self.assertEqual(ddpm_num_timesteps(SimpleNamespace(num_timesteps=1000)), 1000)
        self.assertEqual(ddpm_num_timesteps(SimpleNamespace(timesteps=777)), 777)
        with self.assertRaisesRegex(ValueError, "num_timesteps or timesteps"):
            ddpm_num_timesteps(SimpleNamespace())

    def test_static_history_condition_uses_strict_mask_only(self):
        geometry = {
            "warped_rgb": torch.zeros(3, 4, 4).numpy(),
            "support_mask": torch.ones(1, 4, 4, dtype=torch.bool).numpy(),
            "strict_mask": torch.zeros(1, 4, 4, dtype=torch.bool).numpy(),
            "confidence": torch.full((1, 4, 4), 0.25).numpy(),
        }
        geometry["strict_mask"][0, 1, 2] = True
        history = static_history_to_tensors(geometry, torch.zeros(1, 4, 2, 2), torch.device("cpu"))
        self.assertEqual(int(history["static_mask"].sum()), 1)
        self.assertEqual(float(history["static_confidence"].sum()), 1.0)

    def test_recursive_batch_preserves_lidar_context_pyramid_and_none(self):
        sample0 = {
            "context": torch.ones(1, 3, 4),
            "lidar_context": {
                "features": [torch.ones(1, 8, 16, 64), torch.ones(1, 16, 8, 32) * 2],
                "masks": [torch.ones(1, 1, 16, 64, dtype=torch.bool), torch.zeros(1, 1, 8, 32, dtype=torch.bool)],
            },
            "optional": None,
        }
        sample1 = {
            "context": torch.zeros(1, 3, 4),
            "lidar_context": {
                "features": [torch.zeros(1, 8, 16, 64), torch.zeros(1, 16, 8, 32)],
                "masks": [torch.zeros(1, 1, 16, 64, dtype=torch.bool), torch.ones(1, 1, 8, 32, dtype=torch.bool)],
            },
            "optional": None,
        }
        batched = recursive_batch_to_device([sample0, sample1], torch.device("cpu"))
        self.assertIsInstance(batched["lidar_context"], dict)
        self.assertEqual(len(batched["lidar_context"]["features"]), 2)
        self.assertEqual(batched["lidar_context"]["features"][0].shape, (2, 8, 16, 64))
        self.assertEqual(batched["lidar_context"]["features"][1].shape, (2, 16, 8, 32))
        self.assertEqual(batched["lidar_context"]["masks"][0].shape, (2, 1, 16, 64))
        self.assertIsNone(batched["optional"])
        with self.assertRaisesRegex(ValueError, "dict keys differ"):
            recursive_batch_to_device([sample0, {"context": torch.zeros(1, 3, 4)}], torch.device("cpu"))
        bad_none = dict(sample1)
        bad_none["optional"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "None mixed"):
            recursive_batch_to_device([sample0, bad_none], torch.device("cpu"))

    def test_static_checkpoint_saves_only_adapter_payload(self):
        adapter = Adapter()
        model = SimpleNamespace(DDPM=SimpleNamespace(denoise_model=SimpleNamespace(temporal_history=adapter)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_static_checkpoint(
                path,
                model,
                optimizer=None,
                step=3,
                base={"sha256": "base"},
                args={"depth_tol_m": 0.75},
            )
            payload = torch.load(path, map_location="cpu")
        self.assertEqual(payload["version"], CHECKPOINT_VERSION)
        self.assertEqual(payload["artifact_kind"], "temporal_static_adapter")
        self.assertEqual(payload["base_checkpoint"], {"sha256": "base"})
        self.assertEqual(payload["hidden_dim"], 16)
        self.assertIsNone(payload["optimizer"])
        self.assertEqual(set(payload["state_dict"]), set(adapter.state_dict()))
        self.assertFalse(any(key.startswith("output_blocks.") for key in payload["state_dict"]))
        self.assertEqual(payload["geometry_config"]["condition_policy"], CONDITION_POLICY)

    def test_static_checkpoint_loader_rejects_schema_and_base_mismatch(self):
        adapter = Adapter()
        model = SimpleNamespace(DDPM=SimpleNamespace(denoise_model=SimpleNamespace(temporal_history=adapter)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.pt"
            save_static_checkpoint(path, model, None, 3, {"sha256": "base"}, {"depth_tol_m": 0.75})
            target = Adapter()
            target.proj.weight.data.zero_()
            load_static_adapter(path, target, {"sha256": "base"})
            for key, value in adapter.state_dict().items():
                torch.testing.assert_close(target.state_dict()[key], value)
            with self.assertRaisesRegex(ValueError, "base checkpoint identity mismatch"):
                load_static_adapter(path, Adapter(), {"sha256": "other"})
            payload = torch.load(path, map_location="cpu")
            payload["state_dict"]["proj.weight"] = torch.full_like(payload["state_dict"]["proj.weight"], float("nan"))
            bad = Path(tmp) / "bad.pt"
            torch.save(payload, bad)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_static_adapter(bad, Adapter(), {"sha256": "base"})

    def test_static_payload_allows_correct_off_and_rejects_wrong_modes(self):
        history = {
            "latent": torch.zeros(1, 4, 2, 2),
            "static_rgb": torch.zeros(1, 3, 4, 4),
            "static_mask": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "static_confidence": torch.ones(1, 1, 4, 4),
        }
        off, _ = prepare_history_from_payload({"history": history}, torch.device("cpu"), "off")
        self.assertIsNone(off)
        correct, _ = prepare_history_from_payload({"history": history}, torch.device("cpu"), "correct")
        self.assertTrue(torch.equal(correct["enabled"], torch.ones(1, dtype=torch.bool)))
        with self.assertRaisesRegex(ValueError, "static payload requires explicit aligned RGB"):
            prepare_history_from_payload({"history": history}, torch.device("cpu"), "wrong_history")

    def test_infer_cli_rejects_mixed_temporal_and_static_checkpoints(self):
        argv = [
            "--config", "cfg.yaml",
            "--checkpoint", "base.pt",
            "--temporal-checkpoint", "temporal.pt",
            "--static-checkpoint", "static.pt",
            "--input", "payload.pt",
            "--out", "out.pt",
        ]
        with self.assertRaises(SystemExit):
            parse_args(argv)


class Adapter(torch.nn.Module):
    mode = "static"
    hidden_dim = 16

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(5, 7, 1)


class FakeDataset:
    def __init__(self, records):
        self.records = records

    def __getitem__(self, index):
        return self.records[index]


if __name__ == "__main__":
    unittest.main()
