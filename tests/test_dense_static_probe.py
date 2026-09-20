from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ldm.modules.static_history import DenseStaticHistoryAdapter
from tools.infer_temporal import load_static_adapter, parse_args, prepare_history_from_payload
from tools.train_dense_static_history import (
    CHECKPOINT_VERSION,
    load_reference_npz,
    save_dense_checkpoint,
    sparse_dense_history,
)


def entry():
    return {
        "name": "p0",
        "split": "train",
        "previous": "2011_09_30/d/0000000001",
        "current": "2011_09_30/d/0000000002",
    }


def reference_arrays():
    h, w = 128, 512
    valid = np.zeros((h, w), dtype=bool)
    measured = np.zeros((h, w), dtype=bool)
    estimated = np.zeros((h, w), dtype=bool)
    measured[:10, :20] = True
    estimated[20:30, 30:40] = True
    valid[:] = measured | estimated
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[valid] = 0.5
    return rgb, valid, measured, estimated


class DenseStaticProbeTests(unittest.TestCase):
    def test_reference_npz_validates_identity_partition_and_holes(self):
        rgb, valid, measured, estimated = reference_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.npz"
            np.savez(path, warped_rgb=rgb, support_mask=valid, measured_mask=measured,
                     estimated_mask=estimated, previous=entry()["previous"], current=entry()["current"])
            history, meta = load_reference_npz(path, entry(), torch.zeros(1, 4, 16, 64))
            self.assertEqual(history["dense_rgb"].shape, (1, 3, 128, 512))
            self.assertTrue(torch.equal(history["dense_valid"], history["dense_measured"] | history["dense_estimated"]))
            self.assertGreater(meta["sha256"], "")
            bad_rgb = rgb.copy()
            bad_rgb[~valid] = 0.25
            bad = Path(tmp) / "bad.npz"
            np.savez(bad, warped_rgb=bad_rgb, support_mask=valid, measured_mask=measured,
                     estimated_mask=estimated, previous=entry()["previous"], current=entry()["current"])
            with self.assertRaisesRegex(ValueError, "zero outside"):
                load_reference_npz(bad, entry(), torch.zeros(1, 4, 16, 64))
            missing_id = Path(tmp) / "missing_id.npz"
            np.savez(missing_id, warped_rgb=rgb, support_mask=valid, measured_mask=measured, estimated_mask=estimated)
            with self.assertRaisesRegex(ValueError, "previous/current identity"):
                load_reference_npz(missing_id, entry(), torch.zeros(1, 4, 16, 64))
            wrong = Path(tmp) / "wrong.npz"
            np.savez(wrong, warped_rgb=rgb, support_mask=valid, measured_mask=measured,
                     estimated_mask=estimated, previous="other", current=entry()["current"])
            with self.assertRaisesRegex(ValueError, "previous identity"):
                load_reference_npz(wrong, entry(), torch.zeros(1, 4, 16, 64))

    def test_sparse_history_uses_strict_mask_as_measured_only(self):
        strict = np.zeros((1, 128, 512), dtype=bool)
        strict[:, :4, :5] = True
        support = np.ones((1, 128, 512), dtype=bool)
        rgb = np.ones((3, 128, 512), dtype=np.float32)
        item = {"z_prev": torch.zeros(1, 4, 16, 64), "geometry": {"strict_mask": strict, "support_mask": support, "warped_rgb": rgb}}
        history = sparse_dense_history(item)
        self.assertTrue(torch.equal(history["dense_valid"], history["dense_measured"]))
        self.assertEqual(int(history["dense_estimated"].sum()), 0)
        self.assertEqual(int(history["dense_valid"].sum()), 20)
        self.assertEqual(float(history["dense_rgb"].masked_select(~history["dense_valid"].expand_as(history["dense_rgb"])).sum()), 0.0)

    def test_v2_checkpoint_loads_strictly_and_rejects_variant_mismatch(self):
        adapter = DenseStaticHistoryAdapter(8, hidden_dim=4, input_variant="types")
        model = SimpleNamespace(DDPM=SimpleNamespace(denoise_model=SimpleNamespace(temporal_history=adapter)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dense.pt"
            save_dense_checkpoint(path, model, None, 5, {"sha256": "base"},
                                  {"reference_kind": "dense", "input_variant": "types"})
            target = DenseStaticHistoryAdapter(8, hidden_dim=4, input_variant="types")
            load_static_adapter(path, target, {"sha256": "base"})
            for key, value in adapter.state_dict().items():
                torch.testing.assert_close(target.state_dict()[key], value)
            with self.assertRaisesRegex(ValueError, "input_variant"):
                load_static_adapter(path, DenseStaticHistoryAdapter(8, hidden_dim=4, input_variant="rgb"), {"sha256": "base"})
        self.assertEqual(CHECKPOINT_VERSION, "temporal_static_adapter_v2")

    def test_dense_payload_allows_correct_off_and_rejects_wrong_modes(self):
        rgb, valid, measured, estimated = reference_arrays()
        history = {
            "latent": torch.zeros(1, 4, 16, 64),
            "dense_rgb": torch.as_tensor(rgb.transpose(2, 0, 1)).unsqueeze(0),
            "dense_valid": torch.as_tensor(valid).unsqueeze(0).unsqueeze(0),
            "dense_measured": torch.as_tensor(measured).unsqueeze(0).unsqueeze(0),
            "dense_estimated": torch.as_tensor(estimated).unsqueeze(0).unsqueeze(0),
        }
        off, _ = prepare_history_from_payload({"history": history}, torch.device("cpu"), "off")
        self.assertIsNone(off)
        correct, _ = prepare_history_from_payload({"history": history}, torch.device("cpu"), "correct")
        self.assertTrue(torch.equal(correct["enabled"], torch.ones(1, dtype=torch.bool)))
        with self.assertRaisesRegex(ValueError, "dense static payload"):
            prepare_history_from_payload({"history": history}, torch.device("cpu"), "wrong_history")

    def test_infer_cli_accepts_static_dense_mode_and_variant(self):
        args = parse_args([
            "--config", "cfg.yaml", "--checkpoint", "base.pt", "--static-checkpoint", "dense.pt",
            "--input", "payload.pt", "--out", "out.pt", "--temporal-mode", "static_dense",
            "--input-variant", "valid",
        ])
        self.assertEqual(args.temporal_mode, "static_dense")
        self.assertEqual(args.input_variant, "valid")


if __name__ == "__main__":
    unittest.main()
