"""Unit tests for geometry-history rollout plumbing.

Run:
  python -B -m unittest discover -s tests -p 'test_geometry_history_rollout.py' -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import generate_kitti_geometry_history as gh  # noqa: E402


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = (16, 64)
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.null_token = torch.nn.Parameter(torch.full((1, 1, 64), -1.0))

    def forward(self, z):
        base = z.mean().reshape(1, 1, 1) * self.weight
        return base.expand(z.shape[0], 1024, 64)

    def null_tokens(self, batch):
        return self.null_token.expand(batch, 1, 64)


class TestArgumentHelpers(unittest.TestCase):
    def test_block_indices_are_required_and_ordered(self):
        self.assertEqual(gh.parse_block_indices("2,5,9"), (2, 5, 9))
        self.assertEqual(gh.parse_block_indices("after_bottleneck"), "after_bottleneck")
        with self.assertRaises(ValueError):
            gh.parse_block_indices("")
        with self.assertRaises(ValueError):
            gh.parse_block_indices("2,2")

    def test_cfg_batch_factor_matches_sampler_duplication(self):
        self.assertEqual(gh.cfg_batch_factor(0.0), 1)
        self.assertEqual(gh.cfg_batch_factor(1.0), 1)
        self.assertEqual(gh.cfg_batch_factor(3.0), 2)

    def test_fresh_out_dir_rejects_existing_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out"
            gh.ensure_fresh_out_dir(path)
            (path / "records.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                gh.ensure_fresh_out_dir(path)


class TestSequenceBoundary(unittest.TestCase):
    def test_consecutive_rows_same_drive_only(self):
        prev = {"drive": "d1", "frame_index": 10}
        self.assertTrue(gh.consecutive_rows(prev, {"drive": "d1", "frame_index": 11}))
        self.assertFalse(gh.consecutive_rows(prev, {"drive": "d1", "frame_index": 12}))
        self.assertFalse(gh.consecutive_rows(prev, {"drive": "d2", "frame_index": 11}))

    def test_first_rgb_bootstrap_is_one_frame_only(self):
        self.assertEqual(gh.next_history_latent_source(0, True), "first_rgb")
        self.assertEqual(gh.next_history_latent_source(1, True), "generated")
        self.assertEqual(gh.next_history_latent_source(0, False), "generated")
        self.assertEqual(gh.next_history_latent_source(0, True, "gt"), "first_rgb")
        self.assertEqual(gh.next_history_latent_source(1, True, "gt"), "gt_rgb")
        self.assertEqual(gh.next_history_latent_source(0, False, "gt"), "gt_rgb")

    def test_first_rgb_observed_frame_selection(self):
        self.assertTrue(gh.use_observed_initial_frame(0, True))
        self.assertFalse(gh.use_observed_initial_frame(1, True))
        self.assertFalse(gh.use_observed_initial_frame(0, False))

    def test_diagnostic_sequence_rejects_gaps_and_truncation(self):
        rows = [{'drive': 'a', 'frame_index': i} for i in range(32)]
        gh.require_contiguous_sequence(rows, 32)
        with self.assertRaises(ValueError):
            gh.require_contiguous_sequence(rows[:31], 32)
        rows[-1]['frame_index'] = 33
        with self.assertRaises(ValueError):
            gh.require_contiguous_sequence(rows, 32)

    def test_future_rgb_can_only_change_target(self):
        first = {'target': torch.ones(1), 'cond_label': torch.ones(2), 'range_img': None}
        gh.assert_sampling_inputs_equal(first, dict(first, target=torch.zeros(1)))
        with self.assertRaisesRegex(RuntimeError, 'cond_label'):
            gh.assert_sampling_inputs_equal(first, dict(first, cond_label=torch.zeros(2)))

    def test_noise_bank_and_bootstrap_are_reproducible(self):
        from ar_dyn_utils import seed_step_noise
        first = [torch.randn(1, 4, 2, 2) for _ in range(3)]
        second = [torch.randn(1, 4, 2, 2) for _ in range(3)]
        seed_step_noise(first, 42)
        seed_step_noise(second, 42)
        self.assertEqual(gh.tensor_sha256(torch.stack(first)), gh.tensor_sha256(torch.stack(second)))


class TestGeometryPayload(unittest.TestCase):
    def test_cfg_payload_repeats_history_and_geometry_for_both_branches(self):
        enc = _Encoder()
        z = torch.ones(1, 4, 16, 64)
        grid = np.zeros((16, 64, 2), dtype=np.float32)
        grid[..., 0] = 7
        valid = np.ones((16, 64), dtype=bool)
        payload = gh.build_geometry_history_payload(
            enc,
            z,
            {"history_grid": grid, "history_valid": valid},
            True,
            batch_factor=2,
        )
        self.assertEqual(tuple(payload["history_tokens"].shape), (2, 1024, 64))
        self.assertFalse(payload["history_tokens"].requires_grad)
        self.assertEqual(payload["history_hw"], (16, 64))
        self.assertEqual(tuple(payload["history_grid"].shape), (2, 16, 64, 2))
        self.assertEqual(tuple(payload["history_valid"].shape), (2, 16, 64))
        self.assertTrue(torch.equal(payload["history_tokens"][0], payload["history_tokens"][1]))
        self.assertTrue(torch.equal(payload["history_grid"][0], payload["history_grid"][1]))
        self.assertTrue(torch.equal(payload["history_valid"][0], payload["history_valid"][1]))
        self.assertTrue(payload["has_history"])

    def test_no_history_payload_is_explicitly_invalid_and_null(self):
        enc = _Encoder()
        payload = gh.build_geometry_history_payload(enc, None, None, False, batch_factor=2)
        self.assertFalse(payload["has_history"])
        self.assertEqual(payload["history_hw"], (16, 64))
        self.assertEqual(tuple(payload["history_tokens"].shape), (2, 1, 64))
        self.assertEqual(float(payload["history_grid"].abs().max()), 0.0)
        self.assertFalse(bool(payload["history_valid"].any()))

    def test_history_requires_geometry_maps(self):
        enc = _Encoder()
        with self.assertRaises(KeyError):
            gh.build_geometry_history_payload(enc, torch.ones(1, 4, 16, 64), {}, True)


class TestStrictCheckpoint(unittest.TestCase):
    def setUp(self):
        self.encoder = torch.nn.Linear(1, 1, bias=False)
        self.blocks = [
            SimpleNamespace(history_attn=torch.nn.Linear(1, 1, bias=False)),
            SimpleNamespace(history_attn=torch.nn.Linear(1, 1, bias=False)),
        ]
        self.args = SimpleNamespace(
            ckpt="/tmp/base.pt",
            block_indices=(3, 7),
            history_dim=64,
        )
        self.payload = {
            "step": 12,
            "history_encoder": self.encoder.state_dict(),
            "history_attn": {
                "0": self.blocks[0].history_attn.state_dict(),
                "1": self.blocks[1].history_attn.state_dict(),
            },
            "mode": gh.HISTORY_MODE,
            "base_ckpt": "/tmp/base.pt",
            "block_indices": (3, 7),
            "history_dim": 64,
        }

    def test_matching_payload_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hist.pt"
            torch.save(self.payload, path)
            loaded = gh.load_geometry_history_checkpoint(path, self.args, self.encoder, self.blocks)
        self.assertEqual(loaded["step"], 12)

    def test_mode_mismatch_fails_loudly(self):
        bad = dict(self.payload, mode="temporal_v2_history_phaseA")
        with self.assertRaises(ValueError):
            gh.require_geometry_history_payload(bad, self.args, self.blocks)

    def test_block_mismatch_fails_loudly(self):
        bad = dict(self.payload, block_indices=(3,))
        with self.assertRaises(ValueError):
            gh.require_geometry_history_payload(bad, self.args, self.blocks)

    def test_attention_key_mismatch_fails_loudly(self):
        bad = dict(self.payload, history_attn={"0": self.blocks[0].history_attn.state_dict()})
        with self.assertRaises(ValueError):
            gh.require_geometry_history_payload(bad, self.args, self.blocks)

    def test_unfreeze_host_is_required_and_restored(self):
        class HostBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.history_attn = torch.nn.Linear(1, 1, bias=False)
                self.ff = torch.nn.Linear(2, 2, bias=False)
                self.norm3 = torch.nn.LayerNorm(2)

        blocks = [HostBlock(), HostBlock()]
        payload = dict(self.payload)
        payload["history_attn"] = {str(i): b.history_attn.state_dict() for i, b in enumerate(blocks)}
        payload["unfreeze_host"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hist.pt"
            torch.save(payload, path)
            with self.assertRaises(KeyError):
                gh.load_geometry_history_checkpoint(path, self.args, self.encoder, blocks)
        from temporal_history import history_host_state_dict

        payload["history_host"] = history_host_state_dict(blocks)
        saved = float(blocks[0].ff.weight[0, 0].detach())
        blocks[0].ff.weight.data.fill_(3.0 if saved != 3.0 else 4.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hist.pt"
            torch.save(payload, path)
            gh.load_geometry_history_checkpoint(path, self.args, self.encoder, blocks)
        self.assertEqual(float(blocks[0].ff.weight[0, 0].detach()), saved)


if __name__ == "__main__":
    unittest.main()
