import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import train_centered_decoder_probe as probe


def cache_item(name="x", split="train"):
    valid = torch.zeros(1, 128, 512, dtype=torch.bool)
    valid[:, :32] = True
    return {
        "name": name,
        "split": split,
        "previous": name + "_p",
        "current": name + "_c",
        "previous_rgb": torch.rand(3, 128, 512),
        "current_rgb": torch.rand(3, 128, 512),
        "source_flat_index": torch.arange(128 * 512, dtype=torch.long).reshape(128, 512),
        "valid": valid,
        "measured": valid.clone(),
        "estimated": torch.zeros_like(valid),
        "z_cur": torch.zeros(1, 4, 16, 64),
        "cond": {"context": torch.zeros(1, 4, 3)},
    }


class ProbeHelperTests(unittest.TestCase):
    def test_selection_requires_exact_counts(self):
        payload = {
            "train": [f"tr{i}" for i in range(8)],
            "heldout": [f"he{i}" for i in range(8)],
            "observation": [f"ob{i}" for i in range(10)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(payload))
            self.assertEqual(probe.load_selection(path), payload)
            payload["train"].append("extra")
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                probe.load_selection(path)

    def test_decoder_trainable_name_scope(self):
        self.assertFalse(probe.decoder_trainable_name("output_blocks.8.0.weight"))
        self.assertTrue(probe.decoder_trainable_name("output_blocks.9.0.weight"))
        self.assertTrue(probe.decoder_trainable_name("output_blocks.12.1.bias"))
        self.assertFalse(probe.decoder_trainable_name("output_blocks.9.1.transformer_blocks.0.attn1.sampling_offsets.weight"))
        self.assertFalse(probe.decoder_trainable_name("output_blocks.10.1.transformer_blocks.0.attn1.value_proj.bias"))
        self.assertFalse(probe.decoder_trainable_name("output_blocks.11.1.transformer_blocks.0.attn2.to_q.weight"))
        self.assertFalse(probe.decoder_trainable_name("output_blocks.11.1.transformer_blocks.0.attn2.to_out.0.bias"))
        self.assertTrue(probe.decoder_trainable_name("output_blocks.9.1.transformer_blocks.0.attn1.to_q.weight"))
        self.assertTrue(probe.decoder_trainable_name("output_blocks.9.1.transformer_blocks.0.attn2.sampling_offsets.weight"))
        self.assertTrue(probe.decoder_trainable_name("out.2.weight"))
        self.assertFalse(probe.decoder_trainable_name("input_blocks.9.0.weight"))
        self.assertFalse(probe.decoder_trainable_name("temporal_history.fusion.output.weight"))

    def test_wrong_history_uses_donor_rgb_and_recipient_masks(self):
        recipient = cache_item("recipient")
        donor = cache_item("donor")
        donor["previous_rgb"].zero_()
        donor["previous_rgb"][0].fill_(1.0)
        recipient["valid"].zero_()
        recipient["valid"][:, 16:48] = True
        history = probe.wrong_history(recipient, donor, torch.device("cpu"))
        self.assertTrue(torch.equal(history["dense_valid"], recipient["valid"].unsqueeze(0)))
        self.assertTrue(torch.equal(history["dense_measured"], recipient["measured"].unsqueeze(0)))
        self.assertEqual(history["dense_rgb"].shape, (1, 3, 128, 512))
        self.assertGreater(history["dense_rgb"][:, 0].sum().item(), 0)

    def test_filtered_hash_excludes_adapter_and_trainable_decoder(self):
        class FakeUnet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.temporal_history = torch.nn.Linear(2, 2)
                self.output_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(10)])
                self.out = torch.nn.Linear(2, 2)

        model = SimpleNamespace(DDPM=SimpleNamespace(denoise_model=FakeUnet()))
        trainable = {"output_blocks.9.weight", "output_blocks.9.bias"}
        before = probe.filtered_state_hash(model, trainable)
        with torch.no_grad():
            model.DDPM.denoise_model.temporal_history.weight.add_(1)
            model.DDPM.denoise_model.output_blocks[9].weight.add_(1)
        self.assertEqual(before, probe.filtered_state_hash(model, trainable))
        with torch.no_grad():
            model.DDPM.denoise_model.output_blocks[8].weight.add_(1)
        self.assertNotEqual(before, probe.filtered_state_hash(model, trainable))

    def test_group_b_uses_separate_adapter_and_decoder_lrs(self):
        class FakeUnet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.temporal_history = torch.nn.Linear(2, 2)
                self.output_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(10)])
                self.out = torch.nn.Linear(2, 2)
            def configure_temporal_history(self, **_kwargs):
                return None

        model = SimpleNamespace(DDPM=SimpleNamespace(denoise_model=FakeUnet()))
        groups, names = probe.configure_trainables(model, "B")
        self.assertEqual([g["name"] for g in groups], ["adapter", "decoder"])
        self.assertEqual([g["lr"] for g in groups], [1e-4, 1e-5])
        self.assertTrue(any(name.startswith("temporal_history.") for name in names))
        self.assertTrue(any(name.startswith("output_blocks.9.") for name in names))
        self.assertTrue(any(name.startswith("out.") for name in names))

    def test_render_samples_reuses_same_seed_across_modes_and_split_donor(self):
        calls = []
        groups = {
            "train": [cache_item("tr0", "train"), cache_item("tr1", "train")],
            "heldout": [cache_item("he0", "heldout"), cache_item("he1", "heldout")],
            "observation": [cache_item("ob0", "observation"), cache_item("ob1", "observation")],
        }
        class FakeModel:
            pass
        def fake_sample_frame(_model, _cond, shape, seed, device, history, steps, guidance):
            calls.append(seed)
            return torch.zeros(shape), {}, "hash-%d" % seed
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(probe, "sample_frame", side_effect=fake_sample_frame), \
             mock.patch.object(probe, "decode", side_effect=lambda _model, z: torch.zeros(z.shape[0], 3, 128, 512)), \
             mock.patch.object(probe, "save_rgb", side_effect=lambda *_args, **_kwargs: None), \
             mock.patch.object(probe, "unwrapped", side_effect=lambda _model: __import__("contextlib").nullcontext()):
            probe.render_samples(FakeModel(), groups, 100, Path(directory), torch.device("cpu"), 123, "A", False, 2, 1.0)
            rows = [json.loads(line) for line in (Path(directory) / "sample_metrics.jsonl").read_text().splitlines()]
        by_name = {}
        for row in rows:
            by_name.setdefault(row["name"], []).append(row)
        self.assertEqual(len(by_name["tr0"]), 3)
        self.assertEqual({row["sample_seed"] for row in by_name["tr0"]}, {by_name["tr0"][0]["sample_seed"]})
        self.assertEqual({row["noise_hash"] for row in by_name["tr0"]}, {by_name["tr0"][0]["noise_hash"]})
        self.assertEqual({row["donor_name"] for row in by_name["tr0"]}, {"tr1"})
        self.assertEqual({row["donor_name"] for row in by_name["he0"]}, {"he1"})
        self.assertNotIn("ob0", by_name)

    def test_save_probe_checkpoint_executes_and_records_hashes(self):
        class FakeUnet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.temporal_history = torch.nn.Linear(2, 2)
                self.output_blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(10)])
                self.out = torch.nn.Linear(2, 2)

        model = SimpleNamespace(DDPM=SimpleNamespace(denoise_model=FakeUnet()))
        args = SimpleNamespace(group="B", keep_checkpoints=2, sentinel="ok")
        trainable = ["output_blocks.9.weight", "output_blocks.9.bias", "out.weight", "out.bias"]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "checkpoints").mkdir()
            path = probe.save_probe_checkpoint(model, 7, out, {"base": "x"}, args, trainable,
                                               {"cache": "id"}, "frozen", "adapterhash", "decoderhash")
            payload = torch.load(path, map_location="cpu")
        self.assertEqual(payload["initial_adapter_sha256"], "adapterhash")
        self.assertEqual(payload["initial_decoder_sha256"], "decoderhash")
        self.assertEqual(set(payload["trainable_state"]), set(trainable))


if __name__ == "__main__":
    unittest.main()
