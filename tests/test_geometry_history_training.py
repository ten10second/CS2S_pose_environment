"""CPU checks for fixed probes, explicit CFG dropout and drive separation."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile
from PIL import Image

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from train_kitti_geometry_history import (GeometryPairs, GeometryTrainingStep, pinned_denoising,
                                         split_drive_pairs, fixed_probe)
from temporal_history import history_host_parameters, history_trainable_parameters
from ldm.modules.temporal_history_attention import HistoryLatentEncoder


class DummyDiffusion:
    def __init__(self):
        self.last_loss_metrics = {}

    def p_losses(self, x, t, noise=None):
        value = x.square().mean() + noise.square().mean() + t.float().mean() / 1000
        self.last_loss_metrics = {"loss_denoise": float(value.detach())}
        return value


class DummyModel(torch.nn.Module):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub
        self.DDPM = DummyDiffusion()
        self.seen_condition = None

    def apply_satellite_condition_dropout(self, cond):
        raise AssertionError("wrapper must control dropout explicitly")

    def training_step(self, batch, index):
        self.seen_condition = self.apply_satellite_condition_dropout(torch.ones(1))
        x = batch["grd_left_imgs"] + torch.randn_like(batch["grd_left_imgs"])
        loss = self.DDPM.p_losses(x, torch.tensor([7]), noise=torch.randn_like(x))
        return loss + self.hub.payload["history_tokens"].sum() * 0


class Hub:
    def set(self, payload):
        self.payload = payload

    def clear(self):
        self.payload = None


class TrainingContracts(unittest.TestCase):
    def test_previous_frame_loads_rgb_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "previous.png"
            Image.new("RGB", (4, 4), "red").save(path)
            class Dataset:
                records = [{"image_02_path": str(path)}, {}]
                reads = []
                grd_transform = staticmethod(lambda image: torch.ones(3, 4, 4))

                def __getitem__(self, index):
                    self.reads.append(index)
                    return {"grd_left_imgs": torch.zeros(3, 4, 4)}
            dataset = Dataset()
            geometry = {"history_grid": torch.zeros(2, 2, 2), "history_valid": torch.ones(2, 2)}
            with patch("train_kitti_geometry_history.build_pair_geometry", return_value=geometry):
                item = GeometryPairs(dataset, [{}, {"sample_id": "current"}], [(0, 1, False)], "")[0]
            self.assertEqual(dataset.reads, [1])
            self.assertEqual(set(item["prev"]), {"grd_left_imgs"})

    def test_drive_split_no_shared_endpoints(self):
        rows = [{"drive": drive, "frame_index": i} for drive in ("a", "b", "c") for i in range(4)]
        train, val, held = split_drive_pairs(rows, ["b"])
        self.assertEqual(held, ["b"])
        self.assertFalse({i for p in train for i in p[:2]} & {i for p in val for i in p[:2]})
        with self.assertRaises(ValueError):
            split_drive_pairs(rows, ["missing"])

    def test_pinned_diffusion_is_repeatable_and_restored(self):
        ddpm = DummyDiffusion()
        original = ddpm.p_losses
        x = torch.ones(1, 4, 2, 2)
        with pinned_denoising(ddpm, 250, 11):
            first = ddpm.p_losses(x, torch.tensor([999]))
        with pinned_denoising(ddpm, 250, 11):
            second = ddpm.p_losses(x, torch.tensor([1]))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(ddpm.p_losses, original)

    def test_pinned_diffusion_requires_a_call(self):
        ddpm = DummyDiffusion()
        original = ddpm.p_losses
        with self.assertRaises(RuntimeError):
            with pinned_denoising(ddpm, 250, 11):
                pass
        self.assertEqual(ddpm.p_losses, original)

    def test_no_history_cfg_and_fixed_probe(self):
        hub = Hub()
        model = DummyModel(hub).eval()
        encoder = HistoryLatentEncoder(hidden=8, out_dim=8, grid=(2, 2))
        module = GeometryTrainingStep(model, encoder, hub)
        batch = {"grd_left_imgs": torch.ones(1, 3, 2, 2)}
        geom = {"history_grid": torch.zeros(1, 2, 2, 2),
                "history_valid": torch.ones(1, 2, 2, dtype=torch.bool)}
        original = model.apply_satellite_condition_dropout
        module(batch, None, geom, False, True).backward()
        self.assertEqual(model.seen_condition.item(), 0)
        self.assertFalse(model.training)
        self.assertIsNone(hub.payload)
        self.assertEqual(model.apply_satellite_condition_dropout, original)
        self.assertTrue(all(p.grad is not None for p in encoder.parameters()))
        latent = torch.randn(1, 4, 2, 2)
        rng = torch.random.get_rng_state().clone()
        first = fixed_probe(module, batch, latent, geom, latent + 1, [250, 750], 42, False)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        second = fixed_probe(module, batch, latent, geom, latent + 1, [250, 750], 42, False)
        self.assertEqual(first, second)
        for t in (250, 750):
            self.assertEqual(len({r["loss_total"] for r in first if r["t"] == t}), 1)

    def test_fixed_probe_satellite_axis_is_labelled_and_paired(self):
        hub = Hub()
        model = DummyModel(hub).eval()
        encoder = HistoryLatentEncoder(hidden=8, out_dim=8, grid=(2, 2))
        module = GeometryTrainingStep(model, encoder, hub)
        batch = {"grd_left_imgs": torch.ones(1, 3, 2, 2)}
        geom = {"history_grid": torch.zeros(1, 2, 2, 2),
                "history_valid": torch.ones(1, 2, 2, dtype=torch.bool)}
        latent = torch.randn(1, 4, 2, 2)
        both = fixed_probe(module, batch, latent, geom, latent + 1, [250], 7, False,
                           satellite_arms=(False, True))
        self.assertEqual(len(both), 8)
        self.assertEqual({r["satellite_blind"] for r in both}, {False, True})
        for blind in (False, True):
            cells = [r for r in both if r["satellite_blind"] is blind]
            self.assertEqual(len(cells), 4)
            self.assertEqual({r["condition"] for r in cells},
                             {"disabled", "correct", "wrong_geometry", "wrong_history"})
        # Adding the satellite-blind arm must not disturb the conditioned arm.
        single = fixed_probe(module, batch, latent, geom, latent + 1, [250], 7, False)
        self.assertTrue(all(r["satellite_blind"] is False for r in single))
        self.assertEqual([r["loss_total"] for r in single],
                         [r["loss_total"] for r in both if r["satellite_blind"] is False])

    def test_appearance_x0_applies_to_disabled_history(self):
        hub = Hub()
        model = DummyModel(hub).eval()
        encoder = HistoryLatentEncoder(hidden=8, out_dim=8, grid=(2, 2))
        module = GeometryTrainingStep(model, encoder, hub, appearance_x0_weight=1.0)
        seen = {}
        original = model.training_step

        def wrapped(batch, idx):
            seen["pair"] = model.DDPM._history_appearance_x0
            return original(batch, idx)

        model.training_step = wrapped
        batch = {"grd_left_imgs": torch.ones(1, 3, 2, 2)}
        geom = {"history_grid": torch.zeros(1, 2, 2, 2),
                "history_valid": torch.ones(1, 2, 2, dtype=torch.bool)}
        module(batch, None, geom, False, False)
        self.assertIsNotNone(seen["pair"])
        self.assertEqual(seen["pair"][1], 1.0)

    def test_unfreeze_host_adds_feedforward_parameters(self):
        encoder = HistoryLatentEncoder(hidden=8, out_dim=8, grid=(2, 2))

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.history_attn = torch.nn.Linear(2, 2, bias=False)
                self.ff = torch.nn.Linear(4, 4, bias=False)
                self.norm3 = torch.nn.LayerNorm(4)

        block = Block()
        frozen = history_trainable_parameters(encoder, [block], unfreeze_host=False)
        thawed = history_trainable_parameters(encoder, [block], unfreeze_host=True)
        self.assertGreater(sum(p.numel() for p in thawed), sum(p.numel() for p in frozen))
        self.assertTrue(history_host_parameters([block]))


if __name__ == "__main__":
    unittest.main()
