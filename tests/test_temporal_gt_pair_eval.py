from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch

import tools.eval_temporal_gt_pairs as gt_eval
from tools.eval_temporal_gt_pairs import GTPairEvaluator, parse_args, preserve_torch_rng, unwrapped_denoiser, validate_args


def geometry(h=2, w=2, k=1):
    return {
        "history_grid": torch.zeros(h, w, k, 2).numpy(),
        "sat_grid": torch.zeros(h, w, k, 2).numpy(),
        "valid": torch.ones(h, w, k, dtype=torch.bool).numpy(),
        "sat_valid": torch.ones(h, w, k, dtype=torch.bool).numpy(),
        "positions": torch.zeros(h, w, k, 4).numpy(),
        "metrics": {"valid_fraction": 1.0},
    }


class FakePosterior:
    def __init__(self, value):
        self.value = value
    def mode(self):
        rgb = self.value
        return torch.cat([rgb[:, :1], rgb[:, :1], rgb[:, :1], rgb[:, :1]], dim=1)


class FakeAE(torch.nn.Module):
    def encode(self, value):
        return FakePosterior(value)
    def decode(self, value):
        return value[:, :3]


class FakeDenoiser(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return args[0]


class FakeDDPM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_model = FakeDenoiser()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.DDPM = FakeDDPM()
        self.pre_AE_model = FakeAE()
        self.scale_factor = 1.0
        self.lidar_condition_key = "lidar"
    def get_input(self, batch, key):
        return batch[key]
    def get_optional_lidar_batch_tensor(self, batch, key, device):
        return None
    def make_condition(self, sat, batch):
        return sat.mean(dim=(2, 3))
    def make_lidar_context(self, lidar_cond, **kwargs):
        return lidar_cond
    def make_lidar_evidence(self, lidar_cond):
        return lidar_cond[:, :1]
    def make_lidar_geometry_mask(self, evidence):
        return evidence > -1


class FakeDataset:
    def __init__(self, n=6):
        self.pairs = [(i * 2, i * 2 + 1) for i in range(n)]
        self.base = Namespace(records=[])
        for i in range(n):
            self.base.records.append({"sample_id": "prev_%02d" % i})
            self.base.records.append({"sample_id": "cur_%02d" % i})
    def __getitem__(self, index):
        prev_value = float(index + 1) / 10.0
        cur_value = float(index + 101) / 10.0
        image_shape = (3, 2, 2)
        cur = {
            "sat_map": torch.ones(image_shape) * 0.25,
            "grd_left_imgs": torch.ones(image_shape) * cur_value,
            "lidar": torch.ones(1, 2, 2),
            "camera_to_lidar": torch.zeros(4, 4, 1),
            "left_camera_k": torch.eye(3).unsqueeze(-1),
            "gt_shift_x": torch.tensor(0.0),
            "gt_shift_y": torch.tensor(0.0),
            "theta": torch.tensor(0.0),
        }
        return {
            "prev": {"grd_left_imgs": torch.ones(image_shape) * prev_value},
            "cur": cur,
            "prev_row": self.base.records[index * 2],
            "cur_row": self.base.records[index * 2 + 1],
            "geometry": geometry(),
            "geometry_metrics": {"valid_fraction": 1.0},
        }


class FakeSamplerEvaluator(GTPairEvaluator):
    def __init__(self, *args, dataset=None, fail_sample=False, bad_info="", nonfinite=False,
                 off_noise_mismatch=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset or FakeDataset()
        self.sample_calls = []
        self.fail_sample = fail_sample
        self.bad_info = bad_info
        self.nonfinite = nonfinite
        self.off_noise_mismatch = off_noise_mismatch
    def _load_dataset(self):
        return self.dataset
    def _sample(self, cond, shape, seed, history):
        if self.fail_sample:
            with unwrapped_denoiser(self.model):
                raise RuntimeError("sample exploded")
        self.sample_calls.append({"seed": int(seed), "history": history})
        latent_value = history["latent"].float().mean() if history is not None else torch.tensor(0.0)
        z = torch.ones(tuple(shape), device=self.device) * latent_value
        if self.nonfinite:
            z = z.clone()
            z.reshape(-1)[0] = float("nan")
        info = {"x_inter": [torch.zeros(tuple(shape))], "pred_x0": [torch.ones(tuple(shape))]}
        if history is not None:
            if self.bad_info == "missing_history":
                pass
            elif self.bad_info == "missing_steps":
                info["history"] = {"mode": "persistent_condition"}
            elif self.bad_info == "wrong_steps":
                info["history"] = {"mode": "persistent_condition", "steps": int(self.args.pair_eval_ddim_steps) + 1}
            else:
                info["history"] = {"mode": "persistent_condition", "steps": int(self.args.pair_eval_ddim_steps)}
        elif self.bad_info == "off_has_history":
            info["history"] = {"mode": "persistent_condition", "steps": int(self.args.pair_eval_ddim_steps)}
        noise = "noise_%d" % int(seed)
        if history is None and self.off_noise_mismatch:
            noise += "_off"
        return z, info, noise


def settings_payload(base=None):
    rows = [{"pair_index": i, "previous": "prev_%02d" % i, "current": "cur_%02d" % i} for i in range(5)]
    return {"base": base or {"sha256": "base-a", "step": 200000}, "subset": {"heldout": rows}}


class GTPairEvaluatorTests(unittest.TestCase):
    def make_eval(self, td, rank=0, world=1, settings=None, base=None, pair_eval_off=False, **kwargs):
        path = ""
        if settings is not None:
            path = str(Path(td) / "settings.json")
            Path(path).write_text(json.dumps(settings))
        args = Namespace(
            pair_eval_settings=path,
            fixed_eval_settings="",
            pair_eval_count=3,
            pair_eval_ddim_steps=4,
            pair_eval_guidance=7.5,
            pair_eval_off=pair_eval_off,
            geometry_depth_candidates=1,
            latent_grid_height=2,
            latent_grid_width=2,
            seed=123,
            kitti_root="",
        )
        info = {"rank": rank, "world": world, "device": torch.device("cpu"), "distributed": False}
        return FakeSamplerEvaluator(FakeModel(), Namespace(), args, info, Path(td) / "out", base=base, **kwargs)

    def test_settings_selection_validates_identity_and_sample_ids(self):
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td, settings=settings_payload(), base={"sha256": "base-a", "step": 200000}).prepare()
            self.assertEqual([x["pair_index"] for x in ev.selection], [0, 1, 2])
            self.assertTrue((Path(td) / "out" / "gt_pair_eval" / "settings.json").exists())
        bad = settings_payload()
        bad["subset"]["heldout"][0]["current"] = "wrong"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                self.make_eval(td, settings=bad).prepare()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                self.make_eval(td, settings=settings_payload(), base={"sha256": "other"}).prepare()

    def test_empty_settings_uses_deterministic_evenly_spaced_heldout_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td).prepare()
            self.assertEqual([x["pair_index"] for x in ev.selection], [0, 2, 5])
            persisted = json.loads((Path(td) / "out" / "gt_pair_eval" / "settings.json").read_text())
            self.assertEqual([x["current"] for x in persisted["selected_pairs"]], ["cur_00", "cur_02", "cur_05"])

    def test_encode_pair_uses_previous_gt_latent_not_current_gt_for_history(self):
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td).prepare()
            first = ev.cache[0]
            self.assertAlmostEqual(float(first["history"]["latent"].mean()), -0.8, places=5)
            self.assertAlmostEqual(float(first["cur_latent"].mean()), 1.0, places=5)
            self.assertFalse(torch.equal(first["history"]["latent"], first["cur_latent"]))

    def test_per_pair_seed_is_stable_across_rank_sharding(self):
        with tempfile.TemporaryDirectory() as td:
            full = self.make_eval(td).prepare().selection
            rank0 = self.make_eval(td, rank=0, world=2).prepare().cache
            rank1 = self.make_eval(td, rank=1, world=2).prepare().cache
            seen = {item["meta"]["ordinal"]: item["meta"]["seed"] for item in rank0 + rank1}
            self.assertEqual(seen, {row["ordinal"]: row["seed"] for row in full})

    def test_evaluate_resets_each_pair_to_real_previous_gt_history(self):
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td).prepare()
            rows = ev.evaluate(step=7, epoch_fraction=0.5)
            means = [float(call["history"]["latent"].mean()) for call in ev.sample_calls]
            self.assertEqual([round(x, 3) for x in means], [-0.8, -0.4, 0.2])
            self.assertEqual(len(rows), 3)
            self.assertTrue((Path(td) / "out" / "gt_pair_eval" / "step_0000007" / "index.html").exists())
            metrics = json.loads((Path(td) / "out" / "gt_pair_eval" / "step_0000007" / "metrics.json").read_text())
            self.assertEqual(metrics["metrics"]["history_source"], "previous_gt")
            self.assertIsNone(metrics["metrics"]["temporal_checkpoint"])
            self.assertEqual(metrics["pairs"][0]["sampler_info"], {
                "history": {"mode": "persistent_condition", "steps": 4},
                "history_active": True,
            })
            self.assertFalse((Path(td) / "out" / "gt_pair_eval" / "step_0000007" / "pair_000" / "prev_00").exists())

    def test_rejects_bad_sampler_info_and_nonfinite_outputs(self):
        cases = [
            {"bad_info": "missing_history"},
            {"bad_info": "missing_steps"},
            {"bad_info": "wrong_steps"},
            {"nonfinite": True},
        ]
        for kwargs in cases:
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(RuntimeError):
                    self.make_eval(td, dataset=FakeDataset(1), **kwargs).prepare().evaluate(step=1, epoch_fraction=0.0)

    def test_off_diagnostic_uses_same_noise_and_rejects_history(self):
        with tempfile.TemporaryDirectory() as td:
            rows = self.make_eval(td, dataset=FakeDataset(1), pair_eval_off=True).prepare().evaluate(step=1, epoch_fraction=0.0)
            self.assertEqual(rows[0]["noise_hash"], rows[0]["off_noise_hash"])
            self.assertEqual(rows[0]["off_sampler_info"], {"history": None, "history_active": False})
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                self.make_eval(td, dataset=FakeDataset(1), pair_eval_off=True, off_noise_mismatch=True).prepare().evaluate(step=1, epoch_fraction=0.0)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                self.make_eval(td, dataset=FakeDataset(1), pair_eval_off=True, bad_info="off_has_history").prepare().evaluate(step=1, epoch_fraction=0.0)

    def test_base_sample_preserves_rng_and_restores_ddp_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td, dataset=FakeDataset(1)).prepare()
            class Wrapper(torch.nn.Module):
                def __init__(self, module):
                    super().__init__(); self.module = module
            wrapper = Wrapper(ev.model.DDPM.denoise_model)
            ev.model.DDPM.denoise_model = wrapper
            original = gt_eval.sample_frame
            def fake_sample_frame(model, cond, shape, seed, device, history=None, steps=50, guidance=7.5):
                torch.rand(8)
                return torch.zeros(tuple(shape)), {"history": {"mode": "persistent_condition", "steps": steps}}, "noise_%d" % seed
            gt_eval.sample_frame = fake_sample_frame
            try:
                state = torch.random.get_rng_state()
                ev._sample(ev.cache[0]["cond"], ev.cache[0]["cur_latent"].shape, 99, ev.cache[0]["history"])
                self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
                self.assertIs(ev.model.DDPM.denoise_model, wrapper)
            finally:
                gt_eval.sample_frame = original

    def test_cuda_rng_preservation_is_limited_to_requested_device(self):
        original_get = torch.cuda.get_rng_state
        original_set = torch.cuda.set_rng_state
        original_get_all = torch.cuda.get_rng_state_all
        original_set_all = torch.cuda.set_rng_state_all
        original_current = torch.cuda.current_device
        calls = []
        torch.cuda.get_rng_state = lambda device=None: calls.append(("get", device)) or torch.ByteTensor([1, 2, 3])
        torch.cuda.set_rng_state = lambda state, device=None: calls.append(("set", device))
        torch.cuda.get_rng_state_all = lambda: (_ for _ in ()).throw(AssertionError("all gpu get should not be called"))
        torch.cuda.set_rng_state_all = lambda states: (_ for _ in ()).throw(AssertionError("all gpu set should not be called"))
        torch.cuda.current_device = lambda: 7
        try:
            with preserve_torch_rng(torch.device("cuda:4")):
                pass
            self.assertEqual(calls, [("get", 4), ("set", 4)])
            calls.clear()
            with preserve_torch_rng(torch.device("cuda")):
                pass
            self.assertEqual(calls, [("get", 7), ("set", 7)])
        finally:
            torch.cuda.get_rng_state = original_get
            torch.cuda.set_rng_state = original_set
            torch.cuda.get_rng_state_all = original_get_all
            torch.cuda.set_rng_state_all = original_set_all
            torch.cuda.current_device = original_current

    def test_standalone_arg_validation_accepts_positive_finite_values(self):
        args = parse_args([
            "--config", "cfg.yaml",
            "--checkpoint", "base.pt",
            "--temporal-checkpoint", "temporal.pt",
            "--train-manifest", "train.jsonl",
            "--val-manifest", "val.jsonl",
            "--kitti-root", "/kitti",
            "--sd-base-ckpt", "sd.ckpt",
            "--lidar-pixel-feature-cache-root", "pix",
            "--image-semantic-cache-root", "sem",
            "--out-dir", "out",
            "--pair-eval-count", "1",
            "--pair-eval-ddim-steps", "1",
            "--pair-eval-guidance", "7.5",
        ])
        validate_args(args)
        self.assertEqual(args.pair_eval_count, 1)
        self.assertEqual(args.pair_eval_ddim_steps, 1)
        for field, value in (("pair_eval_count", 0), ("pair_eval_ddim_steps", 0), ("pair_eval_guidance", float("inf"))):
            bad = Namespace(**vars(args))
            setattr(bad, field, value)
            with self.assertRaises(ValueError):
                validate_args(bad)

    def test_ddp_unwrap_restores_wrapper_on_success_and_exception(self):
        class Wrapper(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
        model = FakeModel()
        wrapper = Wrapper(model.DDPM.denoise_model)
        model.DDPM.denoise_model = wrapper
        with unwrapped_denoiser(model):
            self.assertIs(model.DDPM.denoise_model, wrapper.module)
        self.assertIs(model.DDPM.denoise_model, wrapper)
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td, dataset=FakeDataset(1), fail_sample=True).prepare()
            model2 = ev.model
            wrapper2 = Wrapper(model2.DDPM.denoise_model)
            model2.DDPM.denoise_model = wrapper2
            with self.assertRaises(RuntimeError):
                ev.evaluate(step=1, epoch_fraction=0.0)
            self.assertIs(model2.DDPM.denoise_model, wrapper2)


if __name__ == "__main__":
    unittest.main()
