from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ldm.modules.temporal_pair_training import (
    configure_temporal_pair_trainables, load_temporal_checkpoint,
    save_temporal_checkpoint, training_contract, validate_resume_settings,
)
from tools.train_temporal_pairs import TrainingBatchSampler, parse_args, prepare_training_pair, shutdown_training_loader


class FakeUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_blocks = torch.nn.ModuleList([torch.nn.Conv2d(4, 4, 1)])
        self.out = torch.nn.Conv2d(4, 4, 1)

    def configure_temporal_history(self, enabled=True, mode="geometry", hidden_dim=64):
        self.temporal_history = torch.nn.Conv2d(4, 4, 1, bias=False)
        self.temporal_history.mode = mode
        self.temporal_history.hidden_dim = hidden_dim


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.DDPM = torch.nn.Module()
        self.DDPM.denoise_model = FakeUnet()


def cli_args():
    keys = ("config", "checkpoint", "train-manifest", "val-manifest", "kitti-root",
            "sd-base-ckpt", "lidar-pixel-feature-cache-root", "image-semantic-cache-root", "out-dir")
    return [part for key in keys for part in ("--" + key, "unused")]


class GTNextFrameTrainingTests(unittest.TestCase):
    def test_batch_window_preserves_distributed_order_and_resume(self):
        from torch.utils.data import BatchSampler, DistributedSampler
        dataset = list(range(31))
        for rank in range(2):
            sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True, seed=7)
            batches = TrainingBatchSampler(sampler, batch_size=2)
            for epoch in (0, 1):
                sampler.set_epoch(epoch)
                expected = list(BatchSampler(sampler, batch_size=2, drop_last=True))
                batches.set_window(3, 5)
                self.assertEqual(len(batches), 2)
                self.assertEqual(list(batches), expected[3:5])
                batches.set_window(0, len(expected))
                self.assertEqual(list(batches), expected)
            for start, stop in ((0, 0), (2, 1), (-1, 2), (0, 9)):
                with self.assertRaises(ValueError): batches.set_window(start, stop)

    def test_loader_cleanup_handles_worker_and_inprocess_iterators(self):
        from unittest.mock import Mock
        worker_iterator = Mock()
        shutdown_training_loader(worker_iterator)
        worker_iterator._shutdown_workers.assert_called_once_with()
        shutdown_training_loader(iter([1, 2]))

    def test_default_cli_requires_clean_reference_and_visual_pair_evaluation(self):
        args = parse_args(cli_args())
        contract = training_contract(args.training_task, args.history_policy, args.history_dropout)
        self.assertEqual(args.training_task, "gt_next_frame")
        self.assertEqual(args.history_dropout, 0)
        self.assertTrue(contract["history_required"])
        self.assertEqual(contract["history_source"], "previous_gt")
        self.assertGreater(args.pair_eval_every, 0)
        self.assertFalse(args.pair_eval_off)

    def test_gt_task_rejects_missing_reference_and_dropouts(self):
        for policy, dropout in (("off", 0), ("correct", 0.1), ("correct", 1)):
            with self.assertRaises(ValueError):
                training_contract("gt_next_frame", policy, dropout)
        for dropout in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                training_contract("history_ablation", "correct", dropout)
        self.assertEqual(training_contract("history_ablation", "off", 0)["history_source"], "none")

    def test_reference_and_target_remain_separate_and_all_references_enabled(self):
        args = parse_args(cli_args())
        class Model:
            satellite_condition_dropout_prob = 0.0
            @staticmethod
            def get_input(batch, key):
                return batch[key]
        previous = torch.full((2, 3, 2, 2), .2, requires_grad=True)
        current = torch.full((2, 3, 2, 2), .8, requires_grad=True)
        geometry = {"history_grid": np.zeros((2, 2, 1, 2), dtype=np.float32),
                    "sat_grid": np.zeros((2, 2, 1, 2), dtype=np.float32),
                    "valid": np.ones((2, 2, 1), dtype=bool),
                    "sat_valid": np.ones((2, 2, 1), dtype=bool),
                    "positions": np.zeros((2, 2, 1, 4), dtype=np.float32)}
        batch = {"prev": {"grd_left_imgs": previous}, "cur": {"grd_left_imgs": current},
                 "geometries": [geometry, geometry]}
        def encode(_model, rgb):
            return torch.cat([rgb, rgb[:, :1]], dim=1).detach()
        with patch("tools.train_temporal_pairs.encode_conditions", return_value=({"context": torch.ones(2, 3, 4)}, current)), \
             patch("tools.train_temporal_pairs.encode_latent", side_effect=encode), \
             patch("tools.train_temporal_pairs.apply_history_dropout", side_effect=AssertionError("GT references must not be dropped")):
            _, target, history, _, _ = prepare_training_pair(Model(), batch, args, torch.device("cpu"), 12)
        torch.testing.assert_close(history["latent"], encode(None, previous))
        torch.testing.assert_close(target, encode(None, current))
        self.assertFalse(history["latent"].requires_grad)
        self.assertTrue(history["enabled"].all())

    def test_weights_only_initialization_does_not_restore_optimizer_or_step(self):
        source = FakeModel(); configure_temporal_pair_trainables(source)
        with torch.no_grad(): source.DDPM.denoise_model.temporal_history.weight.fill_(.25)
        source_optimizer = torch.optim.AdamW([p for p in source.parameters() if p.requires_grad])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old.pt"
            save_temporal_checkpoint(path, source, source_optimizer, None, 1808,
                                     {"sha256": "base"}, {"history_dropout": .1})
            target = FakeModel(); configure_temporal_pair_trainables(target)
            optimizer = torch.optim.AdamW([p for p in target.parameters() if p.requires_grad])
            state = load_temporal_checkpoint(path, target, {"sha256": "base"})
            self.assertEqual(state["step"], 1808)
            self.assertEqual(len(optimizer.state), 0)
            torch.testing.assert_close(target.DDPM.denoise_model.temporal_history.weight,
                                       torch.full_like(target.DDPM.denoise_model.temporal_history.weight, .25))
            with self.assertRaisesRegex(ValueError, "init-temporal-ckpt"):
                validate_resume_settings(state["args"], {"training_task": "gt_next_frame"})

    def test_exact_resume_keeps_task_and_data_contract(self):
        settings = {"training_task": "gt_next_frame", "history_policy": "correct", "history_dropout": 0.,
                    "world_size": 2, "seed": 42, "batch_size": 4, "kitti_root": "data",
                    "sd_base_ckpt": "vae.pt", "lidar_pixel_feature_cache_root": "lidar",
                    "image_semantic_cache_root": "semantic"}
        validate_resume_settings(settings, settings)
        for key, value in (("history_dropout", .1), ("world_size", 1), ("seed", 43),
                           ("kitti_root", "other_data"), ("sd_base_ckpt", "other_vae.pt"),
                           ("lidar_pixel_feature_cache_root", "other_lidar"),
                           ("image_semantic_cache_root", "other_semantic")):
            with self.assertRaises(ValueError):
                validate_resume_settings(settings, dict(settings, **{key: value}))


if __name__ == "__main__":
    unittest.main()
