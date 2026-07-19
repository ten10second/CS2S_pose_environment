import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from tools.generate_kitti_raea_samples import (
    load_checkpoint_into_model,
    make_lidar_geometry_mask_for_sampling,
)
from tools.train_kitti_raea import (
    TrainingStepModule,
    build_loader,
    configure_cfg,
    ensure_fresh_run_directory,
    load_training_checkpoint,
    merge_distributed_records,
    validate_memmap_cache,
)


class _TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.lidar_context_model = None

    def training_step(self, batch, _step):
        return (self.weight * batch["x"]).mean()


class _StrictResumeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.DDPM = torch.nn.Module()
        self.DDPM.denoise_model = torch.nn.Linear(2, 2)
        self.condition_model_sat = torch.nn.Linear(2, 2)
        self.lidar_context_model = torch.nn.Linear(2, 2)


class KittiDistributedTrainingTest(unittest.TestCase):
    def test_current_config_is_fixed_to_fresh_raea_contract(self):
        args = SimpleNamespace(
            lr=1e-5,
            sd_base_ckpt="sd-v1-4.ckpt",
            lidar_support_loss_weight=1.0,
            lidar_support_dilation=8,
            lidar_depth_loss_weight=1.0,
            lidar_depth_log_eps=1e-3,
            lidar_semantic_alignment_weight=0.2,
            lidar_reference_window=3,
            ray_evidence_sat_bias=2.0,
            ray_evidence_lidar_bias=-2.0,
            ray_evidence_null_bias=-6.0,
            lidar_evidence_dilation=4,
            lidar_evidence_free_space_dilation=14,
            lidar_token_structure_target_ratio=0.08,
            batch_size=2,
            num_workers=2,
            train_manifest="train.jsonl",
            val_manifest="test.jsonl",
            kitti_root="/data/KITTI_RAW",
            lidar_point_feature_cache_root="/cache/utonia",
            image_semantic_cache_root="/cache/dino",
        )
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml"
        )
        cfg = configure_cfg(OmegaConf.load(config_path), args)

        self.assertIsNone(OmegaConf.select(cfg, "model.params.freeze_for_lidar_control"))
        self.assertEqual(
            cfg.model.params.Lidar_context_config.target,
            "models.KITTI_geo_ldm.lidar_condition_model.Lidar3DPointRayTokenEncoder",
        )
        self.assertEqual(cfg.model.params.DDPM_config.params.unet_config.params.lidar_context_dim, 768)
        self.assertEqual(cfg.data.params.train.params.raw_lidar_point_count, 4096)
        self.assertFalse(cfg.data.params.train.params.include_tracklets)
        self.assertIsNone(OmegaConf.select(cfg, "model.params.dynamic_point_x0_loss_weight"))
        self.assertIsNone(OmegaConf.select(cfg, "model.params.foreground_image_loss_weight"))
        self.assertIsNone(OmegaConf.select(cfg, "model.params.lidar_counterfactual_weight"))

    def test_training_step_wrapper_preserves_gradients(self):
        model = _TinyTrainingModel()
        wrapper = TrainingStepModule(model, token_structure_loss_weight=0.0)
        loss = wrapper({"x": torch.tensor([3.0])}, 1)
        loss.backward()
        self.assertAlmostEqual(float(model.weight.grad), 3.0)

    def test_distributed_sampler_partitions_dataset(self):
        dataset = TensorDataset(torch.arange(8))
        args = SimpleNamespace(
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            seed=3407,
        )
        _, sampler0 = build_loader(dataset, args, rank=0, world_size=2)
        _, sampler1 = build_loader(dataset, args, rank=1, world_size=2)
        rank0 = set(iter(sampler0))
        rank1 = set(iter(sampler1))
        self.assertFalse(rank0 & rank1)
        self.assertEqual(rank0 | rank1, set(range(len(dataset))))

    def test_metric_records_are_averaged(self):
        merged = merge_distributed_records(
            [
                {"step": 20, "loss": 1.0, "mode": "ray_evidence"},
                {"step": 20, "loss": 3.0, "mode": "ray_evidence"},
            ]
        )
        self.assertEqual(merged["step"], 20)
        self.assertEqual(merged["mode"], "ray_evidence")
        self.assertEqual(merged["loss"], 2.0)
        self.assertEqual(merged["distributed_metrics_world_size"], 2)

    def test_distributed_counters_are_summed(self):
        merged = merge_distributed_records(
            [
                {"num_projected_lidar_points": 10},
                {"num_projected_lidar_points": 20},
            ]
        )
        self.assertEqual(merged["num_projected_lidar_points"], 30)

    def test_memmap_cache_must_cover_every_manifest_sample(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "cache"
            root.mkdir()
            (root / "features.npy").write_bytes(b"present")
            (root / "masks.npy").write_bytes(b"present")
            manifest = Path(temp_dir) / "train.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps({"sample_id": sample_id})
                    for sample_id in ("drive/frame0", "drive/frame1")
                )
            )
            metadata = {
                "format": "kitti_feature_memmap_v1",
                "kind": "point",
                "feature_shape": [4, 3],
                "mask_shape": [4],
                "features_file": "features.npy",
                "masks_file": "masks.npy",
                "index": {"drive__frame0": 0},
            }
            (root / "memmap_meta.json").write_text(json.dumps(metadata))

            with self.assertRaisesRegex(RuntimeError, "misses 1/2 required samples"):
                validate_memmap_cache(root, "point", (4, 3), (4,), [manifest])

            metadata["index"]["drive__frame1"] = 1
            (root / "memmap_meta.json").write_text(json.dumps(metadata))
            stats = validate_memmap_cache(root, "point", (4, 3), (4,), [manifest])
            self.assertEqual(stats["point_cache_required_rows"], 2)

    def test_fresh_run_refuses_non_empty_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            (run_dir / "old_checkpoint.pt").write_text("stale")
            with self.assertRaises(FileExistsError):
                ensure_fresh_run_directory(run_dir, resume_ckpt="")
            ensure_fresh_run_directory(run_dir, resume_ckpt="checkpoint.pt")

    def test_resume_accepts_only_current_complete_checkpoint(self):
        model = _StrictResumeModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        payload = {
            "step": 37,
            "denoise_model": model.DDPM.denoise_model.state_dict(),
            "condition_model_sat": model.condition_model_sat.state_dict(),
            "lidar_context_model": model.lidar_context_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "current.pt"
            torch.save(payload, checkpoint)
            self.assertEqual(
                load_training_checkpoint(model, optimizer, checkpoint, scaler=scaler),
                37,
            )

            legacy = Path(temp_dir) / "legacy.pt"
            torch.save({"step": 12, "denoise_model_trainable": {}}, legacy)
            with self.assertRaisesRegex(RuntimeError, "not a current fresh-RAEA"):
                load_training_checkpoint(model, optimizer, legacy, scaler=scaler)

    def test_inference_rejects_legacy_partial_checkpoint(self):
        model = _StrictResumeModel()
        payload = {
            "denoise_model": model.DDPM.denoise_model.state_dict(),
            "condition_model_sat": model.condition_model_sat.state_dict(),
            "lidar_context_model": model.lidar_context_model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            current = Path(temp_dir) / "current.pt"
            torch.save(payload, current)
            load_checkpoint_into_model(model, current)

            legacy = Path(temp_dir) / "legacy.pt"
            torch.save({"denoise_model_trainable": {}}, legacy)
            with self.assertRaisesRegex(RuntimeError, "not a current fresh-RAEA"):
                load_checkpoint_into_model(model, legacy)

    def test_sampling_geometry_mask_uses_projected_lidar_hits(self):
        model = SimpleNamespace(DDPM=SimpleNamespace(ray_evidence_mask_mode="lidar_hit"))
        evidence = torch.randn(2, 8, 4, 6)
        mask = make_lidar_geometry_mask_for_sampling(model, evidence)
        torch.testing.assert_close(mask, evidence[:, 1:2])


if __name__ == "__main__":
    unittest.main()
