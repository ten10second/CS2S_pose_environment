from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ldm.modules.persistent_history import repeat_history
from ldm.modules.temporal_pair_training import (
    anchor_trainable_loss,
    apply_history_dropout,
    assert_same_base,
    build_history,
    configure_temporal_pair_trainables,
    epsilon_prediction_loss,
    flatten_trainable_groups,
    load_temporal_checkpoint,
    make_optimizer_param_groups,
    prune_checkpoints,
    save_temporal_checkpoint,
    seed_training_step,
    trainable_state_dict,
)


class FakeUnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Conv2d(4, 4, 1)
        self.output_blocks = torch.nn.ModuleList([torch.nn.Conv2d(4, 4, 1)])
        self.out = torch.nn.Conv2d(4, 4, 1)
        self.configured = None

    def configure_temporal_history(self, enabled=True, mode='geometry', hidden_dim=64):
        self.configured = (enabled, mode, hidden_dim)
        self.temporal_history = torch.nn.Conv2d(4, 4, 1, bias=False)
        self.temporal_history.mode = mode
        self.temporal_history.hidden_dim = int(hidden_dim)

    def forward(self, x, t, history=None, **kwargs):
        out = self.out(self.output_blocks[0](self.backbone(x)))
        if history is not None:
            enabled = history.get('enabled')
            scale = enabled.float().view(-1, 1, 1, 1) if enabled is not None else 1.0
            out = out + self.temporal_history(history['latent'].to(x.dtype)) * scale
        return out


class FakeDDPM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_model = FakeUnet()
        self.timesteps = 10
    def q_sample(self, x_start, t, noise=None):
        return x_start + noise * ((t.float().view(-1, 1, 1, 1) + 1) / 10.0)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.DDPM = FakeDDPM()


def geometry(n=2, h=3, w=5, k=2):
    grid = np.zeros((h, w, k, 2), dtype=np.float32)
    grid[..., 0] = np.linspace(-0.5, 0.5, w)[None, :, None]
    return [{
        'history_grid': grid.copy(), 'sat_grid': grid.copy(),
        'valid': np.ones((h, w, k), dtype=bool), 'sat_valid': np.ones((h, w, k), dtype=bool),
        'positions': np.zeros((h, w, k, 4), dtype=np.float32),
    } for _ in range(n)]


class TemporalPairTrainingTests(unittest.TestCase):
    def test_configure_selects_history_and_decoder_but_freezes_encoder(self):
        model = FakeModel()
        groups = configure_temporal_pair_trainables(model, mode='content', hidden_dim=32)
        params = flatten_trainable_groups(groups)
        self.assertEqual(model.DDPM.denoise_model.configured, (True, 'content', 32))
        self.assertEqual(len(params), 5)
        self.assertTrue(all(p.requires_grad for p in params))
        self.assertFalse(model.DDPM.denoise_model.backbone.weight.requires_grad)
        self.assertEqual(sorted(trainable_state_dict(model)), ['out.bias', 'out.weight', 'output_blocks.0.bias', 'output_blocks.0.weight', 'temporal_history.weight'])
        groups = {'temporal': [model.DDPM.denoise_model.temporal_history.weight], 'decoder': list(model.DDPM.denoise_model.output_blocks.parameters()) + list(model.DDPM.denoise_model.out.parameters())}
        opt_groups = make_optimizer_param_groups(groups, 1e-4, 2e-6)
        self.assertEqual([g['lr'] for g in opt_groups], [1e-4, 2e-6])

    def test_build_history_preserves_full_candidate_axis_and_cfg_repeat(self):
        z = torch.randn(2, 4, 3, 5)
        h = build_history(z, geometry(2, 3, 5))
        self.assertEqual(h['latent'].shape, (2, 4, 3, 5))
        self.assertEqual(h['history_grid'].shape, (2, 3, 5, 2, 2))
        self.assertEqual(h['sat_grid'].shape, (2, 3, 5, 2, 2))
        self.assertEqual(h['valid'].shape, (2, 3, 5, 2))
        self.assertEqual(h['positions'].shape, (2, 3, 5, 2, 4))
        r = repeat_history(h, 2)
        self.assertEqual(r['latent'].shape[0], 4)
        with self.assertRaises(ValueError): build_history(z, geometry(1, 3, 5))

    def test_seed_training_step_sets_python_numpy_and_torch(self):
        seed_training_step(44, torch.device("cpu"))
        torch_a = torch.rand(1)
        np_a = np.random.rand()
        seed_training_step(44, torch.device("cpu"))
        self.assertTrue(torch.equal(torch_a, torch.rand(1)))
        self.assertEqual(np_a, np.random.rand())

    def test_history_dropout_deterministic_and_rng_isolated(self):
        z = torch.zeros(6, 4, 2, 2)
        h = {'latent': z}
        before = torch.random.get_rng_state()
        a = apply_history_dropout(h, 0.5, 123)['enabled']
        b = apply_history_dropout(h, 0.5, 123)['enabled']
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertGreaterEqual(a.numel(), 1)

    def test_epsilon_loss_trains_history_and_decoder_with_reproducible_noise(self):
        model = FakeModel(); params = flatten_trainable_groups(configure_temporal_pair_trainables(model))
        z = torch.randn(2, 4, 3, 5)
        h = build_history(torch.randn_like(z), geometry(2, 3, 5)); h['enabled'] = torch.tensor([True, False])
        loss1, out1 = epsilon_prediction_loss(model.DDPM, z, {}, h, 99)
        loss2, out2 = epsilon_prediction_loss(model.DDPM, z, {}, h, 99)
        self.assertTrue(torch.equal(out1['t'], out2['t']))
        self.assertTrue(torch.equal(out1['noise'], out2['noise']))
        loss1.backward()
        self.assertIsNotNone(model.DDPM.denoise_model.temporal_history.weight.grad)
        self.assertIsNotNone(model.DDPM.denoise_model.out.weight.grad)
        self.assertIsNone(model.DDPM.denoise_model.backbone.weight.grad)


    def test_off_history_anchor_marks_all_trainables_used(self):
        model = FakeModel(); params = flatten_trainable_groups(configure_temporal_pair_trainables(model))
        z = torch.randn(2, 4, 3, 5)
        loss, _ = epsilon_prediction_loss(model.DDPM, z, {}, None, 101)
        loss = anchor_trainable_loss(loss, params)
        loss.backward()
        for parameter in params:
            self.assertIsNotNone(parameter.grad)
        self.assertIsNone(model.DDPM.denoise_model.backbone.weight.grad)

    def test_checkpoint_rejects_mixed_base(self):
        model = FakeModel(); configure_temporal_pair_trainables(model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'ckpt.pt'
            save_temporal_checkpoint(p, model, opt, None, 7, {'sha256': 'base-a', 'step': 1}, {'x': 1}, epoch=2, dataloader_offset=3)
            payload = load_temporal_checkpoint(p, model, {'sha256': 'base-a', 'step': 1}, opt)
            self.assertEqual(payload['step'], 7)
            with self.assertRaises(ValueError): load_temporal_checkpoint(p, model, {'sha256': 'base-b', 'step': 1})


    def test_checkpoint_rejects_history_mode_and_hidden_mismatch(self):
        model = FakeModel(); configure_temporal_pair_trainables(model, mode='geometry', hidden_dim=64)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'ckpt.pt'
            save_temporal_checkpoint(p, model, None, None, 1, {'sha256': 'base-a'}, {'temporal_mode': 'geometry', 'temporal_hidden_dim': 64})
            other_mode = FakeModel(); configure_temporal_pair_trainables(other_mode, mode='content', hidden_dim=64)
            with self.assertRaises(ValueError):
                load_temporal_checkpoint(p, other_mode, {'sha256': 'base-a'})
            other_width = FakeModel(); configure_temporal_pair_trainables(other_width, mode='geometry', hidden_dim=32)
            with self.assertRaises(ValueError):
                load_temporal_checkpoint(p, other_width, {'sha256': 'base-a'})

    def test_prune_checkpoints_keeps_latest_two_temporal_pair_files_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ['temporal_pair_step_0000001.pt', 'temporal_pair_step_0000002.pt', 'temporal_pair_step_0000003.pt', 'unrelated.pt']:
                (root / name).write_text(name)
            prune_checkpoints(root, keep=2)
            self.assertFalse((root / 'temporal_pair_step_0000001.pt').exists())
            self.assertTrue((root / 'temporal_pair_step_0000002.pt').exists())
            self.assertTrue((root / 'temporal_pair_step_0000003.pt').exists())
            self.assertTrue((root / 'unrelated.pt').exists())

    def test_base_identity_helper_rejects_mismatch(self):
        with self.assertRaises(ValueError): assert_same_base({'sha256': 'a'}, {'sha256': 'b'})


if __name__ == '__main__': unittest.main()
