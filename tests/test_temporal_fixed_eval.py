from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from tools.temporal_fixed_eval import FixedTemporalEvaluator


class FakeDataset:
    def __init__(self, n):
        self.pairs = [(i * 2, i * 2 + 1) for i in range(n)]
        self.base = Namespace(records=[])
        for i in range(n):
            self.base.records.append({'sample_id': 'prev_%02d' % i})
            self.base.records.append({'sample_id': 'cur_%02d' % i})


class FakeUnet(torch.nn.Module):
    def forward(self, x, t, history=None, **cond):
        return torch.zeros_like(x)


class FakeDDPM(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.denoise_model = FakeUnet()
    def q_sample(self, x_start, t, noise=None):
        return x_start + noise


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.DDPM = FakeDDPM()


def settings_payload(n_train=16, n_heldout=16, base=None):
    def rows(n):
        return [{'pair_index': i, 'previous': 'prev_%02d' % i, 'current': 'cur_%02d' % i} for i in range(n)]
    return {'base': base or {'sha256': 'base-a', 'step': 120000}, 'subset': {'train': rows(n_train), 'heldout': rows(n_heldout)}}


class FakeEvaluator(FixedTemporalEvaluator):
    def __init__(self, *args, datasets=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = datasets or {'train': FakeDataset(16), 'heldout': FakeDataset(16)}
    def _load_datasets(self):
        return self.datasets
    def _encode_pairs(self, dataset, pair_indices):
        return [self._fake_pair(index) for index in pair_indices]
    def _fake_pair(self, pair_index):
        z = torch.ones(1, 4, 2, 2) * float(pair_index + 1)
        return {'cond': {}, 'z': z, 'history': {'latent': z.clone(), 'history_grid': torch.zeros(1,2,2,1,2),
                'sat_grid': torch.zeros(1,2,2,1,2), 'valid': torch.ones(1,2,2,1,dtype=torch.bool),
                'sat_valid': torch.ones(1,2,2,1,dtype=torch.bool), 'positions': torch.zeros(1,2,2,1,4)}}


class FixedTemporalEvaluatorTests(unittest.TestCase):
    def make_eval(self, td, rank=0, settings=None, base=None):
        path = Path(td) / 'settings.json'
        path.write_text(json.dumps(settings or settings_payload()))
        args = Namespace(fixed_eval_settings=str(path), seed=3407)
        info = {'rank': rank, 'world': 2, 'device': torch.device('cpu'), 'distributed': False}
        return FakeEvaluator(FakeModel(), None, args, info, Path(td) / 'out', base=base)

    def test_prepare_selects_prior_order_and_local_counts(self):
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td, rank=1).prepare()
            self.assertEqual(len(ev.cache['train_monitor']), 8)
            self.assertEqual(len(ev.cache['heldout']), 8)
            # rank1 takes odd pair indices in the original settings order.
            self.assertTrue(torch.equal(ev.cache['train_monitor'][0]['z'], torch.ones(1,4,2,2) * 2))
            self.assertTrue(torch.equal(ev.cache['heldout'][0]['z'], torch.ones(1,4,2,2) * 2))

    def test_rejects_pair_index_sample_id_mismatch(self):
        bad = settings_payload()
        bad['subset']['train'][0]['current'] = 'wrong_cur'
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td, settings=bad)
            with self.assertRaises(ValueError):
                ev.prepare()

    def test_rejects_world_not_two_and_base_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td)
            ev.info['world'] = ev.world = 1
            with self.assertRaises(ValueError): ev.prepare()
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td, base={'sha256': 'other', 'step': 120000})
            with self.assertRaises(ValueError): ev.prepare()

    def test_fixed_noise_is_repeatable_and_does_not_change_global_rng(self):
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td, rank=0)
            state = torch.random.get_rng_state()
            a = ev._noise((2, 4, 2, 2), torch.float32, torch.device('cpu'), ev._noise_seed('heldout', 4, 750))
            b = ev._noise((2, 4, 2, 2), torch.float32, torch.device('cpu'), ev._noise_seed('heldout', 4, 750))
            c = ev._noise((2, 4, 2, 2), torch.float32, torch.device('cpu'), ev._noise_seed('train_monitor', 4, 750))
            self.assertTrue(torch.equal(a, b))
            self.assertFalse(torch.equal(a, c))
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_evaluate_writes_rank0_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            ev = self.make_eval(td, rank=0).prepare()
            rows = ev.evaluate(step=12, epoch_fraction=0.25)
            self.assertEqual(len(rows), 80)  # 2 splits * 4 B2 batches * 5 timesteps * 2 samples
            out = Path(td) / 'out' / 'fixed_eval' / 'step_0000012.json'
            self.assertTrue(out.exists())
            data = json.loads(out.read_text())
            self.assertEqual(data['metrics']['summary']['train_monitor']['pairs'], 8)
            self.assertEqual(data['metrics']['summary']['heldout']['evaluated_pair_timesteps'], 40)


if __name__ == '__main__':
    unittest.main()
