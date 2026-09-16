"""Evaluation-only regression checks; no KITTI data or CUDA required."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from evaluate_geometry_history_regions import (
    attention_metrics, capture_existing_tensors, region_metrics, select_spaced_pairs,
    repeat_disabled,
)
from ldm.modules.temporal_history_attention import GeometryHistoryAttention


class RegionEvaluationTests(unittest.TestCase):
    def test_disabled_repeat_uses_outer_loss_not_ddpm_subtotal(self):
        class DDPM:
            def p_losses(self, x, t, noise=None):
                loss = noise.square().mean()
                self.last_loss_metrics = {'loss_total': float(loss)}
                return loss
        class Module:
            model = SimpleNamespace(DDPM=DDPM())
            def __call__(self, *args):
                return self.model.DDPM.p_losses(torch.ones(1, 4, 2, 2), torch.tensor([1])) + .125
        module = Module()
        result = repeat_disabled(module, {}, None, {}, 250, 42, False)
        self.assertEqual(result['loss_total'], module.model.DDPM.last_loss_metrics['loss_total'] + .125)

    def test_exact_region_decomposition_and_empty_regions(self):
        raw = torch.tensor([[[[1., 3.], [5., 7.]], [[3., 5.], [7., 9.]]]])
        mask = torch.tensor([[[True, False], [False, False]]])
        result = region_metrics(raw, mask)
        self.assertEqual(result['all'], .25 * result['valid'] + .75 * result['invalid'])
        self.assertEqual(result['valid'], 2)
        self.assertIsNone(region_metrics(raw, mask & False)['valid'])
        self.assertIsNone(region_metrics(raw, mask | True)['invalid'])
        with self.assertRaises(ValueError):
            region_metrics(raw, torch.ones(1, 1, 1))

    def test_spaced_selection_caps_short_drives_and_excludes_original_endpoints(self):
        rows = [dict(drive=d, frame_index=i) for d, n in [('a', 24), ('b', 100)] for i in range(n)]
        pairs = [(i - 1, i, False) for i in range(1, len(rows)) if rows[i - 1]['drive'] == rows[i]['drive']]
        chosen = select_spaced_pairs(rows, pairs, ['a', 'b'], {0, 1}, per_drive=5, min_gap=10)
        self.assertEqual(chosen, select_spaced_pairs(rows, pairs, ['a', 'b'], {0, 1}, 5, 10))
        self.assertFalse({i for p in chosen for i in p[:2]} & {0, 1})
        for drive in ('a', 'b'):
            frames = [rows[p[1]]['frame_index'] for p in chosen if rows[p[1]]['drive'] == drive]
            self.assertTrue(all(b - a >= 10 for a, b in zip(frames, frames[1:])))
            self.assertLessEqual(len(frames), 5)
        self.assertEqual(sum(rows[p[1]]['drive'] == 'a' for p in chosen), 3)

    def test_profiler_does_not_change_forward_or_rng_and_restores_on_error(self):
        attn = GeometryHistoryAttention(8, history_dim=8, heads=2, dim_head=4)
        torch.nn.init.normal_(attn.to_out.weight, std=.01)
        block = SimpleNamespace(history_attn=attn, history_block_index=12)
        x, cond, hist = (torch.randn(1, 4, 8) for _ in range(3))
        valid = torch.tensor([[[True, False], [True, True]]])
        grid = attn._identity_grid(1, 2, 2, x.device, x.dtype)
        class DDPM:
            def p_losses(self):
                out = attn(x, cond, hist, True, grid, valid, (2, 2), (2, 2))
                loss_mask, loss_mask_weight = None, 0
                loss_raw = out.transpose(1, 2).reshape(1, 8, 2, 2).square()
                return loss_raw.mean()
        ddpm = DDPM()
        expected = ddpm.p_losses()
        state = torch.random.get_rng_state().clone()
        with capture_existing_tensors(ddpm, [block], valid) as results:
            actual = ddpm.p_losses()
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertIsNone(sys.getprofile())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['attention'][0]['query_valid_fraction'], .75)
        self.assertAlmostEqual(results[0]['region_eps']['all'], float(actual), places=7)
        with self.assertRaisesRegex(RuntimeError, 'test exception'):
            with capture_existing_tensors(ddpm, [block], valid):
                raise RuntimeError('test exception')
        self.assertIsNone(sys.getprofile())

    def test_null_valid_is_not_all_query_null_and_denominator_is_explicit(self):
        x = torch.ones(1, 2, 2)
        local = dict(x=x, cond_summary=x * 2,
                     local_valid=torch.tensor([[[True], [False]]]),
                     attn=torch.tensor([[[[.8, .2], [0., 1.]]]]), query_hw=(1, 2))
        result = attention_metrics(local, x * .1, 12)
        self.assertAlmostEqual(result['null_all'], .6, places=6)
        self.assertAlmostEqual(result['null_valid'], .2, places=6)
        self.assertAlmostEqual(result['residual_to_x_all'], .1, places=6)
        self.assertAlmostEqual(result['residual_to_cond_all'], .05, places=6)
        local.pop('local_valid')
        result = attention_metrics(local, torch.zeros_like(x), 12)
        self.assertIsNone(result['null_valid'])
        self.assertEqual(result['null_all'], 1.)


if __name__ == '__main__':
    unittest.main()
