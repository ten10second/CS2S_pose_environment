import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch.nn.functional as F
from tools import eval_temporal_pairs as ev


class EvalTests(unittest.TestCase):
    def test_fp16_metrics_use_float32_and_sobel_axis_sum(self):
        target = torch.zeros(1, 3, 128, 512, dtype=torch.float16)
        rgb = torch.ones_like(target) * .5
        stats = ev.image_metrics(rgb, target)
        self.assertEqual(stats['rgb_mae'], .5)
        self.assertEqual(stats['rgb_mse'], .25)
        self.assertEqual(stats['sobel_mae'], 0)
        with self.assertRaises(ValueError):
            ev.image_metrics(rgb * float('nan'), target)

    def test_end_to_end_four_matched_controls_and_artifacts(self):
        torch.set_num_threads(1)
        prev = torch.full((3, 128, 512), .2)
        target = torch.full_like(prev, .8)
        valid = torch.ones(1, 128, 512, dtype=torch.bool)
        ref = dict(previous_rgb=prev, current_rgb=target, valid=valid,
                   measured=valid, estimated=~valid,
                   source_flat_index=torch.arange(128 * 512).view(128, 512))
        class Dataset:
            records = [{'sample_id': 'd/v/0001'}, {'sample_id': 'd/v/0002'}]
            def __getitem__(self, i):
                return {'grd_left_imgs': target.clone()}
        model = torch.nn.Linear(1, 1)
        model.loaded = False
        cfg = SimpleNamespace(data=SimpleNamespace(params=SimpleNamespace(train='train', test='test')))
        calls, history_rgbs = [], []
        def make(model, rgb, *masks):
            history_rgbs.append(rgb.clone())
            return dict(latent=F.avg_pool2d(torch.cat((rgb, rgb[:, :1]), 1), 8),
                        masks=F.avg_pool2d(torch.cat(masks, 1).float(), 8))
        def sample(model, cond, shape, seed, device, history, steps, guidance):
            calls.append((model.loaded, history is None, seed, steps))
            self.assertFalse(model.training)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            value = .4 if history is None else float(history['latent'].mean())
            return torch.full(shape, value), {}, str(seed)
        def load(*args):
            model.loaded = True
            return {'step': 648}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = root / 'model.pt'; checkpoint.touch()
            selection = root / 'selection.json'
            selection.write_text(json.dumps({'observation': [dict(name='p', previous='d/v/0001',
                current='d/v/0002', source_split='train', reference_npz='ref.npz')], 'heldout': []}))
            with patch.object(ev, 'load_settings_args', return_value=None), \
                 patch.object(ev, 'load_base', return_value=(model, {'sha256': 'base'}, cfg)), \
                 patch('utils.util.instantiate_from_config', return_value=Dataset()), \
                 patch.object(ev, 'load_reference', return_value=ref), \
                 patch.object(ev, 'previous_rgb_sample', return_value={'grd_left_imgs': prev}), \
                 patch.object(ev, 'encode_conditions', return_value=({'context': torch.ones(1, 1)}, None)), \
                 patch.object(ev, 'make_history_condition', side_effect=make), \
                 patch.object(ev, 'sample_frame', side_effect=sample), \
                 patch.object(ev, 'decode', side_effect=lambda model, z: F.interpolate(z[:, :3], scale_factor=8)), \
                 patch.object(ev, 'load_history_checkpoint', side_effect=load):
                ev.main(['--settings', 's', '--eval-selection', str(selection), '--checkpoint', str(checkpoint),
                         '--out', str(root / 'out'), '--device', 'cpu'])
            self.assertEqual(calls, [(False, True, 4307, 50), (True, False, 4307, 50),
                                     (True, True, 4307, 50), (True, False, 4307, 50)])
            self.assertTrue(torch.equal(history_rgbs[0][0], prev))
            self.assertEqual(float(history_rgbs[1].abs().sum()), 0)
            self.assertEqual(len(list((root / 'out').glob('*/*.png'))), 8)
            summary = json.loads((root / 'out' / 'summary.json').read_text())
            self.assertEqual(set(summary['groups']['observation']), {'base', 'correct', 'off', 'zero_rgb'})
            self.assertTrue(json.loads((root / 'out' / 'done.json').read_text())['done'])


if __name__ == '__main__':
    unittest.main()
