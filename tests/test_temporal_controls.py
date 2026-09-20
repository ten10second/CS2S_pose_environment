from __future__ import annotations

import unittest

import numpy as np
import torch

from ldm.modules.temporal_pair_training import build_history
from tools.infer_temporal import prepare_history_from_payload, evaluation_device


def candidate_geometry(h=3, w=4, k=2):
    grid = np.zeros((h, w, k, 2), dtype=np.float32)
    grid[..., 0] = np.linspace(-0.75, 0.75, w)[None, :, None]
    grid[..., 1] = np.linspace(-0.5, 0.5, h)[:, None, None]
    sat = grid.copy(); sat[..., 0] *= 0.5
    return {
        'history_grid': grid,
        'sat_grid': sat,
        'valid': np.ones((h, w, k), dtype=bool),
        'sat_valid': np.ones((h, w, k), dtype=bool),
        'positions': np.zeros((h, w, k, 4), dtype=np.float32),
        'metrics': {'coverage': 1.0},
    }


class TemporalControlTests(unittest.TestCase):
    def test_evaluation_gpu_restriction_survives_visible_device_remapping(self):
        from unittest.mock import patch
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "5,4"}):
            self.assertEqual(evaluation_device("cuda:4"), torch.device("cuda:1"))
            self.assertEqual(evaluation_device("cuda:5"), torch.device("cuda:0"))
            with self.assertRaises(ValueError):
                evaluation_device("cuda:0")
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4"}):
            with self.assertRaises(ValueError):
                evaluation_device("cuda:5")

    def test_wrong_history_batch1_requires_explicit_distractor(self):
        latent = torch.randn(1, 4, 3, 4)
        history = build_history(latent, [candidate_geometry()])
        with self.assertRaises(ValueError):
            prepare_history_from_payload({'history': history}, torch.device('cpu'), 'wrong_history')
        wrong = torch.randn_like(latent)
        prepared, _ = prepare_history_from_payload({'history': history, 'wrong_history_latent': wrong}, torch.device('cpu'), 'wrong_history')
        self.assertTrue(torch.equal(prepared['latent'], wrong))
        self.assertTrue(torch.equal(prepared['valid'], history['valid']))

    def test_wrong_history_batch_rotation_requires_actual_different_latent(self):
        latent = torch.stack([torch.zeros(4, 3, 4), torch.ones(4, 3, 4)], 0)
        history = build_history(latent, [candidate_geometry(), candidate_geometry()])
        prepared, _ = prepare_history_from_payload({'history': history}, torch.device('cpu'), 'wrong_history')
        self.assertTrue(torch.equal(prepared['latent'][0], latent[1]))
        self.assertTrue(torch.equal(prepared['latent'][1], latent[0]))

    def test_wrong_geometry_changes_history_coordinates_only(self):
        latent = torch.randn(1, 4, 3, 4)
        history = build_history(latent, [candidate_geometry()])
        prepared, _ = prepare_history_from_payload({'history': history}, torch.device('cpu'), 'wrong_geometry')
        self.assertTrue(torch.equal(prepared['latent'], history['latent']))
        self.assertTrue(torch.equal(prepared['valid'], history['valid']))
        self.assertTrue(torch.equal(prepared['sat_grid'], history['sat_grid']))
        self.assertTrue(torch.equal(prepared['history_grid'][..., 1], history['history_grid'][..., 1]))
        self.assertTrue(torch.equal(prepared['history_grid'][..., 0], -history['history_grid'][..., 0]))

    def test_batched_history_geometry_shape_is_full_candidate_shape(self):
        latent = torch.randn(3, 4, 3, 4)
        history = build_history(latent, [candidate_geometry(k=3), candidate_geometry(k=3), candidate_geometry(k=3)])
        self.assertEqual(history['history_grid'].shape, (3, 3, 4, 3, 2))
        self.assertEqual(history['sat_grid'].shape, (3, 3, 4, 3, 2))
        self.assertEqual(history['valid'].shape, (3, 3, 4, 3))
        self.assertEqual(history['positions'].shape, (3, 3, 4, 3, 4))



if __name__ == '__main__':
    unittest.main()
