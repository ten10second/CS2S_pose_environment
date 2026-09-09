import unittest

import numpy as np
import torch

from tools.ar_dyn_utils import consecutive_rows, seed_step_noise


def mask(height, width, cells):
    result = torch.zeros((height, width), dtype=torch.bool)
    for row, col in cells:
        result[row, col] = True
    return result


def base_latent(height=4, width=5, channels=3):
    values = torch.arange(channels * height * width, dtype=torch.float32)
    return values.reshape(1, channels, height, width)


class ComposeHistoryTest(unittest.TestCase):
    def test_returns_true_for_next_frame_in_same_sequence(self):
        previous = {"drive": "0001", "date": "2011_09_26", "frame_index": "7"}
        current = {"drive": "0001", "date": "2011_09_26", "frame_index": "8"}

        self.assertTrue(consecutive_rows(previous, current))

    def test_returns_false_when_frame_has_gap(self):
        previous = {"drive": "0001", "date": "2011_09_26", "frame_index": 7}
        current = {"drive": "0001", "date": "2011_09_26", "frame_index": 9}

        self.assertFalse(consecutive_rows(previous, current))

    def test_returns_false_when_date_changes(self):
        previous = {"drive": "0001", "date": "2011_09_26", "frame_index": 7}
        current = {"drive": "0001", "date": "2011_09_27", "frame_index": 8}

        self.assertFalse(consecutive_rows(previous, current))


class SeedStepNoiseTest(unittest.TestCase):
    def test_generates_same_noise_for_same_seed(self):
        first = [torch.empty((1, 2, 2), dtype=torch.float32),
                 torch.empty((1, 1, 3), dtype=torch.float32)]
        second = [torch.empty_like(first[0]), torch.empty_like(first[1])]

        seed_step_noise(first, 1234)
        seed_step_noise(second, 1234)

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))

    def test_does_not_mutate_global_torch_rng_state(self):
        torch.manual_seed(99)
        expected = torch.randn((2, 3))
        torch.manual_seed(99)
        noise_bank = [torch.empty((1, 2, 2), dtype=torch.float32)]

        seed_step_noise(noise_bank, 1234)
        actual = torch.randn((2, 3))

        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
