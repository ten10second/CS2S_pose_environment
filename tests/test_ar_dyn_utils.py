import unittest

import numpy as np
import torch

from tools.ar_dyn_utils import compose_history, consecutive_rows, seed_step_noise


def mask(height, width, cells):
    result = torch.zeros((height, width), dtype=torch.bool)
    for row, col in cells:
        result[row, col] = True
    return result


def base_latent(height=4, width=5, channels=3):
    values = torch.arange(channels * height * width, dtype=torch.float32)
    return values.reshape(1, channels, height, width)


class ComposeHistoryTest(unittest.TestCase):
    def test_carries_translated_object_values_when_refresh_overlaps_destination(self):
        previous = base_latent()
        fresh_noise = torch.full_like(previous, -100.0)
        source = mask(4, 5, [(1, 1), (1, 2), (2, 1), (2, 2)])
        destination = mask(4, 5, [(1, 2), (1, 3), (2, 2), (2, 3)])
        refresh = destination.clone()

        result, stats = compose_history(
            previous,
            [{
                "cell_mask": destination,
                "prev_cell_mask": source,
                "d_lat": np.array([1.0, 0.0], dtype=np.float32),
                "depth": 5.0,
            }],
            refresh,
            fresh_noise,
        )

        self.assertTrue(torch.equal(result[:, :, 1, 2], previous[:, :, 1, 1]))
        self.assertTrue(torch.equal(result[:, :, 1, 3], previous[:, :, 1, 2]))
        self.assertTrue(torch.equal(result[:, :, 2, 2], previous[:, :, 2, 1]))
        self.assertTrue(torch.equal(result[:, :, 2, 3], previous[:, :, 2, 2]))
        self.assertEqual(stats["transport_protected_cells"], 4)

    def test_missing_near_history_does_not_reveal_far_history(self):
        previous = base_latent()
        fresh = torch.full_like(previous, -9.0)
        destination = mask(4, 5, [(1, 1)])
        far = {"cell_mask": destination, "prev_cell_mask": mask(4, 5, [(1, 0)]),
               "d_lat": [1.0, 0.0], "depth": 9.0}
        near = {"cell_mask": destination, "prev_cell_mask": mask(4, 5, []),
                "d_lat": [0.0, 0.0], "depth": 3.0}
        result, stats = compose_history(previous, [far, near], destination, fresh)
        self.assertTrue(torch.equal(result[:, :, 1, 1], fresh[:, :, 1, 1]))
        self.assertEqual(stats["obj_transported"], 0)

    def test_fractional_shift_rejects_partially_supported_source(self):
        previous = base_latent()
        fresh = torch.full_like(previous, -9.0)
        destination = mask(4, 5, [(1, 2)])
        obj = {"cell_mask": destination, "prev_cell_mask": mask(4, 5, [(1, 1)]),
               "d_lat": [0.5, 0.0], "depth": 5.0}
        result, stats = compose_history(previous, [obj], destination, fresh)
        self.assertTrue(torch.equal(result[:, :, 1, 2], fresh[:, :, 1, 2]))
        self.assertEqual(stats["obj_transported"], 0)

    def test_refreshes_vacated_source_cells_after_transport(self):
        previous = base_latent()
        fresh_noise = torch.full_like(previous, -7.0)
        source = mask(4, 5, [(1, 1), (1, 2)])
        destination = mask(4, 5, [(1, 2), (1, 3)])

        result, _ = compose_history(
            previous,
            [{
                "cell_mask": destination,
                "prev_cell_mask": source,
                "d_lat": np.array([1.0, 0.0], dtype=np.float32),
                "depth": 5.0,
            }],
            torch.zeros((4, 5), dtype=torch.bool),
            fresh_noise,
        )

        self.assertTrue(torch.equal(result[:, :, 1, 1], fresh_noise[:, :, 1, 1]))

    def test_leaves_static_cells_unchanged(self):
        previous = base_latent()
        fresh_noise = torch.full_like(previous, -7.0)
        source = mask(4, 5, [(1, 1)])
        destination = mask(4, 5, [(1, 2)])

        result, _ = compose_history(
            previous,
            [{
                "cell_mask": destination,
                "prev_cell_mask": source,
                "d_lat": np.array([1.0, 0.0], dtype=np.float32),
                "depth": 5.0,
            }],
            torch.zeros((4, 5), dtype=torch.bool),
            fresh_noise,
        )

        self.assertTrue(torch.equal(result[:, :, 0, 0], previous[:, :, 0, 0]))

    def test_refreshes_destination_when_source_is_out_of_bounds(self):
        previous = base_latent()
        fresh_noise = torch.full_like(previous, -3.0)
        source = mask(4, 5, [(0, 0)])
        destination = mask(4, 5, [(0, 0)])

        result, stats = compose_history(
            previous,
            [{
                "cell_mask": destination,
                "prev_cell_mask": source,
                "d_lat": np.array([1.0, 0.0], dtype=np.float32),
                "depth": 5.0,
            }],
            torch.zeros((4, 5), dtype=torch.bool),
            fresh_noise,
        )

        self.assertTrue(torch.equal(result[:, :, 0, 0], fresh_noise[:, :, 0, 0]))
        self.assertEqual(stats["obj_transported"], 0)

    def test_refreshes_destination_when_object_depth_is_unsupported(self):
        previous = base_latent()
        fresh_noise = torch.full_like(previous, -3.0)
        source = mask(4, 5, [(1, 1)])
        destination = mask(4, 5, [(1, 2)])

        result, stats = compose_history(
            previous,
            [{
                "cell_mask": destination,
                "prev_cell_mask": source,
                "d_lat": np.array([1.0, 0.0], dtype=np.float32),
                "depth": 1.0,
            }],
            torch.zeros((4, 5), dtype=torch.bool),
            fresh_noise,
        )

        self.assertTrue(torch.equal(result[:, :, 1, 2], fresh_noise[:, :, 1, 2]))
        self.assertEqual(stats["obj_transported"], 0)

    def test_uses_nearer_object_for_overlapping_destinations(self):
        previous = base_latent()
        fresh_noise = torch.full_like(previous, -3.0)
        near_source = mask(4, 5, [(2, 2)])
        far_source = mask(4, 5, [(0, 0)])
        destination = mask(4, 5, [(1, 1)])

        result, _ = compose_history(
            previous,
            [
                {
                    "cell_mask": destination,
                    "prev_cell_mask": far_source,
                    "d_lat": np.array([1.0, 1.0], dtype=np.float32),
                    "depth": 9.0,
                },
                {
                    "cell_mask": destination,
                    "prev_cell_mask": near_source,
                    "d_lat": np.array([-1.0, -1.0], dtype=np.float32),
                    "depth": 3.0,
                },
            ],
            torch.zeros((4, 5), dtype=torch.bool),
            fresh_noise,
        )

        self.assertTrue(torch.equal(result[:, :, 1, 1], previous[:, :, 2, 2]))

    def test_accepts_plain_refresh_mask_when_no_objects_are_present(self):
        previous = base_latent()
        fresh_noise = torch.full_like(previous, -5.0)
        refresh = [[False, True, False, False, False],
                   [False, False, False, False, False],
                   [False, False, False, False, False],
                   [False, False, False, False, False]]

        result, stats = compose_history(previous, [], refresh, fresh_noise)

        self.assertTrue(torch.equal(result[:, :, 0, 1], fresh_noise[:, :, 0, 1]))
        self.assertEqual(stats["obj_matched"], 0)

    def test_does_not_transport_when_transport_is_disabled(self):
        previous = base_latent()
        fresh_noise = torch.full_like(previous, -11.0)
        source = mask(4, 5, [(1, 1)])
        destination = mask(4, 5, [(1, 2)])

        result, stats = compose_history(
            previous,
            [{
                "cell_mask": destination,
                "prev_cell_mask": source,
                "d_lat": np.array([1.0, 0.0], dtype=np.float32),
                "depth": 5.0,
            }],
            torch.zeros((4, 5), dtype=torch.bool),
            fresh_noise,
            transport=False,
        )

        self.assertTrue(torch.equal(result[:, :, 1, 2], fresh_noise[:, :, 1, 2]))
        self.assertEqual(stats["obj_transported"], 0)

    def test_rejects_previous_latent_without_single_batch_dimension(self):
        previous = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        fresh_noise = torch.zeros_like(previous)

        with self.assertRaisesRegex(ValueError, "single BCHW latent"):
            compose_history(previous, [], torch.zeros((4, 5), dtype=torch.bool), fresh_noise)

    def test_rejects_fresh_noise_with_different_shape(self):
        previous = base_latent()
        fresh_noise = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "fresh noise"):
            compose_history(previous, [], torch.zeros((4, 5), dtype=torch.bool), fresh_noise)

    def test_rejects_mask_with_wrong_spatial_shape(self):
        previous = base_latent()
        fresh_noise = torch.zeros_like(previous)

        with self.assertRaisesRegex(ValueError, "wrong spatial shape"):
            compose_history(previous, [], torch.zeros((3, 5), dtype=torch.bool), fresh_noise)


class ConsecutiveRowsTest(unittest.TestCase):
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
