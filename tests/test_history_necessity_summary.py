"""Verdict logic for the history-necessity probe summary."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools import summarize_history_necessity as shn  # noqa: E402


def payload(pair_index, on_benefit, blind_benefit, raw, read, eps_factor=0.6,
            residual=0.004):
    cells = []
    for is_blind, benefit in ((False, on_benefit), (True, blind_benefit)):
        cells.append({
            "t": 250,
            "satellite_blind": is_blind,
            "benefit_disabled_minus_correct": benefit,
            "benefit_disabled_minus_correct_eps": benefit * eps_factor,
            "benefit_disabled_minus_wrong_geometry": benefit * 1.1,
            "benefit_wrong_history_minus_correct": benefit * 0.9,
        })
    return {
        "pair_index": pair_index,
        "coverage": {
            "raw_fraction_16x64": raw,
            "effective_fraction_at_block": read,
            "null_fraction_at_block": 1.0 - read,
            "residual_to_condition": residual,
        },
        "timesteps": cells,
    }


class VerdictTest(unittest.TestCase):
    def judge(self, payloads, min_benefit=0.002):
        rows, coverage = shn.collect(payloads)
        lines, cause = shn.verdict(rows, coverage, min_benefit)
        return "\n".join(lines), cause

    def test_positive_blind_benefit_in_every_pair_means_redundant_condition(self):
        text, cause = self.judge([payload(i, 0.0002, 0.0081 + i * 1e-4, 0.44, 0.02)
                                  for i in range(5)])
        self.assertEqual(cause, "a")
        self.assertIn("REDUNDANT CONDITION", text)

    def test_stage_f_shape_is_named_neutral_residual(self):
        """The measured Stage F shape: benefit unstable in BOTH arms, healthy
        coverage, large residual. That is (d), not (c)."""
        pairs = [payload(0, 0.0021, 0.0143, 0.44, 0.21, residual=0.671),
                 payload(1, -0.0018, -0.0040, 0.44, 0.18, residual=0.428),
                 payload(2, 0.0020, 0.0070, 0.44, 0.16, residual=0.600),
                 payload(3, -0.0011, -0.0030, 0.44, 0.12, residual=0.520),
                 payload(4, -0.0009, -0.0010, 0.44, 0.11, residual=0.640)]
        text, cause = self.judge(pairs)
        self.assertEqual(cause, "d")
        self.assertIn("NEUTRAL RESIDUAL", text)
        self.assertIn("positive 2/5", text)

    def test_zeroing_the_satellite_is_what_decides_redundancy(self):
        # blind positive, on flat -> (a); both flat -> not (a)
        _text, cause = self.judge([payload(i, -0.001, 0.010, 0.44, 0.30)
                                   for i in range(4)])
        self.assertEqual(cause, "a")
        _text, cause = self.judge([payload(i, -0.001, -0.001, 0.44, 0.30, residual=0.5)
                                   for i in range(4)])
        self.assertEqual(cause, "d")

    def test_a_split_sign_is_called_out_as_no_consistent_benefit(self):
        pairs = [payload(0, -0.001, 0.010, 0.44, 0.30, residual=0.5),
                 payload(1, 0.010, -0.001, 0.44, 0.30, residual=0.5),
                 payload(2, -0.001, 0.010, 0.44, 0.30, residual=0.5),
                 payload(3, 0.010, -0.001, 0.44, 0.30, residual=0.5)]
        text, cause = self.judge(pairs)
        self.assertEqual(cause, "d")
        self.assertIn("sign is split", text)

    def test_no_benefit_with_collapsed_mask_means_mask_collapse(self):
        text, cause = self.judge([payload(i, -1e-5, -3e-5, 0.44, 0.004)
                                  for i in range(4)])
        self.assertEqual(cause, "b")
        self.assertIn("MASK COLLAPSE", text)
        self.assertIn("grid=(8,32)", text)

    def test_no_benefit_with_a_tiny_residual_means_inert_readout(self):
        text, cause = self.judge([payload(i, -2e-5, -5e-5, 0.44, 0.38, residual=0.01)
                                  for i in range(4)])
        self.assertEqual(cause, "c")
        self.assertIn("INERT READOUT", text)

    def test_direction_is_disabled_minus_correct(self):
        # history helping must produce a POSITIVE benefit
        _text, cause = self.judge([payload(0, 0.0, 0.05, 0.44, 0.30)])
        self.assertEqual(cause, "a")
        _text, cause = self.judge([payload(0, 0.0, -0.05, 0.44, 0.30, residual=0.01)])
        self.assertEqual(cause, "c")

    def test_threshold_is_configurable(self):
        high = [payload(i, 0.0, 0.0015, 0.44, 0.30) for i in range(3)]
        _text, cause = self.judge(high, min_benefit=0.001)
        self.assertEqual(cause, "a")
        high = [payload(i, 0.0, 0.0015, 0.44, 0.30, residual=0.5) for i in range(3)]
        _text, cause = self.judge(high, min_benefit=0.002)
        self.assertEqual(cause, "d")

    def test_missing_blind_arm_is_reported_not_guessed(self):
        only_on = payload(0, 0.0, 0.0, 0.44, 0.30)
        only_on["timesteps"] = [c for c in only_on["timesteps"] if not c["satellite_blind"]]
        text, cause = self.judge([only_on])
        self.assertIsNone(cause)
        self.assertIn("no satellite-zeroed arm", text)

    def test_main_reads_a_directory_and_prints_the_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, blind in ((0, 0.008), (1, 0.009)):
                (Path(tmp) / f"necessity_train_pair{index}.json").write_text(
                    json.dumps(payload(index, 0.0001, blind, 0.43, 0.02))
                )
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                sys.argv = ["summarize_history_necessity.py", "--dir", tmp]
                shn.main()
            output = buffer.getvalue()
        self.assertIn("pairs: 2", output)
        self.assertIn("VERDICT", output)
        self.assertIn("REDUNDANT CONDITION", output)


if __name__ == "__main__":
    unittest.main()
