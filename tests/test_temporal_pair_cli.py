from __future__ import annotations

import subprocess
import sys
import unittest


class TemporalPairCliTests(unittest.TestCase):
    def test_cli_help(self):
        scripts = [
            'tools/train_centered_decoder_full.py',
            'tools/train_centered_decoder_probe.py',
            'tools/prepare_full_centered_history.py',
            'tools/prepare_centered_history_data.py',
            'tools/cache_centered_conditions.py',
            'tools/summarize_centered_decoder_probe.py',
            'tools/train_temporal_pairs.py',
            'tools/infer_temporal.py',
            'tools/eval_temporal_gt_pairs.py',
        ]
        for script in scripts:
            with self.subTest(script=script):
                proc = subprocess.run([sys.executable, script, '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn('usage:', proc.stdout)


if __name__ == '__main__':
    unittest.main()
