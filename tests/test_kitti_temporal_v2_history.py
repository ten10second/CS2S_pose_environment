"""Unit tests for temporal-history v2 (route-history) — P1/P3 acceptance.

Run: python -B -m unittest discover -s tests -p 'test_kitti_temporal_v2_history.py' -v
"""
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ldm.modules.temporal_history_attention import (  # noqa: E402
    HistoryCrossAttention,
    HistoryLatentEncoder,
)
from temporal_history import (  # noqa: E402
    HistoryState,
    build_payload,
    should_use_history,
)


class TestHistoryLatentEncoder(unittest.TestCase):
    def test_shape_and_grid(self):
        enc = HistoryLatentEncoder(out_dim=256, grid=(16, 64))
        z = torch.randn(2, 4, 16, 64)
        tokens = enc(z)
        self.assertEqual(tuple(tokens.shape), (2, 1024, 256))

    def test_null_tokens_batch(self):
        enc = HistoryLatentEncoder(out_dim=256)
        t = enc.null_tokens(3)
        self.assertEqual(tuple(t.shape), (3, 1, 256))
        self.assertTrue(t.requires_grad)  # learned parameter


class TestHistoryCrossAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.attn = HistoryCrossAttention(dim=128, history_dim=256, heads=4, dim_head=32)
        self.enc = HistoryLatentEncoder(out_dim=256)
        self.x = torch.randn(1, 64, 128)
        self.cond = torch.randn(1, 64, 128)
        self.tokens = self.enc(torch.randn(1, 4, 16, 64))

    def test_zero_init_inert_with_history(self):
        """P3 acceptance: with the out projection zero-initialised, the stream
        output is exactly zero — enabling is a no-op on the frozen model."""
        out = self.attn(self.x, self.cond, self.tokens, has_history=True)
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_zero_init_inert_without_history(self):
        out = self.attn(self.x, self.cond, self.enc.null_tokens(1), has_history=False)
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_grads_flow_with_history(self):
        """P3 acceptance: after the zero-init step, gradients reach the out
        projection (and through it Q/K/V + encoder) because the upstream
        gradient at the addition site is nonzero in the real composition."""
        for p in self.attn.parameters():
            p.grad = None
        w = torch.eye(128)  # emulate downstream: loss reads x_base+fused+hist
        out = self.attn(self.x, self.cond, self.tokens, has_history=True)
        loss = (w @ out.transpose(-1, -2)).sum()  # any nonzero upstream
        loss.backward()
        self.assertIsNotNone(self.attn.to_out.weight.grad)
        self.assertGreater(float(self.attn.to_out.weight.grad.abs().sum()), 0)
        self.assertIsNotNone(self.attn.to_q.weight.grad)

    def test_no_history_keeps_params_in_graph(self):
        """P4-04: the null-token path must produce (possibly zero) grads for
        every attention parameter — never None — so DDP stays consistent."""
        for p in self.attn.parameters():
            p.grad = None
        out = self.attn(self.x, self.cond, self.enc.null_tokens(1), has_history=False)
        (out * 1.0).sum().backward()  # upstream 0 → zero grads but present
        none_grads = [n for n, p in self.attn.named_parameters() if p.grad is None]
        self.assertEqual(none_grads, [])

    def test_null_value_is_fixed_zero(self):
        """P3-04: mass on the null reference contributes exactly zero, so
        'reject history' is exactly 'no history'."""
        self.assertTrue(torch.all(self.attn.null_value == 0))
        self.assertFalse(self.attn.null_value.requires_grad)

    def test_null_fraction_recorded(self):
        out = self.attn(self.x, self.cond, self.tokens, has_history=True)
        self.assertIsNotNone(self.attn.last_null_frac)
        self.assertGreaterEqual(self.attn.last_null_frac, 0.0)
        self.assertLessEqual(self.attn.last_null_frac, 1.0)


class TestHistoryStateBoundary(unittest.TestCase):
    def test_consecutive_same_drive(self):
        prev = HistoryState("d20", 100, None)
        self.assertTrue(should_use_history(prev, "d20", 101))

    def test_first_frame(self):
        self.assertFalse(should_use_history(None, "d20", 100))

    def test_gap(self):
        prev = HistoryState("d20", 100, None)
        self.assertFalse(should_use_history(prev, "d20", 105))

    def test_drive_change(self):
        prev = HistoryState("d20", 100, None)
        self.assertFalse(should_use_history(prev, "d27", 101))


class TestPayloadBuilder(unittest.TestCase):
    def test_has_history(self):
        enc = HistoryLatentEncoder(out_dim=64)
        tokens = torch.randn(1, 8, 64)
        p = build_payload(enc, tokens, True)
        self.assertTrue(p["has_history"])
        self.assertIs(p["history_tokens"], tokens)

    def test_no_history_routes_null_token(self):
        """P4-04: no-history payloads still carry real tensors through the
        K/V weights — the DDP-safe explicit empty-history path."""
        enc = HistoryLatentEncoder(out_dim=64)
        p = build_payload(enc, None, False)
        self.assertFalse(p["has_history"])
        self.assertEqual(tuple(p["history_tokens"].shape), (1, 1, 64))
        self.assertTrue(p["history_tokens"].requires_grad)


if __name__ == "__main__":
    unittest.main()
