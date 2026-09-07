"""Unit tests for temporal-history v2 (route-history) — P1/P3 acceptance.

Run: python -B -m unittest discover -s tests -p 'test_kitti_temporal_v2_history.py' -v
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    history_latent_from_gt,
    should_use_history,
)
from tools.train_kitti_temporal_v2 import (  # noqa: E402
    TemporalHistoryTrainingStepModule,
    probe_history_effect,
    require_resume_payload,
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
        self.assertIsNone(self.attn.to_out.bias)

    def test_zero_null_value_stays_zero_after_output_weight_changes(self):
        with torch.no_grad():
            self.attn.to_out.weight.fill_(1.0)
        projected = self.attn.to_out(torch.zeros(1, 2, self.attn.inner))
        self.assertEqual(float(projected.abs().max()), 0.0)

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

    def test_no_history_preserves_supplied_graph_tokens(self):
        enc = HistoryLatentEncoder(hidden=8, out_dim=8)
        body = enc(torch.zeros(1, 4, 16, 64))
        tokens = enc.null_tokens(1) + body.mean(dim=1, keepdim=True) * 0.0
        p = build_payload(enc, tokens, False)
        self.assertIs(p["history_tokens"], tokens)


class _PayloadHub:
    def __init__(self):
        self.payload = None

    def set(self, payload):
        self.payload = payload

    def clear(self):
        self.payload = None


class _PayloadReadingModel(torch.nn.Module):
    def __init__(self, hub):
        super().__init__()
        self.hub = hub

    def training_step(self, batch, _batch_idx):
        return self.hub.payload["history_tokens"].sum()


class TestTemporalHistoryTrainingWrapper(unittest.TestCase):
    def setUp(self):
        self.hub = _PayloadHub()
        self.encoder = HistoryLatentEncoder(hidden=8, out_dim=8)
        self.wrapper = TemporalHistoryTrainingStepModule(
            _PayloadReadingModel(self.hub), self.encoder, self.hub
        )
        self.cur = {"grd_left_imgs": torch.zeros(1, 3, 128, 512)}

    def _none_grads(self):
        return [n for n, p in self.encoder.named_parameters() if p.grad is None]

    def test_wrapper_owns_history_encoder(self):
        names = dict(self.wrapper.named_parameters())
        self.assertIn("history_encoder.in_proj.weight", names)

    def test_history_step_reaches_encoder_and_null_token(self):
        z = torch.randn(1, 4, 16, 64)
        self.wrapper(self.cur, z, True).backward()
        self.assertEqual(self._none_grads(), [])
        self.assertIsNone(self.hub.payload)

    def test_no_history_step_keeps_entire_encoder_in_graph(self):
        self.wrapper(self.cur, None, False).backward()
        self.assertEqual(self._none_grads(), [])
        self.assertIsNone(self.hub.payload)


class TestFrozenVaeBoundary(unittest.TestCase):
    def test_history_latent_is_detached_before_trainable_encoder(self):
        class _Posterior:
            def sample(self):
                return torch.ones(1, 4, 16, 64, requires_grad=True)

        model = SimpleNamespace(
            pre_AE_model=SimpleNamespace(encode=lambda _x: _Posterior()),
            scale_factor=0.5,
        )
        latent = history_latent_from_gt(
            model, {"grd_left_imgs": torch.zeros(1, 3, 128, 512)}
        )
        self.assertFalse(latent.requires_grad)
        enc = HistoryLatentEncoder(hidden=8, out_dim=8)
        enc(latent).sum().backward()
        body = [p for n, p in enc.named_parameters() if n != "null_token"]
        self.assertTrue(all(p.grad is not None for p in body))


class TestStrictResume(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(ckpt="/tmp/base.pt")
        self.blocks = [SimpleNamespace(), SimpleNamespace()]
        self.payload = {
            "step": 10,
            "history_encoder": {},
            "history_attn": {"0": {}, "1": {}},
            "optimizer": {},
            "mode": "temporal_v2_history_phaseA",
            "base_ckpt": "/tmp/base.pt",
        }

    def test_complete_payload_is_accepted(self):
        require_resume_payload(self.payload, self.args, self.blocks)

    def test_missing_state_fails_loudly(self):
        payload = dict(self.payload)
        del payload["history_encoder"]
        with self.assertRaises(KeyError):
            require_resume_payload(payload, self.args, self.blocks)

    def test_history_block_mismatch_fails_loudly(self):
        payload = dict(self.payload)
        payload["history_attn"] = {"0": {}}
        with self.assertRaises(ValueError):
            require_resume_payload(payload, self.args, self.blocks)


class _ProbeModel(torch.nn.Module):
    def forward(self, _cur, history_latent=None, has_history=True):
        random_term = torch.rand(())
        history_term = history_latent.mean() if has_history else torch.zeros(())
        return random_term + history_term


class TestHistoryProbe(unittest.TestCase):
    def test_all_modes_share_identical_random_draw(self):
        correct = torch.zeros(1, 4, 1, 1)
        wrong = torch.ones(1, 4, 1, 1)
        losses = probe_history_effect(
            _ProbeModel(), {}, correct, seed=123, amp=False, wrong_history_latent=wrong
        )
        self.assertEqual(float(losses[0]), float(losses[1]))
        self.assertAlmostEqual(float(losses[2] - losses[0]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
