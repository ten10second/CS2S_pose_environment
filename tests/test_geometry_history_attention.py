import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ldm.modules.temporal_history_attention import GeometryHistoryAttention  # noqa: E402
from temporal_history import build_payload, enable_history_attention  # noqa: E402


def identity_grid(batch=1, height=4, width=4, device="cpu", dtype=torch.float32):
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()


class TestGeometryHistoryAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.attn = GeometryHistoryAttention(dim=32, history_dim=16, heads=2, dim_head=8)
        self.x = torch.randn(1, 16, 32)
        self.cond = torch.randn(1, 16, 32)
        self.tokens = torch.randn(1, 16, 16, requires_grad=True)
        self.grid = identity_grid()
        self.valid = torch.ones(1, 4, 4, dtype=torch.bool)

    def test_zero_init_is_inert_with_valid_geometry(self):
        out = self.attn(
            self.x,
            self.cond,
            self.tokens,
            has_history=True,
            history_grid=self.grid,
            history_valid=self.valid,
            query_hw=(4, 4),
            history_hw=(4, 4),
        )
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_resize_identity_grid_preserves_query_cell_centers(self):
        grid = identity_grid(batch=1, height=16, width=64)
        valid = torch.ones(1, 16, 64, dtype=torch.bool)
        resized, resized_valid = GeometryHistoryAttention._resize_grid(grid, valid, (8, 32))
        expected = identity_grid(batch=1, height=8, width=32)
        self.assertLess(float((resized - expected).abs().max()), 1e-7)
        self.assertTrue(torch.equal(resized_valid, torch.ones(1, 8, 32, dtype=torch.bool)))

    def test_resize_valid_is_conservative(self):
        grid = identity_grid(batch=1, height=16, width=64)
        valid = torch.ones(1, 16, 64, dtype=torch.bool)
        valid[0, 0, 0] = False
        _resized, resized_valid = GeometryHistoryAttention._resize_grid(grid, valid, (8, 32))
        self.assertFalse(bool(resized_valid[0, 0, 0]))
        self.assertTrue(bool(resized_valid[0, -1, -1]))

    def test_missing_geometry_with_history_fails_loudly(self):
        with self.assertRaises(ValueError):
            self.attn(
                self.x,
                self.cond,
                self.tokens,
                has_history=True,
                query_hw=(4, 4),
                history_hw=(4, 4),
            )

    def test_invalid_geometry_is_exact_zero_after_training(self):
        with torch.no_grad():
            self.attn.to_out.weight.fill_(0.25)
        out = self.attn(
            self.x,
            self.cond,
            self.tokens,
            has_history=True,
            history_grid=self.grid,
            history_valid=torch.zeros_like(self.valid),
            query_hw=(4, 4),
            history_hw=(4, 4),
        )
        self.assertEqual(float(out.abs().max()), 0.0)
        self.assertEqual(self.attn.last_valid_frac, 0.0)

    def test_no_history_keeps_all_parameters_in_graph(self):
        with torch.no_grad():
            self.attn.to_out.weight.normal_(std=0.02)
        out = self.attn(
            self.x,
            self.cond,
            self.tokens,
            has_history=False,
            history_grid=self.grid,
            history_valid=self.valid,
            query_hw=(4, 4),
            history_hw=(4, 4),
        )
        out.sum().backward()
        none_grads = [name for name, param in self.attn.named_parameters() if param.grad is None]
        self.assertEqual(none_grads, [])
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_gradients_flow_with_valid_geometry(self):
        with torch.no_grad():
            self.attn.to_out.weight.normal_(std=0.02)
        out = self.attn(
            self.x,
            self.cond,
            self.tokens,
            has_history=True,
            history_grid=self.grid,
            history_valid=self.valid,
            query_hw=(4, 4),
            history_hw=(4, 4),
        )
        out.square().mean().backward()
        none_grads = [name for name, param in self.attn.named_parameters() if param.grad is None]
        self.assertEqual(none_grads, [])
        self.assertIsNotNone(self.tokens.grad)
        self.assertGreater(float(self.tokens.grad.abs().sum()), 0.0)

    def test_coordinates_affect_local_readout(self):
        with torch.no_grad():
            self.attn.to_out.weight.normal_(std=0.02)
        out_a = self.attn(
            self.x,
            self.cond,
            self.tokens,
            has_history=True,
            history_grid=self.grid,
            history_valid=self.valid,
            query_hw=(4, 4),
            history_hw=(4, 4),
        )
        shifted = self.grid.clone()
        shifted[..., 0] = (shifted[..., 0] + 0.5).clamp(-1.0, 1.0)
        out_b = self.attn(
            self.x,
            self.cond,
            self.tokens,
            has_history=True,
            history_grid=shifted,
            history_valid=self.valid,
            query_hw=(4, 4),
            history_hw=(4, 4),
        )
        self.assertGreater(float((out_a - out_b).abs().max()), 1e-7)

    def test_cfg_batch_repetition(self):
        out = self.attn(
            self.x.repeat(2, 1, 1),
            self.cond.repeat(2, 1, 1),
            self.tokens,
            has_history=True,
            history_grid=self.grid,
            history_valid=self.valid,
            query_hw=(4, 4),
            history_hw=(4, 4),
        )
        self.assertEqual(tuple(out.shape), (2, 16, 32))

    def test_cfg_repeats_base_batch_order(self):
        with torch.no_grad():
            self.attn.to_out.weight.normal_(std=0.02)
        x = torch.zeros(2, 16, 32)
        cond = torch.zeros(2, 16, 32)
        base = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16) / 31.0
        tokens = torch.stack([base, -base], dim=0)
        grid = identity_grid(batch=2)
        valid = torch.ones(2, 4, 4, dtype=torch.bool)
        out = self.attn(
            x.repeat(2, 1, 1),
            cond.repeat(2, 1, 1),
            tokens,
            has_history=True,
            history_grid=grid,
            history_valid=valid,
            query_hw=(4, 4),
            history_hw=(4, 4),
        )
        self.assertLess(float((out[0] - out[2]).abs().max()), 1e-7)
        self.assertLess(float((out[1] - out[3]).abs().max()), 1e-7)
        self.assertGreater(float((out[0] - out[1]).abs().max()), 1e-7)

    def test_cfg_bad_batch_fails_loudly(self):
        with self.assertRaises(ValueError):
            self.attn(
                self.x.repeat(3, 1, 1),
                self.cond.repeat(3, 1, 1),
                self.tokens.repeat(2, 1, 1),
                has_history=True,
                history_grid=self.grid.repeat(2, 1, 1, 1),
                history_valid=self.valid.repeat(2, 1, 1),
                query_hw=(4, 4),
                history_hw=(4, 4),
            )

    def test_invalid_nan_grid_is_exact_zero_after_training(self):
        with torch.no_grad():
            self.attn.to_out.weight.fill_(0.25)
        bad_grid = self.grid.clone()
        bad_grid[:] = float("nan")
        out = self.attn(
            self.x,
            self.cond,
            self.tokens,
            has_history=True,
            history_grid=bad_grid,
            history_valid=self.valid,
            query_hw=(4, 4),
            history_hw=(4, 4),
        )
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(float(out.abs().max()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA autocast check requires GPU")
    def test_cuda_autocast_forward(self):
        attn = GeometryHistoryAttention(dim=32, history_dim=16, heads=2, dim_head=8).cuda()
        with torch.cuda.amp.autocast(True):
            out = attn(
                self.x.cuda(),
                self.cond.cuda(),
                self.tokens.detach().cuda(),
                has_history=True,
                history_grid=self.grid.cuda(),
                history_valid=self.valid.cuda(),
                query_hw=(4, 4),
                history_hw=(4, 4),
            )
        self.assertEqual(tuple(out.shape), (1, 16, 32))


class _FusionBlock(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.ray_fusion_mode = "ray_posterior"
        self.ray_posterior_fusion = SimpleNamespace(dim=dim)
        self.history_attn = None
        self.temporal_hub = None


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FusionBlock(), _FusionBlock(), _FusionBlock()])


class TestGeometryHistoryPlumbing(unittest.TestCase):
    def test_enable_geometry_selects_explicit_blocks(self):
        model = _TinyModel()
        hub, encoder, blocks = enable_history_attention(model, geometry=True, block_indices=(1,))
        self.assertEqual(len(blocks), 1)
        self.assertIs(blocks[0], model.blocks[1])
        self.assertIs(blocks[0].temporal_hub, hub)
        self.assertTrue(getattr(blocks[0].history_attn, "uses_geometry", False))
        self.assertEqual(blocks[0].history_block_index, 1)
        self.assertEqual(blocks[0].history_attn_index, 0)
        self.assertEqual(encoder.out_dim, 64)

    def test_enable_geometry_rejects_bad_block_index(self):
        with self.assertRaises(ValueError):
            enable_history_attention(_TinyModel(), geometry=True, block_indices=(3,))

    def test_enable_geometry_rejects_empty_or_duplicate_blocks(self):
        with self.assertRaises(ValueError):
            enable_history_attention(_TinyModel(), geometry=True, block_indices=())
        with self.assertRaises(ValueError):
            enable_history_attention(_TinyModel(), geometry=True, block_indices=(1, 1))

    def test_enable_geometry_preserves_explicit_hyperparameters(self):
        model = _TinyModel()
        _hub, encoder, blocks = enable_history_attention(
            model,
            geometry=True,
            block_indices=(0,),
            history_dim=128,
            heads=8,
            dim_head=16,
        )
        self.assertEqual(encoder.out_dim, 128)
        self.assertEqual(blocks[0].history_attn.heads, 8)
        self.assertEqual(blocks[0].history_attn.dim_head, 16)

    def test_payload_validates_geometry_fields(self):
        model = _TinyModel()
        _hub, encoder, _blocks = enable_history_attention(model, geometry=True, block_indices=(0,))
        tokens = torch.randn(1, 16, encoder.out_dim)
        grid = identity_grid(height=4, width=4)
        valid = torch.ones(1, 4, 4, dtype=torch.bool)
        payload = build_payload(encoder, tokens, True, history_grid=grid, history_valid=valid)
        self.assertIs(payload["history_grid"], grid)
        self.assertTrue(torch.equal(payload["history_valid"], valid))
        self.assertEqual(payload["history_hw"], encoder.grid)
        with self.assertRaises(ValueError):
            build_payload(encoder, tokens, True, history_grid=grid)


if __name__ == "__main__":
    unittest.main()
