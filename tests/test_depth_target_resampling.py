import ast
from pathlib import Path

import unittest
import torch
import torch.nn.functional as F


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "KITTI_geo_ldm_diffusion"
    / "latent_diffusion.py"
)


def load_depth_helpers():
    tree = ast.parse(SOURCE.read_text())
    wanted = {
        "_as_nchw_depth_tensor",
        "resize_masked_lidar_depth",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "F": F,
        "ValueError": ValueError,
        "LIDAR_DEPTH_RESAMPLE_MODES": {"legacy_nearest", "masked_area"},
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["resize_masked_lidar_depth"]


resize_masked_lidar_depth = load_depth_helpers()


def masked_log_l1(depth_pred, depth_target, depth_mask, eps=1e-3):
    raw = (torch.log(depth_pred.clamp_min(eps)) - torch.log(depth_target.clamp_min(eps))).abs()
    masked = raw * depth_mask
    denom = depth_mask.mean(dim=[1, 2, 3]).clamp_min(1e-6)
    return (masked.mean(dim=[1, 2, 3]) / denom).mean()


def test_legacy_nearest_reproduces_missed_valid_target():
    depth = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros_like(depth)
    depth[..., 1, 1] = 0.5
    mask[..., 1, 1] = 1.0

    target, target_support = resize_masked_lidar_depth(
        depth,
        mask,
        size=(1, 1),
        mode="legacy_nearest",
    )

    assert target_support is None
    assert target.item() == 0.0
    assert F.interpolate(mask, size=(1, 1), mode="area").item() > 0.0


def test_masked_area_keeps_non_nearest_hit_depth():
    depth = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros_like(depth)
    depth[..., 1, 1] = 0.5
    mask[..., 1, 1] = 1.0

    target, support = resize_masked_lidar_depth(depth, mask, size=(1, 1), mode="masked_area")

    assert torch.allclose(target, torch.tensor([[[[0.5]]]]))
    assert support.item() > 0.0


def test_empty_cells_have_zero_support_and_zero_target():
    depth = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros_like(depth)

    target, support = resize_masked_lidar_depth(depth, mask, size=(2, 2), mode="masked_area")

    assert torch.count_nonzero(target) == 0
    assert torch.count_nonzero(support) == 0


def test_constant_depth_is_preserved_under_resizing():
    depth = torch.full((1, 1, 8, 8), 0.375)
    mask = torch.ones_like(depth)

    target, support = resize_masked_lidar_depth(depth, mask, size=(2, 4), mode="masked_area")

    assert torch.allclose(target, torch.full((1, 1, 2, 4), 0.375))
    assert torch.allclose(support, torch.ones(1, 1, 2, 4))


def test_masked_area_averages_only_valid_depths():
    depth = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros_like(depth)
    depth[..., 0, 0] = 0.2
    depth[..., 0, 1] = 0.6
    depth[..., 1, 0] = 0.9
    mask[..., 0, 0] = 1.0
    mask[..., 0, 1] = 1.0
    mask[..., 1, 0] = 0.0

    target, support = resize_masked_lidar_depth(depth, mask, size=(2, 2), mode="masked_area")

    assert torch.allclose(target[..., 0, 0], torch.tensor([[0.4]]))
    assert support[..., 0, 0].item() > 0.0
    assert target[..., 1, 1].item() == 0.0
    assert support[..., 1, 1].item() == 0.0


def test_masked_area_supports_different_head_sizes():
    for size in [(1, 1), (2, 8), (3, 5)]:
        depth = torch.rand(2, 1, 16, 64)
        mask = (torch.rand(2, 1, 16, 64) > 0.7).float()
    
        target, support = resize_masked_lidar_depth(depth, mask, size=size, mode="masked_area")
    
        assert target.shape[-2:] == size
        assert support.shape[-2:] == size
        assert torch.isfinite(target).all()
        assert torch.isfinite(support).all()


def test_masked_area_loss_and_gradient_are_finite():
    depth = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros_like(depth)
    depth[..., 1, 1] = 0.5
    mask[..., 1, 1] = 1.0
    pred = torch.full((1, 1, 1, 1), 0.25, requires_grad=True)

    target, support = resize_masked_lidar_depth(depth, mask, size=pred.shape[-2:], mode="masked_area")
    loss_mask = F.interpolate(mask, size=pred.shape[-2:], mode="area") * (support > 0).float()
    loss = masked_log_l1(pred, target.clamp_min(1e-6), loss_mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0.0


def test_invalid_resample_mode_raises():
    depth = torch.ones(1, 1, 2, 2)
    mask = torch.ones_like(depth)

    with unittest.TestCase().assertRaises(ValueError):
        resize_masked_lidar_depth(depth, mask, size=(1, 1), mode="unknown")


def test_fix_preserves_original_loss_weights():
    mask = torch.zeros(1, 1, 16, 64)
    mask[..., 1, 1] = 1
    mask[..., 3, 5] = 1
    mask[..., 15, 63] = 1
    _, support = resize_masked_lidar_depth(mask * .5, mask, (2, 8))
    original_weight = F.interpolate(mask, size=(2, 8), mode="area")
    assert torch.equal(original_weight, original_weight * (support > 0).float())


def test_empty_loss_has_zero_gradient():
    target, support = resize_masked_lidar_depth(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), (1, 1))
    pred = torch.full((1, 1, 1, 1), .25, requires_grad=True)
    loss = masked_log_l1(pred, target, support)
    loss.backward()
    assert loss.item() == 0
    assert pred.grad.item() == 0


def test_valid_depth_correction_reverses_wrong_gradient_direction():
    depth = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros_like(depth)
    depth[..., 1, 1] = .5
    mask[..., 1, 1] = 1
    weight = F.interpolate(mask, size=(1, 1), mode="area")
    gradients = []
    for mode in ('legacy_nearest', 'masked_area'):
        target, _ = resize_masked_lidar_depth(depth, mask, (1, 1), mode)
        pred = torch.full((1, 1, 1, 1), .25, requires_grad=True)
        masked_log_l1(pred, target, weight).backward()
        gradients.append(pred.grad.item())
    assert gradients[0] > 0  # Gradient descent wrongly drives depth down.
    assert gradients[1] < 0  # Correct 0.5 target drives prediction upward.


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn))


if __name__ == "__main__":
    unittest.main(verbosity=2)
