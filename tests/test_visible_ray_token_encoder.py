import sys
import unittest
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from models.KITTI_geo_ldm.lidar_condition_model import (  # noqa: E402
    LidarRayDepthSemanticTokenEncoder,
    LidarVisibleRaySemanticTokenEncoder,
)


def make_encoder():
    return LidarVisibleRaySemanticTokenEncoder(
        point_in_channels=10,
        hidden_channels=4,
        token_dim=6,
        token_grid=(2, 2),
        image_size=(4, 4),
        use_evidence_maps=False,
        token_output_norm="none",
        fixed_coord_pos_scale=0.0,
        point_feature_dim=3,
    )


def test_v2_keeps_v1_class_available_without_depth_embedding_in_v2():
    encoder = make_encoder()
    v1_encoder = LidarRayDepthSemanticTokenEncoder(
        ray_depth_bins=4,
        point_in_channels=10,
        hidden_channels=4,
        token_dim=6,
        token_grid=(2, 2),
        image_size=(4, 4),
        use_evidence_maps=False,
        point_feature_dim=3,
    )

    assert "depth_bin_embed" not in encoder.state_dict()
    assert v1_encoder.state_dict()["depth_bin_embed"].shape == (1, 4, 1, 1, 4)


def test_visible_ray_encoder_emits_one_token_per_patch():
    encoder = make_encoder()
    lidar_cond = torch.zeros(2, 10, 4, 4)
    features = torch.randn(2, 3, 1, 2, 2)
    mask = torch.ones(2, 1, 1, 2, 2)

    tokens = encoder(lidar_cond, lidar_ray_features=features, lidar_ray_features_mask=mask)

    assert tokens.shape == (2, 4, 6)
    assert torch.isfinite(tokens).all()


def test_masked_patch_feature_cannot_change_tokens():
    torch.manual_seed(7)
    encoder = make_encoder().eval()
    lidar_cond = torch.zeros(1, 10, 4, 4)
    feature_a = torch.randn(1, 3, 1, 2, 2)
    feature_b = feature_a.clone()
    feature_b[:, :, :, 0, 0] = 1000.0
    mask = torch.ones(1, 1, 1, 2, 2)
    mask[:, :, :, 0, 0] = 0.0

    output_a = encoder(lidar_cond, lidar_ray_features=feature_a, lidar_ray_features_mask=mask)
    output_b = encoder(lidar_cond, lidar_ray_features=feature_b, lidar_ray_features_mask=mask)

    assert torch.allclose(output_a, output_b)


def test_non_singleton_cache_plane_is_rejected():
    encoder = make_encoder()
    lidar_cond = torch.zeros(1, 10, 4, 4)
    features = torch.randn(1, 3, 2, 2, 2)

    with unittest.TestCase().assertRaises(ValueError):
        encoder(lidar_cond, lidar_ray_features=features)


def test_visible_ray_encoder_has_finite_feature_gradients():
    encoder = make_encoder()
    lidar_cond = torch.zeros(1, 10, 4, 4)
    features = torch.randn(1, 3, 1, 2, 2, requires_grad=True)
    mask = torch.ones(1, 1, 1, 2, 2)

    encoder(lidar_cond, lidar_ray_features=features, lidar_ray_features_mask=mask).square().mean().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0.0


def load_tests(loader, tests, pattern):
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    return unittest.TestSuite(unittest.FunctionTestCase(function) for function in functions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
