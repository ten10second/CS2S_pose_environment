import torch
import unittest

from models.KITTI_geo_ldm.txt_control import Boost_Sat2Den_ddpm


class OldTokenEncoder(torch.nn.Module):
    def forward(
        self,
        lidar_cond,
        raw_lidar_cond=None,
        range_img=None,
        range_mask=None,
        camera_k=None,
        camera_to_lidar=None,
        lidar_points=None,
        lidar_points_mask=None,
        lidar_point_features=None,
        lidar_point_features_mask=None,
        lidar_ray_features=None,
        lidar_ray_features_mask=None,
    ):
        return {"old": lidar_cond, "raw": raw_lidar_cond}


class PixelEncoder(torch.nn.Module):
    uses_pixel_features = True

    def forward(self, lidar_cond, raw_lidar_cond=None, lidar_pixel_features=None, lidar_pixel_features_mask=None, **kwargs):
        return {
            "features": lidar_pixel_features,
            "mask": lidar_pixel_features_mask,
            "unexpected": kwargs.get("lidar_pixel_features_available"),
        }


def make_model(encoder):
    model = Boost_Sat2Den_ddpm.__new__(Boost_Sat2Den_ddpm)
    torch.nn.Module.__init__(model)
    model.lidar_context_model = encoder
    model.lidar_geom_mode = "raw"
    return model


def test_old_token_encoder_does_not_receive_pixel_kwargs():
    model = make_model(OldTokenEncoder())
    lidar_cond = torch.zeros(1, 10, 4, 4)

    result = model.make_lidar_context(
        lidar_cond,
        lidar_pixel_features=torch.ones(1, 576, 4, 4),
        lidar_pixel_features_mask=torch.ones(1, 1, 4, 4),
        lidar_pixel_features_available=torch.ones(1),
    )

    assert result["old"] is lidar_cond


def test_pixel_encoder_receives_pixel_kwargs():
    model = make_model(PixelEncoder())
    lidar_cond = torch.zeros(1, 10, 4, 4)
    features = torch.ones(1, 576, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    available = torch.zeros(1)

    result = model.make_lidar_context(
        lidar_cond,
        lidar_pixel_features=features,
        lidar_pixel_features_mask=mask,
        lidar_pixel_features_available=available,
    )

    assert result["features"] is features
    assert result["mask"] is mask
    assert result["unexpected"] is available


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith("test_"))


def test_training_and_sampling_report_spatial_lidar_messages():
    from types import SimpleNamespace
    from models.KITTI_geo_ldm.lidar_pixel_condition import LidarSpatialResidual
    from tools.train_kitti_raea import lidar_attention_stats as train_stats
    from tools.generate_kitti_raea_samples import lidar_attention_stats as sample_stats

    residual = LidarSpatialResidual(2, 4)
    with torch.no_grad():
        residual.projection.weight.fill_(0.25)
        residual(torch.ones(1, 4, 2, 2), torch.ones(1, 2, 2, 2), torch.ones(1, 1, 2, 2))
    model = SimpleNamespace(DDPM=SimpleNamespace(denoise_model=torch.nn.Sequential(residual)))
    for stats_fn in (train_stats, sample_stats):
        stats = stats_fn(model)
        assert stats["lidar_spatial_modules"] == 1
        assert stats["lidar_spatial_message_ratio_mean"] > 0
        assert stats["lidar_spatial_support_mean"] == 1
