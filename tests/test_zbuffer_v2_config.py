import ast
import unittest
from pathlib import Path

import yaml


CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CODE_ROOT / "configs" / "Boost_Sat2Den" / "train" / "KITTI_raw_sat_lidar_raea_cfgdrop10.yaml"
PIXEL_CONFIG_PATH = CODE_ROOT / "configs" / "Boost_Sat2Den" / "train" / "KITTI_raw_sat_lidar_pixel_cfgdrop10.yaml"
TRAIN_PATH = CODE_ROOT / "tools" / "train_kitti_raea.py"


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_pixel_config():
    return yaml.safe_load(PIXEL_CONFIG_PATH.read_text())


def test_yaml_selects_visible_ray_encoder_and_single_plane_cache():
    config = load_config()
    params = config["model"]["params"]
    lidar_encoder = params["Lidar_context_config"]

    assert lidar_encoder["target"].endswith("LidarVisibleRaySemanticTokenEncoder")
    assert "ray_depth_bins" not in lidar_encoder["params"]
    for split in ("train", "test"):
        data = config["data"]["params"][split]["params"]
        assert data["lidar_ray_depth_bins"] == 1
        assert "zbuffer_visible_ray" in data["lidar_ray_feature_cache_root"]


def test_yaml_keeps_geometry_supervision_and_posterior_fusion_contract():
    params = load_config()["model"]["params"]
    unet = params["DDPM_config"]["params"]["unet_config"]["params"]

    assert params["lidar_depth_resample_mode"] == "masked_area"
    assert params["lidar_depth_loss_weight"] == 0.1
    assert unet["use_lidar_cross_attention"] is True
    assert unet["lidar_reference_window"] == 3
    assert unet["ray_fusion_mode"] == "ray_posterior"
    assert unet["use_lidar_ray_posterior"] is True


def test_training_defaults_match_yaml():
    tree = ast.parse(TRAIN_PATH.read_text())
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"LIDAR_CONTEXT_BACKBONE", "LIDAR_RAY_CACHE_PLANES"}
    }

    assert assignments["LIDAR_CONTEXT_BACKBONE"] == "utonia_zbuffer_visible_ray"
    assert assignments["LIDAR_RAY_CACHE_PLANES"] == 1
    source = TRAIN_PATH.read_text()
    assert 'default=0.1' in source
    assert 'cfg.model.params.lidar_depth_resample_mode = "masked_area"' in source


def test_pixel_yaml_selects_spatial_encoder_and_ragged_cache():
    config = load_pixel_config()
    params = config["model"]["params"]
    unet = params["DDPM_config"]["params"]["unet_config"]["params"]
    lidar_encoder = params["Lidar_context_config"]

    assert lidar_encoder["target"].endswith("LidarPixelConditionEncoder")
    assert lidar_encoder["params"]["pyramid_channels"] == [64, 128, 256, 256]
    assert unet["use_lidar_cross_attention"] is False
    assert unet["lidar_spatial_channels"] == [64, 128, 256, 256]
    assert "lidar_reference_window" not in unet
    for split in ("train", "test"):
        data = config["data"]["params"][split]["params"]
        assert "lidar_ray_feature_cache_root" not in data
        assert "lidar_ray_feature_dim" not in data
        assert "lidar_ray_depth_bins" not in data
        assert "pixel_lidar_v21" in data["lidar_pixel_feature_cache_root"]


def load_tests(loader, tests, pattern):
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    return unittest.TestSuite(unittest.FunctionTestCase(function) for function in functions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
