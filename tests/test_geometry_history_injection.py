"""End-to-end check of the live history injection path on a tiny UNet.

The real backbone is a checkpoint we cannot load in a unit test, and at
initialisation both the SpatialTransformer proj_out and the UNet output
convolution are zero_module, which would hide every injection. This test
therefore builds a tiny ray-posterior UNet, un-zeros those two convolution
stacks to emulate a trained backbone, and stubs the satellite cross-attention
(it needs calibration/theta inputs). Everything between the history hub and the
UNet output stays real.

Contract under test:
  * a has_history=False payload is bit-identical to no payload at all;
  * a has_history=True payload on valid correspondences changes the output.
"""
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ldm.modules.KITTI_attention import BasicTransformerBlock, SpatialTransformer  # noqa: E402
from models.KITTI_geo_ldm_diffusion.openaimodel import UNetModel  # noqa: E402
from temporal_history import enable_history_attention, fusion_block_stages  # noqa: E402


def _zero_attn2(x, *args, **kwargs):
    """Installed as an instance attribute, so it is NOT bound to the module."""
    return torch.zeros_like(x)


def build_unet():
    unet = UNetModel(
        image_size=32, in_channels=4, out_channels=4, model_channels=32,
        attention_resolutions=[4, 2, 1], num_res_blocks=2, channel_mult=[1, 2, 4, 4],
        num_heads=2, context_dim=32, use_spatial_transformer=True, transformer_depth=1,
        use_checkpoint=False, legacy=False, use_lidar_cross_attention=True,
        lidar_context_dim=32, ray_fusion_mode="ray_posterior",
        use_lidar_ray_posterior=True, lidar_reference_window=3,
    )
    for module in unet.modules():
        if isinstance(module, BasicTransformerBlock) and module.attn2 is not None:
            module.attn2.forward = _zero_attn2
        if isinstance(module, SpatialTransformer):
            torch.nn.init.normal_(module.proj_out.weight, std=0.02)
        gate = getattr(getattr(module, "ray_posterior_fusion", None), "lidar_gate", None)
        if gate is not None:
            torch.nn.init.normal_(gate.weight, std=0.1)
    torch.nn.init.normal_(unet.out[-1].weight, std=0.02)
    return unet


class HistoryInjectionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.unet = build_unet().eval()
        self.hub, self.encoder, self.blocks = enable_history_attention(self.unet)
        self.tokens = torch.randn(1, 16 * 64, 64)

    def forward(self, payload):
        if payload is None:
            self.hub.clear()
        else:
            self.hub.set(payload)
        with torch.no_grad():
            out = self.unet(
                torch.zeros(1, 4, 16, 64),
                torch.zeros(1, dtype=torch.long),
                context=torch.zeros(1, 77, 32),
                lidar_context=torch.zeros(1, 1024, 32),
                lidar_evidence=torch.zeros(1, 4, 16, 64),
                lidar_geometry_mask=torch.ones(1, 1, 128, 512),
                latent_hw=(16, 64),
            )
        self.hub.clear()
        return out

    def payload(self, has_history):
        data = {
            "history_tokens": self.tokens,
            "history_hw": (16, 64),
            "has_history": has_history,
        }
        if has_history:
            data["history_grid"] = torch.zeros(1, 16, 64, 2)
            data["history_valid"] = torch.ones(1, 16, 64, dtype=torch.bool)
        return data

    def test_hub_is_wired_to_the_selected_decoder_block(self):
        self.assertTrue(self.blocks)
        for block in self.blocks:
            self.assertIs(block.history_hub, self.hub)
            self.assertEqual(block.history_block_stage, "decoder")

    def test_no_history_is_bit_identical_to_no_payload(self):
        baseline = self.forward(None)
        self.assertGreater(float(baseline.abs().max()), 0.0)
        off = self.forward(self.payload(False))
        self.assertTrue(torch.equal(off, baseline))

    def test_history_with_valid_geometry_changes_the_output(self):
        baseline = self.forward(None)
        on = self.forward(self.payload(True))
        self.assertFalse(torch.allclose(on, baseline))
        self.assertGreater(float((on - baseline).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
