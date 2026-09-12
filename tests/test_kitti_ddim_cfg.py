import torch

from models.KITTI_geo_ldm_diffusion.ddim_KITTI import KITTI_DDIMSampler


class FakeDenoiseModel:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        x,
        t,
        context=None,
        lidar_context=None,
        lidar_evidence=None,
        lidar_geometry_mask=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "x_shape": tuple(x.shape),
                "t_shape": tuple(t.shape),
                "context": context.detach().clone(),
                "lidar_context": lidar_context,
                "lidar_evidence": lidar_evidence.detach().clone(),
                "lidar_geometry_mask": lidar_geometry_mask.detach().clone(),
            }
        )
        values = torch.where(
            context.flatten(1).abs().sum(dim=1) > 0,
            torch.full((x.shape[0],), 3.0, device=x.device),
            torch.full((x.shape[0],), 1.0, device=x.device),
        )
        return values.view(-1, 1, 1, 1).expand_as(x)


class FakeModel:
    def __init__(self):
        self.denoise_model = FakeDenoiseModel()


def make_sampler():
    sampler = KITTI_DDIMSampler.__new__(KITTI_DDIMSampler)
    sampler.model = FakeModel()
    sampler.ddim_alphas = torch.tensor([1.0])
    sampler.ddim_alphas_prev = torch.tensor([1.0])
    sampler.ddim_sqrt_one_minus_alphas = torch.tensor([0.0])
    sampler.ddim_sigmas = torch.tensor([0.0])
    return sampler


def test_missing_unconditional_conditioning_uses_zero_satellite_cfg_and_reuses_lidar():
    sampler = make_sampler()
    x = torch.zeros(1, 4, 16, 64)
    t = torch.tensor([9])
    conditioning = torch.full((1, 2, 3), 2.0)
    lidar_context = {
        "features": (torch.ones(1, 64, 16, 64), torch.ones(1, 128, 8, 32)),
        "masks": (torch.ones(1, 1, 16, 64), torch.ones(1, 1, 8, 32)),
    }
    lidar_evidence = torch.ones(1, 3, 16, 64)
    lidar_geometry_mask = torch.ones(1, 1, 16, 64)

    e_t, _, _ = sampler.p_sample_ddim(
        x,
        conditioning,
        t,
        index=0,
        unconditional_guidance_scale=2.5,
        unconditional_conditioning=None,
        lidar_context=lidar_context,
        lidar_evidence=lidar_evidence,
        lidar_geometry_mask=lidar_geometry_mask,
    )

    call = sampler.model.denoise_model.calls[0]
    assert call["x_shape"][0] == 2
    assert call["t_shape"][0] == 2
    assert torch.equal(call["context"][0], torch.zeros_like(conditioning[0]))
    assert torch.equal(call["context"][1], conditioning[0])
    assert call["lidar_context"]["features"][0].shape[0] == 2
    assert call["lidar_context"]["masks"][0].shape[0] == 2
    assert call["lidar_evidence"].shape[0] == 2
    assert call["lidar_geometry_mask"].shape[0] == 2
    assert torch.allclose(e_t, torch.full_like(e_t, 6.0))


def load_tests(loader, tests, pattern):
    import unittest

    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    )
