"""Exercise actual sampler methods without loading CLIP or auxiliary models."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from ldm.modules.temporal_condition import repeat_history, validate_history


SOURCE = Path(__file__).resolve().parents[1] / "models/KITTI_geo_ldm_diffusion/ddim_KITTI.py"
namespace = {"torch": torch, "np": np, "tqdm": lambda values, **kwargs: values,
             "validate_history": validate_history, "repeat_history": repeat_history,
             "default_noise": torch.zeros(100, 1, 4, 4, 8)}
tree = ast.parse(SOURCE.read_text())
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {"repeat_batch_conditioning", "zero_like_conditioning", "concat_conditioning_pair"}:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KITTI_DDIMSampler")
for name in ("sample", "ddim_sampling", "condition_score", "p_sample_ddim"):
    function = next(n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)


class RecordingDenoiser:
    def __init__(self):
        self.calls = []

    def __call__(self, x, t, history=None, **kwargs):
        self.calls.append((x.clone(), t.clone(), history))
        return torch.zeros_like(x) if history is None else history["latent"] * .01


class Sampler:
    sample = namespace["sample"]
    ddim_sampling = namespace["ddim_sampling"]
    condition_score = namespace["condition_score"]
    p_sample_ddim = namespace["p_sample_ddim"]

    def __init__(self):
        self.model = SimpleNamespace(device=torch.device("cpu"), denoise_model=RecordingDenoiser())
        self.ddim_timesteps = np.array([1, 101, 201, 301, 401, 501, 601, 701])
        self.ddim_alphas = torch.linspace(.99, .3, 8)
        self.ddim_alphas_prev = torch.cat([torch.ones(1), self.ddim_alphas[:-1]])
        self.ddim_sqrt_one_minus_alphas = (1 - self.ddim_alphas).sqrt()
        self.ddim_sigmas = torch.zeros(8)

    def make_schedule(self, **kwargs):
        pass


class SamplingTests(unittest.TestCase):
    def test_history_is_read_at_every_step_without_changing_initial_state(self):
        sampler = Sampler()
        noise = torch.randn(2, 4, 4, 8)
        history = {"latent": torch.randn_like(noise), "masks": torch.cat([torch.ones(2,2,4,8),torch.zeros(2,1,4,8)],1)}
        _, info = sampler.sample(8, 2, [4, 4, 8], x_T=noise, history=history)
        calls = sampler.model.denoise_model.calls
        self.assertEqual([int(c[1][0]) for c in calls], [701, 601, 501, 401, 301, 201, 101, 1])
        self.assertTrue(torch.equal(calls[0][0], noise))
        self.assertTrue(all(c[2] is history for c in calls))
        self.assertEqual(info["history"]["steps"], 8)

    def test_history_off_preserves_full_single_frame_schedule(self):
        sampler = Sampler()
        noise = torch.randn(2, 4, 4, 8)
        out, info = sampler.sample(8, 2, [4, 4, 8], x_T=noise)
        torch.testing.assert_close(out, noise / sampler.ddim_alphas[-1].sqrt())
        self.assertEqual(len(sampler.model.denoise_model.calls), 8)
        self.assertNotIn("history", info)

    def test_cfg_keeps_sample_history_on_both_branches(self):
        sampler = Sampler()
        noise = torch.randn(2, 4, 4, 8)
        history = {"latent": torch.stack([torch.ones_like(noise[0]), torch.ones_like(noise[0]) * 7]),
                   "masks": torch.cat([torch.ones(2,2,4,8),torch.zeros(2,1,4,8)],1)}
        sampler.sample(8, 2, [4, 4, 8], x_T=noise, conditioning=torch.ones(2, 16, 12),
                       unconditional_guidance_scale=7.5, history=history)
        for _, _, read in sampler.model.denoise_model.calls:
            self.assertTrue(torch.equal(read["latent"], torch.cat([history["latent"]] * 2)))
            self.assertTrue(torch.equal(read["masks"], torch.cat([history["masks"]]*2)))

    def test_legacy_midpoint_history_rejected(self):
        with self.assertRaises(ValueError):
            Sampler().sample(8, 2, [4, 4, 8], history={"latent": torch.zeros(2, 4, 4, 8), "timestep": 500})


if __name__ == "__main__":
    unittest.main()
