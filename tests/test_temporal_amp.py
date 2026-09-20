import math
import socket
import unittest

import torch

from ldm.modules.temporal_amp import optimizer_step_with_retry


class FakeScaler(object):
    def __init__(self, scale=128.0, enabled=True, backoff=0.5):
        self._scale = float(scale)
        self._enabled = bool(enabled)
        self._backoff = float(backoff)
        self._growth_tracker = 0
        self.step_calls = 0
        self.update_calls = []

    def scale(self, loss):
        return loss * self._scale if self._enabled else loss

    def unscale_(self, optimizer):
        if not self._enabled:
            return
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.detach().div_(self._scale)

    def step(self, optimizer):
        self.step_calls += 1
        optimizer.step()

    def update(self, new_scale=None):
        if new_scale is not None:
            self._scale = float(new_scale)
        self.update_calls.append(new_scale)

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        return self._scale

    def get_backoff_factor(self):
        return self._backoff

    def state_dict(self):
        return {"scale": self._scale, "_growth_tracker": self._growth_tracker}

    def load_state_dict(self, state):
        self._scale = float(state["scale"])
        self._growth_tracker = int(state.get("_growth_tracker", 0))


class FiniteLossInfGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.shape = tensor.shape
        ctx.device = tensor.device
        ctx.dtype = tensor.dtype
        return tensor.detach().new_tensor(0.0)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full(ctx.shape, float("inf"), device=ctx.device, dtype=ctx.dtype)


class FiniteLossHugeFiniteGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, magnitude):
        ctx.shape = tensor.shape
        ctx.device = tensor.device
        ctx.dtype = tensor.dtype
        ctx.magnitude = float(magnitude)
        return tensor.detach().new_tensor(0.0)

    @staticmethod
    def backward(ctx, grad_output):
        signs = torch.ones(ctx.shape, device=ctx.device, dtype=ctx.dtype)
        signs.reshape(-1)[1::2].mul_(-1.0)
        return signs * grad_output.to(ctx.dtype) * ctx.magnitude, None


def _finite_loss(param):
    sample = torch.randn_like(param)
    return ((param - sample) ** 2).mean(), sample.detach().clone()


def _adam_step_value(optimizer, param):
    step = optimizer.state[param]["step"]
    if torch.is_tensor(step):
        return float(step.item())
    return float(step)


class OptimizerStepWithRetryTests(unittest.TestCase):
    def test_finite_path_matches_plain_scaled_step(self):
        seed = 1234
        initial = torch.tensor([0.25, -0.5], dtype=torch.float32)

        torch.manual_seed(seed)
        expected = torch.nn.Parameter(initial.clone())
        expected_optimizer = torch.optim.Adam([expected], lr=1e-2)
        expected_scaler = FakeScaler(scale=64.0)
        expected_optimizer.zero_grad(set_to_none=True)
        expected_loss, expected_sample = _finite_loss(expected)
        expected_scaler.scale(expected_loss).backward()
        expected_scaler.unscale_(expected_optimizer)
        expected_norm = float(torch.nn.utils.clip_grad_norm_([expected], 1.0))
        expected_scaler.step(expected_optimizer)
        expected_scaler.update()
        expected_rng = torch.get_rng_state()

        torch.manual_seed(seed)
        actual = torch.nn.Parameter(initial.clone())
        actual_optimizer = torch.optim.Adam([actual], lr=1e-2)
        actual_scaler = FakeScaler(scale=64.0)

        def closure():
            loss, sample = _finite_loss(actual)
            return loss, {"sample": sample}

        loss, auxiliary, grad_norm, attempts = optimizer_step_with_retry(
            closure, [actual], actual_optimizer, actual_scaler
        )

        self.assertEqual(attempts, 0)
        self.assertEqual(actual_scaler.step_calls, 1)
        self.assertTrue(torch.allclose(actual, expected, atol=0, rtol=0))
        self.assertTrue(torch.allclose(auxiliary["sample"], expected_sample, atol=0, rtol=0))
        self.assertAlmostEqual(float(loss.detach()), float(expected_loss.detach()), places=7)
        self.assertAlmostEqual(grad_norm, expected_norm, places=7)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_overflow_retry_updates_once_and_does_not_corrupt_adam_state(self):
        seed = 4321
        initial = torch.tensor([0.2, -0.4], dtype=torch.float32)

        torch.manual_seed(seed)
        expected = torch.nn.Parameter(initial.clone())
        expected_optimizer = torch.optim.Adam([expected], lr=3e-3)
        expected_scaler = FakeScaler(scale=128.0)
        expected_optimizer.zero_grad(set_to_none=True)
        expected_loss, expected_sample = _finite_loss(expected)
        expected_scaler.scale(expected_loss).backward()
        expected_scaler.unscale_(expected_optimizer)
        torch.nn.utils.clip_grad_norm_([expected], 1.0)
        expected_scaler.step(expected_optimizer)
        expected_scaler.update()
        expected_rng = torch.get_rng_state()

        torch.manual_seed(seed)
        actual = torch.nn.Parameter(initial.clone())
        actual_optimizer = torch.optim.Adam([actual], lr=3e-3)
        actual_scaler = FakeScaler(scale=128.0)
        calls = []
        retry_events = []

        def closure():
            loss, sample = _finite_loss(actual)
            calls.append(sample)
            if len(calls) == 1:
                loss = loss + FiniteLossInfGrad.apply(actual)
            return loss, {"sample": sample}

        loss, auxiliary, grad_norm, attempts = optimizer_step_with_retry(
            closure, [actual], actual_optimizer, actual_scaler,
            max_retries=2, on_retry=retry_events.append
        )

        self.assertEqual(attempts, 1)
        self.assertEqual(actual_scaler.step_calls, 1)
        self.assertEqual(len(retry_events), 1)
        self.assertEqual(retry_events[0]["retry"], 1)
        self.assertEqual(actual_scaler.get_scale(), 64.0)
        self.assertTrue(math.isfinite(grad_norm))
        self.assertTrue(torch.allclose(actual, expected, atol=0, rtol=0))
        self.assertTrue(torch.allclose(auxiliary["sample"], expected_sample, atol=0, rtol=0))
        self.assertTrue(torch.allclose(calls[0], calls[1], atol=0, rtol=0))
        self.assertAlmostEqual(float(loss.detach()), float(expected_loss.detach()), places=7)
        self.assertEqual(_adam_step_value(actual_optimizer, actual), 1.0)
        self.assertTrue(torch.isfinite(actual_optimizer.state[actual]["exp_avg"]).all())
        self.assertTrue(torch.isfinite(actual_optimizer.state[actual]["exp_avg_sq"]).all())
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_huge_finite_gradients_use_fp64_norm_and_clip_without_retry(self):
        param = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        optimizer = torch.optim.SGD([param], lr=0.1)
        scaler = FakeScaler(scale=128.0)

        def closure():
            loss = param.sum() * 0.0 + FiniteLossHugeFiniteGrad.apply(param, 1e20)
            return loss, {}

        _, _, grad_norm, attempts = optimizer_step_with_retry(
            closure, [param], optimizer, scaler, max_retries=2
        )

        expected_norm = math.sqrt(2.0) * 1e20
        expected_update = torch.tensor([-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]) * 0.1
        self.assertEqual(attempts, 0)
        self.assertEqual(scaler.step_calls, 1)
        self.assertTrue(math.isfinite(grad_norm))
        self.assertAlmostEqual(grad_norm / expected_norm, 1.0, places=6)
        self.assertTrue(torch.allclose(param.detach(), expected_update, rtol=1e-6, atol=0))
        self.assertEqual(scaler.get_scale(), 128.0)

    def test_persistent_bad_gradients_fail_without_optimizer_step(self):
        param = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        optimizer = torch.optim.Adam([param], lr=1e-2)
        scaler = FakeScaler(scale=16.0)

        def closure():
            loss = (param ** 2).mean() + FiniteLossInfGrad.apply(param)
            return loss, {}

        with self.assertRaisesRegex(RuntimeError, "non-finite gradient norm persists"):
            optimizer_step_with_retry(closure, [param], optimizer, scaler, max_retries=1)

        self.assertEqual(scaler.step_calls, 0)
        self.assertEqual(optimizer.state, {})
        self.assertTrue(torch.allclose(param.detach(), torch.tensor([1.0, -1.0])))

    def test_nonfinite_loss_fails_before_backward_or_step(self):
        param = torch.nn.Parameter(torch.tensor([0.5, -0.5]))
        optimizer = torch.optim.Adam([param], lr=1e-2)
        scaler = FakeScaler(scale=16.0)

        def closure():
            return param.sum() * float("nan"), {}

        with self.assertRaisesRegex(RuntimeError, "non-finite loss"):
            optimizer_step_with_retry(closure, [param], optimizer, scaler, max_retries=1)

        self.assertEqual(scaler.step_calls, 0)
        self.assertIsNone(param.grad)
        self.assertEqual(optimizer.state, {})

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.device_count() >= 2,
                         "requires at least two CUDA devices")
    def test_cuda_gradscaler_ddp_synchronizes_retry_and_nonfinite_loss(self):
        results = _run_ddp_smoke()
        self.assertEqual(sorted(results.keys()), [0, 1])
        for rank, payload in results.items():
            self.assertEqual(payload["retry_attempts"], 1, payload)
            self.assertEqual(payload["closure_calls"], 2, payload)
            self.assertTrue(payload["retry_event"], payload)
            self.assertTrue(payload["weight_is_finite"], payload)
            self.assertEqual(payload["scale"], 64.0, payload)
            self.assertTrue(payload["cuda_rng_replayed"], payload)
            self.assertIn("non-finite loss", payload["loss_error"])
        self.assertEqual(results[0]["weight"], results[1]["weight"])
        self.assertEqual(results[0]["scale"], results[1]["scale"])


def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _run_ddp_smoke():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    port = _free_port()
    mp.spawn(_ddp_worker, args=(2, port, queue), nprocs=2, join=True)
    results = {}
    while len(results) < 2:
        rank, payload = queue.get()
        results[rank] = payload
    return results


def _ddp_worker(rank, world_size, port, queue):
    import torch.distributed as dist
    from torch.cuda.amp import GradScaler
    from torch.nn.parallel import DistributedDataParallel

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:%d" % port,
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(1000)
        device = torch.device("cuda", rank)
        model = torch.nn.Linear(2, 1, bias=False).to(device)
        model.weight.data.fill_(0.25)
        ddp = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-2)
        scaler = GradScaler(init_scale=128.0, growth_interval=2000)
        calls = []
        rng_draws = []
        retry_events = []

        def finite_loss():
            x = torch.tensor([[1.0, -2.0], [0.5, 0.25]], device=device)
            y = torch.tensor([[0.25], [-0.5]], device=device)
            return ((ddp(x) - y) ** 2).mean()

        def overflow_closure():
            calls.append(len(calls))
            rng_draws.append(torch.rand(3, device=device).detach().cpu())
            loss = finite_loss()
            if rank == 0 and len(calls) == 1:
                loss = loss + FiniteLossInfGrad.apply(next(ddp.parameters()).view(-1)[:2])
            return loss, {}

        _, _, _, attempts = optimizer_step_with_retry(
            overflow_closure, list(ddp.parameters()), optimizer, scaler,
            max_retries=2, on_retry=retry_events.append
        )

        def nonfinite_loss_closure():
            loss = finite_loss()
            if rank == 0:
                loss = loss * float("nan")
            return loss, {}

        try:
            optimizer_step_with_retry(
                nonfinite_loss_closure, list(ddp.parameters()), optimizer, scaler,
                max_retries=1
            )
        except RuntimeError as exc:
            loss_error = str(exc)
        else:
            loss_error = ""

        queue.put((rank, {
            "retry_attempts": attempts,
            "closure_calls": len(calls),
            "retry_event": bool(retry_events),
            "scale": float(scaler.get_scale()),
            "weight": [round(float(v), 8) for v in model.weight.detach().cpu().reshape(-1)],
            "weight_is_finite": bool(torch.isfinite(model.weight.detach()).all().item()),
            "cuda_rng_replayed": bool(len(rng_draws) == 2 and torch.equal(rng_draws[0], rng_draws[1])),
            "loss_error": loss_error,
        }))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
