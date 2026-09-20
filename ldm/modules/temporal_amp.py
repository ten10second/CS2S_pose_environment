"""Bounded, DDP-synchronized AMP retries without dropping training batches."""
import math

import torch


def _any_rank(flag, device):
    value = torch.tensor(int(flag), device=device, dtype=torch.int32)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return bool(value.item())



def _clip_gradients(parameters):
    try:
        return float(torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True).detach().cpu())
    except RuntimeError as exc:
        if "non-finite" not in str(exc):
            raise
    # FP32 norm accumulation can overflow with finite gradient elements. The
    # error_if_nonfinite path leaves gradients untouched, so retry only the
    # reduction in FP64 before deciding whether the gradients need AMP backoff.
    gradients = [p.grad for p in parameters if p.grad is not None]
    norm = torch.stack([g.detach().norm(2, dtype=torch.float64) for g in gradients]).norm(2)
    if torch.isfinite(norm):
        coefficient = (1.0 / (norm + 1e-6)).clamp(max=1.0)
        for grad in gradients:
            grad.detach().mul_(coefficient.to(device=grad.device, dtype=grad.dtype))
    return float(norm.detach().cpu())

def optimizer_step_with_retry(closure, parameters, optimizer, scaler, max_retries=8, on_retry=None):
    """Call closure for (loss, auxiliary), applying exactly one finite update.

    Failed gradients never reach optimizer.step. All ranks retry together.
    RNG state is restored so retrying does not change dropout/noise realization.
    """
    parameters = list(parameters)
    device = parameters[0].device
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    for attempt in range(max_retries + 1):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss, auxiliary = closure()
        if _any_rank(not bool(torch.isfinite(loss.detach()).all()), device):
            raise RuntimeError("non-finite loss on at least one rank")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = _clip_gradients(parameters)
        if not _any_rank(not math.isfinite(grad_norm), device):
            scaler.step(optimizer)
            scaler.update()
            return loss, auxiliary, grad_norm, attempt
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled() or attempt == max_retries:
            raise RuntimeError("non-finite gradient norm persists after %d AMP retries" % attempt)
        old_scale = scaler.get_scale()
        new_scale = old_scale * scaler.get_backoff_factor()
        if not math.isfinite(new_scale) or new_scale <= 0:
            raise RuntimeError("invalid AMP recovery scale")
        # Public update clears per-optimizer UNSCALED state. Reset the public
        # state-dict growth tracker too: explicit new_scale does not reset it.
        scaler.update(new_scale=new_scale)
        state = scaler.state_dict()
        state["_growth_tracker"] = 0
        scaler.load_state_dict(state)
        if on_retry is not None:
            on_retry({"event": "amp_overflow_retry", "retry": attempt + 1,
                      "loss": float(loss.detach().cpu()), "scale_before": old_scale,
                      "scale_after": new_scale})
        del loss, auxiliary
