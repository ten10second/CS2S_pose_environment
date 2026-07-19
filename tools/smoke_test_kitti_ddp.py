#!/usr/bin/env python3
"""Minimal torchrun smoke test for the KITTI DDP training infrastructure."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

# Keep the default smoke test independent of the host GPU count. Set
# KITTI_DDP_SMOKE_USE_CUDA=1 to exercise NCCL on a multi-GPU host.
if os.environ.get("KITTI_DDP_SMOKE_USE_CUDA", "0") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.nn.parallel import DistributedDataParallel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.train_kitti_raea import (
    TrainingStepModule,
    cleanup_distributed,
    distributed_barrier,
    ensure_fresh_run_directory_distributed,
    gather_distributed_record,
    init_distributed,
)


class TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.lidar_context_model = None

    def training_step(self, batch, _step):
        return (self.weight * batch["x"]).mean()


def main():
    args = SimpleNamespace(local_rank=-1, dist_backend="auto")
    info = init_distributed(args)
    device = info["device"]
    model = TinyTrainingModel().to(device)
    wrapper = TrainingStepModule(model, token_structure_loss_weight=0.0).to(device)
    ddp_kwargs = {"find_unused_parameters": False}
    if device.type == "cuda":
        ddp_kwargs.update(
            {
                "device_ids": [info["local_rank"]],
                "output_device": info["local_rank"],
            }
        )
    wrapper = DistributedDataParallel(wrapper, **ddp_kwargs)
    value = torch.tensor([float(info["rank"] + 1)], device=device)
    wrapper({"x": value}, 1).backward()
    expected_grad = (info["world_size"] + 1) / 2.0
    if not torch.allclose(model.weight.grad, torch.tensor(expected_grad, device=device)):
        raise RuntimeError(f"unexpected synchronized gradient: {model.weight.grad}")

    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "manual").replace("/", "_")
    run_dir = Path(tempfile.gettempdir()) / f"kitti_ddp_smoke_{run_id}"
    if info["is_main"]:
        shutil.rmtree(run_dir, ignore_errors=True)
    distributed_barrier(info["distributed"])
    ensure_fresh_run_directory_distributed(
        run_dir,
        resume_ckpt="",
        distributed=info["distributed"],
        is_main=info["is_main"],
    )
    if info["is_main"]:
        run_dir.mkdir()
        (run_dir / "stale.pt").write_text("stale")
    distributed_barrier(info["distributed"])
    guard_caught = False
    try:
        ensure_fresh_run_directory_distributed(
            run_dir,
            resume_ckpt="",
            distributed=info["distributed"],
            is_main=info["is_main"],
        )
    except FileExistsError:
        guard_caught = True
    if not guard_caught:
        raise RuntimeError("distributed run-directory guard did not reach every rank")
    record = gather_distributed_record(
        {"rank_value": float(info["rank"] + 1)},
        info["distributed"],
        info["rank"],
        info["world_size"],
    )
    if info["is_main"]:
        if record["rank_value"] != expected_grad:
            raise RuntimeError(f"unexpected gathered metric: {record}")
        print(
            {
                "ddp_smoke": "ok",
                "world_size": info["world_size"],
                "gradient": float(model.weight.grad),
                "metric": record["rank_value"],
                "pid": os.getpid(),
            }
        )
        shutil.rmtree(run_dir, ignore_errors=True)
    distributed_barrier(info["distributed"])
    cleanup_distributed()


if __name__ == "__main__":
    main()
