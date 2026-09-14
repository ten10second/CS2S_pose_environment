#!/usr/bin/env python3
"""Monitor saved checkpoints and render held-out clips on a separate idle GPU."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.generate_kitti_test_sequences import write_json  # noqa: E402
from tools.select_kitti_test_sequences import sha256_file, validate_fixed_test_sequence_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--training-control-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    return parser.parse_args()


def completed_preview(output):
    status = output / "status.json"
    if not status.exists():
        return False
    return (json.loads(status.read_text()).get("phase") == "complete"
            and (output / "metadata.json").is_file() and (output / "comparison.html").is_file())


def run(args):
    control = args.control_dir
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "watcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (control / "watcher.pid").write_text(str(os.getpid()) + "\n")
    manifest = args.split_dir / "fixed_test_sequences.jsonl"
    validate_fixed_test_sequence_manifest(
        manifest, args.split_dir / "train_manifest.jsonl", args.split_dir / "test_manifest.jsonl",
        report_json=args.split_dir / "fixed_test_sequences.report.json",
    )
    manifest_hash = sha256_file(manifest)
    initial_checkpoints = sorted((args.run_dir / "checkpoints").glob("step_*.pt"))
    if not initial_checkpoints:
        raise RuntimeError("No saved checkpoint available for initial test preview")
    first_step = int(initial_checkpoints[-1].stem.split("_")[-1])
    preview_root = args.run_dir / "test_sequences"
    preview_root.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "effect_preview_policy.json", {
        "split": "test2", "manifest": str(manifest), "manifest_sha256": manifest_hash,
        "selection_report": str(args.split_dir / "fixed_test_sequences.report.json"),
        "authoritative_preview_directory": str(preview_root), "control_directory": str(control),
        "clips": 2, "frames_per_clip": 16, "cfg_scales": [3.0, 7.5],
        "seed_policy": "2026 + clip_index, fixed within each clip and across checkpoints/CFG",
        "temporal_conditioning": False, "first_checkpoint_step": first_step,
        "inline_samples_scope": "Legacy running job uses training samples; internal diagnostics only",
        "training_probes_scope": "Train-set supervision diagnostics; not generation-effect evidence",
    })
    child = None
    pin = preview_root / "active_checkpoint.pt"

    def status(phase, **values):
        write_json(control / "status.json", {"phase": phase, "updated_unix": time.time(), **values})

    def check_stop():
        for root in (control, args.training_control_dir):
            if (root / "user_stop_request.json").exists():
                raise InterruptedError("User requested stop")

    def stop_signal(signum, frame):
        raise InterruptedError("Watcher signal %d" % signum)

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    try:
        while True:
            check_stop()
            if sha256_file(manifest) != manifest_hash:
                raise RuntimeError("Fixed test manifest changed during monitoring")
            candidates = []
            for checkpoint in sorted((args.run_dir / "checkpoints").glob("step_*.pt")):
                step = int(checkpoint.stem.split("_")[-1])
                if step >= first_step and not completed_preview(preview_root / checkpoint.stem):
                    candidates.append(checkpoint)
            if not candidates:
                training_status_path = args.training_control_dir / "status.json"
                training = json.loads(training_status_path.read_text()) if training_status_path.exists() else {}
                train_pid = training.get("train_pid")
                if train_pid:
                    try:
                        os.kill(int(train_pid), 0)
                    except ProcessLookupError:
                        status("stopped_training_not_running")
                        return
                elif training.get("phase") in ("complete", "completed", "failed", "stopped"):
                    status("stopped_training_" + training["phase"])
                    return
                status("waiting_for_checkpoint", latest_completed=max(
                    (int(path.name.split("_")[-1]) for path in preview_root.glob("step_*")
                     if completed_preview(path)), default=None))
                time.sleep(30)
                continue
            used = subprocess.check_output([
                "nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits",
            ], text=True)
            if int(used.strip()) >= 512:
                status("waiting_for_idle_gpu", gpu=args.gpu)
                time.sleep(30)
                continue
            checkpoint = candidates[0]
            step = int(checkpoint.stem.split("_")[-1])
            output = preview_root / checkpoint.stem
            pin.unlink(missing_ok=True)
            try:
                # Atomic checkpoint saves plus a same-filesystem hardlink protect
                # this reader from retention pruning without an 11 GiB copy.
                os.link(checkpoint, pin)
            except FileNotFoundError:
                continue
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, str(REPO / "tools/generate_kitti_test_sequences.py"),
                       "--run-dir", str(args.run_dir), "--checkpoint", str(pin),
                       "--checkpoint-step", str(step), "--split-dir", str(args.split_dir),
                       "--out-dir", str(output)]
            with (control / (checkpoint.stem + ".log")).open("a") as log:
                child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            write_json(control / (checkpoint.stem + ".command.json"),
                       {"command": command, "pid": child.pid, "gpu": args.gpu})
            status("generating", checkpoint_step=step, child_pid=child.pid, output=str(output), gpu=args.gpu)
            while child.poll() is None:
                check_stop()
                time.sleep(5)
            (control / (checkpoint.stem + ".exit")).write_text(str(child.returncode) + "\n")
            if child.returncode != 0 or not completed_preview(output):
                raise RuntimeError("Test preview failed at step %d; inspect checkpoint log" % step)
            child = None
            pin.unlink(missing_ok=True)
            write_json(preview_root / "latest.json", {"checkpoint_step": step, "output_directory": str(output),
                                                       "comparison_html": str(output / "comparison.html")})
    except InterruptedError as exc:
        status("stopped", reason=str(exc))
    except BaseException as exc:
        status("failed", error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        pin.unlink(missing_ok=True)


if __name__ == "__main__":
    run(parse_args())
