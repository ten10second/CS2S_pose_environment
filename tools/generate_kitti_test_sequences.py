#!/usr/bin/env python3
"""Render fixed held-out clips with the existing single-frame AMP inference path."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.select_kitti_test_sequences import (  # noqa: E402
    flatten_clips,
    sha256_file,
    validate_fixed_test_sequence_manifest,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def run(args):
    split = args.split_dir
    manifest = split / "fixed_test_sequences.jsonl"
    report = split / "fixed_test_sequences.report.json"
    clips = validate_fixed_test_sequence_manifest(
        manifest, split / "train_manifest.jsonl", split / "test_manifest.jsonl",
        report_json=report, expected_clips=2, frames_per_clip=16,
    )
    selected = flatten_clips(clips)
    if (args.out_dir / "metadata.json").exists():
        raise FileExistsError("Completed test preview already exists: " + str(args.out_dir))

    # Keep split validation CPU-only and ahead of expensive model/cache loading.
    import torch
    from omegaconf import OmegaConf
    from tools.build_kitti_test_sequence_gallery import build_gallery
    from tools.generate_kitti_raea_samples import (
        generate_prediction, load_checkpoint_into_model, make_lidar_overlay,
        safe_sample_id, sample_to_batch, save_tensor_image,
    )
    from utils.util import instantiate_from_config

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(3407)
    config_path = args.run_dir / "run_config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.data.params.test.params.manifest = str(manifest)
    dataset = instantiate_from_config(cfg.data.params.test)
    if len(dataset) != len(selected):
        raise RuntimeError("Dataset length differs from validated test manifest")
    status_path = args.out_dir / "status.json"
    write_json(status_path, {"phase": "loading_model", "started_unix": time.time()})
    model = instantiate_from_config(cfg.model).cuda().eval()
    load_checkpoint_into_model(model, args.checkpoint)
    records = []
    scales = (3.0, 7.5)
    for index, expected in enumerate(selected):
        sample = dataset[index]
        if sample["sample_id"] != expected["sample_id"]:
            raise RuntimeError("Dataset sample order differs from validated test manifest")
        clip_index = index // 16
        # Same noise throughout a clip and across checkpoints/CFG values.
        # Each frame is generated independently: there is no temporal conditioning.
        seed = 2026 + clip_index
        safe_id = safe_sample_id(sample["sample_id"])
        sources = {}
        for kind, tensor in (("satellite", sample["sat_map"]), ("gt", sample["grd_left_imgs"])):
            path = args.out_dir / "images" / kind / (safe_id + ".png")
            save_tensor_image(tensor, path)
            sources[kind] = str(path.relative_to(args.out_dir))
        overlay = args.out_dir / "images/lidar_overlay" / (safe_id + ".png")
        overlay.parent.mkdir(parents=True, exist_ok=True)
        make_lidar_overlay(sample["grd_left_imgs"], sample["lidar_cond"]).save(overlay)
        sources["lidar_overlay"] = str(overlay.relative_to(args.out_dir))
        batch = sample_to_batch(sample)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            initial = torch.randn((1, 4, 16, 64), device="cuda")
            noise_hash = hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest()
            del initial
        outputs = {}
        for scale in scales:
            started = time.time()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.cuda.amp.autocast(enabled=True):
                prediction, _, _, _ = generate_prediction(
                    model, batch, probe="normal", ddim_steps=50, seed=seed,
                    guidance_scale=scale, eta=0.0, temperature=1.0, key_stats_max_tokens=256,
                )
            if not torch.isfinite(prediction).all():
                raise RuntimeError("Non-finite prediction: " + safe_id)
            path = args.out_dir / ("images/cfg_%g" % scale) / (safe_id + ".png")
            save_tensor_image(prediction[0], path)
            outputs[str(scale)] = str(path.relative_to(args.out_dir))
            del prediction
            progress = {"phase": "generating", "sample_index": index, "sample_id": sample["sample_id"],
                        "cfg": scale, "seconds": time.time() - started}
            print(json.dumps(progress), flush=True)
            write_json(status_path, progress)
        records.append({
            "sample_id": sample["sample_id"], "split": "test2", "drive": expected["drive"],
            "frame_index": expected["frame_index"], "clip_index": clip_index, "seed": seed,
            "initial_noise_sha256": noise_hash, "sources": sources, "outputs": outputs,
        })
        write_json(args.out_dir / "records.json", records)
        del batch, sample
        torch.cuda.empty_cache()

    from PIL import Image
    for record in records:
        for relative in list(record["sources"].values()) + list(record["outputs"].values()):
            with Image.open(args.out_dir / relative) as image:
                image.verify()
    metadata = {
        "checkpoint_step": args.checkpoint_step,
        "source_checkpoint": str(args.run_dir / "checkpoints" / ("step_%06d.pt" % args.checkpoint_step)),
        "inference_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "run_config_sha256": sha256_file(config_path), "selected_manifest_sha256": sha256_file(manifest),
        "selection_report": json.loads(report.read_text()), "split": "test2", "num_samples": len(records),
        "num_clips": 2, "frames_per_clip": 16, "cfg_scales": list(scales), "amp": True,
        "matmul_precision": "high", "ddim_steps": 50, "eta": 0.0, "temperature": 1.0,
        "seed_policy": "2026 + clip_index; identical noise within each clip and across CFG/checkpoints",
        "temporal_conditioning": False, "gt_rgb_conditioning": False,
        "cfg_formula": "eps(0,L) + s * (eps(S,L) - eps(0,L))",
        "scope": "fixed consecutive test frames for qualitative monitoring; not a quantitative benchmark",
    }
    write_json(args.out_dir / "metadata.json", metadata)
    build_gallery(args.out_dir)
    write_json(status_path, {"phase": "complete", "outputs": len(records) * len(scales),
                             "completed_unix": time.time()})


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments)
    except BaseException as exc:
        write_json(arguments.out_dir / "status.json", {"phase": "failed", "error": repr(exc)})
        raise
