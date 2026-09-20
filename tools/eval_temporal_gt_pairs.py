"""Generate next-frame images from real previous-frame GT history pairs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.infer_temporal import decode, evaluation_device, sample_frame
from tools.train_temporal_pairs import (
    ConsecutivePairDataset,
    collate_pairs,
    encode_conditions,
    encode_latent,
    load_base,
    _to_device,
)
from ldm.modules.temporal_pair_training import build_history, load_temporal_checkpoint


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(tensor.shape)).encode())
    h.update(str(tensor.dtype).encode())
    h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def tree_to(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False)
    if isinstance(value, dict):
        return {k: tree_to(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_to(v, device) for v in value)
    if isinstance(value, list):
        return [tree_to(v, device) for v in value]
    return value


def split_tree(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        return value[index:index + 1].detach().cpu()
    if isinstance(value, dict):
        return {k: split_tree(v, index) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(split_tree(v, index) for v in value)
    if isinstance(value, list):
        return [split_tree(v, index) for v in value]
    return value


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    from PIL import Image
    import numpy as np
    arr = (tensor.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def edge_map(tensor: torch.Tensor) -> torch.Tensor:
    gray = tensor.float().mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    return torch.sqrt(F.conv2d(gray, kx, padding=1).square() + F.conv2d(gray, ky, padding=1).square() + 1e-8)


def diagnostic_metrics(rgb: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    mse = F.mse_loss(rgb.float(), gt.float()).item()
    return {
        "rgb_mae": F.l1_loss(rgb.float(), gt.float()).item(),
        "rgb_mse": mse,
        "rgb_psnr": float(-10.0 * torch.log10(torch.tensor(max(mse, 1e-12))).item()),
        "edge_mae": F.l1_loss(edge_map(rgb), edge_map(gt)).item(),
    }


def require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError("non-finite %s" % name)


@contextmanager
def unwrapped_denoiser(model: Any):
    """Temporarily expose the raw UNet to the sampler and always restore DDP."""
    ddpm = model.DDPM
    original = ddpm.denoise_model
    if hasattr(original, "module"):
        ddpm.denoise_model = original.module
    try:
        yield ddpm.denoise_model
    finally:
        ddpm.denoise_model = original


@contextmanager
def preserve_torch_rng(device: torch.device):
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    cuda_device = None
    if getattr(device, "type", None) == "cuda":
        cuda_device = device.index if device.index is not None else torch.cuda.current_device()
        cuda_state = torch.cuda.get_rng_state(cuda_device)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_device)


def checked_history_info(info: Any, expected_steps: int, expect_history: bool) -> Dict[str, Any]:
    if not isinstance(info, Mapping):
        raise RuntimeError("sampler returned non-mapping info")
    history = info.get("history")
    if expect_history:
        if not isinstance(history, Mapping):
            raise RuntimeError("sampler did not report active persistent history")
        mode = history.get("mode")
        steps = int(history.get("steps", -1))
        if mode != "persistent_condition" or steps != int(expected_steps):
            raise RuntimeError("sampler history mismatch: mode=%r steps=%r expected=%d" % (mode, steps, int(expected_steps)))
        return {"history": {"mode": mode, "steps": steps}, "history_active": True}
    if history is not None:
        raise RuntimeError("OFF sampler unexpectedly reported history")
    return {"history": None, "history_active": False}


class GTPairEvaluator:
    """Visual evaluator for the direct task: previous GT frame -> current frame."""

    def __init__(self, model: Any, cfg: Any, args: Any, info: Mapping[str, Any], out_dir: str | Path, base=None):
        self.model = model
        self.cfg = cfg
        self.args = args
        self.info = dict(info)
        self.base = base
        self.rank = int(self.info.get("rank", 0))
        self.world = int(self.info.get("world", 1))
        self.device = self.info.get("device", torch.device("cpu"))
        self.distributed = bool(self.info.get("distributed", False))
        self.out_dir = Path(out_dir)
        self.eval_dir = self.out_dir / "gt_pair_eval"
        self.selection: Optional[List[Dict[str, Any]]] = None
        self.cache: List[Dict[str, Any]] = []
        self.start = time.time()

    def _settings_path(self) -> Optional[Path]:
        path = getattr(self.args, "pair_eval_settings", "") or getattr(self.args, "fixed_eval_settings", "")
        return Path(path) if path else None

    def _load_settings(self) -> Optional[Mapping[str, Any]]:
        path = self._settings_path()
        if not path:
            return None
        return json.loads(path.read_text())

    def _validate_base(self, settings: Optional[Mapping[str, Any]]) -> None:
        if settings is None or self.base is None:
            return
        expected = settings.get("base")
        if expected is None:
            raise ValueError("pair eval settings has no base identity")
        for key in ("sha256", "denoise_model_sha256", "step"):
            if key in expected or key in self.base:
                if expected.get(key) != self.base.get(key):
                    raise ValueError("pair eval base identity mismatch for %s" % key)

    def _load_dataset(self):
        from utils.util import instantiate_from_config
        return ConsecutivePairDataset(
            instantiate_from_config(self.cfg.data.params.test),
            self.args.kitti_root,
            grid=(self.args.latent_grid_height, self.args.latent_grid_width),
            depth_candidates=self.args.geometry_depth_candidates,
            build_geometry=True,
        )

    @staticmethod
    def _pair_sample_ids(dataset: Any, pair_index: int):
        prev_i, cur_i = dataset.pairs[pair_index]
        return dataset.base.records[prev_i].get("sample_id"), dataset.base.records[cur_i].get("sample_id")

    def _row_from_pair(self, dataset: Any, pair_index: int, ordinal: int) -> Dict[str, Any]:
        prev_id, cur_id = self._pair_sample_ids(dataset, pair_index)
        return {
            "ordinal": int(ordinal),
            "pair_index": int(pair_index),
            "previous": prev_id,
            "current": cur_id,
            "seed": int(getattr(self.args, "seed", 0)) + 900000 + int(ordinal),
        }

    def _evenly_spaced_indices(self, total: int, count: int) -> List[int]:
        if count < 1:
            raise ValueError("pair_eval_count must be positive")
        if total < 1:
            raise ValueError("heldout dataset has no consecutive pairs")
        if count == 1:
            return [0]
        if count >= total:
            return list(range(total))
        return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})

    def _selected_entries(self, settings: Optional[Mapping[str, Any]], dataset: Any) -> List[Dict[str, Any]]:
        count = int(getattr(self.args, "pair_eval_count", 8))
        if settings is not None:
            rows = list(settings.get("subset", {}).get("heldout", []))[:count]
            if not rows:
                raise ValueError("pair eval settings must contain subset.heldout entries")
            selected = []
            for ordinal, row in enumerate(rows):
                pair_index = int(row["pair_index"])
                if pair_index < 0 or pair_index >= len(dataset.pairs):
                    raise ValueError("pair eval pair index out of range: %d" % pair_index)
                prev_id, cur_id = self._pair_sample_ids(dataset, pair_index)
                if prev_id != row.get("previous") or cur_id != row.get("current"):
                    raise ValueError("pair eval pair index/sample-id mismatch at %d" % pair_index)
                selected.append(self._row_from_pair(dataset, pair_index, ordinal))
            return selected
        indices = self._evenly_spaced_indices(len(dataset.pairs), count)
        return [self._row_from_pair(dataset, pair_index, ordinal) for ordinal, pair_index in enumerate(indices)]

    def _persist_selection(self, settings: Optional[Mapping[str, Any]]) -> None:
        if self.rank != 0:
            return
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "history_source": "previous_gt",
            "settings_source": str(self._settings_path()) if self._settings_path() else "deterministic_evenly_spaced_heldout",
            "source_settings_base": settings.get("base") if settings else None,
            "selected_pairs": self.selection,
            "args": {
                "pair_eval_count": int(getattr(self.args, "pair_eval_count", 8)),
                "pair_eval_ddim_steps": int(getattr(self.args, "pair_eval_ddim_steps", 50)),
                "pair_eval_guidance": float(getattr(self.args, "pair_eval_guidance", 7.5)),
                "pair_eval_off": bool(getattr(self.args, "pair_eval_off", False)),
                "geometry_depth_candidates": int(self.args.geometry_depth_candidates),
                "latent_grid_height": int(self.args.latent_grid_height),
                "latent_grid_width": int(self.args.latent_grid_width),
                "seed": int(getattr(self.args, "seed", 0)),
            },
        }
        (self.eval_dir / "settings.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _encode_pair(self, dataset: Any, row: Mapping[str, Any]) -> Dict[str, Any]:
        batch = collate_pairs([dataset[int(row["pair_index"])]])
        cond, cur_rgb = encode_conditions(self.model, batch["cur"], self.device)
        cur_latent = encode_latent(self.model, cur_rgb)
        prev_batch = _to_device(batch["prev"], self.device)
        prev_rgb = self.model.get_input(prev_batch, "grd_left_imgs").clamp(0, 1)
        prev_latent = encode_latent(self.model, prev_rgb)
        history = build_history(prev_latent, batch["geometries"])
        record = {
            "meta": dict(row),
            "cond": cond,
            "prev_rgb": prev_rgb,
            "prev_latent": prev_latent,
            "cur_rgb": cur_rgb,
            "cur_latent": cur_latent,
            "history": history,
            "geometry_metrics": batch.get("geometry_metrics", [{}])[0],
        }
        return split_tree(record, 0)

    @torch.no_grad()
    def prepare(self):
        settings = self._load_settings()
        self._validate_base(settings)
        dataset = self._load_dataset()
        self.selection = self._selected_entries(settings, dataset)
        self._persist_selection(settings)
        local_rows = self.selection[self.rank::self.world]
        self.cache = [self._encode_pair(dataset, row) for row in local_rows]
        print(json.dumps({"event": "gt_pair_eval_cache_ready", "rank": self.rank, "pairs": len(self.cache)}), flush=True)
        return self

    def _gather_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.distributed:
            gathered = [None] * self.world
            torch.distributed.all_gather_object(gathered, rows)
            return [row for part in gathered for row in part]
        return list(rows)

    def _sample(self, cond: Mapping[str, Any], shape: Sequence[int], seed: int, history: Optional[Mapping[str, Any]]):
        with preserve_torch_rng(self.device), unwrapped_denoiser(self.model):
            return sample_frame(
                self.model,
                cond,
                tuple(shape),
                int(seed),
                self.device,
                history=history,
                steps=int(getattr(self.args, "pair_eval_ddim_steps", 50)),
                guidance=float(getattr(self.args, "pair_eval_guidance", 7.5)),
            )

    def _write_pair_outputs(self, step_dir: Path, row: Mapping[str, Any], prev_rgb: torch.Tensor,
                            generated: torch.Tensor, current_gt: torch.Tensor,
                            off_rgb: Optional[torch.Tensor] = None) -> Dict[str, str]:
        name = "pair_%03d" % int(row["ordinal"])
        pair_dir = step_dir / name
        save_rgb(prev_rgb, pair_dir / "prev_gt.png")
        save_rgb(generated, pair_dir / "generated.png")
        save_rgb(current_gt, pair_dir / "current_gt.png")
        panel = torch.cat([prev_rgb, generated, current_gt] + ([off_rgb] if off_rgb is not None else []), dim=-1)
        save_rgb(panel, pair_dir / "comparison.png")
        return {
            "prev_gt": str(pair_dir / "prev_gt.png"),
            "generated": str(pair_dir / "generated.png"),
            "current_gt": str(pair_dir / "current_gt.png"),
            "comparison": str(pair_dir / "comparison.png"),
        }

    def _write_index(self, step_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
        rows = []
        for rec in sorted(records, key=lambda x: int(x["ordinal"])):
            rel = Path(rec["files"]["comparison"]).relative_to(step_dir)
            rows.append("<tr><td>%03d</td><td>%s -> %s</td><td><img src='%s'></td></tr>" %
                        (int(rec["ordinal"]), rec["previous"], rec["current"], rel.as_posix()))
        html = """<!doctype html><html><head><meta charset='utf-8'><title>GT Pair Eval</title>
<style>body{font-family:sans-serif;margin:24px}img{max-width:100%%;image-rendering:auto}td{padding:8px;border-top:1px solid #ddd}</style>
</head><body><h1>GT Pair Eval</h1><p>Columns: previous GT reference / generated next / GT next%s.</p><table>%s</table></body></html>""" % (
            " / OFF diagnostic" if bool(getattr(self.args, "pair_eval_off", False)) else "", "\n".join(rows))
        (step_dir / "index.html").write_text(html)

    def _aggregate(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"pairs": len(records), "history_source": "previous_gt"}
        for mode in ("generated", "off"):
            selected = [r for r in records if mode in r.get("metrics", {})]
            if selected:
                summary[mode] = {
                    key: sum(float(r["metrics"][mode][key]) for r in selected) / len(selected)
                    for key in ("rgb_mae", "rgb_mse", "rgb_psnr", "edge_mae")
                }
        return summary

    @torch.no_grad()
    def evaluate(self, step: int, epoch_fraction: float):
        if not self.cache:
            self.prepare()
        step_dir = self.eval_dir / ("step_%07d" % int(step))
        rows = []
        for item in self.cache:
            item_dev = tree_to(item, self.device)
            meta = item_dev["meta"]
            seed = int(meta["seed"])
            expected_steps = int(getattr(self.args, "pair_eval_ddim_steps", 50))
            sample_shape = item_dev["history"]["latent"].shape
            z, info, noise_hash = self._sample(item_dev["cond"], sample_shape, seed, item_dev["history"])
            require_finite("generated latent", z)
            rgb = decode(self.model, z)
            require_finite("generated rgb", rgb)
            require_finite("current gt rgb", item_dev["cur_rgb"])
            history_info = checked_history_info(info, expected_steps, expect_history=True)
            metrics = {"generated": diagnostic_metrics(rgb, item_dev["cur_rgb"])}
            off_rgb = None
            off_noise_hash = None
            off_history_info = None
            if bool(getattr(self.args, "pair_eval_off", False)):
                z_off, off_info, off_noise_hash = self._sample(item_dev["cond"], sample_shape, seed, None)
                if off_noise_hash != noise_hash:
                    raise RuntimeError("OFF diagnostic did not reuse the same initial noise")
                require_finite("off latent", z_off)
                off_rgb = decode(self.model, z_off)
                require_finite("off rgb", off_rgb)
                off_history_info = checked_history_info(off_info, expected_steps, expect_history=False)
                metrics["off"] = diagnostic_metrics(off_rgb, item_dev["cur_rgb"])
            files = {}
            if self.rank == 0 or self.distributed:
                files = self._write_pair_outputs(step_dir, meta, item_dev["prev_rgb"], rgb, item_dev["cur_rgb"], off_rgb)
            rows.append({
                "ordinal": int(meta["ordinal"]),
                "rank": self.rank,
                "pair_index": int(meta["pair_index"]),
                "previous": meta["previous"],
                "current": meta["current"],
                "seed": seed,
                "noise_hash": noise_hash,
                "off_noise_hash": off_noise_hash,
                "latent_hash": tensor_hash(z),
                "rgb_hash": tensor_hash(rgb),
                "history_source": "previous_gt",
                "history_used_all_ddim_steps": bool(history_info["history_active"]),
                "sampler_info": history_info,
                "off_sampler_info": off_history_info,
                "geometry": item_dev.get("geometry_metrics", {}),
                "metrics": metrics,
                "files": files,
            })
        all_rows = sorted(self._gather_rows(rows), key=lambda x: int(x["ordinal"]))
        if self.rank == 0:
            step_dir.mkdir(parents=True, exist_ok=True)
            for row in all_rows:
                (step_dir / ("pair_%03d.json" % int(row["ordinal"]))).write_text(json.dumps(row, indent=2, sort_keys=True))
            record = {
                "step": int(step),
                "epoch_fraction": float(epoch_fraction),
                "history_source": "previous_gt",
                "history_used_all_ddim_steps": all(bool(row["history_used_all_ddim_steps"]) for row in all_rows),
                "sampler": {"ddim_steps": int(getattr(self.args, "pair_eval_ddim_steps", 50)), "eta": 0.0,
                            "guidance": float(getattr(self.args, "pair_eval_guidance", 7.5))},
                "base_checkpoint": self.base,
                "temporal_checkpoint": getattr(self.args, "temporal_checkpoint", None),
                "summary": self._aggregate(all_rows),
                "elapsed_sec": round(time.time() - self.start, 3),
            }
            (step_dir / "metrics.json").write_text(json.dumps({"metrics": record, "pairs": all_rows}, indent=2, sort_keys=True))
            with (self.eval_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._write_index(step_dir, all_rows)
            print(json.dumps({"event": "gt_pair_eval", "step": int(step), "summary": record["summary"]}, sort_keys=True), flush=True)
        if self.distributed:
            torch.distributed.barrier()
        return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="single-frame base checkpoint")
    p.add_argument("--temporal-checkpoint", required=True)
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--kitti-root", required=True)
    p.add_argument("--sd-base-ckpt", required=True)
    p.add_argument("--lidar-pixel-feature-cache-root", required=True)
    p.add_argument("--image-semantic-cache-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pair-eval-settings", default="")
    p.add_argument("--fixed-eval-settings", default="")
    p.add_argument("--pair-eval-count", type=int, default=8)
    p.add_argument("--pair-eval-ddim-steps", type=int, default=50)
    p.add_argument("--pair-eval-guidance", type=float, default=7.5)
    p.add_argument("--pair-eval-off", action="store_true")
    p.add_argument("--geometry-depth-candidates", type=int, default=16)
    p.add_argument("--latent-grid-height", type=int, default=16)
    p.add_argument("--latent-grid-width", type=int, default=64)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--device", default="cuda:4")
    p.add_argument("--temporal-mode", choices=["geometry", "content"], default="geometry")
    p.add_argument("--temporal-hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--temporal-lr", type=float, default=1e-4)
    p.add_argument("--decoder-lr", type=float, default=2e-6)
    return p.parse_args(argv)


def validate_args(args) -> None:
    if min(int(args.pair_eval_count), int(args.pair_eval_ddim_steps)) < 1:
        raise ValueError("pair eval count and DDIM steps must be positive")
    if not math.isfinite(float(args.pair_eval_guidance)):
        raise ValueError("pair eval guidance must be finite")


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    device = evaluation_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    model, base, cfg = load_base(args, device)
    unet = model.DDPM.denoise_model
    if not hasattr(unet, "configure_temporal_history"):
        raise ValueError("UNet lacks configure_temporal_history")
    unet.configure_temporal_history(enabled=True, mode=args.temporal_mode, hidden_dim=args.temporal_hidden_dim)
    payload = load_temporal_checkpoint(args.temporal_checkpoint, model, base, strict=True)
    if payload.get("args", {}).get("temporal_mode") != args.temporal_mode:
        raise ValueError("evaluation history mode differs from training")
    trained_step = int(payload.get("step", 0))
    del payload
    gc.collect()
    model.eval().requires_grad_(False)
    info = {"rank": 0, "world": 1, "device": device, "distributed": False, "is_main": True}
    evaluator = GTPairEvaluator(model, cfg, args, info, out_dir, base=base)
    evaluator.evaluate(step=trained_step, epoch_fraction=0.0)


if __name__ == "__main__":
    main()
