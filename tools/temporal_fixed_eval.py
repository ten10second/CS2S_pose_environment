"""Fixed temporal validation cache/evaluator for persistent-history training."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.cuda.amp import autocast

TIMESTEPS = (50, 250, 500, 750, 950)
MODES = ("correct", "off")


def concat_tree(items: Sequence[Any]) -> Any:
    first = items[0]
    if torch.is_tensor(first):
        return torch.cat(items, dim=0)
    if isinstance(first, dict):
        return {k: concat_tree([x[k] for x in items]) for k in first}
    if isinstance(first, tuple):
        return tuple(concat_tree([x[i] for x in items]) for i in range(len(first)))
    if isinstance(first, list):
        return [concat_tree([x[i] for x in items]) for i in range(len(first))]
    if first is None:
        return None
    raise TypeError(type(first))


def split_tree(data: Any, index: int) -> Any:
    if torch.is_tensor(data):
        return data[index:index + 1].detach().cpu()
    if isinstance(data, dict):
        return {k: split_tree(v, index) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(split_tree(v, index) for v in data)
    if isinstance(data, list):
        return [split_tree(v, index) for v in data]
    if data is None:
        return None
    raise TypeError(type(data))


def tree_to(data: Any, device: torch.device) -> Any:
    if torch.is_tensor(data):
        return data.to(device=device, non_blocking=False)
    if isinstance(data, dict):
        return {k: tree_to(v, device) for k, v in data.items()}
    if isinstance(data, tuple):
        return tuple(tree_to(v, device) for v in data)
    if isinstance(data, list):
        return [tree_to(v, device) for v in data]
    return data


class FixedTemporalEvaluator:
    """Prepare and run the prior fixed 128-pair probe evaluation inside training."""

    def __init__(self, model: Any, cfg: Any, args: Any, info: Mapping[str, Any], out_dir: str | Path, base=None):
        self.model = model
        self.base = base
        self.cfg = cfg
        self.args = args
        self.info = dict(info)
        self.rank = int(self.info.get("rank", 0))
        self.world = int(self.info.get("world", 1))
        self.device = self.info.get("device", torch.device("cpu"))
        self.distributed = bool(self.info.get("distributed", False))
        self.out_dir = Path(out_dir)
        self.eval_dir = self.out_dir / "fixed_eval"
        self.settings = None
        self.cache: Dict[str, List[Dict[str, Any]]] = {}
        self.start = time.time()

    def _settings_path(self) -> Path:
        path = getattr(self.args, "fixed_eval_settings", "")
        if not path:
            raise ValueError("args.fixed_eval_settings is required for fixed temporal evaluation")
        return Path(path)

    def _load_settings(self) -> Mapping[str, Any]:
        settings = json.loads(self._settings_path().read_text())
        if "subset" not in settings or "train" not in settings["subset"] or "heldout" not in settings["subset"]:
            raise ValueError("fixed eval settings must contain subset.train and subset.heldout")
        return settings

    def _validate_base(self, settings: Mapping[str, Any]) -> None:
        observed = self.base
        if observed is None:
            return
        expected = settings.get("base")
        if expected is None:
            raise ValueError("settings has no base identity to validate against")
        for key in ("sha256", "denoise_model_sha256", "step"):
            if key in expected or key in observed:
                if expected.get(key) != observed.get(key):
                    raise ValueError("fixed eval base identity mismatch for %s" % key)

    def _load_datasets(self):
        from utils.util import instantiate_from_config
        from tools.train_temporal_pairs import ConsecutivePairDataset
        train = ConsecutivePairDataset(instantiate_from_config(self.cfg.data.params.train), self.args.kitti_root,
                                       depth_candidates=self.args.geometry_depth_candidates)
        heldout = ConsecutivePairDataset(instantiate_from_config(self.cfg.data.params.test), self.args.kitti_root,
                                         depth_candidates=self.args.geometry_depth_candidates)
        return {"train": train, "heldout": heldout}

    @staticmethod
    def _pair_sample_ids(dataset: Any, pair_index: int):
        prev_i, cur_i = dataset.pairs[pair_index]
        return dataset.base.records[prev_i].get("sample_id"), dataset.base.records[cur_i].get("sample_id")

    def _selected_entries(self, settings: Mapping[str, Any], datasets: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
        if self.world != 2:
            raise ValueError("fixed temporal evaluator requires world size 2")
        selected = {}
        specs = {
            "train_monitor": ("train", settings["subset"]["train"][self.rank::self.world][:8], 8),
            "heldout": ("heldout", settings["subset"]["heldout"][self.rank::self.world], 8),
        }
        for out_name, (dataset_name, rows, expected_count) in specs.items():
            if len(rows) != expected_count:
                raise ValueError("%s expected %d local pairs, got %d" % (out_name, expected_count, len(rows)))
            dataset = datasets[dataset_name]
            checked = []
            for row in rows:
                pair_index = int(row["pair_index"])
                if pair_index < 0 or pair_index >= len(dataset.pairs):
                    raise ValueError("%s pair index out of range: %d" % (out_name, pair_index))
                prev_id, cur_id = self._pair_sample_ids(dataset, pair_index)
                if prev_id != row.get("previous") or cur_id != row.get("current"):
                    raise ValueError("%s pair index/sample-id mismatch at %d" % (out_name, pair_index))
                checked.append(row)
            selected[out_name] = checked
        return selected

    def _encode_pairs(self, dataset: Any, pair_indices):
        from tools.train_temporal_pairs import collate_pairs, encode_conditions, encode_latent, _to_device
        from ldm.modules.temporal_pair_training import build_history
        batch = collate_pairs([dataset[index] for index in pair_indices])
        cond, rgb = encode_conditions(self.model, batch["cur"], self.device)
        z = encode_latent(self.model, rgb)
        prev = _to_device(batch["prev"], self.device)
        zprev = encode_latent(self.model, self.model.get_input(prev, "grd_left_imgs").clamp(0, 1))
        history = build_history(zprev, batch["geometries"])
        return [split_tree({"cond": cond, "z": z, "history": history}, i) for i in range(len(pair_indices))]

    def prepare(self):
        self.settings = self._load_settings()
        self._validate_base(self.settings)
        datasets = self._load_datasets()
        selected = self._selected_entries(self.settings, datasets)
        self.cache = {"train_monitor": [], "heldout": []}
        for split, rows in selected.items():
            dataset = datasets["train" if split == "train_monitor" else "heldout"]
            for start in range(0, len(rows), 2):
                indices = [int(row["pair_index"]) for row in rows[start:start + 2]]
                self.cache[split].extend(self._encode_pairs(dataset, indices))
            print(json.dumps({"event": "fixed_eval_cache_ready", "rank": self.rank, "split": split, "pairs": len(rows)}), flush=True)
        return self

    def _noise_seed(self, split: str, local_batch_start: int, timestep: int) -> int:
        settings_args = (self.settings or {}).get("args", {})
        probe_seed = int(settings_args.get("seed", self.args.seed))
        return probe_seed + 700000 + self.rank * 10007 + local_batch_start * 101 + int(timestep) + (100000 if split == "heldout" else 0)

    def _noise(self, shape, dtype, device, seed: int) -> torch.Tensor:
        with torch.random.fork_rng(devices=[device.index] if getattr(device, "type", None) == "cuda" else []):
            gen = torch.Generator(device=device).manual_seed(int(seed))
            return torch.randn(shape, generator=gen, device=device, dtype=dtype)

    def _gather_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.distributed:
            gathered = [None] * self.world
            torch.distributed.all_gather_object(gathered, rows)
            return [row for part in gathered for row in part]
        return list(rows)

    def _summary(self, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        summary = {}
        for split in ("train_monitor", "heldout"):
            selected = [r for r in rows if r["split"] == split]
            means = {mode: sum(float(r[mode]) for r in selected) / len(selected) for mode in MODES}
            means["history_gain_percent"] = 100.0 * (means["off"] - means["correct"]) / means["off"]
            per_pair = {}
            for r in selected:
                per_pair.setdefault((r["rank"], r["local_sample"]), []).append(float(r["off"]) - float(r["correct"]))
            means["pairs_correct_better"] = sum(sum(v) > 0 for v in per_pair.values())
            means["pairs"] = len(per_pair)
            means["evaluated_pair_timesteps"] = len(selected)
            summary[split] = means
        return summary

    @torch.no_grad()
    def evaluate(self, step: int, epoch_fraction: float):
        if not self.cache:
            self.prepare()
        from ldm.modules.temporal_pair_training import raw_unet
        unet = raw_unet(self.model)
        allowed = {"context", "lidar_context", "lidar_evidence", "lidar_geometry_mask", "control_grd",
                   "left_camera_k", "gt_shift_x", "gt_shift_y", "theta"}
        rows = []
        for split, items in self.cache.items():
            for bi in range(0, len(items), 2):
                batch = tree_to(concat_tree(items[bi:bi + 2]), self.device)
                cond = {k: v for k, v in batch["cond"].items() if k in allowed}
                for tvalue in TIMESTEPS:
                    noise = self._noise(batch["z"].shape, batch["z"].dtype, self.device, self._noise_seed(split, bi, tvalue))
                    t = torch.full((batch["z"].shape[0],), int(tvalue), device=self.device, dtype=torch.long)
                    x = self.model.DDPM.q_sample(x_start=batch["z"], t=t, noise=noise)
                    losses = {}
                    for mode in MODES:
                        history = batch["history"] if mode == "correct" else None
                        with autocast(enabled=getattr(self.device, "type", None) == "cuda"):
                            pred = unet(x, t, history=history, **cond)
                        mse = (pred.float() - noise.float()).square().flatten(1).mean(1)
                        if not torch.isfinite(mse).all():
                            raise RuntimeError("non-finite fixed temporal eval")
                        losses[mode] = mse.detach().cpu().tolist()
                    for j in range(len(losses["correct"])):
                        rows.append({"split": split, "rank": self.rank, "local_sample": bi + j, "t": int(tvalue),
                                     "correct": losses["correct"][j], "off": losses["off"][j]})
        all_rows = self._gather_rows(rows)
        if self.rank == 0:
            self.eval_dir.mkdir(parents=True, exist_ok=True)
            record = {"step": int(step), "epoch_fraction": float(epoch_fraction), "summary": self._summary(all_rows),
                      "elapsed_sec": round(time.time() - self.start, 3)}
            (self.eval_dir / ("step_%07d.json" % int(step))).write_text(json.dumps({"metrics": record, "per_pair_timestep": all_rows}, indent=2, sort_keys=True))
            with (self.eval_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({"event": "fixed_temporal_eval", "step": int(step), "summary": record["summary"]}, sort_keys=True), flush=True)
        if self.distributed:
            torch.distributed.barrier()
        return rows
