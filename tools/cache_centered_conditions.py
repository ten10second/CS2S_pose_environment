#!/usr/bin/env python3
"""Cache frozen conditions and target latents for centered static history training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.train_static_history import (
    IMAGE_SIZE,
    evaluation_device,
    index_dataset,
    load_settings_args,
    manifest_key_for_split,
    previous_rgb_sample,
    sample_from_entry_dataset,
    validate_pair_splits,
)
from tools.train_temporal_pairs import encode_conditions, encode_latent, load_base, stack_samples


REFERENCE_ALIASES = ("reference_npz", "reference", "reference_path", "path", "npz")
RGB_PREV_ALIASES = ("previous_rgb", "prev_rgb", "warped_source_rgb")
RGB_CUR_ALIASES = ("current_rgb", "cur_rgb", "target_rgb")
SOURCE_INDEX_ALIASES = ("source_flat_index", "source_index", "flat_source_index")
VALID_ALIASES = ("valid_mask", "support_mask", "dense_valid")
MEASURED_ALIASES = ("measured_mask", "dense_measured")
ESTIMATED_ALIASES = ("estimated_mask", "dense_estimated")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--wait-reference-seconds", type=int, default=0)
    return parser.parse_args(argv)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(tensor.shape)).encode())
    h.update(str(tensor.dtype).encode())
    h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def find_data_manifest(data_root: Path) -> Path:
    candidates = [data_root / name for name in ("pairs.json", "data_manifest.json", "manifest.json")]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("data root must contain manifest.json, data_manifest.json, or pairs.json")


def manifest_entries(payload: Any) -> list[dict]:
    entries = payload.get("items", payload.get("pairs", payload)) if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("data manifest must contain a non-empty item/pairs list")
    out = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError("manifest entry %d is not an object" % idx)
        item = dict(entry)
        item.setdefault("name", "item_%06d" % idx)
        for key in ("split", "previous", "current"):
            if key not in item or not item[key]:
                raise ValueError("manifest entry %s is missing %s" % (item["name"], key))
        if item["split"] not in {"train", "heldout", "observation"}:
            raise ValueError("manifest entry %s split must be train, heldout, or observation" % item["name"])
        if "source_split" not in item:
            item["source_split"] = item["split"]
        if item["source_split"] not in {"train", "heldout"}:
            raise ValueError("manifest entry %s source_split must be train or heldout" % item["name"])
        out.append(item)
    return out


def normalized_for_split_validation(entries: Sequence[Mapping[str, Any]]) -> list[dict]:
    normalized = []
    for entry in entries:
        item = dict(entry)
        item["split"] = item["source_split"]
        normalized.append(item)
    return normalized


def reference_path(data_root: Path, entry: Mapping[str, Any]) -> Path:
    for key in REFERENCE_ALIASES:
        if key in entry and entry[key]:
            path = Path(str(entry[key]))
            return path if path.is_absolute() else data_root / path
    return data_root / str(entry["name"]) / "reference.npz"


def npz_value(data: Mapping[str, Any], aliases: Sequence[str], label: str) -> np.ndarray:
    for key in aliases:
        if key in data:
            return np.asarray(data[key])
    raise ValueError("reference NPZ is missing " + label)


def rgb_chw(array: np.ndarray, name: str) -> torch.Tensor:
    raw = np.asarray(array)
    if raw.dtype == np.uint8:
        arr = raw.astype(np.float32) / 255.0
    else:
        arr = raw.astype(np.float32)
    if arr.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        arr = arr.transpose(2, 0, 1)
    if arr.shape != (3, IMAGE_SIZE[1], IMAGE_SIZE[0]):
        raise ValueError("%s must be [3,128,512] or [128,512,3]" % name)
    if not np.isfinite(arr).all() or arr.min() < 0.0 or arr.max() > 1.0:
        raise ValueError("%s must be finite float RGB in [0,1]" % name)
    return torch.as_tensor(arr, dtype=torch.float32)


def mask_1hw(array: np.ndarray, name: str) -> torch.Tensor:
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
        raise ValueError("%s must be [1,128,512], [128,512,1], or [128,512]" % name)
    return torch.as_tensor(arr.astype(bool), dtype=torch.bool).unsqueeze(0)


def source_index_hw(array: np.ndarray) -> torch.Tensor:
    arr = np.asarray(array)
    if arr.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
        raise ValueError("source_flat_index must be [128,512]")
    out = torch.as_tensor(arr.astype(np.int64), dtype=torch.long)
    valid_values = out[out >= 0]
    if valid_values.numel() and int(valid_values.max()) >= IMAGE_SIZE[0] * IMAGE_SIZE[1]:
        raise ValueError("source_flat_index points outside previous RGB")
    return out


def load_reference(path: Path, entry: Mapping[str, Any], wait_seconds: int = 0) -> dict:
    deadline = time.time() + max(0, int(wait_seconds))
    last_error = None
    while True:
        if path.is_file():
            try:
                data = np.load(path, allow_pickle=False)
                break
            except Exception as exc:
                last_error = exc
        if time.time() >= deadline:
            if last_error is not None:
                raise RuntimeError("reference NPZ is not readable after wait: %s" % path) from last_error
            raise FileNotFoundError(path)
        time.sleep(2.0)
    previous_rgb = rgb_chw(npz_value(data, RGB_PREV_ALIASES, "previous_rgb"), "previous_rgb")
    current_rgb = rgb_chw(npz_value(data, RGB_CUR_ALIASES, "current_rgb"), "current_rgb")
    source_flat_index = source_index_hw(npz_value(data, SOURCE_INDEX_ALIASES, "source_flat_index"))
    valid = mask_1hw(npz_value(data, VALID_ALIASES, "valid mask"), "valid")
    measured = mask_1hw(npz_value(data, MEASURED_ALIASES, "measured mask"), "measured")
    estimated = mask_1hw(npz_value(data, ESTIMATED_ALIASES, "estimated mask"), "estimated")
    if torch.any(measured & estimated):
        raise ValueError("%s measured/estimated masks overlap" % entry["name"])
    if not torch.equal(valid, measured | estimated):
        raise ValueError("%s valid mask must equal measured|estimated" % entry["name"])
    if torch.any(source_flat_index[valid[0]] < 0):
        raise ValueError("%s has valid pixels with missing source_flat_index" % entry["name"])
    return {
        "previous_rgb": previous_rgb,
        "current_rgb": current_rgb,
        "source_flat_index": source_flat_index,
        "valid": valid,
        "measured": measured,
        "estimated": estimated,
        "sha256": file_sha256(path),
        "path": str(path),
    }


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(item) for item in value)
    return value


def tree_hash(value: Any) -> str:
    h = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            h.update(b"T")
            h.update(tensor_hash(item).encode())
        elif isinstance(item, Mapping):
            h.update(b"D")
            for key in sorted(item):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(b"L")
            h.update(str(len(item)).encode())
            for element in item:
                visit(element)
        elif item is None:
            h.update(b"N")
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def build_warped_rgb(previous_rgb: torch.Tensor, source_flat_index: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    flat = previous_rgb.reshape(3, -1)
    clamped = source_flat_index.clamp_min(0).reshape(-1)
    gathered = flat[:, clamped].reshape(3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    return gathered * valid.float()


def compare_rgb(label: str, cached: torch.Tensor, dataset: torch.Tensor, tolerance: float = 1.0 / 255.0 + 1e-5) -> dict:
    diff = (cached - dataset).abs()
    max_abs = float(diff.max().cpu())
    mean_abs = float(diff.mean().cpu())
    if max_abs > tolerance:
        raise ValueError("%s differs from dataset transform by %.6f" % (label, max_abs))
    return {
        label + "_max_abs": max_abs,
        label + "_mean_abs": mean_abs,
        label + "_u8_rounding_only": bool(max_abs <= tolerance),
    }


def build_datasets(cfg):
    from utils.util import instantiate_from_config

    return {
        "train_manifest": instantiate_from_config(cfg.data.params.train),
        "val_manifest": instantiate_from_config(cfg.data.params.test),
    }


def dataset_indexes(datasets: Mapping[str, Any]) -> dict:
    return {name: {"dataset": dataset, "index": index_dataset(dataset)} for name, dataset in datasets.items()}


def dataset_split(entry: Mapping[str, Any]) -> str:
    return str(entry.get("source_split", entry["split"]))


def row_sample(indexes: Mapping[str, Mapping[str, Any]], entry: Mapping[str, Any], sample_id: str) -> dict:
    manifest_key = manifest_key_for_split(dataset_split(entry))
    item = indexes[manifest_key]
    idx = item["index"].get(sample_id)
    if idx is None:
        raise KeyError("%s not found in %s" % (sample_id, manifest_key))
    return item["dataset"][idx]


def previous_rgb_from_dataset(indexes: Mapping[str, Mapping[str, Any]], entry: Mapping[str, Any]) -> torch.Tensor:
    manifest_key = manifest_key_for_split(dataset_split(entry))
    dataset = indexes[manifest_key]["dataset"]
    row_idx = indexes[manifest_key]["index"].get(entry["previous"])
    if row_idx is None:
        raise KeyError("%s not found in %s" % (entry["previous"], manifest_key))
    sample = previous_rgb_sample(dataset, dataset.records[row_idx])
    return sample["grd_left_imgs"].detach().float().clamp(0, 1)


def item_fingerprint(global_fingerprint: Mapping[str, Any], entry: Mapping[str, Any], ref: Mapping[str, Any]) -> str:
    payload = {
        "global": global_fingerprint,
        "name": entry["name"],
        "split": entry["split"],
        "source_split": entry.get("source_split", entry["split"]),
        "fixed_name": entry.get("fixed_name", ""),
        "previous": entry["previous"],
        "current": entry["current"],
        "reference_sha256": ref["sha256"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_cached_item(path: Path, expected_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        item = torch.load(path, map_location="cpu")
    except Exception:
        return False
    meta = item.get("cache_metadata", {})
    if meta.get("item_fingerprint") != expected_hash:
        return False
    required = {
        "name", "split", "previous", "current", "cond", "z_cur", "previous_rgb", "current_rgb",
        "source_flat_index", "valid", "measured", "estimated", "source_flat_index", "cache_metadata",
    }
    if not required <= set(item):
        return False
    checks = [
        torch.is_tensor(item["z_cur"]) and item["z_cur"].shape == (1, 4, 16, 64),
        torch.is_tensor(item["previous_rgb"]) and item["previous_rgb"].shape == (3, IMAGE_SIZE[1], IMAGE_SIZE[0]),
        torch.is_tensor(item["current_rgb"]) and item["current_rgb"].shape == (3, IMAGE_SIZE[1], IMAGE_SIZE[0]),
        torch.is_tensor(item["source_flat_index"]) and item["source_flat_index"].shape == (IMAGE_SIZE[1], IMAGE_SIZE[0])
        and item["source_flat_index"].dtype == torch.long,
        all(torch.is_tensor(item[key]) and item[key].shape == (1, IMAGE_SIZE[1], IMAGE_SIZE[0])
            and item[key].dtype == torch.bool for key in ("valid", "measured", "estimated")),
    ]
    return all(checks)


def main(argv=None):
    args = parse_args(argv)
    settings_path = Path(args.settings)
    data_root = Path(args.data_root)
    out = Path(args.out)
    manifest_path = find_data_manifest(data_root)
    manifest_payload = load_json(manifest_path)
    entries = manifest_entries(manifest_payload)
    names = [entry["name"] for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("cache item names collide")
    rel_paths = ["%06d_%s.pt" % (idx, entry["name"].replace("/", "_")) for idx, entry in enumerate(entries)]
    if len(set(rel_paths)) != len(rel_paths):
        raise ValueError("cache output filenames collide")
    validate_pair_splits(normalized_for_split_validation(entries))
    counts = {
        "train": sum(e["split"] == "train" for e in entries),
        "heldout": sum(e["split"] == "heldout" for e in entries),
        "observation": sum(e["split"] == "observation" for e in entries),
    }
    source_counts = {"train": sum(e["source_split"] == "train" for e in entries),
                     "heldout": sum(e["source_split"] == "heldout" for e in entries)}
    extra_unique = max(0, len({(e["previous"], e["current"]) for e in entries}) - 512 - 64)

    device = evaluation_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    settings = load_settings_args(settings_path)
    model, base, cfg = load_base(settings, device)
    model.eval().requires_grad_(False)
    datasets = build_datasets(cfg)
    indexes = dataset_indexes(datasets)

    global_fingerprint = {
        "script": "cache_centered_conditions_v1",
        "base_checkpoint": base,
        "settings_path": str(settings_path),
        "settings_sha256": file_sha256(settings_path),
        "data_root": str(data_root),
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": file_sha256(manifest_path),
        "counts": counts,
        "source_counts": source_counts,
    }
    item_dir = out / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    skipped = 0
    built = 0
    rgb_validation = []

    for idx, entry in enumerate(entries):
        ref_path = reference_path(data_root, entry)
        ref = load_reference(ref_path, entry, args.wait_reference_seconds)
        item_hash = item_fingerprint(global_fingerprint, entry, ref)
        rel_path = Path("items") / ("%06d_%s.pt" % (idx, entry["name"].replace("/", "_")))
        cache_path = out / rel_path
        if validate_cached_item(cache_path, item_hash):
            skipped += 1
        else:
            cur_sample = row_sample(indexes, entry, entry["current"])
            cur_batch = stack_samples([cur_sample])
            with torch.no_grad():
                cond, dataset_cur_rgb = encode_conditions(model, cur_batch, device)
                dataset_cur_rgb = dataset_cur_rgb.detach().float().cpu()[0].clamp(0, 1)
                z_cur = encode_latent(model, dataset_cur_rgb.unsqueeze(0).to(device).float()).detach().float().cpu()
            dataset_prev_rgb = previous_rgb_from_dataset(indexes, entry).detach().float().cpu().clamp(0, 1)
            cur_cmp = compare_rgb("current_rgb", ref["current_rgb"], dataset_cur_rgb)
            prev_cmp = compare_rgb("previous_rgb", ref["previous_rgb"], dataset_prev_rgb)
            warped_rgb = build_warped_rgb(ref["previous_rgb"], ref["source_flat_index"], ref["valid"])
            history = {
                "latent": torch.zeros_like(z_cur),
                "dense_rgb": warped_rgb.unsqueeze(0),
                "dense_valid": ref["valid"].unsqueeze(0),
                "dense_measured": ref["measured"].unsqueeze(0),
                "dense_estimated": ref["estimated"].unsqueeze(0),
                "enabled": torch.ones((1,), dtype=torch.bool),
            }
            item = {
                "name": entry["name"],
                "split": entry["split"],
                "source_split": entry["source_split"],
                "fixed_name": entry.get("fixed_name", ""),
                "previous": entry["previous"],
                "current": entry["current"],
                "cond": tree_to_cpu(cond),
                "z_cur": z_cur,
                "previous_rgb": ref["previous_rgb"],
                "current_rgb": ref["current_rgb"],
                "source_flat_index": ref["source_flat_index"],
                "valid": ref["valid"],
                "measured": ref["measured"],
                "estimated": ref["estimated"],
                "history": history,
                "cache_metadata": {
                    "item_fingerprint": item_hash,
                    "reference_path": ref["path"],
                    "reference_sha256": ref["sha256"],
                    "cond_sha256": tree_hash(tree_to_cpu(cond)),
                    "z_cur_sha256": tensor_hash(z_cur),
                    **cur_cmp,
                    **prev_cmp,
                },
            }
            atomic_torch_save(item, cache_path)
            rgb_validation.append({"name": entry["name"], **cur_cmp, **prev_cmp})
            built += 1
        manifest_rows.append({
            "name": entry["name"],
            "split": entry["split"],
            "source_split": entry["source_split"],
            "fixed_name": entry.get("fixed_name", ""),
            "previous": entry["previous"],
            "current": entry["current"],
            "path": str(cache_path.resolve()),
            "cache": str(rel_path),
            "item_fingerprint": item_hash,
            "reference": ref["path"],
            "reference_sha256": ref["sha256"],
        })
        if (idx + 1) % 10 == 0 or idx + 1 == len(entries):
            print(json.dumps({"event": "cache_progress", "done": idx + 1, "total": len(entries),
                              "built": built, "skipped": skipped}, sort_keys=True), flush=True)

    done = {
        "done": True,
        "version": "centered_condition_cache_v1",
        "built": built,
        "skipped": skipped,
        "items": len(entries),
        "counts": counts,
        "source_counts": source_counts,
        "observation_extra_unique_entries": extra_unique,
        "fingerprint": global_fingerprint,
        "manifest": "manifest.json",
    }
    atomic_json({"version": done["version"], "items": manifest_rows, **global_fingerprint}, out / "manifest.json")
    atomic_json({"rgb_validation": rgb_validation}, out / "rgb_validation.json")
    atomic_json(done, out / "cache_done.json")
    print(json.dumps(done, sort_keys=True))


if __name__ == "__main__":
    main()
