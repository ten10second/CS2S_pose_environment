#!/usr/bin/env python3
"""Summarize centered decoder ABC probes.

Read-only report generator for the A/B/C centered-history decoder probe.  It
does not start training, load a model, or require GPU access.

Expected runner contract from `tools/train_centered_decoder_probe.py`:
  run_root/
    A/eval_metrics.jsonl
    B/eval_metrics.jsonl
    C/eval_metrics.jsonl
    {A,B,C}/samples/step_0000100/{name}/{off,correct,wrong}.png
    {A,B,C}/samples/step_0000250/{name}/{off,correct,wrong}.png
    {A,B,C}/samples/step_0000500/{name}/{off,correct,wrong}.png

Group C only produces `off` images/metrics.  Missing expected files are a hard
error by default; pass `--allow-incomplete` for an explicit partial report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_RUN_ROOT = "/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/centered_decoder_abc_20260920"
DEFAULT_SELECTION = "/mnt/shizhm/CS2S_run_control/centered_decoder_abc_20260920/selection.json"
DEFAULT_REFERENCE_ROOT = "/mnt/shizhm/DATA/KITTI/CS2S_results/temporal_static/centered_a1_20260920_data"
RUNS = ("A", "B", "C")
MODES = ("off", "correct", "wrong")
SUMMARY_STEPS = (100, 250, 500)
PANEL_COLUMNS = ("prevGT", "warp", "targetGT", "Acorrect", "Bcorrect", "Boff", "Bwrong", "Coff")


def expected_modes(run: str) -> Tuple[str, ...]:
    return ("off",) if run == "C" else MODES


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2))


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = Path(path)
    if not path.is_file():
        return rows
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from error
    return rows


def infer_step(row: Mapping[str, Any]) -> Optional[int]:
    if "step" in row:
        try:
            return int(row["step"])
        except Exception:
            return None
    phase = row.get("phase")
    if isinstance(phase, str) and phase.startswith("step_"):
        try:
            return int(phase.split("_", 1)[1])
        except Exception:
            return None
    return None


def infer_group(row: Mapping[str, Any]) -> str:
    for key in ("group", "eval_group", "timestep", "bucket"):
        if key in row:
            return str(row[key])
    return "all"


def infer_loss(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("epsilon_loss", "loss", "eps_loss", "mse"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def nearest_wanted_step(step: Optional[int], wanted_steps: Sequence[int]) -> Optional[int]:
    if step is None:
        return None
    return int(step) if int(step) in set(wanted_steps) else None


def mean(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else None


def load_monitor_rows(run_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for run in RUNS:
        path = run_root / run / "eval_metrics.jsonl"
        if not path.is_file():
            # Backward-compatible fallback for older probes, but the actual ABC
            # runner writes eval_metrics.jsonl.
            path = run_root / run / "monitor.jsonl"
        rows = load_jsonl(path)
        for row in rows:
            row["_run"] = run
            row["_step"] = infer_step(row)
            row["_group"] = infer_group(row)
            row["_loss"] = infer_loss(row)
        out[run] = rows
    return out


def expected_eval_keys(selection: Mapping[str, Sequence[str]], steps: Sequence[int]) -> set[Tuple[str, int, str, str, str, str]]:
    keys = set()
    for run in RUNS:
        for step in steps:
            for split in ("train", "heldout", "observation"):
                for name in selection.get(split, []):
                    for timestep in (100, 500, 900):
                        for mode in expected_modes(run):
                            keys.add((run, int(step), split, str(name), str(timestep), mode))
    return keys


def observed_eval_keys(rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]], steps: Sequence[int]) -> set[Tuple[str, int, str, str, str, str]]:
    keys = set()
    wanted = set(int(x) for x in steps)
    for run, rows in rows_by_run.items():
        for row in rows:
            step = infer_step(row)
            if step not in wanted:
                continue
            mode = str(row.get("mode", ""))
            name = row.get("name")
            split = row.get("split")
            timestep = row.get("timestep")
            if mode and name is not None and split is not None and timestep is not None:
                keys.add((run, int(step), str(split), str(name), str(timestep), mode))
    return keys


def check_expected_eval(run_root: Path, rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]], selection: Mapping[str, Sequence[str]], steps: Sequence[int], allow_incomplete: bool) -> List[str]:
    missing_files = [str(run_root / run / "eval_metrics.jsonl") for run, rows in rows_by_run.items() if not rows]
    expected = expected_eval_keys(selection, steps)
    observed = observed_eval_keys(rows_by_run, steps)
    missing = sorted(expected - observed)
    messages = []
    if missing_files:
        messages.append("empty_or_missing_eval_metrics: " + ", ".join(missing_files))
    if missing:
        preview = "; ".join(str(x) for x in missing[:12])
        messages.append(f"missing_eval_records count={len(missing)} preview={preview}")
    if messages and not allow_incomplete:
        raise FileNotFoundError("ABC probe is incomplete: " + " | ".join(messages))
    return messages


def epsilon_summary(
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
    wanted_steps: Sequence[int],
) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, int, str, str, str], List[float]] = defaultdict(list)
    for run, rows in rows_by_run.items():
        for row in rows:
            step = nearest_wanted_step(row.get("_step"), wanted_steps)
            loss = row.get("_loss")
            mode = str(row.get("mode", ""))
            if step is None or mode not in MODES or not isinstance(loss, (int, float)):
                continue
            split = str(row.get("split", "unknown"))
            group = str(row.get("_group", "all"))
            buckets[(run, step, split, group, mode)].append(float(loss))
    records: List[Dict[str, Any]] = []
    key_prefixes = sorted({k[:4] for k in buckets})
    for run, step, split, group in key_prefixes:
        record = {"run": run, "step": step, "split": split, "group": group}
        for mode in MODES:
            values = buckets.get((run, step, split, group, mode), [])
            record[f"{mode}_mean"] = mean(values)
            record[f"{mode}_n"] = len(values)
        records.append(record)
    return records


def paired_mode_table(rows: Sequence[Mapping[str, Any]], wanted_steps: Sequence[int]) -> Dict[Tuple[int, str, str, str, Any, str], float]:
    table: Dict[Tuple[int, str, str, str, Any, str], float] = {}
    for row in rows:
        step = nearest_wanted_step(row.get("_step"), wanted_steps)
        loss = row.get("_loss")
        mode = str(row.get("mode", ""))
        name = row.get("name")
        if step is None or mode not in MODES or not isinstance(loss, (int, float)) or name is None:
            continue
        split = str(row.get("split", "unknown"))
        group = str(row.get("_group", "all"))
        seed = row.get("seed", "no_seed")
        table[(step, split, group, str(name), seed, mode)] = float(loss)
    return table


def correct_vs_wrong_wins(
    rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]],
    wanted_steps: Sequence[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for run, rows in rows_by_run.items():
        table = paired_mode_table(rows, wanted_steps)
        prefixes = sorted({key[:5] for key in table})
        buckets: Dict[Tuple[int, str, str], Dict[str, int]] = defaultdict(lambda: {"pairs": 0, "correct_wins": 0, "wrong_wins": 0, "ties": 0})
        for prefix in prefixes:
            correct = table.get(prefix + ("correct",))
            wrong = table.get(prefix + ("wrong",))
            if correct is None or wrong is None:
                continue
            bucket = buckets[(prefix[0], prefix[1], prefix[2])]
            bucket["pairs"] += 1
            if correct < wrong:
                bucket["correct_wins"] += 1
            elif correct > wrong:
                bucket["wrong_wins"] += 1
            else:
                bucket["ties"] += 1
        for (step, split, group), data in sorted(buckets.items()):
            total = max(data["pairs"], 1)
            records.append({
                "run": run,
                "step": step,
                "split": split,
                "group": group,
                **data,
                "correct_win_rate": data["correct_wins"] / float(total),
            })
    return records


def boff_vs_coff(rows_by_run: Mapping[str, Sequence[Mapping[str, Any]]], wanted_steps: Sequence[int]) -> List[Dict[str, Any]]:
    b_table = paired_mode_table(rows_by_run.get("B", []), wanted_steps)
    c_table = paired_mode_table(rows_by_run.get("C", []), wanted_steps)
    prefixes = sorted({key[:5] for key in b_table} & {key[:5] for key in c_table})
    buckets: Dict[Tuple[int, str, str], Dict[str, Any]] = defaultdict(lambda: {"pairs": 0, "b_better": 0, "c_better": 0, "ties": 0, "diffs": []})
    for prefix in prefixes:
        b = b_table.get(prefix + ("off",))
        c = c_table.get(prefix + ("off",))
        if b is None or c is None:
            continue
        bucket = buckets[(prefix[0], prefix[1], prefix[2])]
        bucket["pairs"] += 1
        diff = b - c
        bucket["diffs"].append(diff)
        if b < c:
            bucket["b_better"] += 1
        elif b > c:
            bucket["c_better"] += 1
        else:
            bucket["ties"] += 1
    records: List[Dict[str, Any]] = []
    for (step, split, group), data in sorted(buckets.items()):
        diffs = data.pop("diffs")
        records.append({
            "step": step,
            "split": split,
            "group": group,
            **data,
            "Boff_minus_Coff_mean": mean(diffs),
            "interpretation": "negative means B off has lower epsilon loss than C off under matched seed/name/group",
        })
    return records


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def read_selection(path: str | Path) -> Dict[str, List[str]]:
    data = json.loads(Path(path).read_text())
    return {str(k): [str(x) for x in v] for k, v in data.items()}


def load_ref_npz(reference_root: Path, name: str) -> Optional[Dict[str, np.ndarray]]:
    path = reference_root / name / "reference.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def image01_from_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def open_rgb01(path: str | Path, size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def step_dir(step: int) -> str:
    return "step_%07d" % int(step)


def find_sample_image(run_dir: Path, name: str, mode: str, step: int) -> Optional[Path]:
    candidates = [
        run_dir / "samples" / step_dir(step) / name / f"{mode}.png",
        run_dir / "samples" / step_dir(step) / mode / f"{name}.png",
        run_dir / "samples" / step_dir(step) / f"{name}_{mode}.png",
        run_dir / "samples" / mode / f"{name}.png",
        run_dir / "samples" / name / f"{mode}.png",
        run_dir / "generated" / mode / f"{name}.png",
        run_dir / name / f"{mode}.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted((run_dir / "samples" / step_dir(step)).glob(f"**/*{name}*{mode}*.png")) if (run_dir / "samples" / step_dir(step)).is_dir() else []
    return matches[0] if matches else None


def expected_sample_names(selection: Mapping[str, Sequence[str]], step: int) -> List[Tuple[str, str]]:
    splits = ("train", "heldout", "observation") if int(step) == 500 else ("train", "heldout")
    return [(split, name) for split in splits for name in selection.get(split, [])]


def check_expected_samples(run_root: Path, selection: Mapping[str, Sequence[str]], steps: Sequence[int], allow_incomplete: bool) -> List[str]:
    missing: List[str] = []
    for run in RUNS:
        for step in steps:
            for _split, name in expected_sample_names(selection, int(step)):
                for mode in expected_modes(run):
                    if find_sample_image(run_root / run, name, mode, int(step)) is None:
                        missing.append(f"{run}/{step_dir(int(step))}/{name}/{mode}.png")
    if missing and not allow_incomplete:
        preview = "; ".join(missing[:20])
        raise FileNotFoundError(f"missing expected generated samples count={len(missing)} preview={preview}")
    return missing


def masked_mae(image: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[float]:
    if image.shape != target.shape:
        raise ValueError(f"shape mismatch {image.shape} != {target.shape}")
    if mask is None:
        return float(np.abs(image - target).mean())
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != image.shape[:2] or not mask.any():
        return None
    return float(np.abs(image[mask] - target[mask]).mean())


def edge_strength(image: np.ndarray) -> float:
    gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    return float(gx.mean() + gy.mean())


def generation_metrics(
    run_root: Path,
    reference_root: Path,
    selection: Mapping[str, Sequence[str]],
    steps: Sequence[int],
    allow_incomplete: bool,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int, str, str], Path], List[str]]:
    rows: List[Dict[str, Any]] = []
    image_paths: Dict[Tuple[str, int, str, str], Path] = {}
    missing = check_expected_samples(run_root, selection, steps, allow_incomplete)
    for step in steps:
        for split, name in expected_sample_names(selection, int(step)):
            ref = load_ref_npz(reference_root, name)
            if ref is None:
                if allow_incomplete:
                    missing.append(f"reference/{name}/reference.npz")
                    continue
                raise FileNotFoundError(f"missing reference npz for {name}")
            target = image01_from_uint8(ref["current_rgb"])
            measured = np.asarray(ref.get("measured_mask", np.zeros(target.shape[:2], bool)), dtype=bool)
            estimated = np.asarray(ref.get("estimated_mask", np.zeros(target.shape[:2], bool)), dtype=bool)
            size = (target.shape[1], target.shape[0])
            loaded: Dict[Tuple[str, str], np.ndarray] = {}
            for run in RUNS:
                for mode in expected_modes(run):
                    path = find_sample_image(run_root / run, name, mode, int(step))
                    if path is None:
                        continue
                    image_paths[(run, int(step), mode, name)] = path
                    image = open_rgb01(path, size)
                    loaded[(run, mode)] = image
                    rows.append({
                        "run": run,
                        "step": int(step),
                        "mode": mode,
                        "name": name,
                        "split": split,
                        "mae_full": masked_mae(image, target),
                        "mae_measured": masked_mae(image, target, measured),
                        "mae_estimated": masked_mae(image, target, estimated),
                        "edge_strength": edge_strength(image),
                        "edge_strength_note": "auxiliary sharpness proxy only; not a perceptual quality proof",
                        "path": str(path),
                    })
            for run in RUNS:
                for left, right in (("correct", "off"), ("correct", "wrong"), ("off", "wrong")):
                    a = loaded.get((run, left))
                    b = loaded.get((run, right))
                    if a is not None and b is not None:
                        rows.append({
                            "run": run,
                            "step": int(step),
                            "mode": f"{left}_minus_{right}",
                            "name": name,
                            "split": split,
                            "mae_full": float(np.abs(a - b).mean()),
                            "mae_measured": masked_mae(a, b, measured),
                            "mae_estimated": masked_mae(a, b, estimated),
                            "edge_strength": None,
                            "edge_strength_note": "difference MAE between generated modes",
                            "path": "",
                        })
    return rows, image_paths, missing


def to_pil(array: np.ndarray) -> Image.Image:
    arr = np.asarray(np.clip(array, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def panel_cell(image: Image.Image, label: str, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height + 22), "#111827")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), label, fill="white")
    canvas.paste(image.resize((width, height), Image.BILINEAR), (0, 22))
    return canvas


def build_panel_page(
    names: Sequence[str],
    run_root: Path,
    reference_root: Path,
    out_path: Path,
    step: int,
    allow_incomplete: bool,
    cell_size: Tuple[int, int] = (256, 64),
) -> None:
    cell_w, cell_h = cell_size
    rows: List[Image.Image] = []
    for name in names:
        ref = load_ref_npz(reference_root, name)
        if ref is None:
            continue
        prev = image01_from_uint8(ref["prev_rgb"])
        warp = np.asarray(ref["warped_rgb"], dtype=np.float32)
        target = image01_from_uint8(ref["current_rgb"])
        size = (target.shape[1], target.shape[0])
        arrays: Dict[str, Optional[np.ndarray]] = {
            "prevGT": prev,
            "warp": warp,
            "targetGT": target,
        }
        for label, run, mode in (
            ("Acorrect", "A", "correct"),
            ("Bcorrect", "B", "correct"),
            ("Boff", "B", "off"),
            ("Bwrong", "B", "wrong"),
            ("Coff", "C", "off"),
        ):
            path = find_sample_image(run_root / run, name, mode, step)
            if path is None and not allow_incomplete:
                raise FileNotFoundError(f"missing panel image: {run}/{step_dir(step)}/{name}/{mode}.png")
            arrays[label] = open_rgb01(path, size) if path is not None else None
        cells: List[Image.Image] = []
        for label in PANEL_COLUMNS:
            arr = arrays.get(label)
            image = to_pil(arr) if arr is not None else Image.new("RGB", size, "#404040")
            cells.append(panel_cell(image, f"{name} {label}" if label == "prevGT" else label, cell_w, cell_h))
        row = Image.new("RGB", (cell_w * len(cells), cell_h + 22), "#111827")
        for idx, cell in enumerate(cells):
            row.paste(cell, (idx * cell_w, 0))
        rows.append(row)
    if not rows:
        return
    page = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "#111827")
    for idx, row in enumerate(rows):
        page.paste(row, (0, idx * row.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)


def make_panels(run_root: Path, reference_root: Path, selection: Mapping[str, Sequence[str]], report_dir: Path, allow_incomplete: bool) -> List[str]:
    panel_paths: List[str] = []
    for split in ("train", "heldout"):
        names = list(selection.get(split, []))
        for page_idx in range(0, len(names), 4):
            chunk = names[page_idx:page_idx + 4]
            if not chunk:
                continue
            path = report_dir / f"panel_{split}_{page_idx // 4 + 1}.png"
            build_panel_page(chunk, run_root, reference_root, path, step=500, allow_incomplete=allow_incomplete)
            if path.is_file():
                panel_paths.append(str(path))
    return panel_paths


def markdown_report(
    report_dir: Path,
    run_root: Path,
    epsilon_rows: Sequence[Mapping[str, Any]],
    wins_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    gen_rows: Sequence[Mapping[str, Any]],
    panel_paths: Sequence[str],
    incomplete_messages: Sequence[str],
    missing_samples: Sequence[str],
) -> None:
    lines = [
        "# Centered decoder ABC probe report",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Files",
        "",
        "- `epsilon_summary.csv/json`: epsilon loss means by run, step, split, group, mode.",
        "- `correct_vs_wrong_wins.csv/json`: paired correct-vs-wrong wins.",
        "- `Boff_vs_Coff.csv/json`: matched baseline comparison; negative Boff-Coff means B off is lower.",
        "- `generation_metrics.csv/json`: RGB MAE to target GT and generated-mode difference MAE.",
        "- `panel_train_*.png`, `panel_heldout_*.png`: visual grids.",
        "",
        "## Notes",
        "",
        "- Edge strength is only a sharpness proxy, not proof of visual quality.",
        "- No instance labels are assumed or used.",
        "- Target GT is used only for reporting image MAE, not for producing samples.",
        "",
        "## Availability",
        "",
        f"- epsilon rows: {len(epsilon_rows)}",
        f"- correct/wrong win rows: {len(wins_rows)}",
        f"- Boff/Coff rows: {len(baseline_rows)}",
        f"- generation metric rows: {len(gen_rows)}",
        f"- panel pages: {len(panel_paths)}",
    ]
    if incomplete_messages or missing_samples:
        lines += [
            "",
            "## Incomplete inputs",
            "",
            "This report was generated with `--allow-incomplete`.",
            "",
        ]
        for item in list(incomplete_messages)[:20]:
            lines.append(f"- {item}")
        if missing_samples:
            lines.append(f"- missing_samples count={len(missing_samples)} preview={missing_samples[:20]}")
    report_dir.joinpath("report.md").write_text("\n".join(lines) + "\n")


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    run_root = Path(args.run_root)
    report_dir = Path(args.report_dir) if args.report_dir else run_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    selection = read_selection(args.selection)
    rows_by_run = load_monitor_rows(run_root)
    incomplete_messages = check_expected_eval(run_root, rows_by_run, selection, args.steps, args.allow_incomplete)
    epsilon_rows = epsilon_summary(rows_by_run, args.steps)
    wins_rows = correct_vs_wrong_wins(rows_by_run, args.steps)
    baseline_rows = boff_vs_coff(rows_by_run, args.steps)
    gen_rows, image_paths, missing_samples = generation_metrics(run_root, Path(args.reference_root), selection, args.steps, args.allow_incomplete)
    panel_paths = make_panels(run_root, Path(args.reference_root), selection, report_dir, args.allow_incomplete)

    outputs = {
        "epsilon_summary": epsilon_rows,
        "correct_vs_wrong_wins": wins_rows,
        "Boff_vs_Coff": baseline_rows,
        "generation_metrics": gen_rows,
        "sample_images_found": len(image_paths),
        "missing_samples": missing_samples,
        "incomplete_messages": incomplete_messages,
        "panels": panel_paths,
    }
    for stem, rows in (
        ("epsilon_summary", epsilon_rows),
        ("correct_vs_wrong_wins", wins_rows),
        ("Boff_vs_Coff", baseline_rows),
        ("generation_metrics", gen_rows),
    ):
        write_json(report_dir / f"{stem}.json", rows)
        write_csv(report_dir / f"{stem}.csv", rows)
    write_json(report_dir / "summary.json", outputs)
    markdown_report(report_dir, run_root, epsilon_rows, wins_rows, baseline_rows, gen_rows, panel_paths, incomplete_messages, missing_samples)
    return outputs


def make_fake_png(path: Path, value: float) -> None:
    arr = np.zeros((16, 32, 3), dtype=np.float32)
    arr[..., 0] = value
    arr[..., 1] = np.linspace(0, 1, arr.shape[1])[None, :]
    arr[..., 2] = np.linspace(0, 1, arr.shape[0])[:, None]
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(arr).save(path)


def run_self_test() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="centered_decoder_summary_"))
    try:
        run_root = tmp / "runs"
        ref_root = tmp / "refs"
        selection = {"train": ["train_0000"], "heldout": ["heldout_0000"], "observation": ["obs_0000"]}
        selection_path = tmp / "selection.json"
        write_json(selection_path, selection)
        for name in ["train_0000", "heldout_0000", "obs_0000"]:
            folder = ref_root / name
            folder.mkdir(parents=True)
            prev = np.full((16, 32, 3), 80, dtype=np.uint8)
            cur = np.full((16, 32, 3), 120, dtype=np.uint8)
            warp = np.full((16, 32, 3), 0.4, dtype=np.float32)
            mask = np.ones((16, 32), dtype=bool)
            np.savez_compressed(folder / "reference.npz", prev_rgb=prev, current_rgb=cur, warped_rgb=warp,
                                support_mask=mask, measured_mask=mask, estimated_mask=np.zeros_like(mask),
                                previous=name + "_prev", current=name + "_cur")
        for run in RUNS:
            run_dir = run_root / run
            run_dir.mkdir(parents=True)
            with (run_dir / "eval_metrics.jsonl").open("w") as stream:
                for step in SUMMARY_STEPS:
                    for split, names in selection.items():
                        for name in names:
                            for timestep in (100, 500, 900):
                                for mode in expected_modes(run):
                                    loss = 1.0 + 0.01 * step + 0.0001 * timestep + {"correct": 0.0, "off": 0.1, "wrong": 0.2}[mode]
                                    if run == "C" and mode == "off":
                                        loss -= 0.05
                                    stream.write(json.dumps({"step": step, "split": split, "name": name, "mode": mode,
                                                             "timestep": timestep, "seed": 1, "loss": loss}) + "\n")
                    for _split, name in expected_sample_names(selection, step):
                        for mode in expected_modes(run):
                            value = {"correct": 0.55, "off": 0.45, "wrong": 0.35}[mode]
                            make_fake_png(run_dir / "samples" / step_dir(step) / name / f"{mode}.png", value)
        args = argparse.Namespace(run_root=str(run_root), selection=str(selection_path), reference_root=str(ref_root),
                                  report_dir=str(tmp / "report"), steps=SUMMARY_STEPS, self_test=False,
                                  allow_incomplete=False)
        outputs = summarize(args)
        assert outputs["epsilon_summary"], "epsilon summary is empty"
        assert outputs["correct_vs_wrong_wins"], "wins summary is empty"
        assert outputs["Boff_vs_Coff"], "baseline summary is empty"
        assert outputs["generation_metrics"], "generation metrics is empty"
        assert outputs["panels"], "panels were not created"
        missing_path = run_root / "A" / "samples" / step_dir(500) / "train_0000" / "correct.png"
        missing_path.unlink()
        try:
            summarize(args)
        except FileNotFoundError as exc:
            assert "missing expected generated samples" in str(exc)
        else:
            raise AssertionError("strict summary should fail when an expected image is missing")
        args.allow_incomplete = True
        partial = summarize(args)
        assert partial["missing_samples"], "allow-incomplete report should record missing samples"
        print(f"SELF_TEST_PASS {tmp / 'report'}")
    finally:
        if not (tmp / "KEEP").exists():
            shutil.rmtree(tmp, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--selection", default=DEFAULT_SELECTION)
    parser.add_argument("--reference-root", default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--steps", type=int, nargs="+", default=list(SUMMARY_STEPS))
    parser.add_argument("--allow-incomplete", action="store_true", help="write a partial report instead of raising on missing expected eval/sample outputs")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    outputs = summarize(args)
    print(json.dumps(json_safe({
        "report_dir": args.report_dir or str(Path(args.run_root) / "report"),
        "epsilon_rows": len(outputs["epsilon_summary"]),
        "correct_vs_wrong_rows": len(outputs["correct_vs_wrong_wins"]),
        "Boff_vs_Coff_rows": len(outputs["Boff_vs_Coff"]),
        "generation_metric_rows": len(outputs["generation_metrics"]),
        "sample_images_found": outputs["sample_images_found"],
        "missing_samples": len(outputs["missing_samples"]),
        "incomplete_messages": len(outputs["incomplete_messages"]),
        "panels": outputs["panels"],
    }), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
