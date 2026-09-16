"""Summarize fixed geometry-history evaluation records.

This intentionally treats RGB denoising as the primary signal. Total loss is
not aggregated here because it can hide a weak RGB branch behind auxiliary
depth improvements.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


CONDITIONS = ("disabled", "correct", "wrong_history", "wrong_geometry")
BASELINES = ("disabled", "wrong_history", "wrong_geometry")
REGIONS = ("all", "valid", "invalid")
TIMESTEPS = (250, 750)


class EvaluationSummaryError(ValueError):
    """Raised when evaluation records are incomplete or internally inconsistent."""


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_file"] = path.name
            row["_line"] = line_number
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_float(value, name: str) -> float:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise EvaluationSummaryError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def quantiles(values: Sequence[float]) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    if vals.size == 0:
        return {"median": None, "q25": None, "q75": None}
    return {
        "median": float(np.quantile(vals, 0.50)),
        "q25": float(np.quantile(vals, 0.25)),
        "q75": float(np.quantile(vals, 0.75)),
    }


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return float(np.asarray(vals, dtype=np.float64).mean())


def ci95(values: Sequence[float]) -> dict:
    if not values:
        return {"low": None, "high": None}
    vals = np.asarray(values, dtype=np.float64)
    return {"low": float(np.quantile(vals, 0.025)), "high": float(np.quantile(vals, 0.975))}


def evaluation_paths(eval_dir: Path) -> List[Path]:
    paths = sorted(eval_dir.glob("evaluation_rank*.jsonl")) + sorted(eval_dir.glob("original_rank*.jsonl"))
    if not paths:
        raise EvaluationSummaryError(f"no evaluation_rank*.jsonl or original_rank*.jsonl files found in {eval_dir}")
    return paths


def reproduction_paths(eval_dir: Path) -> List[Path]:
    return sorted(eval_dir.glob("reproduction_rank*.json"))


def parse_rank(path: Path) -> Optional[int]:
    marker = "_rank"
    if marker not in path.stem:
        return None
    rank_text = path.stem.rsplit(marker, 1)[1]
    if rank_text.isdigit():
        return int(rank_text)
    return None


def selected_pair_ids(items) -> set[str]:
    ids: set[str] = set()
    if not isinstance(items, list):
        return ids
    for item in items:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, dict):
            pair_id = item.get("pair_id", item.get("id"))
            if isinstance(pair_id, str):
                ids.add(pair_id)
    return ids


def parse_selection(selection) -> dict:
    """Extract the concrete selection shape written by the evaluator."""

    if not isinstance(selection, dict):
        raise EvaluationSummaryError("selection.json must be an object")
    timesteps = selection.get("timesteps", list(TIMESTEPS))
    if not isinstance(timesteps, list) or not timesteps:
        raise EvaluationSummaryError("selection.timesteps must be a non-empty list")
    world_size = selection.get("world_size")
    if world_size is not None and (not isinstance(world_size, int) or world_size <= 0):
        raise EvaluationSummaryError("selection.world_size must be a positive integer when present")
    expanded_ids = selected_pair_ids(selection.get("expanded"))
    original_present = "original" in selection
    original_ids = selected_pair_ids(selection.get("original")) if original_present else set()
    if not expanded_ids:
        raise EvaluationSummaryError("selection.json did not expose expanded pair ids in a supported format")
    if original_present and not original_ids:
        raise EvaluationSummaryError("selection.original is present but has no supported pair ids")
    return {
        "expanded_ids": expanded_ids,
        "original_ids": original_ids,
        "original_present": original_present,
        "timesteps": tuple(int(t) for t in timesteps),
        "world_size": world_size,
    }



def validate_rank_files(paths: Sequence[Path], expected_ranks: Optional[set[int]], label: str) -> None:
    if expected_ranks is None:
        return
    ranks = {rank for path in paths for rank in [parse_rank(path)] if rank is not None}
    missing = sorted(expected_ranks.difference(ranks))
    extra = sorted(ranks.difference(expected_ranks))
    if missing or extra:
        raise EvaluationSummaryError(f"{label} rank coverage mismatch: missing={missing}, extra={extra}")


def validate_reproduction(eval_dir: Path, world_size: Optional[int]) -> dict:
    paths = reproduction_paths(eval_dir)
    if not paths:
        raise EvaluationSummaryError("missing reproduction_rank*.json files")
    expected_ranks = set(range(world_size)) if world_size else None
    validate_rank_files(paths, expected_ranks, "reproduction")
    statuses = {}
    failures = []
    for path in paths:
        payload = read_json(path)
        passed = payload.get("passed") is True
        statuses[path.name] = payload
        if not passed:
            failures.append(path.name)
    if failures:
        raise EvaluationSummaryError(f"reproduction checks failed: {failures}")
    completion_statuses = {}
    completion_failures = []
    completion_paths = sorted(eval_dir.glob("*_complete_rank*.json"))
    if expected_ranks is not None and not completion_paths:
        raise EvaluationSummaryError("missing *_complete_rank*.json files")
    # Original replay completion cannot stand in for expanded-run completion.
    validate_rank_files(sorted(eval_dir.glob("expanded_complete_rank*.json")), expected_ranks, "completion")
    for path in completion_paths:
        payload = read_json(path)
        completion_statuses[path.name] = payload
        passed = payload.get("passed", payload.get("complete", True))
        if passed is not True:
            completion_failures.append(path.name)
    if completion_failures:
        raise EvaluationSummaryError(f"completion checks failed: {completion_failures}")
    return {
        "files": [path.name for path in paths],
        "statuses": statuses,
        "completion_files": sorted(completion_statuses),
        "completion_statuses": completion_statuses,
    }


def region_meta(row: dict) -> Tuple[float, int, int]:
    region = row.get("region_eps")
    if not isinstance(region, dict):
        raise EvaluationSummaryError(f"row {row_id(row)} missing region_eps object")
    valid_fraction = as_float(region.get("valid_fraction"), "region_eps.valid_fraction")
    valid_cells = region.get("valid_cells")
    total_cells = region.get("total_cells")
    if not isinstance(valid_cells, int) or not isinstance(total_cells, int):
        raise EvaluationSummaryError(f"row {row_id(row)} valid_cells/total_cells must be integers")
    if total_cells <= 0 or valid_cells < 0 or valid_cells > total_cells:
        raise EvaluationSummaryError(f"row {row_id(row)} has invalid valid_cells/total_cells")
    if valid_fraction < 0.0 or valid_fraction > 1.0:
        raise EvaluationSummaryError(f"row {row_id(row)} valid_fraction out of range")
    if valid_fraction != (valid_cells / total_cells):
        raise EvaluationSummaryError(f"row {row_id(row)} valid_fraction does not equal valid_cells/total_cells")
    if region.get("all") is None:
        raise EvaluationSummaryError(f"row {row_id(row)} region_eps.all must be non-null")
    as_float(region.get("all"), "region_eps.all")
    if valid_cells > 0:
        if region.get("valid") is None:
            raise EvaluationSummaryError(f"row {row_id(row)} region_eps.valid must be non-null when valid_cells > 0")
        as_float(region.get("valid"), "region_eps.valid")
    return valid_fraction, valid_cells, total_cells


def row_id(row: dict) -> str:
    return f"{row.get('_source_file', '?')}:{row.get('_line', '?')}"


def validate_and_group(rows: Sequence[dict], selection: dict) -> Tuple[dict, dict]:
    grouped: Dict[Tuple[str, str, str, int], dict] = defaultdict(dict)
    duplicate_keys = []
    selected_by_cohort = {"expanded": selection["expanded_ids"]}
    if selection["original_present"]:
        selected_by_cohort["original"] = selection["original_ids"]
    seen_by_cohort = defaultdict(set)
    disabled_repeat_failures = []

    for row in rows:
        required = ("cohort", "split", "pair_id", "drive", "frame_index", "t", "condition")
        missing = [key for key in required if key not in row]
        if missing:
            raise EvaluationSummaryError(f"row {row_id(row)} missing keys {missing}")
        condition = row["condition"]
        if condition not in CONDITIONS:
            raise EvaluationSummaryError(f"row {row_id(row)} has unknown condition {condition!r}")
        t = int(row["t"])
        key = (row["cohort"], row["split"], row["pair_id"], t)
        if condition in grouped[key]:
            duplicate_keys.append((*key, condition, row_id(row)))
        grouped[key][condition] = row
        if row["cohort"] in selected_by_cohort:
            seen_by_cohort[row["cohort"]].add((row["pair_id"], t))
        if condition == "disabled" and row.get("disabled_repeat_equal") is not True:
            disabled_repeat_failures.append(row_id(row))
        if condition != "disabled" and "disabled_repeat_equal" in row:
            raise EvaluationSummaryError(f"row {row_id(row)} disabled_repeat_equal is only valid for disabled rows")

    if duplicate_keys:
        raise EvaluationSummaryError(f"duplicate records found: {duplicate_keys[:5]}")

    missing_conditions = []
    inconsistent_meta = []
    depth_violations = []
    for key, conds in grouped.items():
        absent = [condition for condition in CONDITIONS if condition not in conds]
        if absent:
            missing_conditions.append({"key": key, "missing": absent})
            continue
        metas = {condition: region_meta(row) for condition, row in conds.items()}
        if len(set(metas.values())) != 1:
            inconsistent_meta.append({"key": key, "metas": metas})
        depths = {
            condition: as_float(row.get("loss_lidar_bottleneck_depth_log_l1"), "loss_lidar_bottleneck_depth_log_l1")
            for condition, row in conds.items()
        }
        if len(set(depths.values())) != 1:
            depth_violations.append({"key": key, "depths": depths})

    if missing_conditions:
        raise EvaluationSummaryError(f"incomplete condition groups: {missing_conditions[:5]}")
    if inconsistent_meta:
        raise EvaluationSummaryError(f"inconsistent valid masks/cell counts: {inconsistent_meta[:5]}")
    if depth_violations:
        raise EvaluationSummaryError(f"depth loss changed across conditions: {depth_violations[:5]}")
    if disabled_repeat_failures:
        raise EvaluationSummaryError(f"disabled repeat checks failed: {disabled_repeat_failures[:5]}")

    coverage = {"validated": True, "cohorts": {}}
    for cohort, ids in selected_by_cohort.items():
        expected = {(pair_id, timestep) for pair_id in ids for timestep in selection["timesteps"]}
        seen = seen_by_cohort.get(cohort, set())
        missing = sorted(expected.difference(seen))
        extra = sorted(seen.difference(expected))
        if missing or extra:
            raise EvaluationSummaryError(
                f"{cohort} selection coverage mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )
        coverage["cohorts"][cohort] = {
            "expected_pairs": len(ids),
            "seen_pairs": len({pair_id for pair_id, _t in seen}),
            "expected_pair_timesteps": len(expected),
            "seen_pair_timesteps": len(seen),
        }
    coverage["expected_expanded_pairs"] = coverage["cohorts"]["expanded"]["expected_pairs"]
    coverage["seen_expanded_pairs"] = coverage["cohorts"]["expanded"]["seen_pairs"]

    return grouped, coverage


def region_loss(row: dict, region: str) -> Optional[float]:
    value = row["region_eps"].get(region)
    if value is None:
        return None
    return as_float(value, f"region_eps.{region}")


def paired_items(grouped: dict) -> List[dict]:
    items = []
    for (cohort, split, pair_id, t), conds in sorted(grouped.items()):
        correct = conds["correct"]
        item = {
            "cohort": cohort,
            "split": split,
            "pair_id": pair_id,
            "drive": correct["drive"],
            "frame_index": int(correct["frame_index"]),
            "block": int(correct["frame_index"]) // 100,
            "t": int(t),
            "valid_fraction": region_meta(correct)[0],
            "valid_cells": region_meta(correct)[1],
            "total_cells": region_meta(correct)[2],
            "regions": {},
            "attention": correct.get("attention", []),
        }
        for region in REGIONS:
            correct_loss = region_loss(correct, region)
            if correct_loss is None:
                item["regions"][region] = None
                continue
            baselines = {}
            for baseline in BASELINES:
                baseline_loss = region_loss(conds[baseline], region)
                if baseline_loss is None:
                    baselines[baseline] = None
                else:
                    baselines[baseline] = {
                        "baseline_loss": baseline_loss,
                        "correct_loss": correct_loss,
                        "gain": baseline_loss - correct_loss,
                    }
            item["regions"][region] = baselines
        items.append(item)
    return items


def summarize_gain_rows(items: Sequence[dict], baseline: str, region: str) -> dict:
    rows = []
    baseline_losses = []
    correct_losses = []
    for item in items:
        region_payload = item["regions"].get(region)
        if not region_payload:
            continue
        payload = region_payload.get(baseline)
        if not payload:
            continue
        rows.append(payload["gain"])
        baseline_losses.append(payload["baseline_loss"])
        correct_losses.append(payload["correct_loss"])
    if not rows:
        return {
            "n": 0,
            "wins": 0,
            "mean_abs": None,
            "relative_percent": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    mean_baseline = float(np.asarray(baseline_losses, dtype=np.float64).mean())
    mean_correct = float(np.asarray(correct_losses, dtype=np.float64).mean())
    rel = None if mean_baseline == 0.0 else float(100.0 * (mean_baseline - mean_correct) / mean_baseline)
    return {
        "n": len(rows),
        "wins": int(sum(value > 0.0 for value in rows)),
        "mean_abs": float(np.asarray(rows, dtype=np.float64).mean()),
        "relative_percent": rel,
        **quantiles(rows),
    }


def resample_drive_once(items: Sequence[dict], rng: np.random.Generator) -> List[dict]:
    by_block = defaultdict(list)
    for item in items:
        by_block[item["block"]].append(item)
    blocks = sorted(by_block)
    if not blocks:
        return []
    sampled = rng.choice(blocks, size=len(blocks), replace=True)
    result = []
    for block in sampled:
        result.extend(by_block[int(block)])
    return result


def resample_stratified_by_drive_once(items: Sequence[dict], rng: np.random.Generator) -> List[dict]:
    by_drive = defaultdict(list)
    for item in items:
        by_drive[item["drive"]].append(item)
    sampled = []
    for _drive, drive_items in sorted(by_drive.items()):
        sampled.extend(resample_drive_once(drive_items, rng))
    return sampled


def bootstrap_group(items: Sequence[dict], baseline: str, region: str, samples: int, seed: int) -> dict:
    if not items or samples <= 0:
        return {"mean_abs_ci95": ci95([]), "relative_percent_ci95": ci95([]), "samples": 0, "degenerate": True}
    block_count = len({(item["drive"], item["block"]) for item in items})
    drive_count = len({item["drive"] for item in items})
    rng = np.random.default_rng(seed)
    means = []
    rels = []
    for _ in range(samples):
        sampled = resample_stratified_by_drive_once(items, rng)
        stats = summarize_gain_rows(sampled, baseline, region)
        if stats["mean_abs"] is not None:
            means.append(stats["mean_abs"])
        if stats["relative_percent"] is not None:
            rels.append(stats["relative_percent"])
    return {
        "mean_abs_ci95": ci95(means),
        "relative_percent_ci95": ci95(rels),
        "samples": samples,
        "block_count": block_count,
        "drive_count": drive_count,
        "degenerate": any(
            len({item["block"] for item in items if item["drive"] == drive}) < 2
            for drive in {item["drive"] for item in items}
        ),
    }


def macro_bootstrap(drives: Dict[str, Sequence[dict]], baseline: str, region: str, samples: int, seed: int) -> dict:
    if not drives or samples <= 0:
        return {"mean_abs_ci95": ci95([]), "relative_percent_ci95": ci95([]), "samples": 0, "degenerate_drives": []}
    rng = np.random.default_rng(seed)
    means = []
    rels = []
    drive_items = sorted(drives.items())
    for _ in range(samples):
        per_drive_mean = []
        per_drive_rel = []
        for _drive, items in drive_items:
            sampled = resample_drive_once(items, rng)
            stats = summarize_gain_rows(sampled, baseline, region)
            if stats["mean_abs"] is not None:
                per_drive_mean.append(stats["mean_abs"])
            if stats["relative_percent"] is not None:
                per_drive_rel.append(stats["relative_percent"])
        if per_drive_mean:
            means.append(float(np.asarray(per_drive_mean, dtype=np.float64).mean()))
        if per_drive_rel:
            rels.append(float(np.asarray(per_drive_rel, dtype=np.float64).mean()))
    degenerate = sorted(drive for drive, items in drives.items() if len({item["block"] for item in items}) < 2)
    return {
        "mean_abs_ci95": ci95(means),
        "relative_percent_ci95": ci95(rels),
        "samples": samples,
        "degenerate_drives": degenerate,
    }


def summarize_attention(items: Sequence[dict]) -> dict:
    rows = []
    for item in items:
        for attention in item.get("attention") or []:
            if isinstance(attention, dict):
                rows.append(attention)
    if not rows:
        return {}
    fields = (
        "query_valid_fraction",
        "null_all",
        "null_valid",
        "residual_to_x_all",
        "residual_to_x_valid",
        "residual_to_cond_all",
    )
    by_block = defaultdict(list)
    for row in rows:
        by_block[str(row.get("block_index", "unknown"))].append(row)
    summary = {}
    for block, block_rows in sorted(by_block.items()):
        summary[block] = {field: mean_or_none(row.get(field) for row in block_rows) for field in fields}
    summary["all_blocks"] = {field: mean_or_none(row.get(field) for row in rows) for field in fields}
    return summary


def summarize_scope(items: Sequence[dict], bootstrap_samples: int, seed: int) -> dict:
    summary = {
        "num_pairs": len({item["pair_id"] for item in items}),
        "num_records": len(items) * len(CONDITIONS),
        "valid_fraction_mean": mean_or_none(item["valid_fraction"] for item in items),
        "valid_cells_mean": mean_or_none(float(item["valid_cells"]) for item in items),
        "total_cells_mean": mean_or_none(float(item["total_cells"]) for item in items),
        "temporal_block_count": len({item["block"] for item in items}),
        "bootstrap_note": (
            "Degenerate uncertainty: fewer than two frame_index//100 temporal blocks."
            if len({item["block"] for item in items}) < 2
            else "Block bootstrap uses frame_index//100 temporal blocks."
        ),
        "gains": {},
        "attention_correct": summarize_attention(items),
    }
    for region in REGIONS:
        summary["gains"][region] = {}
        for baseline in BASELINES:
            stats = summarize_gain_rows(items, baseline, region)
            stats["bootstrap"] = bootstrap_group(items, baseline, region, bootstrap_samples, seed)
            summary["gains"][region][f"{baseline}_minus_correct"] = stats
    return summary


def build_aggregates(items: Sequence[dict], bootstrap_samples: int, seed: int) -> dict:
    by_cohort_t_drive = defaultdict(list)
    for item in items:
        by_cohort_t_drive[(item["cohort"], item["t"], item["drive"])].append(item)

    result = {}
    for (cohort, timestep, drive), group_items in sorted(by_cohort_t_drive.items()):
        cohort_payload = result.setdefault(cohort, {}).setdefault(str(timestep), {"drives": {}})
        cohort_payload["drives"][drive] = summarize_scope(group_items, bootstrap_samples, seed)

    for cohort, by_t in result.items():
        for timestep, payload in by_t.items():
            drives = {
                drive: [item for item in items if item["cohort"] == cohort and str(item["t"]) == timestep and item["drive"] == drive]
                for drive in payload["drives"]
            }
            pooled_items = [item for group in drives.values() for item in group]
            payload["pooled_micro"] = summarize_scope(pooled_items, bootstrap_samples, seed)
            payload["pooled_macro_equal_drive"] = macro_summary(drives, bootstrap_samples, seed)
    return result


def macro_summary(drives: Dict[str, Sequence[dict]], bootstrap_samples: int, seed: int) -> dict:
    per_drive_scopes = {drive: summarize_scope(items, 0, seed) for drive, items in drives.items()}
    summary = {
        "num_drives": len(drives),
        "num_pairs": sum(scope["num_pairs"] for scope in per_drive_scopes.values()),
        "valid_fraction_mean": mean_or_none(scope["valid_fraction_mean"] for scope in per_drive_scopes.values()),
        "bootstrap_note": "Macro bootstrap averages per-drive block resamples equally; one-block drives have degenerate within-drive uncertainty.",
        "gains": {},
    }
    for region in REGIONS:
        summary["gains"][region] = {}
        for baseline in BASELINES:
            key = f"{baseline}_minus_correct"
            drive_stats = [scope["gains"][region][key] for scope in per_drive_scopes.values()]
            n = int(sum(stat["n"] for stat in drive_stats))
            wins = int(sum(stat["wins"] for stat in drive_stats))
            mean_abs = mean_or_none(stat["mean_abs"] for stat in drive_stats)
            rel = mean_or_none(stat["relative_percent"] for stat in drive_stats)
            summary["gains"][region][key] = {
                "n": n,
                "wins": wins,
                "mean_abs": mean_abs,
                "relative_percent": rel,
                "mean_drive_median": mean_or_none(stat["median"] for stat in drive_stats),
                "mean_drive_q25": mean_or_none(stat["q25"] for stat in drive_stats),
                "mean_drive_q75": mean_or_none(stat["q75"] for stat in drive_stats),
                "bootstrap": macro_bootstrap(drives, baseline, region, bootstrap_samples, seed),
            }
    return summary


def original_reproduction_summary(aggregates: dict) -> dict:
    if "original" not in aggregates:
        return {"present": False}
    result = {"present": True, "timesteps": {}}
    for timestep, payload in sorted(aggregates["original"].items()):
        scope = payload["pooled_micro"]
        result["timesteps"][timestep] = {
            "disabled_minus_correct_all": scope["gains"]["all"]["disabled_minus_correct"],
            "wrong_history_minus_correct_all": scope["gains"]["all"]["wrong_history_minus_correct"],
            "wrong_geometry_minus_correct_all": scope["gains"]["all"]["wrong_geometry_minus_correct"],
        }
    return result


def load_records(eval_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for path in evaluation_paths(eval_dir):
        file_rows = read_jsonl(path)
        if path.name.startswith("original_"):
            for row in file_rows:
                row.setdefault("cohort", "original")
        rows.extend(file_rows)
    return rows


def summarize_evaluation(
    eval_dir: str | Path,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict:
    eval_dir = Path(eval_dir)
    selection_path = eval_dir / "selection.json"
    if not selection_path.exists():
        raise EvaluationSummaryError(f"missing {selection_path}")
    selection = read_json(selection_path)
    parsed_selection = parse_selection(selection)
    reproduction = validate_reproduction(eval_dir, parsed_selection["world_size"])
    rows = load_records(eval_dir)
    grouped, coverage = validate_and_group(rows, parsed_selection)
    items = paired_items(grouped)
    aggregates = build_aggregates(items, bootstrap_samples, seed)
    cohorts = sorted({item["cohort"] for item in items})
    drives = sorted({item["drive"] for item in items if item["cohort"] == "expanded"})
    return {
        "schema_version": 1,
        "eval_dir": str(eval_dir),
        "note": (
            "Exploratory fixed-probe summary. Bootstrap resamples contiguous frame_index//100 "
            "blocks within each drive, and macro results average drives equally; do not read "
            "these as iid-frame significance tests. Short drives with one temporal block have "
            "degenerate within-drive uncertainty, not stronger evidence."
        ),
        "selection_coverage": coverage,
        "reproduction": reproduction,
        "cohorts": cohorts,
        "expanded_drives": drives,
        "timesteps": sorted({item["t"] for item in items}),
        "conditions": list(CONDITIONS),
        "regions": list(REGIONS),
        "primary_metric": "loss_eps_base / region_eps paired baseline_minus_correct gains",
        "aggregates": aggregates,
        "original_reproduction": original_reproduction_summary(aggregates),
    }


def fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def render_report(summary: dict) -> str:
    lines = [
        "# Geometry History Evaluation Summary",
        "",
        summary["note"],
        "",
        f"- Eval dir: `{summary['eval_dir']}`",
        f"- Primary metric: `{summary['primary_metric']}`",
        f"- Expanded pairs: {summary['selection_coverage']['seen_expanded_pairs']}",
        f"- Expanded drives: {', '.join(summary['expanded_drives'])}",
        "",
        "## Original Four-Probe Reproduction",
        "",
    ]
    original = summary["original_reproduction"]
    if not original["present"]:
        lines.append("Original cohort was not present in this evaluation.")
    else:
        lines.append("| t | disabled-correct all | wins/n | wrong_history-correct all | wrong_geometry-correct all |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for timestep, payload in sorted(original["timesteps"].items(), key=lambda kv: int(kv[0])):
            off = payload["disabled_minus_correct_all"]
            wrong = payload["wrong_history_minus_correct_all"]
            wrong_geo = payload["wrong_geometry_minus_correct_all"]
            lines.append(
                f"| {timestep} | {fmt(off['mean_abs'])} ({fmt(off['relative_percent'], 4)}%) | "
                f"{off['wins']}/{off['n']} | {fmt(wrong['mean_abs'])} | {fmt(wrong_geo['mean_abs'])} |"
            )
    lines.extend(["", "## Expanded Macro Results", ""])
    for cohort in ("expanded",):
        if cohort not in summary["aggregates"]:
            continue
        for timestep, payload in sorted(summary["aggregates"][cohort].items(), key=lambda kv: int(kv[0])):
            macro = payload["pooled_macro_equal_drive"]
            lines.append(f"### {cohort} t={timestep}")
            lines.append("")
            lines.append("| region | baseline-correct | mean_abs | relative % | wins/n | 95% block bootstrap mean_abs |")
            lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
            for region in REGIONS:
                for baseline in BASELINES:
                    key = f"{baseline}_minus_correct"
                    stats = macro["gains"][region][key]
                    ci = stats["bootstrap"]["mean_abs_ci95"]
                    lines.append(
                        f"| {region} | {key} | {fmt(stats['mean_abs'])} | {fmt(stats['relative_percent'], 4)} | "
                        f"{stats['wins']}/{stats['n']} | [{fmt(ci['low'])}, {fmt(ci['high'])}] |"
                    )
            lines.append("")
            lines.append(f"Valid fraction mean: {fmt(macro['valid_fraction_mean'], 4)} across {macro['num_drives']} drives.")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def output_path(eval_dir: Path, filename: str, overwrite: bool) -> Path:
    path = eval_dir / filename
    if path.exists() and not overwrite:
        raise EvaluationSummaryError(f"{path} exists; pass --overwrite to replace it")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize fixed geometry-history evaluation JSONL files.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default="summary.json")
    parser.add_argument("--out-md", default="report.md")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    summary = summarize_evaluation(eval_dir, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    json_path = output_path(eval_dir, args.out_json, args.overwrite)
    md_path = output_path(eval_dir, args.out_md, args.overwrite)
    write_json(json_path, summary)
    md_path.write_text(render_report(summary))
    print(json.dumps({"summary": str(json_path), "report": str(md_path), "status": "ok"}))


if __name__ == "__main__":
    main()
