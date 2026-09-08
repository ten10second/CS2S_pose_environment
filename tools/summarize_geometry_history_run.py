"""Summarize a geometry-history training run without importing torch."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


GRAD_KEYS = ("encoder_grad_l2", "cond_query_grad_l2", "out_grad_l2")
PROBE_CONDITIONS = ("disabled", "correct", "wrong_geometry", "wrong_history")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return float(np.asarray(vals, dtype=np.float64).mean())


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and bool(np.isfinite(float(value)))


def rank_paths(run_dir: Path, prefix: str) -> Dict[int, Path]:
    found = {}
    for path in sorted(run_dir.glob(f"{prefix}_rank*.jsonl")):
        stem = path.stem
        rank_text = stem.split("_rank", 1)[1]
        if rank_text.isdigit():
            found[int(rank_text)] = path
    return found


def expected_ranks(metadata: dict, metrics_by_rank: Dict[int, list], probes_by_rank: Dict[int, list]) -> List[int]:
    world = metadata.get("world_size")
    if isinstance(world, int) and world > 0:
        return list(range(world))
    ranks = sorted(set(metrics_by_rank) | set(probes_by_rank))
    return ranks


def aggregate_probe_records(probes_by_rank: Dict[int, list]) -> Tuple[dict, list]:
    buckets = defaultdict(list)
    per_probe = defaultdict(dict)
    for rank, rows in probes_by_rank.items():
        for row in rows:
            key = (row.get("split"), int(row.get("step")), int(row.get("t")), row.get("condition"))
            buckets[key].append(float(row["loss_total"]))
            paired_key = (
                int(row.get("rank", rank)),
                row.get("pair_id", ""),
                row.get("split"),
                int(row.get("step")),
                int(row.get("t")),
            )
            per_probe[paired_key][row.get("condition")] = float(row["loss_total"])

    records = []
    for (split, step, timestep, condition), losses in sorted(buckets.items()):
        records.append(
            {
                "split": split,
                "step": step,
                "t": timestep,
                "condition": condition,
                "loss_total_mean": mean(losses),
                "count": len(losses),
            }
        )

    paired_by_probe = []
    paired_buckets = defaultdict(list)
    positive_counts = defaultdict(int)
    paired_counts = defaultdict(int)
    for (rank, pair_id, split, step, timestep), conds in sorted(per_probe.items()):
        if "correct" not in conds or "disabled" not in conds:
            continue
        correct = conds["correct"]
        disabled = conds["disabled"]
        item = {
            "rank": rank,
            "pair_id": pair_id,
            "split": split,
            "step": step,
            "t": timestep,
            "benefit_disabled_minus_correct": disabled - correct,
        }
        for wrong in ("wrong_geometry", "wrong_history"):
            if wrong in conds:
                item[f"benefit_{wrong}_minus_correct"] = conds[wrong] - correct
        paired_by_probe.append(item)
        for key, value in item.items():
            if key.startswith("benefit_"):
                paired_counts[(split, step, timestep, key)] += 1
                if value > 0:
                    positive_counts[(split, step, timestep, key)] += 1
                paired_buckets[(split, step, timestep, key)].append(value)

    paired = []
    grouped = defaultdict(dict)
    for (split, step, timestep, key), values in sorted(paired_buckets.items()):
        grouped[(split, step, timestep)][key] = mean(values)
        grouped[(split, step, timestep)][f"{key}_count"] = len(values)
    for (split, step, timestep), values in sorted(grouped.items()):
        paired.append({"split": split, "step": step, "t": timestep, **values})

    positives = []
    for (split, step, timestep, key), count in sorted(paired_counts.items()):
        positives.append(
            {
                "split": split,
                "step": step,
                "t": timestep,
                "metric": key,
                "positive_count": positive_counts[(split, step, timestep, key)],
                "count": count,
            }
        )
    return {
        "means": records,
        "paired": paired,
        "paired_by_rank_pair": paired_by_probe,
        "positive_counts": positives,
    }, records


def check_step0_identical(probes_by_rank: Dict[int, list], start_step: int = 0) -> dict:
    if start_step > 0:
        return {"status": "not_applicable", "pass": None, "checked_groups": 0, "failures": []}
    failures = []
    checked = 0
    for rank, rows in probes_by_rank.items():
        grouped = defaultdict(dict)
        for row in rows:
            if int(row.get("step", -1)) == 0:
                grouped[(row.get("split"), int(row.get("t")))][row.get("condition")] = float(row["loss_total"])
        for key, conds in grouped.items():
            if not all(condition in conds for condition in PROBE_CONDITIONS):
                failures.append({"rank": rank, "key": key, "reason": "missing_condition", "conditions": sorted(conds)})
                continue
            checked += 1
            values = [conds[condition] for condition in PROBE_CONDITIONS]
            if len(set(values)) != 1:
                failures.append({"rank": rank, "key": key, "reason": "loss_mismatch", "values": values})
    return {"pass": not failures and checked > 0, "checked_groups": checked, "failures": failures}


def check_disabled_constant(probes_by_rank: Dict[int, list]) -> dict:
    failures = []
    checked = 0
    for rank, rows in probes_by_rank.items():
        grouped = defaultdict(list)
        for row in rows:
            if row.get("condition") == "disabled":
                grouped[(row.get("split"), int(row.get("t")))].append(float(row["loss_total"]))
        for key, values in grouped.items():
            if len(values) < 2:
                continue
            checked += 1
            if len(set(values)) != 1:
                failures.append({"rank": rank, "key": key, "values": values})
    return {"pass": not failures and checked > 0, "checked_groups": checked, "failures": failures}


def check_training_metrics(metrics_by_rank: Dict[int, list], ranks: List[int], start_step: int, final_step: int) -> dict:
    rank_status = {}
    expected_steps = set(range(start_step + 1, final_step + 1)) if final_step > start_step else set()
    for rank in ranks:
        rows = metrics_by_rank.get(rank, [])
        steps = {int(row["step"]) for row in rows if "step" in row}
        enabled_rows = [row for row in rows if row.get("history") is True]
        step2_rows = [row for row in rows if int(row.get("step", -1)) == 2]
        step3_rows = [row for row in rows if int(row.get("step", -1)) == 3]

        finite_grad_rows = [
            row for row in rows
            if all(finite_number(row.get(key)) for key in GRAD_KEYS)
        ]
        finite_loss_rows = [row for row in rows if finite_number(row.get("loss"))]
        nonzero_by_key = {}
        for key in GRAD_KEYS:
            nonzero_by_key[key] = any(float(row.get(key, 0.0)) > 0.0 for row in enabled_rows if int(row.get("step", 0)) >= 5)
        if start_step > 0:
            step2_zero = {"status": "not_applicable", "pass": None}
            step3_cfg = {"status": "not_applicable", "pass": None}
        else:
            step2_zero = {
                "status": "checked",
                "pass": bool(step2_rows) and all(
                    all(finite_number(row.get(key)) and float(row.get(key)) == 0.0 for key in GRAD_KEYS)
                    for row in step2_rows
                ),
            }
            step3_cfg = {
                "status": "checked",
                "pass": bool(step3_rows) and all(row.get("satellite_dropped") is True for row in step3_rows),
            }
        missing_steps = sorted(expected_steps.difference(steps))
        complete = final_step in steps and not missing_steps
        enabled_after_step5_count = len([r for r in enabled_rows if int(r.get("step", 0)) >= 5])
        enabled_nonzero_required = enabled_after_step5_count >= 5
        enabled_nonzero_pass = (not enabled_nonzero_required) or all(nonzero_by_key.values())
        rank_status[str(rank)] = {
            "num_metric_rows": len(rows),
            "max_step": max(steps) if steps else 0,
            "missing_steps": missing_steps,
            "complete_final_step": complete,
            "finite_loss_rows": len(finite_loss_rows),
            "finite_loss_all_rows": len(finite_loss_rows) == len(rows) and bool(rows),
            "finite_grad_metric_rows": len(finite_grad_rows),
            "finite_grad_metrics_all_rows": len(finite_grad_rows) == len(rows) and bool(rows),
            "enabled_history_rows_after_step5": enabled_after_step5_count,
            "enabled_history_nonzero_grad_after_step5": nonzero_by_key,
            "enabled_history_nonzero_grad_after_step5_required": enabled_nonzero_required,
            "enabled_history_nonzero_grad_after_step5_pass": enabled_nonzero_pass,
            "no_history_step2_zero_gradients": step2_zero,
            "cfg_step3_logged_true": step3_cfg,
            "loss_mean": mean(row["loss"] for row in rows if finite_number(row.get("loss"))),
        }
    return rank_status


def summarize_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "run.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    metrics_paths = rank_paths(run_dir, "metrics")
    probes_paths = rank_paths(run_dir, "probes")
    metrics_by_rank = {rank: read_jsonl(path) for rank, path in metrics_paths.items()}
    probes_by_rank = {rank: read_jsonl(path) for rank, path in probes_paths.items()}
    ranks = expected_ranks(metadata, metrics_by_rank, probes_by_rank)
    final_step = int(metadata.get("args", {}).get("steps", metadata.get("step", 0)) or 0)
    start_step = int(metadata.get("start_step", metadata.get("args", {}).get("start_step", 0)) or 0)
    ckpt_path = run_dir / f"geometry_history_step_{final_step}.pt" if final_step else None

    train_status = check_training_metrics(metrics_by_rank, ranks, start_step, final_step)
    complete_ranks = [rank for rank in ranks if train_status.get(str(rank), {}).get("complete_final_step")]
    all_rank_files_present = all(rank in metrics_by_rank and rank in probes_by_rank for rank in ranks)
    status = "complete" if ranks and len(complete_ranks) == len(ranks) and ckpt_path and ckpt_path.exists() else "incomplete"

    probe_summary, _probe_means = aggregate_probe_records(probes_by_rank)
    step0 = check_step0_identical(probes_by_rank, start_step)
    disabled = check_disabled_constant(probes_by_rank)
    final_enabled_steps = sum(
        train_status[str(rank)]["enabled_history_rows_after_step5"]
        for rank in ranks if str(rank) in train_status
    )
    smoke = {
        "all_rank_files_present": all_rank_files_present,
        "complete_steps_all_world_ranks": status == "complete",
        "checkpoint_exists_at_final_step": bool(ckpt_path and ckpt_path.exists()),
        "step0_all_conditions_identical": step0,
        "disabled_probe_constant": disabled,
        "finite_loss_all_ranks": all(
            train_status.get(str(rank), {}).get("finite_loss_all_rows", False)
            for rank in ranks
        ),
        "finite_grad_metrics_all_ranks": all(
            train_status.get(str(rank), {}).get("finite_grad_metrics_all_rows", False)
            for rank in ranks
        ),
        "enabled_history_gradients_nonzero_after_step5": {
            "required_after_total_enabled_steps_at_least": 5,
            "total_enabled_steps_after_step5": final_enabled_steps,
            "pass": all(
                train_status.get(str(rank), {}).get("enabled_history_nonzero_grad_after_step5_pass", False)
                for rank in ranks
            ),
            "by_rank": {
                str(rank): train_status.get(str(rank), {}).get("enabled_history_nonzero_grad_after_step5", {})
                for rank in ranks
            },
            "by_key_all_ranks": {
                key: all(
                    train_status.get(str(rank), {}).get("enabled_history_nonzero_grad_after_step5", {}).get(key, False)
                    or not train_status.get(str(rank), {}).get("enabled_history_nonzero_grad_after_step5_required", True)
                    for rank in ranks
                )
                for key in GRAD_KEYS
            },
        },
        "no_history_step2_zero_gradients_all_ranks": (
            {"status": "not_applicable", "pass": None}
            if start_step > 0
            else {
                "status": "checked",
                "pass": all(
                    train_status.get(str(rank), {}).get("no_history_step2_zero_gradients", {}).get("pass", False)
                    for rank in ranks
                ),
            }
        ),
        "cfg_step3_logged_true_all_ranks": (
            {"status": "not_applicable", "pass": None}
            if start_step > 0
            else {
                "status": "checked",
                "pass": all(
                    train_status.get(str(rank), {}).get("cfg_step3_logged_true", {}).get("pass", False)
                    for rank in ranks
                ),
            }
        ),
    }

    return {
        "status": status,
        "metadata": {
            "mode": metadata.get("mode"),
            "world_size": metadata.get("world_size"),
            "steps": final_step,
            "start_step": start_step,
            "train_pairs": metadata.get("train_pairs"),
            "val_pairs": metadata.get("val_pairs"),
            "val_drives": metadata.get("val_drives"),
            "note": "Held-out probes are a small per-rank smoke check, not a full benchmark.",
        },
        "files": {
            "run_dir": str(run_dir),
            "checkpoint_final": str(ckpt_path) if ckpt_path else "",
            "metric_ranks": sorted(metrics_by_rank),
            "probe_ranks": sorted(probes_by_rank),
        },
        "training": train_status,
        "fixed_probes": probe_summary,
        "smoke": smoke,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a geometry-history run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = summarize_run(args.run_dir)
    out_path = Path(args.out_json) if args.out_json else Path(args.run_dir) / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"summary": str(out_path), "status": summary["status"]}))


if __name__ == "__main__":
    main()
