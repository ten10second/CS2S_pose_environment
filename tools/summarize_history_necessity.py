"""Summarize archived history-necessity probe output.

The adapter and probe runners were retired on 2026-09-11. This standalone
JSON reader is retained for existing experiment artifacts; it loads no model.
See docs/temporal_history.md for the evaluated checkpoint and recovery refs.

Reads the JSON files written by tools/probe_history_necessity.py (or by
tools/run_history_necessity_probe.sh, which loops over pairs) and prints:

  - per-pair coverage: the raw 16x64 correspondence rate next to the rate the
    injection block actually reads, and their ratio;
  - the benefit table averaged over pairs, per (timestep, satellite arm);
  - the verdict, which is the whole point: redundant condition (a), collapsed
    mask (b), or inert readout (c).

Usage:
  python tools/summarize_history_necessity.py --dir <run>/probe_necessity
  python tools/summarize_history_necessity.py --files a.json b.json --min-benefit 0.002

benefit = disabled - correct, so a positive value means correct history lowered
the loss relative to the no-history baseline.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

MASK_COLLAPSE_RATIO = 0.2  # read@block below this fraction of the raw mask
RESIDUAL_FLOOR = 0.1      # ||history delta|| / ||condition delta|| below this is inert


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="", help="directory holding necessity_*.json")
    p.add_argument("--files", nargs="*", default=[], help="explicit JSON files")
    p.add_argument("--min-benefit", type=float, default=0.002,
                   help="benefit above this counts as history helping")
    return p.parse_args()


def load(paths):
    payloads = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


def collect(payloads):
    rows = defaultdict(list)
    coverage = []
    for payload in payloads:
        coverage.append((payload.get("pair_index", "?"), payload["coverage"]))
        for entry in payload["timesteps"]:
            rows[(entry["t"], bool(entry["satellite_blind"]))].append(entry)
    return rows, coverage


def values_of(entries, key):
    return [e[key] for e in entries if e.get(key) is not None]


def mean_of(entries, key):
    values = values_of(entries, key)
    return statistics.mean(values) if values else None


def median_of(entries, key):
    values = values_of(entries, key)
    return statistics.median(values) if values else None


def sign_split(entries, key):
    """(positive, negative) counts. With a handful of pairs the mean hides a
    split sign, so the verdict needs the counts as well."""
    values = values_of(entries, key)
    return sum(1 for v in values if v > 0), sum(1 for v in values if v < 0)


def arm_stats(entries):
    benefit_key = "benefit_disabled_minus_correct"
    positive, negative = sign_split(entries, benefit_key)
    return {
        "median": median_of(entries, benefit_key),
        "mean": mean_of(entries, benefit_key),
        "eps_median": median_of(entries, "benefit_disabled_minus_correct_eps"),
        "positive": positive,
        "total": positive + negative,
    }


def helps(stats, min_benefit):
    """A consistent, non-trivial benefit. The median carries the decision so one
    outlier pair cannot flip it; the positive count is reported next to it."""
    return stats["median"] >= min_benefit and stats["positive"] == stats["total"]


def verdict(rows, coverage, min_benefit):
    """Name the cause: (a) redundant condition, (b) collapsed mask, (c) inert
    readout, or (d) neutral residual (a large, content-sensitive correction that
    the objective never asked to be useful).

    Satellite redundancy can only be claimed by comparing the two arms: if
    zeroing the satellite does not unlock a benefit, the satellite is not what
    was standing in history's way.
    """
    on = [e for (t, is_blind), entries in rows.items() if not is_blind for e in entries]
    blind = [e for (t, is_blind), entries in rows.items() if is_blind for e in entries]
    if not blind:
        return ["no satellite-zeroed arm in the input; nothing to judge"], None
    blind_stats, on_stats = arm_stats(blind), arm_stats(on) if on else None
    if blind_stats["median"] is None:
        return ["satellite-zeroed arm has no paired cells"], None

    ratios = [c["effective_fraction_at_block"] / max(c["raw_fraction_16x64"], 1e-9)
              for _, c in coverage]
    read = statistics.mean(c["effective_fraction_at_block"] for _, c in coverage)
    raw = statistics.mean(c["raw_fraction_16x64"] for _, c in coverage)
    residual = statistics.median(c["residual_to_condition"] for _, c in coverage)
    collapsed = statistics.mean(ratios) < MASK_COLLAPSE_RATIO

    def arm_line(name, stats):
        if stats is None:
            return f"  {name:<7} (absent)"
        return (f"  {name:<7} median {stats['median']:+.4f}  mean {stats['mean']:+.4f}  "
                f"eps {stats['eps_median']:+.4f}  positive {stats['positive']}/{stats['total']}")

    lines = [
        arm_line("on", on_stats),
        arm_line("zeroed", blind_stats),
        f"  coverage: raw 16x64 {raw:.3f} -> read at the injection block {read:.3f}"
        f" (ratio {statistics.mean(ratios):.3f}); residual/condition {residual:.3f}",
    ]

    blind_helps = helps(blind_stats, min_benefit)
    on_helps = bool(on_stats and helps(on_stats, min_benefit))
    if blind_helps and not on_helps:
        lines += [
            "  ->  (a) REDUNDANT CONDITION: history helps only once the satellite is",
            "  zeroed, so the satellite is what made it unnecessary. Nothing in the",
            "  objective requires reading history while the conditions can answer.",
            "  Next: mechanism M1 (training-free, reuse the backbone's own K/V) or M3.",
        ]
        return lines, "a"
    if blind_helps and on_helps:
        return lines + [
            "  ->  history helps in both arms, so the on==off result is not reproduced",
            "  on this pair set. Check that these are the pairs and the checkpoint that",
            "  produced it before acting.",
        ], "none"
    if not blind_helps:
        for stats, name in ((blind_stats, "satellite-zeroed"), (on_stats, "satellite-on")):
            if stats and 0 < stats["positive"] < stats["total"]:
                lines.append(f"  note: the {name} sign is split "
                             f"({stats['positive']}/{stats['total']} pairs positive), so the "
                             "honest reading is \"no consistent benefit\", not a small win.")
        if collapsed:
            lines += [
                f"  ->  (b) MASK COLLAPSE: the block reads under "
                f"{MASK_COLLAPSE_RATIO:.0%} of the correspondence mask.",
                "  Build the correspondence at the injection resolution (grid=(8,32)",
                "  matching the block's query grid) and retrain the adapter on it.",
            ]
            return lines, "b"
        if residual < RESIDUAL_FLOOR:
            lines += [
                f"  ->  (c) INERT READOUT: coverage is fine but the residual is under",
                f"  {RESIDUAL_FLOOR:.2f} of the condition magnitude, so it cannot move",
                "  anything. Replace the readout rather than enlarging it (M1).",
            ]
            return lines, "c"
        lines += [
            "  ->  (d) NEUTRAL RESIDUAL: the block reads a large, content- and",
            "  geometry-sensitive correction (see the wrong-history / wrong-geometry",
            "  columns) that does not lower the loss. The objective never asked it to",
            "  be useful, and with the host tail unfrozen the adapter and the tail can",
            "  co-adapt to a loss-neutral perturbation.",
            "  This is an objective problem, not a geometry problem. Next: M3, but a",
            "  term the conditional mean cannot satisfy; Stage F's masked x0 already",
            "  was a consistency term and did not create the pressure.",
        ]
        return lines, "d"


def main():
    args = parse_args()
    if args.files:
        paths = [Path(p) for p in args.files]
    elif args.dir:
        paths = sorted(Path(args.dir).glob("*.json"))
    else:
        raise SystemExit("pass --dir or --files")
    if not paths:
        raise SystemExit(f"no probe JSON found in {args.dir}")
    payloads = load(paths)
    rows, coverage = collect(payloads)

    print(f"pairs: {len(payloads)}  ({', '.join(p['_path'] for p in payloads[:3])}"
          + (" ..." if len(payloads) > 3 else "") + ")\n")
    print("coverage per pair")
    for index, cov in coverage:
        ratio = cov["effective_fraction_at_block"] / max(cov["raw_fraction_16x64"], 1e-9)
        print(f"  pair {str(index):>3}: raw(16x64)={cov['raw_fraction_16x64']:.3f} "
              f"read@block={cov['effective_fraction_at_block']:.3f} ratio={ratio:.3f} "
              f"residual/condition={cov['residual_to_condition']}")

    print(f"\n{'t':>5} {'satellite':>10} {'mean':>10} {'median':>10} {'+/-':>7} "
          f"{'eps only':>10} {'vs wrongGeom':>13} {'vs wrongHist':>13}")
    for (t, is_blind), entries in sorted(rows.items()):
        positive, negative = sign_split(entries, "benefit_disabled_minus_correct")
        cells = [
            f"{t:>5} {'zeroed' if is_blind else 'on':>10}",
            f"{mean_of(entries, 'benefit_disabled_minus_correct'):>+10.4f}",
            f"{median_of(entries, 'benefit_disabled_minus_correct'):>+10.4f}",
            f"{f'{positive}/{positive + negative}':>7}",
            f"{mean_of(entries, 'benefit_disabled_minus_correct_eps'):>+10.4f}",
            f"{mean_of(entries, 'benefit_disabled_minus_wrong_geometry'):>+13.4f}",
            f"{mean_of(entries, 'benefit_wrong_history_minus_correct'):>+13.4f}",
        ]
        print(" ".join(cells))

    lines, _cause = verdict(rows, coverage, args.min_benefit)
    print("\nVERDICT")
    for line in lines:
        print("  " + line)


if __name__ == "__main__":
    main()
