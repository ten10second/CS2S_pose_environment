"""Turn history-necessity probe output into a verdict.

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


def mean_of(entries, key):
    values = [e[key] for e in entries if e.get(key) is not None]
    return statistics.mean(values) if values else None


def verdict(rows, coverage, min_benefit):
    blind = [e for (t, is_blind), entries in rows.items() if is_blind for e in entries]
    if not blind:
        return ["no satellite-zeroed arm in the input; nothing to judge"], None
    benefit = mean_of(blind, "benefit_disabled_minus_correct")
    benefit_eps = mean_of(blind, "benefit_disabled_minus_correct_eps")
    lines = []
    if benefit is None:
        return ["satellite-zeroed arm has no paired cells"], None
    lines.append(f"satellite-zeroed benefit (disabled - correct) = {benefit:+.4f}"
                 + (f"  [eps-only {benefit_eps:+.4f}]" if benefit_eps is not None else ""))
    if benefit >= min_benefit:
        lines.append(f"  >= min-benefit {min_benefit:+.4f}  ->  (a) REDUNDANT CONDITION")
        lines.append("  History carries usable appearance; the satellite condition makes it")
        lines.append("  unnecessary, so nothing in the objective requires reading it.")
        lines.append("  Next: mechanism M1 (training-free, reuse the backbone's own K/V) or M3.")
        return lines, "a"
    ratios = [c["effective_fraction_at_block"] / max(c["raw_fraction_16x64"], 1e-9)
              for _, c in coverage]
    read = statistics.mean(c["effective_fraction_at_block"] for _, c in coverage)
    raw = statistics.mean(c["raw_fraction_16x64"] for _, c in coverage)
    lines.append(f"  <  min-benefit {min_benefit:+.4f}")
    lines.append(f"  coverage: raw 16x64 {raw:.3f} -> read at the injection block {read:.3f}"
                 f" (ratio {statistics.mean(ratios):.3f})")
    if statistics.mean(ratios) < MASK_COLLAPSE_RATIO:
        lines.append(f"  ->  (b) MASK COLLAPSE: the block reads under "
                     f"{MASK_COLLAPSE_RATIO:.0%} of the correspondence mask.")
        lines.append("  Fix the geometry before touching the mechanism: build the correspondence")
        lines.append("  at the injection resolution (grid=(8,32) matching the block's query")
        lines.append("  grid), which removes the all-four-children-valid rule. The adapter must")
        lines.append("  be retrained on the new grid.")
        return lines, "b"
    lines.append("  ->  (c) INERT READOUT: coverage is fine, the stream runs, but it changes")
    lines.append("  nothing measurable. Replace the readout rather than enlarging it (M1).")
    return lines, "c"


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

    print(f"\n{'t':>5} {'satellite':>10} {'benefit':>10} {'eps only':>10} "
          f"{'vs wrongGeom':>13} {'vs wrongHist':>13} {'n':>3}")
    for (t, is_blind), entries in sorted(rows.items()):
        cells = [
            f"{t:>5} {'zeroed' if is_blind else 'on':>10}",
            f"{mean_of(entries, 'benefit_disabled_minus_correct'):>+10.4f}",
            f"{mean_of(entries, 'benefit_disabled_minus_correct_eps'):>+10.4f}",
            f"{mean_of(entries, 'benefit_disabled_minus_wrong_geometry'):>+13.4f}",
            f"{mean_of(entries, 'benefit_wrong_history_minus_correct'):>+13.4f}",
            f"{len(entries):>3}",
        ]
        print(" ".join(cells))

    lines, _cause = verdict(rows, coverage, args.min_benefit)
    print("\nVERDICT")
    for line in lines:
        print("  " + line)


if __name__ == "__main__":
    main()
