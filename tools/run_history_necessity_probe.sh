#!/usr/bin/env bash
# Root-cause probe: is history necessary, or is the readout broken?
#
# Runs tools/probe_history_necessity.py over several frame pairs on one GPU and
# prints the benefit table averaged over pairs. Read-only: no training, no
# checkpoints written. See docs/temporal_mechanism_options.md section 4.
#
#   GPU=4 PAIRS="0 1 2 -1" bash tools/run_history_necessity_probe.sh
#
# Override HIST_CKPT / SPLIT if you evaluated a different Stage F checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

experiment_root="${EXPERIMENT_ROOT:-/mnt/shizhm/DATA/KITTI/CS2S_results/geometry_history_20260908}"
experiment_python="${EXPERIMENT_PYTHON:-/home/shizhm/miniconda3/envs/ControlS2S/bin/python}"
hist_ckpt="${HIST_CKPT:-$experiment_root/stage_f_generator/geometry_history_step_1000.pt}"
base_ckpt="${BASE_CKPT:-$experiment_root/base/cfg_step_250000.pt}"
split="${SPLIT:-train}"
pairs="${PAIRS:-0 1 2 -1}"
gpu="${GPU:-4}"
out_root="${OUT_ROOT:-$experiment_root/probe_necessity}"
timesteps="${TIMESTEPS:-250,750}"

test -f "$hist_ckpt" || { echo "missing history checkpoint: $hist_ckpt" >&2; exit 1; }
# The adapter checkpoint records its base, and the loader refuses a mismatch,
# so BASE_CKPT must be the backbone the adapter was trained against.
test -f "$base_ckpt" || { echo "missing frozen backbone: $base_ckpt" >&2; exit 1; }

# Refuse to start on a GPU that is already doing something.
usage=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' -v g="$gpu" '$1 == g {print $2}')
if [ -z "$usage" ]; then
    echo "GPU $gpu not found" >&2; exit 1
fi
if [ "$usage" -gt 1024 ]; then
    echo "GPU $gpu already has ${usage} MiB in use; refusing to start" >&2; exit 1
fi

mkdir -p "$out_root"
for index in $pairs; do
    out="$out_root/necessity_${split}_pair${index}.json"
    echo "=== $split pair $index -> $out ==="
    CUDA_VISIBLE_DEVICES="$gpu" "$experiment_python" tools/probe_history_necessity.py \
        --config "$experiment_root/base/cfg_run_config.yaml" \
        --sd-base-ckpt /mnt/shizhm/BasicModel/checkpoints/sd-v1-4.ckpt \
        --ckpt "$base_ckpt" \
        --hist-ckpt "$hist_ckpt" \
        --manifest dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/train_manifest.jsonl \
        --kitti-root /mnt/shizhm/DATA/KITTI/KITTI_RAW \
        --lidar-ray-feature-cache-root /mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/utonia_ray_depth_all_fp16 \
        --image-semantic-cache-root /mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/dino_vits14_8x32_all_fp16 \
        --block-indices after_bottleneck --history-dim 64 --heads 4 --dim-head 32 \
        --appearance-x0-weight "${APPEARANCE_X0_WEIGHT:-1.0}" \
        --split "$split" --pair-index "$index" --timesteps "$timesteps" \
        --out "$out"
done

echo
echo "=== averaged over pairs (benefit = disabled - correct; >0 means history helped) ==="
"$experiment_python" - "$out_root" "$pairs" "$split" <<'PY'
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out_root, pairs, split = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3]
rows = defaultdict(list)
coverage = []
for index in pairs:
    path = out_root / f"necessity_{split}_pair{index}.json"
    if not path.is_file():
        continue
    payload = json.loads(path.read_text())
    coverage.append((index, payload["coverage"]))
    for entry in payload["timesteps"]:
        rows[(entry["t"], entry["satellite_blind"])].append(entry)

if not rows:
    raise SystemExit("no probe output found")

print(f"{'t':>5} {'satellite':>10} {'benefit':>10} {'vs wrongGeom':>13} "
      f"{'vs wrongHist':>13} {'n':>3}")
for (t, blind), entries in sorted(rows.items()):
    benefit = statistics.mean(e["benefit_disabled_minus_correct"] for e in entries)
    vs_geom = statistics.mean(e["benefit_disabled_minus_wrong_geometry"] for e in entries)
    vs_hist = statistics.mean(e["benefit_wrong_history_minus_correct"] for e in entries)
    print(f"{t:>5} {'zeroed' if blind else 'on':>10} {benefit:>+10.4f} "
          f"{vs_geom:>+13.4f} {vs_hist:>+13.4f} {len(entries):>3}")

print()
for index, cov in coverage:
    print(f"pair {index:>3}: raw(16x64)={cov['raw_fraction_16x64']:.3f} "
          f"read@block={cov['effective_fraction_at_block']:.3f} "
          f"residual/condition={cov['residual_to_condition']}")

blind_rows = [e for (t, blind), entries in rows.items() if blind for e in entries]
if blind_rows:
    blind_benefit = statistics.mean(e["benefit_disabled_minus_correct"] for e in blind_rows)
    print(f"\nsatellite-zeroed benefit (the verdict): {blind_benefit:+.4f}")
    print("  > 0 : history carries usable appearance -> the cause is a redundant")
    print("        condition, go to mechanism M1/M3")
    print("  ~ 0 : read the two coverage numbers above; if read@block is near 0 the")
    print("        correspondence mask collapsed at the injection resolution (fix")
    print("        the geometry first), otherwise the readout is inert (cause c)")
PY
