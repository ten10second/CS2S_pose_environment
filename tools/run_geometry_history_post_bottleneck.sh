#!/usr/bin/env bash
# Mechanism control: geometry history only after the bottleneck depth head.
# Does not resume 2,12 checkpoints. Fresh adapter. GPUs 4-7 only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
experiment_root=/mnt/shizhm/DATA/KITTI/CS2S_results/geometry_history_20260908
experiment_python=/home/shizhm/miniconda3/envs/ControlS2S/bin/python
out_dir="$experiment_root/stage_d_post_bottleneck"

exec 9>"$experiment_root/pilot.lock"
flock -n 9
"$experiment_python" - "$experiment_root" "$out_dir" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
out = Path(sys.argv[2])
geometry = json.loads((root / 'stage_a/summary.json').read_text())
for key in ('pilot_gate_valid_coverage_pass', 'pilot_gate_non_ground_proxy_all_pass', 'pilot_gate_rgb_motion_pass'):
    if geometry.get(key) is not True:
        raise RuntimeError('Stage A gate failed: ' + key)
if out.exists():
    raise FileExistsError('post-bottleneck output already exists; inspect it before resubmitting')
output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                  '--format=csv,noheader,nounits'], text=True)
usage = dict(tuple(map(int, line.split(','))) for line in output.strip().splitlines())
if any(usage.get(index, 999999) > 128 for index in (4, 5, 6, 7)):
    raise RuntimeError('GPUs 4-7 are not all idle; not starting a duplicate/conflicting job')
PY

exec env CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 \
    "$experiment_python" -m torch.distributed.run --standalone --nproc_per_node=4 \
    tools/train_kitti_geometry_history.py \
    --config "$experiment_root/base/cfg_run_config.yaml" \
    --sd-base-ckpt /mnt/shizhm/BasicModel/checkpoints/sd-v1-4.ckpt \
    --ckpt "$experiment_root/base/cfg_step_250000.pt" \
    --manifest dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/train_manifest.jsonl \
    --kitti-root /mnt/shizhm/DATA/KITTI/KITTI_RAW \
    --out-dir "$out_dir" \
    --lidar-ray-feature-cache-root /mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/utonia_ray_depth_all_fp16 \
    --image-semantic-cache-root /mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/dino_vits14_8x32_all_fp16 \
    --block-indices after_bottleneck \
    --steps 1000 --probe-every 100 --save-every 100 --log-every 20
