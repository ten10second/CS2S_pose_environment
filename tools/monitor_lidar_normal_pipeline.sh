#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LABEL_ROOT="${LABEL_ROOT:-dataset/kitti_lidar_normal_labels_metric3d}"
PIPELINE_PATTERN="${PIPELINE_PATTERN:-run_lidar_normal_full_pipeline|build_lidar_normal_pseudolabels|train_lidar_normal_encoder}"
METRICS_PATH="${METRICS_PATH:-results/lidar_normal_distill/full_kitti_metric3d/metrics/train_metrics.jsonl}"

while true; do
  date
  echo "label_count $(find "$LABEL_ROOT" -name '*.npz' | wc -l)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits || true
  fi
  pgrep -af "$PIPELINE_PATTERN" || true
  if [ -f "$METRICS_PATH" ]; then
    tail -3 "$METRICS_PATH"
  fi
  echo "---"
  if ! pgrep -f "$PIPELINE_PATTERN" >/dev/null 2>&1; then
    break
  fi
  sleep 300
done
