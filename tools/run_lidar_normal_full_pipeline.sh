#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LABEL_ROOT="${LABEL_ROOT:-dataset/kitti_lidar_normal_labels_metric3d}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-dataset/kitti_raw_lidar_normal/train_manifest.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-dataset/kitti_raw_lidar_normal/val_manifest.jsonl}"
RUN_NAME="${RUN_NAME:-full_kitti_metric3d}"
TRAIN_STEPS="${TRAIN_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_POINTS_PER_SAMPLE="${MAX_POINTS_PER_SAMPLE:-8000}"

date

conda run -n depth python tools/build_lidar_normal_pseudolabels.py \
  --manifest "$TRAIN_MANIFEST" \
  --out-root "$LABEL_ROOT" \
  --teacher metric3d_hub \
  --metric3d-model metric3d_vit_small

conda run -n depth python tools/build_lidar_normal_pseudolabels.py \
  --manifest "$VAL_MANIFEST" \
  --out-root "$LABEL_ROOT" \
  --teacher metric3d_hub \
  --metric3d-model metric3d_vit_small

conda run -n ControlS2S python tools/train_lidar_normal_encoder.py \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --label-root "$LABEL_ROOT" \
  --run-name "$RUN_NAME" \
  --steps "$TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers 4 \
  --save-every 5000 \
  --eval-every 2500 \
  --vis-every 5000 \
  --log-every 500 \
  --val-batches 16 \
  --max-points-per-sample "$MAX_POINTS_PER_SAMPLE"

date
