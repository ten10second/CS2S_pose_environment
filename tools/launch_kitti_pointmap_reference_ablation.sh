#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/shizhm/anaconda3/envs/ControlS2S/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shizhm/codespace/CS2S_pose_environment}"
OUT_ROOT="${OUT_ROOT:-/media/shizhm/sda2/CS2S_results/kitti_pointmap_reference_ablation}"
RUN_PREFIX="${RUN_PREFIX:-pointmap_reference_from_hardcase6k_$(date +%Y%m%d_%H%M%S)}"
START_CKPT="${START_CKPT:-/media/shizhm/sda2/CS2S_results/kitti_pointmap_train/pointmap_hardcase128_from160k_warmstart_detached_20260703_214248/checkpoints/step_006000.pt}"

TRAIN_MIXED30="${TRAIN_MIXED30:-/media/shizhm/sda2/CS2S_results/kitti_raw_manifests/geofence_test2_buffer30_train_mixed30_hardcase128_yolo11sam2_lidarhit_sda2_raw.jsonl}"
TRAIN_MIXED20="${TRAIN_MIXED20:-/media/shizhm/sda2/CS2S_results/kitti_raw_manifests/geofence_test2_buffer30_train_mixed20_hardcase128_yolo11sam2_lidarhit_sda2_raw.jsonl}"
TRAIN_MIXED10="${TRAIN_MIXED10:-/media/shizhm/sda2/CS2S_results/kitti_raw_manifests/geofence_test2_buffer30_train_mixed10_hardcase128_yolo11sam2_lidarhit_sda2_raw.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-/media/shizhm/sda2/CS2S_results/kitti_raw_manifests/geofence_test2_buffer30_test_sda2_raw.jsonl}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-/media/shizhm/sda2/CS2S_results/kitti_raw_manifests/geofence_test2_buffer30_test_yolo11_lidarmask_hardcases_sda2_raw.jsonl}"
FG_MASK_ROOT="${FG_MASK_ROOT:-/media/shizhm/sda2/CS2S_results/kitti_foreground_mask_cache/sam2_yolo11_box_lidarmask_geofence_test2_buffer30_train}"

MIXED30_TARGET_STEPS="${MIXED30_TARGET_STEPS:-18000}"
MIXED20_TARGET_STEPS="${MIXED20_TARGET_STEPS:-30000}"
MIXED10_TARGET_STEPS="${MIXED10_TARGET_STEPS:-50000}"

mkdir -p "$OUT_ROOT"

COMMON_ARGS=(
  --config configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dual_attn.yaml
  --condition-mode raw_lidar_pointmap
  --sd-base-ckpt /home/shizhm/Downloads/sd-v1-4.ckpt
  --val-manifest "$VAL_MANIFEST"
  --sample-manifest "$SAMPLE_MANIFEST"
  --foreground-mask-root "$FG_MASK_ROOT"
  --foreground-mask-suffix _foreground.png
  --out-root "$OUT_ROOT"
  --batch-size 1
  --num-workers 2
  --lr 5e-5
  --shuffle
  --pin-memory
  --amp
  --sample-num-samples 8
  --sample-ddim-steps 50
  --sample-probes normal,zero
  --log-every 20
  --train-denoise partial
  --lidar-unfreeze-transformers output_middle
  --lidar-unfreeze-output-blocks 2
  --lidar-unfreeze-out
  --lidar-unet-lr-scale 0.1
  --lidar-unet-new-lr-scale 1.0
  --lidar-context-lr-scale 1.0
  --lidar-attn-gate-init 0.05
  --lidar-attention-mode reference
  --lidar-fusion-mode ray_evidence
  --lidar-geom-mode ray_depth_inv
  --lidar-reference-window 3
  --lidar-token-output-norm center_layernorm
  --lidar-depth-loss-weight 2.0
  --foreground-loss-weight 1.0
  --foreground-x0-loss-weight 1.0
  --foreground-image-loss-weight 5.0
  --foreground-lidar-intersection
  --lidar-zero-reconstruction-loss-weight 1.0
  --lidar-zero-reconstruction-mask-mode background
  --lidar-counterfactual-weight 0.0
  --no-save-optimizer
)

run_stage() {
  local stage_name="$1"
  local manifest="$2"
  local ckpt_arg_name="$3"
  local ckpt="$4"
  local target_steps="$5"
  local sample_every="$6"
  local save_every="$7"
  local run_name="${RUN_PREFIX}_${stage_name}"
  local run_dir="${OUT_ROOT}/${run_name}"
  mkdir -p "$run_dir"
  {
    echo "[$(date --iso-8601=seconds)] stage=${stage_name}"
    echo "ckpt_arg=${ckpt_arg_name}"
    echo "ckpt=${ckpt}"
    echo "manifest=${manifest}"
    echo "target_steps=${target_steps}"
    echo "lidar_attention_mode=reference"
    echo "lidar_fusion_mode=ray_evidence"
    echo "lidar_geom_mode=ray_depth_inv"
    echo "lidar_to_rgb_path=ray_aligned_evidence_attention"
  } | tee "$run_dir/stage_info.txt"
  "$PYTHON_BIN" tools/train_kitti_lidar_dominant.py \
    "${COMMON_ARGS[@]}" \
    "$ckpt_arg_name" "$ckpt" \
    --train-manifest "$manifest" \
    --run-name "$run_name" \
    --steps "$target_steps" \
    --sample-every "$sample_every" \
    --save-every "$save_every" \
    --keep-step-checkpoints 4 \
    > "$run_dir/train.log" 2>&1
  echo "${run_dir}/checkpoints/last.pt"
}

cd "$PROJECT_ROOT"
echo "run_prefix=${RUN_PREFIX}"
echo "out_root=${OUT_ROOT}"

ckpt="$(run_stage mixed30_hardcase30_full70 "$TRAIN_MIXED30" --lidar-warmstart-ckpt "$START_CKPT" "$MIXED30_TARGET_STEPS" 2000 6000 | tail -n 1)"
ckpt="$(run_stage mixed20_hardcase20_full80 "$TRAIN_MIXED20" --resume-ckpt "$ckpt" "$MIXED20_TARGET_STEPS" 2000 6000 | tail -n 1)"
ckpt="$(run_stage mixed10_hardcase10_full90 "$TRAIN_MIXED10" --resume-ckpt "$ckpt" "$MIXED10_TARGET_STEPS" 2000 10000 | tail -n 1)"

echo "complete_ckpt=${ckpt}"
