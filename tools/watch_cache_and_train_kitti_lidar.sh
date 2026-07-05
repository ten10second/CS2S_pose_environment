#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/shizhm/codespace/CS2S_pose_environment}"
PYTHON_BIN="${PYTHON_BIN:-/home/shizhm/anaconda3/envs/ControlS2S/bin/python}"
RAW_ROOT="${RAW_ROOT:-/media/shizhm/Lenovo/KITTI_RAW}"
RESULT_ROOT="${RESULT_ROOT:-/media/shizhm/sda2/CS2S_results}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/media/shizhm/sda2/CS2S_results/kitti_xlidar_overfit/train_manifest.geofence_test2_buffer30.filtered_exfat_stalls_3.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-dataset/kitti_raw_sat_lidar_geofence_test2_buffer30/test_manifest.jsonl}"
RESUME_CKPT="${RESUME_CKPT:-/media/shizhm/sda2/CS2S_results/kitti_xlidar_overfit/sd14_cs2scope_sam2fg_lidarhit_rgbsupport_dil6_gate02_resume8k_to36k/checkpoints/last.pt}"
FOREGROUND_MASK_ROOT="${FOREGROUND_MASK_ROOT:-/media/shizhm/sda2/CS2S_results/kitti_foreground_mask_cache/sam2_box_geofence_test2_buffer30}"
RUN_NAME="${RUN_NAME:-sd14_cs2scope_sam2fg_lidarhit_rgbsupport_dil6_gate02_cache_sda2_to36k}"
CACHE_ROOT="${CACHE_ROOT:-/media/shizhm/sda2/CS2S_results/kitti_raw_cache/geofence_test2_buffer30_train}"
CACHED_MANIFEST="${CACHED_MANIFEST:-/media/shizhm/sda2/CS2S_results/kitti_raw_cache/geofence_test2_buffer30_train_manifest.cached.jsonl}"
CACHE_LIMIT="${CACHE_LIMIT:-0}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"
WATCH_DIR="${WATCH_DIR:-/media/shizhm/sda2/CS2S_results/watchdogs}"
WATCH_LOG="${WATCH_LOG:-${WATCH_DIR}/${RUN_NAME}.watch.log}"

mkdir -p "${WATCH_DIR}"
cd "${REPO_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${WATCH_LOG}"
}

has_blocked_raw_io() {
  ps -eo stat=,wchan:32=,cmd= | awk '
    $1 ~ /D/ && ($2 ~ /exfat|bread_gfp/ || $0 ~ /spawn_main/) { found=1 }
    END { exit(found ? 0 : 1) }
  '
}

raw_probe_ok() {
  local probe="${RAW_ROOT}/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000048.png"
  timeout 20 bash -c "test -r '${probe}' && dd if='${probe}' of=/dev/null bs=1 count=1 status=none"
}

log "watchdog started"
log "raw_root=${RAW_ROOT}"
log "cache_root=${CACHE_ROOT}"
log "cached_manifest=${CACHED_MANIFEST}"
log "resume_ckpt=${RESUME_CKPT}"

while has_blocked_raw_io; do
  log "waiting: blocked raw/exfat I/O still present"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

until raw_probe_ok; do
  log "waiting: raw probe failed or timed out"
  sleep "${CHECK_INTERVAL_SECONDS}"
done

log "raw probe succeeded; caching manifest files"
"${PYTHON_BIN}" tools/cache_kitti_raw_manifest_files.py \
  --manifest "${TRAIN_MANIFEST}" \
  --source-root "${RAW_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --out-manifest "${CACHED_MANIFEST}" \
  --limit "${CACHE_LIMIT}" \
  --no-tracklets 2>&1 | tee -a "${WATCH_LOG}"

log "starting training from cached manifest"
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" tools/train_kitti_lidar_dominant.py \
  --config configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dual_attn.yaml \
  --train-manifest "${CACHED_MANIFEST}" \
  --val-manifest "${VAL_MANIFEST}" \
  --condition-mode raw_lidar_pointmap \
  --sd-base-ckpt /home/shizhm/Downloads/sd-v1-4.ckpt \
  --resume-ckpt "${RESUME_CKPT}" \
  --run-name "${RUN_NAME}" \
  --out-root "${RESULT_ROOT}/kitti_xlidar_overfit" \
  --steps 36000 \
  --batch-size 1 \
  --num-workers 1 \
  --lr 1e-6 \
  --shuffle \
  --amp \
  --save-every 2000 \
  --no-save-optimizer \
  --keep-step-checkpoints 1 \
  --log-every 20 \
  --cs2s-train-scope \
  --train-sat-condition \
  --train-denoise all \
  --foreground-mask-root "${FOREGROUND_MASK_ROOT}" \
  --foreground-loss-weight 0.5 \
  --foreground-x0-loss-weight 0.5 \
  --foreground-image-loss-weight 1.0 \
  --foreground-lpips-loss-weight 0.02 \
  --foreground-lidar-intersection \
  --lidar-support-dilation 6 \
  --lidar-support-loss-weight 1.0 \
  --lidar-support-x0-loss-weight 1.0 \
  --lidar-support-image-loss-weight 3.0 \
  --force-lidar-gate 0.20 \
  --no-tracklets 2>&1 | tee -a "${WATCH_LOG}"

log "training command exited"
