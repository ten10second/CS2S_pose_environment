#!/usr/bin/env bash
# V2.2 from-scratch CFG training: pixel LiDAR features, dropout=0.1,
# Native 128x512 depth, no bottleneck depth loss.
# 4 GPUs on the same NUMA node, batch 1 per rank (global 4), lr 1e-5,
# and 300k steps (= 1.2M samples).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MANIFEST_REPO="${MANIFEST_REPO:-/mnt/shizhm/CS2S_pose_environment_sat-lidar-ray-posterior-evidence}"
PIXEL_CACHE_ROOT="${PIXEL_CACHE_ROOT:-/home/shizhm/CS2S_cache_npz/utonia_pixel_lidar_v21_fp64_all_fp16}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-${MANIFEST_REPO}/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/train_manifest.jsonl}"

if [[ "${RUN_V22_PIXEL_CFGDROP10:-0}" != "1" ]]; then
    cat <<'EOF'
V2.2 pixel CFG-drop10 launcher is ready but did not start training.
Set RUN_V22_PIXEL_CFGDROP10=1 to run it explicitly.
EOF
    exit 0
fi

cd "$ROOT"
exec env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    /home/shizhm/miniconda3/envs/ControlS2S/bin/torchrun --nproc_per_node=4 --master_port=29623 tools/train_kitti_raea.py \
    --config configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_pixel_v22_cfgdrop10.yaml \
    --sd-base-ckpt /mnt/shizhm/BasicModel/checkpoints/sd-v1-4.ckpt \
    --kitti-root /mnt/shizhm/DATA/KITTI/KITTI_RAW \
    --allow-ddp-inline-sampling \
    --train-manifest "$MANIFEST_REPO/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/train_manifest.jsonl" \
    --val-manifest "$MANIFEST_REPO/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/test_manifest.jsonl" \
    --out-root /mnt/shizhm/DATA/KITTI/CS2S_results/kitti_ray_posterior \
    --run-name pixel_lidar_v22_cfgdrop10 \
    --steps 300000 \
    --batch-size 1 \
    --num-workers 2 \
    --dataloader-timeout 180 \
    --dist-timeout-seconds 1800 \
    --min-free-disk-gb 50 \
    --min-free-host-memory-gb 12 \
    --lr 1e-05 \
    --shuffle --amp \
    --log-every 20 \
    --save-every 5000 \
    --keep-step-checkpoints 2 \
    --sample-every 1000 \
    --sample-manifest "$SAMPLE_MANIFEST" \
    --sample-num-samples 2 \
    --sample-ddim-steps 50 \
    --sample-seed 2026 \
    --sample-fixed-seed --sample-eta 0.0 \
    --sample-probes normal,zero \
    --seed 3407 \
    --lidar-pixel-feature-cache-root "$PIXEL_CACHE_ROOT" \
    --image-semantic-cache-root /mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/dino_vits14_8x32_all_fp16 \
    --lidar-depth-loss-weight 0.1 \
    --lidar-support-loss-weight 1.0 \
    --lidar-semantic-alignment-weight 0.2 \
    --lidar-token-structure-loss-weight 0.0 \
    --lidar-token-structure-target-ratio 0.08 \
    --lidar-evidence-dilation 4 \
    --lidar-evidence-free-space-dilation 14 \
    --lidar-support-dilation 8 \
    --lidar-depth-log-eps 0.001 "$@"
