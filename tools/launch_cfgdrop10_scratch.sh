#!/usr/bin/env bash
# V2 from-scratch CFG training: z-buffer-visible LiDAR features, dropout=0.1,
# 4 GPUs on the same NUMA node, batch 1 per rank (global 4), lr 1e-5,
# and 300k steps (= 1.2M samples).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MANIFEST_REPO="${MANIFEST_REPO:-/mnt/shizhm/CS2S_pose_environment_sat-lidar-ray-posterior-evidence}"

cd "$ROOT"
exec env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    /home/shizhm/miniconda3/envs/ControlS2S/bin/torchrun --nproc_per_node=4 --master_port=29613 tools/train_kitti_raea.py \
    --config configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea_cfgdrop10.yaml \
    --sd-base-ckpt /mnt/shizhm/BasicModel/checkpoints/sd-v1-4.ckpt \
    --kitti-root /mnt/shizhm/DATA/KITTI/KITTI_RAW \
    --allow-ddp-inline-sampling \
    --train-manifest "$MANIFEST_REPO/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/train_manifest.jsonl" \
    --val-manifest "$MANIFEST_REPO/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/test_manifest.jsonl" \
    --out-root /mnt/shizhm/DATA/KITTI/CS2S_results/kitti_ray_posterior \
    --run-name zbuffer_visible_ray_posterior_cfgdrop10_v2 \
    --steps 300000 \
    --batch-size 1 \
    --num-workers 2 \
    --dataloader-timeout 180 \
    --dist-timeout-seconds 600 \
    --min-free-disk-gb 100 \
    --min-free-host-memory-gb 12 \
    --lr 1e-05 \
    --shuffle --amp \
    --log-every 20 \
    --save-every 5000 \
    --keep-step-checkpoints 2 \
    --sample-every 1000 \
    --sample-manifest "$MANIFEST_REPO/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/test_manifest.jsonl" \
    --sample-num-samples 2 \
    --sample-ddim-steps 50 \
    --sample-seed 2026 \
    --sample-probes normal \
    --seed 3407 \
    --lidar-ray-feature-cache-root /mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/utonia_zbuffer_visible_ray_all_fp16 \
    --image-semantic-cache-root /mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/dino_vits14_8x32_all_fp16 \
    --lidar-depth-loss-weight 0.1 \
    --lidar-support-loss-weight 1.0 \
    --lidar-semantic-alignment-weight 0.2 \
    --lidar-token-structure-loss-weight 0.01 \
    --lidar-token-structure-target-ratio 0.08 \
    --lidar-evidence-dilation 4 \
    --lidar-evidence-free-space-dilation 14 \
    --lidar-support-dilation 8 \
    --lidar-reference-window 3 \
    --lidar-depth-log-eps 0.001
