#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/shizhm/anaconda3/envs/ControlS2S/bin/python"
ROOT="/home/shizhm/codespace/CS2S_pose_environment"
RUN_ROOT="/media/shizhm/sda2/CS2S_results/kitti_raea_utonia_dino_curriculum"
RUN_NAME="raea_utonia_dino_full100_routerfix_lr1e5_resume440k"
CHECKPOINT="${RUN_ROOT}/raea_utonia_dino_full100_memmap_b2_resume_200k/checkpoints/step_440000.pt"

cd "${ROOT}"
exec "${PYTHON}" -u tools/train_kitti_lidar_dominant.py \
    --config configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_dual_attn.yaml \
    --condition-mode raw_lidar_pointmap \
    --sd-base-ckpt /home/shizhm/Downloads/sd-v1-4.ckpt \
    --resume-ckpt "${CHECKPOINT}" \
    --reinit-ray-evidence-router \
    --train-manifest /media/shizhm/sda2/CS2S_results/kitti_raw_manifests/geofence_test2_buffer30_train_sda2_raw.jsonl \
    --val-manifest /media/shizhm/sda2/CS2S_results/kitti_raw_manifests/geofence_test2_buffer30_train_hardcase64_numdyn_sda2_raw.jsonl \
    --out-root "${RUN_ROOT}" \
    --run-name "${RUN_NAME}" \
    --steps 1000000 \
    --batch-size 2 \
    --num-workers 2 \
    --lr 1e-5 \
    --shuffle \
    --amp \
    --log-every 20 \
    --save-every 10000 \
    --sample-every 5000 \
    --sample-manifest /media/shizhm/sda2/CS2S_results/kitti_raw_manifests/geofence_test2_buffer30_train_hardcase64_numdyn_sda2_raw.jsonl \
    --sample-num-samples 2 \
    --sample-ddim-steps 50 \
    --sample-probes normal \
    --keep-step-checkpoints 3 \
    --no-save-optimizer \
    --lidar-context-backbone point3d_ray \
    --lidar-raw-point-count 4096 \
    --lidar-point-in-channels 10 \
    --lidar-point-feature-cache-root /home/shizhm/CS2S_cache_memmap/utonia_4096_train_fp16 \
    --lidar-point-feature-cache-suffix .npz \
    --lidar-point-feature-dim 576 \
    --image-semantic-cache-root /home/shizhm/CS2S_cache_memmap/dino_vits14_8x32_train_fp16 \
    --image-semantic-cache-suffix .npz \
    --image-semantic-feature-key dino_feat \
    --image-semantic-feature-dim 384 \
    --image-semantic-height 8 \
    --image-semantic-width 32 \
    --lidar-semantic-alignment-weight 0.2 \
    --lidar-semantic-alignment-mask-mode lidar_hit \
    --lidar-depth-loss-weight 1.0 \
    --lidar-token-output-norm center_layernorm \
    --lidar-token-structure-target-ratio 0.08 \
    --lidar-token-structure-loss-weight 0.01 \
    --lidar-attention-mode reference \
    --lidar-reference-window 3 \
    --lidar-fusion-mode ray_evidence \
    --ray-evidence-sat-bias 2.0 \
    --ray-evidence-lidar-bias -2.0 \
    --ray-evidence-null-bias -6.0 \
    --ray-evidence-mask-mode lidar_hit \
    --cs2s-train-scope \
    --train-denoise all \
    --train-sat-condition \
    --lidar-unet-lr-scale 0.05 \
    --lidar-unet-new-lr-scale 1.0
