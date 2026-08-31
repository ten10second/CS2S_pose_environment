#!/usr/bin/env bash
# Full test-set inference sharded across 8 GPUs (7542 frames, ~943 per GPU).
set -u
REPO=/mnt/shizhm/CS2S_pose_environment_sat-lidar-ray-posterior-evidence
PY=/home/shizhm/miniconda3/envs/ControlS2S/bin/python
RUN_ROOT=/mnt/shizhm/DATA/KITTI/CS2S_results/kitti_ray_posterior/ray_posterior_utonia_dino_sd14_4gpu_20260812
CKPT=${CKPT:-$RUN_ROOT/checkpoints/step_500000.pt}
OUT_BASE=$RUN_ROOT/inference/full_test_500k
MANIFEST=$REPO/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/test_manifest.jsonl
CONFIG=$REPO/configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_raea.yaml
TOTAL=7542
SHARD=943

pids=()
for i in 0 1 2 3 4 5 6 7; do
  start=$((i * SHARD))
  count=$SHARD
  if [ $((start + count)) -gt $TOTAL ]; then count=$((TOTAL - start)); fi
  if [ $count -le 0 ]; then break; fi
  out="$OUT_BASE/shard$i"
  mkdir -p "$out"
  echo "[launcher] shard $i: gpu=$i start=$start count=$count out=$out"
  CUDA_VISIBLE_DEVICES=$i $PY -u "$REPO/tools/generate_kitti_raea_samples.py" \
    --config "$CONFIG" \
    --sd-base-ckpt /mnt/shizhm/BasicModel/checkpoints/sd-v1-4.ckpt \
    --ckpt "$CKPT" \
    --manifest "$MANIFEST" \
    --kitti-root /mnt/shizhm/DATA/KITTI/KITTI_RAW \
    --out-dir "$out" \
    --lidar-ray-feature-cache-root /mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/utonia_ray_depth_all_fp16 \
    --image-semantic-cache-root /mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/dino_vits14_8x32_all_fp16 \
    --start-index "$start" \
    --num-samples "$count" \
    --ddim-steps 50 \
    --probes normal \
    > "$out/run.log" 2>&1 &
  pids+=($!)
  sleep 10
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "[launcher] all shards done, fail=$fail"
exit $fail
