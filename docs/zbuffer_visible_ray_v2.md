# Z-buffer visible-ray V2

V2 keeps the existing 128x512 projected LiDAR depth/hit evidence, local LiDAR attention, satellite posterior weighting, and fusion gate. It changes only the offline Utonia feature path:

```text
3D LiDAR points + Utonia features
  -> camera projection
  -> nearest positive-depth point per 128x512 pixel
  -> mean pooling of visible point features per 8x32 camera patch
  -> [576,1,8,32] cache + occupancy mask
  -> visible-ray semantic token encoder (no depth-bin embedding)
```

The singleton cache plane preserves the existing Dataset/memmap tensor interface. It is not a depth bin.

## Build cache

Build train and test NPZ files into the same raw cache directory. The established split contains no duplicate sample IDs.

```bash
ROOT=/mnt/shizhm/CS2S_pose_environment_sat-lidar-zbuffer-v2
MANIFEST_REPO=/mnt/shizhm/CS2S_pose_environment_sat-lidar-ray-posterior-evidence
RAW_CACHE=/mnt/shizhm/DATA/KITTI/CS2S_cache/utonia_zbuffer_visible_ray_all_npz
MEMMAP_CACHE=/mnt/shizhm/DATA/KITTI/CS2S_cache_memmap/utonia_zbuffer_visible_ray_all_fp16
UTONIA_PY=/home/shizhm/miniconda3/envs/sd21/bin/python
TRAIN_PY=/home/shizhm/miniconda3/envs/ControlS2S/bin/python
UTONIA_ROOT="$MANIFEST_REPO/third_party/Utonia"
UTONIA_CKPT=/mnt/shizhm/BasicModel/checkpoints/utonia.pth

cd "$ROOT"
"$UTONIA_PY" tools/build_kitti_utonia_ray_cache.py \
  --manifest "$MANIFEST_REPO/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/train_manifest.jsonl" \
  --out-root "$RAW_CACHE" \
  --kitti-root /mnt/shizhm/DATA/KITTI/KITTI_RAW \
  --utonia-root "$UTONIA_ROOT" \
  --ckpt "$UTONIA_CKPT" \
  --device cuda \
  --skip-existing

"$UTONIA_PY" tools/build_kitti_utonia_ray_cache.py \
  --manifest "$MANIFEST_REPO/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/test_manifest.jsonl" \
  --out-root "$RAW_CACHE" \
  --kitti-root /mnt/shizhm/DATA/KITTI/KITTI_RAW \
  --utonia-root "$UTONIA_ROOT" \
  --ckpt "$UTONIA_CKPT" \
  --device cuda \
  --skip-existing

"$TRAIN_PY" tools/convert_kitti_feature_cache_memmap.py \
  --kind ray \
  --source-root "$RAW_CACHE" \
  --output-root "$MEMMAP_CACHE" \
  --point-dim 576 \
  --ray-depth-bins 1 \
  --ray-height 8 \
  --ray-width 32
```

The float16 feature memmap requires about 6.8 GiB (7.25 GB) for 24,597 frames. Raw compressed NPZ files require additional temporary space.
The server's Utonia dependencies are installed in the `sd21` environment; model training remains in `ControlS2S`. Passing the local checkpoint prevents an unavailable Hugging Face network request.

## Train

V2 must use its new cache and should start a new run because the LiDAR representation changed.

```bash
cd /mnt/shizhm/CS2S_pose_environment_sat-lidar-zbuffer-v2
bash tools/launch_cfgdrop10_scratch.sh
```

The launcher uses corrected masked-area auxiliary depth targets with weight 0.1. The weight is the result of the limited adaptation screen; it is a selected operating point rather than a claim of global optimality.
