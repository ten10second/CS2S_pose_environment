# Fixed Test Sequence Visualization

Qualitative V2.2 effect figures should use held-out consecutive `test2` frames, not training-set inline diagnostics.

Build the fixed sample manifest once per split directory:

```bash
python tools/select_kitti_test_sequences.py \
  --train-manifest /mnt/shizhm/CS2S_pose_environment_sat-lidar-ray-posterior-evidence/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/train_manifest.jsonl \
  --test-manifest /mnt/shizhm/CS2S_pose_environment_sat-lidar-ray-posterior-evidence/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/test_manifest.jsonl \
  --output-manifest /mnt/shizhm/CS2S_pose_environment_sat-lidar-ray-posterior-evidence/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30/fixed_test_sequences.jsonl
```

The selector is deterministic. It sorts held-out drives by `(date, drive)`, picks the first two drives that contain at least 16 strictly consecutive `frame_index` values, and selects the middle 16 frames from the first eligible run in each drive. It writes the original JSONL records unchanged, plus a sibling `fixed_test_sequences.report.json` with source SHA256 values, clip boundaries, sample IDs, and train/test overlap checks.

Inference or watcher code should validate the fixed manifest before rendering:

```python
from pathlib import Path
from tools.select_kitti_test_sequences import validate_fixed_test_sequence_manifest

clips = validate_fixed_test_sequence_manifest(
    Path("fixed_test_sequences.jsonl"),
    Path("train_manifest.jsonl"),
    Path("test_manifest.jsonl"),
    Path("fixed_test_sequences.report.json"),
    expected_clips=2,
    frames_per_clip=16,
)
```

This checks the fixed records against the authoritative test manifest, `split == "test2"`, no train sample overlap, no train drive overlap, two 16-frame clips, strictly increasing `frame_index + 1` within each clip, and the report's train/test source SHA256 values.

The V2.2 launcher defaults to this fixed test manifest, `SAMPLE_NUM_SAMPLES=32`, and low-frequency inline previews every 5000 steps. Inline sampling loads and renders samples one by one, so the main concern is DDP wait time during 50-step DDIM rather than stacking all 32 LiDAR feature maps in memory. Training-set manifests may still be passed explicitly for internal debugging, but figures used to discuss model behavior should use the fixed test sequence manifest or the external checkpoint watcher.

## Running V2.2 job: separate checkpoint previews

The active job started with its training-sample dataset already in memory. Updating the launcher does not hot-switch that dataset. Keep training running; use `tools/watch_kitti_test_sequences.py` on idle GPU 6 to render each newly saved checkpoint into `<run>/test_sequences/step_XXXXXX/`. Historical and continuing `<run>/samples/` outputs from this process are internal training diagnostics only. Train-only supervision/gradient probes remain separate as well.

```bash
python tools/watch_kitti_test_sequences.py \
  --run-dir /mnt/shizhm/DATA/KITTI/CS2S_results/kitti_ray_posterior/pixel_lidar_v22_cfgdrop10 \
  --split-dir /mnt/shizhm/CS2S_pose_environment_sat-lidar-ray-posterior-evidence/dataset/KITTI_location/kitti_raw_sat_lidar_geofence_test2_buffer30 \
  --control-dir /home/shizhm/CS2S_run_control/v22_test_sequences_20260914 \
  --training-control-dir /home/shizhm/CS2S_run_control/v22_300k_fresh_20260913 \
  --gpu 6
```

Run the watcher detached with its log redirected, using the `ControlS2S` environment. It does not start or restart training. It starts with the latest saved checkpoint, pins a same-filesystem hardlink only during inference, waits for the assigned GPU to be idle, and stops on inference failure or a `user_stop_request.json` in either control directory. Status and logs are under its control directory; the run's `effect_preview_policy.json` and `test_sequences/latest.json` identify the official previews.

Each checkpoint generates 32 test frames at satellite CFG 3 and 7.5, DDIM 50, eta 0, with the existing AMP inference path. Seed is `2026 + clip_index`, identical within each clip and across CFG/checkpoints. Every frame is independently generated; no previous RGB frame or temporal feature is supplied. Fixed-noise playback helps compare changing conditions but does not establish general temporal consistency across seeds.

The current fixed clips are `2011_09_26_drive_0005_sync` frames 90–105 and `2011_09_26_drive_0039_sync` frames 211–226. Selection is deterministic before inspecting generated results. These clips are for qualitative monitoring, not a replacement for full-test-set quantitative evaluation.

Open `comparison.html` alongside its `images/` directory to select either clip, scrub/play its 16 frames, and switch CFG. Column order is satellite (original aspect ratio), depth-colored LiDAR projected onto RGB, GT, generated RGB. The overlay's GT background is for visualization only, not an inference condition. Raw generated PNGs are retained without postprocessing.
