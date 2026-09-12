# LiDAR Pixel V2.1

V2.1 conditions the diffusion UNet with pixel-aligned LiDAR features instead of the old 8x32 visible-ray token cache.

## Interface

- Dataset cache root: `lidar_pixel_feature_cache_root`
- Batch tensors:
  - `lidar_pixel_features`: `[B, 576, 128, 512]`, float16 cache loaded as tensors
  - `lidar_pixel_features_mask`: `[B, 1, 128, 512]`
  - `lidar_pixel_features_available`: per-sample availability flag
- Context encoder target:
  - `models.KITTI_geo_ldm.lidar_pixel_condition.LidarPixelConditionEncoder`
- Encoder output:
  - `lidar_context["features"]`: four spatial feature maps with channels `[64, 128, 256, 256]`
  - `lidar_context["masks"]`: matching four spatial masks
  - `last_semantic_pred_tokens`: `[B, 256, 384]` for DINO semantic alignment

The encoder no longer emits old LiDAR attention tokens. Token structure loss is fixed to zero for V2.1.
The old token-attention `lidar_reference_window` setting, including 1x1 or 3x3 local windows, is not used by the V2.1 spatial path.

## Cache Format

Pixel cache memmap uses ragged visible-pixel storage:

- metadata: `pixel_memmap_meta.json`
- format: `kitti_pixel_feature_ragged_memmap_v1`
- arrays: `features.npy`, `pixel_index.npy`, `depth.npy`, `offsets.npy`

The training launcher accepts this ragged format or the original lossless per-frame NPZ directory. NPZ preflight validates every required sample using at most eight concurrent decompressions, returning only counts; in DDP, rank 0 validates once and broadcasts the result. Neither path expects dense `kitti_feature_memmap_v1` feature/mask files.

The old V2 8x32 semantic cache cannot be converted back into V2.1 pixel features: its per-point feature identity has been averaged away. V2's separate 128x512 depth/hit evidence still exists. Rebuild the semantic cache from the original single-scan point features.

Visibility is selected on the complete raw scan before restricting semantic features to Utonia's 80 m input range. A pixel without an eligible Utonia feature stays masked even if raw LiDAR depth is available there; cached feature hits must be a depth-consistent subset of raw hits. Missing observations are unknown, not proven free space.

## Network change

The pointwise nonlinear encoder consumes normalized Utonia features, continuous log depth and hit mask before any spatial reduction. Six learned masked stride-2 convolutions produce condition features at 16x64, 8x32, 4x16 and 2x8. They are injected after the first residual block at each U-Net encoder scale, entering both the main path and its skips. Each injection has a support mask, learned gate and zero-initialized projection. The masks indicate propagated observation support, not geometric certainty.

Satellite GCA and the LiDAR depth posterior remain enabled. V2.1 no longer uses the old 1x1/3x3 LiDAR token attention. The 8x32 DINO head remains auxiliary and does not compress the generation condition back to one token per patch. Corrected `masked_area` depth supervision remains at weight 0.1 (bottleneck head only), and satellite condition dropout remains 0.1. These are soft learned conditions; pixel-aligned input does not guarantee exact output geometry.

## Training Entry Points

- Config: `configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_pixel_cfgdrop10.yaml`
- Launcher: `tools/launch_v21_pixel_cfgdrop10.sh`
- Required CLI cache arg: `--lidar-pixel-feature-cache-root`

The launcher is guarded and exits without starting training unless `RUN_V21_PIXEL_CFGDROP10=1` is set.

## Build pixel features before a full run

Use the Utonia environment for extraction and the training environment for conversion/training. Run extraction separately on the final train and test manifests, writing into the same pixel NPZ directory. `--num-shards` / `--shard-index` partition extraction when needed; `--skip-existing` resumes extraction. The NPZ directory can be used directly as the training pixel cache. If storage permits and mmap is preferred, optionally convert once into an empty ragged-cache directory:

```bash
python tools/build_kitti_utonia_pixel_cache.py \
  --manifest /path/to/train_manifest.jsonl \
  --out-root /path/to/pixel_npz \
  --kitti-root /path/to/KITTI_RAW \
  --utonia-root /path/to/Utonia --ckpt /path/to/utonia.pth
python tools/build_kitti_utonia_pixel_cache.py \
  --manifest /path/to/test_manifest.jsonl \
  --out-root /path/to/pixel_npz \
  --kitti-root /path/to/KITTI_RAW \
  --utonia-root /path/to/Utonia --ckpt /path/to/utonia.pth
python tools/convert_kitti_pixel_cache_memmap.py \
  --source-root /path/to/pixel_npz --output-root /path/to/pixel_memmap
```

Extraction stores only occupied pixels; the loader renders them onto the image grid. Conversion uses two bounded-memory passes, with metadata published only after successful writes. Training preflight checks array headers, offsets and complete manifest coverage. V2.1 has a different checkpoint architecture; strict resume rejects V2 checkpoints. The supplied full-training launcher initializes from SD rather than silently treating V2 as a resumable V2.1 model.

## CFG and runtime validation (2026-09-12)

When guidance differs from 1 and no unconditional satellite embedding is supplied, DDIM now constructs the zero-satellite branch used by training dropout. LiDAR features/evidence remain identical in both branches. Merely passing a guidance scale previously left this sampling entry point in conditional-only mode. Spatial injection statistics now have explicit `lidar_spatial_*` fields. AMP-generated images are converted to float32 before CPU image saving.

Validation on the server's ControlS2S environment:

- 60 targeted unit/regression tests passed, including legacy V2 paths, masked-area depth supervision, pixel cache integrity, multiscale gradients and CFG branching.
- One real training scan produced 17,794 visible feature pixels; these matched raw projected depths. The raw depth map had one additional hit outside Utonia's input range, which correctly remained absent from semantic features.
- Two GPUs, AMP, batch 1 per rank, two optimizer steps completed and saved a reloadable checkpoint. Losses were finite; peak reported per-rank allocation was about 19.1 GB.
- The saved checkpoint generated finite 128x512 RGB outputs with two DDIM steps and guidance 7.5. Both satellite branches ran with batch 2; all four LiDAR pyramid levels were duplicated correctly. All four spatial projections produced nonzero messages with normal input and exactly zero messages with the zero-LiDAR probe.

Server validation artifacts are under `/home/shizhm/CS2S_run_control/v21_validation_20260912/` (`tests_final.log`, `ddp_amp_2step.log`, `check_cfg.py`, `cfg_smoke_report.json`). These are functional checks on one training sample, not evidence of generation quality or geometric improvement. The full pixel cache has not been built, and full V2/V2.1 training remains stopped.


## Full training and supervision monitoring

The four-card launcher defaults to the lossless NPZ cache under `/home/shizhm/CS2S_cache_npz/utonia_pixel_lidar_v21_fp64_all_fp16`. The 64-frame size audit estimated 476.8 GiB for uncompressed all-split pixel arrays, while one representative compressed NPZ was 1.25 MiB. NPZ compression preserves feature values. No full memmap is created on the nearly full data disk.

### Projection consistency repair (2026-09-12)

The initial 16-frame supervision probe stopped before formal V2.1 training: cached pixel 36902 was absent from the runtime hit mask in `2011_09_26/2011_09_26_drive_0095_sync/0000000155`. Raw point 28554 had the same coordinates, calibration and depth (10.45220947265625 m) in both environments. However, FP32 matrix products produced x=38.5 under the cache environment (NumPy 2.2.6) and x=38.5000114440918 under the training environment (NumPy 1.24.3). Nearest-pixel rounding selected columns 38 and 39 respectively. Differences already existed before image scaling; making the scale dtype explicit alone was insufficient.

The shared projection now uses FP64, non-BLAS `einsum` contractions (`optimize=False`), explicit Python-float image scales, and one final FP32 conversion before visibility checks and pixel rounding. Camera-coordinate point maps share the same transform. The strict hit/depth validation remains enabled. New caches record `projection_version=float64_einsum_v1` and `source_point_index`.

All 24,597 old cache frames reproduced exactly in their original NumPy environment. With canonical projection, 20,626 frames can reuse existing per-point features; 3,971 require extraction because some newly visible source points have no stored feature row. Migration must validate legacy pixel/depth arrays exactly, remap by original scan index, and defer frames with missing features for complete extraction. Never discard newly visible points or assign a neighbouring point's feature. Keep the old cache and write the repaired cache into a separate root; `--skip-existing` alone does not migrate legacy caches.

The full cross-environment audit found different legacy pixel-index arrays in 2,845 frames. Canonical geometry hashes (visible source indices, pixel indices, and FP32 depths) agreed for all 24,597 frames across NumPy 2.2.6 and 1.24.3. A frame count is not a count of incorrect pixels: the reproduced diagnostic failure involved one boundary point.

This repairs projection and cache consistency. It does not change the V2.1 network, `masked_area` supervision, depth loss weight 0.1, or satellite CFG. It is not evidence that generated RGB geometry has improved; that requires subsequent evaluation.

The run remains 0–3 GPUs, 300,000 steps, one sample per rank, AMP, learning rate 1e-5, satellite dropout 0.1, depth weight 0.1, semantic weight 0.2. Checkpoints retain the latest two step files; the output disk floor is 50 GiB. `PIXEL_CACHE_ROOT` and `SAMPLE_MANIFEST` override the server paths. The launcher accepts trailing training CLI overrides for bounded smoke tests or explicit resume.

Monitoring has three separate scopes:

- Train log: depth/semantic losses, observation support, spatial injection strength and AMP skipped steps. These do not prove RGB geometry quality.
- Fixed training images: every 1,000 steps, normal/zero-LiDAR generations use the same per-image seed, eta 0 and satellite CFG 7.5. AMP sampling is supported with the optimizer still resident, and sampling restores the training RNG state. The diagnostic samples are drawn from the existing training split; they are not a new held-out validation set.
- `tools/probe_kitti_pixel_supervision.py`: a fresh SD baseline and subsequent checkpoints use fixed training samples, seeds and timesteps `[50,250,500,750]`. The probe captures the actual depth targets/masks passed into `p_losses`, checks invalid zeros before clamping, and checks reconstructed depth loss against training's recorded value. It measures weighted depth and remaining-loss gradients with respect to the shared U-Net bottleneck feature, including their norm ratio and cosine. This is not a gradient ratio over all model parameters. The semantic head is upstream of this bottleneck and has no direct gradient with respect to that feature. The probe performs no optimizer updates.

Server control files are under `/home/shizhm/CS2S_run_control/v21_300k_cfg_20260912/`. A detached supervisor waits for all cache shards and the verified launch marker, runs the initial diagnostic, and then starts training. It records health once per minute and schedules checkpoint probes on GPU 5 only when that GPU is idle. A `user_stop_request.json` stops this pipeline; failures are recorded rather than silently restarting training. No loss weights are changed automatically.

Functional validation of this startup support: the NPZ path completed two-rank AMP training with two data-loader workers per rank, live-optimizer CFG sampling for normal/zero conditions, and checkpoint saving. Both fresh-model and checkpoint supervision probes are checked separately. Long-run supervision benefit and fine-object image geometry remain measurements to be established.
