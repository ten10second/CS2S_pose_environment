# V2.2: native-pixel auxiliary depth supervision

Cleanup note for the fresh run: V2.2 should start from the SD base checkpoint
or a strict same-version V2.2 checkpoint saved by the current code. The
V2.1-to-V2.2 weights-only migration path is intentionally removed to avoid a
warm-start confound. Pixel depth mode should instantiate only the
image-resolution pixel head; V1/V2.1 latent mode keeps the legacy heads for
reproducibility.

V2.1's active depth head predicts a 2x8 map from the U-Net bottleneck. Its targets
first average projected depth into the 16x64 latent grid and then average valid
latent cells into the 2x8 grid. Adjacent foreground/background hits can therefore
share one target. V2.2 supervises their original image locations separately.

## Forward and loss

- The existing camera-projected Utonia features, pixel encoder, multiscale LiDAR
  residuals, satellite ray-posterior weighting and satellite CFG are unchanged.
- A new `LidarPixelDepthHead` reads the final U-Net decoder feature, normally
  320x16x64. It reduces channels to 128, then learns three 2x upsampling stages
  (64/32/16 channels) and predicts one normalized depth channel at 128x512.
  It does not directly read the LiDAR target or condition features.
- `lidar_depth_resample_mode: native` keeps original z-buffer depth and hit mask
  at 128x512. It rejects target/prediction size mismatches instead of interpolating
  an already averaged target. Missing pixels contribute neither loss nor gradient
  to the depth prediction. Valid hits retain their normalized camera depth z/80.
- The loss is the valid-hit mean of absolute log-depth error, weighted by 0.1.
  The 2x8 bottleneck loss is disabled. The existing coarse resampling modes remain
  available, with their previous behavior, for V1/V2.1 reproducibility.

```yaml
# configs/Boost_Sat2Den/train/KITTI_raw_sat_lidar_pixel_v22_cfgdrop10.yaml
# U-Net parameter:
lidar_depth_head_mode: pixel
# Outer model parameters:
lidar_depth_resample_mode: native
lidar_depth_output_scale: 1.0
lidar_depth_bottleneck_scale: 0.0
lidar_depth_loss_weight: 0.1
```

The loss weight is a starting setting, not a validated optimum. V2.2's decoder
feature gradients and V2.1's bottleneck feature gradients have different scopes;
their norm ratios must not be compared as if they measured the same tensor.
The auxiliary prediction is not depth measured from generated RGB. Better depth
loss alone does not establish accurate pole/vehicle geometry in the image.

## Checkpoints, launch and monitoring

The V2.1 run was stopped at the user's request. Its last completed checkpoint is
step 20000; stopping does not create a checkpoint for later unsaved updates.
Existing V2.1 weights, cache files, run directory and config remain intact.

V2.2 uses architecture ID `satellite_lidar_pixel_v22_ray_posterior` and a distinct
run name. Strict `--resume-ckpt` accepts only same-version checkpoints and restores
optimizer/AMP/step. V2.2 does not expose a V2.1 weights-only warm-start path;
fresh runs load the SD base checkpoint and start local step 0. Checkpoints from
the original V2.2 smoke code contained extra legacy depth-head keys and should
not be used for strict resume after this cleanup; resume from a checkpoint saved
by the current code instead.

`tools/launch_v22_pixel_cfgdrop10.sh` is opt-in through
`RUN_V22_PIXEL_CFGDROP10=1`. It retains GPUs 0-3, satellite dropout 0.1, CFG 7.5
sampling, the repaired V2.1 pixel cache, and a 300000-step budget. Trailing CLI
overrides support bounded validation. Creating this launcher does not start a run.

Training logs include `window_lidar_depth_log_l1_mean`, its weighted contribution,
native mask coverage and actual head height/width. Disabled bottleneck metrics are
zero; do not interpret them as a fitted 2x8 prediction. The supervision probe
selects the active head, verifies native target/mask shapes, loss agreement and
finite connected gradients, and captures the shared decoder feature via the
U-Net output layer's input. V2.1 probes retain their bottleneck scope.

The bounded checks below predate the fresh training run. They establish
implementation behavior, not that V2.2 improves image quality.

## Verified on the server (2026-09-13)

- 58 targeted tests passed, covering adjacent foreground/background hit targets,
  empty-mask gradients, the learned image-resolution head, V2.1 compatibility,
  native configuration and strict checkpoint handling.
- Four RTX 3090 GPUs completed two AMP optimizer steps during the original
  implementation check initialized from V2.1 step 20000, with zero skipped
  updates. Actual depth logs report 128x512,
  output scale 1, bottleneck scale 0 and native resampling mode 2.
- The resulting complete checkpoint passed strict model loading and a 16-sample
  probe. Targets/masks exactly match the raw input pixels, invalid zero targets
  are zero, independently recomputed depth loss agrees, and all four sampled
  shared-decoder gradients are finite and connected.
- CFG scale 7.5 sampling with normal and zero-LiDAR perturbations produced valid
  128x512 RGB images. The two-step DDIM output only validates execution.

Artifacts: `/home/shizhm/CS2S_run_control/v22_validation_20260913/` contains
`tests_final.log`, `validation_report.json`, `smoke.log`, and `probe/`.
The bounded checkpoint is under
`/mnt/shizhm/DATA/KITTI/CS2S_results/v22_validation_20260913/ddp4_native_depth_smoke/`.
The first launch attempt was blocked before model initialization by the existing
50-GiB disk floor on the home partition. The successful run writes to the data
partition; the disk guard was not weakened.

The new head is still effectively untrained after two steps. Its pixel loss and
decoder-gradient ratio are not directly comparable with the trained V2.1 coarse
head's loss and bottleneck-gradient ratio. Long-run quality and loss balancing
remain unverified. The fresh-start cleanup passed 58 targeted tests on the
server, including pixel-only head construction and strict same-version resume.
The separate fresh-SD 300k run is controlled and monitored under
`/home/shizhm/CS2S_run_control/v22_300k_fresh_20260913/`; its status and fixed-sample
supervision probes are recorded there. It uses no V2.1 checkpoint.
