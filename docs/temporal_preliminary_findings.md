# Preliminary Temporal Exploration

Status: frozen on `temporal-model` on 2026-08-31. These are inference-time
experiments on the converged 500k single-frame checkpoint, not a trained
temporal model.

## Baseline

- Checkpoint: `step_500000.pt`
- Single-frame model inputs: satellite image and current-frame LiDAR evidence
- Temporal metric: mean LPIPS between consecutive generated frames divided by
  mean LPIPS between the corresponding ground-truth frames
- A ratio near 1 means the amount of frame-to-frame change is similar to the
  real sequence. The metric alone does not measure semantic correctness,
  geometry, identity preservation, or perceptual quality.
- Results cover one seed and two clips. They are not confidence intervals.

## Inference Variants

- `per_frame`: independent initial noise for every frame
- `shared`: one initial noise tensor shared by the sequence
- `autoregressive`: previous generated latent is noised and denoised for the
  next frame (`ar_strength=0.7`)
- `posewarp`: warp the previous latent using OXTS ego-motion and a LiDAR-fitted
  ground-plane homography, while dropping LiDAR-inconsistent dynamic cells
  (`ar_strength=0.5`)
- `posewarp2`: apply the homography only where it agrees with a local
  LiDAR-derived affine flow, and preserve unwarped history elsewhere

This is not an equal-compute ablation. The per-frame and shared-noise variants
run all 50 DDIM steps. After their first frame, `autoregressive` runs 15 steps,
while `posewarp` and `posewarp2` run 25 steps. The measurements therefore
compare complete inference strategies, not the isolated effect of warping or
history reuse at a fixed denoising budget.

It is also not a fully seed-controlled ablation. `ddim_KITTI.py` creates its
per-step noise when the module is imported, before this script calls
`torch.manual_seed`. The CLI seed controls the initial `x_T`, but separate
method processes can use different per-step DDIM noise. The sharded 300-frame
baseline can also use different per-step noise in each process. Future runs
must pass explicitly seeded step noise into the sampler.

## Measured Results

| Test clip | Frames | Method | Generated tLPIPS | GT tLPIPS | Ratio |
| --- | ---: | --- | ---: | ---: | ---: |
| drive 0057 | 58 | per-frame | 0.2157 | 0.1090 | 1.978 |
| drive 0057 | 58 | shared noise | 0.2153 | 0.1090 | 1.975 |
| drive 0057 | 58 | autoregressive | 0.1164 | 0.1090 | 1.068 |
| drive 0057 | 58 | posewarp | 0.1739 | 0.1090 | 1.595 |
| drive 0020 | 300 | per-frame | 0.3025 | 0.1856 | 1.630 |
| drive 0020 | 300 | autoregressive | 0.1523 | 0.1858 | 0.820 |
| drive 0020 | 300 | posewarp | 0.2641 | 0.1858 | 1.421 |
| drive 0020 | 300 | posewarp2 | 0.2425 | 0.1858 | 1.305 |

The exact output snapshot is stored at:

`/mnt/shizhm/DATA/KITTI/CS2S_results/kitti_ray_posterior/ray_posterior_utonia_dino_sd14_4gpu_20260812/inference/temporal_flicker_analysis.json`

## Conclusions

1. In the observed drive 0057 runs, sharing only the initial noise changed the
   ratio from 1.978 to 1.975, which is negligible. Because per-step DDIM noise
   was not held fixed across processes, this does not isolate the effect of
   initial-noise sharing and must be repeated as a controlled multi-seed test.
2. Autoregressive latent reuse is the strongest inference-only stabilizer. It
   brings the 58-frame ratio close to 1, but the 300-frame ratio of 0.820 is
   below the real sequence. This indicates over-smoothing or history locking
   and creates a risk of ghosting and accumulated errors around moving cars.
3. Pose-guided warping helps without directly copying the whole previous
   frame, but its gain is smaller. A single ground-plane homography cannot
   correctly warp buildings, vegetation, independently moving objects, or
   newly visible regions. Sparse LiDAR also limits the dynamic rejection mask.
4. These experiments establish that temporal information is useful, but they
   do not establish a final temporal architecture or an overall quality gain.
   Image quality and condition fidelity must be evaluated alongside tLPIPS.
5. The run metadata does not independently preserve every provenance field,
   including checkpoint, manifest slice, sampling arguments, and Git commit.
   Future formal experiments must record those fields in `run_summary.json`.

## Recommended Resume Point

If temporal training is resumed, start with same-drive clips (`T=4`) and OXTS
relative poses. Freeze the 500k single-frame backbone, sequentially compute or
checkpoint frame features, warp only reliable static history, and add
zero-initialized temporal adapters at the bottleneck and low-resolution UNet
blocks. Keep the single-frame objective and add masked static-region warp
consistency plus a temporal perceptual/semantic loss. Dynamic and disoccluded
regions must be excluded from rigid-warp supervision.

The full UNet should not be naively unrolled over four frames: the single-frame
run already uses roughly 21 GB per GPU.
