# Stage D: frozen expanded evaluation (2026-09-09)

## Scope and frozen inputs

This is evaluation only: no optimizer, no additional training, no model edits,
and no generated-history rollout. The baseline is the frozen CFG checkpoint
`base/cfg_step_250000.pt`; adapter is
`stage_d_post_bottleneck/geometry_history_step_1000.pt` (decoder block 12 only).

Server experiment root:
`/mnt/shizhm/DATA/KITTI/CS2S_results/geometry_history_20260908`.
Evaluation output: `stage_d_fixed_eval_20260909_v2`.

The first attempt (`stage_d_fixed_eval_20260909`) stopped during original-probe
reproduction, before expanded evaluation: the new repeat helper compared the
DDPM subtotal with the outer model total. The helper was corrected and a
regression test added. Failed logs are preserved. The successful retry uses
identical selected pairs and seeds; no acceptance tolerance was relaxed.

`selection.json` is written before model evaluation and records checkpoint,
base checkpoint, config, manifest and evaluator hashes, every selected pair,
wrong-history source, and all RNG seeds. Existing experiment directories are
not overwritten. GPUs 4–7 are used only after an idle check; GPUs 0–3 remain
reserved for CFG training.

## Sample availability and interpretation

The nominal target was 32 pairs per held-out drive. Two held-out drives contain
only 23 and 12 adjacent pairs, respectively. Do not duplicate or densely pack
those pairs to manufacture a 160-pair count.

Select greedy frame-spaced candidates (minimum 10 current-frame indices), then
uniformly subsample up to 32 per drive. Exclude both endpoints of the four
original probe pairs. This deterministically yields **101 new pairs**, with
drive counts **3 / 32 / 2 / 32 / 32**, plus the original four probes reported
separately. One-second spacing is not statistical independence. The two short
drives cover very little time and cannot establish within-drive generalization.

## Ordered gates

1. `--phase prepare`: freeze the selection and provenance before inference.
2. `--phase reproduce`: each of the original four ranks reconstructs its held-out
   pair and the corresponding training-pair wrong history using the original
   VAE and probe seeds. Compare ordinary versus instrumented `fixed_probe`
   exactly, then compare all recorded loss components with the original step
   1000 log exactly. Stop on mismatch; do not relax tolerance silently.
3. `--phase expanded`: require all original-reproduction verdicts to pass.
   Evaluate new pairs, with `disabled`, `correct`, `wrong_history`, and
   `wrong_geometry` at t=250 and t=750. Wrong history is from the training
   partition, hence a different drive; wrong geometry preserves target validity
   and reverses the valid source-coordinate ordering, as in the original probe.
4. Summarize only complete outputs. Report original and expanded cohorts
   separately, per drive and pooled; do not use total loss to claim RGB benefit.

## Measurement contracts

- Same current VAE randomness, diffusion noise, history latent and timestep
  across paired conditions. Correct and wrong history latents are each sampled
  once and reused; record their hashes.
- Capture existing DDPM `loss_raw` and attention return locals with a scoped
  Python profiler. Do not request extra x0 decoding or replace any model output.
- Primary regional mask is the **original correct geometry** `history_valid`
  at the RGB latent resolution, common to all conditions, including disabled
  and wrong geometry. No mask-dependent outcome selection.
- Average squared error over RGB latent channels, then report all/valid/invalid
  regions, counts and coverage. Empty regions are null, not zero.
- Check the all-region mean against `loss_eps_base` within 1e-7 for reduction
  rounding; logged replay metrics themselves must match exactly.
- Repeat disabled independently at each timestep and require exact equality.
  Bottleneck depth loss must be exactly equal across all four conditions.
- Report attention's actual eligible-query coverage separately: conservative
  downsampling can erode support relative to the primary geometric mask.
  Report null probability on all queries and eligible queries separately.
- Residual diagnostics use ratio of mean token norms, against both backbone
  input `x` and fused condition summary; these denominators are not equivalent.
- No parameter updates are possible (`requires_grad=False`, no optimizer), and
  parameter version counters are checked at completion.

## Questions, not assumed conclusions

1. Do correct-minus-disabled and correct-minus-wrong comparisons retain their
   signs across additional drives and pairs? Report paired differences, wins,
   distributions and exploratory temporal-block bootstrap uncertainty.
2. Is whole-image benefit small because benefit is confined to supported cells?
   Test per pair: `delta_all = p * delta_valid + (1-p) * delta_invalid`.
   Do not substitute product-of-aggregate-means for this identity.
3. Does improvement in valid cells trade off against degradation elsewhere?
4. Is read support mostly removed at the injected resolution, or are eligible
   queries predominantly choosing null? Neither small residuals nor null mass
   alone prove the cause of small RGB gains.

Bootstrap blocks are contiguous `frame_index // 100` bins within each drive.
Equal-drive macro results prevent long drives from automatically dominating.
The few drives, short-drive degenerate blocks, single checkpoint and fixed
noise realization limit inference. These are teacher-forced denoising probes,
not evidence of generated-video motion, appearance persistence or rollout
stability. No architecture or supervision change follows automatically.

## Execution results

Completed on 2026-09-09. All 4 original validation probes reproduced exactly:
32 condition/timestep records, zero loss-component mismatches. Expanded output
contains 808 records = 101 pairs × 2 timesteps × 4 conditions. All 202 repeated
disabled checks pass exactly; bottleneck depth differences are exactly zero.
All four expanded processes completed, parameter versions remained unchanged,
and GPUs 4–7 were released. No new training or rollout was run.

Local and server verification: 129 tests run, zero failures, one CUDA-only skip
in CPU test runs. The evaluator itself ran on all four available server GPUs.

### 1. Does the signal survive expansion? Yes, at a small magnitude.

Positive relative change means lower correct-history RGB denoising error.
Primary table averages per-pair loss, excluding the original four probes.

| Comparison | t=250 | t=750 |
| --- | ---: | ---: |
| Correct vs disabled, relative improvement | 0.1668% | 0.4402% |
| Correct vs disabled, wins | 82/101 | 83/101 |
| Correct vs wrong history, wins | 90/101 | 90/101 |
| Correct vs wrong geometry, wins | 92/101 | 91/101 |
| Equal-drive mean of relative improvements vs disabled | 0.1573% | 0.2808% |

Exploratory within-drive temporal-block bootstrap 95% intervals for mean
absolute disabled-minus-correct RGB error are [0.0002300, 0.0003805] at t=250
and [0.00005022, 0.00007328] at t=750. These intervals condition on these drives
and fixed VAE/noise draws; they are not uncertainty over new drives or training
seeds. Two short drives contain only one temporal block each.

| Held-out drive suffix | New pairs | t=250 improvement / wins | t=750 improvement / wins |
| --- | ---: | ---: | ---: |
| 0002 | 3 | +0.2679% / 3/3 | +0.3332% / 3/3 |
| 0023 | 32 | +0.1507% / 26/32 | +0.4433% / 26/32 |
| 0079 | 2 | +0.0209% / 1/2 | **−0.2822% / 0/2** |
| 0117 | 32 | +0.1637% / 25/32 | +0.3532% / 25/32 |
| 0034 | 32 | +0.1834% / 27/32 | +0.5564% / 29/32 |

Thus this is not merely the original four-pair positive sign. It is also not
uniform success: drive 0079's two t=750 probes are worse with correct history,
and its mean correct-vs-wrong contrasts are negative. Its sample count is too
small to diagnose a drive-specific cause. The three longer drives provide
96/101 pairs and each shows positive mean off/history/geometry contrasts at
both timesteps.

### 2. Is there spatial dilution? Yes, but no hidden large effect.

The following region means first normalize each pair by its region's cell
count, then average pairs. They are not pooled-pixel averages across frames.

| Region: relative correct-vs-disabled improvement | t=250 | t=750 |
| --- | ---: | ---: |
| Full image | 0.1668% | 0.4402% |
| Correct-geometry valid cells | 0.3511% | 0.9240% |
| Invalid cells | 0.0235% | 0.0766% |

Exact per-pair weighted decomposition attributes **91.42% / 89.54%** of the
full-image absolute benefit to valid cells. Mean weighted contributions are:

- t=250: valid +0.0002789296, invalid +0.0000261683.
- t=750: valid +0.0000543773, invalid +0.0000063499.

This supports spatial dilution of the reported full-image metric. It does
**not** establish that training gradients were diluted or that changing to
valid-only supervision will improve the model. Valid-cell improvements remain
below 1% in this evaluation.

### 3. Is the model harming unsupported regions? Not on average.

Invalid-region mean benefit is positive at both timesteps, so there is no
aggregate cancellation by invalid-region degradation. There are local
tradeoffs: valid improves while invalid worsens in 25/101 pairs at t=250 and
21/101 at t=750. Do not turn the positive average into an all-samples safety
claim. Geometry-invalid cells are not equivalent to all disoccluded regions
or a static/dynamic segmentation.

### 4. Is the history branch closed or almost zero? Not where reading is allowed.

Mean primary geometry coverage is **47.44%**, while actual eligible decoder
queries cover **33.11%**. Conservative resizing/local bounds remove some read
support, but the remaining coverage is not near zero. Do not infer that
loosening the geometry gate is safe from these measurements alone.

| Correct-history diagnostic | t=250 | t=750 |
| --- | ---: | ---: |
| Null probability over all queries | 70.43% | 70.37% |
| Null probability over eligible queries only | 10.49% | 10.31% |
| History residual / backbone norm, all queries | 2.43% | 2.47% |
| History residual / backbone norm, eligible queries | 7.35% | 7.44% |
| History residual / fused-condition norm, all queries | 26.23% | 26.04% |

The high all-query null average is largely forced by geometric ineligibility,
not evidence of learned rejection of useful history. Eligible-query residuals
are appreciable; small output error gains do not demonstrate an unopened
output projection. Conversely, norm magnitude does not prove useful influence
on the final image.

Wrong-history eligible null probabilities are 9.03% / 9.01%, lower than correct
history. Thus this experiment does not demonstrate selective null rejection
of incorrect history, even though the correct-history loss is usually better.

## Decision boundary

Evidence supports retaining Stage D as a meaningful small-positive baseline.
The small effect is not explained solely by the old four-pair sampling, a
depth-loss shortcut, invalid-region mean harm, or an almost entirely closed
history branch. Spatial metric dilution is present but only part of the story.
No next architecture, supervision change, longer training, or rollout is
automatically authorized by this report. GT-history denoising effectiveness
is not generated-video temporal effectiveness.

Artifacts in `stage_d_fixed_eval_20260909_v2`:
`selection.json`, `original_rank*.jsonl`, `reproduction_rank*.json`,
`evaluation_rank*.jsonl`, `*_complete_rank*.json`, `summary.json`, `report.md`,
and both execution logs. Local copies are under
`artifacts/geometry_history_20260908/stage_d_fixed_eval_20260909_v2` in the
workspace. Raw records preserve all four conditions' diagnostics, seeds and
history-latent hashes for independent analysis.
