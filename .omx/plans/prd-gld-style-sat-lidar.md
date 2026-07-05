# GLD-Style Sat-LiDAR Plan

## Requirements Summary

Goal: build a GLD-inspired, observation-anchored satellite-to-street-view generation project on the current `sat-lidar` branch.

Task input is:

```text
satellite/map + current LiDAR + ego pose + target camera query -> target-view RGB
```

The first paper-level claim should not be "we add LiDAR to CS2S." The defensible claim is:

> In under-constrained satellite-to-street generation, a frozen satellite/street diffusion prior can complete unknown appearance while current-frame LiDAR acts as a target-view geometry anchor whose supported regions are controllable, checkable, and less seed-dependent than unknown regions.

Current repository baseline:
- `dataloader/kitti_raw_lidar_utils.py:24` supports only 4-channel condition modes: `none`, `bbox_dynamic`, `dynamic_points`, `raw_lidar`, `dynamic_full`.
- `dataloader/kitti_raw_lidar_utils.py:443` maps `raw_lidar` into projected point mask and normalized depth.
- `dataloader/KITTI_raw_sat_lidar.py:116` already loads satellite, `image_02`, calibration, OXTS, tracklets, and LiDAR condition fields.
- `models/KITTI_geo_ldm/lidar_condition_model.py:29` provides `LidarMultiScaleControl` for multi-scale residual injection.
- `models/KITTI_geo_ldm_diffusion/openaimodel.py:736` adds LiDAR residuals into UNet middle/skip features.
- `models/KITTI_geo_ldm/txt_control.py:574` can freeze the base prior and train LiDAR control.
- `tools/eval_kitti_raw_sat_lidar_runs.py:262` already supports LiDAR zero/shift probes.
- `tools/eval_kitti_raw_sat_lidar_runs.py:679` includes MiDaS-vs-LiDAR depth consistency diagnostics.
- `tools/build_lidar_normal_pseudolabels.py:173` and `models/KITTI_geo_ldm/lidar_normal_encoder.py:9` provide a separate Metric3D/normal-distillation path.

Out of scope for the first phase:
- Do not directly port full GLD/DA3/VGGT/cascade DiT.
- Do not make object boxes or class tokens the main contribution.
- Do not claim full geometry truth from generated RGB alone.
- Do not depend on the deleted `sat_lidar_plan.md`.

## RALPLAN-DR Summary

Principles:
1. Evidence first: separate LiDAR-observed, satellite/map-prior, occluded, unknown, and conflict regions.
2. Minimal first: prove anchoring on the existing CS2S latent path before replacing the whole latent space.
3. Geometry must be checkable: every geometry claim needs a metric or perturbation probe.
4. Prior and sensor have different jobs: SD/CS2S completes unknown appearance; LiDAR constrains observed geometry.
5. Avoid category leakage: tracklets may define masks/evaluation, but the core condition should remain class-free LiDAR geometry.

Decision drivers:
1. Fastest route to a defensible experiment on the current branch.
2. Clear distinction from ordinary ControlNet-style condition injection.
3. Compatibility with 24GB GPU and existing KITTI raw assets.

Viable options:
- Option A: GLD-inspired control on current SD latent. Predict a target-view geometry latent from satellite/layout prior + current LiDAR evidence + pose, then inject it through geometry-aware residuals. Best first path: small diff, strong diagnostics, compatible with current branch.
- Option B: true geometry latent replacement. Replace/augment SD VAE latent with pretrained geometry features and joint RGB/geometry decoders. Better novelty, but high engineering and memory risk.
- Option C: normal-distilled LiDAR encoder only. Use Metric3D normal supervision to improve LiDAR features, then feed them to control branch. Good support lane, but not sufficient as the main paper claim unless linked to generation commitments.

Decision: start with Option A, integrate Option C as a feature source, and keep Option B as a later stretch.

## Acceptance Criteria

Stage 0 baseline reproducibility:
- Build sat+LiDAR manifests from KITTI raw without relying on committed generated files.
- Run a `raw_lidar` data smoke and one model backward smoke in `ControlS2S`.
- Produce fixed-sample panels for `none` vs `raw_lidar` with identical seed.

Stage 1 evidence field:
- Add a new condition mode, tentatively `observed_geometry`, with at least:
  - sparse point mask;
  - metric or inverse depth;
  - camera-frame xyz or normalized ray coordinates;
  - intensity or density;
  - depth edge/surface discontinuity;
  - visibility/gate mask.
- `lidar_condition_channels(mode)` and train/smoke/eval choices agree.
- Zero-gate test makes LiDAR residuals numerically zero where the gate is zero.

Stage 1.5 DA3-style geometry latent:
- Define teacher latent:
  - `z_geo* = frozen_DA3(target_RGB)` or the chosen frozen geometry foundation encoder output, projected to the CS2S latent grid.
- Define student latent:
  - `z_geo_hat = GeoFusionEncoder(satellite/map, LiDAR evidence, ego pose, target camera query)`.
- The LiDAR part anchors observed metric geometry; the satellite/map part supplies target-view static layout prior for regions LiDAR does not constrain.
- Supervision is mask-aware: strong feature/depth supervision in LiDAR-observed regions, weaker DA3-feature supervision in non-observed regions, and explicit unknown/conflict masks to avoid pretending all geometry is observed.

Stage 2 geometry commitments:
- Add at least one checkable geometry output or commitment:
  - preferred minimal version: generated-image depth consistency using existing MiDaS diagnostic plus direct LiDAR-supported sparse-depth metric;
  - stronger version: lightweight depth/free-space head on predicted x0 or UNet feature.
- Evaluation reports observed-region geometry consistency separately from unknown-region appearance quality.

Stage 3 anchoring proof:
- Same seed, zero/shift/wrong LiDAR changes are localized to observed/gated regions.
- Same evidence, multiple random seeds show lower geometry variance in observed regions than unknown regions.
- `raw_lidar` or `observed_geometry` improves at least two observed-region metrics over `none` while static/background metrics do not regress beyond a declared tolerance.

## Implementation Steps

1. Rebuild current baseline artifacts.
   - Use `tools/build_kitti_raw_sat_lidar_manifest.py` to create `dataset/kitti_raw_sat_lidar/{train,val}_manifest.jsonl`.
   - Run `tools/check_kitti_sat_lidar_alignment.py` on a few samples to verify satellite/camera/LiDAR alignment.
   - Run `tools/smoke_kitti_raw_sat_lidar.py --condition-mode raw_lidar --model-backward`.

2. Make condition-mode contracts explicit.
   - Update `dataloader/kitti_raw_lidar_utils.py` so every mode has documented channel semantics.
   - Add `observed_geometry` instead of reviving stale names from the deleted plan.
   - Update mode choices in `tools/smoke_kitti_raw_sat_lidar.py`, `tools/train_kitti_raw_sat_lidar_control.py`, and `tools/eval_kitti_raw_sat_lidar_runs.py`.

3. Implement target-view evidence field.
   - Extend `generate_lidar_condition` with z-buffered projected LiDAR statistics.
   - Produce explicit validity/gate masks instead of relying on `dynamic_mask`.
   - Keep tracklet-derived `dynamic_mask` only for object-region loss/eval, not as the main condition.

4. Add geometry-aware control gating.
   - Use `LidarMultiScaleControl.gate_channel` and `gate_residuals` for observed-region residual permissions.
   - Preserve `static_teacher_loss_mask` in `models/KITTI_geo_ldm/txt_control.py:196` as a background-preservation regularizer outside the gate.
   - Add a tiny numerical test through smoke script: zero gate implies zero control residual.

5. Add satellite-LiDAR geometry fusion for `z_geo_hat`.
   - Implement a `GeoFusionEncoder` with three inputs:
     - satellite/map features from the existing sat encoder or a lightweight layout encoder;
     - LiDAR evidence features from a PointPillars-lite / target-view adapter;
     - pose/camera-query embeddings.
   - Fuse in target-view latent coordinates and output `z_geo_hat`.
   - Keep explicit observed/unknown/conflict masks so LiDAR-supported regions can override prior completion while unobserved regions remain prior-guided.

6. Integrate normal-distilled features only after Stage 1 works.
   - Train or load `LidarNormalEncoder` from `models/KITTI_geo_ldm/lidar_normal_encoder.py`.
   - Rasterize predicted per-point normal/feature channels into the same target-view condition grid.
   - Compare manual evidence field vs evidence field plus normal-distilled channels.

7. Add commitment/evaluation layer.
   - Reuse `tools/eval_kitti_raw_sat_lidar_runs.py` for zero/shift probes and depth consistency.
   - Add observed-vs-unknown region summaries.
   - Add same-evidence multi-seed evaluation and report variance inside gate vs outside gate.

8. Train in escalating scopes.
   - 4-8 frame overfit: prove wiring and localized response.
   - 1k object-rich subset: prove signal before full run.
   - Full `2011_09_26` sat+tracklet set: compare `none`, `raw_lidar`, `observed_geometry`.
   - Multi-date normal pretraining stays separate unless it improves observed-region metrics.

## Risks And Mitigations

- Risk: richer condition still behaves like generic ControlNet.
  Mitigation: wrong/shifted LiDAR probes and gate-localized change ratios are required acceptance checks.

- Risk: RGB metrics reward texture, not geometry.
  Mitigation: report observed-region sparse-depth/free-space consistency and seed stability separately.

- Risk: tracklet masks leak object semantics.
  Mitigation: use tracklets for masking/evaluation and optional losses; keep core LiDAR condition class-free.

- Risk: normal pseudo-labels are noisy.
  Mitigation: keep normal-distilled features optional until baseline evidence field passes; use label weights and depth-consistency filtering from `tools/build_lidar_normal_pseudolabels.py`.

- Risk: full GLD-style latent replacement is too large for current branch.
  Mitigation: treat it as Phase 2 only after anchored-control metrics show a real effect.

## Verification Steps

Core smoke:
- `conda run -n ControlS2S python tools/smoke_kitti_raw_sat_lidar.py --condition-mode raw_lidar --batch-size 1 --model-backward`
- After Stage 1: same command with `--condition-mode observed_geometry`.

Data/alignment:
- `conda run -n ControlS2S python tools/check_kitti_sat_lidar_alignment.py --manifest dataset/kitti_raw_sat_lidar/val_manifest.jsonl --num-samples 8`

Training proof:
- Overfit run produces finite loss and visible localized changes in validation snapshots.
- Gate-zero residual max is zero or within floating tolerance.

Evaluation proof:
- Generate `none`, `raw_lidar`, and `observed_geometry` with fixed seeds.
- Run `--lidar-probe normal`, `zero`, and `shift_x`.
- Run depth consistency where dependencies/checkpoints permit.
- Report observed/unknown seed variance.

## ADR

Decision: implement a GLD-inspired target-view geometry state on the current CS2S latent path before attempting true GLD latent replacement. This state is predicted from satellite/map + current LiDAR + pose/camera query, not from LiDAR alone.

Drivers:
- Current branch already supports SD latent diffusion, satellite conditioning, LiDAR residual control, KITTI raw projection, and normal pseudo-label assets.
- The main research gap is evidence anchoring, not another generic condition module.
- A full GLD port has high memory and integration risk.

Alternatives considered:
- Direct GLD port: rejected for first phase because it would replace too much of the current code and delay falsifiable experiments.
- Keep 4-channel `raw_lidar` only: rejected because it cannot support the pasted R1 claim about depth/free-space/visibility commitments.
- Object-box conditioning: rejected as main path because it weakens the category-free LiDAR observation story.

Consequences:
- First paper version is "GLD-inspired" rather than "GLD reproduction."
- Success depends on stronger evaluation, not just prettier RGB.
- Later true geometry-latent work remains possible if Stage 1-3 prove the phenomenon.

Follow-ups:
- Create a short test spec for `observed_geometry` channel semantics.
- Decide whether the first commitment head should be explicit depth/free-space prediction or evaluation-only sparse-depth consistency.
- Once baseline results exist, write the paper story around observability boundaries and evidence attribution.
