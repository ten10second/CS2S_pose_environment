# Stage D: diagnostic generated-history rollout, 2026-09-09

## What was run

- Frozen single-frame CFG base: `base/cfg_step_250000.pt`.
- Frozen history adapter: `stage_d_post_bottleneck/geometry_history_step_1000.pt`, decoder block 12.
- Held-out drive: `2011_10_03_drive_0034_sync`, frames 1228–1259, manifest start index 13333.
- Exactly 32 consecutive frames: one observed RGB followed by 31 generated frames.
- Two conditions: history ON versus history OFF, both loading the same weights.
- Actual satellite-zero CFG = 3, with LiDAR retained in both CFG branches.
- 50 DDIM steps, eta=1, temperature=1, seed=20260909.
- Independent processes on GPUs 4 and 5; no training and no use of GPUs 0–3.

The sequence was selected before generation by viewing the first real RGB of
two available contiguous held-out candidates. Drive 0034 has visible facades,
roadside geometry and a motorcycle; the initially considered drive 0023 was
tree-dominated. No generated result was used to select or replace the clip.

## Minimal sampler changes needed for a fair comparison

`tools/generate_kitti_geometry_history.py` now supports `--disable-history`
and a strict contiguous-clip guard. The disabled run clears the history hub,
not the checkpoint or backbone. No network structure or learned weight changed.

The original sampler created DDIM step noise at module import and did not pin
the first-RGB VAE sample. Both were explicitly seeded before this experiment.
Without that fix, equal per-frame seeds would not pair separate eta>0 runs.

Per-frame records include initial-noise hashes, input/output history-latent
hashes, actual denoiser batch sizes, and per-step history diagnostics. These
observers do not replace the denoiser output. Future RGB is used for GT display
only after bootstrap; replacing the first generated frame's RGB/semantic
features with zeros left all sampling condition tensors exactly unchanged in
both runs. This runtime check complements the inspected sampling code path;
it is not a provenance audit of external caches.

## Verification

The comparison assembler verified:

- Both first displayed frames are pixel-identical to their GT.
- All 31 per-frame initial-noise hashes match ON/OFF.
- The full DDIM step-noise bank and bootstrap VAE latent match ON/OFF.
- All 31 ON input-history hashes equal the preceding output-latent hash.
- Every generated frame uses exactly 50 denoiser calls, each with actual CFG
  batch size 2.
- History ON executes 31 × 50 = **1,550** history attention calls; OFF executes
  **zero** history attention calls.
- Only offset 0 is an observed frame. Subsequent history comes from generated
  latent content, not from repeated GT encoding.
- Both videos encode as H.264, 1024×852, 32 frames. Normal playback is 10 fps
  (3.2 s); slow playback is 5 fps (6.4 s).

The first local video assembly failed because the conda FFmpeg build lacks
libx264. The assembler now checks available encoders and uses libopenh264 on
this machine, with an encoder-selection regression test. No generation was
repeated for this packaging issue.

## Observations and limits

Visual inspection of the selected chronological frames shows facade edges,
curb geometry and the parked car changing screen position/scale in both ON and
OFF, consistent qualitatively with camera advance. This clip does **not** show
the whole-background screen-lock failure of raw latent inheritance. It does
not establish exact pose-consistent geometry over the complete image.

The ON/OFF images are very similar at ordinary viewing scale. Differences are
more apparent in the signed-difference montage amplified ×8. On the raw PNGs,
excluding the shared observed frame:

- Mean absolute ON/OFF RGB difference: **0.9152 / 255** (about 0.36% of range).
- Per-frame mean difference ranges from about **0.5785** to **2.0417 / 255**.
- Some individual pixels differ substantially (maximum 159/255); a small
  global mean does not mean identical output at every pixel.

These are output-sensitivity diagnostics, **not** flicker, motion or image
quality scores. No clear temporal improvement is established by this small
change. The current visual evidence fits the user's "nearly indistinguishable
from history off" category more closely than "history was never executed".
There is insufficient evidence to claim improved disocclusion synthesis,
dynamic identity preservation, reduced trails, or absence of long-run drift.

Runtime ON diagnostics across generated frames and denoising steps:

- Original geometric valid coverage: mean **41.50%**, range **36.52–51.86%**.
- Mean history residual / backbone-input norm: **2.016%** over all queries.
- Mean history residual / fused-condition norm: **24.33%**.
- Mean null probability over all queries: **76.02%**, including queries forced
  invalid by geometry. This is not an eligible-query rejection rate.

The history route is active and nonzero under real CFG, yet its visible output
influence is small on this clip. This does not uniquely identify the cause:
teacher-forcing mismatch, supervision, CFG-conditioned residual response and
effective appearance transfer remain hypotheses. One seed and 3.1 seconds of
generated continuation cannot prove that epsilon-MSE can never learn temporal
behavior or that a particular replacement mechanism is required.

No additional training, new layers, supervision changes or strength sweep were
started. This run supplies the requested diagnostic evidence, not a paper-level
temporal-effectiveness claim.

## Files

Workspace artifact root:
`artifacts/geometry_history_20260908/stage_d_rollout32_20260909`.

Server artifact root:
`/mnt/shizhm/DATA/KITTI/CS2S_results/geometry_history_20260908/stage_d_rollout32_20260909`.

- `comparison/comparisons.mp4`: normal speed; rows GT / HISTORY OFF / HISTORY ON.
- `comparison/comparisons_slow.mp4`: half speed, same frame order.
- `comparison/montage_selected.png`: offsets 0, 1, 8, 16, 24, 31.
- `comparison/difference_montage.png`: signed ON−OFF ×8, gray means zero.
- `comparison/verification.json`: pairing checks and raw-image differences.
- `history_on/`, `history_off/`: all source PNGs, records and run summaries.
- `plan.json`, `jobs.json`, and per-run logs: frozen setup and execution provenance.
