# Geometry-local history: A/B/C execution record

Stage A only checks whether sparse geometry can provide useful current-query to
previous-history coordinates. It is not a semantic building/facade detector and
does not train a temporal model.

## Core Output

`tools.temporal_history_geometry.build_pair_geometry(prev_row, cur_row, kitti_root, grid=(16, 64))`
returns:

- `history_grid`: `(H, W, 2)` float32 grid-sample coordinates for the previous
  feature map, using `align_corners=False`.
- `history_valid`: `(H, W)` bool. Invalid means unknown, not identity fallback.
- `metrics`: JSON-serializable coverage and motion diagnostics.

The projection core uses the full `P_rect_02` 3x4 matrix, including its
translation column, plus `R_rect_00`, Velodyne calibration, OXTS pose, and
previous-scan z-buffer support.

## Diagnostic Command

```bash
python tools/audit_temporal_history_geometry.py \
  --manifest /path/to/train_manifest.jsonl \
  --kitti-root /path/to/KITTI_RAW \
  --out-dir /tmp/geometry_audit \
  --num-pairs 32
```

The audit writes:

- `geometry_metrics.jsonl`: one JSON record per pair.
- `summary.json`: aggregate coverage and preregistered pilot gate flags.
- `overlays/*.png`: current valid cells and previous lookup coordinates.

## Preregistered Pilot Gate

These are feasibility flags for deciding whether Stage B is worth running, not
claims about semantic building stability:

- `pilot_gate_valid_coverage_pass`: mean `coverage_all >= 0.05`.
- `pilot_gate_non_ground_proxy_all_pass`: mean
  `coverage_non_ground_proxy_all >= 0.05`.
- `pilot_gate_rgb_motion_pass`: at least 16 motion-bearing pairs
  (`prev_to_cur_translation_m >= 0.2`) and median
  `rgb_error_identity_minus_mapped_mean > 0`.

Important metric names:

- `coverage_all`
- `coverage_of_current_lidar`
- `coverage_ground_proxy_all`
- `coverage_non_ground_proxy_all`
- `mean_reprojection_vs_identity_cells`
- `prev_to_cur_translation_m`
- `rgb_error_mapped_mean`
- `rgb_error_identity_mean`
- `rgb_error_identity_minus_mapped_mean`

## Frozen base and safety boundary

This is an engineering feasibility experiment, not a novelty claim. Actual
current-frame LiDAR is supplied: this is conditional rendering, not future
prediction. Legacy v2/ar_dyn entrypoints remain available as controls.

- Server experiment root:
  `/mnt/shizhm/DATA/KITTI/CS2S_results/geometry_history_20260908`.
- Frozen base: `base/cfg_step_250000.pt`, hard-linked from the complete current
  CFG run step checkpoint (not `last.pt`). The training run's retention cannot
  delete this experiment's weights.
- SHA256: `45f2d561302efec8817d7ac1f1c5307ba89546d0f1c38f250f058144c2ed183c`.
- Paired original config/args are copied into `base/cfg_run_config.yaml` and
  `base/cfg_run_args.json`.
- Only physical GPUs 4,5,6,7 may run this experiment. GPUs 0–3 and the existing
  CFG training must not be stopped, restarted or reconfigured.
- No new dependencies; no failed v2 weights are reused. Pre-edit copies of the
  three shared legacy files are stored under `code_before/`.

## A: measured server result

64 motion-bearing pairs, round-robin across drives; synthetic identity,
off-centre identity, translation, rotation, P_rect translation, ground-plane
sign, occlusion and unknown-support tests pass.

| Metric | Result |
| --- | --- |
| Valid query coverage | 43.63% |
| Ground proxy supported cells / whole grid | 20.99% |
| Non-ground proxy supported cells / whole grid | 22.64% |
| Pointwise RGB error, identity sampling | 0.10729 |
| Pointwise RGB error, projected sampling | 0.05575 |
| Median per-pair error reduction | 0.05063 |

All three preregistered pilot gates passed. Two projection overlays were also
visually inspected. These are geometry feasibility results, not generated-video
quality results. Non-ground is not a semantic building mask, and sparse depth
agreement cannot reliably exclude every slow-moving vehicle. Lookup uses the
measured point's displacement at its supported query cell, a local approximation
rather than dense surface reconstruction.

## B: adapter and four-GPU smoke

- `GeometryHistoryAttention`: condition-aware Query, projected local 3x3 history
  K/V, learned null key, fixed zero null value, bias-free zero-initialized out.
- Explicit initial fusion indices `2,12`; history dimension 64, four heads of
  dimension 32. Encoder and attention are the only trainable modules.
- No-history stays graph-connected inside the same DDP wrapper. Backbone stays
  eval; satellite dropout is explicit to cover CFG without dropping LiDAR/history.
- Original training objective retained, with decomposition logged. No unaligned
  temporal image loss, frequency split, long-term memory or raw AR initialization.
- First 20 steps exercise no-history on step 2 and satellite-unconditional on
  step 3. Require step-0 exact baseline equality, finite gradients and all
  trainable parameters in the graph. Encoder/Query gradients may be zero at
  zero initialization, but must become nonzero after output weights update.

## C: bounded multi-drive pilot and evaluation

`tools/train_kitti_geometry_history.py` uses all consecutive pairs from training
drives with epoch shuffling; entire drives are independently held out. Each
rank owns one fixed train probe and one held-out probe spread across the pair
list. This small set is a pilot diagnostic, not a full validation benchmark.

Probes cache history latent and pin current VAE sampling, timestep (250,750)
and diffusion noise. Compare disabled, correct, wrong geometry and wrong history.
Wrong geometry preserves target validity and the set of source coordinates.
Disabled loss must stay exactly constant across steps. Every rank/pair/timestep/
condition and loss component is written separately.

After smoke succeeds, budget at most 1000 total optimizer steps for the first
pilot, checking probes every 100 steps. Loss reduction on changing training
samples is not temporal-effectiveness evidence. Negative held-out results must
prevent escalation to longer training.

If held-out results support the method, evaluate a 32-frame generated-history
rollout with `tools/generate_kitti_geometry_history.py --first-rgb --uncond-cfg 3`.
Only the first observed RGB initializes history; later frames use generated
latents. Fresh current noise is used, and gaps/new drives reset. Evaluate
camera-consistent building motion, corresponding-surface appearance and
disocclusions, not merely smaller adjacent-frame LPIPS.

## Artifacts and resume contract

Training saves `run.json`, per-rank `metrics_rank*.jsonl`, `probes_rank*.jsonl`
and atomic `geometry_history_step_N.pt` checkpoints. Checkpoint mode is
`geometry_history_v1`, incompatible with legacy v2. Resume requires a fresh
output directory and matching base, manifest and adapter settings. It continues
optimizer state but is not exact mid-epoch sampler/RNG replay.

## B execution and connection interruption

- First four-GPU attempt (`stage_b_smoke`) reached step 1, then all ranks
  synchronously aborted on non-finite gradients during the forced no-history
  step 2. Preserve the failed log; it is not a successful smoke.
- Retried from fresh adapter initialization in `stage_b_smoke_scale1024` with
  AMP initial scale 1024 instead of 65536. The previous no-history failure did
  not recur, and CFG/no-satellite step 3 had nonzero encoder/Query/output grads.
- Latest received rank-0 log reached step 20: encoder gradient 0.00039974,
  condition Query 0.00005588, output 0.0108774. Fixed disabled probes at steps
  0/10/20 stayed exactly unchanged.
- At step 20, rank-0 held-out total-loss benefit is +0.00013188 at t=250 and
  -0.00009403 at t=750. Mixed, tiny effects are NOT evidence of temporal success.
- SSH multiplexed TCP connection stopped receiving data after these logs.
  Final checkpoint existence, other-rank completion and process exit could not
  yet be confirmed. Fresh SSH reaches the server but requires password login.
- **C has not started.** `tools/run_geometry_history_pilot.sh` checks A and all
  B smoke gates, prevents duplicate launch and restricts execution to idle
  physical GPUs 4–7. After connection recovery, verify B before invoking it.

Checks before connection recovery: 51 geometry/attention/training/rollout/summary
tests and 27 legacy v2/CFG tests passed locally. Pilot shell syntax and git diff
whitespace checks passed. Earlier server checks passed 61 tests.

## Connection recovered: B verified, C launched

After password-authenticated SSH was restored, the final smoke checkpoint was
verified on disk (7.1 MB) and GPUs 4–7 were idle. The run summarizer confirmed
all four ranks have every step 1–20, finite loss/gradient records, exact step-0
equality and invariant disabled probes (16 groups), no-history zero gradients
on step 2, and satellite-unconditional conditioning on step 3. All smoke gates
passed. The complete summary is `stage_b_smoke_scale1024/summary.json`.

All pending code updates were synchronized and the full targeted 78-test suite
passed on the server. The pilot launcher passed shell syntax validation.

C was launched detached from SSH (`nohup`, launcher PID 970021) using the guarded
`tools/run_geometry_history_pilot.sh`. It resumes step 20 and budgets 1000 total
optimizer steps, with a probe/checkpoint every 100 steps. Training partition:
10605 pairs; held-out partition: 3858 pairs from five separate drives. Per-rank
fixed probes still cover only four training and four validation pairs, not the
entire partitions. Only the adapter is trainable (exact parameter count is
recorded in `stage_c_pilot/run.json`). GPUs 0–3 remain reserved for the original CFG
training. C log: `stage_c_pilot.log`; artifacts: `stage_c_pilot/`.

The four-rank smoke's mean held-out benefit at step 20 was +0.00008006 (t=250)
and -0.00000973 (t=750). These small mixed effects do not establish temporal
effectiveness.

C startup is verified: all four ranks reached optimizer step 31. The resumed
step-20 fixed probes exactly reproduced the previous smoke's rank-0 values.
All ranks reported identical synchronized gradient norms at step 31: encoder
0.0009646243, condition Query 0.0001363657, output 0.0195291123. Trainable
parameter count is 608576; selected fusion indices 2 and 12 both have width 640.
The pilot is running detached; its 100-step held-out evaluation and final
temporal effectiveness results remain pending. Do not equate successful
training startup with improved generated-video consistency.

## D: post-bottleneck coupling control

Stage C at `stage_c_pilot/geometry_history_step_1000.pt` did not pass the RGB
effectiveness gate. Held-out RGB denoising stayed slightly worse than disabled
history; depth improved; correct history beat wrong history/geometry on RGB.
A later frozen-checkpoint gradient audit found encoder-local conflict at t=750
and zero depth gradient at fusion index 12, which sits after bottleneck depth
prediction in `openaimodel.py`. That audit is a snapshot: it does not prove
that removing depth loss would pass RGB vs disabled.

Code change (do not resume 2,12 weights):

- Geometry history defaults to `after_bottleneck`: the finest 640-d decoder
  fusion block, after `lidar_bottleneck_depth_head`.
- Explicit encoder/middle indices raise unless
  `--allow-pre-bottleneck-history` is set (2,12 control only).
- Local 3x3 geometry attention, null token, and frozen backbone are unchanged.
- New run directory: `stage_d_post_bottleneck`. Fresh adapter. Same 1000-step
  budget and the same eight fixed probes.
- Expected diagnostic if placement is correct: bottleneck depth metrics are
  identical across disabled/correct/wrong history. Judge RGB `loss_eps_base`
  only. Do not announce temporal success from total loss.

Launch: `tools/run_geometry_history_post_bottleneck.sh`. Do not start it while
GPUs 0–3 CFG training or a previous geometry job is using 4–7. This run answers
whether encoder-side depth coupling caused on>off; it is not itself a rollout.

## E: appearance transport through the correspondence gate

Stage D left geometry lookup correct but on/off videos nearly identical: the
zero-initialised attention residual was too weak to change pixels. Stage E
does not refine the projection. It adds a bilinear appearance skip of history
tokens at the projected cell, with a non-zero `to_skip` map. Invalid cells and
`has_history=False` remain exact zero. Attention `to_out` stays zero-init.

- Checkpoint mode: `geometry_history_v1_transport`. Do not resume Stage D
  `geometry_history_v1` weights.
- Placement remains `after_bottleneck`. Frozen backbone unchanged.
- Step 0 is no longer all-conditions-identical: correct history must differ
  from disabled. Disabled probes must stay bit-identical across steps.
- Judge RGB `loss_eps_base` and whether generated on/off videos diverge on
  corresponding surfaces. Depth must stay invariant across history conditions.
- New run directory: `stage_e_appearance_transport`. Fresh adapter. Same
  1000-step budget and the same eight fixed probes.

Launch: `tools/run_geometry_history_appearance_transport.sh`. This answers
whether transporting appearance along the already-open gate can change pixels;
it is not a claim that temporal consistency is solved.
