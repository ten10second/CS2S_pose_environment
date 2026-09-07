
## Phase A multi-sample round (2026-09-08)

Setup: 48 train + 16 held-out consecutive pairs (same drive stretch), 1200
steps, lr 1e-4, dual GPU, teacher-forced GT history, pure denoising
objective. Probes: fixed-noise comparisons of correct / disabled / wrong
history (same-drive-far, other-drive) on the training frame and on held-out
pairs; per-component gradient L2 logging.

Result: train benefit grows (+0.05 -> +0.10) while held-out benefit is
negative and worsens (-0.002 -> -0.038); correct-vs-wrong discrimination on
held-out pairs stays at noise level (+-0.003). Checkpoints: hist_v2_step_
{400,800,1200}.pt; metrics: hist_v2_multisample_20260907/metrics.jsonl
(wrong-history sources and per-pair values recorded per entry).

Verdict: under the phase-A regime (48 pairs, 25.4M history params, 1200
steps) the history module memorizes the trained pairs instead of learning
generalizable history use; on unseen pairs the conditioning is actively
harmful and degrades with training. Go/no-go criterion 5 (correct history
must beat disabled) fails on held-out pairs — effectiveness claim paused.

Candidate next steps (decision pending): scale to the full 14.4k pair plan
with a short probe first; reduce injected capacity (bottleneck-only) with
regularization; or move to P4-B explicit temporal auxiliary loss. Do not
extend training on the 48-pair slice further.

## Fixed-probe re-evaluation (2026-09-08, post review)

Re-evaluated fresh/400/800/1200 checkpoints with all four review-mandated
probe fixes applied (tools/evaluate_kitti_temporal_v2.py): per-pair frozen
history latents, intercepted fixed timesteps (250 / 750), identical noise
per (pair, t, draw) across all four history conditions, full component
breakdown from DDPM.last_loss_metrics, per-row JSONL records
(temporal_v2_reeval.jsonl, 2048 rows: 4 ckpts x 32 pairs x 2 t x 2 draws
x 4 conditions).

Harness check: the fresh (untrained) reference shows all four conditions
bit-identical at every split and timestep.

Findings at fixed t (component means, val t=250, step_1200):

  condition            denoise  depth  point  total
  disabled              0.1889  0.0147 0.1738 0.3774
  correct               0.2351  0.0154 0.2256 0.4761
  wrong:other_drive     0.2320  0.0152 0.2218 0.4690
  wrong:same_drive_far  0.2272  0.0153 0.2165 0.4590

  1. History conditioning provides no benefit at ANY measured (split, t):
     at train t=250 (fixed) denoise is 0.1925 (correct) vs 0.1886
     (disabled) — slightly WORSE, not better. The training-loop probe's
     +0.10 train "benefit" does not survive fixed timesteps and fixed
     history latents; it was an artifact of uncontrolled timestep/latent
     sampling (as the review predicted).
  2. On held-out pairs the harm concentrates in the DENOISING term
     (+0.046 at t=250) and the point-region term (+0.052); the LiDAR depth
     term is negligible (+0.0007). Reviewer hypothesis 1 (depth-driven
     auxiliary optimization) is therefore ruled out as the main mechanism.
  3. Correct history is consistently MORE harmful than wrong history on
     held-out pairs — the learned corrections are scene-coupled to the 48
     trained pairs and mislead elsewhere. Attention focus grew (0.11 ->
     0.285) but what it reads does not transfer.

Decision (per the review's tree): this is not a training-objective problem
and not a training-duration problem. The condition-aware residual adapter
as configured learns scene-coupled corrections with no reject/generalize
behaviour. Pausing this architecture. Any continuation requires an explicit
mechanism change (correspondence constraint, capacity reduction to
bottleneck-only blocks, or a different history carrier), evaluated with the
fixed-probe protocol above before any long training.
