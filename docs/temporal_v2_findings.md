
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
