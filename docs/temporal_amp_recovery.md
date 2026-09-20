# AMP recovery for the fresh GT-pair run

## Failure and checkpoint
Training stopped before logging step 12791. The latest complete checkpoint is step 12656 (epoch 7). All 549 trainable tensors and Adam state tensors in that checkpoint were finite. GradScaler scale was 4194304, growth tracker 656. The former loop raised on a nonfinite gradient norm before scaler recovery could execute. The original log alone does not distinguish element-level gradient overflow from norm-reduction overflow.

## Fix
- A nonfinite loss is synchronized across ranks and remains fatal.
- Gradient clipping first rejects a nonfinite norm without modifying gradients. FP64 norm reduction handles finite gradient elements whose FP32 norm overflows.
- True nonfinite gradients cause no optimizer update. All ranks back off AMP scale, reset growth tracking and retry the same batch/noise/RNG state, up to eight retries.
- Persistent nonfinite gradients remain fatal; invalid updates are never applied.
- Training metrics record amp_scale and amp_retries; metrics/amp_rank*.jsonl records recovery attempts.

## Validation
80 pre-existing temporal/persistent regression tests passed. Six new AMP tests passed on server Torch 1.13.1, including real GPU4/5 NCCL DDP, one-rank overflow, synchronized weights/scales, RNG restoration, finite huge-gradient clipping, and persistent-error rejection. Independent review found no remaining blockers.

## Resume
Resumed at 2026-09-19 14:01 from this run's step 12656 with the same data, learning rates, B4/card, GPUs4/5, and target36160 (20epochs). No old temporal-experiment weights used. Original partial metrics/log/args archived under amp_recovery/before; active metrics rewound to step12656, so replay does not double-count134 steps. Checkpoints remain latest two, each full epoch.

Reference: https://docs.pytorch.org/docs/stable/notes/amp_examples.html

## Actual failure replay verified
At step 12791 both ranks reproduced true nonfinite gradients. Both backed off scale 4194304 -> 2097152, retried the same batch once, and completed a finite optimizer update. Training continued beyond the former crash. Active metric step IDs are unique. See recovery_verified.json for observed progress and GPU state.
