# Cleanup before the first temporal-branch commit

Scope: temporal/v22-causal-appearance worktree only. Preserve the active one-epoch
B/C full-data experiment, producer, model weights, all data/results, and the original
single-frame worktree. No model/optimizer/sampler behavior changes in this pass.

Backup: before_source.tar.gz, before.diff, before.json and before_status.txt in this
control directory preserve all 73 modified/untracked files from c0cd741.

Behavior lock: run all 190 temporal tests from a real test-runner file (the first
stdin invocation cannot host multiprocessing spawn). Keep current centered history,
paired color, CFG/sampling, zero/OFF fallback, geometric/reference correctness,
AMP retry, full epoch coverage and matched input tests.

Delete six independent retired experiment entrypoints, confirmed outside the active
AST import closure and without tracked-code users:
- tools/eval_adaptive_history_ablation.py
- tools/eval_adaptive_history_path.py
- tools/eval_dense_history_interventions.py
- tools/eval_temporal_clip.py
- tools/prepare_dense_static_reference.py
- tools/validate_temporal_pipeline.py

For clip retirement remove only its build_geo import and one clip-specific test
from tests/test_temporal_controls.py. Preserve the other five inference tests.
Update CLI smoke list to cover current full/probe/cache/preparation/summary entrypoints.
Preserve geometric diagnostic tools and tests: they remain useful, unlike superseded
training experiment launchers. Preserve ALL active dependency-closure source bytes.

Documentation: rewrite docs/temporal_history.md to identify centered A1 and current
B/C commands, add full B/C protocol, update cleanup plan, label former experiment
plans/results historical and point to current guide. Keep numerical evidence/results.
Do not promote training loss or successful runtime checks to appearance-quality claims.

Fallback inventory: single-frame/OFF, zero-initialized residual, empty-mask support,
CFG history duplication, AMP retry and explicit device remapping are grounded tested
behavior and retained. No new fallback or compatibility branch is introduced. Older
geometry/content/static modes are retained dependencies/checkpoint contracts; removal
would be a separate model/API change while training is running, so excluded here.

Verification: tests before/after; CLI imports, compile checks, diff --check; exact hash
comparison of the active source closure; active process progress and failures; independent
review of plan and final diff. If lint/type checker absent, report unavailable rather
than install dependencies. Commit only scoped source/docs/tests, with Lore trailers,
to the existing temporal branch. No push requested; do not push.
