# Work Package — Tolaria Execution Remediation (WP-T)

## Context
Key findings from `TOLARIA_REVIEW_FINDINGS.md` and `lint_static_analysis.md`:
- PCGrad implementation (`combine_flat_grads`) projects only the first gradient; weighted aggregation and empty tensor handling need fixes.
- Legacy/compatibility artefacts: legacy flags in `__init__.py`, synthetic profiler paths, emergency level bypass states.
- Timeout handling uses per-call `ThreadPoolExecutor` (Tamiyo/Kasmina apply); cancellation ineffective.
- Rollback/profiler emit duplicate files, unsafe `torch.load`, missing failure telemetry.
- Complexity hotspots: `TolariaTrainer._train_single_epoch` (F/210), `__init__` (F/49), `run` (E/33), `_emit_telemetry` (D/24).

## Work Streams
| ID | Goal |
|----|------|
| WP-T1 | Gradient aggregation correctness & safety |
| WP-T2 | Async/timeout unification & emergency behaviour |
| WP-T3 | Rollback/profiler hardening |
| WP-T4 | Telemetry/backcompat cleanup & refactors |

### WP-T1 — Gradient Aggregation Correctness
Tasks:
1. Fix `combine_flat_grads` to implement full PCGrad projection with shuffle + pairwise projection.
2. Normalize weighted aggregation to broadcast across tensor ranks.
3. Handle empty gradient tensors on correct device/dtype; fail fast if no gradients.
4. Add unit tests covering multi-seed scenarios, GPU/CPU splits.
Acceptance:
- Tests verifying PCGrad conflicts and weighted sum pass.
- No device/dtype mismatches in AMP/GPU pipelines.
Risks & Mitigation:
- Regression in training speed: benchmark before/after, compare telemetry `tolaria.grad_agg.*`.
- Pairwise projection O(n^2): short-circuit when only single gradient.

### WP-T2 — Async Worker & Emergency Handling
Tasks:
1. Integrate shared async worker (from `05_shared_foundations.md`) for Tamiyo/Kasmina calls.
2. Remove per-call `ThreadPoolExecutor`; ensure cancellation interrupts running work.
3. Update emergency controller to log broadcast failures, use new worker for escalations.
4. Add timeout telemetry (success/failure events).
Acceptance:
- Simulated timeout cancels tasks; no leaked threads.
- Telemetry `timeout_inference` appears when Tamiyo worker fails.
Risks:
- Worker reuse may leak state; add integration tests hitting repeated timeouts.
- Need Tamiyo/Kasmina alignment to avoid inconsistent behaviour.

### WP-T3 — Rollback & Profiler Hardening
Tasks:
1. Fix FastRollback cache size accounting, GPU `torch.load(weights_only=True)` when available.
2. Ensure timeout threads are cancelled in `attempt_two_tier_rollback` (no blocked executor exit).
3. Profiler: add timestamp-based filenames, raise/log on export failure.
4. Add telemetry for rollback success/failure latency.
Acceptance:
- Rollback test suite verifying deadlines & cache size passes.
- Profiler generates unique files; failure surfaces in telemetry.
Risks:
- `weights_only` only in PT>=2.0; guard via try/except.
- Potential performance hit; run regression tests.

### WP-T4 — Telemetry & Backcompat Cleanup
Tasks:
1. Remove legacy exports, drop references to `old/01-tolaria.md`.
2. Ensure telemetry packets include new fields (e.g., command fail reasons).
3. Refactor `TolariaTrainer._train_single_epoch`, `__init__`, `run`, `_emit_telemetry` to reduce complexity to ≤ C.
4. Update docs & changelog; ensure lint/complexity doc updated.
Acceptance:
- Complexity report shows functions at ≤ C.
- Telemetry inspection shows blend/timeout metrics as expected.
Risks:
- Large refactor; break into incremental PRs.
- Need thorough unit tests and control-loop integration test before merge.

## Testing
- Unit: `tests/tolaria/test_aggregation.py` (new/updated), rollback tests, profiler tests.
- Integration: `tests/integration/test_control_loop.py` verifying new worker + telemetry.
- Performance: existing benchmark scripts; compare step latency.

## Rollback Plan
- Keep shared worker behind feature flag during initial integration; toggle off if regressions found.
- Retain previous aggregator implementation in branch for quick revert until validated.

## Telemetry Verification
- `tolaria.timeout.*` events recorded when new worker cancels.
- `tolaria.rollback.*` latency metrics emitted.
- Aggregation metrics show PCGrad conflict ratio consistent.

## Sign-off
- WP-T1..T4 tasks complete with updated tests & telemetry proofs.
- `lint_static_analysis.md` shows complexity reductions.
- `CHANGELOG_RC1.md` captures Tolaria updates.
