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
| WP-T5 | Seed aggregation helper simplification |

## Status Summary (2025-10-03)
- **WP-T1** — ✅ Complete. Pairwise PCGrad, weighted aggregation, and empty-gradient guards landed (`src/esper/tolaria/aggregation.py:77`); covered by updated aggregation tests (`tests/tolaria/test_aggregation.py:1`).
- **WP-T2** — ✅ Complete. Shared async worker adoption and timeout/emergency telemetry are live (`src/esper/tolaria/trainer.py:2174`, `docs/.../CHANGELOG_RC1.md:108`).
- **WP-T3** — ✅ Complete. Rollback/profiler hardening, telemetry, and cancellation tests shipped (`tests/tolaria/test_rollback_cache.py`, `tests/tolaria/test_durability.py`).
- **WP-T4** — ✅ Complete. Telemetry/backcompat refactor delivered; integration baselines captured in `baselines/phase5_snapshots/`.
- **WP-T5** — ✅ Complete. Seed aggregation helpers split out; Radon reports `SeedAggregationTracker.combine` **A (2)** and `_build_seed_metrics` **A (2)** with regression suites green (2025-10-01).

### WP-T1 — Gradient Aggregation Correctness
Tasks:
1. Fix `combine_flat_grads` to implement full PCGrad projection with shuffle + pairwise projection.
2. Normalize weighted aggregation to broadcast across tensor ranks.
3. Handle empty gradient tensors on correct device/dtype; fail fast if no gradients.
4. Add unit tests covering multi-seed scenarios, GPU/CPU splits.

#### WP-T1 Phased Workplan

**Phase 0 – Baseline & Guardrails**
- Step 0.1 – Snapshot current behaviour
  - Task: review `src/esper/tolaria/aggregation.py` and trainer call sites (micro/seed aggregation) for shape/device/dtype assumptions and conflict metrics.
  - Task: run the current aggregation attribution test suite (`pytest tests/tolaria/test_aggregation_attribution.py`) and record missing cases (multi-rank gradients, zero gradients, explicit PCGrad conflicts).
- Step 0.2 – Prepare failure surfaces
  - Task: define the error path or sentinel for "no gradients" so the trainer can fail fast instead of emitting CPU float64 tensors.
  - Task: document expected telemetry behaviour for `grad_conflict_rate` after the refactor so we can validate metrics parity.

**Phase 1 – PCGrad Algorithm Fix**
- Step 1.1 – Implement full pairwise projection
  - Task: shuffle/rotate incoming flat gradients and apply pairwise PCGrad projections so every gradient participates in conflict resolution.
  - Task: accumulate adjusted gradients (without in-place mutation) and count negative dot products.
- Step 1.2 – Stabilise numerical behaviour
  - Task: short-circuit when ≤1 gradient is present.
  - Task: introduce configurable epsilon handling and guard zero-norm vectors.

**Phase 2 – Weighted Aggregation & Broadcasting**
- Step 2.1 – Align weights with tensor metadata
  - Task: create the weight tensor on each gradient's device/dtype (upcasting only when safe) and normalise by the total weight, raising on zero/NaN.
- Step 2.2 – Broadcast across ranks
  - Task: reshape weights to broadcast across arbitrary gradient ranks, keeping a fast path for rank-1 tensors.

**Phase 3 – Empty/Partial Gradient Handling**
- Step 3.1 – Preserve dtype/device in `grads_to_flat`
  - Task: carry dtype/device metadata and raise a clear error when no gradients exist instead of returning an empty CPU tensor.
- Step 3.2 – Reconstruct faithfully in `flat_to_grads`
  - Task: validate offsets match total elements and rebuild tensors on the original device/dtype, including scalar gradients.
- Step 3.3 – Remove unused scaffolding
  - Task: drop the `AggregationResult` dataclass and tidy exports to match the new API.

**Phase 4 – Test Suite Extension**
- Step 4.1 – Add dedicated unit coverage
  - Task: extend `tests/tolaria/test_aggregation.py` with conflict scenarios, multi-gradient permutations, weighted aggregation on >1D tensors, and empty input failures.
- Step 4.2 – Extend trainer-level regression tests
  - Task: broaden `tests/tolaria/test_aggregation_attribution.py` (including AMP/GPU skip-aware cases) to assert `grad_conflict_rate` and device/dtype preservation.
- Step 4.3 – Optional property checks
  - Task: add lightweight property or randomized tests comparing the refactored aggregator with a reference implementation for small vectors.

**Phase 5 – Telemetry & Documentation**
- Step 5.1 – Update docs and status tracking
  - Task: record plan execution and findings in this document and `08_status_tracker.md`, update lint/changelog entries as work lands.
- Step 5.2 – Verification & sign-off
  - Task: document the test commands (`pytest …`) and telemetry snapshots demonstrating conflicts/weights behave as expected post-change.

**Status (2025-09-29)**
- Pairwise PCGrad shuffle + conflict counting implemented (`src/esper/tolaria/aggregation.py:151`) with weighted aggregation/device guards (`src/esper/tolaria/aggregation.py:124`).
- Empty gradient handling now raises `RuntimeError("no gradients available for aggregation")` (`src/esper/tolaria/aggregation.py:146`).
- Extended coverage in `tests/tolaria/test_aggregation.py:1` exercises conflict and weighting scenarios.
- Documented in `CHANGELOG_RC1.md:60` and status tracker updated.

##### Phase 0 status (2025-09-28)
- `combine_flat_grads` currently projects only the first gradient; conflict counts therefore under-report true collisions and the trainer assumes 1-D weight broadcasting.
- `grads_to_flat` returns an empty CPU float32 tensor when no gradients exist, which later violates device/dtype expectations; we will raise `RuntimeError("no gradients available for aggregation")` instead of returning an empty tensor.
- Existing test coverage is limited to `tests/tolaria/test_aggregation_attribution.py`; there is no direct PCGrad or multi-rank coverage, nor explicit zero-gradient handling.
- `grad_conflict_rate` telemetry today uses `conflicts / max(1, n-1)` and we will preserve that formula once the conflict counter reflects pairwise projections.

##### Phase 3 status (2025-09-28)
- `grads_to_flat` now enforces shared device/dtype and raises `RuntimeError("no gradients available for aggregation")` when the caller provides no tensors.
- `flat_to_grads` validates total element counts before reshaping and raises on mismatch, preventing silent tensor truncation.
- Removed the unused `AggregationResult` scaffolding and kept the module exports minimal.

##### Phase 4 status (2025-09-28)
- Added unit coverage (`tests/tolaria/test_aggregation.py`) for PCGrad conflicts, weighted aggregation broadcasting, invalid weights, mixed devices/dtypes, and empty gradient guards.
- Extended trainer attribution tests to assert `tolaria.grad_agg.conflict_ratio` telemetry and metrics snapshots when PCGrad engages (`tests/tolaria/test_aggregation_attribution.py::test_trainer_records_grad_conflict_rate`).
- Trainer close path in tests now uses a shortened async worker shutdown timeout to keep suites fast while exercising the real cleanup path.

##### Phase 5 status (2025-09-28)
- Verification commands run: `pytest tests/tolaria/test_aggregation.py tests/tolaria/test_aggregation_attribution.py -q` and targeted trainer conflict checks via `python -m esper.tolaria` helper script (see aggregation module snippet).
- Manual telemetry probe confirms `tolaria.grad_agg.conflict_ratio` reports `1.0` when the test harness stubs PCGrad conflicts, matching the metrics snapshot and telemetry packets.
- Ready for WP-T1 sign-off; remaining WP-T workstreams (T2–T4) now unblock without gradient aggregation debt.

##### WP-T2 Phase 0 status (2025-09-28)
- Async inventory: Tolaria already accepts an `async_worker` dependency and uses it for Tamiyo/Kasmina invocation (`_invoke_tamiyo_generic`, `_apply_kasmina_command`), but rollback still spins a per-call `ThreadPoolExecutor` (`rollback.py:209`) and Weatherlight owns the shared worker instance today (`service_runner.py:149`).
- Timeouts & telemetry: Tamiyo/Kasmina failures emit `tolaria.tamiyo_timeout` / `tolaria.kasmina_timeout` events with conservative-mode entry, yet no dedicated metrics exist; emergency escalations swallow broadcast exceptions in `EmergencyController.escalate`.
- Shutdown: trainer closes its worker with default 5s timeout; tests override the timeout to keep suites responsive, indicating we should make graceful shutdown configurable during WP-T2.
- Observability gaps: timeout events rely on `TelemetryEvent` warnings without structured metrics or priority propagation; telemetry packets currently set conflict metrics but lack `tolaria.timeout.*` counters for dashboards.

##### WP-T2 Phase 1 status (2025-09-28)
- Added `_submit_async` helper so Tamiyo/Kasmina calls route through the shared adapter with unified timeout handling and descriptive task names (`src/esper/tolaria/trainer.py`).
- Behaviour falls back to synchronous execution when no worker/timeout is provided, keeping legacy code paths intact while setting the stage for shared worker injection.
- Async worker defaults now come from `EsperSettings` (with Tolaria overrides), and Weatherlight honours the same settings when instantiating/shutting down its shared worker (`src/esper/core/config.py`, `.env.example`, `src/esper/weatherlight/service_runner.py`).

##### WP-T2 Phase 2 status (2025-09-28)
- Timeout telemetry now emits structured metrics (`tolaria.timeout.*`) for Tamiyo/Kasmina success and failure paths, with increments on timeout events and last-latency gauges exposed in telemetry packets (`src/esper/tolaria/trainer.py`).
- Added regression coverage to assert timeout counters and telemetry export (`tests/tolaria/test_tolaria_trainer.py::test_tolaria_timeout_metrics_incremented`).

##### WP-T2 Phase 3 status (2025-09-28)
- Emergency dispatch now runs through the shared async worker with configurable timeouts, recording latency/failure metrics and emitting CRITICAL telemetry on dispatch errors (`src/esper/tolaria/trainer.py`, `src/esper/core/config.py`, `.env.example`).
- `EmergencyController` reports the most recent escalation (including errors), Tolaria resets the controller on recovery, and new tests cover success/failure metrics for emergency dispatch (`tests/tolaria/test_tolaria_trainer.py::test_tolaria_emergency_dispatch_success` / `_failure`).

##### WP-T2 Phase 4/5 status (2025-09-28)
- Documentation and changelog updated with timeout/emergency telemetry changes and new configuration knobs; status tracker reflects WP-T2 progress through Phase 3.
- Verification commands: `pytest tests/tolaria/test_aggregation.py`, `pytest tests/tolaria/test_aggregation_attribution.py`, `pytest tests/tolaria/test_tolaria_trainer.py`.
- Additional metrics considered (e.g., rolling emergency latency averages, queue depth) but deferred until WP-T2 final sign-off once observability dashboards incorporate the new counters.

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

#### WP-T2 Phased Workplan

**Phase 0 – Baseline & Interface Survey**
- Step 0.1 – Async usage inventory
  - Task: map current Tolaria async entry points (`_invoke_tamiyo_*`, `_apply_kasmina_command`, rollback helpers) and document where AsyncWorker vs ThreadPoolExecutor is still referenced.
  - Task: confirm Weatherlight and service runners do not yet share the worker instance; capture current constructor signatures and shutdown flow.
- Step 0.2 – Timeout & telemetry audit
  - Task: record which telemetry packets/events fire on Tamiyo/Kasmina timeouts today and the metrics keys involved (`tolaria.tamiyo_timeout`, breaker transitions, etc.).
  - Task: identify missing emergency escalation logs (e.g., swallow exceptions in `EmergencyController.escalate`).

**Phase 1 – Shared Worker Integration**
- Step 1.1 – Worker lifecycle refactor
  - Task: introduce a shared worker handle (via `EsperSettings` or constructor injection) so Tolaria no longer instantiates its own AsyncWorker by default.
  - Task: add a thin adapter method (`_submit_async`) centralising worker submission, timeout, and cancellation behaviour.
- Step 1.2 – Settings & dependency wiring
  - Task: update Weatherlight/Tamiyo service runners to construct and pass the shared worker; ensure trainer factories accept the dependency.
  - Task: document new configuration knobs (max concurrency, graceful shutdown) and default them via `EsperSettings`.

**Phase 2 – Timeout Semantics & Telemetry**
- Step 2.1 – Unified timeout handling
  - Task: replace legacy timeout logic with the shared helper, ensuring Tamiyo/Kasmina timeouts emit consistent metrics/events.
  - Task: ensure cancellation propagates to long-running coroutines and resets breaker/conservative mode as defined in WP-T.
- Step 2.2 – Telemetry enrichment
  - Task: emit structured telemetry for success/timeout/cancellation (`tolaria.timeout.tamiyo`, `tolaria.timeout.kasmina`) with latency attributes.
  - Task: wire metrics into `metrics_snapshot` and telemetry packets so dashboards observe timeout rates.

**Phase 3 – Emergency Broadcast Hardening**
- Step 3.1 – Controller improvements
  - Task: update `EmergencyController.escalate` to capture broadcast exceptions, emit CRITICAL telemetry, and surface latency metrics.
  - Task: add state reset helpers (clear level/timestamps) aligned with shared memory semantics.
- Step 3.2 – Worker-backed dispatch
  - Task: dispatch emergency broadcasts via the shared worker (or a non-blocking pathway) so the training loop is not stalled by network IO.
  - Task: ensure fallback path logs explicit errors and increments failure counters.

**Phase 4 – Test & Simulation Coverage**
- Step 4.1 – Unit tests
  - Task: add tests for the new async submission helper (mock worker, timeout path) and emergency controller telemetry.
  - Task: cover successful vs timed-out Tamiyo/Kasmina calls, verifying metrics/events.
- Step 4.2 – Integration tests
  - Task: extend `tests/integration/test_control_loop.py` (or add a new slice) to simulate Tamiyo timeout and assert the training loop recovers without hangs.
  - Task: add regression test ensuring shared worker shutdown completes within the configured timeout.

**Phase 5 – Documentation & Sign-off**
- Step 5.1 – Update docs & tracker
  - Task: record outcomes in this document (Phase notes), bump `08_status_tracker.md`, and capture emergency/timeout behaviours in the observability runbook.
  - Task: update `CHANGELOG_RC1.md` and `lint_static_analysis.md` with async/telemetry changes.
- Step 5.2 – Verification log
  - Task: document test commands (unit + integration) and provide sample telemetry snippets showing timeout metrics and emergency logs.

**Status (2025-09-29)**
- Shared async worker submission helper and timeout counters live in `src/esper/tolaria/trainer.py:970` and telemetry output includes timeout metrics (`src/esper/tolaria/trainer.py:2174`).
- Weatherlight/Tamiyo share the worker (`src/esper/weatherlight/service_runner.py:1180`, `src/esper/tamiyo/service.py:320`).
- Tests executed: `tests/tolaria/test_tolaria_trainer.py:1`, `tests/integration/test_async_worker_backpressure.py:1`, `tests/integration/test_control_loop.py:1` (per `CHANGELOG_RC1.md:108`).
- Observability runbook records timeout/emergency telemetry (`docs/project/observability_runbook.md:110`).

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

- **Status (2025-09-29)**
- Fast rollback cache now handles snapshot replacement/eviction correctly, loads snapshots via `weights_only` when available, and raises on optimizer restore failures with telemetry/logging (`src/esper/tolaria/rollback.py:24`, `tests/tolaria/test_rollback_cache.py:1`).
- WAL restores use the shared loader with device guards and emit `tolaria.rollback.restore_failed` events plus failure metrics on corruption (`src/esper/tolaria/trainer.py:2790`, `tests/tolaria/test_durability.py:96`).
- Full restores now run through Tolaria's shared `AsyncWorker`, cancelling on deadlines and surfacing `deadline_exceeded` failures via telemetry (`src/esper/tolaria/rollback.py:214`, `tests/tolaria/test_rollback_cache.py:54`).
- Profiler traces include timestamped filenames and emit warning telemetry on export failure (`src/esper/tolaria/profiler.py:12`, `src/esper/tolaria/trainer.py:1689`, `tests/tolaria/test_profiler.py:72`).
- Shared rollback signalling validated via integration with Weatherlight (`tests/integration/test_rollback_shared_signal.py:1`).
- WP-T3 acceptance criteria met; follow-on telemetry/complexity work captured in WP-T4/WP-T5 (see sections below).

### WP-T4 — Telemetry & Backcompat Cleanup
Tasks:
1. Remove legacy exports, drop references to `old/01-tolaria.md`.
2. Ensure telemetry packets include new fields (e.g., command fail reasons).
3. Refactor `TolariaTrainer.run`, `_emit_telemetry`, `_optimizer_step`, `_finalize_epoch` to reduce complexity and centralise telemetry helpers.
4. Update docs & changelog; ensure lint/complexity doc updated.
Acceptance:
- Telemetry packets assembled via helper APIs; regression suite proves metric/event parity.
- Complexity hotspots isolated to the new helper seams (≤F) with follow-up tickets to drive to ≤C.
Risks:
- Large refactor; break into incremental PRs.
- Need thorough unit tests and control-loop integration test before merge.

- **Status (2025-09-30)**
- `_optimizer_step` now delegates to `MicrobatchAccumulator`/`SeedAggregationTracker`, `_finalize_epoch` produces a `SeedMetricSet`, and telemetry builder helpers drive `_emit_telemetry`.
- Regression coverage: Tolaria unit suite + integration (`tests/integration/test_control_loop.py`, `tests/tolaria/test_tolaria_trainer.py`) executed along with new budget/rollback assertions.
- Complexity snapshot (`radon cc`) shows legacy F functions retired; helper seams now grade at ≤C with WP‑T5 delivering the remaining reductions (`SeedAggregationTracker.combine` **A (2)**, `_build_seed_metrics` **A (2)**).
- Telemetry baselines refreshed under `baselines/phase5_snapshots/`; observability runbook documents helper-based metric/event assembly.

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

### WP-T5 — Seed Aggregation Helper Simplification
Tasks:
1. Split `SeedAggregationTracker.combine` into smaller helpers (teacher attribution, per-layer norm, weight accumulation).
2. Reduce `_build_seed_metrics` complexity by delegating compact vs non-compact emission.
3. Add targeted unit coverage for the new helper seams.
Acceptance:
- `SeedAggregationTracker.combine` and `_build_seed_metrics` achieve ≤C complexity.
- Regression suite proves telemetry parity.
Risks:
- Additional helper boundaries may require fixture updates; reuse existing telemetry baselines to guard behaviour.

#### Phase 1 Helper Plan (2025-09-30)
- `SeedGradientAccumulator`: helper to collapse `micro_flats` × `seed_masks` into per-seed vectors; returns consistent seed ordering and zero-fill for missing contributions.
- `TeacherAttributionMixer`: applies attribution weights and updates `EpochContext.teacher_overall_*` counters when the teacher seed participates.
- `SeedWeightResolver`: consumes exporter-derived `SeedState` data, computes stage/alpha-derived weights, and records alpha per seed.
- `SeedMetricUpdater`: centralises weight/norm/share/conflict bookkeeping against `EpochContext`, including last-share delta checks (`seed_share_sum`, `seed_conflicts_total`, `seed_weight_sum`).
- `PerLayerMetricUpdater`: iterates offsets/shapes to accumulate per-layer norms only when enabled.
- Each helper exposes a focused, unit-testable method; `SeedAggregationTracker.combine` orchestrates them and invokes `MicrobatchAccumulator` for final PCGrad merge.

##### Phase 1 Status (2025-09-30)
- Helper extraction landed inside `SeedAggregationTracker`; `combine` now delegates to accumulator, teacher attribution, weight resolution, and metric/per-layer updaters.
- Radon now reports `combine` **A (2)**, helper grades **A/B**, and `_update_seed_metrics` **C (15)**; `_EpochRunner._build_seed_metrics` remains **F (41)** for Phase 2.
- New tests (`tests/tolaria/test_tolaria_trainer.py::test_seed_aggregation_tracker_*`) cover weight/metric updates, teacher attribution flows, and per-layer norm aggregation using the new helper seams.

##### Phase 2.1 Status (2025-09-30)
- Added `EpochRunResult` wrapper so `_run_epoch` returns stats + telemetry context instead of relying on global attributes.
- `_profile_epoch` and `_finalize_epoch_stats` now accept/return the structured result, encapsulating profiler handling and duration stamping while preserving seed metric sets.
- `TolariaTrainer.run` consumes `EpochRunResult`, pulling hook latency directly from the helper output and seeding `_seed_metric_set`/`_seed_agg_metrics` via `_run_epoch`.

##### Phase 2.2 Status (2025-09-30)
- `_handle_epoch_failure` now funnels breaker decisions, conservative-mode transitions, rollback attempts, and emergency escalation through helper seams (`_evaluate_epoch_failure_inputs`, `_record_epoch_failure`, `_record_rollback_metrics`, `_maybe_trigger_deadline_emergency`). Complexity drops to **C (17)** with all supporting helpers ≤B.
- Rollback telemetry updates (`tolaria.rollback.fast_hits_total`, `fast_misses_total`, `deadline_exceeded_total`) and emergency counters live in one path; deadline escalations trigger L4 halts via `_maybe_trigger_deadline_emergency`.
- New unit coverage exercises success, fast-hit rollback, and deadline-escalation flows (`tests/tolaria/test_tolaria_trainer.py::test_handle_epoch_failure_*`), asserting metric deltas and emergency behaviour.

##### Phase 3.1 Status (2025-09-30)
- `_build_seed_metrics` decomposed into snapshot helpers (`_collect_seed_snapshots`, `_build_seed_snapshot`) with per-feature helpers covering share, alpha, conflict ratios, per-layer norms, and compact health events. Complexity now **A (2)**, with `_build_seed_snapshot` **A (3)**.
- Telemetry generation remains deterministic; `SeedMetricSet` assembly consumes the helper outputs so `_emit_telemetry` can reuse the structured data in Phase 3.2.
- Regression suites stay green post-refactor, confirming parity with the recorded telemetry baselines.

##### Phase 3.2 Status (2025-09-30)
- `_emit_telemetry` consumes the `SeedMetricSet` built during epoch finalisation, eliminating duplicate per-seed assembly and keeping complexity at **C (15)**.
- Static metrics remain in `_build_basic_metrics`, while seed events/metrics flow straight from the shared snapshots; cached `_seed_agg_metrics` are no longer required.
- Regression + integration sweeps (`pytest tests/tolaria/test_tolaria_trainer.py`, `tests/integration/test_control_loop.py`, `tests/integration/test_rollback_shared_signal.py`) confirm telemetry parity and emergency/rollback counters after the cleanup, with new telemetry-focused unit tests covering baseline metric supersets, hook budget warnings, and rollback failure events.

##### Phase 4.2 Status (2025-09-30)
- Integration coverage now exercises Tamiyo and Kasmina timeout paths end-to-end (`tests/integration/test_control_loop.py::test_control_loop_handles_*_timeout`), asserting control-loop recovery and telemetry counters.
- Rollback deadline simulation patched into the control-loop slice to validate L4 halt signalling and shared-worker shutdown behaviour under emergency escalation.
- Full integration suite passes (`PYTHONPATH=. pytest tests/integration`); timeout paths remain deterministic via stubs.

##### Phase 5 Status (2025-10-01)
- Regression sweeps re-run (`PYTHONPATH=. pytest tests/tolaria`, `pytest tests/integration/test_control_loop.py tests/integration/test_rollback_shared_signal.py`) confirming seed aggregation helpers keep telemetry parity; phase snapshots stored alongside WP-T4 baselines.
- Radon verification (`radon cc -s src/esper/tolaria/trainer.py`) now reports `SeedAggregationTracker.combine` **A (2)** and `_EpochRunner._build_seed_metrics` **A (2)**; results captured in `lint_static_analysis.md`.
- Documentation and status trackers updated (`02_wp_TOLARIA.md`, `08_status_tracker.md`, `CHANGELOG_RC1.md`, `KNOWLEDGE_DUMP.md`) to record WP-T5 closure.

## Sign-off Status
- WP-T1, WP-T2, and WP-T3 complete with updated tests & telemetry proofs (`CHANGELOG_RC1.md:60`, `:108`, `:126`).
- WP-T4 telemetry/backcompat refactor delivered; WP-T5 now closes out the helper complexity follow-up with ≤C targets achieved and telemetry parity proven.
- `lint_static_analysis.md` captures the updated complexity grades (Seed helpers ≤A, telemetry helpers ≤C) for Tolaria post-WP-T5.

##### WP-T2 Phase 4/5 status (2025-09-28)
- Added unit coverage for timeout and emergency metrics (`tests/tolaria/test_tolaria_trainer.py::test_tolaria_timeout_metrics_incremented`, `::test_tolaria_emergency_dispatch_success`, `::test_tolaria_emergency_dispatch_failure`).
- Integration coverage via `tests/tolaria/test_aggregation_attribution.py` validates shared worker shutdown and step telemetry; fixture-based parity tests assert new metrics remain a superset of recorded baselines.
- Documentation updated (`CHANGELOG_RC1.md`, `.env.example`) and status tracker now reflects WP-T2 Phase 3 completion with remaining observability follow-ups tracked separately.
- Additional telemetry (running emergency latency averages, queue depth) deferred to future observability workstreams; current metrics satisfy WP-T2 acceptance.

## Next Steps
- Tolaria-specific work packages are complete. Remaining RC1 scope is covered by WP-CS1 (cross-system QA & performance harness).
