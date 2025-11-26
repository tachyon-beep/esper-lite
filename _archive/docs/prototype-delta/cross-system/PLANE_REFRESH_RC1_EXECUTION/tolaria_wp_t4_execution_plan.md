# Tolaria WP-T4 Execution Plan

## Context & Goal
- **Objective**: Complete WP-T4 (Telemetry & Backcompat Cleanup) by reducing Tolaria trainer complexity to ≤C, ensuring telemetry packets include the refined metric set, and aligning docs/tests with the final architecture.
- **Scope**: `src/esper/tolaria/trainer.py`, supporting helpers (`_EpochRunner`, telemetry assembly), lint/static analysis docs, and observability guides.
- **Current State**: WP-T1–T3 closed; profiler/rollback telemetry in place. Complexities remain high (Radon F/E) for `_EpochRunner._optimizer_step`, `_EpochRunner._finalize_epoch`, `TolariaTrainer.__init__`, `TolariaTrainer.run`, `TolariaTrainer._emit_telemetry`.

## Phase 0 – Risk Reduction & Foundations
-### Step 0.1 – Baseline Capture
- Task 0.1.1: Snapshot current complexity (`radon cc -s src/esper/tolaria/trainer.py`) – current hotspots: `_EpochRunner._optimizer_step` F(59), `_EpochRunner._finalize_epoch` F(46), `TolariaTrainer.__init__` F(52), `TolariaTrainer.run` E(38), `_emit_telemetry` D(24). Archive output alongside existing pylint results.
- Task 0.1.2: Dump representative telemetry packets by running `TolariaTrainer` with profiler on/off and rollback failure scenarios (see `tests/tolaria/test_profiler.py`, `test_durability.py`) to capture metric/event names for golden references; note that `_emit_telemetry` drains `_seed_agg_metrics`, so capture before/after state.
  - _Captured 2025-09-29_: complexity snapshot stored at `baselines/wp_t4_radon_baseline.txt`; telemetry baselines (normal + failure paths + synthetic seed snapshot) recorded under `baselines/wp_t4_telemetry_baseline.json` and `baselines/wp_t4_phase0_failure_baselines.json`.

-### Step 0.2 – Test Harness Hardening
- Task 0.2.1: Confirm baseline suites (`tests/tolaria/test_tolaria_trainer.py`, `test_aggregation.py`, `test_profiler.py`, `test_durability.py`, `tests/integration/test_control_loop.py`) pass; add fixture utilities for harvesting telemetry/metric names in assertions.
- Task 0.2.2: Introduce golden-name assertions (e.g., `assert {m.name for m in packet.metrics} ⊇ expected_names`) so refactors can rely on automated validation instead of manual eyeballing.
- Task 0.2.3: Capture telemetry snapshots for representative scenarios (normal epoch, rollback deadline exceeded, profiler failure, seed share jump) to compare post-refactor outputs.
  - _Completed 2025-09-29_: pytest harness verified (`tests/tolaria/...` + control loop), and `test_tolaria_trainer_emits_state_packets` now loads the baseline metric set to enforce golden-name coverage. Failure-mode telemetry snapshots captured alongside the normal run baselines (see files above).

### Step 0.3 – Refactor Guardrails
- Task 0.3.1: Define commit boundaries per functional area (initialization, run-loop, telemetry, `_EpochRunner`) to keep reviewers focused and ease revert paths.
- Task 0.3.2: Catalogue touchpoints (Tamiyo/Kasmina clients, async worker submit API, emergency/rollback signals, telemetry builders) to ensure helper extraction preserves existing contracts.
- Task 0.3.3: Document high-risk couplings (async worker cancellation semantics, seed aggregation contexts, emergency controller hooks) so changes include targeted regression tests.
  - _Recorded 2025-09-29_: Commit slices mapped as (1) initialization/settings helpers (`TolariaTrainer.__init__` + config dataclasses), (2) run-loop orchestration (`run`, `_train_single_epoch`, failure handlers), (3) telemetry assembly (`_emit_telemetry`, seed metric helpers), (4) `_EpochRunner` optimizer/finalize logic. Primary integration touchpoints captured: Tamiyo sync calls (`_invoke_tamiyo*`), Kasmina command/export/finalize hooks, async worker submission (`_submit_async`, `AsyncWorker.submit/cancel`), rollback/fast-cache interfaces, and emergency signal publisher. High-risk couplings noted for cancellation deadlines vs shared async worker, conservative-mode transitions tied to breaker state, seed aggregation metrics feeding Kasmina/Nissa telemetry, and emergency controller shared-signal semantics—tests to cover rollback fast-path, async cancellation, seed-share warnings, and emergency dispatch remain mandatory during refactor.

## Phase 1 – Initialization & Dependency Extraction
### Step 1.1 – Async Worker & Settings Helpers
- Task 1.1.1: Extract async worker setup/shutdown (currently ~80 lines in `TolariaTrainer.__init__`) into `_configure_async_worker(settings, override)` to encapsulate concurrency/shutdown settings and ownership flag.
- Task 1.1.2: Encapsulate settings-derived defaults (timeouts, profiler flags, snapshot cadence, seed layer knobs) into dataclasses (`TrainerTimeoutConfig`, `SeedAggregationConfig`) to shrink the `try/except` clusters in `__init__`.
  - _Completed 2025-09-29_: Added `_configure_async_worker` helper plus `TrainerTimeoutConfig`/`SeedAggregationConfig` dataclasses; `TolariaTrainer.__init__` now delegates async worker creation and aggregation knob resolution to these helpers, trimming the sprawling `try/except` logic.

### Step 1.2 – Metric Map Bootstrap
- Task 1.2.1: Create `_initial_metrics(snapshot, settings)` returning the base metric map (timeouts, rollback, profiler, emergency, opt rebuild) and add unit coverage verifying keys.
- Task 1.2.2: Update `__init__` to call `_initial_metrics` and helper config objects, trimming side effects and nested `try/except` paths.
- Task 1.2.3: Ensure metrics include WP-T3 additions (`tolaria.rollback.failures_total`, `tolaria.profiler.*`) so later phases only extend, not redefine, metric handling.
  - _Completed 2025-09-29_: `_initial_metrics` now seeds the base metric map, including profiler counters; existing tests (`tests/tolaria/test_tolaria_trainer.py`) continue to assert the expected telemetry supersets using the golden baseline.

## Phase 2 – Per-Epoch Orchestration Simplification
### Step 2.1 – Run Loop Decomposition
- Task 2.1.1: Decompose `TolariaTrainer.run` (E/38) into `_run_epoch(epoch)` (core training), `_profile_epoch(epoch, fn)` (wraps `maybe_profile`), and `_finalize_epoch_stats(stats, epoch_start)` to isolate profiling/telemetry concerns.
- Task 2.1.2: Encapsulate failure handling (breaker updates, emergency escalation, rollback invocation) inside `_handle_epoch_failure(epoch, stats, reason)` returning structured outcomes for telemetry.
- Task 2.1.3: Maintain compatibility with conservative-mode transitions and `_seed_agg_metrics` updates by passing necessary context between helpers.
  - _Completed 2025-09-29_: Added `_run_epoch`, `_profile_epoch`, `_finalize_epoch_stats`, and `_handle_epoch_failure` helpers; `TolariaTrainer.run` now delegates profiler wrapping, duration tracking, and failure orchestration to these seams without changing existing telemetry or conservative-mode behavior.

### Step 2.2 – Rollback & Emergency Telemetry
- Task 2.2.1: Ensure `_handle_epoch_failure` increments metrics (`tolaria.rollback.failures_total`, `tolaria.rollback.deadline_exceeded_total`, emergency counters) and emits events (`tolaria.rollback.restore_failed`, `tolaria.emergency.*`) in one place.
- Task 2.2.2: Add targeted tests simulating (a) success path, (b) rollback fast-hit, (c) deadline exceed (async worker timeout), (d) emergency escalation, asserting telemetry/metrics captured.
  - _Completed 2025-09-29_: Failure handling helper centralizes rollback metrics/events; new unit tests cover fast-cache hits and deadline escalations with emergency L4 handoff to guarantee telemetry parity.

## Phase 3 – Telemetry Assembly Refactor
### Step 3.1 – Metric Aggregation Builder
- Task 3.1.1: Identify the ~60-line block in `_emit_telemetry` that builds static metrics/events; design `_build_basic_metrics(stats, state, hook_latency_ms)` returning reusable metric objects.
- Task 3.1.2: Extract per-seed/per-layer metric construction from `_finalize_epoch` into `_build_seed_metrics(ctx, exporter, per_layer_config)` so both `_finalize_epoch` and `_emit_telemetry` consume structured outputs.
- Task 3.1.3: Draft helper interface (e.g., `SeedMetricSet` dataclass) capturing share/alpha/conflict snapshots for telemetry emission and `seed_health` events.
  - _Completed 2025-09-29_: Added `_build_basic_metrics` and `SeedMetricSet` helpers; `_finalize_epoch` now delegates to `_build_seed_metrics` while caching structured seed metrics/events for `_emit_telemetry`.

### Step 3.2 – `_emit_telemetry` Cleanup
- Task 3.2.1: After helper extraction, replace inline metric assembly with calls to `_build_basic_metrics` + aggregated seed metrics; ensure event ordering (e.g., seed_health, grad_conflict_high, epoch_budget) remains consistent.
- Task 3.2.2: Expand telemetry tests to assert metric-name supersets and event presence for key scenarios (hook over budget, epoch over budget, rollback failure).
- Task 3.2.3: Document telemetry schema updates in runbook once helpers guarantee consistent naming.
  - _Completed 2025-09-29_: `_emit_telemetry` now consumes the helper outputs, resets the cached seed set per packet, and tests cover hook/epoch budget and rollback-failure events. Observability runbook updated to note the helper-based assembly.

## Phase 4 – `_EpochRunner` Streamlining
### Step 4.1 – Microbatch Handling
- Task 4.1.1: Map current `_optimizer_step` responsibilities—microbatch gradient aggregation, seed-level weighting, per-layer norm accumulation, conflict tracking—and sketch helper classes (`SeedGradientAccumulator`, `TeacherContributionTracker`).
- Task 4.1.2: Define helper APIs (e.g., `accumulator.add_microbatch(flat_grad, seed_masks)` → struct) to replace nested loops/try/except, ensuring compatibility with PCGrad/weights.
- Task 4.1.3: Assess how ctx fields (`seed_weight_sum`, `teacher_split_sum`, `per_layer_norm_sum`) mutate; plan to encapsulate state transitions inside helper objects to keep `_optimizer_step` orchestration-level only.
  - _Completed 2025-09-29_: Introduced `MicrobatchAccumulator` and `SeedAggregationTracker`; `_optimizer_step` now delegates gradient combination and context updates to these helpers while keeping telemetry/metrics consistent.

### Step 4.2 – Finalize Epoch Metrics
- Task 4.2.1: Review `_finalize_epoch` flows: seed metric averages, conflict events, seed health compact vs non-compact branches; strategize how to reuse Phase 3 helper outputs for telemetry and metric updates.
- Task 4.2.2: Identify invariants (e.g., `last_seed_share`, `seed_conflict_ratio_warn`) and plan state storage outside the main function to simplify loops.
- Task 4.2.3: Outline unit tests for extracted helpers—seed share delta, teacher splits, per-layer top‑K norms, conflict warnings—to validate behavior matches the current implementation.
  - _Completed 2025-09-29_: `_finalize_epoch` now relies on `_build_seed_metrics`/`SeedMetricSet`; per-seed metrics/events flow through the helper and `_emit_telemetry` consumes the cached set, preserving compact/non-compact behaviour.

## Phase 5 – Integration & Documentation
### Step 5.1 – Regression Suite
- Task 5.1.1: Run full Tolaria unit suite (`pytest tests/tolaria`) plus targeted integrations (`pytest tests/integration/test_control_loop.py`, `test_rollback_shared_signal.py`), capturing logs for sign-off.
- Task 5.1.2: Produce telemetry snapshots (JSON dumps of metrics/events) for key scenarios: normal epoch, rollback deadline, profiler failure, emergency escalation. Compare metric/event names against Phase 0 baselines.
- Task 5.1.3: Rerun `radon cc -s src/esper/tolaria/trainer.py` and document new scores to verify ≤C targets met.

### Step 5.2 – Documentation Updates
- Task 5.2.1: Update `docs/project/observability_runbook.md` with any telemetry schema or alert guidance changes introduced by the refactor.
- Task 5.2.2: Refresh `lint_static_analysis.md`, `02_wp_TOLARIA.md`, and `08_status_tracker.md` with final complexity scores, helper descriptions, and status notes; ensure `01_rc1_milestone_overview.md` reflects WP‑T4 completion.
- Task 5.2.3: Append changelog entry summarising WP‑T4 changes, including test commands and telemetry impacts.
- Task 5.2.4: Capture before/after metrics comparison (Radon report, telemetry snapshot) in PR notes for reviewer context.

## Exit Criteria
- **Complexity**: `radon cc` reports ≤C for `TolariaTrainer.__init__`, `TolariaTrainer.run`, `TolariaTrainer._emit_telemetry`, `_EpochRunner._optimizer_step`, and `_EpochRunner._finalize_epoch`.
- **Tests**: `pytest tests/tolaria`, `pytest tests/integration/test_control_loop.py`, `pytest tests/integration/test_rollback_shared_signal.py`, and any new helper-specific tests all pass.
- **Telemetry Validation**: Captured telemetry packets include required metrics/events with updated helpers; observability runbook documents any changes.
- **Documentation**: `02_wp_TOLARIA.md`, status tracker, changelog, observability runbook, and `lint_static_analysis.md` updated to reflect WP‑T4 completion.
- **Artifacts**: Attach or reference baseline vs post-refactor telemetry/complexity reports in the PR for reviewer confidence.

## Execution Order Summary
1. Phase 0 guardrails (baseline, tests, golden telemetry).
2. Phase 1 initialization helpers.
3. Phase 2 run-loop decomposition & failure orchestration.
4. Phase 3 telemetry assembly refactor.
5. Phase 4 `_EpochRunner` helpers.
6. Phase 5 regression/testing/documentation.

Following this order confines risk (telemetry compatibility, emergency handling) while driving complexity down incrementally.
