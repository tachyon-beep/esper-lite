## 2025-10-03 — WP-CS1 Performance Harness Execution
- **Summary**: Completed WP-CS1 Phase 3 by running the RC1 cross-system harness (`scripts/run_rc1_harness.py`) across Tolaria steady-train, rollback deadline, Tamiyo timeout, and Kasmina prefetch scenarios. Metrics confirm CPU baselines within RC1 envelopes (Tolaria latency mean 4.35 ms / p95 6.27 ms; rollback restore latency 12 ms with `deadline_exceeded_total = 1`; Tamiyo timeout counter increments once; Kasmina burst latency mean 40.7 ms / p95 41.1 ms). No fallback events were emitted; telemetry snapshots archived under `baselines/perf/wp_cs1_phase3/`. GPU baselines captured under `baselines/perf/wp_cs1_phase3_gpu/` (steady-train latency mean 41.6 ms / p95 135.8 ms; Kasmina burst 20.4 ms / p95 20.7 ms).
- **Tests Run**:
  - `PYTHONPATH=. pytest tests/scripts/test_run_rc1_harness.py`
  - `scripts/run_rc1_harness.py steady-train --device cpu --epochs 3 --batch-size 4 --disable-compile`
  - `scripts/run_rc1_harness.py rollback --device cpu --batch-size 4 --deadline-ms 5 --disable-compile`
  - `scripts/run_rc1_harness.py tamiyo-timeout --device cpu --epochs 3 --batch-size 4 --timeout-every 2 --disable-compile`
  - `scripts/run_rc1_harness.py kasmina-prefetch --device cpu --requests 64 --concurrency 8 --ready-latency-ms 40`
- **Telemetry Verification**: Tolaria packets show expected emergency events (`tolaria.emergency.escalated`, `tolaria.rollback.restore_failed`), timeout counters (`tolaria.timeout.tamiyo_total`, `tolaria.timeout.kasmina_total`), and zero `tolaria.graph_fallback` occurrences; Kasmina telemetry reports latency without isolation violations. Outputs captured in `docs/prototype-delta/cross-system/PLANE_REFRESH_RC1_EXECUTION/baselines/perf/wp_cs1_phase3/`.
- **Notes**: Provide `ESPER_LEYLINE_SECRET` when running in staging/production to avoid unsigned Kasmina command warnings. CI now executes a CPU quick sweep (steady-train, rollback, Tamiyo timeout, Kasmina prefetch) via `scripts/run_rc1_harness.py`; GPU suite remains manual until runners expose CUDA.

# RC1 Change Log

## Template Entry
- **Date**:
- **Module/Work Package**:
- **Summary**:
- **Tests Run**:
  - Unit:
  - Integration:
  - Performance:
- **Telemetry Verification**:
- **Notes**:

(append entries chronologically as work packages complete)

## 2025-09-26 — Shared Foundations
- **Summary**: Added the shared `AsyncWorker`, replaced Tolaria/Tamiyo timeout `ThreadPoolExecutor` usage, plumbed Kasmina prefetch through the common worker, shipped the strict dependency guard primitives (blueprint/training-run IDs), and delivered the soak harness plus developer script for cancellation stress runs.
- **Tests Run**:
  - Unit: `python -m py_compile src/esper/core/async_runner.py tests/helpers/async_worker_harness.py`
  - Integration: `RUN_SOAK_TESTS=1 python -m pytest tests/integration/test_async_worker_soak.py -m soak`
  - Performance: `python scripts/run_async_worker_soak.py --iterations 10 --jobs 128`
  - Targeted: `TAMIYO_ENABLE_COMPILE=0 pytest tests/tamiyo/test_service.py::test_evaluate_step_timeout_inference tests/tamiyo/test_service.py::test_evaluate_step_timeout_urza`
  - Targeted: `pytest tests/tolaria/test_tolaria_trainer.py::test_tolaria_handles_tamiyo_step_timeout`
  - Targeted: `pytest tests/integration/test_control_loop.py::test_control_loop_integration_round_trip`
- **Telemetry Verification**: N/A (harness-focused change).
- **Notes**: Weatherlight and the demo script now share a single `AsyncWorker` across Tamiyo, Tolaria, and Kasmina prefetch; soak harness remains gated behind `RUN_SOAK_TESTS` for opt-in execution.

## 2025-09-27 — Telemetry Routing Load Harness (Risk R3)
- **Summary**: Exposed emergency token-bucket/threshold settings via `EsperSettings`, instrumented Weatherlight publish counters, added Oona token metrics, and landed a repeatable load harness (integration test + CLI script) to exercise telemetry routing under burst conditions.
- **Tests Run**:
  - Targeted: `/home/john/esper-lite/.venv/bin/pytest tests/weatherlight/test_service_priority.py`
  - Targeted: `/home/john/esper-lite/.venv/bin/pytest tests/oona/test_emergency_burst_limit.py`
  - Integration: `/home/john/esper-lite/.venv/bin/pytest tests/integration/test_telemetry_emergency_load.py`
- **Telemetry Verification**: Harness asserts emergency queue depth, drop counters, and token bucket metrics via `OonaClient.metrics_snapshot`; Weatherlight counters expose high/critical publish counts with zero failures.
- **Notes**: Added `scripts/run_telemetry_routing_load.py` for manual load generation; new settings `OONA_EMERGENCY_MAX_PER_MIN`/`OONA_EMERGENCY_THRESHOLD` allow tuning without code changes.
  Manual run on staging Redis (`--count 120 --rate 20`) produced 120 routed CRITICAL packets with 0 drops; Oona metrics recorded `publish_latency_ms≈0.26`, `queue_depth_emergency=120`, breakers closed.

## 2025-09-27 — Confidence Gating Logits (Risk R5)
- **Summary**: Tamiyo now annotates seed commands with the policy-selected blend mode, confidence gate hyperparameters, and a `confidence_logits_required` flag; Kasmina persists the metadata, enforces the logits requirement, and emits telemetry when confidence mode falls back.
- **Tests Run**:
  - Targeted: `/home/john/esper-lite/.venv/bin/pytest tests/tamiyo/test_service.py::test_emits_blend_mode_annotations_when_enabled`
  - Targeted: `/home/john/esper-lite/.venv/bin/pytest tests/kasmina/test_blend_annotations.py`
- **Telemetry Verification**: Kasmina emits `confidence_gate_missing_logits` events when Tamiyo requests confidence mode without logits, ensuring operators see gating degradations; blend telemetry continues to include mode/source tags.
- **Notes**: Confidence mode now fails fast into telemetry when logits are absent instead of silently gating on activations; channel and residual modes unchanged.

## 2025-09-27 — Tolaria Epoch Runner Refactor (Risk R4a)
- **Summary**: `_EpochRunner` now drives the entire Tolaria training loop, replacing `_train_single_epoch_legacy` with helper-driven orchestration and locking behaviour against a refreshed golden fixture. Seed aggregation, optimizer fences, control-loop hand-offs, and epoch finalization all live in runner helpers to keep complexity in check.
- **Tests Run**:
  - Targeted: `/home/john/esper-lite/.venv/bin/pytest tests/tolaria/test_tolaria_trainer.py`
  - Fixture: `/home/john/esper-lite/.venv/bin/python scripts/capture_tolaria_epoch_fixture.py`
- **Telemetry Verification**: `tests/tolaria/test_tolaria_trainer.py::test_tolaria_epoch_fixture_parity` compares live state/telemetry snapshots to the fixture to detect behavioural drift.
- **Notes**: Legacy `_train_single_epoch_legacy` has been removed; docs and status tracker updated to reflect the parity guard. Remaining R4a work focuses on lint/complexity follow-ups.

## 2025-09-27 — Tamiyo Risk Engine Evaluator Cutover (Risk R4b Phase 3E)
- **Summary**: Completed the evaluator refactor for Tamiyo’s risk engine, removing the `_RISK_REFACTOR_ENABLED` feature flag and migrating the legacy `_apply_risk_engine` logic into dedicated evaluators (policy risk, conservative mode, timeouts, blueprint/BSDS, loss/latency/isolation/device, optimizer hints, stabilisation). Complexity drops from `F (92)` to `A (3)` while preserving telemetry and command annotations.
- **Tests Run**:
  - Unit: `/home/john/esper-lite/.venv/bin/pytest tests/tamiyo/test_risk_engine.py`
  - Service: `TAMIYO_ENABLE_COMPILE=0 /home/john/esper-lite/.venv/bin/pytest tests/tamiyo/test_service.py`
  - Static: `/home/john/esper-lite/.venv/bin/pylint --rcfile .codacy/tools-configs/pylint.rc src/esper/tamiyo/service.py`
  - Complexity: `/home/john/esper-lite/.venv/bin/radon cc -s -n A src/esper/tamiyo/service.py`
- **Telemetry Verification**: Fixture-backed tests compare command mutations and telemetry digests against the captured scenarios; conservative-mode enter/exit events observed with no ordering drift.
- **Notes**: `TamiyoRiskOutcome` now records `risk_reason` assignment precedence, ensuring helper parity. Remaining Tamiyo debt focuses on `_evaluate` (`F (70)`) and conservative-mode recovery telemetry (Phase 4/5 scope).

## 2025-09-27 — Tamiyo Evaluation Orchestrator Refactor (Risk R4b Phase 4)
- **Summary**: Reduced `TamiyoService._evaluate` to a high-level orchestration sequence backed by dedicated helpers for policy execution, blueprint lookup, risk enforcement, dependency checks, metric collection, and telemetry finalisation. Complexity improved from `F (70)` to `A (1)` without altering telemetry ordering or command annotations.
- **Tests Run**:
  - Unit: `/home/john/esper-lite/.venv/bin/pytest tests/tamiyo/test_risk_engine.py`
  - Service: `TAMIYO_ENABLE_COMPILE=0 /home/john/esper-lite/.venv/bin/pytest tests/tamiyo/test_service.py`
  - Complexity: `/home/john/esper-lite/.venv/bin/radon cc -s src/esper/tamiyo/service.py`
- **Telemetry Verification**: Fixture comparisons confirm `timeout_inference`, `bp_quarantine`, and coverage/degraded-input events retain ordering; priority indicator still reflects highest emitted level.
- **Notes**: `_prepare_training_metrics`, `_run_risk_engine_with_context`, and `_finalize_evaluation` now own the side effects that previously leaked through `_evaluate`. Sets the stage for Phase 5 conservative-mode cleanup and R4c coordination.

## 2025-09-28 — Kasmina Command Dispatcher Refactor (Risk R4c)
- **Summary**: Enabled the Kasmina command dispatcher by default, removed fallback kernel paths, enforced strict dependency guardrails (blueprint/training-run IDs), and upgraded telemetry to emit CRITICAL gate failures. Hardened `AsyncWorker.shutdown` to prevent Tolaria integration hangs and refreshed integration fixtures with deterministic signing and unique command IDs.
- **Tests Run**:
  - Unit: `/home/john/esper-lite/.venv/bin/pytest tests/kasmina -q`
  - Integration: `/home/john/esper-lite/.venv/bin/pytest tests/integration/test_control_loop.py -q`
  - Static: `/home/john/esper-lite/.venv/bin/radon cc -s src/esper/kasmina/seed_manager.py`
- **Telemetry Verification**: `tests/kasmina/test_seed_manager.py::test_handle_command_logs_tamiyo_annotations` confirms degraded-input/CRITICAL events; integration control loop asserts step-indexed per-seed packets. Gate failures surface as CRITICAL with reasons, and dispatcher telemetry flushes persist.
- **Notes**: WP-K1/WP-K2 complete; command verifier telemetry and prefetch/cache reliability remain tracked under WP-K3/K4. Async worker fix benefits Tolaria/Tamiyo test harnesses and is documented in shared foundations.

## 2025-09-29 — Kasmina Command Verifier Telemetry (Risk R4c / WP-K3)
- **Summary**: Elevated Kasmina command verifier failures to CRITICAL telemetry and added cumulative counters/latency metrics. Nonce ledger now enforces a bounded size with eviction accounting, publishes `kasmina.nonce_ledger.*` gauges, and emits warnings when the cap trims history. Administrative helpers (`reset_registry`, `reset_teacher_model`) clear registry + nonce ledger state with accompanying telemetry.
- **Tests Run**:
  - Unit: `/home/john/esper-lite/.venv/bin/pytest tests/kasmina/test_seed_manager.py --disable-warnings`
  - Unit: `/home/john/esper-lite/.venv/bin/pytest tests/kasmina --disable-warnings`
  - Integration: `/home/john/esper-lite/.venv/bin/pytest tests/integration/test_control_loop.py::test_kasmina_emits_one_packet_per_seed_per_step --disable-warnings`
- **Telemetry Verification**: `tests/kasmina/test_seed_manager.py::test_nonce_replay_emits_critical_and_metrics` confirms CRITICAL routing and verifier counters; `::test_nonce_ledger_truncation_emits_warning` asserts truncation warnings and ledger gauges. Reset helpers emit `registry_reset`/`teacher_deregistered` events with zeroed metrics.
- **Notes**: WP-K3 now covers telemetry, metrics, and registry hooks. Prefetch/cache work (WP-K4) remains outstanding and will extend ledger settings into async prefetch flows.

## 2025-09-28 — Kasmina Prefetch & Cache Reliability (Risk R4c / WP-K4)
- **Summary**: Completed WP-K4 with the async worker spawn fix (disables stale-claim polling on worker clones), stable integration coverage, and a reproducible benchmark harness. Weatherlight now runs Kasmina prefetch without cross-loop crashes, seed manager telemetry captures counters/latency, and observability docs carry alert thresholds.
- **Tests Run**:
  - Unit: `/home/john/esper-lite/.venv/bin/pytest tests/kasmina/test_seed_manager.py --disable-warnings`
  - Unit: `/home/john/esper-lite/.venv/bin/pytest tests/kasmina --disable-warnings`
  - Integration: `/home/john/esper-lite/.venv/bin/pytest tests/integration/test_kasmina_prefetch_async.py --disable-warnings`
  - Integration: `/home/john/esper-lite/.venv/bin/pytest tests/integration/test_control_loop.py::test_kasmina_emits_one_packet_per_seed_per_step --disable-warnings`
- **Telemetry / Benchmark Verification**: Prefetch metrics (`kasmina.prefetch.requests_total{status}`, `kasmina.prefetch.latency_ms`, `kasmina.cache.lock_wait_ms`) asserted in unit/integration suites. `scripts/bench_kasmina_prefetch.py --requests 300 --ready-latency-ms 40 --jitter-ms 8 --concurrency 6` reports 40.1 ms mean / 53.4 ms p95 (0 errors); observability runbook documents thresholds.
- **Notes**: WP-K4 closed; remaining Kasmina follow-ups move under future optimization/alerting tasks.

## 2025-09-28 — Tolaria WP-T1/T2 Aggregation & Emergency Hardening
- **Summary**: Completed WP-T1 gradient aggregation fixes (pairwise PCGrad, weighted broadcast safeguards) with new unit coverage, and advanced WP-T2 through timeout telemetry and emergency dispatch hardening. Tolaria now sources shared async worker settings from `EsperSettings`, emits `tolaria.timeout.*` and emergency metrics, and Weatherlight honours the same configuration knobs.
- **Tests Run**:
  - Unit: `pytest tests/tolaria/test_aggregation.py`
  - Unit: `pytest tests/tolaria/test_tolaria_trainer.py::test_tolaria_timeout_metrics_incremented`
  - Unit: `pytest tests/tolaria/test_tolaria_trainer.py::test_tolaria_emergency_dispatch_success tests/tolaria/test_tolaria_trainer.py::test_tolaria_emergency_dispatch_failure`
  - Integration: `pytest tests/tolaria/test_aggregation_attribution.py`
- **Telemetry Verification**: Timeout counters/latencies appear in telemetry packets; emergency dispatch metrics log successful publishes and CRITICAL failures with error context. Fixture parity tests assert the new metrics are a superset of the recorded baseline.
- **Notes**: Async worker defaults now include `ASYNC_WORKER_MAX_CONCURRENCY`, `ASYNC_WORKER_SHUTDOWN_TIMEOUT_S`, and optional Tolaria overrides; emergency controller resets after successful epochs. Remaining WP-T2 work covers telemetry summaries in observability docs and rollback/emergency integration tests.

## 2025-09-28 — Tamiyo WP-A3 Priority & Routing Wrap-Up
- **Summary**: Finalised WP-A3 by teaching Weatherlight to classify CRITICAL telemetry sources, exposing `weatherlight.emergency.telemetry_total` / `weatherlight.emergency.tamiyo_total`, and adding deterministic harnesses for Tamiyo emergency routing plus shared async-worker back-pressure. Observability docs and status reports now reference the new metrics and drill procedure.
- **Tests Run**:
  - Unit: `pytest tests/weatherlight/test_service_priority.py`
  - Integration: `PYTHONPATH=. pytest tests/integration -k "not async_worker_soak"`
  - Targeted: `pytest tests/integration/test_weatherlight_tamiyo_emergency.py`
  - Targeted: `pytest tests/integration/test_async_worker_backpressure.py`
- **Telemetry Verification**: Weatherlight counters confirm Tamiyo low-coverage events increment emergency telemetry totals; Oona metrics snapshot reports `emergency_published > 0` with `publish_dropped == 0`. Back-pressure harness asserts `publish_total` growth while `queue_depth_max` stays below the configured drop threshold.
- **Notes**: Observability runbook documents the drill, and Phase 3 knowledge/status entries are marked complete. Async worker soak remains opt-in via `RUN_SOAK_TESTS=1`.

## 2025-09-29 — Tamiyo WP-A4 Persistence Hardening
- **Summary**: Hardened Tamiyo field-report persistence: WAL appends now fsync directory entries, `_load_from_disk` surfaces anomalies (strict validation flag via `TAMIYO_WAL_STRICT_VALIDATION`), retry/observation sidecars are validated and telemetry now reports backlog/load-error metrics. Added `scripts/tamiyo_wal_backup.py` for backup/restore workflows and updated runbook documentation.
- **Tests Run**:
  - Unit: `pytest tests/tamiyo/test_persistence.py`
  - Unit: `pytest tests/tamiyo/test_service.py -k "retry_index or observation_window or publish_history_reports_sidecar_errors or strict_wal_validation"`
  - Script: `pytest tests/scripts/test_tamiyo_wal_backup.py`
  - Integration: `PYTHONPATH=. pytest tests/integration/test_weatherlight_tamiyo_emergency.py`
- **Telemetry Verification**: Summary telemetry now emits `tamiyo.field_reports.retry_index_load_errors`, `...window_load_errors`, and backlog metrics; warning events `field_report_sidecar_validation_warning` verified via the targeted publish-history test.
- **Notes**: WAL soak harness remains a TODO (manual drill recommended before production rollout). Strict validation defaults to off via settings flag for safe rollout.
## 2025-09-29 — Tolaria Rollback & Profiler Hardening (Phase 1)
- **Summary**: Improved Tolaria rollback durability—fast-cache replacements now maintain accurate byte accounting, snapshot loads prefer `torch.load(..., weights_only=True)` when available, and both fast/WAL paths raise/log telemetry (`tolaria.rollback.restore_failed`) on deserialization or optimizer failures. Full restores now execute via the shared `AsyncWorker`, cancelling on deadlines and surfacing `deadline_exceeded` telemetry; profiler traces emit timestamped filenames and broadcast warnings when export fails.
- **Tests Run**:
  - `PYTHONPATH=. pytest tests/tolaria/test_rollback_cache.py`
  - `PYTHONPATH=. pytest tests/tolaria/test_durability.py::test_rollback_failure_emits_event`
  - `PYTHONPATH=. pytest tests/tolaria/test_durability.py::test_rollback_uses_inferred_map_location`
  - `PYTHONPATH=. pytest tests/tolaria/test_profiler.py`
- **Telemetry Verification**: Recorded CRITICAL events for forced deserialization failures, observed `tolaria.rollback.failures_total` increments, and captured `tolaria.profiler.export_failed` warnings when trace export raised.
- **Notes**: Remaining WP-T3 work focuses on deeper profiler integrations and extended cross-process rollback signalling.

## 2025-09-29 — Tolaria Telemetry & Helper Refactor (WP-T4)
- **Summary**: Completed WP-T4 by decomposing `_optimizer_step`/`_finalize_epoch` into helper classes (`MicrobatchAccumulator`, `SeedAggregationTracker`, `SeedMetricSet`) and moving metric assembly into `_build_basic_metrics`. Telemetry packets now rely on structured helpers, rollback/emergency events remain intact, and regression baselines have been refreshed (`baselines/phase5_snapshots`).
- **Tests Run**:
  - `PYTHONPATH=. pytest tests/tolaria`
  - `PYTHONPATH=. pytest tests/integration/test_control_loop.py tests/integration/test_rollback_shared_signal.py`
  - Targeted: `PYTHONPATH=. pytest tests/tolaria/test_tolaria_trainer.py::test_telemetry_includes_budget_events`
  - Targeted: `PYTHONPATH=. pytest tests/tolaria/test_tolaria_trainer.py::test_telemetry_includes_rollback_failure_event`
- **Telemetry Verification**: Captured normal/profiler/rollback/emergency snapshots under `baselines/phase5_snapshots/`; metric names match the Phase 0 baseline (only addition is `tolaria.profiler.traces_failed_total`). Observability runbook updated to describe helper-driven metric assembly.
- **Notes**: Radon report after refactor shows legacy F functions retired; remaining F hotspots (`SeedAggregationTracker.combine`, `_build_seed_metrics`) are isolated helper seams tracked in `lint_static_analysis.md` for follow-up to reach ≤C.

## 2025-09-30 — Tolaria WP-T5 Telemetry Finalisation
- **Summary**: Completed WP-T5 by extracting seed telemetry builders into reusable snapshots, refactoring `_emit_telemetry` to consume the shared `SeedMetricSet`, and extending timeout/emergency coverage across unit and integration suites. Profiler exports now write both canonical and timestamped Chrome traces for review.
- **Tests Run**:
  - `PYTHONPATH=. pytest tests/tolaria`
  - `PYTHONPATH=. pytest tests/tamiyo/test_service.py`
  - `PYTHONPATH=. pytest tests/tamiyo/test_policy_gnn.py tests/tamiyo/test_risk_engine.py tests/tamiyo/test_graph_builder_capability_edges.py tests/tamiyo/test_registries.py tests/tamiyo/test_persistence.py`
  - `PYTHONPATH=. pytest tests/kasmina`
  - `PYTHONPATH=. pytest tests/core tests/leyline tests/karn tests/nissa tests/oona tests/quality tests/scripts tests/simic`
  - `PYTHONPATH=. pytest tests/integration`
  - Targeted: `pytest tests/integration/test_profiler_trace.py -vv`
  - Targeted: `pytest tests/tolaria/test_profiler.py -vv`
  - Targeted: `pytest tests/integration/test_control_loop.py`
- **Telemetry Verification**: Timeout events (`tolaria.tamiyo_timeout`, `tolaria.kasmina_timeout`) and rollback deadline simulations confirmed via new tests in `tests/tolaria/test_tolaria_trainer.py` and `tests/integration/test_control_loop.py`; emergency metrics `tolaria.emergency.halts_total` tracked during L4 drills. Phase baselines refreshed with `docs/.../baselines/t5_phase0/seed_epoch_snapshot.json`.
- **Notes**: Observability runbook and knowledge dump document helper-driven telemetry assembly and timeout/emergency expectations; lint snapshot records `_build_seed_metrics` at **A (2)** and `_emit_telemetry` at **C (15)**.

## 2025-10-01 — Tolaria WP-T5 Seed Helper Closure
- **Summary**: Finalised WP-T5 by re-running Tolaria regressions, verifying complexity targets via `radon`, and updating documentation/status trackers to record seed helper completion.
- **Tests Run**:
  - `PYTHONPATH=. pytest tests/tolaria`
  - `PYTHONPATH=. pytest tests/integration/test_control_loop.py tests/integration/test_rollback_shared_signal.py`
- **Telemetry Verification**: Compared regression telemetry against `baselines/phase5_snapshots/` (no metric deltas) and confirmed seed snapshot manifest unchanged.
- **Notes**: `02_wp_TOLARIA.md`, `08_status_tracker.md`, `lint_static_analysis.md`, and `KNOWLEDGE_DUMP.md` updated to mark WP-T5 closed; Radon reports `SeedAggregationTracker.combine` **A (2)** and `_EpochRunner._build_seed_metrics` **A (2)**.

## 2025-10-02 — Tolaria WP-99 (Phase 3 & 4) Closure
- **Summary**: Completed WP-99 by introducing pooled gradient flattening (`GradientBufferPool`), deferring seed/per-layer reductions via `SeedMetricsAccumulator`, and validating telemetry across default, GPU-prefetch, and no-compile permutations. Telemetry packets continue to emit the prior metric/event set while removing per-microbatch overhead.
- **Tests Run**:
  - `PYTHONPATH=. pytest tests/tolaria/test_tolaria_trainer.py`
  - `PYTHONPATH=. pytest tests/integration/test_control_loop.py`
  - `PYTHONPATH=. pytest tests/integration/test_rollback_shared_signal.py`
  - `PYTHONPATH=. python3 scripts/capture_perf_phase0_baselines.py --output-dir baselines/perf/wp99_phase3_flatten`
  - `PYTHONPATH=. python3 scripts/capture_perf_phase0_baselines.py --output-dir baselines/perf/wp99_phase3_seed_metrics`
  - `PYTHONPATH=. python3 scripts/capture_perf_phase0_baselines.py --output-dir baselines/perf/wp99_phase4_validation_default`
  - `PYTHONPATH=. python3 scripts/capture_perf_phase0_baselines.py --output-dir baselines/perf/wp99_phase4_validation_prefetch --enable-gpu-prefetch`
  - `PYTHONPATH=. python3 scripts/capture_perf_phase0_baselines.py --output-dir baselines/perf/wp99_phase4_validation_no_compile --no-compile`
- **Telemetry Verification**: Inspected seed metrics/telemetry packets against `baselines/perf/wp99_phase4_validation_summary.json`; confirmed seed share/teacher share/per-layer norms match prior baselines and timeout/rollback counters remain unchanged.
- **Notes**: Observability runbook now references the new accumulator flow and validation summary; `08_status_tracker.md` marks the performance harness as complete. Remaining Tolaria work: Phase 5 rollout notes and the follow-up eager-graph work under WP-100.

## 2025-10-02 — Tolaria WP-100 Eager Graph Instrumentation (Phases 1–3)
- **Summary**: Prepared the eager graph capture pathway by sequencing staging streams, disabling pin-memory during warm-up, adding `_stage_graph_microbatch`, and capturing metrics/events (`tolaria.graph.stage_copy_ms`, `…capture_ms`, `…replay_ms`). Graph capture still falls back (`AcceleratorError`), but telemetry now surfaces detailed reasons and benchmarks are repeatable via `scripts/run_graph_bench.py`.
- **Tests Run**:
  - `PYTHONPATH=. pytest tests/tolaria/test_tolaria_trainer.py -k graph`
  - `PYTHONPATH=. pytest tests/integration/test_control_loop.py`
  - `ENABLE_TOLARIA_GRAPHS=1 PYTHONPATH=. pytest tests/integration/test_control_loop.py`
  - `PYTHONPATH=. scripts/run_graph_bench.py --epochs 3 --warmup-batches 1 --output baselines/perf/wp100_graph_bench/graph_bench.json`
- **Telemetry Verification**: `baselines/perf/wp100_graph_phase0/` confirms fallback telemetry includes error + message; graph metrics remain zero until capture succeeds. Benchmark output stored in `baselines/perf/wp100_graph_bench/` for future comparison.
- **Notes**: Remaining WP-100 work focuses on eliminating the capture fallback (Phase 4/5). Observability runbook and knowledge dump updated with enablement guidance and rollback steps.

## 2025-10-02 — Kasmina WP-101 Phases 1–3 Progress
- **Summary**: Completed host graph integration for dynamic seeds (`_insert_seed_module`), optimizer/isolator wiring with fail-fast support, and validated blending/lifecycle telemetry end-to-end. Added Tolaria integration assertions for `kasmina.seed.alpha` and isolation metrics, updated observability runbook/knowledge dump, and captured reviewer snapshot (`baselines/perf/wp101_germination/`).
- **Tests Run**:
  - `PYTHONPATH=. pytest tests/kasmina`
  - `PYTHONPATH=. pytest tests/integration/test_control_loop.py`
  - `PYTHONPATH=. pytest tests/kasmina/test_seed_manager.py -k seed_modules_execute_in_order`
  - `PYTHONPATH=. scripts/run_graph_bench.py --epochs 3 --warmup-batches 2 --device cuda` (seed metrics snapshot)
- **Telemetry Verification**: Integration test asserts per-seed telemetry packets include `kasmina.seed.alpha` and isolation counters; runbook documents monitoring commands. Snapshot stored under `baselines/perf/wp101_germination/`.
- **Notes**: Remaining WP-101 work focuses on Phase 4 validation and Phase 5 rollout once performance harness comparisons and rollback SLA confirmations are recorded.

## 2025-10-03 — Tolaria WP-100 Phase 5 Alerting & Dashboards
- Finalized Phase 5 (alert updates + rollout): Prometheus thresholds tightened (stage 0.5 ms, capture warning 200 ms / critical 1000 ms), Grafana dashboard updated, and sandbox→staging→prod guidance documented. Graph pool reuse removes the 5 s slow-start bug (capture now ~63 ms).
- Shared CUDA graph pool reuse eliminates the 5 s first-capture stall; `TrainingLoopConfig.enable_graph_pool_reuse` defaults to on and pools clear on fallback. Baselines refreshed under `baselines/perf/wp100_phase5_prework/` (after_pool variants).
- Added Prometheus rule group `tolaria-graphs` (`infra/prometheus/alert_rules.yml`) covering stage-copy, capture, replay, and fallback counters; Prometheus config now loads the rule file.
- Extended Nissa observability dashboard (`infra/grafana/dashboards/nissa_overview.json`, panel 9) with a 1-minute stage-copy trend and thresholds aligned to the new alerts.
- Updated observability runbook with graph alert budgets and bench/diagnostic commands; captured fresh baselines under `baselines/perf/wp100_phase5_prework/`.
- Added CUDA-backed success test (`tests/tolaria/test_tolaria_trainer.py::test_graph_capture_success_metrics`) to guard the graph-enabled telemetry path.
- Remaining Phase 5 work: warm-up pool optimisation and rollout guidance before enabling graphs beyond prototype.

## 2025-10-03 — Kasmina WP-101 Phase 3 Integration
- Added integration test `tests/integration/test_control_loop_seed_metrics.py` to verify Tamiyo SEED commands drive Kasmina telemetry (`kasmina.seed.alpha`, `seed_stage`) with zero isolation violations.
- Updated observability guidance (seed metrics, rollback toggles) and referenced perf snapshots under `baselines/perf/wp101_germination/`.
- Knowledge dump/status tracker now mark Phase 3 complete; Phase 4 validation/rollout remains.

## 2025-10-03 — Kasmina WP-101 Phase 5 Rollout
- Production checklist recorded (observability runbook) for enabling/disabling dynamic seeds and monitoring `kasmina.seed.*` metrics.
- WP-101 status updated to complete; soak/perf baselines stored under `baselines/perf/wp101_germination`.
- No code changes required beyond tests; documentation and status artefacts refreshed.

