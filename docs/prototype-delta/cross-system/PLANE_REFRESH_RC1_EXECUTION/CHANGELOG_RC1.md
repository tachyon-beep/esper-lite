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

## 2025-09-29 — Kasmina Prefetch & Cache Reliability (Risk R4c / WP-K4)
- **Summary**: Prefetch coordinator now runs on the shared `AsyncWorker`, ensuring cancellable publish/consume loops with failure surfacing. Seed manager tracks per-status prefetch counters/latency, enforces timeouts, and emits CRITICAL `prefetch_timeout` telemetry. Kernel attachment uses per-blueprint locks with contention telemetry, and administrative resets cancel outstanding requests cleanly.
- **Tests Run**:
  - Unit: `/home/john/esper-lite/.venv/bin/pytest tests/kasmina/test_seed_manager.py --disable-warnings`
  - Unit: `/home/john/esper-lite/.venv/bin/pytest tests/kasmina --disable-warnings`
  - Integration: `/home/john/esper-lite/.venv/bin/pytest tests/integration/test_control_loop.py::test_kasmina_emits_one_packet_per_seed_per_step --disable-warnings`
- **Telemetry Verification**: `tests/kasmina/test_seed_manager.py::test_nonce_replay_emits_critical_and_metrics` (latency + counters), `::test_nonce_ledger_truncation_emits_warning` (inflight + truncations), and new prefetch tests assert `prefetch_timeout`, `prefetch_canceled`, and cache lock contention behaviour. Average and last-latency gauges now appear in Kasmina global packets.
- **Notes**: Outstanding follow-up: benchmark prefetch throughput under contention and integrate metrics into observability runbooks. WP-K4 remains open for performance validation but functional reliability changes (async worker clients, telemetry, locking) are in place.

## 2025-09-28 — Tolaria WP-T1/T2 Aggregation & Emergency Hardening
- **Summary**: Completed WP-T1 gradient aggregation fixes (pairwise PCGrad, weighted broadcast safeguards) with new unit coverage, and advanced WP-T2 through timeout telemetry and emergency dispatch hardening. Tolaria now sources shared async worker settings from `EsperSettings`, emits `tolaria.timeout.*` and emergency metrics, and Weatherlight honours the same configuration knobs.
- **Tests Run**:
  - Unit: `pytest tests/tolaria/test_aggregation.py`
  - Unit: `pytest tests/tolaria/test_tolaria_trainer.py::test_tolaria_timeout_metrics_incremented`
  - Unit: `pytest tests/tolaria/test_tolaria_trainer.py::test_tolaria_emergency_dispatch_success tests/tolaria/test_tolaria_trainer.py::test_tolaria_emergency_dispatch_failure`
  - Integration: `pytest tests/tolaria/test_aggregation_attribution.py`
- **Telemetry Verification**: Timeout counters/latencies appear in telemetry packets; emergency dispatch metrics log successful publishes and CRITICAL failures with error context. Fixture parity tests assert the new metrics are a superset of the recorded baseline.
- **Notes**: Async worker defaults now include `ASYNC_WORKER_MAX_CONCURRENCY`, `ASYNC_WORKER_SHUTDOWN_TIMEOUT_S`, and optional Tolaria overrides; emergency controller resets after successful epochs. Remaining WP-T2 work covers telemetry summaries in observability docs and rollback/emergency integration tests.
