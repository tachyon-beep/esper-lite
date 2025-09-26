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
