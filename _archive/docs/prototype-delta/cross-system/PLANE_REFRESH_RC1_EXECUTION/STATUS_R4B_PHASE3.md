# Tamiyo Risk Engine Refactor – Status Report (Phase 3 & 4)

## Current State (2025-09-27)
- Phase 3A–3E are complete. All legacy risk logic now lives in evaluator helpers and `_apply_risk_engine` delegates exclusively to them.
- The `_RISK_REFACTOR_ENABLED` feature flag has been removed; the evaluator pipeline is the only execution path.
- `TamiyoRiskOutcome` tracks the active `risk_reason`, enabling evaluators to reproduce the legacy precedence rules without local state.
- Complexity dropped from `F (92)` to `A (3)` for `_apply_risk_engine`; evaluators cap at `C (15)` (`_risk_evaluator_bsds`) with the rest `A/B`.

## Phase 4 Snapshot (2025-09-27)
- `_evaluate` now orchestrates the risk pipeline through dedicated helpers: policy preparation, blueprint resolution, risk enforcement, dependency checks, metric collection, and telemetry finalisation.
- New helpers (`_prepare_training_metrics`, `_run_risk_engine_with_context`, `_finalize_evaluation`, etc.) keep side effects isolated while preserving telemetry ordering and annotations.
- Complexity for `_evaluate` dropped from `F (70)` to `A (1)`; Tamiyo service and risk engine suites remain green under `TAMIYO_ENABLE_COMPILE=0`.

## Code Orientation
- `TamiyoService._build_risk_evaluators` now returns concrete callables:
  - `policy_risk`, `conservative_mode`, `timeouts`, `blueprint_risk`, `bsds`, `loss_metrics`, `latency_metrics`, `isolation_and_device`, `optimizer_hints`, and `stabilisation`.
- Each evaluator accepts a `TamiyoRiskContext` snapshot and mutates a shared `TamiyoRiskOutcome` (command, telemetry events, blueprint metadata, and reason).
- `_apply_risk_engine` constructs the context/outcome, iterates through the evaluators in the canonical order, then finalises the `risk_reason` annotation.
- `_set_conservative_mode` continues to emit enter/exit telemetry; evaluators surface the existing breaker/timeout transitions so downstream telemetry remains unchanged.

## Validation
- Unit: `.venv/bin/pytest tests/tamiyo/test_risk_engine.py`
- Service: `TAMIYO_ENABLE_COMPILE=0 .venv/bin/pytest tests/tamiyo/test_service.py`
- Static analysis: `.venv/bin/pylint --rcfile .codacy/tools-configs/pylint.rc src/esper/tamiyo/service.py` (10.00/10).
- Complexity: `.venv/bin/radon cc -s src/esper/tamiyo/service.py` confirms `_apply_risk_engine` at `A (3)` and `_evaluate` at `A (1)` after Phase 4.5.
- Fixture parity: existing golden fixtures exercised by `tests/tamiyo/test_risk_engine.py` (command + telemetry digest) all pass without diffs.
- Emergency telemetry routing: `pytest tests/weatherlight/test_service_priority.py` now verifies CRITICAL Tamiyo packets publish with explicit priority enums and update Weatherlight emergency counters (`weatherlight.emergency.telemetry_total` / `tamiyo_total`).
- Integration harness: `pytest tests/integration/test_weatherlight_tamiyo_emergency.py` drives a low-coverage Tamiyo flow through Weatherlight + Oona (FakeRedis) and confirms Oona emergency metrics (`emergency_published`, `publish_dropped`) and Weatherlight counters increment as expected. `tests/integration/test_async_worker_backpressure.py` exercises the shared async worker/Oona path and asserts back-pressure metrics remain within thresholds (`publish_total`, `publish_dropped`, `queue_depth_max`).

## Phase 5–7 Recap (2025-09-29)
- Conservative-mode coordination complete: `_set_conservative_mode` suppresses duplicate events, records enter/exit metadata, and the new `conservative_recovery` evaluator clears the mode when breakers recover. Unit/regression suites updated accordingly.
- Documentation refreshed (observability runbook, changelog, knowledge dump) alongside targeted + integration tests.

## WP-A4 Snapshot (Persistence & WAL) — Completed
- A4.1 baseline audit finished; A4.2 durability improvements landed (strict validation flag, WAL fsync, anomaly reporting).
- A4.3 sidecar validation + telemetry/backup tooling delivered.
- A4.4 validation tooling complete (backup/soak scripts, targeted Tamiyo subsets). A4.5 documentation/sign-off updates now merged into runbooks/changelog.

## Next Steps
1. Monitor soak harness automation in CI; consider extending to full service suite when runtime permits.
2. Prioritize WP-A4 follow-up items (e.g., WAL retention/backoff refinements) or begin planning the next Tamiyo work package.

This report reflects the current WP-A3/A4 state after Phase 5–7 and A4 completion.
