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

## Next Steps
1. Phase 5 — Conservative-mode convergence: audit `_set_conservative_mode` callers to avoid duplicate enter/exit telemetry and introduce recovery evaluators.
2. Expand docs/monitoring updates (Phase 7) and queue R4c planning once Tamiyo cleanup is fully signed off.

This report replaces the pre-cut-over snapshot and records the Phase 3/4 baseline for subsequent work.
