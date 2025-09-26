# Tamiyo Risk Engine Refactor – Status Report (Phase 3)

## Current State (2025-09-27)
- Phase 3A–3E are complete. All legacy risk logic now lives in evaluator helpers and `_apply_risk_engine` delegates exclusively to them.
- The `_RISK_REFACTOR_ENABLED` feature flag has been removed; the evaluator pipeline is the only execution path.
- `TamiyoRiskOutcome` tracks the active `risk_reason`, enabling evaluators to reproduce the legacy precedence rules without local state.
- Complexity dropped from `F (92)` to `A (3)` for `_apply_risk_engine`; evaluators cap at `C (15)` (`_risk_evaluator_bsds`) with the rest `A/B`.

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
- Complexity: `.venv/bin/radon cc -s -n A src/esper/tamiyo/service.py` confirms `_apply_risk_engine` at `A (3)` with evaluator helpers <= `C (15)`.
- Fixture parity: existing golden fixtures exercised by `tests/tamiyo/test_risk_engine.py` (command + telemetry digest) all pass without diffs.

## Next Steps
1. Phase 4 — Orchestrator cleanup: continue reducing complexity in `_evaluate` (still `F (70)`), factoring timeout handling, metadata enrichment, and telemetry finalisation.
2. Phase 5 — Conservative-mode convergence: audit `_set_conservative_mode` callers to avoid duplicate enter/exit telemetry and introduce recovery evaluators.
3. Update risk register / milestone overview to mark R4b complete and queue R4c planning.

This report replaces the pre-cut-over snapshot and records the new baseline for subsequent phases.
