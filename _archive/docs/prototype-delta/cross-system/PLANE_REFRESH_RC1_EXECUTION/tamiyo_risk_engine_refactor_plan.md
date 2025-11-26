# Tamiyo Risk Engine Refactor (R4b)

## Context
- Scope: refactor `TamiyoService._apply_risk_engine` and dependent telemetry/conservative-mode plumbing to reach ≤ C complexity without relying on synthetic fallbacks.
- Baseline: current implementation spans ~240 lines with intertwined policy, blueprint, BSDS, and telemetry logic (see `src/esper/tamiyo/service.py:1359`). No dedicated unit harness exists; regression coverage relies on broad service tests.
- Guardrails: follow prototype strict-failure policy (no synthetic pause defaults), maintain telemetry parity, and use golden fixtures to guarantee behaviour stability.

## Phase Breakdown

### Phase 0 — Baseline & Safeguards
- Record current behaviour via deterministic fixture captures from `tests/tamiyo/test_service.py` (AdaptationCommand + telemetry event sequences).
- Snapshot complexity and lint posture in `lint_static_analysis.md` and note work start in `08_status_tracker.md`.
- Create scratch notes describing baseline inputs/outputs for future comparisons.
- Status (2025-09-27): `TAMIYO_ENABLE_COMPILE=0 PYTHONPATH=src pytest tests/tamiyo/test_service.py` passed; radon confirms `_apply_risk_engine` at F (92); status tracker updated to In Progress.

### Phase 1 — Test & Fixture Harness
- Add `tests/tamiyo/test_risk_engine.py` exercising `_apply_risk_engine` with canned SystemState, blueprint metadata, and training metrics.
- Store reference fixtures under `tests/fixtures/tamiyo_risk_engine/` to verify command mutations and telemetry events.
- Extend service integration tests to assert telemetry priority and conservative-mode toggling following risk evaluation.
- Status (2025-09-27): Capture script `scripts/capture_tamiyo_risk_fixture.py` generates seven scenarios; new fixtures + loader power `tests/tamiyo/test_risk_engine.py` and service priority assertions. Phase 1 testing complete pending doc updates.

### Phase 2 — Architecture & Data Model
- Introduce `RiskContext` dataclass encapsulating command snapshot, metrics, blueprint info, breaker state, and policy annotations.
- Define `RiskOutcome` (mutated command view, telemetry events, conservative-mode flag) as evaluator contract.
- Create ordered evaluator registry to ensure precedence matches current behaviour.
- Status (2025-09-27): Step 2.1 complete — `TamiyoRiskContext`/`TamiyoRiskOutcome` dataclasses landed with helper methods and unit coverage in `tests/tamiyo/test_risk_engine.py`.
- Status (2025-09-27): Step 2.2 complete — registry scaffold `_RISK_EVALUATOR_SEQUENCE` + `_build_risk_evaluators` in place with no-op placeholders; order enforced by `test_risk_evaluator_registry_order`.
- Status (2025-09-27): Step 2.3 complete — helper utilities for risk_reason, blueprint annotations, and optimizer adjustments introduced; `_apply_risk_engine` now references them without behavioural change, guarded by expanded unit tests.
- Status (2025-09-27): Step 2.4 complete — `_apply_risk_engine` now builds `TamiyoRiskContext`/`TamiyoRiskOutcome` and passes through the evaluator registry behind `_RISK_REFACTOR_ENABLED`; legacy path parity certified via flag-off tests.

### Phase 3 — Signal Evaluator Extraction
- Implement granular evaluators: policy risk, conservative mode, timeout handling, blueprint risk, BSDS payload, loss metrics, latency metrics, isolation/device pressure, optimizer hints.
- Ensure evaluators operate purely on `RiskContext`/`RiskOutcome`, eliminating direct `command.annotations.setdefault` usage.
- Status (2025-09-27): Steps 3A–3D landed previously; Step 3E is now complete — all evaluators own their legacy logic, `_RISK_REFACTOR_ENABLED` was deleted, and fixtures/lint/radon confirm parity (10/10 pylint, `_apply_risk_engine` complexity `A (3)`).

### Phase 4 — Orchestrator Refactor
- Rewrite `_apply_risk_engine` to construct context, invoke evaluators in order, aggregate outcomes, and finalise risk_reason annotations.
- Centralise annotation formatting and conservative-mode transitions to avoid duplicated logic.
- Status (2025-09-27): Steps 4.1–4.5 complete — `_evaluate` now delegates to helpers for policy prep, blueprint fetch, risk enforcement, metrics assembly, and telemetry finalisation. Complexity improved from **F (70)** to **A (1)** with Tamiyo service and integration suites green.

### Phase 5 — Conservative Mode & Breaker Coordination
- Audit `_set_conservative_mode` paths to emit telemetry exactly once per transition and clear conservative mode when breakers recover.
- Add evaluator covering recovery pathway (closed breakers + no timeouts).
- Status (2025-09-29): Completed. `_set_conservative_mode` now tracks last-enter metadata and suppresses duplicate enter events; exit telemetry carries previous reason/duration without breaking fixtures. New `conservative_recovery` evaluator clears conservative mode when breakers close with no timeouts, emitting a single `conservative_exited` event. Unit coverage added via `tests/tamiyo/test_risk_engine.py::test_set_conservative_mode_no_duplicate_events` and `::test_conservative_recovery_evaluator_clears_mode`.

### Phase 6 — Regression & Complexity Validation
- Run targeted pytest suite (`tests/tamiyo/test_risk_engine.py`, `tests/tamiyo/test_service.py`).
- Confirm `_apply_risk_engine` complexity ≤ C via `radon cc` and record results.
- Re-run pylint for Tamiyo modules to ensure no regressions.
- Status (2025-09-29): Completed alongside Phase 5. `pytest tests/tamiyo/test_risk_engine.py` and `pytest tests/tamiyo/test_service.py` remain green; `PYTHONPATH=. pytest tests/integration -k "not async_worker_soak"` confirms cross-system parity. `radon cc -s src/esper/tamiyo/service.py` keeps `_apply_risk_engine` at `A (3)` / `_evaluate` at `A (1)` and pylint still reports 10.00.

### Phase 7 — Documentation & Status Updates
- Update `TAMIYO_REVIEW_FINDINGS.md`, `lint_static_analysis.md`, `CHANGELOG_RC1.md`, and `08_status_tracker.md` with outcomes, test runs, and telemetry verification.
- Capture lessons learned in `KNOWLEDGE_DUMP.md` to inform R4c and downstream telemetry consumers.

## Acceptance Criteria
- `_apply_risk_engine` reduced to ≤ C complexity with evaluators ≤ B.
- New unit fixtures guarantee parity across critical decision branches.
- Telemetry priority, conservative-mode toggling, and annotations remain consistent (validated through tests).
- Documentation and status trackers reflect completed R4b work.
