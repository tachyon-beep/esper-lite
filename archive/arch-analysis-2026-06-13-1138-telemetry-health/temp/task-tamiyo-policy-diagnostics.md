## Task: Tamiyo Policy Diagnostics Audit

Write output to: `docs/arch-analysis-2026-06-13-1138-telemetry-health/temp/tamiyo-diagnostics-findings.md`

Read-only scope:

- `src/esper/tamiyo/`
- `src/esper/simic/training/vectorized_trainer.py` decision telemetry call sites
- relevant Leyline action/mask contracts
- tests under `tests/tamiyo/`, `tests/telemetry/test_decision_metrics.py`, and `tests/integration/test_q_values_telemetry.py`

Goal:

- Audit observation features, missingness/freshness handling, action masks, entropy diagnostics, Q-value/value-head telemetry, hidden-state diagnostics, and decision snapshots.
- Identify any healthy defaults, zero/sentinel values, stale values, or disconnected diagnostics.

Required output:

- Policy diagnostic feed inventory.
- Findings with file/line evidence.
- Proposed acceptance tests.

Constraints:

- Do not edit source.
- Do not recommend masking missing telemetry with fallback values.

