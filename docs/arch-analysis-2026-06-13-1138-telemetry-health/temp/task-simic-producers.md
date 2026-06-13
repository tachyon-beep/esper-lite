## Task: Simic Producer Audit

Write output to: `docs/arch-analysis-2026-06-13-1138-telemetry-health/temp/simic-producer-findings.md`

Read-only scope:

- `src/esper/simic/training/`
- `src/esper/simic/agent/`
- `src/esper/simic/rewards/`
- `src/esper/simic/telemetry/`
- related tests under `tests/simic/`, `tests/simic/telemetry/`, and `tests/integration/test_reward_telemetry_flow.py`

Goal:

- Audit PPO/training/reward/action telemetry emission, rollout buffer telemetry, reward components, gradient/LSTM/value/anomaly feeds, action outcome timing, normalized-vs-raw metric semantics, and feeds implemented but not wired.
- Pay special attention to current dirty files as live source state.

Required output:

- Feed paths for PPO update, action decision, reward components, episode outcome, gradient telemetry, LSTM health, value metrics, anomalies, and rollback penalties.
- Findings with failure mode and file/line evidence.
- Proposed tests that would catch each defect.

Constraints:

- Do not edit source.
- Distinguish policy-learning bugs from telemetry-only correctness bugs.

