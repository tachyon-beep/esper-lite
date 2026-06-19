# Simic Producer Audit Findings

Audit date: 2026-06-13
Agent: Simic producer audit agent
Scope: `src/esper/simic/training/`, `src/esper/simic/agent/`, `src/esper/simic/rewards/`, `src/esper/simic/telemetry/`, plus Simic and reward telemetry tests.

## Source State

- Read-only audit only. No source or test files were edited.
- Live working-tree files were treated as current source state. Verification after writing this report showed only temp audit artifacts as untracked files; no source or test files were changed by this audit.
- Loomweave index was stale when checked, so Loomweave was used only for orientation; all evidence below is from live source reads.

## Feed Path Map

| Feed | Current producer path | Real vs placeholder assessment |
| --- | --- | --- |
| PPO update metrics | `VectorizedPPOTrainer.run` calls `PPOCoordinator.run_update` after rollouts (`src/esper/simic/training/vectorized_trainer.py:1633-1640`), `PPOAgent.update` builds metrics, and `VectorizedEmitter.on_ppo_update` emits `PPO_UPDATE_COMPLETED` (`src/esper/simic/telemetry/emitters.py:908-1058`). | Real for successful optimizer updates. Skipped finiteness-gate updates are not emitted before the trainer continues; see `SIMIC-PROD-004`. |
| Action decision and action outcome | Action sampling occurs in `VectorizedPPOTrainer.run` (`src/esper/simic/training/vectorized_trainer.py:1394-1428`), action execution emits last-action analytics through `VectorizedEmitter.on_last_action` (`src/esper/simic/training/action_execution.py:1274-1363`, `src/esper/simic/telemetry/emitters.py:327-369`). | Real for normal actions. Rollback actions emit governor and causal-log events but skip normal last-action analytics. |
| Reward components | Reward computation returns typed `RewardComponentsTelemetry` when requested (`src/esper/simic/rewards/rewards.py:58-184`); action execution stores raw reward and components before reward normalization (`src/esper/simic/training/action_execution.py:1060-1071`); last-action analytics carries the typed component object (`src/esper/simic/telemetry/emitters.py:337-343`). | Real for normal component-enabled reward paths. Component fields are explicit in `RewardComponentsTelemetry` (`src/esper/simic/rewards/reward_telemetry.py:32-148`). |
| Normalized-vs-raw semantics | Normal path stores raw reward in `env_state.episode_rewards` and `action_outcome.reward_raw`, then writes normalized reward to the PPO buffer (`src/esper/simic/training/action_execution.py:1060-1071`, `src/esper/simic/training/action_execution.py:1392-1435`). Rollback coordinator writes normalized penalty to the buffer and raw penalty to episode reward history (`src/esper/simic/training/ppo_coordinator.py:123-141`). | Mostly real and intentional: PPO consumes normalized reward; telemetry histories use raw reward. Rollback raw/normalized penalty details are not fully exposed in emitted telemetry; see `SIMIC-PROD-006`. |
| Episode outcome | Normal episode completion appends `EpisodeOutcome` and emits `EPISODE_OUTCOME` (`src/esper/simic/training/action_execution.py:1509-1580`). | Real for non-rollback episodes. Rollback episodes are explicitly skipped in the producer and no later corrected event is emitted; see `SIMIC-PROD-001`. |
| Gradient telemetry | `PPO_UPDATE_COMPLETED` includes gradient norm, pre-clip norm, per-head gradient norms/states, layer health, clipping split, and gradient CV (`src/esper/simic/telemetry/emitters.py:913-922`, `src/esper/simic/telemetry/emitters.py:959-999`, `src/esper/simic/telemetry/emitters.py:1013-1017`). | Real for successful updates. |
| LSTM health | `PPOAgent.update` computes update-time LSTM health (`src/esper/simic/agent/ppo_agent.py:823-836`) and `PPOUpdateMetricsBuilder` aggregates it (`src/esper/simic/agent/ppo_metrics.py:169-189`). Coordinator later recomputes from rollout hidden state and overwrites metric keys (`src/esper/simic/training/ppo_coordinator.py:378-396`). | Partly real, but emitted values currently describe the wrong source hidden state; see `SIMIC-PROD-003`. |
| Value metrics | `PPO_UPDATE_COMPLETED` emits value loss, explained variance, value distribution, TD/Bellman/return percentiles and variance (`src/esper/simic/telemetry/emitters.py:914-923`, `src/esper/simic/telemetry/emitters.py:947-956`, `src/esper/simic/telemetry/emitters.py:1036-1045`). | Real for successful updates. |
| Anomalies | `PPOCoordinator.run_anomaly_detection` builds ratio/value/numerical, gradient-drift, LSTM, and per-head entropy anomaly reports (`src/esper/simic/training/ppo_coordinator.py:350-422`) and delegates emission to `_emit_anomaly_diagnostics` (`src/esper/simic/training/vectorized.py:478-545`). | Real for generic anomaly events. Ratio diagnostics are produced upstream but not passed into the production emission call; see `SIMIC-PROD-002`. |
| Rollback penalties and attribution | Rollback action execution emits `GOVERNOR_ROLLBACK` and morphology causal-log events, then stores a terminal zero-reward transition (`src/esper/simic/training/action_execution.py:632-722`). Coordinator later injects normalized rollback penalty and attribution into the buffer (`src/esper/simic/training/ppo_coordinator.py:123-136`; `src/esper/simic/agent/rollout_buffer.py:757-811`). | PPO receives the normalized terminal penalty. The buffer attribution feed is implemented but not wired to emitted telemetry; see `SIMIC-PROD-006`. |
| Blueprint proof-baseline producers | Proof cohorts are declared supported and passed to the vectorized runner (`src/esper/simic/training/proof_baselines.py:31-97`, `src/esper/simic/training/proof_baselines.py:136-145`). Runtime action controls convert several policies to WAIT-only masks (`src/esper/simic/training/vectorized_trainer.py:94-117`). | `off_switch` and `static_initial` are plausibly real WAIT-only controls. `static_final` and `fixed_schedule` are placeholders despite being marked supported; see `SIMIC-PROD-005`. |

## Confirmed Issues

### SIMIC-PROD-001: Rollback episodes skip `EPISODE_OUTCOME` and no corrected outcome event is emitted

Priority: P1
Class: telemetry/proof correctness
Policy-learning impact: no direct PPO-learning bug; PPO still receives the corrected terminal penalty through the buffer. This is a telemetry and proof-data loss bug.

Failure mode:
Rollback episodes disappear from `EPISODE_OUTCOME` telemetry. The action layer comments that rollback episode outcome emission is skipped because a corrected outcome will be emitted later, but the coordinator only mutates in-memory `episode_outcomes` and `episode_history`. There is no later `EPISODE_OUTCOME` producer.

Evidence:
- Normal end-of-episode outcome is appended and conditionally emitted in `action_execution.py`; rollback episodes are explicitly excluded: `src/esper/simic/training/action_execution.py:1536-1580`.
- The skip comment says corrected emission will happen later: `src/esper/simic/training/action_execution.py:1552-1554`.
- Rollback correction recomputes raw penalty totals and replaces the latest `EpisodeOutcome`, but it does not emit telemetry: `src/esper/simic/training/ppo_coordinator.py:123-181`.
- Current references to `EPISODE_OUTCOME` are confined to the action execution producer path; coordinator tests assert only in-memory correction: `tests/simic/training/test_ppo_coordinator.py:122-157`.

Proposed tests:
- Add a rollback end-to-end producer test that executes a rollback episode with a capture hub and asserts exactly one corrected `EPISODE_OUTCOME` event is emitted with the post-penalty `episode_reward` and `stability_score`.
- Add a negative assertion that pre-penalty rollback outcomes are not emitted.

Tracker-ready row:
`P1 | bug | Simic rollback episodes do not emit corrected EPISODE_OUTCOME telemetry | Evidence: action_execution.py:1552-1554 skips rollback emission; ppo_coordinator.py:123-181 only corrects in memory | Acceptance: rollback episode emits one corrected EPISODE_OUTCOME carrying raw penalty-adjusted reward/stability.`

### SIMIC-PROD-002: Ratio diagnostics are produced and testable but dropped on the production anomaly path

Priority: P2
Class: anomaly telemetry correctness
Policy-learning impact: no direct policy-learning bug; this removes operator diagnostics for ratio explosion/collapse events.

Failure mode:
`PPOAgent.update` creates structured `ratio_diagnostic` data when ratio thresholds are breached. The anomaly emitter accepts and serializes that diagnostic. The production coordinator call never passes the metric through, so anomaly events carry `ratio_diagnostic=None`.

Evidence:
- `PPOAgent.update` appends `RatioExplosionDiagnostic.to_dict()` into `metrics["ratio_diagnostic"]`: `src/esper/simic/agent/ppo_agent.py:1249-1263`.
- `_emit_anomaly_diagnostics` accepts `ratio_diagnostic` and stores it on `AnomalyDetectedPayload`: `src/esper/simic/training/vectorized.py:478-535`.
- `PPOCoordinator.run_anomaly_detection` calls the emitter without a ratio diagnostic argument, leaving the default `None`: `src/esper/simic/training/ppo_coordinator.py:422-433`.
- Tests cover metric aggregation and direct emitter wiring, but not the coordinator production path: `tests/simic/test_vectorized.py:824-834`, `tests/simic/test_vectorized.py:1289-1308`.

Proposed tests:
- Unit-test `PPOCoordinator.run_anomaly_detection` with metrics containing `ratio_diagnostic` and a ratio anomaly, using a spy `emit_anomaly_diagnostics_fn`; assert the spy receives the exact diagnostic.
- Add an integration-style telemetry test that captures `RATIO_EXPLOSION_DETECTED` and asserts payload `ratio_diagnostic` is not `None`.

Tracker-ready row:
`P2 | bug | Pass PPO ratio_diagnostic through coordinator anomaly emission | Evidence: ppo_agent.py:1249-1263 produces it; vectorized.py:478-535 can emit it; ppo_coordinator.py:422-433 drops it | Acceptance: ratio anomaly payload includes worst-ratio diagnostic from PPO metrics.`

### SIMIC-PROD-003: PPO-update LSTM health is overwritten by rollout action-sampling hidden state

Priority: P2
Class: LSTM health telemetry correctness
Policy-learning impact: no direct optimizer bug; this can hide BPTT/update-time hidden-state corruption and emit misleading LSTM health.

Failure mode:
The PPO update path computes LSTM health from `result.hidden` during PPO update, then the coordinator recomputes health from `batched_lstm_hidden`, which is the action-sampling hidden state from rollout collection. The recomputed dictionary overwrites the existing `lstm_*` metric keys before `PPO_UPDATE_COMPLETED` emission.

Evidence:
- PPO update collects LSTM health from update result hidden: `src/esper/simic/agent/ppo_agent.py:823-836`.
- Metrics builder aggregates those update-time health values: `src/esper/simic/agent/ppo_metrics.py:169-189`.
- The trainer updates `batched_lstm_hidden` from action sampling before rollout execution: `src/esper/simic/training/vectorized_trainer.py:1394-1428`.
- The same rollout hidden state is passed into anomaly detection after the update: `src/esper/simic/training/vectorized_trainer.py:1668-1675`.
- Coordinator computes LSTM health from that passed hidden state and overwrites `metrics` with `metrics.update(lstm_health.to_dict())`: `src/esper/simic/training/ppo_coordinator.py:378-396`.
- The emitter reads LSTM health fields from the final `metrics` dictionary: `src/esper/simic/telemetry/emitters.py:1023-1035`.

Proposed tests:
- Unit-test `run_anomaly_detection` with metrics already containing sentinel `lstm_h_rms`/`lstm_has_nan` values and a different `batched_lstm_hidden`; assert update-time metrics are preserved or emitted under separate names.
- Add a regression test where update metrics report `lstm_has_nan=True` but rollout hidden state is healthy; anomaly detection must still emit/report update-time LSTM corruption.

Tracker-ready row:
`P2 | bug | Do not overwrite PPO update LSTM health with rollout hidden-state health | Evidence: ppo_agent.py:823-836 and ppo_metrics.py:169-189 produce update health; ppo_coordinator.py:378-396 overwrites it from batched_lstm_hidden | Acceptance: emitted PPO_UPDATE_COMPLETED distinguishes update LSTM health from rollout LSTM health and preserves update anomalies.`

### SIMIC-PROD-004: First and second finiteness-gate skipped PPO batches have no PPO or batch telemetry

Priority: P2
Class: skipped-update telemetry correctness
Policy-learning impact: training intentionally skips unsafe optimizer steps; the bug is that proof telemetry for degraded batches is suppressed until the third consecutive failure.

Failure mode:
When every PPO epoch is skipped by the finiteness gate but the consecutive streak is below three, `check_finiteness_gate` returns `should_continue=False`. The trainer immediately continues the outer loop before anomaly detection, before `on_ppo_update`, and before `on_batch_completed`. As a result, degraded batches do not produce `PPO_UPDATE_COMPLETED`, batch stats with `skipped_update=True`, or anomaly telemetry until the third consecutive failure raises.

Evidence:
- Metrics builder returns `ppo_update_performed=False`, `finiteness_gate_skip_count`, and failure details when no optimizer epoch completes: `src/esper/simic/agent/ppo_metrics.py:50-74`.
- Coordinator marks the run degraded and returns `False` for failures one and two; only the third emits a proof-blocking anomaly and raises: `src/esper/simic/training/ppo_coordinator.py:239-270`.
- Trainer continues immediately when `should_continue` is false: `src/esper/simic/training/vectorized_trainer.py:1649-1659`.
- `on_ppo_update` is gated behind `not update_skipped`, and `on_batch_completed` sits after the early `continue`: `src/esper/simic/training/vectorized_trainer.py:1696-1728`.
- Current coordinator tests cover KL skip and third-failure escalation, not producer emission for degraded first/second failures: `tests/simic/training/test_ppo_coordinator.py:81-105`, `tests/simic/training/test_ppo_coordinator.py:188-230`.

Proposed tests:
- Add a trainer-level or extracted-loop test for `metrics={"ppo_update_performed": False, "finiteness_gate_skip_count": 1, ...}` that asserts a skipped-update telemetry event or batch snapshot is emitted before continuing.
- Assert that emitted skipped-update telemetry includes `run_governor_signal`, `run_governor_status`, `run_governor_finiteness_streak`, and `finiteness_gate_failures`.

Tracker-ready row:
`P2 | bug | Emit degraded skipped-update telemetry for first/second PPO finiteness failures | Evidence: ppo_metrics.py:50-74 returns failure details; ppo_coordinator.py:239-270 returns should_continue=False; vectorized_trainer.py:1649-1659 continues before emitters | Acceptance: each finiteness-gate skipped batch emits auditable skipped-update/batch telemetry before the run continues or halts.`

### SIMIC-PROD-005: `static_final` and `fixed_schedule` proof baselines are placeholders but marked runner-supported

Priority: P1
Class: experimental validity / proof-baseline producer correctness
Policy-learning impact: this can invalidate comparison claims; it is not a PPO optimizer bug, but it changes what the baseline cohorts actually do.

Failure mode:
The blueprint-health proof plan marks all cohorts as `current_runner_supported=True`, including `static_final` and `fixed_schedule`. Runtime controls implement both `freeze_replayed_final_topology` and `apply_declared_lifecycle_schedule` as the same WAIT-only mask used for off-switch/static-initial controls. No replayed final topology or declared schedule is consumed. Tests currently lock this placeholder behavior in as expected.

Evidence:
- The proof plan declares `static_final` with `freeze_replayed_final_topology` and `fixed_schedule` with `apply_declared_lifecycle_schedule`, both marked supported: `src/esper/simic/training/proof_baselines.py:60-77`.
- Runner passes the lifecycle policy through as if executable: `src/esper/simic/training/proof_baselines.py:136-145`.
- Runtime action controls classify both policies as WAIT-only and replace the op mask with only WAIT enabled: `src/esper/simic/training/vectorized_trainer.py:105-117`.
- The parameterized proof-baseline test expects `freeze_replayed_final_topology` and `apply_declared_lifecycle_schedule` to force WAIT-only: `tests/simic/training/test_proof_baselines.py:85-110`.

Proposed tests:
- For `fixed_schedule`, provide a declared lifecycle schedule with at least one non-WAIT operation and assert the action controls follow that schedule at the specified epoch.
- For `static_final`, require replayed final topology input and fail closed when missing; assert the cohort cannot be marked supported without replay data.
- Change the existing WAIT-only parameterized test to cover only genuinely wait-only policies.

Tracker-ready row:
`P1 | bug | Mark unimplemented blueprint proof baselines unsupported or implement their real controls | Evidence: proof_baselines.py:60-77 marks static_final/fixed_schedule supported; vectorized_trainer.py:105-117 implements them as WAIT-only; tests lock this at test_proof_baselines.py:85-110 | Acceptance: static_final replays a final topology or fails unsupported; fixed_schedule follows a declared schedule or fails unsupported.`

### SIMIC-PROD-006: Rollout buffer rollback attribution metadata is implemented but not wired to emitted telemetry

Priority: P3
Class: rollback attribution telemetry gap
Policy-learning impact: PPO receives the normalized rollback penalty; this is a diagnostics/traceability gap.

Failure mode:
The rollout buffer stores rollback transition type, severity, triggering action id, watch-window evidence, and stable action ids. Coordinator writes those fields when injecting rollback penalty. `get_batched_sequences` exports them, and tests validate buffer roundtrip. The emitted rollback events do not include the normalized/raw punishment pair, triggering action id, or buffer attribution fields, so the producer feed exists but is not observable outside the buffer.

Evidence:
- Buffer defines rollback attribution fields and stable action ids: `src/esper/simic/agent/rollout_buffer.py:181-186`.
- Buffer exports the fields in batched sequences: `src/esper/simic/agent/rollout_buffer.py:680-693`.
- `mark_terminal_with_penalty` records penalty, severity, stable triggering action id, and watch-window evidence: `src/esper/simic/agent/rollout_buffer.py:757-811`.
- Coordinator calls `mark_terminal_with_penalty` with raw severity and triggering action id while writing normalized penalty to the buffer: `src/esper/simic/training/ppo_coordinator.py:123-136`.
- The rollback action event emits only env/device/reason/loss/consecutive panic fields, not penalty/action attribution: `src/esper/simic/training/action_execution.py:632-643`.
- Current buffer test verifies roundtrip only, not emitted telemetry consumption: `tests/simic/agent/test_rollout_buffer_unit.py:345-371`.

Proposed tests:
- Simulate a rollback and assert emitted telemetry contains stable triggering action id, raw penalty, normalized penalty, severity, and watch-window evidence in either `GOVERNOR_ROLLBACK`, `PPO_UPDATE_COMPLETED`, or a dedicated rollback-penalty event.
- Add a producer test that fails if rollback attribution is present only in the buffer batch and never emitted before the buffer is cleared.

Tracker-ready row:
`P3 | bug | Expose rollback attribution metadata in producer telemetry | Evidence: rollout_buffer.py:181-186,680-693,757-811 records it; ppo_coordinator.py:123-136 writes it; action_execution.py:632-643 omits it from emitted rollback event | Acceptance: rollback telemetry includes stable action id plus raw/normalized penalty and watch evidence.`

## Current Test Coverage Gaps

- No production-path test verifies that coordinator anomaly emission forwards `ratio_diagnostic`; existing tests cover aggregation and direct emitter invocation only.
- No trainer-level test verifies telemetry emission for first/second finiteness-gate skipped PPO batches.
- No producer test verifies corrected rollback `EPISODE_OUTCOME` emission.
- No producer test verifies rollback attribution metadata leaves the buffer and appears in emitted telemetry.
- Proof-baseline tests currently assert placeholder WAIT-only behavior for policies whose names imply replayed final topology or scheduled lifecycle control.

## Tracker-Ready Issue Rows

| Priority | Type | Title | Evidence | Acceptance |
| --- | --- | --- | --- | --- |
| P1 | bug | Rollback episodes do not emit corrected `EPISODE_OUTCOME` telemetry | `action_execution.py:1552-1554` skips rollback emission; `ppo_coordinator.py:123-181` only corrects in memory | Rollback episode emits exactly one corrected `EPISODE_OUTCOME` with raw penalty-adjusted reward/stability. |
| P2 | bug | Pass PPO `ratio_diagnostic` through coordinator anomaly emission | `ppo_agent.py:1249-1263` produces it; `vectorized.py:478-535` can emit it; `ppo_coordinator.py:422-433` drops it | Ratio anomaly payload includes worst-ratio diagnostic from PPO metrics. |
| P2 | bug | Preserve PPO update LSTM health instead of overwriting it with rollout hidden-state health | `ppo_agent.py:823-836` and `ppo_metrics.py:169-189` produce update health; `ppo_coordinator.py:378-396` overwrites it | `PPO_UPDATE_COMPLETED` distinguishes update LSTM health from rollout LSTM health and preserves update anomalies. |
| P2 | bug | Emit degraded skipped-update telemetry for first/second PPO finiteness failures | `ppo_metrics.py:50-74` returns failure details; `ppo_coordinator.py:239-270` returns early; `vectorized_trainer.py:1649-1659` continues before emitters | Each skipped finiteness-gate batch emits auditable skipped-update/batch telemetry before continuing or halting. |
| P1 | bug | Mark unimplemented blueprint proof baselines unsupported or implement real controls | `proof_baselines.py:60-77` marks `static_final`/`fixed_schedule` supported; `vectorized_trainer.py:105-117` makes them WAIT-only; `test_proof_baselines.py:85-110` locks this in | `static_final` replays final topology or fails unsupported; `fixed_schedule` follows declared schedule or fails unsupported. |
| P3 | bug | Expose rollback attribution metadata in producer telemetry | `rollout_buffer.py:181-186`, `rollout_buffer.py:680-693`, `rollout_buffer.py:757-811`, `ppo_coordinator.py:123-136`; emitted rollback payload at `action_execution.py:632-643` omits it | Rollback telemetry includes stable action id, raw/normalized penalty, severity, and watch-window evidence. |
