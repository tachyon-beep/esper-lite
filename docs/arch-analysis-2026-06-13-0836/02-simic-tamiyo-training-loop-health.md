# Simic, Tamiyo, and Training Loop Health Deep Dive

Date: 2026-06-13
Scope: `src/esper/simic/`, `src/esper/tamiyo/`, shared action contracts in `src/esper/leyline/`, and the PPO training loop.
Method: source read, Loomweave hotspot orientation, repo guardrail execution, and focused inspection of current dirty worktree state. This report is analysis only; no source code was changed.

## Executive Summary

Simic and Tamiyo are not toy scaffolding anymore. The current code contains several strong recent repairs: op-conditioned critic consistency in rollout sampling, recurrent PPO hidden-state capture, denormalized value targets for GAE, forced-step masking, availability-based entropy floors, fresh-only counterfactual contribution targets, and fail-loud contribution reward checks.

The remaining health risk is structural. The training loop has outgrown its current shape: `VectorizedPPOTrainer.run()` and `execute_actions()` still own too many phases at once, while a tested lifecycle handler registry exists but is not the production path. This makes governor authority, reward timing, rollback semantics, telemetry, and buffer writes hard to reason about as one system.

The highest-priority model issues are:

1. Per-decision rollout entropy is still absent even though update-time entropy exists, leaving a gap in proof telemetry for action decisiveness and sparse-head collapse.
2. Missing or stale slot telemetry can still enter Tamiyo observations as healthy/default values or as host-drift fallbacks, despite fresh contribution targets now being handled better.
3. PPO finiteness handling is safer than before, but repeated non-finite epochs are expressed as skipped updates rather than a training-run governor failure.
4. The value head is collected in internal gradient telemetry but is not exposed in the PPO update event surface.
5. Documentation and architectural contracts disagree about Obs V3 dimensions, which is dangerous for checkpoint, dashboard, and proof consumers.

## Models Investigated

| Model / subsystem | Current health | Evidence | Key action |
| --- | --- | --- | --- |
| Tamiyo factored recurrent policy | Stronger than the surrounding loop. Sampling reuses the op used for Q(s, op), and canonicalizes irrelevant heads. | `src/esper/tamiyo/networks/factored_lstm.py:1005` to `factored_lstm.py:1055`; `factored_lstm.py:1058` to `factored_lstm.py:1165` | Add rollout entropy to `GetActionResult`; align forward/evaluate hidden-state handling. |
| Tamiyo Obs V3 features | Shape is compact and batched, but missing sensors still become benign values in several places. | `src/esper/tamiyo/policy/features.py:47` to `features.py:55`; `features.py:270` to `features.py:273`; `features.py:750` to `features.py:758` | Encode freshness/missingness as explicit features and fail or mask where required sensors are absent. |
| Simic PPO agent | Substantial correctness work is present: denormalized GAE/value metrics, forced-step masking, entropy floor, gradient-state tracking. | `src/esper/simic/agent/ppo_agent.py:502` to `ppo_agent.py:610`; `ppo_agent.py:919` to `ppo_agent.py:951`; `ppo_agent.py:1138` to `ppo_agent.py:1189` | Make finiteness failures run-visible; expose value-head state; reduce exception/telemetry complexity in the hot update path. |
| Simic rollout buffer | Correctly stores pre-step hidden state and now stores contribution targets only when fresh. | `src/esper/simic/agent/rollout_buffer.py:312` to `rollout_buffer.py:436`; `src/esper/simic/training/action_execution.py:1126` to `action_execution.py:1139` | Add tests that stale contribution targets cannot reach aux loss through any training entrypoint. |
| Simic action execution | Functionally central but architecturally overloaded. It performs reward, rollback, mutation, telemetry, and buffer insertion. | `src/esper/simic/training/action_execution.py:301`; `action_execution.py:765` to `action_execution.py:965`; `action_execution.py:1114` to `action_execution.py:1165` | Move production execution to the lifecycle handler registry or delete the unused handler track. |
| Vectorized PPO trainer | GPU-conscious in places, but the orchestration method remains too large and owns too many contracts. | `src/esper/simic/training/vectorized_trainer.py:1200` to `vectorized_trainer.py:1255`; `vectorized_trainer.py:1423` to `vectorized_trainer.py:1445`; `vectorized_trainer.py:1457` | Split into explicit phases: sense, decide, preflight, apply, reward, record, update. |
| Reward model | Contribution path is improving; simplified reward remains too small to prove economic behavior. | `src/esper/simic/rewards/contribution.py:1075` to `contribution.py:1085`; `contribution.py:1101` to `contribution.py:1127` | Add structural rent/capacity cost and preserve contribution freshness as a first-class reward contract. |
| Evaluation / A-B loop | Useful smoke infrastructure, not yet a causally strong evaluation rig. | `src/esper/simic/training/dual_ab.py:12` to `dual_ab.py:18`; `dual_ab.py:182` to `dual_ab.py:185` | Treat current dual A/B as sequential comparison; build lockstep seeded paired evaluation before claiming proof. |

## Findings

### P1. Rollout Decisions Still Do Not Capture Per-Head Entropy

`VectorizedPPOTrainer.run()` explicitly says per-head entropy is not available during action sampling and sets decision entropy to zero until `get_action()` returns it.

Evidence:

- `src/esper/simic/training/vectorized_trainer.py:1423` to `vectorized_trainer.py:1445`
- `src/esper/tamiyo/networks/factored_lstm.py:751` to `factored_lstm.py:800`
- `src/esper/tamiyo/networks/factored_lstm.py:1173` to `factored_lstm.py:1180`

The PPO update path does compute entropy for training diagnostics, so this is not an entropy-loss bug. It is a rollout/proof gap: the decision snapshot cannot show whether the live policy was decisive, collapsed, or forced at the moment action was taken.

Key action: add entropy fields to `GetActionResult`, compute them from the same masked/floored logits used for sampling, and wire them into decision telemetry without constructing `torch.distributions.Categorical` in the hot path.

### P1. Missing Slot Sensors Still Become Healthy Policy Inputs

The recent contribution-target change is good: stale counterfactual tracking no longer creates fresh aux targets. However, Obs V3 still has multiple places where missing data becomes benign data.

Evidence:

- `src/esper/tamiyo/policy/features.py:47` to `features.py:55` marks counterfactual freshness explicitly.
- `src/esper/tamiyo/policy/features.py:270` to `features.py:273` falls back from missing contribution telemetry to `improvement_since_stage_start`.
- `src/esper/tamiyo/policy/features.py:750` to `features.py:758` writes default contribution, freshness, and gradient health values.
- `src/esper/simic/training/vectorized_trainer.py:1238` to `vectorized_trainer.py:1255` syncs fallback telemetry for active seeds while leaving gradient fields at defaults.

The policy therefore cannot reliably distinguish "this seed is healthy" from "the sensor path did not produce a value" for all health fields. That is especially risky because Tamiyo is a controller over lifecycle interventions; missing health evidence should be observable as missingness, not converted into a healthy state.

Key action: reserve explicit missingness/freshness features for gradient health, contribution health, and telemetry sync source. For active slots, consider fail-loud behavior when a required sensor is absent outside known warmup windows.

### P1. Production Lifecycle Execution Bypasses the Tested Handler Registry

There is a lifecycle handler package with a registry and focused tests, but the live training path still calls the monolithic action executor.

Evidence:

- Production call: `src/esper/simic/training/vectorized_trainer.py:1457`
- Monolith entry: `src/esper/simic/training/action_execution.py:301`
- Handler registry: `src/esper/simic/training/handlers/registry.py:57`
- Search evidence: production references route through `execute_actions()`; handler references are package/tests.

This creates two semantic tracks for lifecycle behavior. The handler path has the shape the architecture wants, but the training loop still owns direct action mutation, reward, rollback, telemetry, and buffer writes in one broad procedure.

Key action: choose one production path. Prefer integrating the registry into `execute_actions()` as the authoritative operation dispatcher, then retiring duplicated direct branches after parity tests pass.

### P1. PPO Finiteness Gate Is Soft at the Run Level

The PPO agent records non-finite log-prob/value sources and skips the affected epoch. If every epoch is skipped, the metrics builder returns `ppo_update_performed=False` with NaN metric placeholders.

Evidence:

- Finiteness gate skip: `src/esper/simic/agent/ppo_agent.py:845` to `ppo_agent.py:891`
- Zero-completed update result: `src/esper/simic/agent/ppo_metrics.py:50` to `ppo_metrics.py:74`
- Contract note: `src/esper/simic/agent/types.py:48` to `types.py:53`

This is a good local safety mechanism, but it is not yet a training-run governor. A run can keep collecting rollouts while updates repeatedly fail unless the caller or dashboard interprets the status correctly.

Key action: promote repeated `ppo_update_performed=False` or any configured streak of finiteness failures into a run-level halt/rollback/escalation signal. Proof packets should treat such runs as invalid, not degraded.

### P2. Value-Head Gradient State Is Collected but Not Emitted

`PPOAgent.update()` collects gradient norm/state for the action heads and `value`, and `HeadGradientNorms` includes `value`. `emit_ppo_update_event()` serializes only the eight action-head gradient fields.

Evidence:

- Collection includes `value`: `src/esper/simic/agent/ppo_agent.py:1153` to `ppo_agent.py:1159`
- Metric type includes `value`: `src/esper/simic/agent/types.py:13` to `types.py:29`
- Event payload writes action heads only: `src/esper/simic/telemetry/emitters.py:968` to `emitters.py:997`

For an op-conditioned critic, value-head health is not secondary. If the critic head is missing, non-finite, or not learnable, advantage quality and Q(s, op) guidance are suspect even when action heads look fine.

Key action: add `head_value_grad_norm` and `head_value_gradient_state` to the PPO update payload and dashboard/proof consumers, or intentionally remove value from the collected head list if it is not meant to be reported there.

### P2. Hidden-State Contracts Diverge Between Forward and Evaluate Paths

The forward path documents that soft clamping was removed and leaves `lstm_ln` as identity, while `evaluate_actions()` still clamps returned cell state with a comment saying it matches forward.

Evidence:

- Forward path comment: `src/esper/tamiyo/networks/factored_lstm.py:670` to `factored_lstm.py:674`
- Evaluate path clamp: `src/esper/tamiyo/networks/factored_lstm.py:1247` to `factored_lstm.py:1250`

The clamp appears to affect the returned hidden state rather than current logits, so this is not necessarily an immediate PPO loss bug. It is still a recurrent-policy contract drift: rollout, forward, evaluation, and hidden-state telemetry should agree on what hidden state means.

Key action: make hidden-state normalization/clamping a single helper or remove the stale branch. Add a test that `get_action()` and `evaluate_actions()` advance hidden state under the same policy contract for equivalent single-step inputs.

### P2. Q(s, op) Telemetry Uses One Representative State

The PPO update computes op-conditioned Q telemetry from the first valid state in the batch.

Evidence:

- `src/esper/simic/agent/ppo_agent.py:678` to `ppo_agent.py:731`

This is useful as a cheap smoke signal, but `q_variance` and `q_spread` can be misleading if treated as batch health. A single state cannot represent policy conditioning across envs, episode phases, or slot saturation states.

Key action: either rename the fields to representative-state Q diagnostics or aggregate across a bounded stratified sample of valid states.

### P2. Simplified Reward Still Cannot Prove Structural Economy

`compute_simplified_reward()` is currently potential-based progress, action cost, terminal accuracy, and fossilize bonus. It does not charge structural rent, parameter cost, active slot cost, or compute footprint.

Evidence:

- `src/esper/simic/rewards/contribution.py:1101` to `contribution.py:1127`

That makes it insufficient for claims about economical morphogenesis. A policy can be shaped toward accuracy movement without learning that unused or overlarge capacity has an ongoing cost.

Key action: add explicit capacity rent and/or resource-normalized reward terms, then evaluate against paired baselines. Keep the existing contribution-freshness guardrails.

### P2. Current Dual A/B Is Sequential, Not Lockstep

The current `DualABTrainer` docstring states it trains A then B sequentially on the same device, not true parallel/lockstep comparison.

Evidence:

- `src/esper/simic/training/dual_ab.py:12` to `dual_ab.py:18`
- `src/esper/simic/training/dual_ab.py:182` to `dual_ab.py:185`

This is acceptable as an operational smoke harness, but not as a high-confidence causal evaluation rig for reward or blueprint claims.

Key action: build a seeded paired evaluation mode where the control and treatment consume equivalent task batches, initial seeds, and budget windows, then report confidence intervals over paired deltas.

### P2. Training Metrics Aggregation Still Uses Broad Generic Merging

`_aggregate_ppo_metrics()` uses generic key iteration, dictionary/list shape checks, and first-value behavior for some dictionaries.

Evidence:

- `src/esper/simic/training/vectorized.py:180` to `vectorized.py:233`

The guardrail currently allows these patterns, but typed proof metrics should not depend on generic merging rules. This is especially relevant for `head_gradient_states`, finiteness failures, ratio diagnostics, and value-function telemetry, where "first value" or "max list" may not be the desired aggregate.

Key action: replace generic aggregation with typed metric reducers. Each proof metric should declare whether it is max, min, sum, mean, any, all, first, or invalid-to-merge.

### P3. Obs V3 Dimension Contracts Have Documentation Drift

The README and current specs say non-blueprint Obs V3 is 116 dims with 128 total network input, while `ROADMAP.md` still describes 113 + 12 = 125.

Evidence:

- `README.md:34` to `README.md:36`
- `ROADMAP.md:266`
- `docs/specifications/simic.md:184`
- `docs/specifications/simic.md:231`

This is not a runtime bug by itself, but shape drift is dangerous for checkpoint compatibility, dashboard interpretation, and any external proof packet that records observation dimensions.

Key action: make Leyline constants the only dimension authority and update stale docs. Add a doc/test check that public shape claims match `get_feature_size()` plus blueprint embedding sizing.

## Positive Health Notes

- `get_action()` now reuses the sampled op for Q(s, op) in stochastic rollout mode and recomputes value for argmax op in deterministic mode. This fixes a class of biased-advantage failures.
- Causal masks and availability masks are centralized in `src/esper/leyline/causal_masks.py`, which is the right direction for factored-action credit assignment.
- PPO update metrics now distinguish completed optimizer steps from finiteness-skipped epochs.
- Fresh contribution targets are now constructed as a typed result and only stored when counterfactual baselines are current.
- The hot rollout path batches action tensor transfers reasonably: stacked actions/log-probs are copied to CPU once per step rather than per head/per env.
- `lint_gpu_sync.py` and `lint_defensive_patterns.py` both pass on the current tree, so the current exception load is at least registered with local governance.

## Priority Action Plan

1. Add rollout entropy to `GetActionResult` and decision telemetry. This is the fastest high-value proof improvement.
2. Introduce explicit missingness/freshness features for gradient health and contribution telemetry. Stop treating absent active-slot sensors as healthy.
3. Promote repeated PPO finiteness-gate failures to a run-level halt or invalid-proof status.
4. Wire value-head gradient norm/state through PPO telemetry, or remove it from collection if it is intentionally internal.
5. Integrate the lifecycle handler registry into the production action execution path.
6. Replace generic PPO metric aggregation with typed reducers.
7. Add capacity/resource rent to reward and evaluate with paired seeded comparisons.
8. Align Obs V3 dimension documentation to Leyline/runtime constants.

## Suggested Tracker Breakdown

- P1 task: rollout decision entropy for all factored heads, including canonicalized irrelevant heads.
- P1 task: active-slot sensor missingness in Obs V3 for gradient health and contribution health.
- P1 task: run-level PPO finiteness failure governor.
- P2 task: production lifecycle handler registry integration.
- P2 task: value-head gradient telemetry payload and dashboard/proof wiring.
- P2 task: typed PPO metric reducer module.
- P2 task: reward rent/resource-cost experiment.
- P3 task: Obs V3 shape documentation and doc-test guard.

## Verification Performed

- Loomweave orientation:
  - `project_status_get` reported the index as stale/dirty but useful for hotspots.
  - Coupling hotspots identified `VectorizedPPOTrainer.run`, `execute_actions`, `PPOAgent.update`, `compute_action_masks`, and `FactoredRecurrentActorCritic` paths for live source verification.
- Guardrails:
  - `uv run python scripts/lint_gpu_sync.py`
    - Checked 185 files, 113 sync points, 113 allowed, 0 violations, 0 stale whitelist entries.
  - `uv run python scripts/lint_defensive_patterns.py`
    - Checked 185 files in strict mode, 71 checked patterns, 71 allowed, 0 violations, 0 stale whitelist entries.
- Source verification:
  - Read current dirty-worktree source for Simic PPO update, rollout buffer, action execution, vectorized trainer, telemetry emitters, Tamiyo policy features, action masks, and recurrent policy network.

No training run or pytest suite was executed for this report; the artifact is a source-grounded architecture and model-health review.
