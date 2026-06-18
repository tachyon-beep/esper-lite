# 0.1.1 Last-24h Defect Report

Date: 2026-06-18

Scope:

- Branch: `0.1.1`
- Review range: `0a0dd397625beb51f0722825c48228de47b1fd8a^..HEAD`
- Current HEAD: `05f7706c chore(mcp): normalize filigree server entry key order in .mcp.json`
- Requested mode: review only. No source/plan edits were made; this report is the only file updated.

Methods:

- Read repository rules from `CLAUDE.md`, `README.md`, and `ROADMAP.md`.
- Applied plan-review, Warpline, deep-RL, PyTorch engineering, quality-engineering, and Python-engineering review lenses.
- Used Warpline `changed` / `reverify` on the review range. Result was useful as a changed-surface checklist, but `reverify` reported `NO_SNAPSHOT`, so it was not treated as a complete impact proof.
- Used four read-only subagents for disjoint review slices: PPO/RL correctness, telemetry contracts/consumers, vectorized trainer/integration behavior, and release/process state.
- Verified returned findings against live source, local tests, guardrail scripts, git history, and Filigree state before including them.

Verdict: CHANGES_REQUESTED

The 0.1.1 branch contains important fixes and several high-risk areas appear sound, but there are actionable P1 defects in the new q-aux finiteness path, leyline guardrail state, Sanctum crash propagation, and q-head telemetry propagation.

## Findings

### P1 - q-aux non-finite values bypass the PPO finiteness gate and can poison the optimizer step

Files:

- `src/esper/simic/agent/ppo_agent.py:991`
- `src/esper/simic/agent/ppo_agent.py:1034`
- `src/esper/simic/agent/ppo_agent.py:1414`
- `src/esper/simic/agent/ppo_update.py:367`
- `src/esper/simic/agent/ppo_update.py:429`

Evidence:

- P0-1 adds `q_values = result.q_value` and slices it with the valid mask at `ppo_agent.py:991` and `ppo_agent.py:1013`.
- The finiteness gate at `ppo_agent.py:1034-1038` checks only new log-probs, old log-probs, and `values`; it does not check `q_values`.
- `q_values` feeds `per_step_q_loss = (q_values - normalized_returns) ** 2` at `ppo_update.py:367`, and `q_aux_coef * q_aux_loss` enters `total_loss` at `ppo_update.py:429`.
- The resulting gradients are clipped globally across `self.policy.network.parameters()` at `ppo_agent.py:1414-1416`.
- Local PyTorch check confirmed that one NaN grad in `clip_grad_norm_` returns a NaN norm and turns an unrelated finite grad into NaN:
  - `norm_is_nan True`
  - `finite_grad_after nan`
  - `nan_grad_after nan`

Impact:

A non-finite q-head auxiliary value can bypass the intended skip/fail-fast path, enter the combined loss, and contaminate actor/V/shared gradients during global clipping. This is an optimizer-safety regression, not just telemetry noise.

Suggested correction:

- Include `q_values` in the pre-backward finiteness gate with explicit source attribution.
- Add regression coverage that forces `q_value` to NaN/Inf and asserts:
  - `ppo_update_performed=False`
  - `finiteness_gate_failures` names the q source
  - no optimizer step mutates parameters
- Consider `error_if_nonfinite=True` or clipping q-head params separately from PPO actor/V/shared params if the intended isolation boundary is stronger than the current global clip.

### P1 - Leyline telemetry contract move leaves stale allowlist entries and fails the guardrail

Files:

- `leyline_boundaries.yaml:49`
- `leyline_boundaries.yaml:109`
- `src/esper/leyline/telemetry_contracts.py`

Evidence:

`src/esper/leyline/telemetry_contracts.py` moved `RewardComponentsTelemetry` and `ObservationStatsTelemetry` into leyline, but `leyline_boundaries.yaml` still whitelists their old simic locations:

- `src/esper/simic/rewards/reward_telemetry.py:dataclass:RewardComponentsTelemetry`
- `src/esper/simic/telemetry/observation_stats.py:dataclass:ObservationStatsTelemetry`

Verification:

`uv run python scripts/lint_leyline_types.py` fails:

```text
Checked 163 files, 148 type definitions
Stale whitelist entries: 2

STALE WHITELIST ENTRIES FOUND:

  src/esper/simic/rewards/reward_telemetry.py:dataclass:RewardComponentsTelemetry
  src/esper/simic/telemetry/observation_stats.py:dataclass:ObservationStatsTelemetry
```

Impact:

The committed branch does not pass the repository's leyline boundary guardrail after the telemetry-contract move.

Suggested correction:

Remove the two stale whitelist entries from `leyline_boundaries.yaml` and keep the stale-entry check active.

### P1 - Sanctum-mode training crashes still do not propagate promptly to the CLI

Files:

- `src/esper/scripts/train.py:64`
- `src/esper/scripts/train.py:1128`
- `src/esper/scripts/train.py:1134`
- `src/esper/scripts/train.py:1180`

Evidence:

- `_run_training_capturing_errors()` records the traceback and sets `shutdown_event` at `train.py:64-68`.
- `main()` then waits up to 60 seconds on `dataloader_ready_event` at `train.py:1126-1130`.
- It unconditionally constructs and runs `SanctumApp` at `train.py:1134-1140`.
- The non-zero exit helper is only called after the TUI exits at `train.py:1180`.

Impact:

A background training crash before the ready event can still leave the CLI waiting and then entering the TUI instead of exiting non-zero promptly. This is materially better than silent success, but it still violates the expected "crash means failed command" behavior for scripted or CI-like use.

Suggested correction:

After the dataloader wait, check `training_error[0]`, `shutdown_event.is_set()`, and `training_thread.is_alive()` before constructing the TUI. Exit non-zero immediately for captured crashes. Add a test that simulates a pre-ready-event crash and asserts `SanctumApp.run()` is not entered.

### P1 - q-head telemetry is emitted but not carried through the reviewed consumers

Files:

- `src/esper/leyline/telemetry.py:799`
- `src/esper/leyline/telemetry.py:888`
- `src/esper/simic/telemetry/emitters.py:1020`
- `src/esper/simic/telemetry/emitters.py:1041`
- `src/esper/karn/mcp/views.py:158`
- `src/esper/karn/sanctum/schema.py:1063`
- `src/esper/karn/sanctum/aggregator.py:1147`
- `src/esper/nissa/wandb_backend.py:359`
- `src/esper/karn/overwatch/web/src/types/sanctum.ts:426`

Evidence:

- `PPOUpdatePayload` adds `head_q_grad_norm` and `q_aux_loss`.
- `emit_ppo_update_event()` sets both fields.
- Karn's `ppo_updates` view projects the older Q fields (`op_q_values`, `op_valid_mask`, `q_variance`, `q_spread`) but stops before `q_aux_loss`.
- `TamiyoState` exposes per-head grad norms only through `head_op_grad_norm`, with no `head_q_grad_norm`.
- The Sanctum aggregator wires only the older Q fields.
- W&B logs PPO metrics through `bellman_error` in the inspected block and has no q-head metrics.
- Generated Overwatch types expose `op_q_values`, `op_valid_mask`, `q_variance`, and `q_spread`, but no `q_aux_loss` or `head_q_grad_norm`.

Impact:

The commit title says q-aux and q-head gradient telemetry are wired end-to-end, but the values are only in the raw typed payload/emitter path. Primary consumers cannot inspect the q-head training signal.

Suggested correction:

Either make these fields first-class telemetry and wire them through Karn SQL, Sanctum schema/aggregator, generated TypeScript, W&B, and consumer tests, or remove them from the payload/emitter if they are not intended to be consumed.

### P2 - q-head telemetry fails soft at emitter/parser boundaries

Files:

- `src/esper/simic/telemetry/emitters.py:1040`
- `src/esper/simic/telemetry/emitters.py:1041`
- `src/esper/leyline/telemetry.py:1026`
- `src/esper/leyline/telemetry.py:1086`
- `src/esper/leyline/telemetry.py:819`
- `src/esper/leyline/telemetry.py:827`

Evidence:

- `head_value_grad_norm` is direct-indexed at `emitters.py:1040`.
- The new `head_q_grad_norm` uses `.get()` immediately below at `emitters.py:1041`, even though `ppo_agent.py` now initializes and records `"q"` head gradient history.
- `PPOUpdatePayload.from_dict()` uses `.get()` for `head_q_grad_norm` and `q_aux_loss`, allowing serialized q-head telemetry to parse back as `None`.
- The payload has gradient-state fields through `head_value_gradient_state`, but no `head_q_gradient_state`, even though the agent records q-head gradient states internally.

Impact:

Missing q-head telemetry can be silently emitted or replayed as absent instead of failing loudly. This conflicts with the branch's own q-head observability goal and the repository's no bug-hiding rule.

Suggested correction:

Make q-head fields required wherever factored PPO updates are required, use direct indexing at the live emitter boundary, add `head_q_gradient_state`, and add missing-field regression tests.

### P2 - No-transition rollback attempts are not represented in structured telemetry

Files:

- `src/esper/simic/agent/rollout_buffer.py:208`
- `src/esper/simic/agent/rollout_buffer.py:842`
- `src/esper/simic/training/ppo_coordinator.py:155`
- `src/esper/simic/training/ppo_coordinator.py:297`

Evidence:

- The buffer comment defines `rollback_count` as the number of governor rollbacks in the rollout window.
- If `step_count == 0`, `mark_terminal_with_penalty()` returns `RollbackPenaltyResult(applied=False, steps_zeroed=0)` without incrementing any counter.
- The coordinator logs a warning for this path.
- `run_update()` emits only `buffer.rollback_count` and `buffer.rollback_steps_zeroed`.

Impact:

High first-step/no-attribution rollback rates remain only in logs, not structured metrics. That weakens the observability goal of making rollback starvation visible.

Suggested correction:

Track attempts separately from attributed transitions, for example `rollback_attempt_count` and `rollback_unattributed_count`, and emit them from coordinator-level metrics/events even when the buffer has no PPO row. Extend `test_handle_rollbacks_drops_penalty_when_no_executed_transition` to assert the structured counter.

### P2 - Public architecture docs still describe the pre-P0-1 critic as current

Files:

- `README.md:38`
- `ROADMAP.md:268`

Evidence:

- `README.md` still says the current critic is an action-conditioned baseline, `Q(s, op)` style.
- `ROADMAP.md` still lists `Q(s,op) critic: action-conditioned value baseline` as a delivered capability.
- The 0.1.1 branch now intentionally replaces the PPO baseline with op-independent `V(s)` and leaves Q as telemetry/aux only.

Impact:

The branch landed a checkpoint-breaking critic semantics change while the public current-state docs still describe the old semantics. This is especially risky because `CLAUDE.md` requires starting from README/ROADMAP and treats them as required context.

Suggested correction:

Update README/ROADMAP to state the current split explicitly: `state_value_head` is the PPO baseline V(s), while `q_head` is op-conditioned telemetry/aux and not the PPO baseline.

### P2 - Live tracker state disagrees with the branch and plan tracker

Files/state:

- `docs/coord/PLAN_TRACKER.md:3`
- `docs/coord/PLAN_TRACKER.md:78`
- `docs/coord/PLAN_TRACKER.md:114`
- Filigree session context

Evidence:

- `PLAN_TRACKER.md` says `esper-lite-6682b3faea` is fixed.
- `0.1.1` contains the code fix via `_select_hidden_for_envs()` and the bootstrap hidden slicing call.
- Live Filigree still reports ready P1 issue `esper-lite-6682b3faea`: "Recurrent vectorized PPO crashes on subset-truncation: GAE bootstrap hidden not sliced to truncated env subset".
- `PLAN_TRACKER.md` marks the post-P0-1 sprint, EV robustness, and main merge plans ready.
- Filigree searches for `post p01 hardening sprint`, `ev telemetry robustness`, `main merge integration`, and `dependency vulnerability triage` returned no matching open work items.

Impact:

The executable work queue does not reflect the branch and planning state. Agents using Filigree will still see a supposedly fixed P1 as ready and will not see the newly-authored sprint dependencies as live work.

Suggested correction:

Close or update `esper-lite-6682b3faea` with commit `31bf8cb7` and verification evidence, or make explicit if another target branch still lacks it. Create/link Filigree work items for the sprint umbrella, EV robustness, main merge, and dependency triage before marking them ready in the tracker.

## Plan Defects Relevant To Landing 0.1.1 Safely

These are not committed branch-code defects, but they affect the safety of the newly authored post-P0-1 hardening and main-merge plans that are currently present in the working tree.

### P1 - Dependency bump verification can pass while high alerts remain vulnerable

File:

- `docs/plans/ready/2026-06-18-main-merge-integration-plan.md:428`

Evidence:

The main-merge plan hardcodes stale or incomplete floors, including:

- `starlette >=0.40.0`
- `python-multipart >=0.0.18`
- `tornado >=6.5`
- `mistune >=3.1.0`

Reviewer live checks found current high/critical patched floors including:

- `starlette >=1.3.1`
- `python-multipart >=0.0.30`
- `tornado >=6.5.6`
- `mistune >=3.2.1`
- plus `jupyter-server >=2.18.0`, `jupyterlab >=4.5.7`, and `notebook >=7.5.6`

Suggested correction:

Generate the floor assertion map from the Dependabot API immediately before the bump, and fail unless every open high/critical package is at the advisory's current `first_patched_version` or is explicitly documented as an exception.

### P1 - EV calibration preflight cannot run from the stated Karn view

Files:

- `docs/plans/ready/2026-06-18-ev-telemetry-robustness-plan.md:158`
- `src/esper/karn/mcp/views.py:122`

Evidence:

The EV plan Step 0 tells the executor to query `ppo_updates` for `bellman_error`, `v_return_correlation`, and `value_loss`. Live `ppo_updates` projects `value_loss`, `explained_variance`, and `return_std`, but does not project `bellman_error` or `v_return_correlation`.

Suggested correction:

Either query `raw_events` JSON directly for `$.bellman_error` and `$.v_return_correlation`, or add those columns to `ppo_updates` before making Step 0 an executable hard preflight. Add a required-column preflight to the plan.

### P1 - EV design still permits suppressing proof blockers at the wrong layer

Files:

- `docs/superpowers/specs/2026-06-18-ev-telemetry-robustness-design.md:283`
- `src/esper/karn/mcp/views.py:649`

Evidence:

The design says flagged updates can be excluded from `proof_blocking` confounder logic. Live `run_confounders` marks surfaced confounders as `true as proof_blocking`, and the executable plan correctly says artifact suppression must happen upstream by preventing artefactual `VALUE_COLLAPSE_DETECTED` emission.

Suggested correction:

Rewrite the design to keep emitted `VALUE_COLLAPSE_DETECTED` rows proof-blocking. Allow low-variance flags to affect EV display rollups and upstream detector emission only, not view-layer proof-blocking suppression.

### P2 - EV gate parameter defaults weaken the required plumbing check

File:

- `docs/plans/ready/2026-06-18-ev-telemetry-robustness-plan.md:294`

Evidence:

The plan adds defaulted robust-signal parameters to `check_all` / `check_value_function`. That allows a missed caller to silently run with fake `bellman_error=0.0`, `value_loss=0.0`, and `ev_low_return_variance=False`, contrary to the same plan's fail-loud mandatory-field rule.

Suggested correction:

Make the new parameters required keyword-only, update the coordinator call in the same commit, and add a regression that the coordinator passes every robust signal explicitly.

### P2 - Post-push Karn false-alarm assertion is not executable as written

File:

- `docs/plans/ready/2026-06-18-main-merge-integration-plan.md:385`

Evidence:

The plan requires a `run_confounders` assertion, but gives no telemetry-producing command, `run_dir`, or SQL/MCP query. The referenced EV liftoff integration test builds an agent and reads `metrics["explained_variance"]` in-process; it does not emit/query Karn telemetry.

Suggested correction:

Add an explicit smoke that writes telemetry to a known `--telemetry-dir`, then query `run_confounders` for that run and assert zero proof-blocking `VALUE_COLLAPSE_DETECTED` rows.

## Inspected And Not Defective

- The op-independent V(s) path is consistently wired through `get_action()`, `get_value()`, `evaluate_actions()`, rollout-stored values, and bootstrap. It no longer uses op-conditioned Q as the PPO baseline.
- The q-aux gradient boundary is structurally correct: `_compute_q()` detaches `lstm_out`, so q-aux trains `q_head` without backpropagating into the LSTM.
- Checkpoint behavior is fail-loud and no-legacy: saves stamp `value_head_schema_version`, and old/mismatched checkpoints are rejected without remap/shim.
- The truncated-env bootstrap hidden-state slicing matches the subset order before the recurrent bootstrap forward.
- The `.mcp.json` change itself is valid JSON and only reorders Filigree server keys.
- Skipped PPO updates do not call `on_ppo_update`, so the new mandatory live `q_aux_loss` emitter key does not create a skipped-update crash.

## Verification Commands

Passed:

```bash
git diff --check 0a0dd397625beb51f0722825c48228de47b1fd8a^..HEAD
uv run pytest tests/simic/agent/test_q_aux_training.py tests/tamiyo/networks/test_state_value_head.py tests/simic/training/test_batch_bootstrap.py tests/simic/telemetry/test_emitters.py -q
```

Focused test result:

```text
40 passed
```

Failed as expected for a reported defect:

```bash
uv run python scripts/lint_leyline_types.py
```

Result:

```text
Stale whitelist entries: 2
```

Auxiliary verification:

- Local PyTorch snippet confirmed `clip_grad_norm_` with one NaN gradient contaminates an otherwise finite gradient.
- Filigree `session_context_get` confirmed `esper-lite-6682b3faea` remains ready P1.
- Filigree searches for the post-P0-1 sprint, EV robustness, main merge, and dependency triage returned no matching open issues.
- Warpline `reverify` returned `NO_SNAPSHOT`; its output was treated as advisory changed-set coverage, not a complete proof.
