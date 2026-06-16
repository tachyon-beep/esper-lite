# Kasmina/Tolaria Signal Audit Findings

Audit date: 2026-06-13

Scope: `src/esper/kasmina/`, `src/esper/tolaria/`, lifecycle/governor telemetry consumers in Simic/Karn, and the specified Kasmina/Tolaria/integration tests. Source was read-only.

Loomweave was used for orientation, but its index reported `stale`; all findings below are from current file reads.

## Lifecycle/Governor Feed Inventory

| Feed | Producer | Consumer | Real-vs-placeholder assessment |
|---|---|---|---|
| Seed lifecycle payloads (`SEED_GERMINATED`, `SEED_GATE_EVALUATED`, `SEED_STAGE_CHANGED`, `SEED_FOSSILIZED`, `SEED_PRUNED`) | `SeedSlot` emits typed payloads from germination, gate checks, transitions, fossilize and prune paths. Evidence: `src/esper/kasmina/slot.py:1446`, `src/esper/kasmina/slot.py:1508`, `src/esper/kasmina/slot.py:1537`, `src/esper/kasmina/slot.py:1564`, `src/esper/kasmina/slot.py:1663`, `src/esper/kasmina/slot.py:1680`. | Simic callback enriches env/episode context; Karn aggregates into `EnvState` and lifecycle timeline. Evidence: `src/esper/simic/training/env_factory.py:74`, `src/esper/simic/telemetry/emitters.py:558`, `src/esper/karn/sanctum/aggregator.py:1174`. | `slot_id`, `blueprint_id`, params, alpha and stage are real local state. `env_id=-1` is a sentinel until Simic enrichment. Germination gradient fields are explicit placeholders (`0.0`/`False`). Stage-change gradient fields are real only after `SeedTelemetry.epoch > 0`; otherwise they collapse to defaults. |
| Morphology causal log (`MORPHOLOGY_CAUSAL_LOG`) | Simic action execution emits proposal, verdict, mutation, watch, commit/fossilization and audit rows. Evidence: `src/esper/simic/training/action_execution.py:770`, `src/esper/simic/training/action_execution.py:798`, `src/esper/simic/training/action_execution.py:1094`, `src/esper/simic/training/action_execution.py:1184`, `src/esper/simic/training/action_execution.py:1204`, `src/esper/simic/training/action_execution.py:1219`. | Currently not consumed by `SanctumAggregator`; `_process_event_unlocked` has no `MORPHOLOGY_CAUSAL_LOG` route. Evidence: `src/esper/karn/sanctum/aggregator.py:346`. | Identity fields are real when emitted from Simic action execution. The `watch`/terminal/audit evidence currently reuses the current env validation loss rather than a post-mutation watch-window measurement. |
| Governor preflight verdict | `TolariaGovernor.preflight_lifecycle_mutation` builds a structured health snapshot and returns approval/veto. Evidence: `src/esper/tolaria/governor.py:89`, `src/esper/tolaria/governor.py:109`, `src/esper/tolaria/governor.py:188`. | Simic emits verdict into causal log. Evidence: `src/esper/simic/training/action_execution.py:781`, `src/esper/simic/training/action_execution.py:798`. | Preflight inputs are real current env state, but `cooldown_epochs_remaining` is always passed as `0`, so cooldown capacity proof is placeholder. Evidence: `src/esper/simic/training/action_execution.py:795`. |
| Governor rollback | `TolariaGovernor.execute_rollback` restores snapshot, prunes live seeds, emits `GOVERNOR_ROLLBACK`, resets panic state and returns report. Evidence: `src/esper/tolaria/governor.py:385`, `src/esper/tolaria/governor.py:458`, `src/esper/tolaria/governor.py:475`, `src/esper/tolaria/governor.py:518`, `src/esper/tolaria/governor.py:569`. | Simic action path also emits a second rollback event; Karn handles rollback events by setting env flash state. Evidence: `src/esper/simic/training/action_execution.py:632`, `src/esper/karn/sanctum/aggregator.py:1934`. | Rollback execution is real. The telemetry feed is duplicated and split: Tolaria emits threshold/panic context, Simic emits a second warning event with partial context. |
| Epoch seed snapshots | `VectorizedEmitter.on_epoch_completed` serializes per-slot telemetry from slot reports. Evidence: `src/esper/simic/telemetry/emitters.py:128`, `src/esper/simic/telemetry/emitters.py:140`. | Karn `EPOCH_COMPLETED` path updates env accuracy/loss and per-seed state. | Real when slot reports have real telemetry. Gradient health is placeholder when Simic fallback sync ran without gradient stats. Evidence: `src/esper/simic/training/vectorized_trainer.py:1280`. |

## Confirmed Findings

### KTS-001: Permissive G2 can pass with no measured gradient evidence

Severity: P1

`SeedTelemetry` defaults to healthy gradient evidence: `gradient_norm=0.0`, `gradient_health=1.0`, `has_vanishing=False`, `has_exploding=False` (`src/esper/leyline/telemetry.py:193`). Every `SeedState` creates this telemetry object in `__post_init__` (`src/esper/kasmina/slot.py:420`). In permissive G2, `QualityGates._check_g2` treats any non-`None` telemetry as safety evidence and passes if `telemetry.gradient_health >= threshold` and `has_exploding` is false (`src/esper/kasmina/slot.py:803`, `src/esper/kasmina/slot.py:818`, `src/esper/kasmina/slot.py:830`).

This makes absent gradient measurement look healthy in the permissive gate. The fallback sync path explicitly calls `sync_telemetry(epoch=..., max_epochs=...)` without gradient arguments when no gradient stats exist (`src/esper/simic/training/vectorized_trainer.py:1280`), and `sync_telemetry` leaves gradient fields at defaults when arguments are `None` (`src/esper/kasmina/slot.py:444`, `src/esper/kasmina/slot.py:487`).

Verification snippet run read-only:

```text
True ['trained_10_epochs', 'no_exploding_gradients', 'gradient_health_1.00'] []
```

Real-vs-placeholder: strict G2 correctly treats `seed_gradient_norm_ratio is None` as `gradient_stats_never_measured` (`src/esper/kasmina/slot.py:870`). Permissive G2 instead accepts the default telemetry snapshot as real gradient health.

### KTS-002: Morphology watch/commit/audit rows do not contain post-mutation evidence

Severity: P1

Simic emits proposal and verdict before applying the lifecycle handler (`src/esper/simic/training/action_execution.py:770`, `src/esper/simic/training/action_execution.py:781`). After handler execution, it emits `watch`, terminal (`commit` or `fossilization`), and `audit` rows immediately in the same action block using `watch_window_evidence=env_state.val_loss` (`src/esper/simic/training/action_execution.py:1184`, `src/esper/simic/training/action_execution.py:1193`, `src/esper/simic/training/action_execution.py:1204`, `src/esper/simic/training/action_execution.py:1213`, `src/esper/simic/training/action_execution.py:1219`, `src/esper/simic/training/action_execution.py:1228`).

That value is the current env loss already used for preflight (`src/esper/simic/training/action_execution.py:788`) and comes from the current epoch state before the lifecycle action was applied (`src/esper/simic/training/vectorized_trainer.py:1202`). There is no delayed validation, post-mutation measurement, or watch-window aggregation between mutation dispatch and the watch/commit/audit rows.

Real-vs-placeholder: proposal/verdict/mutation identity is real; `watch_window_evidence` is not evidence of the mutation outcome. It is a same-step placeholder/reuse of pre-action health.

### KTS-003: Governor rollback is emitted twice with conflicting context

Severity: P1

`TolariaGovernor.execute_rollback` emits a `GOVERNOR_ROLLBACK` directly after state restore and seed pruning (`src/esper/tolaria/governor.py:515`). That payload includes hard-coded `reason="Structural Collapse"`, `loss_at_panic`, `loss_threshold`, `consecutive_panics`, `panic_reason`, and key mismatch diagnostics (`src/esper/tolaria/governor.py:522`).

The Simic rollback branch then emits another `GOVERNOR_ROLLBACK` for the same rollback via `VectorizedEmitter` (`src/esper/simic/training/action_execution.py:632`). This second payload uses `reason=panic_reason or "unknown"` and includes `loss_at_panic` and `consecutive_panics`, but omits `loss_threshold` and `panic_reason` (`src/esper/simic/training/action_execution.py:635`). Tests explicitly encode this second event pattern (`tests/simic/training/test_governor_integration.py:512`) and assert `payload.reason == "governor_nan"` (`tests/simic/training/test_governor_integration.py:539`).

Karn consumes both events the same way and overwrites env rollback state from the latest payload reason (`src/esper/karn/sanctum/aggregator.py:1934`, `src/esper/karn/sanctum/aggregator.py:1952`).

Real-vs-placeholder: rollback execution is real. The feed is not single-source-of-truth; consumers can see two rollback facts with different severity and reason semantics for one causal rollback.

### KTS-004: Karn lifecycle timeline loses terminal transition origins and deltas

Severity: P2

Kasmina emits `SEED_STAGE_CHANGED` before terminal `SEED_FOSSILIZED`/`SEED_PRUNED` events. For fossilization this happens in `advance_stage`: stage change is emitted, then `SEED_FOSSILIZED` is emitted (`src/esper/kasmina/slot.py:1537`, `src/esper/kasmina/slot.py:1564`). For pruning, stage change is emitted, then `SEED_PRUNED` is emitted (`src/esper/kasmina/slot.py:1663`, `src/esper/kasmina/slot.py:1680`).

Karn updates `seed.stage` immediately from `SEED_STAGE_CHANGED` (`src/esper/karn/sanctum/aggregator.py:1247`). When it later builds the terminal lifecycle event, it uses the already-updated `seed.stage` as `from_stage`: fossilization uses `from_stage=seed.stage` after setting `seed.stage = "FOSSILIZED"` (`src/esper/karn/sanctum/aggregator.py:1296`, `src/esper/karn/sanctum/aggregator.py:1322`), and pruning uses `from_stage=seed.stage` after the preceding stage-change handler has already set it to `PRUNED` (`src/esper/karn/sanctum/aggregator.py:1352`).

The same handler drops transition deltas from stage-change timeline entries by setting `accuracy_delta=None` even though the payload carries `accuracy_delta` and the seed state is updated from it (`src/esper/karn/sanctum/aggregator.py:1253`, `src/esper/karn/sanctum/aggregator.py:1266`). Prune timeline entries also set `accuracy_delta=None` even though `SeedPrunedPayload` carries `improvement` and `counterfactual` (`src/esper/karn/sanctum/aggregator.py:1353`).

Real-vs-placeholder: current seed state may have the latest stage/delta, but the lifecycle timeline cannot prove the original terminal transition. It can record `FOSSILIZED -> FOSSILIZED` or `PRUNED -> PRUNED` self-transitions and omit the payload delta.

### KTS-005: Karn drops lifecycle causal IDs even when payloads carry them

Severity: P2

Lifecycle payloads carry morphology causal identity fields: germination (`src/esper/leyline/telemetry.py:1128`), stage change (`src/esper/leyline/telemetry.py:1194`), fossilization (`src/esper/leyline/telemetry.py:1283`), and pruning (`src/esper/leyline/telemetry.py:1345`). `SeedSlot._emit_telemetry` fills those fields from pending Simic context for lifecycle events (`src/esper/kasmina/slot.py:2656`, `src/esper/kasmina/slot.py:2666`).

Karn's `SeedLifecycleEvent` schema has only epoch/action/stage/blueprint/slot/alpha/accuracy fields and no proposal/verdict/mutation/RNG identity (`src/esper/karn/sanctum/schema.py:169`). The aggregator handlers build lifecycle events without copying any morphology IDs (`src/esper/karn/sanctum/aggregator.py:1217`, `src/esper/karn/sanctum/aggregator.py:1266`, `src/esper/karn/sanctum/aggregator.py:1318`, `src/esper/karn/sanctum/aggregator.py:1352`).

Real-vs-placeholder: lifecycle payload identity is real when Simic action execution set pending context. The Karn lifecycle feed discards it, so UI/history cannot join a lifecycle row back to proposal, governor verdict, RNG stream, observation hash, or mutation ID.

### KTS-006: Fossilize/prune payload schemas expose `blending_delta`, but emitters never populate it

Severity: P3

`SeedMetrics.blending_delta` exists and explicitly states it is telemetry/logging only, not causal attribution (`src/esper/kasmina/slot.py:247`). `SeedFossilizedPayload` and `SeedPrunedPayload` both include optional `blending_delta` fields (`src/esper/leyline/telemetry.py:1282`, `src/esper/leyline/telemetry.py:1343`).

Kasmina's fossilize emitter sends `improvement`, `epochs_total`, and `counterfactual`, but not `blending_delta` (`src/esper/kasmina/slot.py:1565`). The prune emitter similarly sends `improvement`, `epochs_total`, and `counterfactual`, but not `blending_delta` (`src/esper/kasmina/slot.py:1680`).

Real-vs-placeholder: `counterfactual` is real when Simic has computed it (`src/esper/simic/training/vectorized_trainer.py:1009`) and G5 requires it for fossilization (`src/esper/kasmina/slot.py:967`). `blending_delta` is a schema placeholder today.

## Tracker-Ready Issue Rows

| ID | Priority | Title | Evidence | Acceptance tests |
|---|---:|---|---|---|
| KTS-001 | P1 | Permissive G2 must fail when gradient health has not been measured | `SeedTelemetry` defaults healthy at `src/esper/leyline/telemetry.py:193`; permissive gate accepts it at `src/esper/kasmina/slot.py:818`; fallback sync omits gradient stats at `src/esper/simic/training/vectorized_trainer.py:1280`. | Add a Kasmina gate test where `SeedState` is freshly constructed, `epochs_in_current_stage >= DEFAULT_MIN_BLENDING_EPOCHS`, no gradient sync has occurred, and permissive G2 fails with `gradient_health_missing` or `gradient_stats_never_measured`. Add a vectorized-trainer test proving fallback accuracy-only sync cannot satisfy G2 safety. |
| KTS-002 | P1 | Morphology causal watch/commit/audit must use post-mutation evidence or be renamed as dispatch-only | Same-step `watch_window_evidence=env_state.val_loss` at `src/esper/simic/training/action_execution.py:1193`, `src/esper/simic/training/action_execution.py:1213`, `src/esper/simic/training/action_execution.py:1228`. | Add a Simic action-execution test that mutates env loss after a validation/watch step and asserts watch/commit/audit rows use that post-mutation measurement. If no watch window exists, assert phase names/messages do not claim watch/audit evidence. |
| KTS-003 | P1 | Make rollback telemetry single-source and complete | Tolaria emits at `src/esper/tolaria/governor.py:518`; Simic emits second event at `src/esper/simic/training/action_execution.py:632`; Karn handles both at `src/esper/karn/sanctum/aggregator.py:1934`. | Add an integration test for one panic rollback that captures hub events and asserts exactly one `GOVERNOR_ROLLBACK` per env, with `panic_reason`, `loss_at_panic`, `loss_threshold`, `consecutive_panics`, `env_id`, `device`, and episode context present. |
| KTS-004 | P2 | Preserve terminal lifecycle from-stage and deltas in Karn timeline | Fossilize terminal event uses already-updated stage at `src/esper/karn/sanctum/aggregator.py:1296` and `src/esper/karn/sanctum/aggregator.py:1322`; prune terminal event uses current seed stage at `src/esper/karn/sanctum/aggregator.py:1353`; stage-change deltas are dropped at `src/esper/karn/sanctum/aggregator.py:1274`. | Add Karn aggregator tests that feed `HOLDING -> FOSSILIZED` stage change followed by `SEED_FOSSILIZED` and assert lifecycle timeline contains the original `HOLDING -> FOSSILIZED`, not `FOSSILIZED -> FOSSILIZED`. Add analogous `BLENDING/HOLDING -> PRUNED` test and assert payload deltas are retained. |
| KTS-005 | P2 | Carry morphology causal IDs into Karn lifecycle rows | Payload IDs exist at `src/esper/leyline/telemetry.py:1128`, `src/esper/leyline/telemetry.py:1194`, `src/esper/leyline/telemetry.py:1283`, `src/esper/leyline/telemetry.py:1345`; Karn schema lacks them at `src/esper/karn/sanctum/schema.py:169`. | Extend `SeedLifecycleEvent` with proposal/verdict/mutation IDs, observation hash, RNG stream/seed as applicable. Add aggregator tests proving IDs from each lifecycle payload are visible in snapshot lifecycle events. |
| KTS-006 | P3 | Populate or remove fossilize/prune `blending_delta` | Schema fields at `src/esper/leyline/telemetry.py:1282` and `src/esper/leyline/telemetry.py:1343`; emitters omit them at `src/esper/kasmina/slot.py:1565` and `src/esper/kasmina/slot.py:1680`. | Add Kasmina lifecycle payload tests that fossilize/prune after blending and assert `blending_delta == state.metrics.blending_delta`, or remove the field and update downstream schemas/tests so no consumer expects it. |

