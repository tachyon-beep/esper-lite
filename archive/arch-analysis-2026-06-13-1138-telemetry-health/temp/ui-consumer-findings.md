# UI Consumer Findings

Date: 2026-06-13

Scope: read-only audit of Sanctum schema/aggregator/widgets, Overwatch backend/web types/components, websocket output, generated TypeScript drift, and sentinel/fallback display paths. Source lines below are current source, not copied from the coordinator summary. No Filigree issues were created.

## Feeds Examined

| Feed | Producer/source | Live backend | UI consumers | Assessment |
| --- | --- | --- | --- | --- |
| Seed lifecycle events | `TelemetryEventType.SEED_*` and seed payloads in `src/esper/leyline/telemetry.py` | `SanctumAggregator._handle_seed_event()` creates `EnvState.lifecycle_events` | Sanctum event/lifecycle widgets, Overwatch `SeedSwimlane`, best-run lifecycle history | Real stage data exists, but live lifecycle rows drop causal identity fields. |
| Morphology causal log | `TelemetryEventType.MORPHOLOGY_CAUSAL_LOG` from Simic action execution | No structured Sanctum handler; only generic event log fallback | Raw WebSocket clients can see events; Overwatch snapshot clients cannot | Raw feed is real but live snapshot path is missing. |
| Aggregated snapshot | `SanctumAggregator.get_snapshot()` | Overwatch `OverwatchBackend` broadcasts `type: "snapshot"` envelope | Vue `useOverwatch()` and all Overwatch panels | Snapshot is authoritative for Overwatch; missing fields before aggregation cannot be recovered by clients. |
| Raw event websocket | `WebSocketOutput.emit()` | Serializes each event with `esper.karn.serialization.serialize_event()` | Separate raw-event dashboard/client path, not the Overwatch app | Can carry causal-log payloads if connected, but it is not the Overwatch snapshot contract. |
| Generated TS schema | `scripts/generate_overwatch_types.py` from Python dataclasses | `src/esper/karn/overwatch/web/src/types/sanctum.ts` | All Vue components | Type generation is mechanically aligned with Python schema, so Python omissions become web omissions. |
| Sentinel/fallback displays | Dataclass defaults and component fallback helpers | `SanctumSnapshot()`/`TamiyoState()`/`EnvState()` defaults | Health gauges, policy diagnostics, action context, seed swimlane, TUI health panel | Several defaults render as zero/green/OK rather than unknown or pending. |

## Producer -> Payload -> Backend -> Consumer Path

1. Seed lifecycle identity is produced in typed payloads. `SeedGerminatedPayload` carries `morphology_proposal_id`, `morphology_verdict_id`, `morphology_mutation_id`, `rng_stream`, and `rng_seed` at `src/esper/leyline/telemetry.py:1128-1132`; `SeedStageChangedPayload` carries the same causal fields at `src/esper/leyline/telemetry.py:1194-1198`; fossilize/prune payloads carry them at `src/esper/leyline/telemetry.py:1283-1287` and `src/esper/leyline/telemetry.py:1345-1349`.
2. Sanctum reduces those payloads into `SeedLifecycleEvent`, whose schema has only `epoch`, `action`, `from_stage`, `to_stage`, `blueprint_id`, `slot_id`, `alpha`, and `accuracy_delta` at `src/esper/karn/sanctum/schema.py:169-183`.
3. The aggregator constructs lifecycle events from seed payloads but does not copy causal IDs or RNG identity at `src/esper/karn/sanctum/aggregator.py:1217-1227`, `src/esper/karn/sanctum/aggregator.py:1266-1275`, `src/esper/karn/sanctum/aggregator.py:1318-1328`, and `src/esper/karn/sanctum/aggregator.py:1352-1362`.
4. Overwatch broadcasts aggregated snapshots only: `OverwatchBackend.emit()` processes events through the aggregator at `src/esper/karn/overwatch/backend.py:141-150`; `_broadcast()` wraps `_json_safe(snapshot)` in a `type: "snapshot"` envelope at `src/esper/karn/overwatch/backend.py:208-224`; `useOverwatch()` only updates state for `message.type === 'snapshot'` at `src/esper/karn/overwatch/web/src/composables/useOverwatch.ts:57-65`.
5. The generated TypeScript interface for `SeedLifecycleEvent` mirrors the reduced Python schema at `src/esper/karn/overwatch/web/src/types/sanctum.ts:72-81`; generation explicitly includes `SeedLifecycleEvent` from Python dataclasses at `scripts/generate_overwatch_types.py:224-250`.
6. Raw-event websocket output is a separate path: `WebSocketOutput.emit()` serializes individual events at `src/esper/karn/websocket_output.py:89-100` and sends queued messages directly at `src/esper/karn/websocket_output.py:282-313`. It can carry the raw causal-log payload, but Overwatch is not consuming that stream.

## Real-vs-Placeholder Data Assessment

| Surface | Real data | Placeholder/default risk |
| --- | --- | --- |
| Raw `MORPHOLOGY_CAUSAL_LOG` | Real typed payload exists (`MorphologyCausalLogPayload` fields at `src/esper/leyline/telemetry.py:2064-2090`) and Simic emits proposal/mutation/watch/commit/audit rows (`src/esper/simic/training/action_execution.py:770-780`, `src/esper/simic/training/action_execution.py:1094-1107`, `src/esper/simic/training/action_execution.py:1184-1233`). | Not present as structured live snapshot state; Overwatch cannot render/join it from the snapshot contract. |
| Lifecycle timeline | Real stage transitions and counts are aggregated. | Causal identity fields already present in seed payloads are omitted, so lifecycle rows are not joinable to causal-log phases. |
| Health gauges/policy diagnostics | Real once PPO/vitals telemetry has arrived. | `TamiyoState` and `SystemVitals` default many fields to `0.0`/`False` (`src/esper/karn/sanctum/schema.py:925-943`, `src/esper/karn/sanctum/schema.py:1139-1157`), and Vue renders those directly as percentages or `OK`. |
| Seed swimlane | Real when `env.seeds` has slot state. | Missing seed data for configured slots is manufactured as a full dormant `SeedState` with zero metrics (`src/esper/karn/overwatch/web/src/components/SeedSwimlane.vue:21-50`). |
| TUI health status | Some missingness is rendered honestly as `No data` or `--` in detail widgets. | Several NaN/no-data policy status helpers return `OK` (`src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py:657-678`, `src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py:685-692`). |

## Findings

### UI-001: `MORPHOLOGY_CAUSAL_LOG` is raw-view-only, not live UI state

Severity: P1

Confidence: High

Blocks solid signals of life: Yes. The live UI cannot prove proposal -> verdict -> mutation -> watch -> terminal causality, even though raw telemetry emits it.

Evidence:

- Event type exists: `TelemetryEventType.MORPHOLOGY_CAUSAL_LOG` is defined at `src/esper/leyline/telemetry.py:100-103`.
- Payload has the join identity fields: `action_id`, `proposal_id`, `verdict_id`, `mutation_id`, `observation_hash`, RNG identity, topology, governor fields, and linked event id at `src/esper/leyline/telemetry.py:2064-2090`.
- Simic emits causal-log rows at proposal, mutation, watch, commit/audit, and rollback paths: `src/esper/simic/training/action_execution.py:770-780`, `src/esper/simic/training/action_execution.py:1094-1107`, `src/esper/simic/training/action_execution.py:1184-1233`, and `src/esper/simic/training/action_execution.py:659-670`.
- Karn MCP has a raw `morphology_causal_log` view at `src/esper/karn/mcp/views.py:268-297`.
- Sanctum dispatch only handles training, epoch, PPO, `SEED_*`, batch, counterfactual, analytics, episode outcome, and governor rollback events at `src/esper/karn/sanctum/aggregator.py:345-363`; there is no `MORPHOLOGY_CAUSAL_LOG` handler.
- Overwatch broadcasts only aggregator snapshots at `src/esper/karn/overwatch/backend.py:208-224`, and the web client ignores non-snapshot messages at `src/esper/karn/overwatch/web/src/composables/useOverwatch.ts:57-65`.

Missing/miswired/mutated fields:

- Entire causal-log payload is absent from `SanctumSnapshot`.
- Event log fallback may show the message string, but it does not expose the structured causal join.

Tracker-ready issue row:

| Field | Value |
| --- | --- |
| Title | Add structured live UI route for morphology causal-log telemetry |
| Severity | P1 |
| Files | `src/esper/karn/sanctum/schema.py`, `src/esper/karn/sanctum/aggregator.py`, `scripts/generate_overwatch_types.py`, Overwatch components/tests |
| Acceptance tests | A `MORPHOLOGY_CAUSAL_LOG` event processed by `SanctumAggregator` appears in snapshot state with `action_id`, `proposal_id`, `verdict_id`, `mutation_id`, `phase`, `watch_window_evidence`, and `linked_event_id`; Overwatch TypeScript generation includes the new type; a Vue test renders or exposes the causal rows from a snapshot; raw-event-only websocket is not required for Overwatch causal visibility. |

### UI-002: Seed lifecycle snapshot drops causal identity already present in seed payloads

Severity: P1

Confidence: High

Blocks solid signals of life: Yes. The UI can show that a seed changed stage but cannot join the stage change to the proposal/verdict/mutation/watch chain.

Evidence:

- Seed payloads carry causal IDs and RNG identity: germinated at `src/esper/leyline/telemetry.py:1128-1132`, stage changed at `src/esper/leyline/telemetry.py:1194-1198`, fossilized at `src/esper/leyline/telemetry.py:1283-1287`, pruned at `src/esper/leyline/telemetry.py:1345-1349`.
- `SeedLifecycleEvent` omits these fields at `src/esper/karn/sanctum/schema.py:169-183`.
- Aggregator constructs lifecycle events without copying them at `src/esper/karn/sanctum/aggregator.py:1217-1227`, `src/esper/karn/sanctum/aggregator.py:1266-1275`, `src/esper/karn/sanctum/aggregator.py:1318-1328`, and `src/esper/karn/sanctum/aggregator.py:1352-1362`.
- Generated TS confirms the web contract lacks them at `src/esper/karn/overwatch/web/src/types/sanctum.ts:72-81`.
- Existing lifecycle tests assert only stage/action fields, not causal identity, at `tests/karn/sanctum/test_aggregator_lifecycle.py:35-43`, `tests/karn/sanctum/test_aggregator_lifecycle.py:75-80`, and `tests/karn/sanctum/test_aggregator_lifecycle.py:150-153`.

Missing/miswired/mutated fields:

- Missing from live lifecycle UI contract: `morphology_proposal_id`, `morphology_verdict_id`, `morphology_mutation_id`, `rng_stream`, `rng_seed`.
- `SeedFossilizedPayload.blending_delta` and prune/fossilize causal context also do not surface in `SeedLifecycleEvent`.

Tracker-ready issue row:

| Field | Value |
| --- | --- |
| Title | Preserve seed lifecycle causal IDs through Sanctum and Overwatch snapshot types |
| Severity | P1 |
| Files | `src/esper/karn/sanctum/schema.py`, `src/esper/karn/sanctum/aggregator.py`, `src/esper/karn/overwatch/web/src/types/sanctum.ts`, `tests/karn/sanctum/test_aggregator_lifecycle.py` |
| Acceptance tests | Given seed lifecycle payloads with causal IDs and RNG identity, `SanctumAggregator.get_snapshot().envs[env].lifecycle_events[-1]` preserves those exact values; generated `SeedLifecycleEvent` TypeScript includes them; an Overwatch component/factory test can assert the IDs are present rather than fabricated. |

### UI-003: Generated Overwatch TypeScript is in sync with Python but faithfully propagates schema omissions

Severity: P2

Confidence: High

Blocks solid signals of life: Indirectly yes. It does not create the data loss, but it locks the web app to the reduced snapshot contract.

Evidence:

- The generator imports `SeedLifecycleEvent` from `esper.karn.sanctum.schema` at `scripts/generate_overwatch_types.py:36-57`.
- It mechanically emits every dataclass field via `dataclasses.fields(cls)` at `scripts/generate_overwatch_types.py:187-210`.
- `SeedLifecycleEvent` is included in the generation list at `scripts/generate_overwatch_types.py:224-250`.
- The generated TypeScript `SeedLifecycleEvent` contains only the reduced fields at `src/esper/karn/overwatch/web/src/types/sanctum.ts:72-81`.

Missing/miswired/mutated fields:

- No generator drift was found for `SeedLifecycleEvent`; the drift is architectural: Python schema omits causal fields, so TypeScript cannot reference them.
- There is no visible generated-type freshness guard in this lane proving `sanctum.ts` was regenerated after schema changes.

Tracker-ready issue row:

| Field | Value |
| --- | --- |
| Title | Add generated TypeScript freshness check for Sanctum/Overwatch schema |
| Severity | P2 |
| Files | `scripts/generate_overwatch_types.py`, `src/esper/karn/overwatch/web/src/types/sanctum.ts`, CI/test guardrail |
| Acceptance tests | A test or CI script regenerates Overwatch types and fails if the committed `sanctum.ts` differs; adding a field to `SeedLifecycleEvent` without regenerating TypeScript fails locally and in CI. |

### UI-004: Web and TUI health displays can present no-data defaults as clean or measured values

Severity: P2

Confidence: Medium-High

Blocks solid signals of life: Partially. It does not erase raw data, but it weakens operator confidence because absent telemetry can look like healthy telemetry.

Evidence:

- `TamiyoState` defaults core policy metrics to `0.0`, ratio metrics to `1.0`, gradient counts to `0`, and entropy-collapse to `False` at `src/esper/karn/sanctum/schema.py:925-990`.
- `SystemVitals` defaults GPU/CPU/RAM/throughput/host params to zero at `src/esper/karn/sanctum/schema.py:1139-1157`.
- Overwatch health gauges render those values directly: GPU utilization at `src/esper/karn/overwatch/web/src/components/HealthGauges.vue:33-38`, entropy at `src/esper/karn/overwatch/web/src/components/HealthGauges.vue:54-65`, clip fraction at `src/esper/karn/overwatch/web/src/components/HealthGauges.vue:67-74`, and explained variance at `src/esper/karn/overwatch/web/src/components/HealthGauges.vue:76-87`.
- `PolicyDiagnostics` renders `dead_layers`, `exploding_layers`, `nan_grad_count`, `entropy_collapsed`, losses, ratios, advantages, entropies, and grad norms directly at `src/esper/karn/overwatch/web/src/components/PolicyDiagnostics.vue:37-93` and `src/esper/karn/overwatch/web/src/components/PolicyDiagnostics.vue:96-260`.
- TUI status helpers treat NaN/no-data as `OK` for skewness, kurtosis, positive-ratio, and log-prob status at `src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py:657-678` and `src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py:685-692`.

Missing/miswired/mutated fields:

- Missing "unknown/pending/no data" discriminators for policy/vitals fields whose zero value is also a valid measurement.
- Web panels cannot tell "no PPO/vitals telemetry yet" from "healthy zero faults and zero load" except in some outer gates.

Tracker-ready issue row:

| Field | Value |
| --- | --- |
| Title | Distinguish missing telemetry from measured zero in UI health panels |
| Severity | P2 |
| Files | `src/esper/karn/sanctum/schema.py`, `src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py`, `src/esper/karn/overwatch/web/src/components/HealthGauges.vue`, `src/esper/karn/overwatch/web/src/components/PolicyDiagnostics.vue` |
| Acceptance tests | Empty/default `SanctumSnapshot()` renders pending/unknown for policy and vitals gauges, not green OK or measured zero; a snapshot with explicit measured zero still renders zero; TUI no-data helpers return neutral/pending labels instead of `OK` where absence is not health. |

### UI-005: Seed swimlane fabricates dormant rows for missing slot state

Severity: P2

Confidence: High

Blocks solid signals of life: Partially. It can make missing per-slot consumer data look like confirmed dormant seed state.

Evidence:

- `SeedSwimlane.getSeedForSlot()` returns a complete default dormant `SeedState` when `props.seeds[slotId]` is absent at `src/esper/karn/overwatch/web/src/components/SeedSwimlane.vue:21-50`.
- The template renders a stage bar from that manufactured state at `src/esper/karn/overwatch/web/src/components/SeedSwimlane.vue:93-109`.
- The existing test codifies the fallback as desired behavior: missing `slot_1` renders a `stage-dormant` bar at `src/esper/karn/overwatch/web/src/components/__tests__/SeedSwimlane.spec.ts:137-159`.
- The aggregator itself treats absent seed state as dormant for aggregate counts at `src/esper/karn/sanctum/aggregator.py:493-505`, so the web component is reinforcing an upstream absence-as-dormant convention.

Missing/miswired/mutated fields:

- Missing separate UI state for "configured slot with no seed row in current env snapshot".
- Missing evidence bit that distinguishes "observed dormant" from "consumer filled absence".

Tracker-ready issue row:

| Field | Value |
| --- | --- |
| Title | Show missing slot state distinctly from observed dormant seed state |
| Severity | P2 |
| Files | `src/esper/karn/sanctum/aggregator.py`, `src/esper/karn/overwatch/web/src/components/SeedSwimlane.vue`, `src/esper/karn/overwatch/web/src/components/__tests__/SeedSwimlane.spec.ts` |
| Acceptance tests | A configured slot absent from `env.seeds` renders as missing/unknown or pending, not `DORMANT`; an explicit `SeedState(stage="DORMANT")` still renders dormant; aggregate slot counts distinguish observed dormant from missing when the data source can provide that distinction. |

## File/Line Evidence Summary

| Topic | Evidence |
| --- | --- |
| Causal-log event type | `src/esper/leyline/telemetry.py:100-103` |
| Causal-log payload identity fields | `src/esper/leyline/telemetry.py:2064-2090` |
| Simic causal-log emissions | `src/esper/simic/training/action_execution.py:770-780`, `src/esper/simic/training/action_execution.py:1094-1107`, `src/esper/simic/training/action_execution.py:1184-1233`, `src/esper/simic/training/action_execution.py:659-670` |
| Sanctum dispatch lacks causal-log handler | `src/esper/karn/sanctum/aggregator.py:345-363` |
| Seed payload causal fields | `src/esper/leyline/telemetry.py:1128-1132`, `src/esper/leyline/telemetry.py:1194-1198`, `src/esper/leyline/telemetry.py:1283-1287`, `src/esper/leyline/telemetry.py:1345-1349` |
| Reduced lifecycle schema | `src/esper/karn/sanctum/schema.py:169-183` |
| Lifecycle event construction | `src/esper/karn/sanctum/aggregator.py:1217-1227`, `src/esper/karn/sanctum/aggregator.py:1266-1275`, `src/esper/karn/sanctum/aggregator.py:1318-1328`, `src/esper/karn/sanctum/aggregator.py:1352-1362` |
| Snapshot-only Overwatch path | `src/esper/karn/overwatch/backend.py:208-224`, `src/esper/karn/overwatch/web/src/composables/useOverwatch.ts:57-65` |
| Generated TS lifecycle type | `src/esper/karn/overwatch/web/src/types/sanctum.ts:72-81` |
| Type generator mechanics | `scripts/generate_overwatch_types.py:187-210`, `scripts/generate_overwatch_types.py:224-250` |
| Raw websocket event path | `src/esper/karn/websocket_output.py:89-100`, `src/esper/karn/websocket_output.py:282-313` |
| Defaults rendered as data | `src/esper/karn/sanctum/schema.py:925-990`, `src/esper/karn/sanctum/schema.py:1139-1157`, `src/esper/karn/overwatch/web/src/components/HealthGauges.vue:33-87`, `src/esper/karn/overwatch/web/src/components/PolicyDiagnostics.vue:37-93` |
| TUI no-data-as-OK helpers | `src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py:657-678`, `src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py:685-692` |
| Seed swimlane fabricated dormant state | `src/esper/karn/overwatch/web/src/components/SeedSwimlane.vue:21-50`, `src/esper/karn/overwatch/web/src/components/SeedSwimlane.vue:93-109`, `src/esper/karn/overwatch/web/src/components/__tests__/SeedSwimlane.spec.ts:137-159` |

## Solid Signals-of-Life Blocking Summary

| Issue | Blocks solid signals of life? | Reason |
| --- | --- | --- |
| UI-001 | Yes | Causal morphology proof exists only in raw/event-store surfaces, not live operator UI state. |
| UI-002 | Yes | Lifecycle UI cannot join stage transitions to causal proposal/verdict/mutation/watch identity. |
| UI-003 | Indirectly | Type generation preserves the omission and can silently leave the web app behind schema changes without a freshness gate. |
| UI-004 | Partially | Missing telemetry can look like clean/zero/OK, weakening UI evidence quality. |
| UI-005 | Partially | Missing slot state can look like observed dormant state. |

