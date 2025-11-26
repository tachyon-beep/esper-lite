# Shared Foundations — Async, Telemetry, Strict Dependencies

## Objectives
- Provide common infrastructure underpinning Tolaria, Tamiyo, and Kasmina during RC1 execution.
- Ensure strict dependency policy (no synthetic fallbacks/IDs) is enforced consistently.
- Standardise async execution and telemetry routing to reduce duplicated logic and failure cases.

## Components
| Component | Description |
|-----------|-------------|
| Async Worker | Shared cancellable task runner used for Tamiyo inference, Kasmina metadata/prefetch, Tolaria service calls. |
| Telemetry Router | Defines priority mapping (INFO/NORMAL vs WARNING/HIGH vs CRITICAL/EMERGENCY) and Weatherlight routing rules; load tested via integration harness + CLI. |
| Strict Dependency Guard | Shared helper raising errors for missing IDs, fallback requests, invalid configurations. |
| Shared Config Schema | Defines PT2.8 assumptions, worker settings, fallback disabling. |
| Telemetry Schema | Common fields for timeout/gate/blend/command verifications across subsystems. |

## Implementation Status (2025-09-29)
- **Async Worker** — Shared `AsyncWorker` module shipped (`src/esper/core/async_runner.py:1`). Tolaria, Tamiyo, Kasmina, and Weatherlight now inject or share a single worker instance (`src/esper/tolaria/trainer.py:970`, `src/esper/tamiyo/service.py:320`, `src/esper/weatherlight/service_runner.py:1180`). Integration harness `tests/integration/test_async_worker_backpressure.py:1` validates cancellation/back-pressure metrics.
- **Telemetry Router** — Telemetry packets carry priority indicators that are honoured end-to-end. Tamiyo stamps priority before publication (`src/esper/tamiyo/service.py:1380`), Weatherlight preserves the enum when flushing (`src/esper/weatherlight/service_runner.py:1148`), and Kasmina dispatcher telemetry includes proper routing hints (`src/esper/kasmina/seed_manager.py:1`). Emergency routing is exercised via `tests/weatherlight/test_service_priority.py:1` and `tests/integration/test_weatherlight_tamiyo_emergency.py:1`.
- **Strict Dependency Guard** — Central guard utilities live in `src/esper/core/dependency_guard.py:1`. Tolaria validates seed commands before application (`src/esper/tolaria/trainer.py:2679`), Tamiyo verifies command dependencies prior to publication (`src/esper/tamiyo/service.py:2412`), and Kasmina treats guard failures as gate errors (`src/esper/kasmina/seed_manager.py:1253`).
- **Observability & Docs** — `docs/project/observability_runbook.md:121` documents emergency metrics, WAL workflows, and async worker telemetry. `CHANGELOG_RC1.md:60` and `:108` capture shared-foundation test runs.
- **Tooling** — CLI harnesses for async worker soak, telemetry routing load, and WAL backup/soak live under `scripts/` and have unit coverage (`scripts/tamiyo_wal_soak.py:1`, `tests/scripts/test_tamiyo_wal_soak.py:1`).

## Outstanding Follow-ups
- **Leyline struct-first cutover** — Prototype still relies on `system_health.indicators["priority"]`. The draft message schema below remains the target; coordinating BusEnvelope/TelemetryPacket updates with Leyline generation is open work.
- **Tamiyo WP-A2 alignment** — Dependency guard uncovers placeholder IDs only after Tamiyo stops emitting defaults (`src/esper/tamiyo/policy.py:897`). This is tracked under WP-A2.
- **Tolaria WP-T3/T4 integration** — Shared worker now in place, but rollback/profiler hardening and telemetry cleanup must adopt the new primitives before RC1 sign-off (`02_wp_TOLARIA.md:186`).

### Communication Review & Struct-First Standardisation Plan

#### Tolaria
- **Outbound payloads today**: `SystemStatePacket` assembled in `_emit_state` and published via `publish_history` (`src/esper/tolaria/trainer.py:1589`, `src/esper/tolaria/trainer.py:2220`); `TelemetryPacket` built in `_emit_telemetry` (`src/esper/tolaria/trainer.py:1940`); `EmergencySignal` sent through `_dispatch_emergency_signal` (`src/esper/tolaria/trainer.py:2158`).
- **Inbound dependencies**: consumes Tamiyo `AdaptationCommand` responses (`src/esper/tolaria/trainer.py:2083`) and relays Kasmina command application latency via executor (`src/esper/tolaria/trainer.py:2108`).
- **Unstructured surfaces**:
  - `SystemStatePacket.training_metrics` stores dozens of ad-hoc keys (e.g., `"tamiyo_latency_ms"`, `"grad_conflict_rate"`, `"gpu_mem_used_gb"`) (`src/esper/tolaria/trainer.py:1128-1639`).
  - Telemetry priority is encoded as `system_health.indicators["priority"]`, so routing depends on string parsing downstream (`src/esper/tolaria/trainer.py:1995-2016`).
- **Standardisation actions**:
  1. Extend `TelemetryPacket` with an explicit `MessagePriority priority` field and update Tolaria to populate it instead of the indicator map.
  2. Introduce a typed `TrainingMetrics` message (or enum-backed metric ids) under `SystemStatePacket` covering: optimizer stats, hook timings, device telemetry, Tamiyo/Kasmina latencies. Deprecate free-form map usage once migration completes.
  3. Provide a `TelemetryEmitter` helper in `esper.core` that accepts structured metric objects and enforces the new priority contract, so Tolaria publishes through a single compliant API.

#### Tamiyo
- **Outbound payloads today**: `AdaptationCommand` from `_build_command` (`src/esper/tamiyo/policy.py:432`), enriched via `command.annotations`; `FieldReport` built in `_emit_field_report` with `metrics_delta` map (`src/esper/tamiyo/service.py:1874`); telemetry packets created alongside decisions (`src/esper/tamiyo/service.py:600-706`); WAL sidecars.
- **Inbound payloads**: `SystemStatePacket` from Tolaria, Urza metadata fetched via Oona, Kasmina feedback via field reports.
- **Unstructured surfaces**:
  - Heavy reliance on `AdaptationCommand.annotations` for policy metadata, risk reasons, blend configuration, feature coverage, alpha vectors, pause flags (`src/esper/tamiyo/policy.py:462-710`, `src/esper/tamiyo/service.py:964-1776`).
  - `CommandSeedOperation.parameters` carries numeric blend parameters (`alpha`, schedule bounds, method index) as a string-key map (`src/esper/tamiyo/policy.py:892-899`).
  - `FieldReport.metrics` is a free-form `map<string, float>` storing loss deltas, risk scores, hook timings (`src/esper/tamiyo/service.py:1874`).
- **Standardisation actions**:
  1. Add a structured `DecisionMetadata` message to `AdaptationCommand` capturing policy action, value estimate, risk posture, blend configuration, coverage summary, and selection context. Maintain `annotations` temporarily for backward compatibility, but gate new data through the typed struct.
  2. Extend `CommandSeedOperation` with explicit optional fields (`blend_alpha`, `blend_mode`, `schedule_start`, `schedule_end`, `resume_flag`, `include_teacher`) so Kasmina can drop string lookups. Retire use of `command.annotations` for control flags.
  3. Replace `FieldReport.metrics` with a dedicated `FieldReportDelta` message (loss delta, param delta, latency slices, risk indices). Provide a translator inside Tamiyo until consumers migrate.

  _Annotation mapping audit_
  - Policy metadata: `policy_action`, `policy_param_delta`, `policy_version`, `policy_value_estimate`, `policy_risk_index`, `policy_risk_score`, `blending_index`, `selected_seed`, `selected_blueprint` (and corresponding scores) — these become `DecisionMetadata.policy` + `DecisionMetadata.selection`.
  - Risk context: `risk_reason`, `conservative_mode` (implied), timeout markers — map to `DecisionMetadata.risk` with typed enums.
  - Blend configuration: `blending_method`, `blend_mode_source`, `alpha_vec`, `gate_k`, `gate_tau`, `alpha_lo`, `alpha_hi`, `blending_schedule_start`, `blending_schedule_end` — migrate into `DecisionMetadata.blend`.
  - `CommandSeedOperation.parameters` presently carries `alpha`, `blending_method_index`, `blending_schedule_start`, `blending_schedule_end` (`src/esper/tamiyo/policy.py:892-899`); once optional fields exist on the proto we will populate them directly and drop the map writes.
  - Coverage summary: `feature_coverage`, potential per-group entries — populate `DecisionMetadata.coverage` (and continue to export per-type telemetry via structured metrics).
  - Control flags: `resume`, `include_teacher` — move to explicit `CommandSeedOperation` optional fields.
  - Security: `signature` stays authoritative until we expose a dedicated `security.signature` field on the command; after the schema update the annotation write will be removed.
  - **Tests/fixtures**: integration/serialization tests still seed map parameters (e.g., `command.seed_operation.parameters["alpha"]`) — these will need to be updated to write the new struct fields when the proto changes (`tests/integration/test_blueprint_pipeline_integration.py`, `tests/leyline/test_serialization.py`).

  _Security flow planning_
  - Tamiyo signs commands by serialising the proto, assigning a new UUID, and storing the HMAC under `annotations["signature"]` (`src/esper/tamiyo/service.py:1768-1776`).
  - Kasmina verification removes the annotation temporarily, validates via `CommandVerifier` (HMAC + nonce replay + freshness), then reinstates the signature (`src/esper/kasmina/seed_manager.py:1476-1498`, `src/esper/kasmina/security.py:35-80`).
  - Weatherlight loads the same `SignatureContext` for downstream services (`src/esper/weatherlight/service_runner.py:138`); Oona transports signatures transparently in Redis payload metadata.
  - Plan: add `CommandSecurity` to `AdaptationCommand` (fields: `signature`, `nonce/command_id`, `issued_at`, optional `freshness_window_ms`). Tamiyo will populate this struct, Kasmina will read from it directly, and Oona/Weatherlight can inspect or log security data without mutating annotations. When implemented, annotation-based signatures disappear entirely.

#### Kasmina
- **Inbound payloads today**: consumes `AdaptationCommand` and inspects both `seed_operation.parameters` and `command.annotations` for coverage, blend, resume, and emergency semantics (`src/esper/kasmina/seed_manager.py:726-908`). Prefetch coordinator sends/receives kernel events via Oona (`src/esper/kasmina/prefetch.py:17-87`).
- **Outbound payloads**: telemetry packets per seed/global context with priority stored as an indicator string (`src/esper/kasmina/seed_manager.py:509-599`); kernel prefetch requests default training IDs to `"prototype"` (`src/esper/kasmina/prefetch.py:28`).
- **Unstructured surfaces**:
  - Relies on arbitrary annotation keys (`feature_coverage`, `risk_reason`, `resume`, `include_teacher`, `blend_mode`, `alpha_vec`) to steer lifecycle decisions (`src/esper/kasmina/seed_manager.py:801-890`).
  - Telemetry embeds seed identifiers and priority inside `system_health.indicators` (`src/esper/kasmina/seed_manager.py:517-521`).
  - Prefetch defaults highlight missing strict ID validation.
- **Standardisation actions**:
  1. Consume the proposed `DecisionMetadata`/`SeedOperation` fields instead of parsing arbitrary maps; implement strict schema validation and reject commands lacking the structured payload.
  2. When `TelemetryPacket.priority` exists, emit priority via the dedicated field and reserve `indicators` for human-readable summaries (e.g., seed id, lifecycle stage).
  3. Update kernel prefetch to require typed identifiers (`training_run_id`, `seed_id`) and remove the fallback strings once dependency guard is in place. ✅
  4. Expose explicit registry APIs: `SeedParameterRegistry.reset()` clears state, `KasminaSeedManager.reset_teacher()` releases teacher models/memory, and both helpers must flush the nonce ledger before accepting new commands to avoid replay deadlocks after administrative resets.

  _Annotation usage mapping_
  - Blend configuration: `blend_mode`, `blend_mode_source`, `alpha_vec`, `gate_k`, `gate_tau`, `alpha_lo`, `alpha_hi` → feed `_apply_blend_annotations` and populate `SeedContext.blend_config` (`src/esper/kasmina/seed_manager.py:1524-1584`).
  - Lifecycle flags: `resume`, `include_teacher` toggle pause/emergency branches (`src/esper/kasmina/seed_manager.py:815-823`).
  - Coverage/risk context: `feature_coverage`, `risk_reason`, `blueprint_risk`, `blueprint_tier`, `blueprint_stage` drive telemetry events and seed metadata (`src/esper/kasmina/seed_manager.py:845-906`).
  - Signatures: `signature` is stripped/verified prior to mutation (`src/esper/kasmina/seed_manager.py:1476-1483`).
  These will migrate to the structured `DecisionMetadata`, `CommandSeedOperation`, and future `CommandSecurity` fields so Kasmina consumes typed enums/scalars rather than string maps.
  - Security: Kasmina’s `CommandVerifier` currently expects the signature via annotations; the structured plan should surface signature + freshness metadata explicitly on the command (e.g., `AdaptationCommand.security.signature`) so verification no longer depends on mutable annotation maps (`src/esper/kasmina/security.py:35-80`).
  - Seed ID fallback: Kasmina still checks `command.seed_operation.parameters["seed_id"]` when `target_seed_id` is missing (`src/esper/kasmina/seed_manager.py:767-776`); the structured `SeedOperation` should carry `seed_id` explicitly or enforce non-empty `target_seed_id` so the fallback can be removed.

#### Weatherlight & Oona Bridge
- Weatherlight currently re-derives priority to pass into `publish_telemetry` (`src/esper/weatherlight/service_runner.py:1055-1138`). When the new `TelemetryPacket.priority` field exists, Weatherlight should forward it directly and drop `_telemetry_priority` duplication.
- `BusEnvelope` lacks a `priority` field, so emergency routing depends on call-site heuristics. Extend `BusEnvelope` with `MessagePriority priority = 4;` and propagate it inside Oona’s `_publish_proto`/`consume` logic (`contracts/leyline/leyline.proto:311`, `src/esper/oona/messaging.py:558`).

#### Leyline Contract Updates (proposed)
1. `BusEnvelope` → add `MessagePriority priority = 4;` (default `MESSAGE_PRIORITY_NORMAL`).
2. `TelemetryPacket` → add `MessagePriority priority = 50;` to formalise routing metadata.
3. `AdaptationCommand` → add `DecisionMetadata metadata = 16;` where:
   ```protobuf
   message DecisionMetadata {
     PolicyMetadata policy = 1;
     RiskMetadata risk = 2;
     BlendConfig blend = 3;
     CoverageSummary coverage = 4;
     SelectionContext selection = 5;
   }
   ```
   Each nested message uses typed fields (enums for blend mode, risk reason, etc.) instead of free-form strings.
4. `CommandSeedOperation` → add optional fields for blend alpha, schedule bounds, operating mode, resume/include-teacher flags. Mark `parameters` map as deprecated once consumers switch.
5. `FieldReport` → replace `map<string, float> metrics` with a structured `FieldReportDelta` message capturing the canonical metrics consumed by Simic.
6. `SystemStatePacket` → add an optional `TrainingMetrics` message (mirroring today’s keys) and leave the map as legacy compatibility during transition.

##### Draft Schema Snippets (planning only)
```protobuf
message BusEnvelope {
  BusMessageType message_type = 1;
  bytes payload = 2;
  map<string, string> attributes = 3; // legacy, to be pruned once structured metadata lands
  MessagePriority priority = 4;
}

message TelemetryPacket {
  string packet_id = 1;
  google.protobuf.Timestamp timestamp = 2;
  string source_subsystem = 3;
  TelemetryLevel level = 4;
  repeated MetricPoint metrics = 10;
  repeated TelemetryEvent events = 20;
  repeated TraceSpan spans = 30;
  SystemHealth system_health = 40;
  MessagePriority priority = 50; // replaces indicator-based routing
}

message AdaptationCommand {
  // existing fields...
  DecisionMetadata metadata = 16;
  message DecisionMetadata {
    PolicyMetadata policy = 1;
    RiskMetadata risk = 2;
    BlendConfig blend = 3;
    CoverageSummary coverage = 4;
    SelectionContext selection = 5;
  }
  message PolicyMetadata {
    uint32 action_id = 1;
    float value_estimate = 2;
    float risk_score = 3;
    string policy_version = 4;
  }
  message RiskMetadata {
    RiskReason reason = 1;
    bool conservative_mode = 2;
    float breaker_confidence = 3;
    enum RiskReason {
      RISK_REASON_UNSPECIFIED = 0;
      RISK_REASON_TIMEOUT_INFERENCE = 1;
      RISK_REASON_TIMEOUT_URZA = 2;
      RISK_REASON_LOSS_SPIKE = 3;
      RISK_REASON_BP_QUARANTINE = 4;
      RISK_REASON_DEVICE_PRESSURE = 5;
      RISK_REASON_OPERATOR_OVERRIDE = 6;
    }
  }
  message BlendConfig {
    BlendMode mode = 1;
    float alpha = 2;
    float schedule_start = 3;
    float schedule_end = 4;
    repeated float alpha_vec = 5; // bounded to ≤64 for proto budget
    bool logits_available = 6;
    enum BlendMode {
      BLEND_MODE_UNSPECIFIED = 0;
      BLEND_MODE_CONVEX = 1;
      BLEND_MODE_RESIDUAL = 2;
      BLEND_MODE_CHANNEL = 3;
      BLEND_MODE_CONFIDENCE = 4;
    }
  }
  message CoverageSummary {
    float average = 1;
    map<string, float> by_group = 2;
  }
  message SelectionContext {
    string seed_id = 1;
    string blueprint_id = 2;
    float seed_score = 3;
    float blueprint_score = 4;
  }
}

message CommandSeedOperation {
  SeedOperation operation = 1;
  string blueprint_id = 2;
  map<string, double> parameters = 3 [deprecated = true];
  optional double blend_alpha = 4;
  optional AdaptationCommand.BlendConfig.BlendMode blend_mode = 5;
  optional double schedule_start = 6;
  optional double schedule_end = 7;
  optional bool resume_flag = 8;
  optional bool include_teacher = 9;
}

message FieldReport {
  // existing identifier fields...
  FieldReportDelta delta = 14;
  message FieldReportDelta {
    float loss_delta = 1;
    float param_delta = 2;
    float risk_index = 3;
    float risk_score = 4;
    float tamiyo_latency_ms = 5;
    float hook_latency_ms = 6;
    float kasmina_apply_ms = 7;
    float kasmina_finalize_ms = 8;
  }
}

message SystemStatePacket {
  // existing fields...
  TrainingMetrics metrics_struct = 50;
  message TrainingMetrics {
    float optimizer_lr = 1;
    float optimizer_momentum = 2;
    float loss_delta = 3;
    float loss_ewma = 4;
    float loss_volatility = 5;
    float grad_norm_ewma = 6;
    float grad_conflict_rate = 7;
    float tamiyo_latency_ms = 8;
    float kasmina_apply_ms = 9;
    float kasmina_finalize_ms = 10;
    float step_latency_ms = 11;
    float gpu_mem_used_gb = 12;
    float gpu_mem_free_gb = 13;
    float gpu_util_percent = 14;
    float cpu_util_percent = 15;
  }
}
```

##### Oona Priority Integration Plan
- **Current behaviour**: `_publish_proto` wraps payloads in `BusEnvelope(message_type, payload)` and infers emergency routing from the optional `priority` argument passed to `publish_telemetry` (`src/esper/oona/messaging.py:244-261, 558-635`). Consumers parse the envelope and surface `envelope.payload` plus `envelope.attributes`; priority context is lost after routing.
- **With structured priority**:
  1. Set `envelope.priority = priority` before serialisation inside `_publish_proto` so downstream consumers can inspect it without relying on attributes or re-deriving from telemetry contents.
  2. Extend `OonaMessage` to carry a `priority: MessagePriority` slot populated during `_consume` (`src/esper/oona/messaging.py:400-456`), allowing handlers (Weatherlight, Kasmina, Simic) to react without decoding the envelope manually.
  3. Update Weatherlight’s telemetry flush loop to pass `message.priority` through when requeueing or routing telemetry (`src/esper/weatherlight/service_runner.py:1055-1138`).
  4. Ensure retry/requeue flows (`_requeue`, `_handle_handler_error`) preserve the priority field when cloning envelopes (`src/esper/oona/messaging.py:793-834`).
- **Verification**: add integration coverage verifying that CRITICAL telemetry published via Tolaria/Tamiyo surfaces with `priority=CRITICAL` in Oona handlers and that emergency stream routing honours the new field.

##### Helper APIs (draft interfaces)
- `esper.core.telemetry`:
  ```python
  @dataclass
  class TelemetryPayload:
      packet_id: str
      source: str
      level: leyline_pb2.TelemetryLevel
      priority: leyline_pb2.MessagePriority
      metrics: Sequence[TelemetryMetric]
      events: Sequence[TelemetryEvent]
      health: TelemetryHealth

  def build_packet(payload: TelemetryPayload) -> leyline_pb2.TelemetryPacket:
      """Populate structured fields and legacy indicators consistently."""

  async def publish(oona: OonaClient, payload: TelemetryPayload) -> None:
      """Send telemetry with correct emergency routing semantics."""
  ```
- `esper.core.decisions`:
  ```python
  @dataclass
  class DecisionContext:
      policy: PolicyMetadata
      risk: RiskMetadata
      blend: BlendConfig
      coverage: CoverageSummary
      selection: SelectionContext

  def attach_metadata(command: leyline_pb2.AdaptationCommand, ctx: DecisionContext) -> None:
      """Populate `metadata`; mirror into annotations when compatibility is required."""

  def parse_metadata(command: leyline_pb2.AdaptationCommand) -> DecisionContext:
      """Read structured fields first, fall back to annotations with warnings."""
  ```
- `esper.core.training_metrics`:
  ```python
  def pack(stats: TrainingStats) -> tuple[
      leyline_pb2.SystemStatePacket.TrainingMetrics,
      Mapping[str, float],
  ]:
      """Return the structured message and legacy map payload."""

  def unpack(packet: leyline_pb2.SystemStatePacket) -> TrainingStats:
      """Prefer structured metrics; absorb legacy keys when present."""
  ```

Helpers will emit structured telemetry for migration telemetry (e.g., `telemetry.schema.legacy_used=1`) whenever they touch deprecated maps, enabling staged enforcement.

##### FieldReport Metric Consumers (current state)
- **Producer**: Tamiyo populates `FieldReport.metrics` in `_emit_field_report` (`src/esper/tamiyo/service.py:670-741`). Baseline keys:
  - Always present: `loss_delta`.
  - From policy last-action cache: `param_delta`, `value_estimate`, `risk_index`, `risk_score`, `blending_index`, `selected_seed_score`, `blending_schedule_start`, `blending_schedule_end` (when available).
  - From `SystemStatePacket.training_metrics`: `tamiyo_latency_ms`, `hook_latency_ms` (when Tolaria attaches them).
  - From command annotations: `blueprint_risk` (optional).
  - Ad-hoc additions are possible (tests call `generate_field_report` with `{"loss": -0.05}`), so mapping needs a controlled extension story.
- **Consumer**: Simic replay buffer ingests field reports and treats `metrics` as an ordered value list (`src/esper/simic/replay.py:56-118`).
  - `loss_delta` is read explicitly to derive `reward` (`_compute_reward`) and stored on each `SimicExperience`.
  - `_populate_features` copies the first `MAX_METRIC_FEATURES` (currently 8) metric values into the feature vector and first `METRIC_SEQUENCE_LENGTH` (16) into the temporal sequence tensor, without relying on keys.
  - The remainder of Simic (trainer, unit tests) assumes at least `loss_delta`, with optional `loss` accepted for legacy scenarios (`tests/simic/test_simic_trainer.py`, `tests/tamiyo/test_service.py`).
- **Implications for struct design**:
  - The new `FieldReportDelta` must expose the explicit scalar slots above while providing a bounded, ordered list for “extra” metrics to preserve Simic’s feature stacking behaviour.
  - Migration helpers should warn when legacy keys outside the canonical set are encountered and decide whether to drop them, map them into generic extension slots, or extend the schema.
  - `MAX_METRIC_FEATURES`/`METRIC_SEQUENCE_LENGTH` should inform the size of any repeated field to avoid silently truncating important signals.
  - `FieldReport` persistence uses protobuf binary records (`field_reports.log`) with no additional JSON schema (`src/esper/tamiyo/persistence.py:10-122`); changing the proto structure simply updates the serialized payload. Scripts/tests call `TamiyoService.generate_field_report` rather than parsing the log, so no extra consumers need migration.

##### Planned Simic/Tamiyo Refactor (no adapters, prototype scope)
- **Proto change prerequisite**: Once `FieldReport.delta` replaces the metrics map, Simic must consume the structured fields directly.
- **Simic replay buffer updates** (`src/esper/simic/replay.py`):
  1. Read `report.delta.loss_delta`, `risk_score`, etc., instead of `metrics.get(...)` when constructing `SimicExperience`.
  2. Populate `features_numeric` / `metric_sequence` from a new `delta.extra_metrics` repeated field (bounded to 8/16 entries) and eliminate generic map iteration.
  3. Remove assumptions about arbitrary map ordering; enforce deterministic stacking so PPO inputs stay stable.
- **Simic trainer/tests** (`tests/simic/test_replay.py`, `test_simic_trainer.py`):
  - Update fixtures to set structured fields; drop legacy `report.metrics[...]` assignments.
  - Assert that `SimicExperience.loss_delta` and rewards reflect the new struct.
- **Tamiyo service updates** (`src/esper/tamiyo/service.py`):
  - Write into `FieldReport.delta` (and optional `delta.extra_metrics`) instead of mutating the map.
  - Prune ad-hoc metric keys; decide which values to keep as formal scalar fields, drop or relocate others.
  - Adjust field-report tests to verify struct population (`tests/tamiyo/test_service.py`).
- **Failure policy**: because we’re pre-1.0 prototype, we will delete the map writes in the same PR and expect downstream consumers to handle the new struct immediately.
- **Follow-up observability**: add a unit/integration test ensuring structured reports reach Simic via ReplayBuffer `ingest_from_oona` to validate Oona serialization after the proto change.
- **Telemetry downstream (Nissa)**: Coverage metrics emitted by Tamiyo telemetry (`tamiyo.gnn.feature_coverage*`) are already keyed by structured metric names; after `DecisionMetadata` lands, Tamiyo should source these gauges from the new coverage struct to keep Nissa ingestion unchanged while eliminating dependence on ad-hoc annotations (`src/esper/nissa/observability.py:232-270`).

##### Training Metrics Map Consumers
- **Producer**: Tolaria writes a rich set of keys into `SystemStatePacket.training_metrics` during step emission (`src/esper/tolaria/trainer.py:1128-1639,1666-1705`). Keys include optimizer stats (`optimizer_lr`, `optimizer_momentum`, `optimizer_family_index`), loss/gradient aggregates (`loss_delta`, `loss_ewma`, `loss_volatility`, `grad_norm_ewma`, `grad_conflict_rate`, `grad_var`), latency measurements (`input_wait_ms`, `h2d_copy_ms`, `tamiyo_latency_ms`, `kasmina.apply_ms`, `kasmina.finalize_ms`, `step_latency_ms`, `hook_latency_ms`), throughput (`samples_per_s`), and device telemetry (`gpu_mem_used_gb`, `gpu_mem_free_gb`, `gpu_util_percent`, `cpu_util_percent`).
- **Consumers**:
  - Tamiyo’s graph builder normalises a subset when building global features (`src/esper/tamiyo/graph_builder.py:300-356`), specifically looking for `loss`, `validation_loss`, `training_loss`, `gradient_norm`, `samples_per_s`, `hook_latency_ms`, `epochs_total`, and optionally `optimizer_family_index`.
  - Tamiyo `_emit_field_report` reuses `tamiyo_latency_ms` and `hook_latency_ms` when emitting field-report deltas (`src/esper/tamiyo/service.py:698-704`).
  - Kasmina does not read the map today; it relies on annotations/telemetry instead.
- **Struct plan**: the proposed `SystemStatePacket.TrainingMetrics` message (above) mirrors the keys the consumers require. When we switch, Tolaria will populate the structured fields instead of the free-form map, Tamiyo will read from the struct, and remaining metrics can be expressed via a bounded `extra_metrics` repeated field if we need to retain long-tail instrumentation.

##### Telemetry & Alert Consumers
- **Nissa ingestion** (`src/esper/nissa/observability.py:232-296`): expects the following Tamiyo metrics/events for downstream gauges/alerts:
  - Coverage gauges: `tamiyo.gnn.feature_coverage` plus per-type metrics prefixed with `tamiyo.gnn.coverage.*` or `tamiyo.gnn.feature_coverage.*`.
  - Blueprint risk gauge: `tamiyo.blueprint.risk`, used to derive `tamiyo.bsds.elevated_risk_flag`.
  - BSDS hazard events converted into metrics: `tamiyo.bsds.hazard_critical_signal`, `tamiyo.bsds.hazard_high_signal` (built from telemetry events `bsds_hazard_*`).
- **Alert engine** (`src/esper/nissa/alerts.py:128-180`): rules fire on `tamiyo.gnn.feature_coverage`, `tamiyo.bsds.hazard_*`, and `tamiyo.bsds.elevated_risk_flag`.
- **Docs / operator runbooks** (`docs/architecture_summary.md`, `docs/project/operator_runbook.md`, `docs/project/observability_runbook.md`) enumerate canonical Tamiyo metrics: `tamiyo.validation_loss`, `tamiyo.loss_delta`, `tamiyo.conservative_mode`, `tamiyo.blueprint.risk`, plus the BSDS coverage gauges.
- **Implication**: when migrating to structured telemetry/metadata, ensure these metric names continue to be emitted exactly as-is (even if sourced from the new `DecisionMetadata` or coverage structs). Any additional telemetry derived from annotations (e.g., degraded-input events) must persist through the new representation so existing dashboards/alerts stay green.
- **Grafana dashboards** (`infra/grafana/dashboards/nissa_overview.json`) visualise `esper_field_reports_total` (summed and per-outcome rate); field-report ingestion must remain intact so these charts and Prometheus counters continue to reflect reality.

##### Integration Touchpoints (Tolaria ↔ Tamiyo ↔ Kasmina ↔ externals)
- **Tolaria → Tamiyo**: synchronous calls into `TamiyoService.evaluate_step/evaluate_epoch` produce `AdaptationCommand` and latency metrics; RC1 changes affect the `AdaptationCommand` schema and telemetry priority, so both sides must adopt the structured metadata simultaneously (`src/esper/tolaria/trainer.py:2048-2145`, `src/esper/tamiyo/service.py:557-742`).
- **Tolaria → Kasmina**: `KasminaSeedManager.apply_command` consumes Tamiyo’s command inside the training loop; enforcing structured command fields (seed/blend/security) is mandatory to keep Kasmina gate logic strict (`src/esper/kasmina/seed_manager.py:720-919`).
- **Tamiyo ↔ Urza**: blueprint metadata fetch via `_urza.get` powers risk/blend decisions and telemetry (`src/esper/tamiyo/service.py:1025-1104`); no schema change, but strict dependency guard must ensure Urza is available at startup.
- **Tamiyo → Oona/Nissa**: telemetry/field reports published via Oona (`src/esper/tamiyo/service.py:1938-2078`) feed Nissa dashboards/alerts. Priority routing, structured telemetry, and `FieldReportDelta` must remain compatible with Weatherlight/Nissa ingestion once updated.
- **Kasmina ↔ Urza/Oona**: Kasmina fetches kernels from Urza and emits telemetry via Weatherlight’s Oona client (`src/esper/kasmina/seed_manager.py:1688-1739`, `src/esper/weatherlight/service_runner.py:1055-1100`). Structured command security and priority propagation ensure Kasmina’s verifier + telemetry loop continue to operate correctly.
- **Weatherlight orchestration**: Weatherlight mediates telemetry, policy updates, and command routing. After adding `TelemetryPacket.priority`/`BusEnvelope.priority`, Weatherlight’s loops must forward the structured priority without re-deriving (`src/esper/weatherlight/service_runner.py:618-1138`).
- **Simic replay/trainer**: consumes field reports via Oona and Tamiyo WAL (`src/esper/simic/replay.py:56-118`, `scripts/run_demo.py:140-214`). The `FieldReportDelta` switch and WAL compatibility plan avoid surprises for offline training.
- **Observability stack**: Prometheus/Grafana rely on Nissa exposing counters derived from telemetry + field reports; the structured telemetry work must keep metric names stable to prevent dashboard regressions (`docs/project/observability_runbook.md`, `infra/grafana/dashboards`).

#### Migration Approach
- Implement proto extensions with backwards-compatible field numbers, regenerate bindings, and introduce struct-based helper classes in `esper.core.contracts` to hydrate/extract the new messages.
- Add validation gates: once both producer and consumer understand the structured fields, log/telemetry warnings when legacy maps are used; eventually flip to hard failures per strict-dependency policy.
- Update subsystem telemetry publishers to rely on the shared helper so routing priority and indicator usage stay consistent across Tolaria, Tamiyo, Kasmina, and Weatherlight.

## Async Worker Plan
- Implement `esper.core.async_runner.AsyncWorker` with:
  - Worker pool (size configurable) using `asyncio` + cancellation tokens.
  - Submit API returning cancellable futures with timeout handling.
  - Metrics hooks (tasks scheduled, cancelled, completed, latency).
- Adoption steps:
  1. Integrate worker into Tolaria WP-T2 (Tamiyo/Kasmina calls).
  2. Replace Tamiyo per-call executors (WP-A1) and Kasmina prefetch loops (WP-K4).
  3. Provide synchronous adapter for contexts without running loop.
- Risks: ensure cancellation stops underlying coroutine; guard exceptions.
- **Implementation status**: Tolaria trainer and Tamiyo service now share `AsyncWorker` instances for timeout handling, and `KasminaPrefetchCoordinator` falls back to the worker when no loop is running (Weatherlight + demo wire a single worker across subsystems). Legacy executor fallbacks were removed to enforce immediate adoption.

## Telemetry Router Plan
- Define priority mapping:
  - CRITICAL events → emergency stream.
  - WARNING/ERROR events → high-priority normal stream.
  - INFO → normal stream.
- Weatherlight integration: route based on `TelemetryPacket.system_health.indicators["priority"]`.
- Provide helper for subsystems to set priority consistent with risk engine.
- Verification: telemetry tests after WP-A3/WP-K3 plus automated load harness (`tests/integration/test_telemetry_emergency_load.py`) and CLI (`scripts/run_telemetry_routing_load.py`).

### 2025-09-27 Update
- Weatherlight now tracks publish counts per priority and failed submissions, surfaced via `telemetry_priority_counters()` for observability.
- `OonaClient.metrics_snapshot` exposes emergency token-bucket state (`emergency_tokens_remaining`, `emergency_refill_rate_per_s`, bucket capacity) so automated tests can assert routing health.
- `EsperSettings` gained `OONA_EMERGENCY_MAX_PER_MIN` and `OONA_EMERGENCY_THRESHOLD` switches to tune throttling without a code change.
- Load harness covers burst limits and backlog control with `FakeRedis` (`tests/integration/test_telemetry_emergency_load.py`); `scripts/run_telemetry_routing_load.py` drives manual experiments against live Redis deployments.
- Operators can trim telemetry queues when Weatherlight is offline via `scripts/drain_telemetry_streams.py`, which issues Redis `XTRIM` calls to reset `oona.normal` / `oona.emergency` and logs resulting depths.
- Tamiyo blend annotations now carry the policy-selected mode plus confidence gate parameters; Kasmina stores the metadata, enforces logits availability, and emits `confidence_gate_missing_logits` telemetry when Tamiyo requests confidence mode without sufficient logits.

## Strict Dependency Guard
- Create `esper.core.dependency_guard` with checks:
  - Validate IDs non-empty.
  - Prevent fallback kernel usage unless explicitly flagged.
  - Ensure training run IDs provided for prefetch.
- Usage: Tamiyo command parsing, Kasmina seed ops, Tolaria rollback/profiler.

## Shared Config Schema
- Define dataclass `ExecutionConfig` with sections:
  - Async worker settings.
  - PT2.8 toggles (matmul precision, inference mode enforcement).
  - Dependency policies (fallback allowed bools).
  - Telemetry options.
- Each subsystem consumes `ExecutionConfig` to remove scattered `EsperSettings` lookups.

## Telemetry Schema
- Document common metrics/events: timeout, gate failure, blend telemetry, command rejection.
- Provide helper to attach coverage map/types, command verification results, fallback flags.
- **Command verifier routing**: Kasmina (and any consumer of the shared verifier) MUST emit `TelemetryEvent(description="command_rejected")` with `TelemetryLevel.CRITICAL` whenever `reason` is one of `{missing_signature, invalid_signature, nonce_replayed, missing_timestamp, stale_command}`. Less severe reasons (e.g., `verifier_unavailable`) stay at `TelemetryLevel.ERROR`. The finaliser must set `system_health.indicators["priority"]` (or the future `TelemetryPacket.priority` field) to `MESSAGE_PRIORITY_CRITICAL` so Weatherlight/Oona route failures to the emergency stream.
- **Verifier metrics**: publish `kasmina.command_verifier.rejections_total{reason="..."}`, `kasmina.command_verifier.accepted_total`, and `kasmina.command_verifier.validation_latency_ms`. Nonce lifecycle is surfaced via gauges/counters `kasmina.nonce_ledger.size`, `kasmina.nonce_ledger.evictions_total`, and `kasmina.nonce_ledger.ttl_seconds` (constant but exported for runbook parity). Metrics live under the shared telemetry helper to keep naming aligned across subsystems once Tamiyo adopts structured `CommandSecurity` fields.
- **Prefetch metrics**: introduce `kasmina.prefetch.requests_total{status="scheduled|ready|error|canceled"}`, `kasmina.prefetch.inflight`, and `kasmina.prefetch.latency_ms` (time from request → ready/error). Emit telemetry events on cancellation/escalation: `prefetch_canceled` (WARNING), `prefetch_timeout` (CRITICAL), and enrich existing `prefetch_ready`/`prefetch_error` packets with latency + queue depth attributes.

## Nonce Ledger Strategy
- Ledger TTL defaults to 300 seconds; expose it through configuration and telemetry. Services must invoke a cleanup sweep at least once per control-loop iteration (Kasmina via `finalize_step`, Tamiyo via evaluation loop) to prevent unbounded growth during idle periods.
- Maintain a hard cap (default 10,000 entries). When the cap is exceeded, evict the oldest entries and emit a `TelemetryEvent(description="nonce_ledger_truncated", level=TelemetryLevel.WARNING, attributes={"size": str(size)})` alongside incrementing `kasmina.nonce_ledger.evictions_total`.
- Registry resets or teacher swaps must clear the ledger to avoid blocking replays after administrative actions; provide explicit hooks rather than relying on process restarts.

## Prefetch & Cache Reliability Plan
- **Async execution**: Prefetch publishing and result consumers should run on the shared `AsyncWorker` so shutdown is deterministic. Coordinators must expose `start()`/`close()` hooks that submit/await worker tasks, and `KasminaSeedManager.finalize_step` should poll for task failures via `poll_task_issue()`.
- **Cancellation policy**: Track per-request deadlines (based on Tamiyo/Kasmina latency budget) and cancel inflight tasks when exceeded. Emit `prefetch_timeout` (CRITICAL) with attributes `{seed_id, blueprint_id, request_id}` and increment `kasmina.prefetch.requests_total{status="canceled"}`.
- **Cache locking**: Introduce a lightweight lock (per blueprint) around GPU cache attach/evict to prevent concurrent writes. Lock contention should surface via `kasmina.cache.lock_wait_ms` and a WARNING `cache_lock_contention` event if wait exceeds 100 ms.
- **Shutdown hygiene**: On `reset_registry`, `reset_teacher_model`, or service shutdown, cancel outstanding prefetch requests, evict cache entries, and emit a single INFO `prefetch_cleanup` event summarising counts removed. Coordinators must join worker tasks within the AsyncWorker timeout to avoid test hangs.

## Timeline & Integration
1. Finalise Async Worker & Config schema (before WP-T2).
2. Weatherlight telemetry routing update.
3. Subsystem adoption as per work packages (Tolaria → Tamiyo → Kasmina).
4. Update documentation & tests.

## Risks & Mitigations
- Async worker needs robust cancellation; include stress tests.
- Telemetry routing changes require Weatherlight deploy; coordinate with ops.
- Config schema adoption may require staged rollout; maintain backward compatibility until all modules switched.

## Deliverables
- `esper/core/async_runner.py` + tests.
- `esper/core/dependency_guard.py` + tests.
- Updated Weatherlight supervisor for priority routing.
- Documentation in `README` + subsystem config updates.
