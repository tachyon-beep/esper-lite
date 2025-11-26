# Leyline Contract Register

## Purpose

This register inventories enums and dataclasses that exist outside the generated Leyline protobuf bindings. It will expand subsystem by subsystem to make hidden cross-coupling visible during the prototype delta review.

See also `docs/design/decisions/ADR-006-leyline-shared-contract-consolidation.md` for the consolidation plan.
## src/esper/core

**Dataclasses**

- `TelemetryMetric` (`src/esper/core/telemetry.py:16`) — Represents a single metric sample added to Leyline telemetry packets; captures name, value, unit, and optional attributes.
- `TelemetryEvent` (`src/esper/core/telemetry.py:26`) — Describes discrete telemetry events with a Leyline `TelemetryLevel`, attributes, and optional event identifier.

**Enums**

- _None defined._

**Notes**

- `EsperSettings` (`src/esper/core/config.py:15`) derives from `pydantic.BaseSettings` rather than `dataclass`, so it is excluded from this register.
- Both dataclasses directly wrap `leyline_pb2.TelemetryPacket` primitives and should eventually be replaced by pure protobuf usage once helper ergonomics are addressed.
- Recommendation: adopt the forthcoming Leyline telemetry builder (ADR-006) and retire these helpers to cut per-packet allocations.

## src/esper/karn

**Dataclasses**

- `BlueprintQuery` (`src/esper/karn/catalog.py:22`) — Front-end request payload for Karn blueprint lookups; captures target blueprint/tier plus parameter overrides, context hashing, and conservative/adversarial flags.
- `KarnSelection` (`src/esper/karn/catalog.py:34`) — Response wrapper returned from catalog queries; carries the chosen `BlueprintDescriptor`, resolved parameters, conservative flag, and resulting tier.

**Enums**

- _None defined._ (Karn re-exports Leyline `BlueprintTier` and `BlueprintDescriptor` enums/messages instead.)

**Notes**

- `BlueprintDescriptor`, `BlueprintTier`, and `BlueprintParameterBounds` in Karn are direct aliases to `leyline_pb2` message/enum types.
- The catalog’s breaker state/metrics lean on shared telemetry helpers; any bespoke dataclasses beyond the two listed above would violate the Leyline-only contract boundary.

## src/esper/kasmina

**Dataclasses**

- `VerificationResult` (`src/esper/kasmina/security.py:15`) — Wraps the result of adaptation-command verification, including acceptance flag and rejection reason.
- `GateInputs` (`src/esper/kasmina/gates.py:13`) — Collects lifecycle gate evaluation context such as blueprint metadata, isolation counts, latency stats, and interface health.
- `GateResult` (`src/esper/kasmina/gates.py:32`) — Encapsulates a gate decision with pass/fail flag, reason string, and optional telemetry attributes.
- `RegistrationRecord` (`src/esper/kasmina/registry.py:12`) — Tracks parameter identifiers owned by a particular seed for update validation.
- `IsolationStats` (`src/esper/kasmina/isolation.py:13`) — Summarises gradient norms and dot product used to check host/seed isolation.
- `KernelCacheStats` (`src/esper/kasmina/kernel_cache.py:16`) — Reports Kasmina GPU cache size, capacity, hit rate, and eviction count.
- `LifecycleTransition` (`src/esper/kasmina/lifecycle.py:19`) — Records a lifecycle state transition attempt (current and next Leyline stage ids).
- `SeedContext` (`src/esper/kasmina/seed_manager.py:89`) — Per-seed state bundle covering lifecycle, gate outcomes, kernel attachment, alpha schedule, pending telemetry, and priority metadata.
- `AlphaSchedule` (`src/esper/kasmina/blending.py:11`) — Maintains blending steps/temperature and computes alpha progression.
- `CacheStats` (`src/esper/kasmina/memory.py:13`) — Surface TTL cache occupancy, hit rate, and eviction count for memory governance.
- `KasminaMemoryManager` (`src/esper/kasmina/memory.py:70`) — Aggregates Kasmina caches with GC bookkeeping and clean-up helpers.
- `BreakerEvent` (`src/esper/kasmina/safety.py:17`) — Describes a circuit-breaker action including resulting state and action keyword.
- `BreakerSnapshot` (`src/esper/kasmina/safety.py:26`) — Snapshot of circuit-breaker counters and open-until timestamp for telemetry/export.

**Enums**

- _None defined._ (Kasmina relies exclusively on Leyline enums such as `SeedLifecycleStage`, `CircuitState`, and `TelemetryLevel`.)

**Notes**

- Several Kasmina helpers alias Leyline message types directly (e.g., `SeedStage`, `BlueprintDescriptor`) rather than defining local dataclasses or enums.
- Persistence of these helpers in the shared package underscores where Kasmina still hosts subsystem-specific state outside the canonical Leyline protobufs; future deltas should migrate telemetry payload builders to pure Leyline message usage.

## src/esper/nissa

**Dataclasses**

- `NissaIngestorConfig` (`src/esper/nissa/observability.py:26`) — Captures Prometheus gateway and Elasticsearch endpoints used to initialise the ingestor.
- `AlertRule` (`src/esper/nissa/alerts.py:22`) — Defines an alerting rule (metric, comparator, threshold, repeat count, routing targets).
- `AlertEvent` (`src/esper/nissa/alerts.py:32`) — Represents a fired alert with metric value, originating source, and routing destinations.
- `_AlertState` (`src/esper/nissa/alerts.py:80`) — Internal counter/state holder tracking consecutive breaches and active state per rule.
- `SLOSample` (`src/esper/nissa/slo.py:11`) — Stores an SLO measurement with objective, actual, timestamp, and helper to check compliance.
- `SLOStatus` (`src/esper/nissa/slo.py:23`) — Aggregated SLO health summary including burn rate for a metric.
- `SLOConfig` (`src/esper/nissa/slo.py:31`) — Configures SLO window length and burn alert threshold for the tracker.

**Enums**

- _None defined._ (Nissa consumes Leyline enums directly from telemetry/system-state messages.)

**Notes**

- Alert comparator dispatch (`COMPARATORS`) hangs off simple callables; introducing Enum wrappers here would centralise comparator names but is currently avoided to keep helpers light.
- Nissa’s primary data structures still rely on Leyline protobuf payloads (`TelemetryPacket`, `SystemStatePacket`, `FieldReport`); the listed dataclasses support local control logic around ingestion and alerting.

## src/esper/oona

**Dataclasses**

- `StreamConfig` (`src/esper/oona/messaging.py:19`) — Configuration bundle for Redis stream names, consumer group metadata, retry behaviour, TTLs, and kernel freshness windows.
- `OonaMessage` (`src/esper/oona/messaging.py:44`) — Handler-facing container exposing stream, message id, typed `BusMessageType`, payload bytes, and optional attributes map.
- `BreakerSnapshot` (`src/esper/oona/messaging.py:55`) — Captures circuit-breaker counters/state for telemetry and decision logic within the client.

**Enums**

- _None defined._ (Oona references Leyline Bus/Circuit enums directly.)

**Notes**

- Oona’s `CircuitBreaker` mirrors Kasmina safety semantics but keeps its own implementation; consider consolidating via Leyline abstractions during delta hardening.
- Signature context defaults to `DEFAULT_SECRET_ENV`; register should cross-link with security package once inventoried.
- Recommendation: move stream/config/breaker helpers into Leyline BusConfig per ADR-006 so subsystems consume protobuf envelopes directly.

## src/esper/security

**Dataclasses**

- `SignatureContext` (`src/esper/security/signing.py:12`) — Encapsulates Leyline signing secret material and HTTP header metadata; constructed from environment variables for HMAC operations.

**Enums**

- _None defined._

**Notes**

- `DEFAULT_SECRET_ENV` is a string constant pointing at the shared Leyline secret; multiple subsystems (Kasmina, Oona, Weatherlight) rely on this helper for signature enforcement.
- Signing helpers currently only support static HMAC; prototype delta may expand to asymmetric keys per detailed design.
- Recommendation: relocate `SignatureContext` and signing APIs under Leyline security per ADR-006 and cache context creation to avoid redundant env reads.

## src/esper/simic

**Dataclasses**

- `SimicExperience` (`src/esper/simic/replay.py:31`) — Processed field-report sample carrying reward/loss metrics, encoded features, indices, and raw Leyline report for PPO training.
- `FieldReportReplayBuffer` (`src/esper/simic/replay.py:91`) — FIFO buffer maintaining processed experiences, timestamps, and configuration for sampling/TTL enforcement.
- `SimicTrainerConfig` (`src/esper/simic/trainer.py:30`) — Parameterises the PPO trainer (hyperparameters, embedding dims, validation thresholds, etc.).
- `ValidationConfig` (`src/esper/simic/validation.py:15`) — Thresholds for policy validation metrics.
- `ValidationResult` (`src/esper/simic/validation.py:26`) — Captures pass/fail state, failure reasons, and metric snapshot from validation.
- `EmbeddingRegistryConfig` (`src/esper/simic/registry.py:11`) — Configures persistent embedding registries (path, max size).

**Enums**

- _None defined._

**Notes**

- `EmbeddingRegistry` persists mapping tables to `Path`s; ensure future refactors consider Leyline-based identifier services to reduce bespoke state.
- `SimicExperience.outcome` stores the Leyline enum name string rather than numeric value, which may require alignment with protobuf schema conventions later.
- Recommendation: migrate embedding vocab persistence to a Leyline `EmbeddingDictionary` store (ADR-006) so Simic and Tamiyo share a single binary snapshot.

## src/esper/tamiyo

**Dataclasses**

- `FieldReportStoreConfig` (`src/esper/tamiyo/persistence.py:23`) — Configures Tamiyo’s field-report WAL path and retention window.
- `TamiyoPolicyConfig` (`src/esper/tamiyo/policy.py:29`) — Defines the GNN policy’s registries, embedding dims, architecture toggles, and blending methods.
- `TamiyoGraphBuilderConfig` (`src/esper/tamiyo/graph_builder.py:108`) — Controls graph construction limits (vocab sizes, feature dims, metadata providers) for policy inference.
- `TamiyoGNNConfig` (`src/esper/tamiyo/gnn.py:26`) — Hyperparameters for the heterogenous GNN layers/heads used in Tamiyo.
- `TamiyoServiceConfig` (`src/esper/tamiyo/service.py:35`) — Runtime configuration for the Tamiyo service (buffer sizes, WAL paths, conservative mode, thresholds).

**Enums**

- _None defined._

**Notes**

- Tamiyo policy/service classes tightly integrate with Simic embedding registries and Leyline `SystemStatePacket`/`AdaptationCommand`; documenting these configs highlights coupling points that should eventually align with Leyline schema metadata.
- `TamiyoPolicyConfig.blending_methods` enumerates strings; consider migrating to Leyline enum types once defined.

## src/esper/tezzeret

**Dataclasses**

- `CompileJobConfig` (`src/esper/tezzeret/compiler.py:23`) — Configures Tezzeret compile jobs (artifact directory, retries, WAL, inductor cache path).
- `CompilationResult` (`src/esper/tezzeret/compiler.py:91`) — Captures metadata from a single compilation run (guard spec, timing, strategy).
- `CompilerMetrics` (`src/esper/tezzeret/compiler.py:106`) — Tracks aggregate compiler counters and durations by strategy.
- `ForgeMetrics` (`src/esper/tezzeret/runner.py:28`) — Telemetry snapshot for the Tezzeret forge (breaker state, job counts, last error).
- `CompilationJob` (`src/esper/tezzeret/runner.py:41`) — Lightweight representation of a pending blueprint compile request.

**Enums**

- _None defined._

**Notes**

- Tezzeret shares blueprint metadata with Karn/Urza; these dataclasses mainly support orchestration and should eventually draw more strongly on Leyline kernel contracts (e.g., guard specs).

## src/esper/tolaria

**Dataclasses**

- `ConstantSchedule` (`src/esper/tolaria/lr_controller.py:18`), `CosineSchedule` (`:24`), `StepSchedule` (`:36`), `WarmupWrapper` (`:46`) — LR controller primitives implementing prototype-friendly schedules and warmup.
- `Escalation` (`src/esper/tolaria/emergency.py:30`) — Records emergency level changes, reason, broadcast flag, and latency.
- `Snapshot` (`src/esper/tolaria/rollback.py:29`) — Encapsulates fast-tier rollback payload metadata (step, byte size, serialized state).
- `RollbackResult` (`src/esper/tolaria/rollback.py:182`) — Summarises rollback attempt outcome (fast-path used, latency, cache hit).
- `OptimizationConfig`? (none) — No other dataclasses in optimizer manager; double-checked.
- `AggregationResult` (`src/esper/tolaria/aggregation.py:16`) — Tracks PCGrad conflict projections during gradient aggregation.
- `TrainingLoopConfig` (`src/esper/tolaria/trainer.py:54`) — Governs training parameters (device, budgets, breaker thresholds, AMP toggles).
- `EpochStats` (`src/esper/tolaria/trainer.py:76`) — Aggregates per-epoch stats (loss sums, accuracy, gradient norms, durations).

**Enums**

- _None defined._ (Emergency levels intentionally use integer constants to avoid shadow enums.)

**Notes**

- Emergency levels remain ad-hoc constants (`Level` class) pending a Leyline schema; convert to protobuf enum once available.
- Rollback helpers include shared-memory signals; they encapsulate state outside Leyline and should be reviewed before cross-subsystem integration.
- LR schedules and aggregation results may become Leyline-controlled metadata in future designs; for now they’re local dataclasses providing structure.

## src/esper/urza

**Dataclasses**

- `UrzaRecord` (`src/esper/urza/library.py:25`) — Represents a stored blueprint artifact and metadata (guard digests, prewarm samples, checksums, extras).
- `BlueprintRequest` (`src/esper/urza/pipeline.py:14`) — Pipeline input describing blueprint id, parameters, and training run context.
- `BlueprintResponse` (`src/esper/urza/pipeline.py:21`) — Pipeline output returning descriptor metadata, artifact path, and optional catalog update.
- `PrefetchMetrics` (`src/esper/urza/prefetch.py:18`) — Collects hit/miss/error counters and latency for kernel prefetch worker.

**Enums**

- _None defined._

**Notes**

- Urza’s record structure duplicates fields found in Leyline `KernelCatalogUpdate`; prototype clean-up should consolidate to the canonical message.
- Prefetch metrics track counters locally; consider migrating to Leyline telemetry packets directly to avoid bespoke dataclasses.
- Recommendation: adopt Leyline `KernelCatalogEntry` messages for storage/prefetch per ADR-006 to remove JSON duplication and speed lookups.

## src/esper/weatherlight

**Dataclasses**

- `WorkerState` (`src/esper/weatherlight/service_runner.py:39`) — Tracks supervisor worker metadata (coroutine factory, task handle, restart/backoff counters, last error info).

**Enums**

- _None defined._

**Notes**

- Weatherlight aggregates telemetry by composing subsystem helpers; no additional contract types beyond `WorkerState` are defined locally.
