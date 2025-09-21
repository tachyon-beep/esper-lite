# Leyline Combined Design

---
File: docs/design/detailed_design/00-leyline-shared-contracts.md
---
# Leyline Shared Contracts

## Metadata
- Version: 1.0 (virtual subsystem, compiled into every build)
- Status: Design complete; enforcement delivered through CI/CD and build tooling
- Ownership: Data Architect (contract authority) with System Architect + Integration Specialist reviewers

## Purpose
Leyline is the canonical library of cross-subsystem data contracts for Esper-Lite. It has no runtime footprint; instead it distributes Protocol Buffer schemas, enums, and constants that every subsystem imports during compilation. The goal is to guarantee consistent serialization, keep the training loop free from schema drift, and uphold tight latency budgets (<80 µs for `SystemStatePacket`).

## Architecture Essentials
- **Canonical Source**: `.proto` files define all payloads, enums, and constants under the `leyline.*` namespace.
- **Distribution Model**: Contracts ship alongside subsystem builds; CI rejects mismatched versions.
- **Governance Hooks**: Schema changes flow through the Leyline governance workflow (see `00.3-leyline-governance-implementation.md`).
- **Performance Guardrails**: Serialization, packet size, and GC-allocation caps are encoded as constants that downstream systems must honour.

## Contract Inventory

| Domain | Key Artifacts | Notes |
| --- | --- | --- |
| Training State | `SystemStatePacket`, `SeedState`, `HardwareContext` | Broadcast by Tolaria; consumed by Kasmina, Tamiyo, Simic, others. Native maps keep allocations ≤4 per message. |
| Control Plane | `AdaptationCommand` (+ `SeedOperation`, `CommandType` enums) | Tamiyo-originated commands with execution constraints and metadata bundles. |
| Observability | `EventEnvelope`, `TelemetryPacket`, tracing structs | Oona and Nissa use these wrappers for bus routing and telemetry fan-out. |
| Learning Feedback | `FieldReport`, `MitigationAction` | Tamiyo publishes adaptation outcomes; Simic consumes them for offline policy training. |
| Constants & Limits | `PerformanceBudgets`, `MemoryBudgets`, `SystemLimits` | Budgets for epoch work, rollback timing, memory ratios, retry ceilings, etc. |

Detailed schemas and enumerations reside in:
- `00.1-leyline-message-contracts.md`
- `00.2-leyline-enums-constants.md`

## Integration Summary

| Consumer | Contracts Pulled | Critical Usage |
| --- | --- | --- |
| Tolaria | `SystemStatePacket`, `PerformanceBudgets` | Publishes training state; enforces epoch timing envelopes. |
| Kasmina | `SystemStatePacket`, checkpoint metadata, `SystemLimits` | Applies control commands; validates kernel injection quotas. |
| Tamiyo | `SystemStatePacket`, `AdaptationCommand`, `FieldReport`, command enums | Makes inference decisions, publishes outcomes, and issues germination directives. |
| Simic | `FieldReport`, telemetry reports | Trains policies offline; relies on provenance fields for replay. |
| Oona / Nissa | `EventEnvelope`, `TelemetryPacket`, delivery enums | Bus routing, telemetry ingestion, SLO enforcement. |

## Performance & Configuration

| Metric / Setting | Target | Enforcement |
| --- | --- | --- |
| `SystemStatePacket` serialization | <80 µs, <280 B | Benchmark scripts under `tests/leyline/contracts`. |
| `AdaptationCommand` serialization | <40 µs | Same harness; alerts pipe to CI. |
| GC allocations / message | ≤4 | Map-based fields validated in integration tests. |
| Schema version | 1 (strict) | `SchemaVersion` helper validates at runtime. |
| Governance approvals | ≥2 | CI enforces `min_approvals`; failures block merge. |

YAML configuration stub (shared by subsystems):
```yaml
leyline:
  schema_version: 1
  max_message_size_bytes: 280
  max_serialization_us: 80
  owner: data-architect
  change_approval_required: 3
```

## Operational Notes
- **Health Checks**: None (virtual component). Success criteria measured by "contract parity" metrics emitted during subsystem startup.
- **Failure Modes**: Build-time schema mismatch, runtime version mismatch, or serialization exceptions. Each triggers circuit breakers defined per subsystem.
- **Security**: Change control is the primary guard. Message payloads may contain sensitive training data, so downstream services must handle transport encryption.

## Evolution Roadmap
1. **Schema Evolution (Phase 2)**: Introduce backward-compatible versioning once multi-version deployments begin.
2. **Dynamic Discovery (Phase 3)**: Optional runtime negotiation for heterogeneous clusters; blocked until after Phase 2 stabilises.

## References
- `docs/design/detailed_design/00.1-leyline-message-contracts.md`
- `docs/design/detailed_design/00.2-leyline-enums-constants.md`
- `docs/design/detailed_design/00.3-leyline-governance-implementation.md`
- Esper HLD (`docs/design/HLD.md`)

---
File: docs/design/detailed_design/00.1-leyline-message-contracts.md
---
# Leyline Message Contracts

## Scope & Status
- Version 1.0 (production)
- Implements Protocol Buffer schemas consumed by every Esper-Lite subsystem
- Owned by Data Architect; changes must pass Leyline governance review

## Contract Groups

| Group | Messages | Purpose |
| --- | --- | --- |
| Core Training State | `SystemStatePacket`, `SeedState`, `HardwareContext` | Tolaria broadcasts consolidated training telemetry to control and execution planes. |
| Control Plane | `AdaptationCommand` with `SeedOperation` / `CommandType` enums | Tamiyo issues germination, rollback, and safety actions executed by Kasmina. |
| Learning Feedback | `FieldReport`, `MitigationAction` | Summarises adaptation outcomes that Simic replays offline. |
| Observability & Bus | `EventEnvelope`, `TelemetryPacket`, telemetry primitives | Oona wraps any payload for routing; Nissa consumes telemetry streams. |
| Checkpoint Metadata | `CheckpointMetadata` and subordinate detail blocks | Kasmina ↔ Tolaria checkpoint coordination. |

## Key Schemas

```protobuf
message SystemStatePacket {
  // Versioning & timing
  uint32 version = 1;
  uint64 timestamp_ns = 5;
  uint64 global_step = 10;
  uint32 current_epoch = 2;

  // Metrics
  float validation_accuracy = 3;
  float validation_loss = 4;
  float training_loss = 11;
  map<string, float> training_metrics = 7;   // Native map -> ≤4 GC allocations

  // Context
  HardwareContext hardware_context = 6;
  repeated SeedState seed_states = 8;
  string source_subsystem = 12;
  string training_run_id = 13;
  string experiment_name = 14;
  string packet_id = 9;
  reserved 100 to 199;                        // Extension window
}

message SeedState {
  string seed_id = 1;
  SeedLifecycleStage stage = 2;
  float gradient_norm = 3;
  float learning_rate = 4;
  uint32 layer_depth = 5;
  map<string, float> metrics = 6;
  uint32 age_epochs = 7;
  float risk_score = 8;
}

message HardwareContext {
  string device_type = 1;      // "cuda", "cpu", "tpu"
  string device_id = 2;
  float total_memory_gb = 3;
  float available_memory_gb = 4;
  float temperature_celsius = 5;
  float utilization_percent = 6;
  uint32 compute_capability = 7;
}
```

`AdaptationCommand` encapsulates all control-plane actions via a `oneof` structure. The command always carries:
- `command_id` (UUID)
- `command_type` (`CommandType` enum)
- `target_seed_id` (if applicable)
- `execution_deadline_ms` (enforces epoch budget)
- A `oneof` with payloads for seed operations, optimizer adjustments, or circuit breaker toggles.

Observability wrappers:
```protobuf
message EventEnvelope {
  string event_id = 1;
  string event_type = 2;
  string source_subsystem = 3;
  google.protobuf.Timestamp created_at = 4;
  google.protobuf.Duration processing_deadline = 5;
  google.protobuf.Duration ttl = 6;
  bytes payload = 10;
  string payload_type = 11;
  string content_encoding = 12;
  MessagePriority priority = 20;
  repeated string routing_keys = 21;
  string correlation_id = 22;
  DeliveryGuarantee delivery_guarantee = 30;
  uint32 max_attempts = 31;
  uint32 current_attempt = 32;
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
}
```

### Telemetry Packet Catalog

Leyline telemetry adheres to the “Option B” budget (<280 B, ≤4 allocations, <80 µs) while
capturing the observability signals mandated in the legacy subsystem designs
(`old/01-tolaria-unified-design.md`, `old/02-kasmina-unified-design.md`,
`old/03-tamiyo-unified-design.md`, `old/10-nissa-unified-design.md`). The following
metrics are emitted today:

| Subsystem | Metric(s) | Notes |
| --- | --- | --- |
| Tolaria | `tolaria.training.loss`, `tolaria.training.accuracy`, `tolaria.training.latency_ms`, `tolaria.seeds.active` | Latency is compared against the 18 ms epoch budget and drives the `training_latency_high` alert. Additional epoch diagnostics live in the richer `SystemStatePacket` payload. |
| Kasmina | `kasmina.seeds.active`, `kasmina.isolation.violations` (+ `kasmina.kernel.fetch_latency_ms` when no fallback is in play) | Isolation violations and fallback fetches raise warning events (`fallback_applied`, `isolation_violations`) that bubble into Nissa’s `kasmina_isolation_violation` alert while keeping each packet within the Option B size budget. |
| Tamiyo | `tamiyo.validation_loss`, `tamiyo.loss_delta`, `tamiyo.conservative_mode`, `tamiyo.blueprint.risk` | Risk-induced pauses (loss spikes, high-risk blueprints, conservative mode) are surfaced via succinct telemetry events (`pause_triggered`, `bp_quarantine`) and degrade the health status to `DEGRADED`. |
| Simic (context) | `simic.training.loss`, `simic.training.reward`, `simic.value.loss`, `simic.training.iterations`, `simic.policy.loss`, `simic.param.loss`, `simic.policy.entropy`, `simic.validation.pass` | Included for completeness; emitted by the PPO trainer and ingested by Nissa for dashboards. |

`TelemetryEvent` entries accompany the metrics with subsystem-specific context (e.g.
“Kasmina handled seed operation”, “Tolaria epoch latency above budget”). The
`SystemHealth` section is populated using the same logic—`status` escalates to
`DEGRADED` when budgets are breached and to `CRITICAL` when Tamiyo applies hard risk
stops.

Examples (JSON, after protobuf decoding):

```json
{
  "packet_id": "training-run-epoch-1",
  "source_subsystem": "tolaria",
  "metrics": [
    {"name": "tolaria.training.loss", "value": 0.342, "unit": "loss"},
    {"name": "tolaria.training.accuracy", "value": 0.91, "unit": "ratio"},
    {"name": "tolaria.training.latency_ms", "value": 16.8, "unit": "ms"},
    {"name": "tolaria.seeds.active", "value": 2.0, "unit": "count"}
  ],
  "events": [
    {
      "description": "latency_high",
      "level": "WARNING"
    }
  ],
  "system_health": {
    "status": "DEGRADED",
    "summary": "latency_high",
    "indicators": {"epoch": "1"}
  }
}
```

```json
{
  "packet_id": "kasmina-telemetry-3",
  "source_subsystem": "kasmina",
  "metrics": [
    {"name": "kasmina.seeds.active", "value": 1.0, "unit": "count"},
    {"name": "kasmina.isolation.violations", "value": 1.0, "unit": "count"}
  ],
  "events": [
    {
      "description": "isolation_violations",
      "level": "WARNING",
      "attributes": {"violations": "1"}
    }
  ],
  "system_health": {
    "status": "UNHEALTHY",
    "summary": "violations",
    "indicators": {
      "seeds": "1"
    }
  }
}
```

These examples provide the canonical payloads used in tests (`tests/leyline/test_serialization.py`) to enforce the Option B budgets.

Field reports feed Simic’s replay buffer with bounded payloads (<280 B, ≤4 allocations):
```protobuf
message FieldReport {
  string report_id = 1;
  string command_id = 2;
  string training_run_id = 3;
  string seed_id = 4;
  string blueprint_id = 5;
  FieldReportOutcome outcome = 6;
  map<string, float> metrics = 7;  // Δaccuracy, Δloss, resource deltas
  uint32 observation_window_epochs = 8;
  google.protobuf.Timestamp issued_at = 9;
  string tamiyo_policy_version = 10;
  repeated MitigationAction follow_up_actions = 11;
  string notes = 12;
}

message MitigationAction {
  string action_type = 1;  // e.g. CONSERVATIVE_MODE, ROLLBACK_REQUEST
  string rationale = 2;
}

enum FieldReportOutcome {
  FIELD_REPORT_OUTCOME_UNSPECIFIED = 0;
  FIELD_REPORT_OUTCOME_SUCCESS = 1;
  FIELD_REPORT_OUTCOME_NEUTRAL = 2;
  FIELD_REPORT_OUTCOME_REGRESSION = 3;
  FIELD_REPORT_OUTCOME_ABORTED = 4;
}
```

## Serialization Strategy
1. Prefer primitive types and native maps; avoid wrapper messages to keep allocations ≤4.
2. Pack numeric repeated fields; reserve number ranges for future expansion.
3. Fail builds if serialized size exceeds `leyline.max_message_size_bytes` (280 B default).
4. Circuit breakers in downstream services trip after three consecutive serialization failures.

## Integration Patterns
- `protoc` generates Python bindings consumed by every subsystem build.
- Schema validation runs in CI via `validate_schema()` and `check_version()`; both reject version drift.
- Version field in each message must equal `SchemaVersion.CURRENT_VERSION (1)` until backward-compatible evolution lands.

## Performance Profile

| Operation | Target | Validation |
| --- | --- | --- |
| `SystemStatePacket` serialize | <80 µs | Microbench in `tests/leyline/contracts/test_performance.py`. |
| `AdaptationCommand` serialize | <40 µs | Same harness. |
| `EventEnvelope` wrap | <20 µs | Ensures Oona bus budget compliance. |

## Testing Expectations
- Unit tests confirm serialization latency and size budgets for representative packets.
- Cross-language integration tests ensure every subsystem can deserialize canonical fixtures.
- Property tests (Hypothesis) retain data integrity for metric ranges (accuracy 0–1, epoch 0–10 000, etc.).

## Change Management
All schema edits require:
1. Change proposal with performance impact estimate.
2. Updated golden fixtures + tests.
3. Approval via the Leyline governance workflow before resealing version 1.x.

---
File: docs/design/detailed_design/00.2-leyline-enums-constants.md
---
# Leyline Enums & Constants

## Scope & Status
- Version 1.0 (production)
- Provides the canonical enum values and system budgets referenced throughout Esper-Lite
- `leyline.enums` and `leyline.constants` modules are generated from these definitions

## Lifecycle & Health Enumerations
Keep exact numeric values to maintain network compatibility.

```protobuf
enum SeedLifecycleStage {
  SEED_STAGE_UNKNOWN = 0;
  SEED_STAGE_DORMANT = 1;
  SEED_STAGE_GERMINATED = 2;
  SEED_STAGE_TRAINING = 3;
  SEED_STAGE_BLENDING = 4;
  SEED_STAGE_SHADOWING = 5;
  SEED_STAGE_PROBATIONARY = 6;
  SEED_STAGE_FOSSILIZED = 7;
  SEED_STAGE_CULLED = 8;
  SEED_STAGE_EMBARGOED = 9;
  SEED_STAGE_RESETTING = 10;
  SEED_STAGE_TERMINATED = 11;
}

enum HealthStatus {
  HEALTH_STATUS_UNKNOWN = 0;
  HEALTH_STATUS_HEALTHY = 1;
  HEALTH_STATUS_DEGRADED = 2;
  HEALTH_STATUS_UNHEALTHY = 3;
  HEALTH_STATUS_CRITICAL = 4;
}

enum CircuitBreakerState {
  CIRCUIT_STATE_UNKNOWN = 0;
  CIRCUIT_STATE_CLOSED = 1;
  CIRCUIT_STATE_OPEN = 2;
  CIRCUIT_STATE_HALF_OPEN = 3;
}
```

## Command & Pruning Enumerations

```protobuf
enum CommandType {
  COMMAND_UNKNOWN = 0;
  COMMAND_SEED = 1;
  COMMAND_ROLLBACK = 2;
  COMMAND_OPTIMIZER = 3;
  COMMAND_CIRCUIT_BREAKER = 4;
  COMMAND_PAUSE = 5;
  COMMAND_EMERGENCY = 6;
  COMMAND_STRUCTURAL_PRUNING = 7;
}

enum SeedOperation {
  SEED_OP_UNKNOWN = 0;
  SEED_OP_GERMINATE = 1;
  SEED_OP_START_TRAINING = 2;
  SEED_OP_START_GRAFTING = 3;
  SEED_OP_STABILIZE = 4;
  SEED_OP_EVALUATE = 5;
  SEED_OP_FINE_TUNE = 6;
  SEED_OP_FOSSILIZE = 7;
  SEED_OP_CULL = 8;
  SEED_OP_CANCEL = 9;
}

enum MessagePriority {
  MESSAGE_PRIORITY_UNSPECIFIED = 0;
  MESSAGE_PRIORITY_LOW = 1;
  MESSAGE_PRIORITY_NORMAL = 2;
  MESSAGE_PRIORITY_HIGH = 3;
  MESSAGE_PRIORITY_CRITICAL = 4;
}

enum DeliveryGuarantee {
  DELIVERY_GUARANTEE_UNKNOWN = 0;
  DELIVERY_GUARANTEE_AT_LEAST_ONCE = 1;
  DELIVERY_GUARANTEE_AT_MOST_ONCE = 2;
  DELIVERY_GUARANTEE_EXACTLY_ONCE = 3;
}

```

## Constants: Performance, Memory, Limits

```protobuf
message PerformanceBudgets {
  uint32 EPOCH_BOUNDARY_MS = 18;
  uint32 TAMIYO_INFERENCE_MS = 12;
  uint32 SYSTEMSTATE_ASSEMBLY_MS = 3;
  uint32 ADAPTATION_PROCESSING_MS = 2;
  uint32 FAST_ROLLBACK_MS = 500;
  uint32 FULL_ROLLBACK_SECONDS = 12;
  uint32 PROTOBUF_SERIALIZATION_US = 80;
  uint32 MESSAGE_VALIDATION_US = 20;
  uint32 BREAKER_TIMEOUT_MS = 1000;
  uint32 BREAKER_RESET_SECONDS = 30;
  uint32 IMPORTANCE_TRACKING_OVERHEAD_PERCENT = 1;
  uint32 CHECKPOINT_ANALYSIS_MINUTES_MIN = 2;
  uint32 CHECKPOINT_ANALYSIS_MINUTES_MAX = 5;
  uint32 STRUCTURED_VALIDATION_SECONDS = 60;
  uint32 ROLLBACK_COORDINATION_SECONDS = 30;
}

message MemoryBudgets {
  float MODEL_PERCENT = 0.40;
  float OPTIMIZER_PERCENT = 0.25;
  float GRADIENTS_PERCENT = 0.15;
  float CHECKPOINTS_PERCENT = 0.08;
  float TELEMETRY_PERCENT = 0.05;
  float MORPHOGENETIC_PERCENT = 0.05;
  float EMERGENCY_PERCENT = 0.02;
  float IMPORTANCE_STATISTICS_PERCENT = 0.03;
  float PRUNING_METADATA_PERCENT = 0.02;
}

message SystemLimits {
  uint32 MAX_SEEDS_PER_EPOCH = 100;
  uint32 MAX_MESSAGE_SIZE_BYTES = 280;
  uint32 MAX_QUEUE_DEPTH = 10000;
  uint32 MAX_RETRY_ATTEMPTS = 3;
  uint32 MAX_PAUSE_QUOTA = 10000;
  uint32 MAX_GC_ALLOCATIONS_PER_MSG = 4;
  uint32 MAX_PRUNING_RATIO_PERCENT = 50;
  uint32 MAX_CONSECUTIVE_PRUNING_FAILURES = 3;
  uint32 MAX_CHECKPOINT_STORAGE_GB = 100;
  uint32 MAX_IMPORTANCE_HISTORY_DAYS = 30;
}
```

## Usage Guidelines
- Reserve enum value `0` for `UNKNOWN`/`UNSPECIFIED` to maintain backward-compatibility headroom.
- Treat constants as **hard limits**; exceeding them should trigger local circuit breakers or refusal to operate.
- CI validates that memory budget percentages sum to ≤1.0 and that new enum values are strictly appended.

## Testing & Validation
- Snapshot tests confirm enum numeric stability across language bindings.
- Budget validation runs during subsystem startup; failure halts boot with actionable logs.
- Property tests ensure all `SeedLifecycleStage` values fall within allowed transitions.

## Change Control
Any new enum or constant requires:
1. Governance approval (minimum two reviewers).
2. Explicit documentation of affected subsystems and migration steps.
3. Updated fixtures in `tests/leyline/enums` before merge.

---
File: docs/design/detailed_design/00.3-leyline-governance-implementation.md
---
# Leyline Governance & Implementation

## Scope & Status
- Version 1.0 governance for all Leyline contracts (production)
- Component type: virtual; enforces process, versioning, and tooling for the shared schemas
- Implementation: core helpers exist (`SchemaVersion`, change proposal structures); CI wiring still in flight

## Roles & Ownership
- **Owner**: Data Architect (final approval + release authority)
- **Reviewers**: System Architect, Integration Specialist (technical validation)
- **Contributors**: Subsystem teams proposing contract changes

## Change Workflow
1. **Proposal** – submit a `ChangeProposal` with protobuf diff, rationale, affected subsystems, performance expectations.
2. **Impact Analysis** – automated script (`analyze_change_impact`) classifies compatibility (`BACKWARD_COMPATIBLE` vs `BREAKING_CHANGE`), highlights serialization and consumer risk.
3. **Review & Approval** – minimum approvals: 2 reviewers + owner sign-off. Review window capped at 3 days.
4. **Implementation** – regenerate bindings (`protoc`), run contract test suite, update fixtures.
5. **Release** – bump schema version if change is breaking; tag new contracts; notify integrators.

Emergency bypass is disabled; all changes must follow the standard path.

## Version Management
```python
class SchemaVersion:
    CURRENT_VERSION = 1

    @staticmethod
    def validate_version(version: int) -> bool:
        return version == SchemaVersion.CURRENT_VERSION

    @staticmethod
    def check_compatibility(message_version: int) -> bool:
        # Strict mode until backward compatibility support ships
        return message_version == SchemaVersion.CURRENT_VERSION
```
- Future roadmap introduces backward-compatible support (Phase 2) and runtime negotiation (Phase 3).

## Core Data Structures
```python
@dataclass
class ChangeProposal:
    proposal_id: str
    proposer: str
    contract_name: str
    change_type: ChangeType
    description: str
    protobuf_diff: str
    performance_impact: dict
    affected_subsystems: list[str]
    risk_level: RiskLevel

@dataclass
class GovernanceDecision:
    proposal_id: str
    decision: Decision
    approvers: list[str]
    conditions: list[str]
    implementation_deadline: datetime
    rollback_plan: str
    decision_rationale: str
```

`analyze_change_impact` scans added/removed fields, recalculates serialization cost deltas, and enumerates all dependent subsystems via contract ownership metadata. Breaking removals automatically flag the proposal as high risk.

## Tooling Pipeline
- `validate_contracts()` (CI) ensures schema consistency and enforces governance configuration.
- `compile_protobuf()` rebuilds generated assets; incremental compilation keeps runtime <2 s.
- `check_version()` runs in subsystem startup to catch mismatched schema versions before live traffic.
- Impact analysis and regression tests execute in parallel to keep total review cycle <30 minutes.

## Configuration Snapshot
```yaml
governance:
  owner: data-architect
  reviewers: [system-architect, integration-specialist]
  min_approvals: 2
  review_timeout_days: 3
  emergency_bypass: false
  current_version: 1
  compatibility_mode: strict
  block_on_validation_failure: true
  require_impact_analysis: true
  auto_generate_bindings: true
```

## Failure Handling
| Failure | Detection | Response |
| --- | --- | --- |
| Schema validation failure | CI `validate_contracts` | Block merge, notify proposer + owner |
| Version mismatch | Runtime `check_version` | Reject message, trigger subsystem circuit breaker |
| Compiler error | Build pipeline | Fail build, ship detailed diagnostics |

Circuit breaker defaults (`failure_threshold=5`, recovery timeout 1 h) halt further deployments if repeated violations occur.

## Metrics & Audit
- Counters: `leyline.governance.proposals`, `...approvals`, `...violations`, `leyline.version.mismatches`.
- Logs: proposal submission, decision events, compilation failures.
- Full audit trail stored in version control + governance dashboard.

## Outstanding Work
- GOV-001: Wire governance validation rules into CI (tracking ticket).
- GOV-002: Finish automated impact analysis reporting in dashboards.
# Enum Policy and Canonicalization (ADR‑003)

Leyline is the canonical source for all enums. Subsystems MUST use Leyline enums
internally and externally. No parallel or mapped lifecycle enums are allowed.
Operational conditions such as degraded or isolated states are reported via
`TelemetryPacket.system_health` and events rather than overloading lifecycle stages.
See `old/decisions/ADR-003-enum-canonicalization.md`.
