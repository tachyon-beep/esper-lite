# Issues — Architecture Docs Sweep

## Critical Issues

- Tezzeret metrics prefix inconsistency: 06 uses `tezzeret_*` while 06.1 defines `esper_*`. Standardize on one prefix (recommended: `tezzeret_*`) or include an explicit mapping block in 06.1’s monitoring section.
- Tezzeret contract naming clarity: Unified doc lists "Shared Contracts (Leyline)" but internals define `esper.compilation.v2` messages. Clarify that Leyline is the transport and the contracts are `esper.compilation.v2` to avoid confusion with Karn’s Leyline shared contracts.
- Tezzeret “Protocol Buffers v2” wording: Snippets use proto3. Clarify that “v2” refers to the contract/schema version, not proto2, to prevent misinterpretation.
- Simic config/limits inconsistencies: 04 unified config uses a single `learning_rate`, while 04.1 uses `policy_lr` and `value_lr`. Also, memory limits conflict: performance target says 12GB max, config has `memory_budget_gb: 12.0`, but resource limits list `max_memory_gb: 8.0`. Standardize config keys and reconcile memory targets/limits.
- Simic 04.1 `SimicConfig` fields vs usage: Code references `gamma`, `rho_bar`, `c_bar`, `max_grad_norm`, `training_step_budget_ms`, and conservative factors, but the `SimicConfig` dataclass in 04.1 doesn’t declare them. Align dataclass fields with usage or adjust code examples.

## 07-urabrask (Evaluation Engine)
- Parity: New 07 docs preserve migration content (C-016 fixes, C-020 pruning validation, 3-stage pipeline, realistic CV targets) and are standalone with performance/config/ops sections.
- No critical blockers found. See minor items below for polish and consistency.

Minor editorial/consistency fixes to apply at the end of the doc pass. Grouped by file for easy pickup.

## 05-karn-unified-design.md
- Add explicit checklist: Reintroduce the “mandatory_before_go_with_leyline” YAML checklist (currently implied by Implementation Status). Suggest placing as a short appendix.
- Fix link path: Change “For complete contract definitions, see: /docs/architecture/00-leyline-shared-contracts.md” to a relative path “./00-leyline-shared-contracts.md”.
- Fix truncation: In Phase 2 resource requirements, complete the comment “disk: 8GB  # Reduced due to opti” → “disk: 8GB  # Reduced due to optimized contract storage”.

## 05.1-karn-template-system.md
- Curriculum mapping (optional): Migration doc had a `curriculum_mapping` YAML. Current doc conveys stages within code blocks (clearer). Optionally add a compact YAML appendix summarizing BP ranges per stage for quick scan.

## 05.2-karn-generative-ai.md
- Helper methods: References exist to `_publish_neural_generation_event`, `_publish_gnn_telemetry`, and `_publish_decoder_telemetry` without example stubs. Either:
  - Add brief code snippets mirroring migration-style examples; or
  - Remove/inline the calls to keep examples self-contained.
- Import pattern snippet: Add a short code block showing Leyline import conventions (as in migration 6.2) to mirror usage in examples.

## 07-urabrask-unified-design.md
- Link path: Change absolute `[/docs/architecture/00-leyline-shared-contracts.md]` to relative `./00-leyline-shared-contracts.md`.
- Metrics units: Clarify `urabrask_benchmark_cv` as fraction vs percent (e.g., 0.12–0.18) and reflect in label description; or rename to `urabrask_benchmark_cv_percent` if reporting 12–18 numerically.
- Timeout naming: Table uses time in minutes/seconds; config uses `*_timeout_ms`. Add a short note aligning units and pointing to config keys.
- Authorization note: “Only accepts requests from Tezzeret and Emrakul via Oona” — if Elesh/Jace observers subscribe, consider noting read-only subscribers to avoid confusion.

## 07.1-urabrask-safety-validation.md
- Exception specificity: Replace bare `except:` around NVML with specific `except Exception:` and log reason to aid ops.
- Telemetry consistency: Reference the metric names from unified doc where applicable (e.g., timing budget overruns into `urabrask_safety_validation_time_ms`).
- Hardware context fields: Confirm fields match Leyline schema names used elsewhere (e.g., `available_memory_gb`, `utilization_percent`); they appear consistent.

## 07.2-urabrask-performance-benchmarks.md
- Telemetry snippets: Ensure benchmark failure/variance telemetry also tags kernel_id and mode for correlation.
- Warmup runs: Note that warmup consumes budget; clarify runs are outside the primary measurement window.
- Metric exposure: Unified doc lists `urabrask_performance_benchmarking_time_ms` and `urabrask_benchmark_cv`; consider adding a brief example of emitting these in the benchmarking code.

### Proposed monitoring metrics block (drop-in for 07/Urabrask)

```python
# Urabrask metrics (align with unified doc)
from prometheus_client import Counter, Gauge, Histogram

urabrask_metrics = {
    # Stage durations
    'urabrask_safety_validation_time_ms': Histogram('urabrask_safety_validation_time_ms', buckets=[5000, 10000, 20000, 30000, 45000]),
    'urabrask_structural_validation_time_ms': Histogram('urabrask_structural_validation_time_ms', buckets=[10000, 20000, 40000, 60000, 90000]),
    'urabrask_performance_benchmarking_time_ms': Histogram('urabrask_performance_benchmarking_time_ms', buckets=[60000, 120000, 240000, 480000, 900000]),
    'urabrask_emergency_validation_time_ms': Histogram('urabrask_emergency_validation_time_ms', buckets=[5000, 10000, 20000, 30000]),

    # Throughput and outcomes
    'urabrask_kernels_validated_total': Counter('urabrask_kernels_validated_total', ['result']),
    'urabrask_validation_failures_total': Counter('urabrask_validation_failures_total', ['stage', 'reason']),
    'urabrask_circuit_breaker_triggers_total': Counter('urabrask_circuit_breaker_triggers_total', ['breaker_name']),

    # Resource and quality
    'urabrask_gpu_memory_usage_gb': Gauge('urabrask_gpu_memory_usage_gb'),
    'urabrask_benchmark_cv': Gauge('urabrask_benchmark_cv'),  # document units (fraction or percent)
}

# Note: If CV is reported as percent, use a *_percent suffix or scale to 0–1 consistently across docs.
```

## 08-urza (Central Library)
- Parity: New 08 docs are standalone and preserve migration content (C-016 fixes, immutable CAS, rich metadata, lifecycle, multi-tier caching, query SLAs). Internals document adds concrete storage/query details.
- No critical blockers found. See polish/consistency items below.

### Notes and nitpicks
- Contract naming clarity: Unified doc lists `leyline.*` contracts while internals import `esper_protocols_v2` and Message Contracts use names like `BlueprintProto`/`KernelProto`. Clarify that the transport is Leyline and the contract namespace is `esper_protocols_v2` (v2 schema), or align names across docs.
- “Protocol Buffers v2” phrasing: As with other subsystems, confirm wording refers to contract/schema version (proto3 syntax in examples), not Google’s proto2.
- Metrics naming consistency: Performance section uses `urza_cache_hits_total`/`urza_queries_total` to derive hit rate, while Ops section references `urza_cache_hit_rate`. Either expose a gauge for `urza_cache_hit_rate` or consistently present the ratio derivation; include both in monitoring.
- L4 cache latency phrasing: “S3 with CDN (10s ms)” likely means “tens of ms”. Reword to avoid ambiguity.
- Search index visibility: Internals include `ElasticsearchIndex`; consider listing “Search Index” as an optional component in the unified component table to match capabilities.
- Example async in S3 store: The snippet uses `await self.s3_client.put_object` (boto3 is sync). Add a note that this is pseudocode or suggest `aiobotocore`/threadpool execution to avoid misleading implementers.
- Identity kernel fallback: Add a one-liner describing the identity kernel artifact (no-op kernel) and where it’s stored/referenced for fallback.
- Link text: “For complete contract definitions” link text shows an absolute path; the actual link is relative. Optionally adjust link text for consistency.

### Proposed monitoring metrics block (drop-in for 08/Urza)

```python
# Urza metrics (align with unified doc)
from prometheus_client import Counter, Gauge, Histogram

urza_metrics = {
    # Latency and throughput
    'urza_query_duration_ms': Histogram('urza_query_duration_ms', buckets=[1, 5, 10, 50, 200, 500, 1000]),
    'urza_queries_total': Counter('urza_queries_total', ['source', 'result']),

    # Cache effectiveness
    'urza_cache_hits_total': Counter('urza_cache_hits_total', ['tier']),
    'urza_cache_hit_rate': Gauge('urza_cache_hit_rate'),  # optional convenience gauge

    # Storage and metadata
    'urza_storage_dedup_ratio': Gauge('urza_storage_dedup_ratio'),  # document units (fraction 0–1 vs percent)
    'urza_metadata_attributes': Gauge('urza_metadata_attributes'),

    # Reliability and safety
    'urza_circuit_breaker_state': Gauge('urza_circuit_breaker_state', ['breaker_name', 'state']),
}

# Implementation note: Update 'urza_cache_hit_rate' from counters periodically
# to keep both raw counts and an instantaneous ratio available in dashboards.
```

## 09-oona (Message Bus)
- Parity: New 09 docs preserve migration content (C-016 fixes: Protocol Buffer v2, circuit breakers, conservative mode, TTL cleanup) and are standalone with architecture, ops, security, and config. Internals provide concrete proto schemas and duration helpers.
- No blockers, but several consistency items to tighten given its central role.

### Critical/important clarifications
- Duration units vs protobuf types: Docs mandate “all durations in milliseconds with _ms suffix,” but the envelope uses protobuf `Duration`. Add a one-liner emphasizing conversion via helpers (as in 09.1 `ProtocolDuration`) to avoid mixing raw ms and `Duration` fields in examples.
- Cleanup cadence mismatch: Architecture section says “GC every 100 epochs” while Memory Management config uses `OONA_GC_INTERVAL_MS: 100000` (100 seconds) and the text says “every 100 seconds.” Pick one unit and align all mentions (recommend milliseconds-based config, with a note that “epoch” refers to training context, not GC cadence).

### Notes and nitpicks
- Envelope field naming: Oona’s `EventEnvelope` uses `payload_data` and `tags`; other subsystems’ examples sometimes use `payload` and routing keys. Add a short alignment note or alias fields in examples to keep consistency across docs.
- Metrics design: Prefer deriving “messages per second” from `oona_messages_total` via dashboards rather than a separate `oona_messages_per_second` gauge; if the gauge stays, document calculation method and sampling window.
- Link to shared contracts: Add cross-reference to `./00-leyline-shared-contracts.md` where the envelope and common types live or are referenced.
- Throughput vs bandwidth: With “10k msgs/sec sustained,” add a note that limits assume typical envelope sizes (e.g., <1KB) and scale with payload size.
- Priority SLO phrasing: “Emergency messages < 10ms” — clarify scope (publish vs end-to-end).

### Proposed monitoring metrics block (drop-in for 09/Oona)

```python
# Oona metrics (align with unified doc and best practices)
from prometheus_client import Counter, Gauge, Histogram

oona_metrics = {
    # Latency (ms)
    'oona_message_publish_duration_ms': Histogram('oona_message_publish_duration_ms', buckets=[1, 5, 10, 25, 50, 100, 250]),
    'oona_message_delivery_duration_ms': Histogram('oona_message_delivery_duration_ms', buckets=[5, 10, 25, 50, 100, 250, 500]),
    'oona_gc_duration_ms': Histogram('oona_gc_duration_ms', buckets=[5, 10, 20, 40, 80, 160]),
    'oona_rollback_propagation_duration_ms': Histogram('oona_rollback_propagation_duration_ms', buckets=[50, 100, 250, 500, 1000]),

    # Throughput and queue
    'oona_messages_total': Counter('oona_messages_total', ['topic', 'priority', 'status']),
    # Derive per-second rates in dashboards from oona_messages_total
    'oona_queue_depth_current': Gauge('oona_queue_depth_current', ['topic']),
    'oona_backpressure_level': Gauge('oona_backpressure_level'),  # 0.0–1.0

    # Reliability and SLOs
    'oona_circuit_breaker_state': Gauge('oona_circuit_breaker_state', ['operation', 'state']),
    'oona_conservative_mode_triggers_total': Counter('oona_conservative_mode_triggers_total', ['reason']),
    'oona_slo_violations_total': Counter('oona_slo_violations_total', ['slo_name']),
    'oona_error_budget_consumption_ratio': Gauge('oona_error_budget_consumption_ratio', ['slo_name']),
}
```

## 10-nissa (Observability Platform)
- Parity: New 10 docs preserve migration content (C-016 fixes) and are standalone with architecture, components, integration, performance, resilience, security, and ops sections. Subdocs add concrete designs for metrics/telemetry, mission control, and alerting/SLOs.
- Important gap: Unified doc lists “Critical metrics to monitor” but does not standardize metric names; subdocs also avoid canonical `nissa_*` metric names. Given Nissa’s centrality, define a stable metric namespace to avoid drift across dashboards and exporters.

### Critical/important clarifications
- Metric naming standardization: Add a “Metrics Namespace” section that canonically defines `nissa_*` metric names (histograms, counters, gauges) for ingestion, query latency, alert evaluation, websocket connections, circuit breakers, error budget, and memory cleanup. Ensure ms-suffixed duration names where appropriate.
- Duration units: Consistently document ms-based fields, and show conversion helpers when protobuf `Duration`/`Timestamp` types are used (as done in 09/ProtocolDuration) to prevent mixed units.
- Topic subscriptions: Consider adding a compact table mapping each Oona topic to data model types and retention to facilitate onboarding and capacity planning.

### Notes and nitpicks
- Storage phrasing: “Prometheus backend” often implies remote-write TSDB. Clarify whether Nissa pushes to Prometheus (remote write) or scrapes, and whether long-term storage uses Thanos/Cortex, or an in-house store.
- RBAC verbs: In “Access Control,” add explicit allowed verbs per role (view, ack, execute, configure) to avoid ambiguity.
- Alert routing examples: Include an example route policy (e.g., severity/tag → channel) to make routing more concrete.
- WebSocket scaling: Add note on sharding or backpressure for 10k–25k connections (e.g., per‑node connection cap and autoscaling triggers).

### Proposed monitoring metrics block (drop-in for 10/Nissa)

```python
# Nissa metrics (define canonical names)
from prometheus_client import Counter, Gauge, Histogram

nissa_metrics = {
    # Ingestion and processing
    'nissa_event_ingestion_duration_ms': Histogram('nissa_event_ingestion_duration_ms', buckets=[1, 5, 10, 25, 50, 100, 250]),
    'nissa_metric_ingestion_duration_ms': Histogram('nissa_metric_ingestion_duration_ms', buckets=[1, 2, 5, 10, 20, 50]),
    'nissa_events_total': Counter('nissa_events_total', ['source', 'status']),
    'nissa_metrics_total': Counter('nissa_metrics_total', ['source']),

    # Querying and APIs
    'nissa_query_latency_ms': Histogram('nissa_query_latency_ms', buckets=[5, 10, 25, 50, 100, 150, 250, 500]),
    'nissa_api_requests_total': Counter('nissa_api_requests_total', ['endpoint', 'status']),

    # Alerting and SLO
    'nissa_alert_evaluation_duration_ms': Histogram('nissa_alert_evaluation_duration_ms', buckets=[10, 20, 50, 100, 200, 500]),
    'nissa_alerts_triggered_total': Counter('nissa_alerts_triggered_total', ['severity']),
    'nissa_error_budget_remaining_percent': Gauge('nissa_error_budget_remaining_percent', ['slo_name']),
    'nissa_error_budget_burn_rate': Gauge('nissa_error_budget_burn_rate', ['slo_name']),

    # Resilience and resources
    'nissa_circuit_breaker_state': Gauge('nissa_circuit_breaker_state', ['component', 'state']),
    'nissa_conservative_mode_active': Gauge('nissa_conservative_mode_active', ['component']),
    'nissa_memory_cleanup_duration_ms': Histogram('nissa_memory_cleanup_duration_ms', buckets=[10, 50, 100, 200, 500, 1000]),
    'nissa_memory_cleanup_operations_total': Counter('nissa_memory_cleanup_operations_total', ['bucket']),

    # Realtime
    'nissa_websocket_connections': Gauge('nissa_websocket_connections'),
}

# Guidance: Derive rates (e.g., events/sec) in dashboards from *_total counters.
```

## 11-jace (Curriculum Coordinator)
- Parity: New 11 docs preserve migration content and are standalone with core architecture, C-016 fixes, coordination modes, compatibility matrix, Leyline integration, and SLO/metrics. Subdocs detail testing frameworks, circuit breakers, and SLO metrics.
- Overall in good shape; a few consistency items to polish.

### Critical/important clarifications
- Message contract naming: Unified doc references Leyline Option B with native maps, but examples sometimes use `SystemStatePacket.training_metrics` as Python dicts. Clarify serialization path and ensure examples explicitly show mapping to Leyline native map fields (or use a helper) to avoid implementer confusion.
- Timing budgets: P95 target 18ms appears in multiple places; add a single source of truth (config table) and reference it from sections to avoid drift during edits.

### Notes and nitpicks
- Metric naming units: Most durations use `_ms`, but `jace_leyline_serialization_duration_us` is microseconds. Keep as-is but add a short note in SLO doc that not all timing metrics are ms; call out the µs ones explicitly.
- Cache metrics: Unified doc mentions L1/L2/L3 hit ratios; SLO subdoc exposes hits/misses. Consider adding `jace_cache_hit_ratio` gauge computed in-process or document dashboard computation from counters.
- Event envelope fields: Ensure payload field naming matches Oona/Leyline examples (`payload` vs `payload_data`) across subsystems to avoid minor friction.
- Cross-links: Add a quick reference to Tamiyo and Simic sections where the compatibility matrix inputs originate (e.g., stage definitions), helping readers navigate.

### Proposed monitoring metrics block (drop-in addendum for 11/Jace)

```python
# Jace metrics (augment existing SLO metrics)
from prometheus_client import Counter, Gauge, Histogram

jace_metrics = {
    # Decision quality and latency
    'jace_coordination_decisions_total': Counter('jace_coordination_decisions_total', ['mode', 'fallback', 'conservative']),
    'jace_coordination_duration_ms': Histogram('jace_coordination_duration_ms', ['mode', 'complexity'], buckets=[1,5,10,15,18,25,35,50,100,500]),
    'jace_coordination_success_rate': Gauge('jace_coordination_success_rate', ['mode']),

    # Cache effectiveness
    'jace_cache_hits_total': Counter('jace_cache_hits_total', ['level']),
    'jace_cache_misses_total': Counter('jace_cache_misses_total'),
    'jace_cache_hit_ratio': Gauge('jace_cache_hit_ratio', ['level']),  # optional convenience gauge

    # Resilience and memory
    'jace_circuit_breaker_state': Gauge('jace_circuit_breaker_state', ['breaker_name', 'state']),
    'jace_circuit_breaker_trips_total': Counter('jace_circuit_breaker_trips_total', ['breaker_name', 'reason']),
    'jace_conservative_mode_active': Gauge('jace_conservative_mode_active'),
    'jace_conservative_mode_triggers_total': Counter('jace_conservative_mode_triggers_total', ['trigger_reason']),
    'jace_memory_gc_operations_total': Counter('jace_memory_gc_operations_total', ['component']),
    'jace_memory_entries_cleaned_total': Counter('jace_memory_entries_cleaned_total', ['component', 'reason']),

    # Leyline integration
    'jace_leyline_message_size_bytes': Histogram('jace_leyline_message_size_bytes', ['message_type'], buckets=[50,100,150,200,250,280,350,500]),
    'jace_leyline_serialization_duration_us': Histogram('jace_leyline_serialization_duration_us', ['message_type'], buckets=[10,20,40,60,80,100,150,200]),
    'jace_leyline_contract_validation_failures': Counter('jace_leyline_contract_validation_failures', ['contract_type', 'failure_reason']),
}

# Note: Compute jace_cache_hit_ratio from hits/misses periodically, or do it in dashboards.
```

## 00-leyline (Shared Contracts — Virtual Subsystem)
- Parity: New Leyline docs are comprehensive: core contracts, enums/constants, and governance. Subdocs provide canonical schemas and governance pipeline. Since this is greenfield, focus is clarity and consistency for downstream consumers.

### Critical/important clarifications
- Versioning consistency: Unified doc shows `SchemaVersion.CURRENT_VERSION = 1` and v1.0, while migration artifacts elsewhere reference Leyline v2.x semantics. Add a clear statement that greenfield resets schema to v1 (Option B), and note implications for subsystems currently mentioning “v2” (they refer to contract generation, not proto2).
- Proto headers: Several message snippets (e.g., SystemStatePacket in 00.1) omit `syntax = "proto3";` and `package esper.leyline;`. Add headers in all code blocks for copy-paste correctness and to prevent ambiguity around package names.
- Constants in proto: `PerformanceBudgets`, `MemoryBudgets`, and `SystemLimits` define fields in UPPER_CASE. Proto3 style recommends lower_snake_case field names. Either convert to lower_snake_case fields or document that these are reference constants not intended for runtime mutation (and provide generated-code usage guidance).

### Notes and nitpicks
- Link style: Replace wiki-style links `[[...]]` and absolute paths with relative Markdown links (e.g., `./00.1-leyline-message-contracts.md`) for consistency with other docs.
- EventEnvelope field naming: Align with Oona usage. 00.1 defines `payload` and `routing_keys`. Ensure all subsystem examples use the same names (avoid `payload_data`/`tags` drift).
- Size/latency targets: Add benchmark harness guidance (sample payloads, environment, serializer version) to ensure the <80µs and <280 bytes targets are measured consistently across languages/toolchains.
- Governance pipeline: Implementation status lists CI/CD validation and protobuf compilation as TODO. Recommend adding a brief CI outline (lint + `buf`/`prototool` + codegen + compatibility check) and a migration gate for any schema changes touching top-level messages.

### Drop-in snippets

Proto header and package (add to each snippet):
```protobuf
syntax = "proto3";
package esper.leyline;
import "google/protobuf/timestamp.proto";
import "google/protobuf/duration.proto";
```

CI policy outline (for governance doc):
```yaml
# .github/workflows/leyline-validate.yml (outline)
name: Validate Leyline Contracts
on: [pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: bufbuild/buf-setup-action@v1
      - run: buf lint && buf generate
      - run: make proto-bench  # verify <80us/<280B targets with sample payloads
      - run: python tools/check_compat.py  # ensure no forbidden field deletions
```

## Proto3/Leyline Wording Sweep (repo-wide)

Goal: Standardize to “Protocol Buffers (proto3), Leyline schema v1”; scope the “no map<>” rule to envelopes only; align EventEnvelope field names; and add proto headers in code blocks. No live changes yet — this is a writer sweep list.

- Tezzeret
  - docs/architecture/06-tezzeret-unified-design.md: Replace “Protocol Buffers v2” with “proto3; Leyline schema v1”. Scope “no map<>” to envelopes or “as defined by Leyline”.
  - docs/architecture/06.1-tezzeret-compilation-internals.md: Same phrasing updates at intro and “Protocol Buffers v2 Integration” section; comments like “NO map<>” should be message-specific. Keep proto3 wording throughout.

- Urza
  - docs/architecture/08-urza-unified-design.md: Update all “Protocol Buffers v2” mentions to “proto3; Leyline schema v1”.
  - docs/architecture/08.1-urza-internals.md: Update wording; change any config like `protocol_version: v2` to `leyline_schema_version: 1 (proto3)`. Scope any “no map<>” claims.

- Oona
  - docs/architecture/09-oona-unified-design.md: Replace v2 wording with proto3; at “NO map<> fields” change to “No map<> in the envelope; maps allowed where Leyline defines them.” Align envelope fields to `payload` and `routing_keys`.
  - docs/architecture/09.1-oona-internals.md: Same; adjust “Verify no map<> fields” to envelope-only; keep proto3 phrasing.

- Nissa
  - docs/architecture/10-nissa-unified-design.md and 10.1/10.3 subdocs: Replace “Protocol Buffers v2” with “proto3; Leyline schema v1” in narrative and logs/errors; retain validation language.

- High-Level/Archive (optional, recommend note)
  - docs/architecture/high-level-design/008-data-contracts-api-specifications.md: Multiple “Protocol Buffer v2” references — update to proto3; clarify map<> per Leyline.
  - docs/architecture/archive/C-016-INTEGRATION-TRACKING.md: Keep historical but add note that current standard is proto3/Leyline v1; maps allowed where defined.

- EventEnvelope naming alignment
  - Ensure examples across subsystems use Leyline’s `payload` and `routing_keys` (not `payload_data`/`tags`). Add clarifying note where aliasing exists.

- Proto headers in Leyline snippets
  - Prepend to all 00.1 code blocks: `syntax = "proto3";` and `package esper.leyline;` with needed imports.

- Version statements
  - Replace any “Protocol Buffers v2” implying proto2 with “Proto3; Leyline schema v1”.
  - Update configs that say “protocol_version: v2” to “leyline_schema_version: 1”.

### Metrics naming and coverage (09 unified vs 09.1 internals)
- Present in unified only (consider ensuring internals implement/expose):
  - `oona_message_publish_duration_ms`, `oona_message_delivery_duration_ms`, `oona_gc_duration_ms`, `oona_rollback_propagation_duration_ms` (histograms)
  - `oona_messages_total` (counter) and derivation guidance for rate
  - `oona_queue_depth_current`, `oona_backpressure_level`
  - `oona_circuit_breaker_state`, `oona_conservative_mode_triggers_total`, `oona_slo_violations_total`, `oona_error_budget_consumption_ratio`
- Internals focus on circuit breaker/conservative mode mechanics; add/confirm metric emission points in publish/subscribe paths and GC/rollback handlers.

### Metrics naming and coverage mismatches (08 unified vs 08.1 internals)
- Present in 08.1 but not in unified list (decide to keep or de-scope):
  - `urza_conservative_mode` (Gauge) — useful ops signal; consider adding to unified Key Metrics.
  - `urza_storage_size_gb` (Gauge) — storage usage; unified lists `urza_storage_dedup_ratio` instead.
- Present in unified but missing from 08.1 metrics table (add to internals):
  - `urza_queries_total` (Counter) — needed to derive cache hit rate and throughput.
  - `urza_storage_dedup_ratio` (Gauge) — matches unified performance target.
  - `urza_cache_hit_rate` (Gauge) — optional convenience gauge; or document derivation from counters.
- Alignment notes:
  - Keep `urza_query_duration_ms` as the single latency histogram used across both docs (p50/p95/p99 derived in dashboards).
  - Ensure `urza_circuit_breaker_state` labels match convention `['breaker_name','state']` used in other subsystems.

## 04-simic-unified-design.md
- Link path: Change “For complete contract definitions, see: /docs/architecture/00-leyline-shared-contracts.md” to relative “./00-leyline-shared-contracts.md”.
- Metrics naming: Consider `simic_gpu_utilization_percent` (with unit in name) for consistency with other subsystems; or explicitly note units for `simic_gpu_utilization`.
- Cleanup cadence phrasing: Text mentions TTL cleanup every “epochs” while config uses `ttl_cleanup_interval_s` (seconds). Align terminology and units.
- ADR link: `[ADR-011]` should be a proper relative link to the ADR location (e.g., `../adrs/ADR-011-simic-policy-training.md`).

## 04.1-simic-rl-algorithms.md
- Config naming alignment: Either adopt `learning_rate` (single) to match unified doc/migration, or keep `policy_lr`/`value_lr` and reflect that in the unified config section; avoid mixed schemes.
- `SimicConfig` completeness: Add missing fields used by code examples (`gamma`, `rho_bar`, `c_bar`, `max_grad_norm`, `training_step_budget_ms`, `conservative_batch_size_factor`, `conservative_timeout_factor`).
- Conservative mode memory check: Replace `torch.cuda.max_memory_allocated()` (peak) with `torch.cuda.mem_get_info()` or NVML to derive a true utilization ratio; current ratio may be misleading.
- Monitoring: Add a short metrics block defining `simic_training_time_ms`, `simic_sampling_time_ms`, `simic_validation_time_ms`, `simic_memory_usage_gb`, `simic_gpu_utilization[_percent]`, `simic_throughput_exp_sec`, `simic_slo_compliance` to mirror unified doc.

## 04.2-simic-experience-replay.md
- Missing import: Code uses `np` but doesn’t import NumPy. Add `import numpy as np` in examples.
- Undeclared helper: `_estimate_memory_usage()` is referenced but not shown; add a brief stub or explanatory note.
- Cleanup cadence: Clarify whether `cleanup_interval` is seconds or operations; align with unified doc phrasing/units.

### Proposed monitoring metrics block (drop-in for 04/Simic)

```python
# Simic metrics (aligns with unified doc naming)
from prometheus_client import Counter, Gauge, Histogram

simic_metrics = {
    'simic_training_time_ms': Histogram('simic_training_time_ms', buckets=[50, 100, 250, 500, 750, 1000]),
    'simic_sampling_time_ms': Histogram('simic_sampling_time_ms', buckets=[5, 10, 20, 30, 50, 80, 120]),
    'simic_validation_time_ms': Histogram('simic_validation_time_ms', buckets=[50, 100, 250, 500, 1000]),
    'simic_memory_usage_gb': Gauge('simic_memory_usage_gb'),
    'simic_gpu_utilization': Gauge('simic_gpu_utilization'),  # or simic_gpu_utilization_percent
    'simic_throughput_exp_sec': Gauge('simic_throughput_exp_sec'),
    'simic_slo_compliance': Gauge('simic_slo_compliance'),
}

# Note: If adopting *_percent convention, rename accordingly and document units.
```

## 06-tezzeret-unified-design.md
- Truncated SLO text: Complete the line “WAL recovery succes” under Operational Considerations → Health Monitoring → SLO Targets (e.g., “WAL recovery success <12s P95”).
- Contract naming consistency: “Shared Contracts (Leyline)” lists `leyline.*` messages, but 06.1 uses `esper.compilation.v2` proto package. Clarify that Leyline is the transport and contracts are `esper.compilation.v2` (or update names for consistency).
- “Protocol Buffers v2” wording: Clarify that “v2” refers to contract/schema version (not proto2). Snippets use `syntax = "proto3"`; adjust phrasing to avoid confusion.
- Performance targets (optional): Add explicit “Sandboxing Overhead: 10–20%” metric preserved from migration doc to keep parity.
- Cross-reference: `[ADR-023-Circuit-Breakers.md]` lacks a relative path; link should point to the correct ADR location (e.g., `../adrs/ADR-023-Circuit-Breakers.md`).

## 06.1-tezzeret-compilation-internals.md
- Proto section: Add a brief note or snippet showing the decode–reencode validation step (called out in migration) to reinforce v2 contract hygiene.
- Metrics consistency: Ensure metric names used here align with those in the unified doc (e.g., `tezzeret_cache_hit_rate`, `tezzeret_wal_recovery_duration_ms`). If names differ, add a short mapping or unify.

### Metrics naming and coverage mismatches (06 vs 06.1)
- Prefix mismatch: 06 uses `tezzeret_*` while 06.1 defines `esper_*` metrics. Choose one prefix (recommended: `tezzeret_*` for subsystem consistency) and standardize across both docs.
- One-to-one mappings to reconcile:
  - `tezzeret_compilation_duration_ms` ↔ `esper_compilation_duration_ms`
  - `tezzeret_circuit_breaker_state` ↔ `esper_circuit_breaker_state`
  - `tezzeret_conservative_mode_active` ↔ `esper_conservative_mode_active`
  - `tezzeret_wal_transactions_total` ↔ `esper_wal_transactions_total`
  - `tezzeret_wal_recovery_duration_ms` ↔ `esper_wal_crash_recovery_duration_ms` (align on one name)
- Metrics present in 06 but missing in 06.1 (add or revise):
  - `tezzeret_gpu_utilization_percent` (add a Gauge; 06.1 has a GPU tracker but no metric)
  - `tezzeret_concurrent_compilations` (add a Gauge reporting current active jobs)
  - `tezzeret_memory_usage_gb` (add a Gauge; 06.1 tracks cache entries/GC but not GB usage)
  - `tezzeret_cache_hit_rate` (add a Gauge or Histogram; 06.1 lacks cache hit metric)
- Metrics in 06.1 but not listed in 06 Performance/Monitoring (consider adding or acknowledging):
  - `esper_wal_fsync_duration_ms`, `esper_wal_recovery_total` (ops depth)
  - `esper_memory_ttl_cleanup_total`, `esper_memory_cache_entries`, `esper_memory_gc_duration_ms`
  - `esper_epoch_boundary_duration_ms` (optional to expose)

#### Proposed monitoring metric mapping block (drop-in for 06.1)

```python
# Standardize to tezzeret_* prefix while keeping semantics unchanged
from prometheus_client import Counter, Gauge, Histogram

metrics = {
    # Circuit breakers / conservative mode
    'tezzeret_circuit_breaker_state': Gauge('tezzeret_circuit_breaker_state', ['circuit_name', 'state']),
    'tezzeret_circuit_breaker_failures_total': Counter('tezzeret_circuit_breaker_failures_total', ['circuit_name']),
    'tezzeret_conservative_mode_triggers_total': Counter('tezzeret_conservative_mode_triggers_total', ['trigger_reason']),
    'tezzeret_conservative_mode_active': Gauge('tezzeret_conservative_mode_active'),

    # Memory management
    'tezzeret_memory_ttl_cleanup_total': Counter('tezzeret_memory_ttl_cleanup_total', ['cache_type']),
    'tezzeret_memory_cache_entries': Gauge('tezzeret_memory_cache_entries', ['cache_type']),
    'tezzeret_memory_gc_duration_ms': Histogram('tezzeret_memory_gc_duration_ms'),
    'tezzeret_memory_usage_gb': Gauge('tezzeret_memory_usage_gb'),  # unify with 06

    # WAL
    'tezzeret_wal_transactions_total': Counter('tezzeret_wal_transactions_total', ['status']),
    'tezzeret_wal_recovery_total': Counter('tezzeret_wal_recovery_total', ['outcome']),
    'tezzeret_wal_fsync_duration_ms': Histogram('tezzeret_wal_fsync_duration_ms'),
    'tezzeret_wal_recovery_duration_ms': Histogram('tezzeret_wal_recovery_duration_ms'),  # rename from crash_recovery

    # Compilation timing / throughput
    'tezzeret_compilation_duration_ms': Histogram(
        'tezzeret_compilation_duration_ms',
        ['strategy', 'status', 'conservative_mode'],
        buckets=[5000, 10000, 30000, 65000, 120000, 250000, 500000, 960000]
    ),
    'tezzeret_epoch_boundary_duration_ms': Histogram(
        'tezzeret_epoch_boundary_duration_ms',
        buckets=[5, 10, 15, 18, 25, 50, 100, 500, 1000]
    ),

    # Resource/utilization
    'tezzeret_gpu_utilization_percent': Gauge('tezzeret_gpu_utilization_percent'),
    'tezzeret_concurrent_compilations': Gauge('tezzeret_concurrent_compilations'),
    'tezzeret_cache_hit_rate': Gauge('tezzeret_cache_hit_rate'),
}

# Migration note:
# - Old names (06.1): esper_* equivalents map 1:1 to the tezzeret_* names above.
# - If changing metric names in a live system, consider exposing both for one release.
```
