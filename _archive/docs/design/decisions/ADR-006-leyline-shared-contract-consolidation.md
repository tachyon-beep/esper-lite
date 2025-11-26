# ADR-006: Leyline Shared Contract Consolidation

## Metadata

| Field | Value |
|-------|-------|
| **ADR Number** | ADR-006 |
| **Status** | PROPOSED |
| **Date** | 2025-09-22 |
| **Author(s)** | Codex (performance audit) |
| **Supersedes** | None |
| **Superseded By** | None |
| **Tags** | leyline, contracts, performance, telemetry, messaging |

## Executive Summary

We will migrate cross-subsystem helper structures (telemetry builders, Oona bus helpers, signature context, embedding registries, and Urza artifact metadata) into the Leyline virtual subsystem. This refactor eliminates Python-level shared dataclasses, restores the contract boundary, and improves high-frequency performance by standardising on generated protobufs.

## Context

Recent audit work uncovered multiple Python dataclasses living outside Leyline that are imported by three or more subsystems. Examples include:

- `TelemetryMetric`/`TelemetryEvent` helpers from `esper.core`
- Oona stream/circuit configuration dataclasses (`StreamConfig`, `OonaMessage`, `BreakerSnapshot`)
- `SignatureContext` from `esper.security`
- Simic embedding registry configuration used by Tamiyo policy inference
- Urza artifact records duplicated across Urza, Tezzeret, and Kasmina

These shunts helped rapid prototyping but they create performance costs (extra allocations, redundant env reads, inconsistent breaker tuning) and violate the "Leyline owns all shared contracts" rule from ADR-001 and ADR-003. As we prepare for scaled workloads, we need canonical, protobuf-first contracts managed by Leyline.

### Performance Drivers

- Telemetry emission already hits >10k packets/minute under perf tests; Python object churn is measurable.
- Oona publish/consume loops pay per-message dataclass wrapping; removing it yields lower latency and GC pressure.
- Signature verification runs hot inside Kasmina; caching the context in Leyline avoids repetitive environment parsing.
- Embedding registry IO introduces lock contention when Tamiyo and Simic both mutate JSON state.
- Urza artifact duplication forces JSON parsing on every lookup and prefetch.

## Decision

Centralise shared contracts inside Leyline with protobuf-first APIs:

1. **Telemetry**: Provide a Leyline telemetry builder that returns pooled `TelemetryPacket` instances. Deprecate `TelemetryMetric`/`TelemetryEvent` dataclasses.
2. **Oona Bus Config**: Define a `BusConfig` proto (streams, TTLs, breaker thresholds) and reusable `CircuitBreakerSnapshot` message so subsystems consume Leyline envelopes directly.
3. **Signing Context**: Move `SignatureContext`, `sign`, and `verify` into Leyline, expose cached context getters, and embed signature metadata in Leyline envelopes.
4. **Embedding Registry Contract**: Specify a Leyline `EmbeddingDictionary` schema (binary + streaming updates) consumed by Simic and Tamiyo. Replace JSON persistence with memory-mapped or protobuf-backed snapshots.
5. **Kernel Catalog Metadata**: Align Urza records with Leyline `KernelCatalogUpdate`/`KernelCatalogEntry` messages; Tezzeret and Kasmina will only exchange protobufs.

## Alternatives Considered

1. **Status Quo** – rejected due to growing performance overhead and architectural drift.
2. **Subsystem-specific copies** – would duplicate code and configs, increasing maintenance burden.
3. **Full microservice boundary** – unnecessary; Leyline already acts as the virtual module for shared contracts.

## Consequences

### Positive

- Reduced allocations and serialisation cost for telemetry and Oona traffic.
- Single source of truth for configuration tuning (streams, breaker thresholds, signing secrets).
- Clear dependency graph: subsystems depend on Leyline, not each other.
- Enables code generation / static analysis for hot paths.

### Negative / Mitigations

- Requires coordinated rollout touching multiple subsystems; we'll gate behind feature flags and staged migrations.
- Telemetry helper rewrite may break bespoke formatting; provide migration guide and shadow mode testing.
- Need to document new Leyline messages thoroughly for downstream teams.

## Rollout Plan

1. Design protobuf updates (Telemetry builder extensions, BusConfig, EmbeddingDictionary, KernelCatalogEntry) and review with subsystem owners.
2. Implement Leyline helper modules (pooled telemetry builder, cached signature context, bus config reader, embedding dictionary API).
3. Update subsystems incrementally:
   - Replace telemetry helper imports.
   - Swap Oona config usage to Leyline BusConfig.
   - Adopt new signature context in Kasmina, Tamiyo, Oona, Weatherlight.
   - Migrate Simic/Tamiyo embedding registry persistence.
   - Align Urza/Tezzeret artifact handling.
4. Remove deprecated dataclasses once all consumers depend on Leyline.

## References

- ADR-001 (tight coupling performance strategy)
- ADR-003 (enum canonicalisation via Leyline)
- Prototype delta audit notes (2025-09-22)
