# WS-PER-LAYER-SEED-SUMMARIES

## Overview

- **Objective**: Emit per-seed per-layer gradient summaries from Tolaria to aid Tamiyo diagnostics.
- **Background**: `TolariaTrainer` maintains optional per-layer telemetry scaffolding, but it is currently disabled by default and lacks a dedicated configuration/telemetry contract.
- **Desired Outcome**: When explicitly enabled, Tolaria should surface deterministic, rate-limited metrics for the top-K layers per seed while keeping the feature opt-in to avoid baseline overhead.

## Scope

- Implement a feature flag (`tolaria_seed_layer_summaries_enabled`) plus optional `top_k` limit.
- Extend telemetry emission to include layer-level metrics/events when enabled.
- Add documentation for configuration, metric semantics, and operational guidance.
- Provide unit/integration tests covering enabled/disabled behavior.

## Assumptions

- Seed counts remain modest in prototype scenarios (≤4 simultaneously active).
- Layer count per model can be large; the feature will emit only the top-K layers per seed.
- Tamiyo/Nissa can ingest metrics tagged with `seed_id` and `layer` attributes.

## Risks

| Risk | Severity | Notes | Mitigation |
| --- | --- | --- | --- |
| Telemetry volume spike | Medium | Seed × layer × metric combinations can grow quickly | Strict top-K + flag; document defaults |
| Training-loop overhead | Medium | Additional aggregation in hot path | Early flag checks; reuse existing tensors |
| Consumer incompatibility | Low | Metric format must align with Tamiyo/Nissa expectations | Pre-brief owners; document schema |
| Noisy/low-value signals | Low | Metrics may fluctuate for tiny batches | Document limitations; consider smoothing later |

## Risk-Reduction Tasks

1. Telemetry sizing mock-up (estimate metric counts) – **Owner**: ML ops – **ETA**: Day 1
2. Consumer alignment (Tamiyo/Nissa) – **Owner**: Tolaria team – **ETA**: Day 1
3. Configuration contract draft – **Owner**: Tolaria team – **ETA**: Day 1
4. Baseline performance profile – **Owner**: ML systems – **ETA**: Day 1
5. Test harness design – **Owner**: QA – **ETA**: Day 1
6. Operational comms prep – **Owner**: Ops – **ETA**: Day 1

## Implementation Steps (post risk reduction)

1. Update configuration & docs with new settings.
2. Implement per-layer aggregation guarded by the flag.
3. Emit metrics/events and ensure they respect top-K limit.
4. Add tests for enabled/disabled & multiple-sletseteed scenarios.
5. Validate telemetry output and update dashboards/alerts.

## Risk-Reduction Findings (2024-09-25)

### 1. Telemetry sizing mock-up

- Gradient aggregation currently emits one `tolaria.grad_agg.seed.layer_norm` metric per `(seed, layer)` entry when `_per_layer_enabled` is active and per-seed telemetry is not compacted.
- With a conservative top-K=3, `N` seeds translate to at most `3 × N` additional metrics per telemetry packet. For the prototype’s expected concurrency (≤4 seeds), this results in ≤12 new metrics, plus existing per-seed metrics already in place.
- Payload remains bounded even for deeper models; the per-layer map is computed once per epoch and sliced to `top_k`, so metric growth is linear in seed count rather than layer count.

### 2. Consumer alignment prep

- Proposed metric schema: `tolaria.grad_agg.seed.layer_norm` with attributes `{seed_id, stage, layer}` and unit `grad`.
- Action item for implementation phase: review naming with Tamiyo/Nissa owners and confirm dashboards/ingestion pipelines accept the new attribute tuple.
- Document expected top-K behaviour and note that metrics are suppressed entirely unless the feature flag is enabled.

### 3. Configuration contract draft

- Introduce `TOLARIA_SEED_LAYER_SUMMARIES_ENABLED` (default `false`) to gate the feature.
- Provide optional `TOLARIA_SEED_LAYER_TOPK` (default `3`, minimum `1`).
- Consider `TOLARIA_SEED_LAYER_COMPACT_EVENTS` if we later want to emit a single event instead of individual metrics; for now, rely on existing `tolaria_seed_health_compact` flag.
- Document new settings in Tolaria README, sample `.env`, and architecture summary.

### 4. Baseline performance profile

- Measured CPU-only training loop (3 epochs, 5-layer MLP, no per-layer summaries): **58 ms per epoch** (total 175 ms) under current settings.
- This serves as the baseline to compare after enabling the feature; any regression beyond ~10 % should be investigated during implementation.

### 5. Test harness design

- Unit test plan: enable the flag, inject two mock seed IDs, assert telemetry packet includes exactly `top_k` metrics per seed with correct attributes; confirm disabled mode emits no layer metrics.
- Integration test plan: reuse the existing Tolaria trainer fixture, populate fake seed states via Kasmina stub, and validate Oona-bound telemetry payloads.
- Add regression test for compact mode (`tolaria_seed_health_compact=True`) to ensure layer metrics remain suppressed when compact telemetry is requested.

### 6. Operational comms

- Notify operations that enabling `TOLARIA_SEED_LAYER_SUMMARIES_ENABLED` increases telemetry volume by ~`3 × seeds` metrics per epoch.
- Recommend dashboards/alerts track the new metric family (`tolaria.grad_agg.seed.layer_norm`) before the flag is toggled in non-test environments.
- Document instructions for disabling the feature quickly via environment variables if telemetry back-pressure is observed.

### Follow-up Actions

- **Metric contract**: ensure telemetry schema is recorded (`tolaria.grad_agg.seed.layer_norm` with attributes `{seed_id, stage, layer}`, unit `grad`). This doc now reflects it; mirror the note in the upcoming README/architecture updates.
- **Nissa/Tamiyo alignment**: no code changes needed, but confirm dashboards or analytics that reference seed telemetry can accommodate the new metric family.
- **Operations hand-off**: when the feature ships, include rollout instructions in the ops changelog (.env example + monitoring guidance).
