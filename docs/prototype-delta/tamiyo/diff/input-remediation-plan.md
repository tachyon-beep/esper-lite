# Tamiyo Input Gap Remediation Checklist

This checklist expands on `gnn-inputs.md` by enumerating the code changes required across subsystems to unblock the full WP1 Tamiyo hetero-GNN. Each subsection names the functions that must be updated, summarises the missing inputs today, and calls out the remediation work needed to deliver the schema described in the WP1 design.【F:docs/prototype-delta/tamiyo/GNN-WP1.md†L38-L145】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L5-L86】

## Tamiyo (Policy, Graph Builder, Service)

- `TamiyoGraphBuilder.build` and `_build_*` helpers (global/seed/layer/activation/parameter) only surface the metrics currently present in Tolaria packets and leave many WP1 fields blank. Extend these paths to ingest BSDS payloads, layer topology, activation stats, optimizer metadata, and per-feature masks, persisting normalisation tables and emitting detailed coverage so the GNN receives the full schema.【F:src/esper/tamiyo/graph_builder.py†L186-L258】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L18-L63】
- `_populate_edges` presently synthesises placeholder attributes; replace these with real adjacency (including connection strength, data-flow volume, monitoring thresholds, and capability edges) sourced from Tolaria, Kasmina, and Urza to satisfy the relation requirements in WP1.【F:src/esper/tamiyo/graph_builder.py†L663-L784】【F:docs/prototype-delta/tamiyo/GNN-WP1.md†L73-L146】
- `TamiyoPolicy.select_action` should attach the feature-coverage summary and per-node scores to the emitted `AdaptationCommand` (e.g., annotations or structured attachments) so downstream consumers can react to missing inputs instead of polling policy internals. Status: Delivered in WP15 — `coverage_map` and `coverage_types` annotations are attached by the service/policy path.
- `TamiyoPolicy.__init__` must grow additional embedding registries (layer type, activation type, optimizer family, hazard class) and checkpoint metadata so Simic and online inference share identical vocabularies per WP1.【F:src/esper/tamiyo/policy.py†L71-L148】【F:docs/prototype-delta/tamiyo/GNN-WP1.md†L73-L145】
- `TamiyoService._evaluate` needs to persist BSDS degradation/missing-feature flags, forward the detailed coverage metrics, and surface blueprint metadata failures once Urabrask data is wired in, rather than averaging coverage into a single ratio. Status: Delivered in WP15 for coverage granularity; BSDS-lite flags surfaced; full Urabrask wiring remains pending by design.
- `TamiyoService.publish_history` should run automatically from the Weatherlight loop (or expose an API to drain Tamiyo telemetry) so `tamiyo.gnn.feature_coverage` reaches Oona/Nissa without manual intervention. Status: Delivered in WP11 — Weatherlight calls `TamiyoService.publish_history()` during `_flush_telemetry_once`.

## Tolaria (Training Orchestrator)

- `TolariaTrainer._build_epoch_state_packet` and `_build_step_state` currently emit only aggregate metrics. Extend them to include per-layer gradient flow, activation saturation, latency breakdowns, optimizer deltas, and hook timings needed for node/edge features.【F:src/esper/tolaria/trainer.py†L1214-L1277】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L18-L52】
- `_populate_seed_states` must request richer exports from Kasmina (lifecycle stage, alpha schedule, blending allowances, risk tolerances) and attach them to each `SeedState` so the builder can populate seed features and monitors correctly.【F:src/esper/tolaria/trainer.py†L1279-L1287】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L24-L33】
- `_emit_telemetry` should add the new training metrics (per-layer throughput, hook latency, gradient variance) that Nissa and Tamiyo need for coverage reconciliation and SLOs.【F:src/esper/tolaria/trainer.py†L1293-L1305】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L18-L41】

## Kasmina (Execution Controller)

- `KasminaSeedManager.export_seed_states` must enrich each export with alpha schedules, blending permissions, optimizer compatibility, monitored-layer lists, and risk tolerances so Tolaria and Tamiyo receive the detailed seed features WP1 expects.【F:src/esper/kasmina/seed_manager.py†L1691-L1702】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L24-L33】
- `KasminaSeedManager.handle_command` should parse the new Tamiyo annotations (coverage status, selected feature masks, BSDS signals) and log or react to degraded inputs, enabling Kasmina to fall back or escalate when Tamiyo lacks data.【F:src/esper/kasmina/seed_manager.py†L627-L726】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L82-L84】
- Seed telemetry emitters (e.g., `_queue_seed_events`, `drain_telemetry_packets`) need to include isolation, blending, and capability metrics that Tamiyo’s aggregation hub will rebroadcast downstream.【F:src/esper/kasmina/seed_manager.py†L600-L726】【F:docs/prototype-delta/tamiyo/implementation-roadmap.md†L5-L16】

## Urza (Blueprint Library)

- `UrzaLibrary._build_extras` and related fetch paths should persist and expose blueprint graph metadata (layer adjacency, activation descriptors, optimizer priors, BSDS cache) so Tamiyo can hydrate layer/activation/parameter nodes without bespoke lookups.【F:src/esper/urza/library.py†L317-L360】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L18-L37】
- `UrzaLibrary.get` should return the cached extras alongside `UrzaRecord` so Tamiyo’s metadata loader can retrieve graph payloads efficiently during inference.【F:src/esper/urza/library.py†L92-L176】【F:docs/prototype-delta/tamiyo/GNN-WP1.md†L38-L145】

## Urabrask (BSDS Producer)

- Implement the BSDS export hook so Weatherlight/Tamiyo can request per-blueprint risk sheets (hazard bands, handling classes, resource requirements). Tamiyo’s service layer must cache and attach these to layer/seed nodes during graph construction.【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L12-L28】【F:docs/prototype-delta/tamiyo/GNN-WP1.md†L73-L145】

## Weatherlight (Telemetry Orchestrator)

- `_flush_telemetry_once` needs to drain Tamiyo and Kasmina telemetry buffers (calling into `TamiyoService.publish_history` and Kasmina equivalents) before pushing to Oona, fulfilling the telemetry aggregation hub requirement from the delta matrix. Status: Delivered in WP11 for Tamiyo; Kasmina drain remains a follow-up.
- `_build_telemetry_packet` should embed aggregated coverage metrics (per-feature ratios, BSDS degradation flags) so Nissa can alert on missing inputs without querying Tamiyo directly.【F:src/esper/weatherlight/service_runner.py†L601-L614】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L81-L84】

## Oona (Messaging Fabric)

- `OonaClient.publish_telemetry` and associated stream wiring must ensure the new aggregated packets (coverage summaries, BSDS warnings) land on the appropriate queues with priority routing so downstream services receive them promptly.【F:src/esper/oona/messaging.py†L239-L256】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L81-L84】
- Extend Oona’s schema helpers to recognise any new telemetry message types (e.g., coverage reports) emitted by Tamiyo/Weatherlight to avoid dead-letter routing once the aggregation hub is active.【F:src/esper/oona/messaging.py†L220-L310】【F:docs/prototype-delta/tamiyo/implementation-roadmap.md†L5-L16】

## Nissa (Observability)

- `NissaIngestor.consume` / `_process_slo_metrics` should record the Tamiyo coverage and BSDS metrics in Prometheus and surface alerts when coverage drops below thresholds, closing the loop on the telemetry hub requirement.【F:src/esper/nissa/observability.py†L160-L215】【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L81-L84】
- Update alert routing so degraded-input events from Tamiyo/Kasmina trigger operator notifications and are reflected in SLO summaries, providing visibility into upstream data gaps.【F:src/esper/nissa/observability.py†L160-L215】【F:docs/prototype-delta/tamiyo/implementation-roadmap.md†L5-L16】

## Simic (Offline Trainer)

- Extend `EmbeddingRegistry` usage to cover the new vocabularies (layer, activation, optimizer, hazard) so offline training remains index-aligned with Tamiyo’s online policy checkpoints.【F:src/esper/simic/registry.py†L1-L51】【F:docs/prototype-delta/tamiyo/GNN-WP1.md†L73-L145】
- Update Simic’s dataset exporters to consume the richer field reports/telemetry (including coverage gaps) so the offline trainer can learn from degraded-input scenarios and maintain registry parity.【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L60-L74】【F:docs/prototype-delta/tamiyo/implementation-roadmap.md†L5-L16】

These changes together close the input gaps highlighted in `gnn-inputs.md`, unblock the WP1 GNN from running at full fidelity, and satisfy the telemetry aggregation commitments captured in the Tamiyo delta plan.【F:docs/prototype-delta/tamiyo/diff/gnn-inputs.md†L75-L84】【F:docs/prototype-delta/tamiyo/implementation-roadmap.md†L5-L38】
