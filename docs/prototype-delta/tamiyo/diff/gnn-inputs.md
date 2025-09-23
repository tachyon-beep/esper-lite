# Tamiyo Hetero-GNN — Required Inputs

This note captures the concrete inputs Tamiyo needs in order to field the full WP1 hetero-GNN stack (GraphSAGE→GAT with structured policy/risk/value heads). It complements `GNN-WP1.md` by enumerating every upstream signal, registry, and normalization artefact the builder must assemble before the model can run at parity with the detailed design.

## 1. Event Streams and Contracts

| Source | Contract | Purpose |
| --- | --- | --- |
| Tolaria | `SystemStatePacket` | Provides per-step training metrics (`loss`, `gradient_norm`, `samples_per_s`, `hook_latency_ms`) plus layer topology and per-layer telemetry hooks. |
| Kasmina | `SeedState` exports | Supplies seed lifecycle, monitored-layer bindings, alpha schedules, risk tolerances, and allowed blending methods. |
| Urza | Blueprint graph cache | Delivers static blueprint structure (layers, activations, parameters), optimizer hyperparameters, and adjacency for edge construction. |
| Urabrask | Blueprint Safety Data Sheet (BSDS) | Emits risk scores, handling classes, resource profiles, and recommended grafting strategies to enrich layer/seed metadata. |
| Simic | Registry snapshots | Seeds and blueprint vocabularies shared with offline training to keep embedding tables aligned. |
| Tamiyo persistence | `var/tamiyo/gnn_norms.json`, registry JSON artefacts | Stores rolling means/variances for normalization and categorical registries for reproducible index assignment. |

## 2. Node Feature Requirements

### Layer Nodes (`layer`)
- Structural: `layer_type`, `input_channels`, `output_channels`, `activation_type`, `dropout_rate`, `weight_norm`, `update_frequency`.
- Live metrics: normalized `loss`, `gradient_norm`, `samples_per_s`, `hook_latency_ms`, each paired with a binary mask flag to indicate missing data.
- Risk & context: BSDS `risk_score`, `hazard_band`, `handling_class`, mapped to embeddings/masks; blueprint `resource_tier` for capacity awareness.
- Derived telemetry: gradient instability class, gradient variance (log-scaled), latency budget utilisation.

### Seed Nodes (`seed`)
- Identity and routing: `seed_id` registry index, `monitoring_scope`, `monitored_layers` (converted to adjacency), lifecycle stage enum, `policy_version` tag.
- Behavioural: `activity_level`, `success_rate`, `alpha`, `alpha_schedule` descriptor, `risk_tolerance`, `quarantine_only` flag.
- Capability: allowed blending methods, optimizer families, isolation exemptions (multi-hot + masks until Kasmina exports land).
- BSDS links: blueprint risk tier currently attached to the seed, including safe-handling notes.

### Activation Nodes (`activation`)
- Blueprint-provided: `activation_type`, `saturation_rate`, `gradient_flow`, `computational_cost` (or learned embeddings when blueprint data absent).
- Live feedback: saturation events per interval, non-linearity stability score from Tolaria hooks (requires new optional metric).

### Parameter Nodes (`parameter`)
- Hyperparameters: `parameter_count` (`log1p`), `learning_rate`, `momentum`, `gradient_variance`, optimizer family enum.
- Performance: convergence rate estimate, update magnitude, compile status (from Tezzeret prewarm telemetry) to inform action heads.
- Risk context: BSDS resource requirement class, cooldown recommendations for aggressive updates.

### Global / Context Nodes (optional per design)
- System summarizers: rolling mean loss, throughput, GPU/CPU utilisation, Tamiyo conservative-mode flag.
- External: Weatherlight health indicators, SLA breach counters to bias policy decisions.

All categorical values require maintained registries with deterministic indexing (layer types, activation types, optimizer families, blending schedules, hazard classes). Each registry must persist alongside checkpoints so Simic can reproduce embeddings offline.

## 3. Edge Relations and Attributes

| Relation | Attributes Needed | Source |
| --- | --- | --- |
| `layer` ↔ `layer` (`connects`) | `connection_strength`, `data_flow_volume`, `latency_ms`, optional `residual_flag` | Urza blueprint graph + Tolaria profiler metrics |
| `layer` → `activation` (`feeds`/`activates`) | `activation_frequency`, `saturation_events` | Urza blueprint + Tolaria hooks |
| `activation` → `layer` (`influences`/`affects`) | `gradient_contribution`, `nonlinearity_strength` | Derived from Tolaria gradient tracing |
| `activation` → `parameter` (`configures`) | link to parameter influence | Blueprint model topology |
| `parameter` → `activation` (`modulates`) | magnitude, effect | Derived/metadata |
| `parameter` → `blueprint` (`targets`) | categorical target | Blueprint metadata |
| `seed` → `layer` (`monitors`) | `monitoring_frequency`, `adaptation_trigger_threshold`, `quarantine_flag` | Kasmina seed exports |
| `seed` → `parameter` (`allowed`) | Multi-hot compatibility (blending/optimizer) | Kasmina capability registry (future export) |
| Global edges (`context` → `layer`/`seed`) | broadcast scalars (latency budget utilisation, conservative mode) | Tamiyo service state |

Each edge type must include forward and reverse indices for PyG hetero convolutions. Missing attributes should be zero-filled with matching mask channels and logged via coverage telemetry. The builder exports per-key coverage and a typed aggregation for node/edge families to support observability.

## 4. Normalisation & Masking Artefacts
- Maintain EWMA mean/variance for core metrics (`loss`, `gradient_norm`, `samples_per_s`, `hook_latency_ms`) in `var/tamiyo/gnn_norms.json` with α=0.1 and epsilon=1e-6.
- Blueprint priors (log scale + z-score) for `parameter_count`, `learning_rate`, `momentum`, `gradient_variance`; seeds default to 0.5 with masks when absent.
- Persist per-feature mask coverage counters and emit `tamiyo.gnn.feature_coverage` telemetry each build to highlight missing exports.
- When Urabrask fields are missing, substitute neutral embeddings and raise a `tamiyo.gnn.bsds_missing` metric for operations.

## 5. Performance & Device Handling Inputs
- Device availability and precision policy (CPU vs CUDA, TF32 enablement) from Tamiyo runtime configuration.
- Pin-memory buffers for graph batches when CUDA present; builder must expose pinned host tensors for non-blocking transfers.
- `torch.compile` success flag persisted per model version so compile failures are reported exactly once (telemetry + log).

## 6. Checkpoint & Registry Metadata
- Policy checkpoints must bundle model weights, registry snapshots (all categorical vocabularies), normalization tables, and version identifiers (`tamiyo_gnn_v1`).
- Simic training pipeline needs access to the same registries and normalization seeds to ensure alignment; document exchange format in `docs/simic`.
- Hot-swap validation requires a manifest describing expected tensor shapes for each node/edge feature; include it with checkpoints for preflight checks.

## 7. Outstanding Gaps to Close
- Tolaria currently lacks per-layer `data_flow_volume`, `gradient_contribution`, and activation saturation metrics—needs profiler extensions.
- Kasmina does not yet export allowed blending/optimizer compatibility; must extend seed capability payloads.
- Urabrask BSDS feed must be wired into Tamiyo runtime (currently unused) with caching to avoid blocking inference.
- Telemetry to populate `tamiyo.gnn.feature_coverage` and BSDS degradation flags — DELIVERED. Average and per‑type coverage metrics emitted; degraded‑inputs events raised based on thresholds.

## 8. Telemetry Export Gaps & Integration Tasks
- Status: Tamiyo telemetry drain — CLOSED (WP11). Weatherlight now calls `TamiyoService.publish_history()` during each flush so Tamiyo coverage/BSDS signals reach Oona/Nissa automatically (no contract changes). See `src/esper/weatherlight/service_runner.py::_flush_telemetry_once` and `src/esper/tamiyo/service.py::publish_history`.
- Status: Per-feature coverage export — CLOSED (WP15). The builder exposes per-key and typed coverage; Tamiyo attaches `coverage_map` and `coverage_types` to `AdaptationCommand` annotations and emits `tamiyo.gnn.feature_coverage.<type>` metrics. See `src/esper/tamiyo/graph_builder.py` and `src/esper/tamiyo/service.py`.
- **Telemetry hub still missing by design.** The delta matrix and roadmap both call for Tamiyo to aggregate telemetry from Tolaria/Kasmina as part of the WP1 upgrade; today the service emits its own metrics only. Wiring Tamiyo’s telemetry export into Weatherlight should be accompanied by the wider aggregation work (ingesting Kasmina packets from the existing queue, normalising them, and re-emitting a consolidated view) so Kasmina, Tolaria, and Tamiyo stay in lock-step with the documented telemetry taxonomy.

## 9. End-State Hetero-GNN Posture

When the upstream contracts and builder changes above land, Tamiyo’s policy will operate as the WP1 “full fidelity” hetero-graph network rather than today’s interim graph. In that steady state:

- **Graph coverage** – Every blueprint entity (layer, activation, parameter, seed, and global context) is represented with complete node features, forward/reverse relations, and mask channels. Coverage telemetry shows 100 % presence for mandatory features, with any degradations bubbling through the `tamiyo.gnn.feature_coverage` export for Nissa alerting.
- **Inference flow** – `TamiyoGraphBuilder` continuously refreshes registries and normalizers, emits pinned, non-blocking tensors on CUDA nodes, and attaches the feature manifest that `TamiyoPolicy` threads into `AdaptationCommand` annotations. Downstream services ingest the annotations to choose the correct adaptation or fallback mode.
- **Model behaviour** – The four-layer GraphSAGE→GAT stack consumes the enriched hetero-data, exposing 32-way policy logits, calibrated risk/value heads, and compile-state embeddings that satisfy the WP1 acceptance tests. Checkpoints (`tamiyo_gnn_v1.pt`) bundle weights, registries, and norms so Simic and online inference remain index-aligned.
- **Operational visibility** – Weatherlight drains Tamiyo’s telemetry buffer every cycle, forwarding BSDS degradation flags, per-feature coverage, and compile status through Oona to Nissa. Operators see a single consolidated panel covering Tolaria/Kasmina feeds and Tamiyo health, matching the telemetry hub requirement in the delta matrix.

Gathering these inputs (and persisting the supporting registries/normalizers) is the prerequisite for lighting up the full WP1 Tamiyo GNN without falling back to placeholder features.
