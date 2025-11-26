# Tamiyo Combined Design

---

File: docs/design/detailed_design/03-tamiyo-unified-design.md
---

# Tamiyo Unified Design (v4.1)

## Snapshot

- **Role**: Strategic controller for Esper-Lite; receives system state, evaluates risk, and issues morphogenetic commands without disrupting Tolaria’s training loop.
- **Scope**: Restored 4-layer HeteroGNN policy, PPO/IMPALA learning stack, multi-dimensional risk engine, and full C-016 production-safety suite.
- **Status**: Production-ready; neural architecture and safeguards reinstated from original v3.1 while retaining all hardening fixes.
- **Invariant**: `tamiyo` owns decision policy yet never mutates training state directly—AdaptationCommands are advisory and always validated by Tolaria/Kasmina.

## Core Responsibilities

- **Strategic Inference**: Encode `SystemStatePacket` graphs, produce policy distributions, value estimates, and risk scores in <45 ms on H100-class GPUs.
- **Risk Governance**: Evaluate gradient, memory, latency, and stability channels; block or downgrade risky actions via conservative mode and rollback hooks.
- **Learning Loop**: Consume telemetry/field reports, run PPO locally, optionally accelerate with IMPALA/V-trace, and push verified policy checkpoints.
- **Field Report Production**: Aggregate adaptation outcomes, build Leyline `FieldReport` payloads, and publish them to Simic without breaching the epoch boundary budget.
- **Integration**: Communicate through Leyline contracts (Option B) with strict latency, timeout, and authentication budgets; respect pause quotas and async delivery deadlines.

## Architecture at a Glance

| Component | Focus | Reference |
| --- | --- | --- |
| Neural policy | 4-layer hetero GNN (GraphSAGE → GAT) with risk/value/policy heads; 256-256-128-128 dims, 4 attention heads | `03.1-tamiyo-gnn-architecture.md` |
| Policy optimisation | PPO + IMPALA/V-trace, graph experience replay (100 K trajectories), LR scheduling via UnifiedLRController | `03.2-tamiyo-policy-training.md` |
| Risk engine | Adaptive thresholds, multi-signal scoring, emergency/go‑no‑go evaluation, structured rollback triggers | `03.3-tamiyo-risk-modeling.md` |
| Integration surface | Leyline contracts, timeout matrix, async coordinator, pause security, telemetry budget | `03.4-tamiyo-integration-contracts.md` |

## Neural Architecture Highlights

- Heterogeneous node encoders for layer (128→256), seed (64→256), activation (32→128), parameter (16→256) feature vectors.
- Two GraphSAGE layers followed by two GAT layers (4 heads) with GELU + dropout (0.2).
- Global mean pooling over layer embeddings feeds three decision heads: risk logits (5 classes), scalar value, 32-way policy logits.
- Inference budget: <45 ms latency, ≤2 GB VRAM; telemetry records per-layer timings and circuit-breaker state.

## Learning & Optimisation

- **PPO**: GAE, clipped policy objectives, entropy bonuses, KL early-stopping; tuned for 128 mini-batch steps, 0.99 γ, 0.95 λ.
- **IMPALA**: Optional distributed learner with V-trace correction, prioritized replay, gradient clipping.
- **Experience Replay**: Graph trajectories compressed and bounded (100 000 entries) to avoid memory growth; GC sweeps enforced per training epoch.
- **UnifiedLRController**: Sole authority for LR mutation; circuit breakers guard integrity and trigger conservative mode if off-policy interference detected.

## Risk & Safety Framework

- **Signals**: Gradient volatility, validation delta, memory utilisation, inference latency, seed lifecycle state.
- **Thresholds**: Dynamically adjusted from baseline stability; risk categories (GREEN→CRITICAL) map to action gating.
- **Emergency Actions**: Immediate downgrade to conservative commands, request Tolaria rollback, or trigger pause with quota enforcement.
- **Circuit Breakers**: Replace asserts across inference, training, async delivery, and LR updates; emit Leyline telemetry on every transition.
- **Conservative Mode**: Reduces policy entropy, lengthens decision deadlines, and suppresses risky blueprint classes when breakers half-open.

## Field Report Workflow

1. **Capture**: After Tolaria confirms an adaptation command, Tamiyo starts an observation window (default 3 epochs) and aggregates telemetry from Tamiyo, Tolaria, and Kasmina.
2. **Synthesis**: `FieldReportBuilder` computes deltas (accuracy, loss, resource impact), attaches risk snapshots, and selects any mitigation actions taken (e.g. conservative mode).
3. **Publish**: A Leyline `FieldReport` is serialized (<280 B) and emitted through Oona (`tamiyo.field_reports` stream). Messages carry the originating `command_id`, blueprint id, observation window, and policy version.
4. **Ack & Retry**: Simic acknowledges via consumer group; missing ack within 2 s triggers one retry and then raises telemetry + conservative-mode hint. Reports persist locally until acked.
5. **Retention**: Reports kept in Tamiyo’s WAL for 24 h to support replays or Simic restarts; checksum validated before deletion.

## Integration & Contracts

- **Leyline Option B**: `SystemStatePacket`, `AdaptationCommand`, `FieldReport`, `TelemetryPacket`, `EventEnvelope`, `SeedLifecycleStage`, `HealthStatus`; single `uint32` schema version, native metric maps (<80 µs serialisation, <280 B payload, ≤4 allocations).
- **Async Coordinator**: Deadline-aware request/acknowledgement pipeline with idempotent command IDs, replay defence, and configurable per-subsystem SLAs.
- **Pause Security**: Server-side quota & auth enforcement to prevent external DoS via pause flooding; audited through telemetry budgets.
- **Decision Delivery**: End-to-end latency target 85 ms (policy inference + risk vetting + messaging). Timeout matrix codified per subsystem (Tamiyo↔Tolaria 12 ms budget, Tamiyo↔Kasmina advisory <5 ms apply window).

## Operational Metrics

| Metric | Target | Escalation |
| --- | --- | --- |
| GNN inference | <45 ms | Circuit breaker opens at 60 ms sustained; conservative mode engaged. |
| Decision latency | <85 ms end-to-end | PagerDuty if >95 ms for 3 consecutive epochs. |
| Risk false-negative rate | <0.5 % | KPI tracked via post-rollback audits. |
| Policy update cadence | ≤30 min between checkpoints | Slower cadence generates warning; halted if LR breaker trips. |
| Message serialization | <80 µs / <280 B | Leyline benchmark alerts on regression. |
| Field report dispatch | <200 ms from observation close | PagerDuty if ack missing after retry; conservative hint raised. |

## Telemetry & Observability

- Structured metrics via Leyline: `tamiyo.inference.latency_ms`, `tamiyo.risk.score`, `tamiyo.pause.quota`, `tamiyo.decision.conservative_mode`.
- Event logging includes adaptation id, risk rationale, breaker state, and timeout compliance; sampled traces captured for top 1 % slowest decisions.
- SLO dashboard anchored on 18 ms epoch boundary reality (C‑016) and decision success rate.

## Production Safeguards (C‑016 Recap)

- Exclusive LR control, pause quota enforcement, garbage-collected async queues, conservative fallback, and full circuit-breaker coverage (inference, async messaging, LR, training).
- Security compliance: HMAC-authenticated commands, role-based pause authorisation, quota tracking, telemetry audit trails.

## References

- `docs/design/detailed_design/03.1-tamiyo-gnn-architecture.md`
- `docs/design/detailed_design/03.2-tamiyo-policy-training.md`
- `docs/design/detailed_design/03.3-tamiyo-risk-modeling.md`
- `docs/design/detailed_design/03.4-tamiyo-integration-contracts.md`
- `docs/design/detailed_design/00-leyline-shared-contracts.md`

---

File: docs/design/detailed_design/03.1-tamiyo-gnn-architecture.md
---

# Tamiyo GNN Architecture (Doc 03.1)

## Scope

Defines the restored 4-layer heterogeneous GNN that powers Tamiyo’s strategic decisions. Architecture, dimensions, node semantics, and operational budgets are unchanged from the original v3.1 design; wording trimmed to the essentials so engineers can reimplement or audit quickly.

## Graph Inputs

- **SystemGraph** assembled from `SystemStatePacket` (Leyline Option B).
- Node types & feature sizes:
  - `layer`: 128-dim structural + performance features.
  - `seed`: 64-dim lifecycle + telemetry snapshot.
  - `activation`: 32-dim statistics (saturation, variance, drift).
  - `parameter`: 16-dim optimizer/meta features.
- Edge types:
  - (`layer`,`connects`,`layer`), (`seed`,`monitors`,`layer`), (`layer`,`feeds`,`activation`), (`activation`,`influences`,`layer`), (`parameter`,`tunes`,`layer`).
- Batch processing uses PyG-style hetero graphs; schema validated against Leyline version 1.

## Network Topology

| Stage | Layer Type | Hidden Dim | Notes |
| --- | --- | --- | --- |
| 1 | GraphSAGE | 256 | Mean aggregation; seed→layer uses max. |
| 2 | GraphSAGE | 256 | Same as layer 1; residual connection optional. |
| 3 | GAT (4 heads) | 128 | Attention across heterogeneous relations, concat=False. |
| 4 | GAT (4 heads) | 128 | Final embedding per node type. |

- Dropout: 0.2 after each layer; activation: GELU.
- LayerNorm per node type inside encoders.
- Node encoders map native features to hidden dims (`layer`: 128→256, `seed`:64→256, `activation`:32→128, `parameter`:16→256).

## Decision Heads

- **Risk head**: 256→128→64→5 logits (softmax) representing risk levels.
- **Value head**: 256→128→64→1 scalar.
- **Policy head**: 256→128→64→32 logits representing kernel choices + control actions.
- Global embedding obtained via mean pooling over `layer` nodes (pooling keyed by batch when present).

## Performance Targets

| Metric | Target | Enforcement |
| --- | --- | --- |
| GNN inference latency | <45 ms (100 K nodes, H100) | Circuit breaker opens at 60 ms sustained; telemetry `tamiyo.gnn.latency_ms`. |
| Memory footprint | ≤2 GB VRAM | Monitored via CUDA stats; breaker on overflow. |
| Serialization | <80 µs / <280 B | Leyline benchmark job. |

## Implementation Notes

```python
class TamiyoGNN(nn.Module):
    def forward(self, x_dict, edge_index_dict):
        encoded = {nt: self.node_encoders[nt](x) for nt, x in x_dict.items()}
        for conv in self.convs:
            encoded = conv(encoded, edge_index_dict)
            encoded = {nt: F.dropout(F.gelu(x), p=0.2, training=self.training)
                       for nt, x in encoded.items()}
        layer_emb = encoded['layer']
        graph_emb = global_mean_pool(layer_emb, batch=None)
        return {
            'risk_prediction': F.softmax(self.risk_head(graph_emb), dim=-1),
            'value_estimate': self.value_head(graph_emb),
            'policy_distribution': F.softmax(self.policy_head(graph_emb), dim=-1),
            'graph_embedding': graph_emb,
        }
```

- Above code omits residual wiring and auxiliary diagnostics but preserves the canonical flow.
- Optional attention edge (seed↔seed) retained in original design—enable via config flag when modelling seed interactions.
- Training uses mixed precision (AMP) by default; fall back to FP32 when breaker detects underflow/overflow anomalies.

## Diagnostics & Telemetry

- Expose per-layer latency, attention entropy, gradient norms, and activation statistics via Leyline telemetry (`tamiyo.gnn.layer{i}.latency_ms`, etc.).
- Gradient clipping (1.0) applied during backprop; recorded for audit.
- Conservative mode reduces dropout to 0.1 and clamps policy logits to limit aggressive exploration.

## Dependencies

- PyTorch 2.8 (baseline), PyG heterogeneous ops, CUDA graphs for inference batching.
- Leyline contracts for schema validation (raises if version mismatch).

### Mission-Critical Behaviours (Authoritative Reference)

Detailed policy flow, risk gating, and field-report handling are described in `docs/design/detailed_design/03-tamiyo.md`. The following capabilities remain mandatory for Esper-Lite deployments:

- **Synchronous Inference Loop:** `Tamiyo.step()` ingests `SystemStatePacket`, runs the hetero-GNN policy, applies the risk engine, and emits an `AdaptationCommand` within the 45 ms budget (Old §"Inference Pipeline").
- **Risk Governance:** Dynamic thresholds, conservative mode, and emergency downgrade paths must guard every command (Old §"Risk Modeling" and §"Conservative Mode").
- **Field Report Lifecycle:** Every adaptation generates WAL-backed field reports that capture outcome metrics, mitigation actions, and policy version for Simic replay (Old §"Field Report Workflow").
- **Telemetry Aggregation:** Tamiyo remains the telemetry hub—ingesting Kasmina/Tolaria signals, normalising them, and forwarding structured metrics to Nissa and Simic (Old §"Telemetry & Observability").
- **Policy Update Interface:** Tamiyo accepts hot-reload of policy checkpoints (and optional LoRA adapters) only after validation gates pass, with rollback on failure (Old §"Policy Deployment").

These behaviours define Tamiyo’s core responsibilities; optional accelerators (e.g., IMPALA fan-out) can be deferred, but the control loop, risk gates, and reporting pipeline must stay intact.

Tamiyo’s GNN remains the cornerstone of strategic inference; preserving these structural and budget constraints is mandatory for Esper-Lite reliability.

---

File: docs/design/detailed_design/03.2-tamiyo-policy-training.md
---

# Tamiyo Policy Training (Doc 03.2)

## Scope

Describes the reinforcement-learning stack that improves Tamiyo’s policy while respecting production guardrails. The restored design keeps PPO + IMPALA, graph-aware replay buffers, and the UnifiedLRController integration from C‑016.

## Training Modes

| Mode | Purpose | Notes |
| --- | --- | --- |
| **PPO (default)** | On-policy fine-tuning using GAE | Mini-batch SGD, clipping, entropy bonus. |
| **IMPALA (optional)** | Distributed learner for large-scale runs | Uses V-trace importance correction, prioritized sampling. |
| **Offline replay** | Batch evaluation & risk calibration | Consumes field reports without affecting live training. |

## Key Hyperparameters

- **Discount γ**: 0.99; **GAE λ**: 0.95.
- **Clip range**: 0.2; **Entropy bonus**: 0.01 → decays to 0.001 in conservative mode.
- **Mini-batch size**: 256 trajectories; **Epochs per update**: 8 (PPO).
- **Learning rate**: Managed exclusively by `UnifiedLRController` (initial 3e-4).
- **Gradient norm clip**: 1.0.

## Training Loop Skeleton

```python
def ppo_update(batch):
    with autocast():
        outputs = gnn(batch.graph)
        value_loss = F.mse_loss(outputs['value_estimate'], batch.returns)
        log_probs = torch.log(outputs['policy_distribution'] + 1e-8)
        ratio = torch.exp(log_probs - batch.old_log_probs)
        surrogate = torch.min(
            ratio * batch.advantages,
            torch.clamp(ratio, 1 - clip, 1 + clip) * batch.advantages,
        )
        entropy = -(outputs['policy_distribution'] * log_probs).sum(dim=-1).mean()
        loss = -(surrogate.mean() - c1 * value_loss - c2 * entropy)
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=1.0)
    lr_controller.step(metrics={'loss': loss.item()})
    optimizer.step()
    optimizer.zero_grad()
```

- `lr_controller.step()` is the only mutation point for learning rates; violations trigger conservative mode and telemetry alerts.
- AMP scaler monitored by circuit breaker to catch inf/nan explosions.

## Graph Experience Replay

- Capacity: 100 000 trajectories (configurable) with on-disk spillover.
- Entries store graph structure, action logits, risk scores, rewards, and metadata (seed ids, lifecycle, timestamps).
- Stratified sampling ensures representation of rare risk events; priority weights derived from TD error + risk severity.
- Periodic GC cleans aged trajectories (>48 h) to prevent memory leak (C‑016 fix).

## IMPALA / V-trace Enhancements

- Learner synchronizes parameters via async checkpoints Tamiyo publishes on the Oona bus.
- Actor-side gradient isolation prevents double updates.
- V-trace coefficients `c̄=1.0`, `ρ̄=1.0` by default; tunable per deployment.

## Safety & Monitoring

- **Circuit breakers** wrap optimisation steps, replay sampling, and parameter broadcasts; on repeated failure we fall back to evaluation-only mode.
- **Conservative mode** reduces LR, entropy, and batch size; disables IMPALA fan-out until stability returns.
- Telemetry: `tamiyo.training.loss`, `tamiyo.training.kl_divergence`, `tamiyo.training.lr`, `tamiyo.training.breaker_state`.
- Policy checkpoints signed & versioned; deployment gating requires successful replay evaluation + risk approval.

## Dependencies & Tooling

- PyTorch 2.8 (baseline), PyTorch Lightning (optional) for trainer orchestration, PyG for graph batching.
- UnifiedLRController configuration shared with Tolaria to guarantee consistent LR semantics.

This training system keeps Tamiyo adaptive while ensuring production stability through tight LR governance, bounded replay storage, and comprehensive monitoring.

---

File: docs/design/detailed_design/03.3-tamiyo-risk-modeling.md
---

# Tamiyo Risk Modeling (Doc 03.3)

## Scope

Captures the restored multi-dimensional risk engine that vetos or downgrades Tamiyo’s decisions. It combines signal scoring, adaptive thresholds, and emergency handling consistent with C‑016 safety guidance.

## Risk Signals

| Signal | Source | Example Metrics |
| --- | --- | --- |
| Gradient Stability | Kasmina telemetry + GNN outputs | Gradient norm variance, seed isolation violations. |
| Performance Drift | Tolaria validation metrics | Δ accuracy, loss plateaus, throughput change. |
| Resource Pressure | Nissa + Tolaria | GPU/CPU utilisation, memory headroom, checkpoint I/O. |
| Latency Budget | Tamiyo inference/async timings | Decision latency, message round-trip, backlog depth. |
| Lifecycle Context | Leyline seed states | Active seeds, grafting density, pause quotas. |

Each signal produces a normalized risk score (0–1). Scores feed the risk assessor with weightings tuned per deployment.

## Adaptive Thresholds

- Baseline thresholds seeded from historical performance; updated via exponential moving averages.
- Conservative mode tightens thresholds (e.g., gradient risk threshold drops from 0.6→0.4).
- Hysteresis prevents oscillation: risk must stay out-of-bounds for N consecutive windows (default 3) before escalation.

## Decision Outcomes

| Risk Level | Action |
| --- | --- |
| Green | Issue command as-is. |
| Yellow | Apply mitigation (e.g., downgrade blueprint, reduce alpha). |
| Orange | Switch to fallback adaptations; flag for policy review. |
| Red | Block decision, request Tolaria rollback/pause. |
| Critical | Immediate emergency path: notify Tolaria, trigger conservative mode, raise PagerDuty. |

## Emergency Handling

- **Automatic rollback**: Compose `AdaptationCommand` with rollback directive when risk > critical and recent adaptations align with suspect seeds.
- **Pause security**: Requests paused seeds through quota-enforced interface; logs auth principal and remaining budget.
- **Incident logging**: Generates structured telemetry with risk vector, thresholds, and chosen mitigation.

## Implementation Outline

```python
def score_decision(signal_bundle):
    scores = {
        'gradient': gradient_model(signal_bundle.gradient_metrics),
        'performance': perf_model(signal_bundle.training_metrics),
        'resource': resource_model(signal_bundle.resource_metrics),
        'latency': latency_model(signal_bundle.latency_metrics),
        'lifecycle': lifecycle_model(signal_bundle.seed_states),
    }
    combined = weighted_sum(scores, weights=config.risk_weights)
    level = classify(combined, thresholds=adaptive_thresholds)
    return RiskAssessment(scores=scores, combined=combined, level=level)
```

- `adaptive_thresholds` updated per epoch; stored in telemetry for audit.
- Risk assessor integrates with decision engine; final output contains rationale & recommended mitigation encoded in metadata.

## Telemetry & Audit

- `tamiyo.risk.combined_score`, `tamiyo.risk.level`, per-signal metrics, mitigation cause.
- Historical records stored for 30 days to support post-mortems (C‑016 requirement).
- Reconciliation job compares risk predictions against actual outcomes (false positives/negatives) and tunes weights offline.

## Dependencies

- Shared risk schema used across Tamiyo, Tolaria, and Simic; defined in Leyline risk contract (enum + fields for rationale, level, thresholds).
- Conservative mode manager subscribes to risk events to adjust behaviour in real time.

The risk engine ensures Tamiyo’s neural policy remains bounded by operational safety, delivering defensible decisions even under degraded conditions.

---

File: docs/design/detailed_design/03.4-tamiyo-integration-contracts.md
---

# Tamiyo Integration Contracts (Doc 03.4)

## Scope

Summarises the APIs, timeouts, and security guarantees Tamiyo observes when interacting with the rest of Esper-Lite. Built entirely on Leyline Option B contracts with C‑016 safety augmentations.

## Key Contracts

| Contract | Purpose | Notes |
| --- | --- | --- |
| `SystemStatePacket` | Primary input describing training state | Leyline versioned; native metric maps; <80 µs serialisation. |
| `AdaptationCommand` | Output command (germination, rollback, etc.) | Includes metadata for risk rationale, deadlines, signatures. |
| `FieldReport` | Post-adaptation outcome payload for Simic | Carries command id, blueprint id, metrics, mitigation actions, policy version. |
| `TelemetryPacket` | Observability events | Structured metrics + risk reports + breaker state. |
| `EventEnvelope` | Bus transport through Oona | Contains HMAC signature, TTL, routing keys. |
| `RiskAssessment` (embedded) | Risk rationale payload | Risk level enum, signal breakdown, mitigation hints. |

## Latency & Timeout Matrix (core excerpts)

| Interaction | Budget | Fallback |
| --- | --- | --- |
| SystemState processing | 12 ms | Conservative mode if >18 ms aggregate with Tolaria hook. |
| Tamiyo→Tolaria command delivery | 5 ms apply window | Drop with telemetry + backlog alert. |
| Async decision ack | 85 ms end-to-end | Block command, enter conservative mode. |
| Pause request | 500 ms | Quota manager aborts & logs incident. |

## Async Coordinator Workflow

1. Receive `SystemStatePacket` via Leyline; validate schema version.
2. Run GNN inference + risk assessment.
3. Construct `AdaptationCommand`, attach risk metadata, signature, nonce, deadline.
4. Submit through Oona `EventEnvelope`; wait for ack within configured deadline.
5. Retry policy: exponential backoff max 3 attempts; after that, escalate to conservative mode.
6. Schedule field-report observation window; register command metadata for later publication.

## Security Controls

- HMAC-SHA256 signatures on all outgoing commands; nonces expire after 5 min.
- Role-based pause authorisation with quota tokens to prevent pause flooding (quota resets per epoch unless manually extended).
- Input validation ensures version match, monotonic timestamps, and bounded payload sizes.
- Circuit breakers wrap message send/receive paths to guard against bus failures.

## Observability

- Metrics: `tamiyo.message.latency_ms`, `tamiyo.message.failures`, `tamiyo.pause.quota_remaining`, `tamiyo.async.queue_depth`.
- Structured logs include correlation IDs, nonce, ack status, risk level, breaker state.
- Traces sampled for highest latency 5 % of decisions; exported via OpenTelemetry.

### Field Report Publication

- Reports published on `tamiyo.field_reports` stream (NORMAL priority); retries mirror command delivery but allow one additional attempt before escalation.
- `field_report_dispatch_latency_ms` recorded end-to-end; breaker trips >500 ms for 3 consecutive reports.
- Simic acks via consumer group; missing ack raises `tamiyo.field_report.retry` telemetry and keeps report in WAL until success (24 h TTL).
- Emergency path: if Tamiyo enters conservative mode due to report failure, it downgrades adaptation entropy until Simic recovers.

## Error Handling Patterns

- Missing ack → conservative mode + optional fallback command with zero-effect adapter.
- Schema mismatch → immediate rejection; request Leyline schema refresh.
- Quota exhaustion → pause request denied, alert raised to operators.
- Deadline violation → command cancelled, breaker increments; Tamiyo waits next epoch.

These integration constraints ensure Tamiyo’s decisions remain timely, authenticated, and observable—critical for safe orchestration within Esper-Lite.
