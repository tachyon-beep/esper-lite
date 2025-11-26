# Simic Combined Design

---

File: docs/design/detailed_design/04-simic-unified-design.md
---

# Simic Unified Design (v3.0)

## Snapshot

- **Role**: Offline policy-training service that improves Tamiyo’s controller without touching production traffic.
- **Scope**: IMPALA/PPO training loop, graph-aware experience replay, policy versioning/validation, and C‑016 safety controls.
- **Status**: Production-ready; design keeps original architecture but trimmed to the essentials so Esper-Lite can run it with minimal surprises.

## Responsibilities

- Ingest Leyline `FieldReport` messages from Tamiyo, map them to rewards/graph features, and store them safely for replay.
- Train policies (Tamiyo, optional Karn) using conventional RL/IL methods, respecting LR governance and conservative-mode triggers.
- Validate, version, and publish new policy checkpoints via Tamiyo’s Oona streams once they pass safety gates.
- Emit telemetry and circuit-breaker state so operators can monitor training health.

## Component Map

| Component | Focus | Reference |
| --- | --- | --- |
| SimicTrainer | IMPALA/PPO training with UnifiedLRController & safety hooks | `04.1-simic-rl-algorithms.md` |
| GraphExperienceBuffer | Graph-compatible replay buffer, TTL cleanup, prioritised sampling | `04.2-simic-experience-replay.md` |
| PolicyManager & Validator | Versioning, chaos/property tests, deployment gating | Included below |
| FieldReportProcessor | Parses field reports, computes rewards, feeds buffer | `04.2` |
| Service Layer | API/health metrics, circuit breakers, conservative mode | Included below |

## Architecture Highlights

- **Offline first**: Simic runs on dedicated GPU nodes; production decisions continue uninterrupted.
- **Event driven**: Field reports arrive over Oona (Leyline `EventEnvelope`) on the `tamiyo.field_reports` stream; policy updates published the same way.
- **Graph native**: Experiences stored as `HeteroData` so Tamiyo’s GNN policy can learn topology-preserving patterns.
- **Schema-aware ingestion**: Field reports expose `command_id`, `blueprint_id`, metrics map, observation window, outcome enum, mitigation actions, and Tamiyo policy version; Simic records each field for replay provenance and reward shaping.
- **Safety baked in**: Circuit breakers wrap training, storage, validation, and LR mutations; conservative mode halves batch size and relaxes timeouts when instability detected.

## Training & Replay (summary)

- IMPALA with V-trace corrections is the default; PPO available for local fine-tuning.
- Experience buffer capacity 100 k trajectories (≈12 GB budget) with TTL cleanup and prioritized replay.
- EWC + LoRA optional modules prevent catastrophic forgetting and reduce weight-update cost.
- UnifiedLRController ensures Simic never double-steps LRs; breaker opens if integrity violated.

## Policy Lifecycle

1. **Ingest**: FieldReportProcessor validates schema version, parses `command_id`, `blueprint_id`, `seed_id`, `observation_window_epochs`, outcome, metrics map, mitigation actions, and `tamiyo_policy_version`, then computes reward/graph features.
2. **Store**: GraphExperienceBuffer writes to circular buffer; TTL & memory budgets enforced; breaker handles overflow.
3. **Train**: SimicTrainer samples batches, runs IMPALA/PPO updates, writes metrics (`simic_training_time_ms`, `simic_loss`, etc.).
4. **Validate**: PolicyValidator runs performance, regression, chaos, property, and security scans; results logged with justification.
5. **Version**: PolicyManager commits weights + metadata (`PolicyVersionMetadata`), updates history, and notifies subscribers.
6. **Deploy**: Policy updates are published on the Oona bus; Tamiyo consumes them after running its safety checks.

## Field Report Stream Handling

- Consumes from Oona `tamiyo.field_reports` stream via dedicated consumer group; ack deadline 2 s aligns with Tamiyo’s retry budget.
- Missing ack triggers replay once; second failure opens messaging breaker, emits `simic.field_report.retry` telemetry, and requests Tamiyo to enter conservative mode until recovery.
- Reports are persisted to disk until acked (24 h TTL) to survive restarts; duplicates detected via `command_id` + `report_id` shard key.

## PolicyManager & Validator Essentials

- Checkpoints stored with metadata (training iteration, metrics, safety scores, approvals).
- Validation pipeline produces `ValidationResult` capturing chaos/property/security outcomes.
- Rollbacks supported by simply switching active version; count tracked for audit.

## Integration Contracts (Leyline Option B)

- `FieldReport` (Tamiyo → Simic) carrying `command_id`, `blueprint_id`, `seed_id`, observation window, outcome enum, metrics map (<280 B), mitigation actions, and Tamiyo policy version.
- `PolicyUpdate` (Simic → Tamiyo/Karn), `TelemetryPacket` (Simic → Nissa), `EventEnvelope` (bus delivery).
- All messages use single `uint32` version, native metric maps, HMAC signatures where applicable.

## Performance Targets

| Metric | Target | Notes |
| --- | --- | --- |
| Training step | <500 ms per 32-experience batch | Breaker opens after sustained overruns. |
| Replay sampling | <50 ms | Prioritized sampling with PyG batching. |
| Validation cycle | <1 s | Includes chaos/property/security scans. |
| Memory usage | ≤12 GB | Monitored per buffer; TTL cleanup every 100 s. |
| Throughput | 180–250 experiences/s | Telemetry verifies; conservative mode lowers target. |

## Safety & Operations

- Circuit breakers for storage, sampling, training, validation, messaging.
- Conservative triggers: high error rate, memory pressure, SLO violation, breaker open, training instability.
- Chaos testing & property-based tests baked into validation to avoid regressions.
- Health endpoints expose breaker states, buffer depth, GPU utilisation, throughput.

### Mission-Critical Behaviours (Authoritative Reference)

Simic’s authoritative design remains codified in `docs/design/detailed_design/04-simic.md`. Esper-Lite keeps the following core behaviours even in the lightweight build:

- **Field Report Pipeline:** Simic consumes Tamiyo’s `FieldReport` stream, validates schema versions, and enriches experiences with provenance before storing them in the graph-aware replay buffer (Old §"Field Report Processor").
- **Replay Buffer Governance:** The `GraphExperienceBuffer` enforces TTL, memory caps (~12 GB), and prioritised sampling to prevent unbounded growth (Old §"Experience Replay").
- **PPO/IMPALA Training Loop:** Even if IMPALA fan-out is deferred, the canonical optimisation flow (PPO objectives, V-trace corrections, LoRA adapters) must stay intact for single-node runs (Old §"Training Stack" and §"Implementation Blueprint").
- **Policy Validation & Versioning:** Before publishing an update, Simic executes the full validation pipeline (performance, regression, chaos/property tests) and records metadata in the policy registry (Old §"PolicyManager & Validator").
- **Policy Deployment:** Validated checkpoints are published via Narset/Oona with accompanying telemetry; Tamiyo consumes them under its own safety gates (Old §"Deploy" stage).

These behaviours constitute Simic’s main job: offline learning that continually improves Tamiyo without disturbing live training.

## References

- `docs/design/detailed_design/04.1-simic-rl-algorithms.md`
- `docs/design/detailed_design/04.2-simic-experience-replay.md`
- `docs/design/detailed_design/00-leyline-shared-contracts.md`

---

File: docs/design/detailed_design/04.1-simic-rl-algorithms.md
---

# Simic Reinforcement-Learning Algorithms (Doc 04.1)

## Scope

Shows how Simic trains Tamiyo’s policy using IMPALA/PPO, UnifiedLRController, and hardening features. Retains numeric targets and safeguard logic from the original document.

## Training Stack

| Element | Purpose |
| --- | --- |
| IMPALA + V-trace | Default distributed learner (32 CPU actors, 4 GPU learners). |
| PPO | Optional on-policy refinement (GAE, clipping). |
| UnifiedLRController | Exclusive LR authority; integrates with conservation and breaker states. |
| EWC & LoRA | Prevent catastrophic forgetting; lightweight fine-tuning. |
| Curriculum hooks | Stage-based data weighting (configurable). |

## Configuration Snapshot

```yaml
algorithm: IMPALA
learning_rate: 3e-4
batch_size: 32
gamma: 0.99
rho_bar: 1.0
c_bar: 1.0
num_actors: 32
num_learners: 4
ewc_lambda: 0.1
lora_rank: 16
```

## SimicTrainer Highlights

```python
class SimicTrainer:
    def __init__(self, config, lr_controller):
        self.policy_net = create_policy_net()
        self.value_net = create_value_net()
        self.policy_opt = AdamW(self.policy_net.parameters(), lr=config.policy_lr)
        self.value_opt = AdamW(self.value_net.parameters(), lr=config.value_lr)
        lr_controller.register_optimizer('simic_policy', self.policy_opt, GroupConfig(...))
        lr_controller.register_optimizer('simic_value', self.value_opt, GroupConfig(...))
        self.training_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_ms=60_000)
```

- Training batches built via PyG `Batch.from_data_list` to preserve graph topology.
- V-trace returns computed per batch; PPO path uses conventional surrogate loss.
- EWC penalty added when prior policies tracked; LoRA adapters initialised when configured.
- Gradient norm clipping (1.0) and AMP scaler monitored by breaker.

## Training Loop (IMPALA path)

```python
async def train_policy(experiences):
    if not training_breaker.is_closed():
        return {"status": "breaker_open"}
    batch = Batch.from_data_list([e.graph_data for e in experiences])
    policy_logits = policy_net(batch)
    values = value_net(batch)
    vtrace = vtrace_module.compute(...)
    policy_loss = compute_policy_loss(policy_logits, experiences, vtrace)
    value_loss = compute_value_loss(values, vtrace)
    total_loss = policy_loss + value_loss + ewc_term()
    if conservative_trigger(total_loss):
        enter_conservative_mode()
    total_loss.backward()
    clip_grad_norm_(parameters, max_norm=1.0)
    lr_controller.step(metrics={"loss": total_loss.item()})
    optimizer.step(); optimizer.zero_grad()
```

- `enter_conservative_mode()` halves batch size, raises timeouts, and reduces LR via controller configuration.
- Training metrics recorded: loss, grad norm, throughput, latency (histograms via telemetry).

## Safety Notes

- Circuit breakers guard storage failures, training instability, LR integrity, AMP explosions.
- Conservative mode triggers: high loss, repeated breaker trips, memory pressure, SLO violations.
- Chaos/property tests run during validation (see unified doc) before policies ship.

## Observability

- Metrics: `simic.training.loss`, `simic.training.kl`, `simic.training.lr`, `simic.training.breaker_state`.
- Logs include batch id, loss components, gradient norms, conservative mode flag.
- Traces sample top latency updates for root-cause analysis.

Simic’s training loop thus remains a standard RL pipeline, hardened with LR governance and breaker-driven safety to support Tamiyo in production.

---

File: docs/design/detailed_design/04.2-simic-experience-replay.md
---

# Simic Experience Replay (Doc 04.2)

## Scope

Details the graph-aware replay buffer and field-report ingestion that feed Simic’s training loop. Content is condensed but retains TTL cleanup, prioritised sampling, and safety hooks.

## Pipeline

1. **FieldReportProcessor** validates Leyline schema, maps outcome→reward, lifts `metrics` map into feature tensors, encodes mitigation actions/policy version into `info`, filters noise, and converts reports to `GraphExperience` objects (`HeteroData` + metadata).
2. **GraphExperienceBuffer** stores experiences in a circular array with TTL and memory budgeting (≈12 GB total).
3. **GraphBatchCreator** samples prioritised batches, attaches importance weights, and hands them to SimicTrainer.

## GraphExperience

```python
@dataclass
class GraphExperience:
    graph_data: HeteroData
    reward: float
    done: bool
    info: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0
```

- Migration note: slated for future Leyline contract once multiple subsystems share graph replay patterns.

## Buffer Mechanics

```python
class GraphExperienceBuffer:
    def __init__(self, capacity=100_000, ttl_seconds=3600, memory_budget_gb=6.0):
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.storage_breaker = CircuitBreaker(failure_threshold=3)
        self.sampling_breaker = CircuitBreaker(failure_threshold=5)
```

- `store()` enforces memory budget (`_estimate_memory_usage()`), triggers `_force_cleanup()` when exceeded, and runs periodic `cleanup_expired()`.
- Prioritised replay: probability ∝ priority^α (α=0.6), importance sampling weight ∝ (N·p)^−β (β=0.4).
- Sampling breaker trips after repeated failures; conservative mode disables prioritisation until restored.

## Memory & TTL Policies

- TTL default 1 h; cleanup invoked every 100 s or when breaker trips.
- Memory usage tracked; forced GC when buffer crosses budget.
- Metadata recorded: hit/miss counts, eviction tally, current memory footprint; exposed via `simic.buffer.*` telemetry.

## Safety Considerations

- Store/sampling operations guarded by circuit breakers; failures produce structured telemetry and skip batch to avoid poisoning training.
- Conservative mode reduces batch size and disables prioritized sampling when buffer health degrades.
- Field reports validated for schema version, reward range, and topology sanity before insertion.

## Telemetry & Ops

- Metrics: `simic.buffer.size`, `simic.buffer.memory_mb`, `simic.buffer.priority_max`, `simic.buffer.breaker_state`.
- Logs tag experiences with correlation IDs so Tamiyo rollbacks can be traced back to replay entries.
- Health endpoint returns buffer depth, TTL cleanup stats, and breaker status.

The replay subsystem preserves the graph structure Tamiyo’s policy expects while enforcing strict memory and safety limits suitable for Esper-Lite deployments.
