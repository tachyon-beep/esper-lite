# Tamiyo — Implementation Roadmap (Closing the Delta)

Goal: close Tamiyo’s gaps to the restored design (Leyline‑first, no tech debt; single set of enums/classes).

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | Strategic policy (GNN) | Replace FFN stub with 4‑layer hetero GNN (GraphSAGE→GAT), heads for policy/value/risk; maintain <45 ms latency; add unit perf tests | Production‑grade inference consistent with design |
| 2 | Risk engine (multi‑signal) | Implement multi‑channel risk scoring (grad volatility, stability, latency, memory, lifecycle); add thresholds, categories, and gating; emit detailed telemetry | Decisions gated by robust risk assessment |
| 3 | Circuit breakers + conservative mode | Introduce breaker framework around inference, IO, policy updates; automatic conservative‑mode activation on repeated faults; explicit resume conditions | Graceful degradation and recovery |
| 4 | Deadlines/timeouts | Enforce strict timeout matrix (inference, Oona IO, Urza metadata lookups); degrade or no‑op on breach; surface in telemetry | Predictable timing behaviour |
| 5 | Learning loop (PPO) | Add PPO with GAE and LR via UnifiedLRController; integrate AMP and gradient clipping; telemetry for KL, loss, LR | Continuous policy improvement with guardrails |
| 6 | Experience replay | Add graph trajectory buffer (100 K), GC and prioritised sampling; stratify rare risk events | Stable training data pipeline |
| 7 | IMPALA (optional) | Add distributed learner with V‑trace; async checkpoints; disable under conservative mode | Scale‑out training capability |
| 8 | Field report lifecycle | Add observation windows (≥3 epochs), synthesis pipeline, and ack/retry semantics; persist WAL index; retries bounded | Reliable reporting to Simic |
| 9 | Telemetry aggregation hub | Ingest Tamiyo/Tolaria/Kasmina telemetry; normalise; forward to Nissa; CRITICAL events bypass | Centralised observability consistent with design |
| 10 | Security envelope | Sign emitted AdaptationCommands (HMAC/nonce/freshness); verify signed PolicyUpdate payloads; telemetry on failures | Authenticated control surface |

Notes
- Leyline remains canonical: use `leyline_pb2` types directly for all contracts and enums.
- Coordinate LR controller integration with Tolaria to ensure single source of truth for LR mutations.
- Keep inference latency budget enforced by tests; add optional Chrome trace capture to profiling docs rather than code.

Acceptance Criteria
- Policy inference is <45 ms in representative conditions; telemetry includes breakdown.
- Risk engine gates decisions across multiple signals; conservative mode and breaker transitions observable.
- PPO training is functional with LR controller; experience replay works and is bounded.
- Field reports implement observation windows, ack/retry, and WAL durability.
- Telemetry aggregation and security envelopes function as specified.

