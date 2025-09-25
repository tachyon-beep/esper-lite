# Tamiyo — Implementation Roadmap (Closing the Delta)

Goal: close Tamiyo’s gaps to the restored design (Leyline‑first, no tech debt; single set of enums/classes). Decision model is 3A (tight coupling, step‑level), per ADR‑001. Weatherlight remains a whole‑of‑system orchestrator only.

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | Strategic policy (GNN) | Replace FFN stub with 4‑layer hetero GNN (GraphSAGE→GAT), heads for policy/value/risk; maintain <45 ms latency; add unit perf tests | Production‑grade inference consistent with design |
| 2 | Risk engine (multi‑signal) | Implement multi‑channel risk scoring (grad volatility, stability, latency, memory, lifecycle); add thresholds, categories, and gating; emit detailed telemetry | Decisions gated by robust risk assessment |
| 3 | Circuit breakers + conservative mode | Introduce breaker framework around inference, IO, policy updates; automatic conservative‑mode activation on repeated faults; explicit resume conditions | Graceful degradation and recovery |
| 4 | Deadlines/timeouts | Enforce strict timeout matrix (inference, Urza metadata lookups); degrade or no‑op on breach; surface in telemetry | Predictable timing behaviour |
| 5 | Learning loop (PPO) | Add PPO with GAE and LR via UnifiedLRController; integrate AMP and gradient clipping; telemetry for KL, loss, LR | Continuous policy improvement with guardrails |
| 6 | Experience replay | Add graph trajectory buffer (100 K), GC and prioritised sampling; stratify rare risk events | Stable training data pipeline |
| 7 | IMPALA (optional) | Add distributed learner with V‑trace; async checkpoints; disable under conservative mode | Scale‑out training capability |
| 8 | Field report lifecycle | Add observation windows (≥3 epochs), synthesis pipeline, and ack/retry semantics; persist WAL index; retries bounded | Reliable reporting to Simic |
| 9 | Telemetry aggregation hub | Ingest Tamiyo/Tolaria/Kasmina telemetry; normalise; forward to Nissa; CRITICAL events bypass | Centralised observability consistent with design |
| 10 | Security envelope | Sign emitted AdaptationCommands (HMAC/nonce/freshness); verify signed PolicyUpdate payloads; telemetry on failures | Authenticated control surface || 9 | Step‑level integration (3A) | Add `evaluate_step` in Tamiyo; call from Tolaria each step; sign commands; α advance and Kasmina `finalize_step` | Tight coupling; timely, signed decisions without added latency |

Notes
- Leyline remains canonical: use `leyline_pb2` types directly for all contracts and enums.
- Coordinate LR controller integration with Tolaria to ensure single source of truth for LR mutations.
- Keep inference latency budget enforced by tests; add optional Chrome trace capture to profiling docs rather than code.
- We do not support 3B (bus‑subscriber for Tamiyo decisions). See `3A-step-level-tight-coupling.md` for the accepted plan.

Acceptance Criteria
- Policy inference is <45 ms in representative conditions; telemetry includes breakdown.
- Risk engine gates decisions across multiple signals; conservative mode and breaker transitions observable.
- PPO training is functional with LR controller; experience replay works and is bounded.
- Field reports implement observation windows, ack/retry, and WAL durability.
- Telemetry aggregation and security envelopes function as specified.

Status update
- Step 1: Implemented — hetero‑GNN (GraphSAGE→GAT) with compile + CUDA warm‑up; p95 budget asserted; Leyline I/O.
- Step 2: Partially implemented — multi‑signal gating (loss delta, latency, device pressure, optimizer hints, blueprint risk, degraded inputs); room to expand grad/memory stability.
- Step 3: Partially implemented — inference/metadata breakers and conservative mode with telemetry; training/IO breakers remain future work.
- Step 4: Implemented — strict step/metadata timeouts with fail‑open; budgets exposed in telemetry indicators.
- Steps 5–7: Missing — no in‑process learning/replay/IMPALA (handled by Simic in prototype).
- Step 8: Partially implemented — WAL + retention present; ack/retry and observation windows pending.
- Step 9: Partially implemented — Weatherlight drains Tamiyo via `publish_history()` and routes priorities; full aggregation hub across subsystems remains future work.
- Step 10: Partially implemented — commands signed; Oona envelope verification on consume; transactional policy updates with version/freshness/registry validation.
