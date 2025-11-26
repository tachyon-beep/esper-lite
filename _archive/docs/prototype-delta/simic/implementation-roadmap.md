# Simic — Implementation Roadmap (Closing the Delta)

Goal: bring Simic in line with the unified design, Leyline‑first, no tech debt.

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | Replay buffer (prioritised + budgeted) | Add memory budgeting (estimate usage), prioritised sampling with IS weights, TTL scheduler; expose metrics | Robust, scalable buffer |
| 2 | Ingestion semantics | Add Oona consumer with ack/retry, duplicate detection (command_id/report_id), WAL‑backed persistence (24 h) | Reliable ingestion pipeline |
| 3 | Trainer algorithms | Add IMPALA with V‑trace as default; keep PPO path; integrate gradient clipping/AMP; telemetry | Production‑grade training |
| 4 | LR governance | Integrate UnifiedLRController; register optimiser groups; open breaker on integrity violations | Single source of LR truth |
| 5 | Circuit breakers + conservative mode | Breakers around storage/training/validation; auto conservative mode (reduce batch, LR, disable IMPALA fan‑out) | Safe degradation |
| 6 | Policy validation & versioning | Extend validator (chaos/property/security checks); track versions/history and approvals in metadata | Safer deployments |
| 7 | Telemetry | Emit `simic.training.*`, breaker state, replay stats; publish via Oona | Operator visibility |

Notes
- Leyline remains canonical: use FieldReport/PolicyUpdate directly and keep message budgets.
- Keep graph‑native path optional (adopt PyG HeteroData if/when Tamiyo GNN inputs appear) to avoid premature complexity.

Acceptance Criteria
- Buffer enforces TTL, memory budget, and prioritisation; metrics exposed.
- Ingestion handles ack/retry and duplicates with minimal loss; persisted for 24 h.
- IMPALA/V‑trace training functional; PPO path retained; LR controller integrated.
- Breakers/conservative mode observable; validator gates publishing; telemetry complete.

