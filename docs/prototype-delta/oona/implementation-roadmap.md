# Oona — Implementation Roadmap (Closing the Delta)

Goal: align Oona with the unified design (Leyline‑first, no tech debt) while keeping it lightweight.

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | At‑least‑once + retries | Add idle claim/retry (XCLAIM) with min‑idle; dead‑letter stream after N failures; idempotent handler guidance | Robust delivery semantics |
| 2 | Circuit breakers | Breakers for publish/consume errors and timeouts; expose breaker state; integrate with conservative mode | Safer under faults |
| 3 | Conservative mode | When breaker trips or depth high, pause NORMAL publishing/consumption; EMERGENCY only for a recovery window | Guarantees emergency path |
| 4 | TTL housekeeping | Periodic cleanup of idle consumer groups and stale pending entries; stream trimming audit | Bounded memory and clean state |
| 5 | Telemetry & health | Emit publish/consume latency, queue depth, backpressure counters, breaker state; optional health endpoint | Operator visibility |
| 6 | Security hardening | Enforce signature presence when secret configured; attach minimal envelope metadata (event_id, timestamps) | Authenticated bus |

Notes
- Keep the Leyline `BusEnvelope` and `BusMessageType` canonical; no local envelope structs.
- Maintain the emergency path as the highest‑priority stream and verify via tests.

Acceptance Criteria
- Retry/claim/dead‑letter behaviours exist with tests; depth‑based conservative mode pauses NORMAL; telemetry exposes latency/backpressure/breakers; signature enforcement active.

