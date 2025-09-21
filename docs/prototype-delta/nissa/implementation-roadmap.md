# Nissa — Implementation Roadmap (Closing the Delta)

Goal: bring Nissa to the Esper‑Lite design with minimal surface, Leyline‑first, no tech debt.

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 1 | Circuit breakers + conservative mode | Add breakers to ingestion and ES/prom export; conservative mode drops low‑priority telemetry and reduces ingest rate | Stable behaviour under load/faults |
| 2 | TTL housekeeping | Periodic job for ES TTL (logs 7 d, metrics 30 d) and internal counters; expose deletions in telemetry | Bounded storage |
| 3 | Self‑telemetry | Add `nissa.ingest.latency_ms`, `nissa.telemetry.dropped_total`, breaker state, queue lag (from Oona) | Operator visibility |
| 4 | Mission control surface | Add `/api/status` and WebSocket stream with alert count, queue lag, conservative mode; auth stub | Operational control |
| 5 | Alert routing stubs | Implement Slack/email/PagerDuty stubs and dead‑letter queue for failed notifications | Reliable alerts |
| 6 | Health endpoint | Expand `/healthz` to include queue lag, breaker state, active alerts | Clear health signal |

Notes
- Keep all message parsing and types via `leyline_pb2`; no local schemas.
- Coordinate Oona queue‑lag metric exposure for the dashboard and alerts.

Acceptance Criteria
- Breakers and conservative mode observable via telemetry; ingestion/export degrade gracefully.
- TTL housekeeping runs and reduces ES indices; metrics reflect deletions.
- Mission control API exposes status and real‑time stream; alert routing stubs function with tests.

