# Oona — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Priority routing | `09-oona.md` | Route to EMERGENCY when NORMAL depth high; EMERGENCY bypasses | `_resolve_stream()` with `emergency_threshold` | Implemented | Must‑have | Verified by tests. |
| Backpressure drop | `09-oona.md` | Drop NORMAL when depth exceeds `backpressure_drop_threshold` | `_resolve_stream()` drop path | Implemented | Should‑have | Metrics incremented. |
| At‑least‑once delivery | `09.1-oona-internals.md` | Consumer groups, ack/retry, claim idle, dead‑letter | `consume()` uses XREADGROUP + XACK | Partially Implemented | Must‑have | No retry/claim/dead‑letter handling. |
| TTL & trimming | `09.1` | Stream maxlen trimming and TTL housekeeper for groups | `xadd` with `maxlen` | Partially Implemented | Should‑have | Maxlen trimming present; no TTL housekeeper/idle group cleanup. |
| Circuit breakers & conservative mode | `09-oona.md` | Breakers on publish/consume; conservative mode pauses NORMAL | — | Missing | Must‑have | Not implemented. |
| Telemetry & health | `09-oona.md` | Publish/consume latency, queue depth, breaker state, backpressure counters | `metrics_snapshot()` (queue depth, publish counters only) | Partially Implemented | Should‑have | No latency or breaker telemetry; no health endpoint. |
| Security envelope | `09-oona.md` | Sign envelopes (HMAC) and verify on consume | HMAC sign/verify when secret present | Implemented | Must‑have | Uses `DEFAULT_SECRET_ENV`. |
| Leyline as canonical | `09-oona.md` | Use Leyline `BusEnvelope`, `BusMessageType` | `BusEnvelope` wrapping payloads | Implemented | Must‑have | Canonical use. |

