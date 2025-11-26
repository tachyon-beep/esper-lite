# Oona — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Priority routing | `09-oona.md` | Route to EMERGENCY when NORMAL depth high; EMERGENCY bypasses | `_resolve_stream()` with `emergency_threshold` | Implemented | Must‑have | Verified by tests. |
| Backpressure drop | `09-oona.md` | Drop NORMAL when depth exceeds `backpressure_drop_threshold` | `_resolve_stream()` drop path | Implemented | Should‑have | Metrics incremented. |
| At‑least‑once delivery | `09.1-oona-internals.md` | Consumer groups, ack/retry, claim idle, dead‑letter | `consume()` retries with `_handle_handler_error`, `_claim_stale_messages()` | Implemented | Must‑have | Retries honour max attempts and push to configurable dead-letter stream. |
| TTL & trimming | `09.1` | Stream maxlen trimming and TTL housekeeper for groups | `housekeeping()` with `XTRIM MINID` | Implemented | Should‑have | Supports TTL trimming alongside existing `maxlen` bounds. |
| Circuit breakers & conservative mode | `09-oona.md` | Breakers on publish/consume; conservative mode pauses NORMAL | `_publish_breaker`, `_consume_breaker`, conservative routing | Implemented | Must‑have | Publish/consume failures open breakers and route through emergency mode. |
| Telemetry & health | `09-oona.md` | Publish/consume latency, queue depth, breaker state, backpressure counters | `metrics_snapshot()`, `health_snapshot()` | Partially Implemented | Should‑have | Breaker/backpressure counters exposed; latency tracking remains TODO. |
| Security envelope | `09-oona.md` | Sign envelopes (HMAC) and verify on consume | HMAC sign/verify when secret present | Implemented | Must‑have | Uses `DEFAULT_SECRET_ENV`. |
| Leyline as canonical | `09-oona.md` | Use Leyline `BusEnvelope`, `BusMessageType` | `BusEnvelope` wrapping payloads | Implemented | Must‑have | Canonical use. |
