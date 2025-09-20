# Oona Combined Design

---
File: docs/design/detailed_design/09-oona-unified-design.md
---
# Oona Unified Design (Esper-Lite)

## Snapshot
- **Role**: Lightweight message bus that moves telemetry and control events between subsystems.
- **Scope**: Provide async publish/subscribe on Redis Streams, enforce simple priority handling, and surface health metrics. No advanced routing or cross-datacenter replication in Esper-Lite.
- **Status**: Production-ready; retains circuit breakers, TTL cleanup, and conservative mode from C‑016.

## Responsibilities
- Accept messages from producers (Tamiyo, Tolaria, Tezzeret, etc.) and write them to Redis Streams queues.
- Deliver messages to subscribers with at-least-once semantics and basic deduplication IDs.
- Enforce message TTL, max stream length, and priority handling (EMERGENCY bypasses limits).
- Emit operational metrics and trigger conservative mode when queues grow beyond thresholds.

## Component Map
| Component | Purpose | Notes |
| --- | --- | --- |
| StreamManager | Wraps Redis Streams publish/ack APIs | Configures TTL, maxlen. |
| PriorityRouter | Places messages into NORMAL vs EMERGENCY streams | EMERGENCY always delivered. |
| CircuitBreakerLayer | Guards publish/consume operations | Falls back to conservative mode. |
| TTLHousekeeper | Periodic cleanup of idle consumer groups | Prevents memory creep. |
| MetricsEmitter | Publishes queue depth, latency, breaker state | Consumed by Nissa. |

## Message Envelope
All payloads wrapped in Leyline `EventEnvelope` (Option B) containing `event_id`, `priority`, timestamps, `payload_data`, and optional correlation info. No map fields. Publish/subscribe clients decode payload into subsystem-specific Protocol Buffers.

## Operations
1. **Publish**: Producer sends `EventEnvelope` → StreamManager writes to appropriate Redis stream → success/failure logged.
2. **Consume**: Subscriber reads via consumer group → acknowledges after processing → re-delivery occurs if ack not received within timeout.
3. **Conservative Mode**: Triggered when breaker trips or queue depth exceeds threshold; disables NORMAL publishing (emergency-only) until recovery window expires.

## Performance Targets
| Metric | Target | Notes |
| --- | --- | --- |
| Publish latency p95 | <25 ms | Redis round-trip. |
| Delivery latency p95 | <100 ms | Consumer group fetch. |
| Queue depth recovery | <10 s | After spike ends. |
| Availability | 99.9 % | Measured via health endpoint. |

## Configuration Highlights
```yaml
oona:
  redis:
    url: redis://oona-bus:6379
    stream_maxlen: 5000
    ttl_seconds: 3600
  breakers:
    publish: {failure_threshold: 3, timeout_ms: 30000}
    consume: {failure_threshold: 3, timeout_ms: 30000}
  conservative_mode:
    queue_depth_high: 4000
    recovery_depth: 1000
    emergency_only_duration_s: 60
```

## Metrics & Health
- `oona.publish.latency_ms`, `oona.consume.latency_ms`, `oona.queue.depth`, `oona.breaker.state`, `oona.conservative_mode_active`.
- Health endpoint reports Redis connectivity, queue depth, breaker status.

Oona therefore acts as a simple, reliable messaging layer for Esper-Lite without the heavier routing features used in the full Esper deployment.

---
File: docs/design/detailed_design/09.1-oona-internals.md
---
# Oona Internals (Esper-Lite)

## Scope
Implementation details for the lightweight messaging bus. Focuses on Redis Streams usage, circuit breakers, and TTL cleanup.

## Publish Path
```python
def publish(envelope: EventEnvelope):
    with publish_breaker.protect():
        stream = 'oona.emergency' if envelope.priority in (PRIORITY_EMERGENCY, PRIORITY_CRITICAL) else 'oona.normal'
        redis_client.xadd(stream, {'payload': envelope.SerializeToString()}, maxlen=config.stream_maxlen)
```
- Envelope validated (schema version, size <256 KB) before publishing.
- Max stream length enforces bounded memory; older entries trimmed automatically.

## Consume Path
```python
def consume(group: str, consumer: str, stream: str):
    entries = redis_client.xreadgroup(group, consumer, {stream: '>'}, count=config.batch_size, block=config.block_ms)
    for entry_id, data in entries:
        try:
            envelope = EventEnvelope.FromString(data['payload'])
            handler(envelope)
            redis_client.xack(stream, group, entry_id)
        except Exception:
            redis_client.xclaim(stream, group, consumer, min_idle_time=config.min_idle_ms, ids=[entry_id])
```
- Consumer groups provide at-least-once delivery; failed messages re-delivered after idle timeout.
- Emergency stream processed before normal stream when both have entries.

## Circuit Breakers
- Publish breaker trips on repeated Redis errors; fallback returns `success=False` so caller can retry/backoff.
- Consume breaker trips when handlers consistently fail; conservative mode pauses normal stream consumption for recovery window.

## TTL & Cleanup
- `TTLHousekeeper` runs every 10 minutes: trims streams to `stream_maxlen`, clears idle consumer groups, removes stale pending entries via `XPENDING`/`XDEL`.

## Metrics & Logging
- Publish/consume latency measured with `time.perf_counter()`; logged to Nissa via telemetry packets.
- Queue depth from `XLEN` exposed as `oona.queue.depth`.
- Breaker transitions logged with reason and stream.

This pared-down implementation keeps Oona dependable while remaining lightweight for Esper-Lite deployments.

