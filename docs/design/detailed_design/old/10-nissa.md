# Nissa Combined Design

---
File: docs/design/detailed_design/10-nissa-unified-design.md
---
# Nissa Unified Design (Esper-Lite)

## Snapshot
- **Role**: Lightweight observability stack for Esper-Lite. Collects metrics from Oona feeds, exposes dashboards, raises basic alerts, and tracks SLOs.
- **Scope**: Metrics ingestion, telemetry forwarding, alert evaluation, and simple mission control APIs. Heavy analytics and advanced automation from full Esper are out of scope.
- **Status**: Production; retains C‑016 safety features (TTL cleanup, circuit breakers, conservative mode).

## Responsibilities
- Ingest telemetry (`TelemetryPacket`) from subsystems via Oona.
- Store metrics/logs in Prometheus + Elasticsearch (single-node) with retention policies.
- Provide mission-control REST endpoints/WebSocket stream for dashboards.
- Evaluate alert rules and error budgets; notify operators through email/PagerDuty stubs.

## Component Map
| Component | Purpose | Reference |
| --- | --- | --- |
| MetricsEngine | Scrape/ingest metrics into Prometheus | `10.1-nissa-metrics-telemetry.md` |
| TelemetryProcessor | Parse Oona events, fan-out to storage | `10.1` |
| AlertManagerLite | Evaluate alert rules & SLOs | `10.3-nissa-alerting-slo.md` |
| MissionControlAPI | Serve dashboards & health APIs | `10.2-nissa-mission-control.md` |
| CircuitBreakerLayer | Protect ingestion/output pipelines | `10.1` |
| TTLHousekeeper | Drop stale telemetry/alerts | `10.1` |

## Simplifications
- Single Prometheus + Elasticsearch instance; no federation.
- Limited alert types (threshold, rate-of-change, error budget burn). No ML anomaly detection.
- Mission control offers read-only dashboards plus simple controls (e.g., ack alert). No workflow automation.

## Data Flow
1. Oona pushes telemetry → TelemetryProcessor validates schema and writes to Prometheus/ES.
2. MetricsEngine exposes `/metrics` for scraping; dashboards read from Prometheus + ES.
3. AlertManagerLite evaluates rules every 30 s, emits notifications via Oona `alert.events` topic and optional webhooks.
4. MissionControlAPI aggregates status for UI/WebSocket clients.

## Reliability & Operations
- Circuit breakers guard ingestion/export; conservative mode skips low-priority telemetry when load high.
- TTL cleanup purges logs >7 days, metrics >30 days (configurable).
- Health endpoint reports queue lag, storage status, breaker state, active alerts count.
- Metrics: `nissa.ingest.latency_ms`, `nissa.alerts.active`, `nissa.breaker.state`, `nissa.telemetry.dropped_total`.

Nissa thus offers a slimmed-down observability surface appropriate for Esper-Lite while keeping the critical safety and monitoring hooks from the full platform.

---
File: docs/design/detailed_design/10.1-nissa-metrics-telemetry.md
---
# Nissa Metrics & Telemetry (Esper-Lite)

## Scope
Defines the ingestion pipeline for metrics/telemetry. Keeps C‑016 safeguards (circuit breakers, TTL cleanup) but removes advanced analytics.

## Ingestion Pipeline
```python
def process_telemetry(envelope: EventEnvelope):
    with ingest_breaker.protect():
        packet = TelemetryPacket.FromString(envelope.payload_data)
        validate_schema(packet)
        metrics_engine.write(packet)
        if packet.events:
            telemetry_log.store(packet)
```
- `metrics_engine` forwards counters/gauges/histograms to Prometheus (via remote-write or exporter).
- `telemetry_log` stores structured events in Elasticsearch with TTL.

## Scraping & Export
- `/metrics` endpoint exposes Prometheus scrape target.
- Downsampling handled by Prometheus recording rules (1 m/5 m). No hierarchical aggregation.
- Optional pushgateway support for batch jobs.

## Safeguards
- Ingestion breaker opens after 3 consecutive failures; fallback drops packet and increments `nissa.telemetry.dropped_total`.
- Conservative mode disables verbose telemetry categories when breaker half-open.
- TTL cleanup runs hourly: deletes logs older than 7 days, metrics older than 30 days.

## Configuration Snippet
```yaml
nissa:
  telemetry:
    redis_topic: telemetry.events
    scrape_interval_s: 15
    retention:
      metrics_days: 30
      logs_days: 7
  breakers:
    ingest: {failure_threshold: 3, timeout_ms: 30000}
  conservative_mode:
    drop_categories: ["debug", "trace"]
```

This simple flow keeps metrics and telemetry available for dashboards/alerts without the heavier tooling from the full Esper deployment.

---
File: docs/design/detailed_design/10.2-nissa-mission-control.md
---
# Nissa Mission Control (Esper-Lite)

## Scope
Provides a thin API and UI layer for viewing system status and acknowledging alerts. No complex automation.

## API Endpoints
| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/status` | Returns breaker states, queue lag, active alerts count. |
| GET | `/api/metrics/summary` | Aggregated KPIs (training throughput, error budget, top alerts). |
| POST | `/api/alerts/{id}/ack` | Ack alert, recording user + timestamp. |
| GET | `/ws/stream` | WebSocket delivering live telemetry snapshots (JSON). |

## Implementation Notes
- FastAPI (or Flask) service with auth middleware (basic token).
- WebSocket stream publishes small JSON payload every 5 s: `queue_depth`, `alert_count`, `conservative_mode` flags.
- Rate limiting prevents overuse (default 20 req/min per user).

## Safety & Ops
- Circuit breaker wraps outbound DB/Prom queries; fallback returns cached snapshot.
- TTL for session tokens (8 h); mission control stores audit logs in PostgreSQL.
- UI built with simple dashboard (Grafana or custom React) pointing at these endpoints.

Mission Control in Esper-Lite is intentionally minimal—operators can view status, acknowledge alerts, and monitor conservative mode without the heavier control-plane features of the full platform.

---
File: docs/design/detailed_design/10.3-nissa-alerting-slo.md
---
# Nissa Alerting & SLO (Esper-Lite)

## Scope
Defines the lean alerting and SLO framework. Focuses on threshold alerts, error budget burn, and simple routing.

## Alert Types
| Alert | Condition | Default Routing |
| --- | --- | --- |
| `training_latency_high` | Tolaria epoch hook >18 ms for 3 epochs | PagerDuty + Slack |
| `kasmina_isolation_violation` | ≥3 violations in 5 min | PagerDuty |
| `oona_queue_depth` | Queue depth >4 000 for 2 min | Slack |
| `tezzeret_conservative_mode` | Conservative mode >15 min | Email |

Rules evaluated every 30 s. Alerts include description, current value, suggested action.

## SLO Tracking
- Metrics: `availability`, `latency`, `error_rate` per subsystem.
- Error budget computed over 30-day window; burn alerts triggered when ≥40 % budget consumed in 24 h.
- Results stored in PostgreSQL table (`slo_snapshots`), exposed via `/api/metrics/summary`.

## Notification Pipeline
```python
def notify(alert):
    for channel in alert.routes:
        try:
            send(channel, alert)
        except Exception:
            dead_letter_queue.add(alert)
```
- Channels: Slack webhook, email SMTP stub, PagerDuty event API (configurable). Dead-letter queue persisted in Redis for manual replay.

## Configuration Example
```yaml
alerts:
  evaluation_interval_s: 30
  rules:
    - name: training_latency_high
      expr: tolaira_epoch_ms > 18
      for: 3m
      routes: [pagerduty, slack]
    - name: kasmina_isolation_violation
      expr: kasmina_isolation_violations_total >= 3
      for: 5m
      routes: [pagerduty]

slo:
  window_days: 30
  burn_alert_threshold: 0.4
```

This streamlined alerting/SLO layer provides the operational guardrails required for Esper-Lite without the advanced analytics from the full platform.
