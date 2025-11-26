# Nissa — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Telemetry ingestion via Oona | `10.1` | Consume `TelemetryPacket` and write to Prom/ES | `NissaIngestor.consume_from_oona()`, `ingest_telemetry()` | Implemented | Must‑have | Covered by tests. |
| Metrics/log storage | `10-nissa.md` | Prometheus + Elasticsearch with retention | Counters + ES index calls | Partially Implemented | Should‑have | No retention/TTL enforcement. |
| Alerting | `10.3` | Threshold/rate alerts with routing stubs | `alerts.py` engine/router + default rules | Implemented | Should‑have | Basic routes logged; no external hooks. |
| SLO tracking | `10.3` | Error budget burn calculation; summary | `slo.py` tracker; summary endpoints | Implemented | Should‑have | Burn threshold used for breaches; works via metrics. |
| Mission control API | `10.2` | `/api/status`, `/api/metrics/summary`, WebSocket | `/metrics`, `/healthz`, `/metrics/summary` only | Partially Implemented | Should‑have | No `/api/status`/WebSocket; summary present. |
| Circuit breakers & conservative mode | `10.1` | Breakers around ingest/export; drop low‑priority in conservative | — | Missing | Must‑have | Not implemented. |
| TTL housekeeping | `10.1` | Periodic TTL cleanup (logs 7 d, metrics 30 d) | — | Missing | Should‑have | No TTL jobs; ES retention assumed external. |
| Telemetry for Nissa itself | `10-nissa.md` | `nissa.ingest.latency_ms`, `nissa.telemetry.dropped_total`, etc. | — | Missing | Should‑have | Prototype lacks self‑metrics. |
| Health surface | `10-nissa.md` | Health endpoint with queue lag, breaker, alerts | `/healthz` basic only | Partially Implemented | Nice‑to‑have | Extend with queue lag + alert count. |
| Leyline contracts | `10-nissa.md` | Use Leyline types and envelopes | `leyline_pb2.*` used | Implemented | Must‑have | Canonical usage. |

