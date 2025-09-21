# Nissa — Prototype Delta (Observability)

Executive summary: the prototype implements a Nissa ingestor that consumes telemetry via Oona, increments Prometheus counters, indexes structured documents to Elasticsearch (or a local stub), evaluates basic alert rules, tracks SLO samples/burn rate, exposes `/metrics`, `/healthz`, and a metrics summary API, and includes a long‑running service runner. The full design specifies circuit breakers and conservative mode in ingestion/output, TTL housekeeping for logs/metrics, richer telemetry (ingest/export latency, dropped counts), alert routing stubs, and a broader mission‑control surface. Leyline remains canonical for telemetry and envelope contracts.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps (leyline‑first, no tech debt)

Design sources:
- `docs/design/detailed_design/10-nissa.md`
- `docs/design/detailed_design/10.1-nissa-metrics-telemetry.md`
- `docs/design/detailed_design/10.2-nissa-mission-control.md`
- `docs/design/detailed_design/10.3-nissa-alerting-slo.md`

Implementation evidence (primary):
- `src/esper/nissa/observability.py`, `src/esper/nissa/alerts.py`, `src/esper/nissa/slo.py`, `src/esper/nissa/server.py`, `src/esper/nissa/service_runner.py`
- Tests: `tests/nissa/test_observability.py`

