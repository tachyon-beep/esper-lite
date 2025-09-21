# Nissa — Prototype Delta (Observability)

Executive summary: the prototype implements a Nissa ingestor that consumes telemetry via Oona, increments Prometheus counters, indexes structured documents to Elasticsearch (or a local stub), evaluates basic alert rules, tracks SLO samples/burn rate, exposes `/metrics`, `/healthz`, and a metrics summary API, and includes a long‑running service runner. The full design specifies circuit breakers and conservative mode in ingestion/output, TTL housekeeping for logs/metrics, richer telemetry (ingest/export latency, dropped counts), alert routing stubs, and a broader mission‑control surface. Leyline remains canonical for telemetry and envelope contracts.

Outstanding Items (for coders)

- Ingestion breakers & conservative mode
  - Add circuit breakers around Oona consume, ES index, and Prom pushes; on repeated faults, enter conservative mode (e.g., buffer or sample-only) and escalate severity.
  - Pointers: `src/esper/nissa/service_runner.py::_ingest_loop`, add breaker state + telemetry.

- Latency & drop metrics
  - Capture ingest latency (Oona→Nissa), ES index latency, and dropped counts; expose via Prom and telemetry summary.
  - Pointers: `src/esper/nissa/observability.py` (ingest paths), extend counters to histograms.

- Mission‑control surface
  - Extend `/metrics/summary` with active alerts (already) plus ack/silence endpoints and SLO breach snapshots; optional WS for live updates.
  - Pointers: `src/esper/nissa/server.py` (FastAPI app), `alerts.py`, `slo.py`.

- Alert routing stubs
  - Implement Slack/Email/PagerDuty stubs (log-only) and route via `AlertRouter` with per-rule config; add tests.
  - Pointers: `src/esper/nissa/alerts.py` (router/engine).

- TTL housekeeping
  - Add retention/TTL cleanup for the in‑memory ES stub and any local buffers to prevent growth in long runs.
  - Pointers: `src/esper/nissa/service_runner.py::MemoryElasticsearch` and ingestion buffers.

- Telemetry publishing
  - Periodically publish Nissa health/metrics to Oona (for central dashboards) using a small packet builder with severity derived from alert/burn state.
  - Pointers: add `metrics_snapshot()` and packet builder; wire into service runner.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps (leyline‑first, no tech debt)

Design sources:
- `docs/design/detailed_design/10-nissa-unified-design.md`
- `docs/design/detailed_design/10.1-nissa-metrics-telemetry.md`
- `docs/design/detailed_design/10.2-nissa-mission-control.md`
- `docs/design/detailed_design/10.3-nissa-alerting-slo.md`

Implementation evidence (primary):
- `src/esper/nissa/observability.py`, `src/esper/nissa/alerts.py`, `src/esper/nissa/slo.py`, `src/esper/nissa/server.py`, `src/esper/nissa/service_runner.py`
- Tests: `tests/nissa/test_observability.py`
