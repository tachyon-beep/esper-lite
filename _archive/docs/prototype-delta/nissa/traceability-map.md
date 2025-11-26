# Nissa — Traceability Map

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Ingest telemetry and index to ES | `10.1-nissa-metrics-telemetry.md` | `NissaIngestor.ingest_telemetry()` → ES index | `tests/nissa/test_observability.py::test_ingest_telemetry_records_metrics_and_indexes` |
| Consume from Oona | `10-nissa.md` | `consume_from_oona()` with handler | `tests/nissa/test_observability.py::test_consume_from_oona_ingests_packets` |
| Prometheus metrics exposure | `10.1` | `registry`, `/metrics` endpoint | `tests/nissa/test_observability.py::test_metrics_endpoint_serves_prometheus` |
| Alert rules & routing | `10.3` | `AlertEngine` + `AlertRouter` + defaults | `tests/nissa/test_observability.py::test_training_latency_alert_triggers_after_consecutive_breaches` |
| SLO tracking and burn rate | `10.3` | `SLOTracker` and summary | `tests/nissa/test_observability.py::test_slo_summary_reports_burn_rate` |
| Circuit breakers, conservative mode, TTL | `10.1` | — | — |
| Mission control API breadth | `10.2` | Basic endpoints only | `tests/nissa/test_observability.py` (summary exists) |

