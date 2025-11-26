# Oona — Traceability Map

| Design Assertion | Source | Implementation | Tests |
| --- | --- | --- | --- |
| Publish/consume with Redis Streams | `09.1-oona-internals.md` | `OonaClient.publish_*()`, `consume()` | `tests/oona/test_messaging.py::test_oona_publish_and_consume` |
| Priority routing to emergency | `09-oona.md` | `_resolve_stream()` emergency threshold | `tests/oona/test_messaging.py::test_oona_emergency_threshold`, `tests/oona/test_messaging_integration.py` |
| Stream trimming (maxlen) | `09.1` | `xadd(..., maxlen=...)` | `tests/oona/test_messaging.py::test_oona_max_stream_length_trims` |
| Backpressure drop and metrics | `09-oona.md` | `_resolve_stream()` drop path; `metrics_snapshot()` | `tests/oona/test_messaging.py::test_oona_backpressure_drop_threshold`, `::test_oona_backpressure_reroute_and_metrics` |
| HMAC signing/verification | `09-oona.md` | `_generate_signature()` and `_verify_payload()` | Covered implicitly; tested via presence of messages (set secret to exercise) |
| At‑least-once retry/claim; dead-letter | `09.1` | `_claim_stale_messages()`, `_handle_handler_error()` | `tests/oona/test_messaging.py::test_oona_retry_and_dead_letter` |
| Circuit breakers, conservative mode | `09-oona.md` | `_publish_breaker`, `_consume_breaker` | `tests/oona/test_messaging.py::test_oona_publish_breaker_records_failures` |
| TTL housekeeping | `09.1` | `OonaClient.housekeeping()` | `tests/oona/test_messaging.py::test_oona_housekeeping_trims_old_messages` |
