# Oona — Prototype Delta (Messaging Bus)

Executive summary: the prototype implements a Redis Streams client with publish/consume methods, optional HMAC signing/verification via a shared secret, basic priority routing (NORMAL → EMERGENCY when depth exceeds a threshold), an optional drop threshold for backpressure, consumer‑group ack semantics, retry/claim/dead-letter handling, TTL housekeeping for aged entries, circuit breakers with conservative mode routing, and richer metrics/health snapshots. Remaining work covers latency instrumentation and the production health surface, but Leyline remains canonical for the envelope and message enums.

Outstanding Items (for coders)

- Latency instrumentation & histograms
  - Track rolling histograms/p50/p95 for publish/consume latency, not only last sample; include queue depth deltas per publish.
  - Pointers: `src/esper/oona/messaging.py` (`_publish`, `consume`, `metrics_snapshot`).

- Health surface
  - Add `health_snapshot`-backed telemetry packet publisher (periodic) and optional lightweight HTTP `/healthz` if needed.
  - Pointers: `src/esper/oona/messaging.py::health_snapshot/emit_metrics_telemetry`.

- Priority routing polish
  - Ensure `MessagePriority` from telemetry (and future command paths) maps to emergency routing consistently; add tests for NORMAL→EMERGENCY mapping under priority.
  - Pointers: `publish_telemetry(priority=...)`, `_resolve_stream`.

- Backpressure SLOs & alerts
  - Emit alerts/counters when backpressure reroute/drop occurs beyond thresholds; integrate with Nissa default alerts.
  - Pointers: `metrics_snapshot()` names (`publish_rerouted`, `publish_dropped`, `queue_depth_*`).

- Dead‑letter auditing
  - Add a small auditor that samples dead‑letters, counts by message type, and emits telemetry for operator action.
  - Pointers: `_handle_handler_error` path and `dead_letter_stream`.

- Kernel stream freshness tests
  - Expand tests for `kernel_freshness_window_ms` and nonce cache to include boundary/expiry and replay under load.
  - Pointers: `_enforce_kernel_freshness()` and OrderedDict caches.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps without tech debt

Design sources:
- `docs/design/detailed_design/09-oona-unified-design.md`
- `docs/design/detailed_design/09.1-oona-internals.md`

Implementation evidence (primary):
- `src/esper/oona/messaging.py`
- Tests: `tests/oona/*`, plus integration usages in Tamiyo/Tolaria/Simic/Nissa tests
- Kernel streams: `oona.kernels.requests`, `.ready`, `.errors` for prefetch UX
