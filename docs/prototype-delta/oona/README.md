# Oona — Prototype Delta (Messaging Bus)

Executive summary: the prototype implements a Redis Streams client with publish/consume methods, optional HMAC signing/verification via a shared secret, basic priority routing (NORMAL → EMERGENCY when depth exceeds a threshold), an optional drop threshold for backpressure, consumer‑group ack semantics, retry/claim/dead-letter handling, TTL housekeeping for aged entries, circuit breakers with conservative mode routing, and richer metrics/health snapshots. Remaining work covers latency instrumentation and the production health surface, but Leyline remains canonical for the envelope and message enums.

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
