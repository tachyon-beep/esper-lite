# Urza — Prototype Delta (Kernel Library)

Executive summary: the prototype implements a local Urza library backed by SQLite + filesystem artifacts with a small in‑process LRU cache and a JSON WAL for recovery. It supports save/get/list, is used by Tezzeret and Tamiyo tests, and powers Kasmina via `UrzaRuntime` to load compiled kernels. The full design specifies multi‑tier caching (memory/Redis/object store), explicit integrity checksums, query circuit breakers, TTL cleanup, telemetry, and a metadata query surface (tags/stage/tier) with latency SLOs. Leyline remains canonical for blueprint descriptors.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps without tech debt

Design sources:
- `docs/design/detailed_design/08-urza.md`
- `docs/design/detailed_design/08.1-urza-internals.md`

Implementation evidence (primary):
- `src/esper/urza/library.py`, `src/esper/urza/runtime.py`, `src/esper/urza/pipeline.py`
- Tests: `tests/urza/*`, `tests/integration/test_blueprint_pipeline_integration.py`

