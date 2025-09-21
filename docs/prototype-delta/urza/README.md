# Urza — Prototype Delta (Kernel Library)

Executive summary: the prototype implements a local Urza library backed by SQLite + filesystem artifacts with a small in‑process LRU cache, checksum‑verified artifacts, and a JSON WAL for recovery. It supports save/get/list, is used by Tezzeret and Tamiyo tests, and powers Kasmina via `UrzaRuntime` to load compiled kernels (now verifying checksums). Single-tier caching includes optional TTL, while prefetchers validate checksums before publishing READY messages. Remaining design work covers multi‑tier caching (Redis/object store), query circuit breakers, richer telemetry, and metadata query surfaces (tags/stage/tier) with latency SLOs. Leyline remains canonical for blueprint descriptors.

Documents in this folder:
- `delta-matrix.md` — requirement‑by‑requirement status with evidence
- `traceability-map.md` — mapping of design assertions to code/tests
- `implementation-roadmap.md` — plan to close gaps without tech debt

Design sources:
- `docs/design/detailed_design/08-urza-unified-design.md`
- `docs/design/detailed_design/08.1-urza-internals.md`

Implementation evidence (primary):
- `src/esper/urza/library.py`, `src/esper/urza/runtime.py`, `src/esper/urza/pipeline.py`, `src/esper/urza/prefetch.py`
- Tests: `tests/urza/*`, `tests/integration/test_blueprint_pipeline_integration.py`

Status (prototype)
- Implemented: SQLite + FS artifacts with JSON WAL; in‑process LRU; get/save/list; query by tier; UrzaRuntime load path (latency measured); Oona prefetch worker with checksum/guard_digest and p50/p95 pre‑warm metrics computed from samples.
- Partial: artifact TTL (via cache_ttl_seconds) and cache metrics; latency measurement; telemetry export via Weatherlight (prefetch only); checksum verification not yet enforced on load; no tag/stage queries.
- Missing: multi‑tier caching (Redis/object store); circuit breakers; SLO enforcement; richer telemetry and health; signing/versioning.

Next Actions (minimal, high‑leverage)
- Add checksum verification on load
  - On `UrzaLibrary.save/get` and/or `UrzaRuntime.fetch_kernel`, compute SHA‑256 of the artifact and compare with the stored checksum (from Tezzeret’s `KernelCatalogUpdate`).
- Expose basic telemetry
  - Add an `emit_metrics_telemetry()` on a small Urza service wrapper or export via Weatherlight: `urza.query.duration_ms`, `urza.cache.{hits,misses,expired}`.
- Wire TTL knobs
  - Surface `cache_ttl_seconds` via `EsperSettings` for deployment control; keep default off.
- Breaker + conservative mode (optional)
  - If repeated IO errors or high latencies occur, switch to cache‑only mode and flag degraded health via telemetry.
- Query API expansion
  - Add tag/stage filters over metadata JSON; index in SQLite if needed.
