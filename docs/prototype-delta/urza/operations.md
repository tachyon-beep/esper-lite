# Urza — Operations & Configuration (Prototype)

Purpose: give a clear checklist for wiring integrity, TTL, telemetry, and (optionally) cache tiers so coders can land improvements with minimal friction.

Environment and settings
- Existing (EsperSettings)
  - `URZA_DATABASE_URL` — SQLite URL (default `sqlite:///./var/urza/catalog.db`).
  - `URZA_ARTIFACT_DIR` — Artifact directory root (default `./var/urza/artifacts`).
- Recommended (add to settings; already supported by library constructor)
  - `URZA_CACHE_TTL_SECONDS` — Expire artifacts based on mtime; pass into `UrzaLibrary(cache_ttl_seconds=...)`.
- Optional (future)
  - `URZA_REDIS_URL` — L2 cache endpoint (if/when cache tiers are added).
  - `URZA_CACHE_SIZE` — In‑process LRU capacity.

Integrity (checksums)
- What: verify artifact SHA‑256 before serve to Kasmina.
- Where:
  - Persist checksum via Tezzeret `KernelCatalogUpdate` → UrzaLibrary extras (already stores extras; include `checksum`).
  - Verify on load in `UrzaLibrary.get` and/or `UrzaRuntime.fetch_kernel`:
    - Compute SHA‑256 for `artifact_path` and compare to stored.
    - On mismatch: increment `urza.integrity.checksum_mismatch`, delete record, and return None (or raise), then let Kasmina fall back.
- Telemetry: emit `urza.integrity.checksum_mismatch` and a `checksum_mismatch` event with blueprint_id.

TTL / retention
- Use `UrzaLibrary(cache_ttl_seconds=...)` to expire stale artifacts by mtime (implemented; tested).
- Add a periodic maintenance hook (e.g., run from Weatherlight housekeeping) to list and remove expired artifacts/rows to bound disk usage.
- Metrics: `urza.cache.expired`, `urza.cache.size`.

Telemetry (names and routing)
- Query/cache metrics (export periodically via Weatherlight):
  - `urza.query.duration_ms` (from `metrics_snapshot()['lookup_latency_ms']`)
  - `urza.cache.hits`, `urza.cache.misses`, `urza.cache.errors`, `urza.cache.expired`
  - `urza.cache.size` (records in LRU)
- Prefetch metrics (already available on worker):
  - `urza.prefetch.hits`, `urza.prefetch.misses`, `urza.prefetch.errors`, `urza.prefetch.latency_ms`
- Integrity:
  - `urza.integrity.checksum_mismatch`
- Breaker (optional):
  - `urza.breaker.state` (0=closed,1=half‑open,2=open), `urza.breaker.open_total`
- Events (TelemetryEvent examples):
  - `artifact_expired`, `checksum_mismatch`, `prefetch_ready`, `prefetch_error{reason}`

Weatherlight integration
- Add `urza.query.*` and `urza.cache.*` from `UrzaLibrary.metrics_snapshot()` in Weatherlight’s telemetry loop (similar to Oona/Urza prefetch metrics already included).
- Run TTL cleanup from Weatherlight housekeeping if you add a maintenance API.

Cache tiers (optional)
- L2 Redis: cache artifact bytes or small serialized modules keyed by blueprint_id + version; expire aggressively.
- L3 object store: out‑of‑scope for Esper‑Lite; local FS is sufficient. If needed later, gate behind a provider interface.

Touchpoints for coders
- `src/esper/urza/library.py`
  - Store `checksum` in extras when saving; verify in `get()`; expose counts in `metrics_snapshot()`.
- `src/esper/urza/runtime.py`
  - Optionally re‑verify checksum before `torch.load` (defence in depth) and record load latency.
- `src/esper/weatherlight/service_runner.py`
  - Include `urza.query.*` and `urza.cache.*` in periodic telemetry; call a future `library.maintenance()` to prune expired artifacts.

Safety defaults
- On checksum mismatch or missing artifact, prefer returning None/raising and let the caller (Kasmina) take the eager/identity fallback path with a WARNING event.

