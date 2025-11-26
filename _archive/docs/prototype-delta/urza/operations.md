# Urza — Operations & Configuration (Prototype)

Purpose: give a clear checklist for wiring integrity, TTL, telemetry, and (optionally) cache tiers so coders can land improvements with minimal friction.

Environment and settings
- Present in `EsperSettings`
  - `URZA_DATABASE_URL` — SQLite URL (default `sqlite:///./var/urza/catalog.db`).
  - `URZA_ARTIFACT_DIR` — Artifact directory root (default `./var/urza/artifacts`).
  - `URZA_CACHE_TTL_SECONDS` — Optional TTL (seconds) applied to on-disk artifacts.
- Optional (future)
  - `URZA_REDIS_URL` — L2 cache endpoint (if/when cache tiers are added).
  - `URZA_CACHE_SIZE` — In‑process LRU capacity.

Integrity (checksums)
- Urza persists the SHA‑256 provided by Tezzeret and verifies artifacts during `UrzaRuntime.fetch_kernel`.
- Prefetch worker cross-checks the checksum before responding; mismatches emit `checksum_mismatch` errors.
- Recommended telemetry counter: `urza.integrity.checksum_mismatch` (Weatherlight can include it once wired).

TTL / retention
- `UrzaLibrary(cache_ttl_seconds=...)` (configurable via settings) expires stale artifacts by mtime.
- Future: add maintenance hook to prune expired rows proactively and surface `urza.cache.expired`.

Telemetry (names and routing)
- Query/cache metrics (Weatherlight now includes `urza.library.*` in its telemetry packet):
  - `urza.library.cache_hits`, `urza.library.cache_misses`, `urza.library.cache_errors`, `urza.library.cache_expired`, `urza.library.lookup_latency_ms`
- Prefetch worker metrics: `urza.prefetch.hits`, `urza.prefetch.misses`, `urza.prefetch.errors`, `urza.prefetch.latency_ms`
- Integrity/TTL counters can be appended once maintenance hooks land.

Weatherlight integration
- Weatherlight aggregates `urza.library.*` metrics each cycle; TTL cleanup hook remains future work.

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
