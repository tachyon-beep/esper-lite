# Urza — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Immutable catalogue & WAL | `08-urza.md`, `08.1` | Versioned metadata + artifact storage with WAL durability | `src/esper/urza/library.py` (SQLite + JSON WAL) | Partially Implemented | Must‑have | WAL present; no CRC/O_DSYNC semantics; single‑entry WAL. |
| Multi‑tier caching | `08.1` | L1 memory, L2 Redis, L3 object store | In‑process LRU only | Missing | Should‑have | Single tier; adequate for prototype, lacks Redis/object store. |
| Integrity checksums | `08.1` | SHA‑256 for artifacts and metadata | `UrzaLibrary.save` persists checksum; `UrzaRuntime.fetch_kernel` verifies; prefetch validates before READY | Implemented | Must‑have | Artifacts verified on load; checksum mismatches emit errors. |
| Query API (tags/stage/tier) | `08-urza.md` | Query by id/tag/stage/tier | `library.fetch_by_tier()`, `get()` | Partially Implemented | Should‑have | Tag/stage queries absent; tier supported in memory. |
| Latency SLOs | `08-urza.md` | p50 <10 ms, p95 <200 ms via cache; breaker on timeout | `library.metrics_snapshot()['lookup_latency_ms']`; no breakers | Partially Implemented | Should‑have | Measures lookup latency; no breakers/SLO enforcement. |
| TTL cleanup & retention | `08.1` | TTL for caches; retention for metadata | `UrzaLibrary(cache_ttl_seconds=...)` expires artifacts by mtime (configurable via settings) | Partially Implemented | Nice‑to‑have | Artifact TTL supported; tiered caches remain future work. |
| Telemetry & health | `08-urza.md` | `urza.query.*`, `urza.cache.*`, breaker state | Library metrics included in Weatherlight telemetry; prefetch metrics exposed | Partially Implemented | Should‑have | Base metrics available; breaker/conservative mode still TODO. |
| Kernel prefetch bus | `08-urza.md` | Consume kernel requests, emit ready/error messages | `src/esper/urza/prefetch.py`, `tests/urza/test_prefetch.py` | Implemented | Must‑have | Async worker loads artifacts, responds via Oona streams. |
| Leyline as canonical | `00-leyline-shared-contracts.md` | Use Leyline for descriptors | `esper.karn.BlueprintDescriptor` | Implemented | Must‑have | Uses Leyline types throughout. |
| UrzaRuntime load path | `08-urza.md` | Load compiled kernel into nn.Module | `src/esper/urza/runtime.py` | Implemented | Should‑have | Measures fetch latency; no cache tier metrics. |
