# Urza — Implementation Roadmap (Closing the Delta)

Goal: move the prototype library toward the design without tech debt.

| Order | Theme | Key Tasks | Outcome |
| --- | --- | --- | --- |
| 0 | Prefetch bus | Async worker consumes kernel requests, emits ready/error updates | ✅ Oona-based prefetch in place |
| 1 | Integrity & WAL | Add SHA‑256 checksums for artifacts; include checksum + version in DB; WAL with CRC and atomic writes | Strong integrity and recovery |
| 2 | Query surface | Expose tag/stage queries; index metadata; simple search | Useful metadata queries for Tamiyo |
| 3 | Circuit breakers & latency SLO | Add query time guards, breaker states; optionally conservative mode (cache‑only) | Predictable performance under load |
| 4 | Telemetry | Emit `urza.query.duration_ms`, `urza.cache.hit_rate`, breaker state via Oona | Operator visibility (prefetch metrics already included in Weatherlight telemetry) |
| 5 | TTL/GC | Add TTL to in‑process cache; periodic GC | Bounded memory |
| 6 | Cache tiers (optional) | Add Redis tier; abstract cache provider; keep object store out of prototype | Better hit rates without over‑building |

Notes
- Keep Leyline descriptors canonical for all metadata.
- Avoid premature S3: local FS + optional Redis tier suffices for Esper‑Lite.

Acceptance Criteria
- Checksums stored and verified on load; WAL recovery tested.
- Tier/Stage/Tag query APIs present with tests.
- Telemetry + breaker guard query times; optional cache‑only conservative mode works.
