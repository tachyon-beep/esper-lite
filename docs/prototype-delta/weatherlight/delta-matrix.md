# Weatherlight — Delta Matrix

Status and evidence per requirement. See rubric in `../rubric.md`.

| Area | Design Source | Expected Behaviour | Prototype Evidence | Status | Severity | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Single foundation service | HLD 001; infra overview | One process boots and supervises core subsystems | `src/esper/weatherlight/service_runner.py`, console `esper-weatherlight-service` | Implemented | Must‑have | Async supervisor with backoff, telemetry, graceful shutdown.
| Oona group initialisation | 09‑oona; 00‑leyline | Ensure consumer groups/streams exist before start | OonaClient.ensure_consumer_group(): src/esper/oona/messaging.py | Implemented | Must‑have | Used in demo; Weatherlight should call on startup.
| HMAC/nonce/freshness | 09‑oona; Security | HMAC enforced; freshness window + replay guard | `OonaClient._verify_payload` + `_enforce_kernel_freshness()` | Implemented | Must‑have | Drops stale/replayed requests/responses; metrics recorded.
| Urza prefetch worker | 08‑urza | Run async prefetch loop (requests→ready/errors) | UrzaPrefetchWorker.run_forever(): src/esper/urza/prefetch.py | Implemented (needs host) | Must‑have | Weatherlight must own lifecycle and backoff.
| Kasmina prefetch coordinator | 02‑kasmina | Start ready/error consumers; notify SeedManager | KasminaPrefetchCoordinator: src/esper/kasmina/prefetch.py | Implemented (needs host) | Must‑have | Owned by Weatherlight.
| Tamiyo policy consumer | 03‑tamiyo | Start policy update consumer loop | TamiyoService.consume_policy_updates: src/esper/tamiyo/service.py | Implemented (needs host) | Should‑have | Weatherlight starts/stops it.
| Tezzeret catalog updates | 06‑tezzeret | Publish KernelCatalogUpdate (guards, latencies) | TezzeretCompiler.latest_catalog_update(): src/esper/tezzeret/compiler.py | Implemented | Should‑have | Weatherlight may forward updates via Oona (optional).
| Health & backoff | HLD 005; ADR‑001 | Per‑task backoff, circuit breakers, graceful shutdown | `WeatherlightService._worker_wrapper()` + Oona breakers | Implemented | Must‑have | Exponential backoff + jitter; restart counters; signal handling.
| Telemetry aggregation | 11‑nissa | Emit foundation metrics (counters/gauges) | `WeatherlightService._build_telemetry_packet()`; Oona metrics snapshot | Implemented | Should‑have | Publishes Weatherlight/Oona/Urza metrics to telemetry stream.
| Config via EsperSettings | Core config | All endpoints/streams/env through EsperSettings | src/esper/core/config.py | Implemented | Must‑have | Weatherlight reads/validates settings.
| Compose integration | infra | Single compose to run Redis + Weatherlight (+ Nissa opt.) | `infra/docker-compose.weatherlight.yml` | Implemented | Nice‑to‑have | Uses `esper-weatherlight-service` entrypoint.
