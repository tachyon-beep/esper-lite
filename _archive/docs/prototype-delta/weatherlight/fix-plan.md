# Weatherlight — Fix Plan (Coder Hand‑Off)

Problem: We lack a single process that boots and supervises the platform. Subsystems exist and are testable in isolation, but orchestration is spread across `scripts/run_demo.py` and ad‑hoc usage.

Status: Implemented (prototype scope)
- Service module: `src/esper/weatherlight/service_runner.py` with `run_service()` and `main()`.
- Console entrypoint: `esper-weatherlight-service` (pyproject configured).
- Ops docs and compose file: `infra/docker-compose.weatherlight.yml`.

Tasks (all completed; kept here for traceability)
1) Service skeleton
- Build `EsperSettings` and log resolved settings (mask secrets).
- Validate `ESPER_LEYLINE_SECRET` present; exit non‑zero if missing.
- Construct `OonaClient` with `StreamConfig` from settings; await `ensure_consumer_group()`.

2) Core workers
- Create `UrzaLibrary` (root from `URZA_ARTIFACT_DIR`, DB from `URZA_DATABASE_URL`).
- Start `UrzaPrefetchWorker.run_forever(interval_ms=200)` as an asyncio Task with exponential backoff on uncaught exceptions (cap at 30 s; jitter).
- Create `KasminaSeedManager` (runtime = `UrzaRuntime`); attach `KasminaPrefetchCoordinator` and call `.start()`.
- Start `TamiyoService.consume_policy_updates()` as a task.

3) Security polish
- On Oona consume paths for kernel messages, validate freshness window and de‑duplicate `request_id` within a TTL map; drop stale/replayed messages; counters recorded.

4) Telemetry
- Every 10 s, publish a `TelemetryPacket` with foundation metrics (task_count, open_breakers, backoff_seconds, last_error_ts, uptime_s) via Oona telemetry stream.

5) Shutdown
- Install SIGINT/SIGTERM handlers; cancel tasks; await graceful completion with a deadline; close Oona.

Nice‑to‑have
- Add `infra/docker-compose.weatherlight.yml` to run Redis + Weatherlight (+ Nissa optional).
- Health probe endpoint (simple TCP or HTTP) for liveness.

Guard‑rails
- Do not alter Leyline messages or enums.
- No code inside subsystem packages should change behaviourally; edits limited to adding the Weatherlight service.

Test Plan (executed ad‑hoc during bring‑up)
- Start/stop: creates Oona groups; tasks reach running; telemetry published.
- Integration: Redis up; `KernelPrefetchRequest` → `KernelArtifactReady`; Kasmina attaches.
- Negative: stale `issued_at` / replayed `request_id` → message dropped; counters increment.

References
- Oona: src/esper/oona/messaging.py (HMAC, groups, retries)
- Urza: src/esper/urza/prefetch.py (worker)
- Kasmina: src/esper/kasmina/prefetch.py (coordinator), src/esper/kasmina/seed_manager.py (attach)
- Tamiyo: src/esper/tamiyo/service.py (consumer)
- Settings: src/esper/core/config.py
