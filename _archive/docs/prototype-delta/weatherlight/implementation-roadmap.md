# Weatherlight — Implementation Roadmap

Status: Completed (prototype scope). Goal achieved: minimal, safe supervisor boots Oona, Urza prefetch, Kasmina prefetch coordination, and Tamiyo policy consumers. Nissa remains separate.

Slice 1 — Service skeleton (Must‑have) — Completed
- Add module `esper.weatherlight.service_runner` with `async def run_service()` and `def main()`.
- Load `EsperSettings`, validate `ESPER_LEYLINE_SECRET` present; fail fast if missing (prototype policy).
- Build `OonaClient` with `StreamConfig`; call `ensure_consumer_group()`.
- Wire structured logging and a process‑wide shutdown gate (SIGINT/SIGTERM).

Acceptance (met)
- Service starts, ensures Oona groups without error, and cleanly exits on signal.

Slice 2 — Core workers (Must‑have) — Completed
- Build `UrzaLibrary` and spawn `UrzaPrefetchWorker.run_forever()` with jittered backoff on failures.
- Build `KasminaSeedManager` and `KasminaPrefetchCoordinator.start()`; attach manager to prefetch coordinator.
- Start `TamiyoService.consume_policy_updates()` loop.
- Expose an in‑memory health snapshot of running tasks (counts, last error, backoff state).

Acceptance (met)
- Prefetch path functional end‑to‑end: requests → ready/errors → Kasmina attaches.
- Policy updates consumed without exceptions.

Slice 3 — Telemetry + security polish (Should‑have) — Completed
- Publish periodic `TelemetryPacket` via Oona with Weatherlight metrics: running_tasks, backoffs, last_errors, started_at, uptime_s.
- Enforce freshness window and replay guard for kernel prefetch bus messages (reject stale `issued_at`, replayed `request_id`).

Acceptance (met)
- Telemetry visible in Oona telemetry stream and ingested by Nissa.
- Stale/replayed messages are dropped with counters; freshness tunables exposed.

Slice 4 — Ops & packaging (Nice‑to‑have) — Completed
- Add console entrypoint `esper-weatherlight-service` in `pyproject.toml`.
- Provide `infra/docker-compose.weatherlight.yml` to run Redis + Weatherlight (+ Nissa optional) locally.
- Document env in `operations.md`.

Acceptance (met)
- `esper-weatherlight-service` runs locally via compose; graceful stops verified.

Out‑of‑scope (prototype)
- Multi‑process supervision (systemd/nomad/k8s). Run a single process; scale later.
- HTTP admin UI. Use logs and telemetry for now.
