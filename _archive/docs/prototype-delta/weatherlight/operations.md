# Weatherlight — Operations & Configuration

Environment variables (via EsperSettings)
- `REDIS_URL` — e.g. `redis://localhost:6379/0`
- `OONA_NORMAL_STREAM` — default `oona.normal`
- `OONA_EMERGENCY_STREAM` — default `oona.emergency`
- `OONA_TELEMETRY_STREAM` — default `oona.telemetry`
- `OONA_POLICY_STREAM` — default `oona.policy`
- `OONA_EMERGENCY_MAX_PER_MIN` — optional; set the emergency token bucket capacity for telemetry (default unlimited)
- `OONA_EMERGENCY_THRESHOLD` — optional; backlog threshold for rerouting to the emergency stream
- `OONA_MESSAGE_TTL_MS` — default `900000` (15 minutes); TTL for trimming streams
- `KERNEL_FRESHNESS_WINDOW_MS` — default `60000`; drop stale kernel requests
- `KERNEL_NONCE_CACHE_SIZE` — default `4096`; replay/nonce cache for kernel messages
- `URZA_DATABASE_URL` — default `sqlite:///./var/urza/catalog.db`
- `URZA_ARTIFACT_DIR` — default `./var/urza/artifacts`
- `ELASTICSEARCH_URL` — for Nissa (optional)
- `PROMETHEUS_PUSHGATEWAY` — for Nissa (optional)
- `ESPER_LEYLINE_SECRET` — REQUIRED for HMAC signing/verification

Runtime policies
- Fail‑fast on missing `ESPER_LEYLINE_SECRET` (prototype policy).
- Freshness window: reject kernel bus messages with `issued_at` older than 60 s (configurable later).
- Dead‑letter replication: after `retry_max_attempts` on a handler error; maintain counters.

Runbook
- Local: `esper-weatherlight-service`.
- Compose (optional): start Redis + Weatherlight (and optionally Nissa) with `docker compose -f infra/docker-compose.weatherlight.yml up -d`.
- Shutdown: Ctrl‑C → supervisor cancels tasks and drains consumers; check logs for “shutdown complete”.
- Manual queue reset: use `scripts/drain_telemetry_streams.py` to trim `oona.normal` / `oona.emergency` when Weatherlight is offline (dry-run support available).

Health and telemetry
- Periodic summary TelemetryPacket published to Oona telemetry stream:
  - `weatherlight.tasks.running`, `weatherlight.tasks.backing_off`
  - `weatherlight.last_error_ts`, `weatherlight.uptime_s`
  - `oona.publish/consume.latency_ms` (forwarded), queue depths
  - `urza.prefetch.{hits,misses,errors}`, `urza.prefetch.latency_ms`
  - (once implemented) `tezzeret.compilation.duration_ms{strategy}`, `tezzeret.prewarm.ms{strategy}`, `tezzeret.breaker.state`

Security notes
- Do not place binary artifacts on Oona; only references (`artifact_ref`, `checksum`, `guard_digest`).
- Treat `request_id` as a nonce across ready/error; de‑duplicate for TTL window.

Limitations (prototype)
- Single process; no HTTP health endpoint (log/telemetry only) unless added as a nice‑to‑have.
