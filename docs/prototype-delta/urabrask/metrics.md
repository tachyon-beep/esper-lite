# Urabrask Prototype Metrics

Focus: BSDS‑Lite production/consumption and crucible readiness. Names are stable for prototype; adjust in full design if Leyline adds canonical messages.

## Producer (Urabrask)
- `urabrask.bsds.issued_total` (count)
- `urabrask.bsds.failed_total` (count)
- `urabrask.crucible.duration_ms` (ms)
- `urabrask.benchmark.latency_ms{profile}` (ms) — optional
- `urabrask.wal.append_errors_total` (count) — WAL append failures (fail‑open); alert on sustained non‑zero
- `urabrask.integrity_failures` (count) — signature verification failures (observability‑only)

## Urza Library
- `urza.library.slow_queries` (count) — existing
- `urza.library.integrity_failures` (count) — existing

## Tamiyo (Consumer)
- `tamiyo.policy.risk_score` (ratio) — existing
- `tamiyo.gnn.feature_coverage` (ratio) — existing
- `tamiyo.bsds.hazard_high_total` (count) — derived from events
- `tamiyo.bsds.hazard_critical_total` (count) — derived from events

## Bench (Worker)
- `urabrask.bench.profiles_total` (count)
- `urabrask.bench.failures_total` (count)
- `urabrask.bench.last_duration_ms` (ms)
- `urabrask.bench.last_processed` (count)
- `urabrask.bench.skipped_cooldown_total` (count) — skipped runs due to `URABRASK_BENCH_MIN_INTERVAL_S`

## Nissa Alerts (proposed)
- `tamiyo_bsds_hazard_critical` — CRITICAL → PagerDuty
- `tamiyo_bsds_hazard_high` — HIGH after 3 consecutive → Slack

## Suggested Alert Rules (Prototype)
- WAL Append Errors
  - Condition: increase(`urabrask.wal.append_errors_total[10m]`) > 0 for 3 consecutive intervals
  - Route: Slack
- BSDS Integrity Failures
  - Condition: increase(`urabrask.integrity_failures[10m]`) > 0 for 3 consecutive intervals
  - Route: Slack
