# Urabrask Prototype Metrics

Focus: BSDS‑Lite production/consumption and crucible readiness. Names are stable for prototype; adjust in full design if Leyline adds canonical messages.

## Producer (Urabrask)
- `urabrask.bsds.issued_total` (count)
- `urabrask.bsds.failed_total` (count)
- `urabrask.crucible.duration_ms` (ms)
- `urabrask.benchmark.latency_ms{profile}` (ms) — optional

## Urza Library
- `urza.library.slow_queries` (count) — existing
- `urza.library.integrity_failures` (count) — existing

## Tamiyo (Consumer)
- `tamiyo.policy.risk_score` (ratio) — existing
- `tamiyo.gnn.feature_coverage` (ratio) — existing
- `tamiyo.bsds.hazard_high_total` (count) — derived from events
- `tamiyo.bsds.hazard_critical_total` (count) — derived from events

## Nissa Alerts (proposed)
- `tamiyo_bsds_hazard_critical` — CRITICAL → PagerDuty
- `tamiyo_bsds_hazard_high` — HIGH after 3 consecutive → Slack

