# Tamiyo — Telemetry Taxonomy (3A)

Purpose
- Standardize metrics/events/indicators and priority mapping for Tamiyo’s decision loop.

Metrics (typical)
- `tamiyo.validation_loss` (loss)
- `tamiyo.loss_delta` (loss)
- `tamiyo.inference.latency_ms` (ms)
- `tamiyo.conservative_mode` (bool)
- `tamiyo.blueprint.risk` (score)
- `tamiyo.gnn.compile_enabled` (bool)
- `tamiyo.gnn.compile_fallback_total` (count; present when >0)
- `tamiyo.gnn.compile_warm_ms` (ms; CUDA warm-up latency when compile active)
- Decision counters by action type (optional)

Events (with severity)
- See `decision-taxonomy.md` for reasons and severities

Indicators (packet.system_health.indicators)
- `reason`, `priority`, `step_index` (when applicable), `policy`
- `timeout_budget_ms`, `metadata_timeout_budget_ms`

Routing
- CRITICAL/HIGH → Oona emergency stream; map severity→priority consistently

Acceptance
- Packets carry correct indicators; priorities match severity; Oona routing verified in tests
 - Compile status and (when applicable) warm-up latency reported under `tamiyo.gnn.*`

## Checklist (for PR)
- [ ] Metrics/events present as documented
- [ ] Indicators include reason/priority/step_index (when applicable)
- [ ] Priorities match severities; routing verified
