# Tamiyo — Field Reports (Per Decision)

Purpose
- Persist lightweight outcomes of control decisions for Simic and ops analysis.

When to Emit
- On every decision from `evaluate_step` (or `evaluate_epoch`), include reason and small metric deltas.

Content (minimal)
- command_id, training_run_id, seed_id (if any), blueprint_id (if any)
- outcome (enum), observation_window_epochs (counter)
- metrics_delta: loss_delta, breaker transitions, latency deltas

Durability
- Append to WAL (`FieldReportStore`); enforce retention window; rewrite file on prune
- Batch publish via existing `publish_history` path

Acceptance
- Report created for each decision; WAL retention keeps recent window only

Tests
- Unit: append + prune; Integration: presence of reports during a run and optional publish

References
- `src/esper/tamiyo/persistence.py`

## Checklist (for PR)
- [ ] Report emitted per decision
- [ ] WAL retention rewrites on prune
- [ ] Optional publish path exercised in integration
