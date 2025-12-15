# Telemetry Overwatch Phase 3 (Deferred / Marginal Value)

Purpose: capture remaining low-priority or high-cost items. Execute after Phase 1–2 if value is justified and performance budget allows.

## Scope (Phase 3 candidates)
- Health and governance:
  - Emit `MEMORY_WARNING` and `PERFORMANCE_DEGRADATION` using allocator/dataloader metrics and governor stats.
  - Add governor periodic health/memory snapshots; optional governor dashboard hooks.
- Reward integrity:
  - Implement `REWARD_HACKING_SUSPECTED` detector (attribution/improvement ratio spikes, anomalous rent bypass).
- Command instrumentation:
  - Emit `COMMAND_*` events from CLI orchestration (start/complete/fail) for external controllers.
- Counterfactual depth:
  - Heavy counterfactual coverage (multi-slot ablation/Shapley) and corresponding telemetry when enabled.

## Nice-to-have
- Enhanced policy pulse metrics (e.g., optimizer state norms) if profiling shows headroom.
- Richer action distribution slices (by stage, success/fail, mask state) beyond Phase 2.

## Deliverables
- Wired emitters for memory/perf warnings, reward hacking detector, and command events.
- Optional governor health snapshot events and heavy counterfactual telemetry toggles.
- Spec and checklist updates to close remaining items.

## Acceptance
- New events emit only when explicitly enabled or when thresholds trip; no default throughput regressions.
- Governor health/memory snapshots integrate without destabilizing training loops.
- Reward hacking detector produces actionable signals with low false-positive rate (configurable).
- Heavy counterfactual telemetry can be toggled on/off; defaults keep Phase 1–2 performance intact.
