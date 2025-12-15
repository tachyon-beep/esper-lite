# Telemetry Overwatch Phase 2 (Deeper Coverage, Moderate Effort)

Purpose: wire the medium-effort items that improve explainability and health visibility without heavy compute cost. Builds on Phase 1.

## Scope (Phase 2 inclusions)
- Gates and lifecycle diagnostics:
  - Emit gate-evaluated events with gate id, pass/fail, and failed checks/reasons.
  - Add per-slot health fields to lifecycle payloads or periodic slot snapshots: gradient health/vanish/explode, isolation_violations, seed_gradient_norm_ratio.
- Rewards and policy clarity:
  - Provide reward breakdown visibility at ops-normal (summary or sampled debug events).
  - Emit per-head/mask hit rates for op/slot/blueprint/blend (aggregated per batch/epoch).
- Diagnostics:
  - Add optional DiagnosticTracker cadence hook in PPO/heuristic loops; when enabled, emit snapshots (likely via `ANALYTICS_SNAPSHOT` payload) every K epochs.
- Counterfactual robustness:
  - Add “counterfactual unavailable” reason codes and ensure per-slot baselines are attempted; consider lightweight per-slot ablation at final epoch for multi-slot runs when feasible.

## Nice-to-have (include if effort stays moderate)
- Policy/state grad_norm surrogate aligned to PPO updates (if not fully done in Phase 1).
- Action distribution broken down by success vs masked for additional debug fidelity.

## Out of Scope (later phases)
- Memory/perf warnings, governor periodic health snapshots.
- Reward hacking detector (`REWARD_HACKING_SUSPECTED`), command events.
- Heavy counterfactual matrix/Shapley integrations beyond the lightweight ablations.

## Deliverables
- New event types/payloads for gates and per-slot health.
- Aggregated per-head/mask metrics and reward summaries at ops-normal cadence.
- DiagnosticTracker hook and emission plumbing behind config flag.
- Updated specs (`docs/specifications/telemetry-audit.md`) and remediation checklist reflecting completed items.

## Acceptance
- Gate pass/fail with reasons visible in backends; slot health fields appear where lifecycle data is consumed.
- Reward summaries and per-head/mask hit rates show up in ops-normal telemetry without excessive overhead.
- DiagnosticTracker can be toggled on and emits snapshots on the configured cadence without breaking main training loops.
- Counterfactual telemetry includes unavailability reasons; optional lightweight ablation runs complete without destabilizing throughput.
