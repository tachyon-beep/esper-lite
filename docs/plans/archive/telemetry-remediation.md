# Telemetry Remediation Plan

> **Status:** Minimum viable telemetry is **COMPLETE** (Phase 1 + Phase 2). Phase 3 is deferred as marginal value.

## Consumer: Esper-Overwatch

This remediation work was driven by the **Esper-Overwatch** TUI requirements — an "Air Traffic Control" style monitoring interface that needs truthful, structured telemetry to render:

- **Policy Pulse:** PPO health metrics (KL, entropy, clip fraction, explained variance, grad norm, lr)
- **Flight Board:** Per-env status, throughput, last action, slot chip strips
- **Slot Chips:** Stage, blueprint, alpha, gate status, health indicators
- **Event Feed:** Structured lifecycle events with env_id/slot/seed attribution

See: [`esper_overwatch.md`](./esper_overwatch.md) (design philosophy) and [`overwatch-textual-ui.md`](./overwatch-textual-ui.md) (implementation plan).

---

## Implementation Status

| Phase | Status | Scope |
|-------|--------|-------|
| **Phase 1** | ✅ Complete | Low-risk telemetry: lifecycle-only mode, env_id/device context, alpha/epoch fields, PPO vitals, throughput metrics |
| **Phase 2** | ✅ Complete | UI truthfulness: gate visibility (`SEED_GATE_EVALUATED`), slot health fields, `fps` in throughput |
| **Phase 3** | ⏸️ Deferred | Marginal value: `MEMORY_WARNING`, `REWARD_HACKING_SUSPECTED`, `COMMAND_*`, heavy counterfactual/Shapley |

See: [`telemetry-phase1.md`](./telemetry-phase1.md), [`telemetry-phase2.md`](./telemetry-phase2.md), [`telemetry-phase3.md`](./telemetry-phase3.md)

---

## Minimum Viable Telemetry (Complete)

The following telemetry is **required** for Overwatch to function correctly:

### Slot/Lifecycle Events
- ✅ `SEED_STAGE_CHANGED` with alpha, inner_epoch, global_epoch
- ✅ `SEED_GATE_EVALUATED` with gate id, pass/fail, checks_passed, checks_failed, reasons
- ✅ `SEED_FOSSILIZED`, `SEED_CULLED` with full context
- ✅ Per-slot health fields: gradient_health, has_vanishing, has_exploding, isolation_violations, seed_gradient_norm_ratio
- ✅ env_id/device attribution on all lifecycle events

### Policy Pulse
- ✅ `PPO_UPDATE_COMPLETED` with lr, grad_norm, update_time_ms, kl_divergence, entropy, clip_fraction, explained_variance
- ✅ Action distribution summaries

### Throughput
- ✅ Per-env step_time_ms, fps, dataloader_wait_ms
- ✅ Counterfactual-unavailable markers with reason codes

### Infrastructure
- ✅ `--telemetry-lifecycle-only` mode for fast runs
- ✅ Warning event when telemetry disabled

---

## Deferred (Phase 3 — Marginal Value)

These items are defined but **not wired**. Event types exist in `leyline/telemetry.py` as forward-declared placeholders:

| Event Type | Intent | Why Deferred |
|------------|--------|--------------|
| `MEMORY_WARNING` | Alert on high VRAM usage via `torch.cuda.memory_stats` | Requires allocator hooks; marginal value vs. external monitoring |
| `PERFORMANCE_DEGRADATION` | Detect rolling accuracy drops | Governor already handles rollbacks; explicit event is redundant |
| `REWARD_HACKING_SUSPECTED` | Flag anomalous attribution/improvement ratios | Needs careful threshold tuning to avoid false positives |
| `COMMAND_START/COMPLETE/FAIL` | CLI orchestration events for external controllers | No current external controller integration |
| Heavy counterfactual (Shapley) | Multi-slot ablation with telemetry emission | Compute-heavy; Shapley exists in `karn/counterfactual.py` but not wired to telemetry |

### If You Need Phase 3

See [`telemetry-phase3.md`](./telemetry-phase3.md) for implementation guidance. Key criteria for justifying Phase 3 work:

1. **External monitoring integration** — If you need to push alerts to PagerDuty/Slack, `MEMORY_WARNING` and `PERFORMANCE_DEGRADATION` become valuable
2. **Multi-run orchestration** — If an external controller manages training runs, `COMMAND_*` events enable coordination
3. **Reward hacking research** — If investigating Goodhart effects, the detector provides structured signals

---

## Historical Context

This document originated as an audit of telemetry gaps found when designing the Overwatch UI. The full audit identified ~30 items across slots, policy, counterfactuals, health, and infrastructure.

**Phase 1** addressed low-risk, high-value items (CLI flags, env context, PPO vitals, throughput).

**Phase 2** closed UI truthfulness gaps (gate visibility, slot health fields, fps).

**Phase 3** contains remaining items where cost exceeds benefit for current use cases.
