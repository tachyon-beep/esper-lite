# Telemetry Record: [TELE-001] Policy Entropy

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-001` |
| **Name** | Policy Entropy |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is the policy maintaining healthy exploration, or is it collapsing to deterministic behavior?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [x] Automated system (alerts/intervention)

### When is this information needed?

- [x] Real-time (every batch/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [ ] Post-hoc (offline analysis)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | nats (natural log base) |
| **Range** | `[0.0, HEAD_MAX_ENTROPIES[head]]` — varies by action head |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Policy entropy measures the "spread" of the action distribution. Computed as:
>
> H(π) = -Σ π(a|s) log π(a|s)
>
> High entropy = exploring many actions equally. Low entropy = converging on specific actions.
> For categorical distributions, max entropy = log(n_actions).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `entropy > 0.3` | Policy maintaining exploration |
| **Warning** | `0.1 < entropy <= 0.3` | Exploration declining, monitor closely |
| **Critical** | `entropy <= 0.1` | Policy collapse imminent or occurring |

**Threshold Source:** `TUIThresholds.ENTROPY_WARNING = 0.3`, `TUIThresholds.ENTROPY_CRITICAL = 0.1`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after action probability computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._compute_policy_loss()` |
| **Line(s)** | ~450-460 |

```python
# Entropy computed from action distribution
entropy = dist.entropy().mean()
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.emit_ppo_update()` | `simic/telemetry/emitters.py` |
| **2. Collection** | Event payload with `entropy` field | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.tamiyo.entropy` | `karn/sanctum/schema.py` |

```
[PPOAgent] --emit_ppo_update()--> [TelemetryEmitter] --event--> [Aggregator] --> [TamiyoState.entropy]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `entropy` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.entropy` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~180 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Entropy trend display with velocity |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | Critical/warning status detection |
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py` | Displayed as metric with sparkline |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent` computes entropy during update
- [x] **Transport works** — Event includes entropy field
- [x] **Schema field exists** — `TamiyoState.entropy: float = 0.0`
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — All 3 widgets access `snapshot.tamiyo.entropy`
- [x] **Display is correct** — Value renders with appropriate formatting
- [x] **Thresholds applied** — StatusBanner uses 0.1/0.3 thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_entropy_computed` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_entropy` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_entropy_reaches_tui` | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Entropy D" row
4. Verify entropy value updates after each PPO batch
5. Artificially reduce entropy coefficient to verify warning/critical coloring

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Action distribution | computation | Requires valid policy forward pass |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-002` entropy_velocity | telemetry | Computed as d(entropy)/d(batch) |
| `TELE-003` collapse_risk_score | telemetry | Uses entropy + velocity for risk assessment |
| `TELE-004` entropy_clip_correlation | telemetry | Pearson correlation with clip_fraction |
| StatusBanner status | display | Drives FAIL/WARN status when critical |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-06-15 | Initial | Created with PPO implementation |
| 2024-09-20 | Refactor | Moved to TamiyoState from flat snapshot |
| 2025-01-03 | Audit | Verified wiring in telemetry audit |

---

## 8. Notes

> **Design Decision:** Entropy is averaged across the batch rather than per-sample to reduce noise. Individual sample entropy would be too volatile for meaningful monitoring.
>
> **Known Issue:** During warmup (first 50 batches), entropy may show artificially high values due to random initialization. The warmup period in StatusBanner accounts for this.
>
> **Future Improvement:** Consider per-head entropy breakdown (slot, blueprint, tempo, etc.) for more granular collapse detection.
