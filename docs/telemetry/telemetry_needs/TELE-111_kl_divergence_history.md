# Telemetry Record: [TELE-111] KL Divergence History

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-111` |
| **Name** | KL Divergence History |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the KL divergence trend over recent batches? Is the policy drifting within acceptable bounds?"

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging)
- [x] Researcher (analysis)
- [ ] Automated system (alerts/intervention)

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
| **Type** | `deque[float]` |
| **Units** | nats (natural log base), dimensionless |
| **Range** | `[0.0, inf)` — typically `0.0 to 0.1+` in practice |
| **Precision** | 4 decimal places (stored), 3 for display |
| **Default** | Empty deque (maxlen=10) |

### Semantic Meaning

> Kullback-Leibler divergence (approx_kl) measures the magnitude of policy change during PPO updates. Computed as a weighted average of per-head KL divergences across action heads (slot, blueprint, tempo, tempo_offset, operations).
>
> **Formula:** For each action head:
> - KL = E[exp(log_ratio) - 1 - log_ratio], where log_ratio = new_policy_log - old_policy_log
> - Head KL is masked to only timesteps where that head is causally relevant
> - Final approx_kl is weighted average: Σ(head_kl × causal_weight) / Σ(causal_weights)
>
> **Interpretation:**
> - KL ≈ 0.0 = policy unchanged (trust region working)
> - KL ≈ 0.01-0.02 = moderate policy update (healthy)
> - KL > 0.03 = excessive drift (policy changing too fast, triggers early stopping at 1.5× target_kl)

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `kl < 0.015` | Policy maintaining trust region, stable updates |
| **Warning** | `0.015 <= kl < 0.03` | Mild policy drift, monitor for instability |
| **Critical** | `kl >= 0.03` | Excessive policy change, early stopping triggered |

**Threshold Source:** `TUIThresholds.KL_WARNING = 0.015`, `TUIThresholds.KL_CRITICAL = 0.03`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO policy update inner loop, computed after all policy gradient steps |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` (inner loop around line 742) |
| **Line(s)** | 730-743 |

```python
# Per-head KL computation with causal masking
kl_per_step = (torch.exp(log_ratio_clamped) - 1) - log_ratio_clamped
head_kl = (kl_per_step * mask).sum() / n_valid
causal_weight = n_valid / total_timesteps
weighted_kl_sum = weighted_kl_sum + causal_weight * head_kl

# Normalize to weighted average
approx_kl = (weighted_kl_sum / total_weight.clamp(min=1e-8)).item()
metrics["approx_kl"].append(approx_kl)
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.emit_ppo_update()` | `simic/telemetry/emitters.py:779-809` |
| **2. Collection** | Event payload with `kl_divergence` field (from `approx_kl`) | `leyline/telemetry.py:621` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` | `karn/sanctum/aggregator.py:813-815` |
| **4. Delivery** | Appended to `snapshot.tamiyo.kl_divergence_history` deque | `karn/sanctum/schema.py:950` |

```
[PPOAgent.update()]
  --approx_kl computed (line 742)-->
[metrics["approx_kl"].append()]
  --TelemetryEmitter.emit_ppo_update()-->
[PPOUpdatePayload.kl_divergence]
  --TelemetryEvent-->
[SanctumAggregator._handle_ppo_update()]
  --self._tamiyo.kl_divergence_history.append()-->
[TamiyoState.kl_divergence_history deque]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `kl_divergence_history` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.kl_divergence_history` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 950 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py:152-162` | Sparkline visualization (width=10) with trend arrow |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py:144-152` | Displays current KL with trend arrow and warning indicator |
| StatusBanner (detection) | `widgets/tamiyo_brain/status_banner.py:209-231` | Uses individual values for critical/warning status assessment |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent.update()` computes `approx_kl` during policy gradient steps
- [x] **Transport works** — Value flows through `TelemetryEmitter.emit_ppo_update()` → `PPOUpdatePayload` → event
- [x] **Schema field exists** — `TamiyoState.kl_divergence_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))`
- [x] **Default is correct** — Empty deque with maxlen=10 (stores last 10 KL values)
- [x] **Consumer reads it** — Both HealthStatusPanel and StatusBanner access `snapshot.tamiyo.kl_divergence_history`
- [x] **Display is correct** — Rendered as sparkline in HealthStatusPanel, trend arrow in StatusBanner
- [x] **Thresholds applied** — StatusBanner uses 0.015/0.03 thresholds for warning/critical coloring

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_approx_kl_computed` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_kl_divergence_history` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_kl_divergence_reaches_tui` | `[x]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 20`
2. Open Sanctum TUI
3. Observe HealthStatusPanel "KL" row with sparkline (shows last 10 KL values)
4. Observe StatusBanner for "KL" metric display with trend arrow
5. Trigger excessive policy drift by setting high learning rate to verify warning/critical coloring
6. Verify early stopping: reduce target_kl threshold to trigger early stopping at `approx_kl > 1.5 * target_kl`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Policy distribution | computation | Requires valid policy forward pass |
| Log probability ratios | computation | Per-head log-ratio between new and old policy |
| Causal masking | computation | Identifies which action heads are relevant per timestep |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HealthStatusPanel sparkline | display | Visualizes trend of last 10 KL values |
| StatusBanner KL metric | display | Shows current KL with trend indicator |
| StatusBanner critical detection | system | Drives CRITICAL status when KL > 0.03 |
| StatusBanner warning detection | system | Drives WARNING status when KL > 0.015 |
| Early stopping logic | algorithm | PPOAgent stops inner loop when `approx_kl > 1.5 * target_kl` |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Audit | Initial creation with full wiring verification |

---

## 8. Notes

> **Design Decision:** KL divergence is computed as a weighted average across all action heads (slot, blueprint, tempo, etc.) rather than per-head. This simplifies monitoring while accounting for the causal relevance of each head (some actions may not affect all heads). The weighting ensures that heads that don't apply to certain timesteps don't artificially inflate the aggregate KL.
>
> **Integration Detail:** The deque maintains a rolling window of the last 10 KL values with maxlen=10. When the 11th value is appended, the oldest value is automatically discarded. This design is optimal for sparkline rendering (fixed width=10) in HealthStatusPanel.
>
> **Policy Trust Region:** The 1.5× multiplier for early stopping (`if approx_kl > 1.5 * target_kl`) is standard in PPO implementations (OpenAI Baselines, Stable-Baselines3). It prevents the policy from drifting too far from the old policy within a single batch, which can cause training instability or off-policy violations.
>
> **Multi-Head Consideration:** Each action head (slot, blueprint, tempo, tempo_offset, operations) produces its own action distribution. The weighted-average KL accounts for the fact that some heads may not be causally relevant for all timesteps (masked out during computation). This prevents spurious KL contributions from irrelevant heads.
>
> **Known Behavior:** During the first few batches of training, KL values may be artificially high due to random initialization of the policy. The warmup period handling in StatusBanner should account for this if needed.
