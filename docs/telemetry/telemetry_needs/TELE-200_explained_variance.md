# Telemetry Record: [TELE-200] Explained Variance

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-200` |
| **Name** | Explained Variance |
| **Category** | `value` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is the value function learning to predict returns accurately, or is it providing no advantage over random estimation?"

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
| **Units** | proportion (normalized 0.0 to 1.0) |
| **Range** | `[0.0, 1.0]` — EV can theoretically exceed 1.0 on training set but 0-1 is normal |
| **Precision** | 3 decimal places for display |
| **Default** | `0.0` (before first PPO update) |

### Semantic Meaning

> Explained Variance (EV) measures how well the value function from rollout collection predicts returns:
>
> **EV = 1 - (Var(returns - values) / Var(returns))**
>
> - **EV = 1.0:** Value function perfectly predicts returns (ideal)
> - **EV = 0.5:** Value function explains 50% of return variance; 50% is noise or unexplainable
> - **EV = 0.0:** Value function provides no advantage over REINFORCE (random/constant estimator)
> - **EV < 0.0:** Value function is worse than random (pathological — indicates bug or data corruption)
>
> EV is computed **before** PPO updates (not after) to measure rollout quality from the policy's perspective. Post-update EV would measure updated value against stale returns (less meaningful).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.5 < value <= 1.0` | Value function strong; accurate return prediction |
| **Good** | `0.3 < value <= 0.5` | Value function learning but still developing |
| **Warning** | `0.0 <= value <= 0.3` | Value function weak; most variance unexplained |
| **Critical** | `value < 0.0` | Value function worse than random (bug/corruption) |

**Threshold Source:** `src/esper/karn/constants.py` — `TUIThresholds.EXPLAINED_VAR_WARNING = 0.3`, `TUIThresholds.EXPLAINED_VAR_CRITICAL = 0.0`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, computed from value function predictions vs actual returns |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` |
| **Line(s)** | 392-407 |

```python
# Compute explained variance BEFORE updates (intentional - standard practice)
# This measures how well the value function from rollout collection predicts returns.
# Post-update EV would measure the updated value function against stale returns,
# which is less meaningful. High pre-update EV (>0.8) indicates good value estimates.
valid_values = data["values"][valid_mask]
valid_returns = data["returns"][valid_mask]
var_returns = valid_returns.var()
explained_variance: float
if var_returns > 1e-8:
    ev_tensor = 1.0 - (valid_returns - valid_values).var() / var_returns
    explained_variance = ev_tensor.item()
else:
    explained_variance = 0.0
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | Included in PPOUpdateMetrics dict | `simic/agent/ppo.py` (line 407) |
| **2. Collection** | PPOUpdateMetrics typed dict with `explained_variance` key | `simic/agent/types.py` (line 57) |
| **3. Aggregation** | Extracted from PPOUpdatePayload in aggregator event handler | `karn/sanctum/aggregator.py` (line 803-806) |
| **4. Delivery** | Written to `snapshot.tamiyo.explained_variance` | `karn/sanctum/schema.py` (line 836) |

```
[PPOAgent.update()]
  --metrics["explained_variance"]-->
  [emit_ppo_update_event()]
  --PPOUpdatePayload.explained_variance-->
  [SanctumAggregator.handle_ppo_update()]
  --_tamiyo.explained_variance-->
  [SanctumSnapshot.tamiyo.explained_variance]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `explained_variance` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.explained_variance` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 836 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| PPOLossesPanel | `widgets/tamiyo_brain/ppo_losses_panel.py` (lines 107-116) | Displayed as gauge row labeled "Expl.Var" with color-coded status (min=-1.0, max=1.0) |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` (lines 203, 226) | Used for critical/warning status detection; drives failure/warning banner display |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent.update()` computes explained_variance at line 392-407
- [x] **Transport works** — Value is collected in `metrics["explained_variance"]` dict, passed to `emit_ppo_update_event()`
- [x] **Schema field exists** — `TamiyoState.explained_variance: float = 0.0` at line 836
- [x] **Default is correct** — `0.0` is appropriate default before first PPO update
- [x] **Consumer reads it** — Both PPOLossesPanel and StatusBanner directly access `snapshot.tamiyo.explained_variance`
- [x] **Display is correct** — PPOLossesPanel renders as gauge with sparkline; StatusBanner uses for status thresholding
- [x] **Thresholds applied** — Both widgets use `TUIThresholds.EXPLAINED_VAR_WARNING` (0.3) and `TUIThresholds.EXPLAINED_VAR_CRITICAL` (0.0)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_vectorized.py` | Implicit in PPO update tests | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_fields` | `[ ]` |
| Integration (end-to-end) | `tests/karn/sanctum/test_backend.py` | Telemetry flow integration | `[ ]` |
| Widget (PPOLossesPanel) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_losses_panel.py` | Gauge rendering | `[ ]` |
| Widget (StatusBanner) | `tests/karn/sanctum/widgets/tamiyo_brain/test_status_banner.py` (line 28) | Status detection logic | `[x]` |

### Manual Verification Steps

1. Start training: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 10`
2. Launch Sanctum TUI (opens automatically or `PYTHONPATH=src uv run python -m esper.karn.sanctum`)
3. Observe PPOLossesPanel — top row "Expl.Var" should show gauge with current EV value
4. Verify color coding: green (0.3-1.0), yellow (0.0-0.3), red (<0.0)
5. Observe StatusBanner critical/warning status updates:
   - At batch ~10-20, EV typically climbs from 0.0 toward healthy range
   - If EV stays near 0.0 after 100 batches, indicates value function not learning
   - If EV goes negative, indicates serious value prediction bug
6. After training, query telemetry: `SELECT explained_variance FROM ppo_updates ORDER BY batch DESC LIMIT 10`

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first complete PPO update |
| Value function | neural module | Requires valid value head forward pass |
| Rollout returns | computation | Requires complete episode returns for variance calculation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| StatusBanner status | display | Drives critical/warning status indicator |
| PPOLossesPanel color | display | Determines gauge color and visual warning level |
| Value health alerts | system | Could trigger auto-intervention if EV remains critical |
| Experiment analysis | research | Used in post-hoc analysis of value function convergence |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation and wiring verification |

---

## 8. Notes

> **Design Decision:** Explained variance is computed **before** PPO weight updates (not after). This is intentional and standard practice. Pre-update EV measures how well the old value function predicted returns in the current rollout. Post-update EV would measure the updated value function against the same stale returns, which is less meaningful for assessing learning quality.
>
> **Warmup Period:** During the first 50 batches, EV values may be unstable or artificially low/high due to:
> - Random initialization of value function
> - Limited data accumulation for variance calculation
> - Potential numerical instability (var_returns < 1e-8 threshold at line 400)
>
> **Edge Case:** If `var_returns` is extremely small (< 1e-8), EV is set to 0.0 rather than attempting division. This prevents NaN from constant returns (all trajectories yield same total reward).
>
> **Thresholds Rationale:**
> - **0.3 (Warning):** Below this, value function is not providing strong guidance for advantage estimation. PPO can still work but with higher gradient variance.
> - **0.0 (Critical):** At or below this, value function is at best useless, at worst harmful (worse than random). Indicates architectural issue, reward scale mismatch, or data leak.
>
> **Wiring Status:** Fully wired and operational. Both producer (PPOAgent) and consumers (widgets) correctly implement the metric. No known issues or broken chains.
