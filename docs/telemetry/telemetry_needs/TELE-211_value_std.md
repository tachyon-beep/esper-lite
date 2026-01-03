# Telemetry Record: [TELE-211] Value Std

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-211` |
| **Name** | Value Std |
| **Category** | `value` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the value function predicting a stable range of values, or is it collapsing to constant predictions (all outputs clustered at one point)?"

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
| **Units** | Scalar value spread (unitless) |
| **Range** | `[0.0, inf)` — non-negative, typically in range [0.0, 5.0] |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> Standard deviation of value function outputs across the batch. Computed as:
>
> value_std = std([V(s_i) for i in batch])
>
> High std = value function outputs spread across wide range of predictions. Low std = value function predicting constant or near-constant outputs (collapse risk).
>
> value_std is paired with value_mean and value_min/max for coefficient-of-variation (CoV) calculation: CoV = value_std / |value_mean|. Used with range collapse detection (range < 0.1 AND std < 0.01) to identify value function failure.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `value_std > 0.01` (within CoV bounds) | Value function predicting diverse outputs |
| **Warning** | `value_std < 0.01` (small range + low std) | Potential value collapse developing, monitor range |
| **Critical** | `value_std < 0.01` AND `value_range < 0.1` | Value function collapsed to constant output |

**Threshold Source:** `PolicyThresholds.VALUE_STD_COLLAPSE = 0.01` in `/home/john/esper-lite/src/esper/karn/constants.py` (line 63)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, value function forward pass on batch |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._perform_ppo_update()` (inner loop) |
| **Line(s)** | ~908 (computed), ~922 (appended to metrics) |

```python
# Value function statistics computed from batch predictions
logging_tensors = torch.stack([
    policy_loss,
    value_loss,
    -entropy_loss,
    joint_ratio.mean(),
    joint_ratio.max(),
    joint_ratio.min(),
    joint_ratio.std(),
    values.mean(),
    values.std(),              # line 908: value_std computation
    values.min(),
    values.max(),
]).cpu().tolist()

metrics["value_std"].append(logging_tensors[8])  # line 922: stored in metrics dict
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `TelemetryEmitter.emit_ppo_update()` with metrics dict | `simic/telemetry/emitters.py` (line 812) |
| **2. Collection** | Event payload field `value_std` | `leyline/telemetry.py` (line 727) |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` assigns payload to state | `karn/sanctum/aggregator.py` (line 932) |
| **4. Delivery** | Written to `snapshot.tamiyo.value_std` | `karn/sanctum/schema.py` (line 984) |

```
[PPOAgent._perform_ppo_update()]
  --values.std()-->
[PPOAgent.metrics dict]
  --emit_ppo_update(metrics)-->
[PPOUpdateEvent.value_std]
  --Aggregator.handle_ppo_update(payload)-->
[TamiyoState.value_std]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `value_std` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_std` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 984 |

```python
# From TamiyoState definition
value_mean: float = 0.0
value_std: float = 0.0       # <-- line 984
value_min: float = 0.0
value_max: float = 0.0
```

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Displayed in "Value Range" row as "s=X.XX" (line 300) |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` | May reference for value collapse status detection |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `values.std()` computed in PPOAgent inner loop (line 908)
- [x] **Transport works** — Event includes `value_std` field (leyline/telemetry.py line 727)
- [x] **Schema field exists** — `TamiyoState.value_std: float = 0.0` (schema.py line 984)
- [x] **Default is correct** — 0.0 appropriate before first PPO update
- [x] **Consumer reads it** — HealthStatusPanel accesses `snapshot.tamiyo.value_std` (line 293)
- [x] **Display is correct** — Value renders as "s=X.XX" in Value Range row (line 300)
- [x] **Thresholds applied** — CollapseDetector uses VALUE_STD_COLLAPSE threshold (0.01) in triggers.py (line 245)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/telemetry/test_emitters.py` | `test_emit_ppo_update_populates_value_stats` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_value_metrics` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_value_stats_reach_widget` | `[ ]` |
| Widget (value collapse) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_health_value.py` | `test_collapsed_values_show_critical` | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Value Range" row
4. Verify value_std updates after each PPO batch update
5. Monitor that value_std changes in healthy range (typically > 0.01)
6. Artificially force value collapse by setting critic to constant (test scenario) to verify critical status coloring

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Value function forward pass | computation | Requires valid critic network predictions on batch |
| Batch statistics | computation | Depends on batch size affecting variance calculation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-206` value_mean | telemetry | Paired with value_std for CoV calculation (value_std / value_mean) |
| `TELE-207` value_min | telemetry | Paired with value_std for range collapse detection |
| `TELE-208` value_max | telemetry | Paired with value_std for range collapse detection |
| `CollapseDetector` system | system | Uses value_std < 0.01 threshold for value collapse alerts |
| HealthStatusPanel display | widget | Renders alongside value range and status indicator |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation, full wiring verified |

---

## 8. Notes

> **Design Decision:** Value std is computed as the standard deviation of value predictions across the entire batch, not per-sample. Per-sample value stds would be too granular and noisy for meaningful monitoring.
>
> **Related to CoV:** The widget computes Coefficient of Variation (CoV) as value_std / |value_mean| to detect instability independent of absolute value range. High CoV (>2.0) triggers warning status even if mean is small.
>
> **Collapse Detection:** Value collapse is a dual-condition check: both `v_range < 0.1` AND `v_std < 0.01` must be true to trigger critical status. This prevents false positives from momentary dips in variance. See HealthStatusPanel._get_value_status() lines 364-366.
>
> **Threshold Rationale:** The 0.01 threshold represents "essentially zero variance" — below this, the value function predictions are so clustered that the critic has failed to differentiate state values. This is slightly more stringent than mathematical zero to avoid floating-point precision artifacts.
>
> **Known Usage:** CollapseDetector in triggers.py maintains a rolling window of value_stds (window_size=3) and checks if average falls below threshold for robust collapse detection (see lines 245-256).
