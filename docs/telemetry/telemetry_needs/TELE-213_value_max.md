# Telemetry Record: [TELE-213] Value Max

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-213` |
| **Name** | Value Max |
| **Category** | `value` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the maximum value prediction in the batch? Is the value function suffering from explosion that could lead to training instability?"

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
| **Units** | discounted cumulative reward (unbounded) |
| **Range** | `(-inf, inf)` |
| **Precision** | 2-4 decimal places for display |
| **Default** | `0.0` |

### Semantic Meaning

> The maximum value prediction (V(s)) among all states in the current batch.
>
> In PPO, the critic network predicts the expected discounted cumulative reward for each state.
> This metric detects value function explosion, where the critic predicts unreasonably high values
> due to:
> - Numerical instability (NaN propagation, unbounded gradient flow)
> - Reward scaling issues (rewards too large before normalization)
> - Architecture imbalance (value head learning too aggressively)
> - Critic overfitting to outlier states
>
> Formula: V_max = max(V(s)) for all s in batch
> Computed as: `values.max()` where values = critic network output

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `abs(value_max) < 5000` | Value predictions within reasonable range |
| **Warning** | `abs(value_max) >= 5000` and `< 10000` | Potential explosion developing, monitor closely |
| **Critical** | `abs(value_max) >= 10000` | Severe explosion detected, immediate attention required |

**Threshold Source:** Metric specification, health_status_panel.py lines 386-389

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, value function forward pass and statistics computation |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` |
| **Line(s)** | 578, 910-924 |

```python
# Line 578: Values extracted from policy evaluation
values = result.value  # Float tensor of shape [batch_size]

# Lines 910-924: Value statistics computed and batched into single GPU→CPU transfer
values.max()  # Computed at line 910, aggregated at line 924
metrics["value_max"].append(logging_tensors[10])  # Index 10 in stacked tensor
```

**Context:** The `values` tensor contains the critic network's predictions for all valid (non-masked) states in the current PPO batch. The max operation detects outliers that could indicate training instability.

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update_event()` extracts from metrics dict | `simic/telemetry/emitters.py:741-852` |
| **2. Collection** | Packed into `PPOUpdatePayload.value_max` field | `leyline/telemetry.py:723-729, 845` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` unpacks payload | `karn/sanctum/aggregator.py:930-934` |
| **4. Delivery** | Written directly to `TamiyoState.value_max` field | `karn/sanctum/schema.py:986` |

```
[PPOAgent.update()] --values.max()--> [metrics["value_max"]]
  --emit_ppo_update_event()--> [PPOUpdatePayload.value_max]
  --handle_ppo_update()--> [TamiyoState.value_max]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `value_max` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_max` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 986 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py:285-305` | Displayed as "[min,max]" in "Value Range" row |
| HealthStatusPanel (status detection) | `widgets/tamiyo_brain/health_status_panel.py:357-391` | Used for threshold checks (lines 386-389) to determine "ok"/"warning"/"critical" status |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent.update()` computes values and calls max() at line 910
- [x] **Transport works** — Event includes value_max in PPOUpdatePayload field (leyline/telemetry.py:729, 845)
- [x] **Schema field exists** — `TamiyoState.value_max: float = 0.0` (schema.py:986)
- [x] **Default is correct** — 0.0 appropriate before first PPO update (schema.py:986)
- [x] **Consumer reads it** — HealthStatusPanel accesses `tamiyo.value_max` at lines 292, 298, 386, 388
- [x] **Display is correct** — Value renders as part of "[min,max]" range display with appropriate formatting
- [x] **Thresholds applied** — HealthStatusPanel checks abs(value_max) > 10000 (critical) and > 5000 (warning) at lines 386-389

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | value_max computed in update | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | ppo_update_populates_value_max | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | value_max_reaches_widget | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 20`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Value Range [min,max]" row
4. Verify value_max updates after each PPO batch
5. Monitor status color (should be "ok" in normal training)
6. Artificially trigger high value predictions (via modified environment rewards) to verify warning/critical coloring

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Policy forward pass | computation | Critic network must produce valid tensor output |
| Valid value predictions | data | Values must be finite (not NaN/Inf) after network forward pass |
| PPO update cycle | event | Only populated after first PPO update completes |
| Masked value filtering | operation | Values filtered by valid_mask at line 584 to exclude invalid timesteps |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-212` value_min | telemetry | Paired with value_min to compute value_range for range-based diagnostics |
| `TELE-214` value_mean | telemetry | Part of value function statistics cluster |
| `TELE-215` value_std | telemetry | Part of value function statistics cluster |
| HealthStatusPanel status | display | Drives visual status indicator when thresholds exceeded |
| `initial_value_spread` (schema.py:987) | computation | Used to compute relative threshold ratios after warmup (line 949) |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Audit | Created with full wiring verification |
| | | |

---

## 8. Notes

> **Design Decision:** Value statistics (min, max, mean, std) are always emitted together as a cluster to enable joint analysis of value function health. Individual metrics in isolation are less useful than the cluster pattern.
>
> **Data Stability:** During the first 50 PPO batches (warmup phase), value predictions are often unstable due to random initialization. The initial_value_spread is only captured after WARMUP_BATCHES=50 (aggregator.py:947-951) to establish a baseline for relative thresholds.
>
> **Explosion Detection:** The absolute fallback thresholds (5000/10000) are conservative by design. Real-world PPO rarely produces values >100 in stable training. Thresholds >1000 indicate something is fundamentally wrong. For relative thresholds, a 10x increase over initial_value_spread suggests either:
>   - Environment reward scale changed unexpectedly
>   - Critic is overfitting to outlier states
>   - Value loss is not clipping effectively (value_clip=True should be checked)
>
> **GPU Transfer Optimization:** value_max is computed in a single GPU→CPU transfer (line 911) along with 9 other metrics to minimize synchronization overhead. This is critical for training speed since value GPU transfer happens every PPO epoch.
>
> **Relationship to value_min:** The value_range = value_max - value_min is the primary diagnostic. A range of 1000 indicates explosion; a range <0.1 indicates collapse. Both extremes are unhealthy.
>
> **Future Improvements:** Consider per-head value statistics for multi-head architectures (similar to per-head entropy in TELE-001). Currently all heads share a single critic, but if critics diverge in future versions, per-head V_max would improve diagnostics.
