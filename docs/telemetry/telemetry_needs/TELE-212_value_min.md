# Telemetry Record: [TELE-212] Value Min

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-212` |
| **Name** | Value Min |
| **Category** | `value` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What is the minimum value prediction (lowest estimate of state value) in the current PPO batch? Is the value function outputting extreme negative predictions?"

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
| **Units** | Value estimate (reward prediction), same units as episode return |
| **Range** | `(-inf, inf)` — unbounded in theory, but typically `[-1000, 100]` in practice |
| **Precision** | 1 decimal place for display (e.g., `-3.0`) |
| **Default** | `0.0` |

### Semantic Meaning

> The minimum value prediction across all valid states in the current batch. Computed as:
>
> value_min = min(V(s) for all states s in batch)
>
> where V(s) is the value function output (scalar) for state s.
>
> This metric is essential for:
> - **Value collapse detection:** When value_min ≈ value_max, the value network has converged to a constant
> - **Value explosion detection:** When |value_min| or |value_max| becomes extreme (>10x initial spread), gradient signals are corrupted
> - **Value range analysis:** Together with value_max, shows the spread of value estimates and whether they're diverse

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | Range < 5x initial spread, CoV < 2.0 | Value network outputting diverse, stable estimates |
| **Warning** | Range 5-10x initial spread, CoV 2.0-3.0 | Value divergence developing, monitor closely |
| **Critical** | Range > 10x initial spread, CoV > 3.0 | Value explosion/collapse, training may fail |

**Threshold Source:** `HealthStatusPanel._get_value_status()` in `src/esper/karn/sanctum/widgets/tamiyo_brain/health_status_panel.py` (lines 357-391)

**Collapse Threshold:** `v_range < 0.1 and v_std < 0.01` → critical

**Coefficient of Variation:** `cov = std/|mean|`
- `cov > 3.0` → critical
- `cov > 2.0` → warning

**Relative Threshold (post-warmup):**
- `range/initial_spread > 10` → critical
- `range/initial_spread > 5` → warning

**Absolute Fallback (during warmup, <50 batches):**
- `range > 1000 or |max| > 10000` → critical
- `range > 500 or |max| > 5000` → warning

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | Value function forward pass during PPO batch update |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent.update()` |
| **Line(s)** | 909-910 |

```python
# Value function predictions (raw network output)
values = base_net_untyped.value_head(policy_features)  # Shape: (batch_size,)

# Batch statistics computed for metrics
# Later in update(), per-epoch metrics are averaged:
logging_tensors = torch.stack([
    ...
    values.min(),  # Index 9 → value_min
    values.max(),  # Index 10 → value_max
]).cpu().tolist()

metrics["value_min"].append(logging_tensors[9])
metrics["value_max"].append(logging_tensors[10])
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `PPOAgent.update()` computes min/max and stores in `metrics` dict | `simic/agent/ppo.py` (lines 909-924) |
| **2. Collection** | Metrics aggregated across epochs, then emitted via `TelemetryEmitter.emit_ppo_update()` | `simic/telemetry/emitters.py` (line 813) |
| **3. Aggregation** | Event payload received by `SanctumAggregator.handle_ppo_update()` | `karn/sanctum/aggregator.py` (line 933) |
| **4. Delivery** | Written directly to `TamiyoState.value_min` field | `karn/sanctum/schema.py` (line 985) |

```
[PPOAgent.update()]
  → values.min()
  → metrics["value_min"]
  → emit_ppo_update()
  → PPOUpdateEvent payload
  → SanctumAggregator.handle_ppo_update()
  → snapshot.tamiyo.value_min
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `value_min` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_min` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 985 |
| **Default** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` (line 298) | Displayed in "Value Range" row as `[min,max]` e.g., `[-3.0,15.0]` |
| StatusBanner | (if integrated) | May use for status detection (value explosion alarm) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — `PPOAgent.update()` computes `values.min()` at line 909
- [x] **Transport works** — Metrics dict passed to `emit_ppo_update()` payload
- [x] **Schema field exists** — `TamiyoState.value_min: float = 0.0` at line 985
- [x] **Default is correct** — `0.0` appropriate before first PPO update
- [x] **Consumer reads it** — `HealthStatusPanel._render_value_stats()` accesses `tamiyo.value_min` at line 291
- [x] **Display is correct** — Rendered as `[{v_min:.1f},{v_max:.1f}]` (e.g., `[-3.0,15.0]`)
- [x] **Thresholds applied** — Color coding via `_get_value_status()` (lines 357-391)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_value_min_max_computed` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_value_min_max` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_value_min_reaches_tui` | `[ ]` |
| Widget rendering | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_health_value.py` | Multiple tests (lines 18-222) | `[x]` |

**Test File Note:** Comprehensive widget rendering tests exist for value function display, verifying:
- Healthy values show "ok" status (test_healthy_values_show_ok)
- Exploding values show "critical" (test_exploding_values_show_critical)
- Collapsed values show "critical" (test_collapsed_values_show_critical)
- High coefficient of variation shows warning/critical (test_high_cov_shows_warning, test_extreme_cov_shows_critical)
- Range rendering works correctly (test_render_value_stats_shows_range_and_std)
- Critical alert indicator displayed (test_render_value_stats_critical_shows_alert)

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 20`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Value Range" row
4. Verify min and max update after each PPO batch
5. Early in training, expect wide range (e.g., `[-50.0, 100.0]`)
6. As value network converges, range should narrow but not collapse
7. Trigger collapse condition: Artificially fix value network output to constant
8. Verify widget shows "!" alert and red coloring

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Value network forward pass | computation | Requires valid policy forward pass with value head |
| Batch completion | event | Min/max computed across entire batch (not per-sample) |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-210` value_mean | telemetry | Paired display in Health panel |
| `TELE-211` value_max | telemetry | Forms the range `[value_min, value_max]` |
| `TELE-213` value_std | telemetry | Used for coefficient of variation (CoV = std/mean) |
| `TELE-214` initial_value_spread | metadata | Used to compute relative explosion ratio |
| HealthStatusPanel | widget | Drives display and status coloring |
| Value divergence detection | system | Used in `_get_value_status()` for alarm logic |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Claude | Created during telemetry audit (TELE-212) |
| 2025-12-29 | (Tamiyo Debugging Enhancements) | Value function metrics added to schema & aggregator |

---

## 8. Notes

> **Design Decision:** Value min/max are computed as simple min() and max() across the batch, not per-episode or rolling statistics. This captures the true range of value estimates in the current update, making it sensitive to sudden value changes.
>
> **Design Rationale (Value Function Metrics):** Per DRL expert review, value function quality is THE primary diagnostic for RL training failures. V-Return correlation + value range are more predictive of training success than policy metrics alone. The value network can show signs of divergence (explosion) or collapse before policy metrics degrade.
>
> **Warmup Period:** During the first 50 batches (warmup), initial_value_spread is not yet set. The widget uses absolute thresholds as a fallback:
> - critical: `range > 1000 or |max| > 10000`
> - warning: `range > 500 or |max| > 5000`
>
> After batch 50, if value spread has stabilized, initial_value_spread is set to the current spread, and relative thresholds take over.
>
> **Known Behavior:** With the current PPO network initialization, value networks often start with `v ≈ 0` and then quickly expand to reasonable ranges. Early batches may show large spreads before settling. This is normal and expected.
>
> **Future Improvement:** Could add per-environment value ranges if multi-env training shows divergent value distributions across environments. Currently, value_min/max are batch-level aggregates regardless of environment.
>
> **Related Metrics:** See also value_mean (TELE-210), value_std (TELE-213), value_max (TELE-211), and the nested `ValueFunctionMetrics` dataclass in TamiyoState which includes V-Return correlation, TD error, Bellman error, and return percentiles for deeper value diagnosis.
