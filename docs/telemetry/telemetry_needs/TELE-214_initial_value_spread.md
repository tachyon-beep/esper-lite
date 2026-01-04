# Telemetry Record: [TELE-214] Initial Value Spread

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-214` |
| **Name** | Initial Value Spread |
| **Category** | `value` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What was the initial value range at training start? Used to compute relative explosion ratio (current_range / initial_spread) to detect value function divergence."

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
| **Type** | `float \| None` |
| **Units** | Absolute value range (value_max - value_min) |
| **Range** | `(0.0, inf)` — spread is positive by definition |
| **Precision** | 3 decimal places for display |
| **Default** | `None` — set after warmup, before that not available |

### Semantic Meaning

> Initial Value Spread captures the first stable estimate of the value function's range (difference between max and min predicted value across a batch). It is recorded once after the warmup period (50 batches) when the value function has stabilized from random initialization.
>
> **Formula:** `initial_spread = max(value_predictions) - min(value_predictions)` (at batch >= 50)
>
> This serves as a baseline for detecting value explosion: if current range grows much larger than initial spread, the value function is diverging catastrophically.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `initial_spread` set and `1.0 < ratio < 5.0` | Normal value range growth |
| **Warning** | `5.0 <= ratio <= 10.0` | Value function experiencing significant drift |
| **Critical** | `ratio > 10.0` or `initial_spread` never set | Severe value explosion or training never stabilized |

**Threshold Source:** Aggregator logic in `src/esper/karn/sanctum/aggregator.py:946-951`, consumed by `HealthStatusPanel._get_value_status()` in `src/esper/karn/sanctum/widgets/tamiyo_brain/health_status_panel.py:376-382`

**Ratio Calculation:** `ratio = (value_max - value_min) / initial_spread`

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, value function forward pass over batch |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._compute_value_loss()` (value prediction extraction) |
| **Line(s)** | ~920-924 (metrics collection) |

```python
# Value statistics collected in PPOAgent._train_epoch
metrics["value_min"].append(logging_tensors[9])
metrics["value_max"].append(logging_tensors[10])
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update_event()` emits value_min/value_max in payload | `simic/telemetry/emitters.py:811-814` |
| **2. Collection** | Event payload includes `value_min` and `value_max` fields | `leyline/telemetry.py` |
| **3. Aggregation** | Aggregator captures min/max, computes spread at batch 50 | `karn/sanctum/aggregator.py:930-951` |
| **4. Delivery** | Spread written to `snapshot.tamiyo.initial_value_spread` | `karn/sanctum/schema.py:987` |

```
[PPOAgent] --extract_values()--> [emit_ppo_update_event()] --event--> [Aggregator] --at_batch_50--> [TamiyoState.initial_value_spread]
```

**Key Detail:** This is NOT emitted by the agent; it's **computed by the aggregator** at batch >= 50:
- Batches 0-49: `initial_value_spread = None`
- Batch 50+: `initial_value_spread = (value_max - value_min)` if spread > 0.1, else remains `None`

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `initial_value_spread` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.initial_value_spread` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~987 |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Line 362-382: Computes relative value explosion ratio (`ratio = v_range / initial`) to determine status (ok/warning/critical) |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPO computes value_min/value_max during update
- [x] **Transport works** — Event includes value_min/value_max fields in PPOUpdatePayload
- [x] **Schema field exists** — `TamiyoState.initial_value_spread: float | None = None`
- [x] **Default is correct** — `None` appropriate before batch 50; set once when stable
- [x] **Consumer reads it** — HealthStatusPanel accesses `snapshot.tamiyo.initial_value_spread`
- [x] **Display is correct** — Value used for relative thresholds; status coloring applied correctly
- [x] **Thresholds applied** — HealthStatusPanel uses 5.0/10.0 ratio thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_value_stats_emitted` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_initial_value_spread_set_at_warmup` | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_value_explosion_detected_with_initial_spread` | `[ ]` |
| Widget (health status) | `tests/karn/sanctum/widgets/tamiyo_brain/test_ppo_health_value.py` | `test_exploding_values_show_critical`, `test_warning_at_5x_spread` | `[x]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. In HealthStatusPanel, observe "Value D" row
4. Run first 49 batches: `initial_value_spread` should be `None`, status should not use ratio thresholds
5. At batch 50+: `initial_value_spread` should populate with a number (e.g., 1.5)
6. Inject artificially large values to trigger value explosion
7. Verify warning/critical coloring appears when ratio exceeds thresholds

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| `value_min`, `value_max` | telemetry | Requires valid value function forward pass; computed in PPO update |
| Warmup period (batch 50) | event | Only populated after 50 batches of stable training |
| Value spread > 0.1 | condition | Only set if initial spread is non-trivial; prevents division by near-zero |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `HealthStatusPanel.value_status` | display | Determines ok/warning/critical coloring for "Value D" row |
| Value explosion detection | system | Relative thresholds (ratio > 10) used to flag divergence early |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2024-12-29 | Implementation | Added to aggregator as part of TamiyoBrainV2 value monitoring |
| 2025-01-03 | Audit | Verified wiring in telemetry audit; documented initial_value_spread behavior |

---

## 8. Notes

> **Design Decision:** Spread is captured once at batch 50 (after warmup) and **never updated**. This provides a stable baseline for comparing later divergence. If the value function is still chaotic before warmup, no baseline is set—this is intentional to avoid spurious thresholds.
>
> **Non-Trivial Threshold:** The condition `if spread > 0.1` prevents setting initial_value_spread for near-constant value functions. This avoids division-by-very-small-numbers when computing the ratio.
>
> **Relative vs Absolute:** HealthStatusPanel checks relative thresholds (ratio > 5/10) when `initial_value_spread` is known, falling back to absolute thresholds (max > 5000 / range > 500) during warmup when `initial_value_spread` is still `None`. This two-tier approach handles both early-training and stable-training scenarios.
>
> **Known Limitation:** If training starts with already-diverged values (unlikely but possible with bad initialization), the "initial" spread will still be captured at batch 50, making relative detection less sensitive. This is rare and acceptable.
>
> **Future Improvement:** Could track multiple "resets" if the value function is re-initialized mid-training (e.g., network reset in multi-phase training). Currently only the first stable spread is captured.

