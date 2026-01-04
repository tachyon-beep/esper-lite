# Telemetry Record: [TELE-144] Advantage Positive Ratio

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[ ] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-144` |
| **Name** | Advantage Positive Ratio |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "What fraction of advantages are positive? Is exploration balanced between favorable and unfavorable actions, or is learning one-sided?"

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
| **Units** | ratio (0.0 to 1.0) |
| **Range** | `[0.0, 1.0]` |
| **Precision** | 1 decimal place for display (rendered as percentage) |
| **Default** | `NaN` (no data available) |

### Semantic Meaning

> The fraction of advantages in the batch that are positive (greater than zero).
>
> Healthy ratio is 0.4-0.6 (40-60%), indicating balanced exploration:
> - 50% positive = perfectly balanced updates on both favorable and unfavorable actions
> - <40% = learning biased toward unfavorable actions (reward shape issue or easy regions)
> - >60% = learning biased toward favorable actions (reward shape issue or hard regions)
>
> Extreme values (<20% or >80%) indicate severe imbalance that suggests fundamental problems with:
> - Reward design (too sparse or misaligned)
> - Environment difficulty distribution (some regions always bad, others always good)
> - Value function convergence (overestimating or underestimating certain states)
>
> Formula: `advantage_positive_ratio = count(advantages > 0) / count(valid_advantages)`

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.4 <= value <= 0.6` | Balanced exploration, symmetric policy updates |
| **Warning** | `0.2 < value < 0.4` or `0.6 < value < 0.8` | Moderately imbalanced, monitor reward design |
| **Critical** | `value <= 0.2` or `value >= 0.8` | Severely imbalanced, likely reward/value collapse |

**Threshold Source:** Hard-coded in `health_status_panel.py` `_get_adv_positive_status()` method (lines 427-431)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO update step, after advantage normalization |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._compute_policy_loss()` |
| **Line(s)** | ~449-452 |

```python
# Fraction of positive advantages (healthy: 40-60%)
# Imbalanced ratios suggest reward design issues or easy/hard regions
adv_positive_ratio = (valid_advantages_for_stats > 0).float().mean().item()
metrics["advantage_positive_ratio"] = [adv_positive_ratio]
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | `emit_ppo_update()` — packages into PPOUpdatePayload | `simic/telemetry/emitters.py` |
| **2. Collection** | Event payload field `advantage_positive_ratio` | `leyline/telemetry.py` |
| **3. Aggregation** | `SanctumAggregator.handle_ppo_update()` — unpacks and stores | `karn/sanctum/aggregator.py` |
| **4. Delivery** | Written to `snapshot.tamiyo.advantage_positive_ratio` | `karn/sanctum/schema.py` |

```
[PPOAgent._compute_policy_loss()]
         ↓
[metrics["advantage_positive_ratio"]]
         ↓
[emit_ppo_update(metrics)]
         ↓
[PPOUpdatePayload.advantage_positive_ratio]
         ↓
[TelemetryEvent(type=PPO_UPDATE_COMPLETED, data=payload)]
         ↓
[hub.emit(event)]
         ↓
[SanctumAggregator.handle_ppo_update(payload)]
         ↓
[self._tamiyo.advantage_positive_ratio = payload.advantage_positive_ratio]
         ↓
[SanctumSnapshot.tamiyo.advantage_positive_ratio]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `advantage_positive_ratio` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.advantage_positive_ratio` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | ~865 |
| **Default Value** | `float("nan")` — signals "no data available" |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| HealthStatusPanel | `widgets/tamiyo_brain/health_status_panel.py` | Displayed as percentage with "+:" prefix (line 98) |
| StatusBanner | `widgets/tamiyo_brain/status_banner.py` (inferred) | May use advantage status for warning/critical detection |

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPOAgent computes ratio during advantage analysis
- [x] **Transport works** — Event includes advantage_positive_ratio field in PPOUpdatePayload
- [x] **Schema field exists** — `TamiyoState.advantage_positive_ratio: float = float("nan")`
- [x] **Default is correct** — NaN appropriate before first PPO update (signals "no data")
- [x] **Consumer reads it** — HealthStatusPanel accesses `snapshot.tamiyo.advantage_positive_ratio` (line 62, 94, 98)
- [x] **Display is correct** — Value renders as percentage with "+:" prefix (line 98: `f"{tamiyo.advantage_positive_ratio:.0%}"`)
- [x] **Thresholds applied** — HealthStatusPanel uses thresholds via `_get_adv_positive_status()` (lines 423-431)

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_vectorized.py` or `tests/simic/telemetry/test_emitters.py` | Advantage computation | `[ ]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | PPO update aggregation | `[ ]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | Metric reaches TUI | `[ ]` |
| Visual (TUI snapshot) | — | Manual verification | `[ ]` |

### Manual Verification Steps

1. Start training with: `uv run esper ppo --episodes 10`
2. Open Sanctum TUI (auto-opens or `uv run sanctum`)
3. Observe HealthStatusPanel "Advantage +:" row
4. Verify value displays as percentage (e.g., "55%") after first PPO update
5. Monitor for status color changes:
   - Green (ok): 40-60%
   - Yellow (warning): 20-40% or 60-80%
   - Red (critical): <20% or >80%
6. Trigger imbalance by manipulating reward signals or environment difficulty
7. Verify status color transitions in real-time

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after first PPO update completes |
| Advantage computation | computation | Requires valid advantage values (mask filtering out invalid steps) |
| Normalization | preprocessing | Advantages must pass through valid_advantages_for_stats mask |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| HealthStatusPanel status | display | Drives warning/critical coloring in training health view |
| StatusBanner status | display | May contribute to overall training status determination |
| User monitoring/alerts | operational | Manual inspection for training quality signals |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Created telemetry record, traced full data flow |

---

## 8. Notes

> **Design Decision:** The metric uses a simple fraction (count > 0 / total), not a weighted average. This keeps interpretation straightforward: if 55% of advantages are positive, 55% of updates were on favorable actions.
>
> **NaN Handling:** Default is `float("nan")` before the first PPO update, not 0.5 (neutral). This explicitly signals "no data" rather than "balanced by assumption." The widget renders NaN as "---" (dim text) to indicate missing data.
>
> **Healthy Range Rationale:** 40-60% is the target because:
> - Symmetric around 50% (perfect balance)
> - Allows ±10% natural fluctuation without warning
> - Leaves room for environment-driven asymmetry without false positives
>
> **Imbalance Interpretation:**
> - High ratio (>0.8): Agent sees mostly positive advantages → easy regions dominate OR reward shaped to encourage current behavior
> - Low ratio (<0.2): Agent sees mostly negative advantages → hard regions dominate OR agent learning from failures
> - Either extreme suggests either skewed reward design or highly non-uniform environment difficulty
>
> **Widget Integration:** HealthStatusPanel displays as "+:55%" inline with other advantage statistics (mean, skewness, kurtosis). The status contribution is combined with other advantage metrics to determine overall training health status via `worst_status` aggregation (line 63-66).
>
> **No Constants in TUIThresholds:** Unlike entropy/gradient metrics, advantage_positive_ratio thresholds are hard-coded in the widget (`_get_adv_positive_status()`) rather than exposed as class constants. This is acceptable because thresholds are specific to this metric's 0-1 range and unlikely to change independently of advantage normalization strategy.
