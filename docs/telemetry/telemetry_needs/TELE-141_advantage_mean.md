# Telemetry Record: [TELE-141] Advantage Mean

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-141` |
| **Name** | Advantage Mean (Normalization Centering) |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the advantage normalization centered around zero, or is the GAE estimate systematically shifted?"

Advantage mean should be near zero after normalization. The normalization step `(A - mean(A)) / (std(A) + ε)` explicitly centers advantages at zero. If mean is significantly non-zero, it suggests:
- Poor GAE computation (bias in return estimation)
- Numerical instability in the advantage centering operation
- Systematic reward bias in the environment

A near-zero mean indicates the value function is well-calibrated and returns are approximately centered.

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging advantage distribution issues)
- [x] Researcher (analyzing policy gradient quality)
- [ ] Automated system (informational only, not critical for intervention)

### When is this information needed?

- [x] Real-time (every PPO update/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating issues)
- [x] Post-hoc (offline analysis of training runs)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | Advantage scale (dimensionless, post-normalization) |
| **Range** | `(-inf, +inf)` (typically `[-0.5, +0.5]` for healthy training) |
| **Precision** | 3 decimal places for display (to distinguish small non-zero from zero) |
| **Default** | `0.0` (before first PPO update) |

### Semantic Meaning

> The **mean of the generalized advantage estimate (GAE)** after normalization.
>
> Computed as:
> - Raw advantages: `A_t = Σ(γλ)ⁿ δ_tⁿ` (GAE λ-return residuals)
> - Normalized: `A_norm = (A - mean(A)) / (std(A) + ε)`
> - Reported: `mean(A_norm)` across all valid transitions in the batch
>
> **Healthy Range:** Approximately **-0.05 to +0.05** after proper normalization.
> **Semantics:**
> - **~0.0 (within ±0.05):** Normalization working correctly; value function well-calibrated
> - **+0.1 to +0.5:** Systematic positive bias; advantages shifted rightward (possible reward shift)
> - **-0.1 to -0.5:** Systematic negative bias; advantages shifted leftward (value function overestimating)
> - **>|0.5|:** Severe bias; normalization centering may be broken or GAE computation has systematic error

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `\|value\| < 0.1` | Normalization centered correctly; value function well-calibrated |
| **Warning** | `0.1 ≤ \|value\| < 0.3` | Minor systematic bias; monitor for value function drift |
| **Critical** | `\|value\| ≥ 0.3` | Severe bias; suggests value function scaling or return distribution issues |

**Note:** No TUIThresholds constants defined yet for advantage_mean. Thresholds above are derived from normalization theory (post-centering mean should be ~0).

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO batch update, after computing normalized advantages |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._perform_ppo_update()` (lines 425–452) |
| **Line(s)** | 425–452 (advantage stats computation) |

**Code Snippet:**
```python
# Compute advantage stats for status banner diagnostics
# These indicate if advantage normalization is working correctly
valid_advantages_for_stats = data["advantages"][valid_mask]
if valid_advantages_for_stats.numel() > 0:
    adv_mean = valid_advantages_for_stats.mean()  # Line 430
    adv_std = valid_advantages_for_stats.std()
    # ... skewness/kurtosis computation ...
    adv_stats = torch.stack([adv_mean, adv_std, adv_skewness, adv_kurtosis]).cpu().tolist()
    metrics["advantage_mean"] = [adv_stats[0]]  # Line 445
else:
    # No valid advantages - use NaN to signal "no data"
    metrics["advantage_mean"] = [float("nan")]  # Line 455
```

### Transport

| Stage | Mechanism | File | Lines |
|-------|-----------|------|-------|
| **1. Emission** | PPO update completion, metrics dict created | `ppo.py` | 425–459 |
| **2. Collection** | Event payload construction via `PPOUpdatePayload` | `leyline/telemetry.py` | 635 |
| **3. Aggregation** | Aggregator updates TamiyoState | `karn/sanctum/aggregator.py` | 825 |
| **4. Delivery** | Schema field assignment | `karn/sanctum/schema.py` | 854 |

**Flow Diagram:**
```
[PPOAgent._perform_ppo_update()]
  └─ computes: adv_mean = mean(normalized_advantages)
     └─ metrics["advantage_mean"] = [adv_mean]
        └─ PPOUpdatePayload(advantage_mean=..., ...)
           └─ VectorizedEmitter._emit()
              └─ [EventHub]
                 └─ SanctumAggregator._handle_ppo_update()
                    └─ self._tamiyo.advantage_mean = payload.advantage_mean
                       └─ snapshot.tamiyo.advantage_mean
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `advantage_mean` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.advantage_mean` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | 854 (field definition) |

**Schema Definition:**
```python
# From TamiyoState (line 854–865)
@dataclass
class TamiyoState:
    # Advantage statistics (from PPO update)
    # Post-normalization stats (should be ~0 mean, ~1 std if normalization working)
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_skewness: float = float("nan")
    advantage_kurtosis: float = float("nan")
```

### Consumers (Display)

| Widget | File | Usage | Lines |
|--------|------|-------|-------|
| **HealthStatusPanel** | `widgets/tamiyo_brain/health_status_panel.py` | Displayed as `mean±std` with color coding based on std | 59, 71, 394–402 |

**HealthStatusPanel Display:**
```python
# Line 59: Get status color based on advantage_std thresholds
# (Note: mean doesn't have its own status function, uses std status)
adv_status = self._get_advantage_status(tamiyo.advantage_std)

# Line 71: Display formatted as mean±std
f"{tamiyo.advantage_mean:+.3f}±{tamiyo.advantage_std:.2f}",
style=self._status_style(adv_status),

# Example output: "+0.025±0.87" in green, "-0.015±0.95" in green, "+0.150±0.92" in yellow
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPOAgent computes advantage mean during _perform_ppo_update()
- [x] **Transport works** — Event includes advantage_mean field (PPOUpdatePayload, line 635)
- [x] **Schema field exists** — TamiyoState.advantage_mean: float = 0.0 (line 854)
- [x] **Default is correct** — 0.0 is sensible default (pre-update state)
- [x] **Consumer reads it** — HealthStatusPanel accesses snapshot.tamiyo.advantage_mean (line 71)
- [x] **Display is correct** — Rendered as `+mean±std` format in HealthStatusPanel
- [ ] **Thresholds applied** — No dedicated color thresholds for mean (only std); mean is informational

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_advantage_stats_computed` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_advantage_mean` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_advantage_mean_reaches_tui` | `[x]` |
| Widget (HealthStatusPanel) | `tests/karn/sanctum/widgets/test_health_status_panel.py` | `test_advantage_mean_display_format` | `[x]` |

### Manual Verification Steps

1. **Setup:** Start training with: `uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 5`
2. **Open Sanctum:** Launch TUI (auto-opens or `uv run sanctum` in separate terminal)
3. **Locate metric:** Navigate to **HealthStatusPanel** in Tamiyo Brain section
4. **Verify updates:** Watch the "Advantage" row for advantage_mean±advantage_std
5. **Verify formatting:**
   - Values should display with `+/-` sign (e.g., `+0.025±0.87`)
   - Mean should be close to zero in healthy training (typically `[-0.05, +0.05]`)
   - Color coding follows advantage_std thresholds (green when std is healthy)
6. **Verify edge cases:**
   - Early training: may show `[---,---]` if no valid advantages yet (NaN display)
   - During training: should stabilize to small non-zero values
   - Training drift: if mean drifts beyond ±0.3, indicates value function miscalibration

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after PPO update completes |
| Valid advantage data | computation | Requires at least one valid transition in batch |
| Normalized advantages | computation | Assumes `data["advantages"]` are post-normalization `(A - mean(A)) / (std(A) + ε)` |
| Valid mask | event | Filters out terminal/invalid transitions before mean computation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-140` advantage_std | telemetry | Pair display: mean±std indicates full distribution health |
| `TELE-142` advantage_skewness | telemetry | Complementary: skewness + mean indicate distribution asymmetry |
| `TELE-143` advantage_kurtosis | telemetry | Complementary: kurtosis + mean indicate tail behavior |
| HealthStatusPanel.adv_display | display | Drives "Advantage mean±std" row formatting |
| Training operator observation | system | Non-zero persistent mean triggers investigation of value function calibration |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation and full wiring verification |
| | | Confirmed end-to-end flow from PPO → Sanctum UI |
| | | Identified that mean uses std thresholds, no dedicated mean thresholds yet |

---

## 8. Notes

### Design Decisions

1. **NaN for missing data:** When valid_advantages_for_stats is empty (no valid transitions in batch), we emit `float("nan")` instead of 0.0. This distinguishes "genuinely zero mean" from "no data available". The UI treats NaN as "unknown" rather than "zero".

2. **Post-normalization only:** The schema tracks the **normalized** advantage mean (after subtracting the original mean and dividing by std). This should mathematically be exactly 0.0 per the normalization formula. Non-zero values indicate numerical precision issues or actual deviations from the expected normalization.

3. **3-decimal display precision:** HealthStatusPanel uses `.3f` formatting (line 71) to distinguish small non-zero values (e.g., `-0.005`) from true zero (`0.000`). This sensitivity helps detect subtle value function calibration drift.

4. **Paired display with std:** Advantage mean is always displayed alongside `advantage_std` in the format `mean±std`. Following ML convention, this shows both location and spread at a glance.

5. **No dedicated color thresholds:** Currently, the color status of the advantage row is driven by `_get_advantage_status(advantage_std)` only. Advantage mean is informational. Future improvements could add dedicated mean thresholds (e.g., `ADVANTAGE_MEAN_BIAS_WARNING = 0.1`).

### Known Issues

1. **Empty batch edge case:** If a batch has only terminal transitions (all invalid_mask == False), advantage_mean will be NaN. This is rare but can happen in early training or with very low success rates. The UI gracefully handles this by displaying "---".

2. **Numerical precision:** Due to finite floating-point precision, the normalized mean may not be exactly 0.0 even with correct normalization. Values in the range [-1e-6, 1e-6] are effectively zero and not indicative of problems.

3. **No thresholds yet:** Unlike advantage_std, there are no TUIThresholds constants for advantage_mean. Implementing diagnostic thresholds (e.g., warn if |mean| > 0.1) would require additional configuration.

### Future Improvements

1. **Dedicated thresholds:** Add `ADVANTAGE_MEAN_WARNING = 0.1` and `ADVANTAGE_MEAN_CRITICAL = 0.3` constants to TUIThresholds, and implement `_get_advantage_mean_status()` in HealthStatusPanel for independent color coding.

2. **Mean velocity:** Track `d(advantage_mean)/d(epoch)` to detect whether bias is growing, shrinking, or stable (similar to entropy velocity in TELE-002).

3. **Correlation with value loss:** Add a computed metric showing whether non-zero mean correlates with high value loss (would indicate value function scaling issues).

4. **Per-action breakdown:** Track advantage_mean per action head (e.g., mean for GERMINATE vs mean for WAIT) to detect action-specific value estimation bias.

### Related Telemetry

- **TELE-140** (Advantage Std): Measures advantage spread; advantage_mean measures bias/centering
- **TELE-001** (Policy Entropy): Measures exploration breadth; advantage_mean measures credit assignment centering
- **TELE-011** (Explained Variance): Measures value function quality; non-zero mean + high EV suggests value scaling is off
- **TELE-142-PRE** (Pre-norm Advantage Mean): Raw mean before normalization; for diagnosing whether bias is in returns or value function
- **TELE-142** (Advantage Skewness): Distribution asymmetry; use alongside mean to interpret skew direction
- **TELE-143** (Advantage Kurtosis): Tail heaviness; use alongside mean to understand extreme advantage values

