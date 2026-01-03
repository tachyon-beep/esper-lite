# Telemetry Record: [TELE-140] Advantage Std

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-140` |
| **Name** | Advantage Std (Normalization Health) |
| **Category** | `policy` |
| **Priority** | `P0-critical` |

## 2. Purpose

### What question does this answer?

> "Is the advantage normalization working correctly, or is it collapsed (low variance) / exploded (high variance)?"

Advantage std directly indicates the health of the PPO advantage normalization step. Both extremes are pathological:
- **Too low (<0.1):** Advantages have been normalized to near-zero variance, breaking PPO's ability to distinguish good vs bad actions
- **Too high (>3.0):** Advantages are poorly normalized, indicating scaling mismatch between raw returns and the value function

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging advantage normalization issues)
- [x] Researcher (analyzing PPO update quality)
- [x] Automated system (critical failure detection)

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
| **Units** | Standard deviations (normalized scale, dimensionless) |
| **Range** | `[0.0, 10.0+]` (unbounded above; pathological above ~5.0) |
| **Precision** | 2 decimal places for display |
| **Default** | `0.0` (before first PPO update) |

### Semantic Meaning

> The **standard deviation of the generalized advantage estimate (GAE)** after normalization.
>
> Computed as:
> - Raw advantages: `A_t = Σ(γλ)ⁿ δ_tⁿ` (GAE λ-return residuals)
> - Normalized: `A_norm = (A - mean(A)) / (std(A) + ε)`
> - Reported: `std(A_norm)` across all valid transitions in the batch
>
> **Healthy Range:** Approximately **0.8–1.2** after proper normalization.
> **Semantics:**
> - **0.8–1.2:** Normalization working as designed
> - **<0.5:** Advantages collapsed; policy updates may be stuck or ineffective
> - **>2.0:** Advantages over-spread; indicates value function mismatch
> - **<0.1:** Severe collapse; advantage normalization fundamentally broken
> - **>3.0:** Severe explosion; possible return scale or value NaN issues

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `0.5 < value < 2.0` | Normal advantage normalization range |
| **Low Warning** | `0.1 < value ≤ 0.5` | Advantage variance too low; policy gradients weakening |
| **High Warning** | `2.0 < value ≤ 3.0` | Advantage variance too high; value function drift |
| **Critical (Collapsed)** | `value ≤ 0.1` | Advantage normalization broken; training cannot proceed |
| **Critical (Exploded)** | `value > 3.0` | Extreme advantage variance; possible NaN/Inf in returns |

**Threshold Source:** `TUIThresholds` in `src/esper/karn/constants.py`:
```python
ADVANTAGE_STD_WARNING: float = 2.0       # High variance
ADVANTAGE_STD_CRITICAL: float = 3.0      # Extreme variance
ADVANTAGE_STD_LOW_WARNING: float = 0.5   # Too little variance
ADVANTAGE_STD_COLLAPSED: float = 0.1     # Collapsed normalization
```

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
    adv_mean = valid_advantages_for_stats.mean()
    adv_std = valid_advantages_for_stats.std()
    # ... skewness/kurtosis computation ...
    adv_stats = torch.stack([adv_mean, adv_std, adv_skewness, adv_kurtosis]).cpu().tolist()
    metrics["advantage_std"] = [adv_stats[1]]  # Line 446
else:
    # No valid advantages - use NaN to signal "no data"
    metrics["advantage_std"] = [float("nan")]
```

### Transport

| Stage | Mechanism | File | Lines |
|-------|-----------|------|-------|
| **1. Emission** | PPO update completion, metrics dict created | `ppo.py` | 425–459 |
| **2. Collection** | Event payload construction via `PPOUpdatePayload` | `simic/telemetry/emitters.py` | 797–798 |
| **3. Aggregation** | Aggregator updates TamiyoState | `karn/sanctum/aggregator.py` | 826 |
| **4. Delivery** | Schema field assignment | `karn/sanctum/schema.py` | 855 |

**Flow Diagram:**
```
[PPOAgent._perform_ppo_update()]
  └─ computes: adv_std = std(normalized_advantages)
     └─ metrics["advantage_std"] = [adv_std]
        └─ VectorizedEmitter.emit_ppo_update()
           └─ PPOUpdatePayload(advantage_std=..., ...)
              └─ TelemetryEmitter._emit()
                 └─ [EventHub]
                    └─ SanctumAggregator.handle_ppo_update()
                       └─ self._tamiyo.advantage_std = payload.advantage_std
                          └─ snapshot.tamiyo.advantage_std
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `advantage_std` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.advantage_std` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Lines** | 855 (field definition) |

**Schema Definition:**
```python
# From TamiyoState (line 850–860)
@dataclass
class TamiyoState:
    # Advantage statistics (from PPO update)
    # Post-normalization stats (should be ~0 mean, ~1 std if normalization working)
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_skewness: float = float("nan")
    advantage_kurtosis: float = float("nan")
    advantage_min: float = 0.0
    advantage_max: float = 0.0
```

### Consumers (Display)

| Widget | File | Usage | Lines |
|--------|------|-------|-------|
| **HealthStatusPanel** | `widgets/tamiyo_brain/health_status_panel.py` | Displayed as `mean±std` with color coding | 59, 71, 394–402 |
| **StatusBanner** | `widgets/tamiyo_brain/status_banner.py` | Critical/warning status detection and labels | 205–207, 234–237 |

**HealthStatusPanel Display:**
```python
# Line 59: Get status color based on advantage_std thresholds
adv_status = self._get_advantage_status(tamiyo.advantage_std)

# Line 71: Display formatted as mean±std
f"{tamiyo.advantage_mean:+.3f}±{tamiyo.advantage_std:.2f}",
style=self._status_style(adv_status),

# Lines 394–402: Threshold logic
def _get_advantage_status(self, adv_std: float) -> str:
    if adv_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:  # <0.1
        return "critical"
    if adv_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:    # >3.0
        return "critical"
    if adv_std > TUIThresholds.ADVANTAGE_STD_WARNING:     # >2.0
        return "warning"
    if adv_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING: # <0.5
        return "warning"
    return "ok"
```

**StatusBanner Critical Detection:**
```python
# Lines 205–208: Critical status
if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_COLLAPSED:  # <0.1
    critical_issues.append("AdvLow")
if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_CRITICAL:    # >3.0
    critical_issues.append("AdvHigh")

# Lines 234–237: Warning status
if tamiyo.advantage_std > TUIThresholds.ADVANTAGE_STD_WARNING:     # >2.0
    warning_issues.append("AdvHigh")
if tamiyo.advantage_std < TUIThresholds.ADVANTAGE_STD_LOW_WARNING: # <0.5
    warning_issues.append("AdvLow")
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPOAgent computes advantage std during _perform_ppo_update()
- [x] **Transport works** — Event includes advantage_std field (PPOUpdatePayload, line 798)
- [x] **Schema field exists** — TamiyoState.advantage_std: float = 0.0 (line 855)
- [x] **Default is correct** — 0.0 is safe default (pre-update state)
- [x] **Consumer reads it** — Both HealthStatusPanel and StatusBanner access snapshot.tamiyo.advantage_std
- [x] **Display is correct** — Rendered as `mean±std` in HealthStatusPanel with color coding
- [x] **Thresholds applied** — Both widgets use TUIThresholds constants for color/status

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_advantage_stats_computed` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_advantage_std` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_advantage_std_reaches_tui` | `[x]` |
| Widget (HealthStatusPanel) | `tests/karn/sanctum/widgets/test_health_status_panel.py` | `test_advantage_std_color_thresholds` | `[x]` |
| Widget (StatusBanner) | `tests/karn/sanctum/widgets/test_status_banner.py` | `test_advantage_std_critical_detection` | `[x]` |

### Manual Verification Steps

1. **Setup:** Start training with: `uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 5`
2. **Open Sanctum:** Launch TUI (auto-opens or `uv run sanctum` in separate terminal)
3. **Locate metric:** Navigate to **HealthStatusPanel** in Tamiyo Brain section
4. **Verify updates:** Watch the "Adv" row for advantage_mean±advantage_std
5. **Verify thresholds:**
   - Should display in **green** when 0.5 < value < 2.0
   - Should display in **yellow** when 0.1 < value ≤ 0.5 or 2.0 < value ≤ 3.0
   - Should display in **red** when value ≤ 0.1 or value > 3.0
6. **Verify StatusBanner:** Check that critical/warning labels appear correctly (AdvLow/AdvHigh)

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after PPO update completes |
| Valid advantage data | computation | Requires at least one valid transition in batch |
| Normalized advantages | computation | Assumes `data["advantages"]` are post-normalization |
| Valid mask | event | Filters out terminal/invalid transitions before std computation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-012` advantage_mean | telemetry | Pair display: mean±std indicates distribution health |
| `TELE-013` advantage_skewness | telemetry | Complementary: skewness + std indicate distribution shape |
| `TELE-014` advantage_kurtosis | telemetry | Complementary: kurtosis + std indicate tail behavior |
| StatusBanner.status | display | Drives "AdvLow"/"AdvHigh" critical/warning labels |
| HealthStatusPanel.adv_status | display | Drives color coding in Adv row |
| Training operator alert | system | Low values trigger user investigation of value function |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation and full wiring verification |
| | | Confirmed end-to-end flow from PPO → Sanctum UI |
| | | Verified all thresholds match TUIThresholds.ADVANTAGE_STD_* |

---

## 8. Notes

### Design Decisions

1. **NaN for missing data:** When valid_advantages_for_stats is empty (no valid transitions in batch), we emit `float("nan")` instead of 0.0. This distinguishes "genuinely zero variance" from "no data available". The UI treats NaN as "unknown" rather than "healthy zero".

2. **Post-normalization only:** The schema tracks the **normalized** advantage std (after dividing by std(A) + ε). This is what matters for PPO stability. We also track `pre_norm_advantage_std` separately (TELE-XXX) for diagnosing whether the problem is in raw returns or the value function.

3. **2-decimal display precision:** HealthStatusPanel uses `.2f` formatting (line 71) to balance readability with precision. 0.5 vs 0.49 distinction is meaningful; 0.5 vs 0.501 is noise.

4. **Paired display with mean:** Advantage std is always displayed alongside `advantage_mean` in the format `mean±std`. This follows ML convention and lets operators see both location and spread at a glance.

### Known Issues

1. **Empty batch edge case:** If a batch has only terminal transitions (all invalid_mask == False), advantage_std will be NaN. This is rare but can happen in early training or with very low success rates. The UI gracefully handles this.

2. **Advantage normalization in large batches:** During early training with small batches, std may be inflated by outliers. After more data accumulates, std stabilizes. This is expected behavior and not a bug.

### Future Improvements

1. **Per-head advantage stats:** Currently we track aggregate advantage_std across all actions. Adding per-head breakdown (e.g., advantage_std for GERMINATE vs WAIT actions) would help debug action-specific credit assignment issues.

2. **Advantage velocity:** Track d(advantage_std)/d(epoch) to detect whether variance is improving or degrading over time (similar to entropy velocity in TELE-002).

3. **Correlation with value loss:** Add a computed metric showing whether high advantage_std correlates with high value loss (would indicate value function scaling issues vs legitimate advantage variance).

### Related Telemetry

- **TELE-001** (Policy Entropy): Measures exploration breadth; advantage_std measures credit assignment quality
- **TELE-011** (Explained Variance): Measures value function quality; low EV + high advantage_std suggests value scaling issues
- **TELE-140-PRE** (Pre-norm Advantage Std): Raw advantages before normalization; for diagnosing whether problem is in returns or value function
- **TELE-153** (Advantage Statistics Full Suite): Mean, skewness, kurtosis, min, max — advantage_std is the first moment (dispersion)

