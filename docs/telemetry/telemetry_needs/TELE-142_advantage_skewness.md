# Telemetry Record: [TELE-142] Advantage Skewness

> **Status:** `[x] Planned` `[x] In Progress` `[x] Wired` `[x] Tested` `[x] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-142` |
| **Name** | Advantage Skewness |
| **Category** | `policy` |
| **Priority** | `P1-important` |

## 2. Purpose

### What question does this answer?

> "Is the advantage distribution symmetric or biased? High skew indicates asymmetric TD errors."

Advantage skewness measures asymmetry in the temporal difference (TD) error distribution. This indicates whether the learning signal is balanced across favorable vs unfavorable trajectories:
- **Symmetric (skewness ≈ 0):** Well-balanced TD errors, healthy learning signal
- **Right-skewed (skewness > 0):** Few large positive advantages (occasional big wins), many small penalties
- **Left-skewed (skewness < 0):** Few large negative advantages (occasional big losses), many small gains
- **Extreme skew (|skewness| > 2.0):** Pathological advantage distribution, suggests reward design issues or catastrophic failures

### Who needs this information?

- [x] Training operator (real-time monitoring)
- [x] Developer (debugging reward function and TD error accumulation)
- [x] Researcher (analyzing advantage distribution shape and asymmetry)
- [ ] Automated system (secondary metric for alerts)

### When is this information needed?

- [x] Real-time (every PPO update/epoch)
- [ ] Periodic (every N episodes)
- [ ] On-demand (when investigating distribution issues)
- [x] Post-hoc (offline analysis of training trajectory)

---

## 3. Data Specification

### Type and Format

| Property | Value |
|----------|-------|
| **Type** | `float` |
| **Units** | Standardized third moment (dimensionless) |
| **Range** | `(-inf, inf)` (unbounded; pathological outside [-2.0, 2.0]) |
| **Precision** | 1 decimal place for display |
| **Default** | `NaN` (before first PPO update or when advantage std too low) |

### Semantic Meaning

> **Skewness** measures asymmetry in the advantage distribution:
>
> Skewness = E[(X-μ)³] / σ³
>
> Where:
> - X = normalized advantages
> - μ = mean advantage
> - σ = standard deviation of advantages
>
> **Interpretation:**
> - **≈ 0:** Symmetric distribution (Gaussian-like)
> - **> 0:** Right-skewed (positive tail, few big wins)
> - **< 0:** Left-skewed (negative tail, few big losses)
> - **> 2.0 or < -1.0:** Pathological asymmetry
>
> **Semantics:**
> - **-0.5 to +1.0:** Normal operating range (slight asymmetry acceptable)
> - **-1.0 to -0.5 or +1.0 to +2.0:** Warning range (increasing skew, monitor)
> - **< -1.0 or > +2.0:** Critical (severe asymmetry, suggests reward/env issues)
>
> **Undefined (NaN):** Emitted when advantage std ≤ 0.05 (too low variance for meaningful moment estimation).

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `-0.5 < value < +1.0` | Approximately symmetric, balanced learning signal |
| **Warning (Left)** | `value < -0.5 or value ≤ -1.0` | Increasingly left-skewed; few catastrophic losses |
| **Warning (Right)** | `value > +1.0 or value ≤ +2.0` | Increasingly right-skewed; few exceptional wins |
| **Critical (Left)** | `value < -1.0` | Severe left skew; pathological negative TD errors |
| **Critical (Right)** | `value > +2.0` | Severe right skew; pathological positive TD errors |
| **Undefined** | `isnan(value)` | No data available (insufficient variance or no updates yet) |

**Threshold Source:** `HealthStatusPanel._get_skewness_status()` in `src/esper/karn/sanctum/widgets/tamiyo_brain/health_status_panel.py`:
```python
if skewness < -1.0 or skewness > 2.0:
    return "critical"
if skewness < -0.5 or skewness > 1.0:
    return "warning"
return "ok"
```

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | PPO batch update, after advantage normalization |
| **File** | `/home/john/esper-lite/src/esper/simic/agent/ppo.py` |
| **Function/Method** | `PPOAgent._perform_ppo_update()` |
| **Line(s)** | 432–443 (skewness computation) |

**Code Snippet:**
```python
# Compute advantage stats for status banner diagnostics
# These indicate if advantage normalization is working correctly
valid_advantages_for_stats = data["advantages"][valid_mask]
if valid_advantages_for_stats.numel() > 0:
    adv_mean = valid_advantages_for_stats.mean()
    adv_std = valid_advantages_for_stats.std()
    # Compute skewness and kurtosis (both use same std threshold)
    # Use 0.05 threshold to match UI "low warning" - below this, σ^n division unstable
    if adv_std > 0.05:
        centered = valid_advantages_for_stats - adv_mean
        # Skewness: E[(X-μ)³] / σ³ - asymmetry indicator
        adv_skewness = (centered ** 3).mean() / (adv_std ** 3)  # Line 437
        # ... (kurtosis computation)
    else:
        # NaN signals "undefined" - std too low for meaningful higher moments
        adv_skewness = torch.tensor(float("nan"), device=adv_mean.device, dtype=adv_mean.dtype)
    adv_stats = torch.stack([adv_mean, adv_std, adv_skewness, adv_kurtosis]).cpu().tolist()
    metrics["advantage_skewness"] = [adv_stats[2]]  # Line 447
else:
    # No valid advantages - use NaN to signal "no data"
    metrics["advantage_skewness"] = [float("nan")]  # Line 457
```

### Transport

| Stage | Mechanism | File | Lines |
|-------|-----------|------|-------|
| **1. Emission** | PPO update completion, metrics dict created | `simic/agent/ppo.py` | 425–459 |
| **2. Collection** | Event payload construction via `PPOUpdatePayload` | `simic/telemetry/emitters.py` | 799 |
| **3. Aggregation** | Aggregator updates TamiyoState | `karn/sanctum/aggregator.py` | 827 |
| **4. Delivery** | Schema field assignment | `karn/sanctum/schema.py` | 857 |

**Flow Diagram:**
```
[PPOAgent._perform_ppo_update()]
  └─ computes: adv_skewness = E[(A-μ)³] / σ³
     └─ metrics["advantage_skewness"] = [adv_skewness]
        └─ VectorizedEmitter.emit_ppo_update()
           └─ PPOUpdatePayload(advantage_skewness=..., ...)
              └─ TelemetryEmitter._emit()
                 └─ [EventHub]
                    └─ SanctumAggregator.handle_ppo_update()
                       └─ self._tamiyo.advantage_skewness = payload.advantage_skewness
                          └─ snapshot.tamiyo.advantage_skewness
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `TamiyoState` |
| **Field** | `advantage_skewness` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.advantage_skewness` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 857 |

**Schema Definition:**
```python
# From TamiyoState (lines 852–865)
# Advantage statistics (from PPO update)
# Post-normalization stats (should be ~0 mean, ~1 std if normalization working)
advantage_mean: float = 0.0
advantage_std: float = 0.0
# NaN = no data (std too low or no valid advantages); 0 = symmetric/normal
advantage_skewness: float = float("nan")  # >0 right-skewed (few big wins), <0 left-skewed (few big losses)
advantage_kurtosis: float = float("nan")  # >0 heavy tails (outliers), <0 light tails; >3 is super-Gaussian
```

### Consumers (Display)

| Widget | File | Usage | Lines |
|--------|------|-------|-------|
| **HealthStatusPanel** | `widgets/tamiyo_brain/health_status_panel.py` | Displayed as `sk:` with value and hint | 60, 76–83, 405–412 |

**HealthStatusPanel Display:**
```python
# Line 60: Get status color based on advantage_skewness thresholds
skew_status = self._get_skewness_status(tamiyo.advantage_skewness)

# Lines 76–83: Display formatted with NaN handling
result.append(" sk:", style="dim")
if math.isnan(tamiyo.advantage_skewness):
    result.append("---", style="dim")
else:
    result.append(
        f"{tamiyo.advantage_skewness:+.1f}",
        style=self._status_style(skew_status),
    )
result.append(self._skewness_hint(tamiyo.advantage_skewness), style="dim")

# Lines 405–412: Threshold logic
def _get_skewness_status(self, skewness: float) -> str:
    if math.isnan(skewness):
        return "ok"  # No data yet - neutral status
    if skewness < -1.0 or skewness > 2.0:
        return "critical"
    if skewness < -0.5 or skewness > 1.0:
        return "warning"
    return "ok"
```

---

## 5. Wiring Verification

### Checklist

- [x] **Emitter exists** — PPOAgent computes skewness during _perform_ppo_update()
- [x] **Transport works** — Event includes advantage_skewness field (PPOUpdatePayload, line 637)
- [x] **Schema field exists** — TamiyoState.advantage_skewness: float = float("nan") (line 857)
- [x] **Default is correct** — NaN is correct (indicates "no data", not "zero skew")
- [x] **Consumer reads it** — HealthStatusPanel accesses snapshot.tamiyo.advantage_skewness
- [x] **Display is correct** — Rendered with `sk:` prefix and +/- formatting
- [x] **Thresholds applied** — HealthStatusPanel uses _get_skewness_status() with correct thresholds

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | `tests/simic/test_ppo.py` | `test_advantage_skewness_computed` | `[x]` |
| Unit (aggregator) | `tests/karn/sanctum/test_aggregator.py` | `test_ppo_update_populates_advantage_skewness` | `[x]` |
| Integration (end-to-end) | `tests/integration/test_telemetry_flow.py` | `test_advantage_skewness_reaches_tui` | `[x]` |
| Widget (HealthStatusPanel) | `tests/karn/sanctum/widgets/test_health_status_panel.py` | `test_advantage_skewness_color_thresholds` | `[x]` |

### Manual Verification Steps

1. **Setup:** Start training with: `uv run python -m esper.scripts.train ppo --preset cifar10 --episodes 5`
2. **Open Sanctum:** Launch TUI (auto-opens or `uv run sanctum` in separate terminal)
3. **Locate metric:** Navigate to **HealthStatusPanel** in Tamiyo Brain section
4. **Verify updates:** Watch the "Advantage" row for skewness display (`sk:` prefix)
5. **Verify formatting:**
   - Should display as "---" when skewness is NaN (insufficient variance)
   - Should display as "+X.X" or "-X.X" when skewness is defined
6. **Verify thresholds:**
   - Should display in **green** when -0.5 < value < +1.0
   - Should display in **yellow** when -1.0 < value ≤ -0.5 or +1.0 < value ≤ +2.0
   - Should display in **red** when value ≤ -1.0 or value > +2.0
7. **Trigger critical state:** If possible, construct a batch with extreme asymmetric advantages to verify red coloring

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| PPO update cycle | event | Only populated after PPO update completes |
| Valid advantage data | computation | Requires at least one valid transition in batch |
| Normalized advantages | computation | Assumes `data["advantages"]` are post-normalization |
| Advantage std | computation | Requires advantage_std > 0.05 for skewness computation |
| Valid mask | event | Filters out terminal/invalid transitions before skewness computation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| `TELE-140` advantage_std | telemetry | Pair display: std + skewness characterize distribution |
| `TELE-141` advantage_kurtosis | telemetry | Complementary: kurtosis + skewness describe distribution shape |
| HealthStatusPanel.adv_status | display | Drives color coding in Advantage row |
| Training operator alert | system | Extreme skew suggests reward design or environment pathology |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2025-01-03 | Telemetry Audit | Initial creation and full wiring verification |
| | | Confirmed end-to-end flow from PPO → Sanctum UI |
| | | Verified threshold logic in HealthStatusPanel._get_skewness_status() |

---

## 8. Notes

### Design Decisions

1. **NaN for insufficient variance:** When advantage std ≤ 0.05, skewness is set to NaN rather than 0.0. This is correct because dividing by σ³ where σ is tiny produces numerical instability. NaN signals "undefined/unreliable" rather than "zero skew". The UI gracefully treats NaN as "no data" and displays "---".

2. **Asymmetric thresholds:** The warning/critical thresholds are asymmetric: warning triggers at -0.5 (left) or +1.0 (right), critical at -1.0 or +2.0. This reflects domain knowledge that right skew (few big wins) is more dangerous than mild left skew (few big losses), because right-skewed distributions often indicate rare catastrophic successes that may not generalize.

3. **Pair with advantage_std:** Skewness is always displayed alongside advantage_std in the HealthStatusPanel. High skewness + low std indicates a distribution concentrated near zero with rare extremes. High skewness + high std indicates a wide, asymmetric distribution. Together they characterize the full shape.

4. **1-decimal display precision:** HealthStatusPanel uses `.1f` formatting to balance readability with precision. Skewness values are typically in range [-2, +3], so 0.1 granularity is appropriate.

### Known Issues

1. **Empty batch edge case:** If a batch has only terminal transitions (valid_advantages_for_stats is empty), skewness is NaN. This is rare but can happen in early training or with very low success rates. The UI gracefully handles this.

2. **Low variance skewness unreliable:** When advantage std is just above 0.05 (the threshold), skewness can exhibit numerical instability from dividing by σ³. For std = 0.06, the divisor is (0.06)³ ≈ 0.00022, making skewness sensitive to rounding errors. This is inherent to the third moment and difficult to fix without more sophisticated estimators.

### Future Improvements

1. **Robust skewness estimators:** Current implementation uses the standardized third moment, which is sensitive to outliers. Consider quartile-based skewness (Bowley skewness) for outlier-robust estimation.

2. **Per-head skewness:** Currently we track aggregate skewness across all actions. Adding per-head breakdown (e.g., skewness for GERMINATE vs WAIT actions) would help debug action-specific credit assignment asymmetries.

3. **Skewness velocity:** Track d(advantage_skewness)/d(epoch) to detect whether asymmetry is improving or worsening over time (similar to entropy velocity in TELE-002).

4. **Skewness-kurtosis joint visualization:** Since skewness and kurtosis are paired (both describe distribution shape), consider a 2D dashboard showing their correlation (e.g., are right-skewed distributions also high-kurtosis?).

### Related Telemetry

- **TELE-140** (Advantage Std): Measures variance; skewness measures asymmetry. Together characterize distribution.
- **TELE-141** (Advantage Kurtosis): Measures tail heaviness; skewness measures asymmetry. Together describe higher moments.
- **TELE-139** (Advantage Mean): Location parameter; skewness is relative to this mean.
- **TELE-143** (Advantage Positive Ratio): Complementary metric — fraction positive vs distribution shape.
- **TELE-001** (Policy Entropy): Both entropy and advantage skewness indicate policy quality, but from different angles.
