# Telemetry Record: [TELE-228] Return Skewness

> **Status:** `[x] Planned` `[ ] In Progress` `[ ] Wired` `[ ] Tested` `[ ] Verified`

---

## 1. Identity

| Field | Value |
|-------|-------|
| **ID** | `TELE-228` |
| **Name** | Return Skewness |
| **Category** | `value` |
| **Priority** | `P2-medium` |

## 2. Purpose

### What question does this answer?

> "Is the return distribution asymmetric? Are there rare big wins or rare catastrophic failures that mean return would mask?"

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
| **Type** | `float` |
| **Units** | dimensionless (standardized third moment) |
| **Range** | unbounded (typically -3 to +3) |
| **Precision** | 1 decimal place for display |
| **Default** | `0.0` (before return data available) |

### Semantic Meaning

> Return Skewness measures the asymmetry of the episode return distribution:
>
> **Skewness = E[((G - mu) / sigma)^3]**
>
> Where G = episode return, mu = mean, sigma = std.
>
> - **Skew near 0:** Symmetric distribution (normal-like)
> - **Skew > 0:** Right-skewed (few big wins, many moderate results)
> - **Skew < 0:** Left-skewed (few catastrophic failures, many moderate results)
> - **|Skew| > 2:** Severely asymmetric distribution
>
> Skewness reveals whether mean return is representative or if outliers dominate.

### Health Thresholds

| Level | Condition | Meaning |
|-------|-----------|---------|
| **Healthy** | `abs(value) < 1.0` | Displayed in cyan; roughly symmetric distribution |
| **Warning** | `1.0 <= abs(value) < 2.0` | Displayed in yellow; moderately asymmetric |
| **Critical** | `abs(value) >= 2.0` | Displayed in red bold; severely skewed |

**Threshold Source:** `src/esper/karn/sanctum/widgets/tamiyo_brain/value_diagnostics_panel.py` - `_get_skewness_style()` method (lines 189-202)

---

## 4. Data Flow

### Source (Emitter)

| Property | Value |
|----------|-------|
| **Origin** | NOT YET IMPLEMENTED |
| **File** | N/A |
| **Function/Method** | N/A |
| **Line(s)** | N/A |

```python
# PLANNED IMPLEMENTATION:
# Compute from episode return buffer (last N completed episodes)
#
# from scipy.stats import skew
# episode_returns = np.array([...])  # Collected from episode completions
# return_skewness = skew(episode_returns)
#
# Or using manual calculation:
# mean = episode_returns.mean()
# std = episode_returns.std()
# if std > 1e-8:
#     return_skewness = ((episode_returns - mean) ** 3).mean() / (std ** 3)
# else:
#     return_skewness = 0.0
```

### Transport

| Stage | Mechanism | File |
|-------|-----------|------|
| **1. Emission** | NOT IMPLEMENTED | N/A |
| **2. Collection** | NOT IMPLEMENTED | N/A |
| **3. Aggregation** | Field exists in schema, awaiting emitter | `karn/sanctum/schema.py` (line 755) |
| **4. Delivery** | Schema ready at `snapshot.tamiyo.value_function.return_skewness` | `karn/sanctum/schema.py` |

```
[Episode completion - PLANNED]
  --return_skewness-->
  [emit_episode_stats_event() - NOT IMPLEMENTED]
  --ValueFunctionMetrics-->
  [SanctumAggregator - HANDLER NOT IMPLEMENTED]
  -->
  [SanctumSnapshot.tamiyo.value_function.return_skewness]
```

### Schema Location

| Property | Value |
|----------|-------|
| **Dataclass** | `ValueFunctionMetrics` |
| **Field** | `return_skewness` |
| **Path from SanctumSnapshot** | `snapshot.tamiyo.value_function.return_skewness` |
| **Schema File** | `/home/john/esper-lite/src/esper/karn/sanctum/schema.py` |
| **Schema Line** | 755 |
| **Default Value** | `0.0` |

### Consumers (Display)

| Widget | File | Usage |
|--------|------|-------|
| ValueDiagnosticsPanel | `widgets/tamiyo_brain/value_diagnostics_panel.py` (lines 114-116) | Displayed as "Skew" with sign prefix and color-coded style (cyan/yellow/red based on absolute value) |

---

## 5. Wiring Verification

### Checklist

- [ ] **Emitter exists** — NOT IMPLEMENTED: No code computes return_skewness
- [ ] **Transport works** — NOT IMPLEMENTED: No event payload includes this field
- [x] **Schema field exists** — `ValueFunctionMetrics.return_skewness: float = 0.0` at line 755
- [x] **Default is correct** — `0.0` is appropriate before return data available
- [x] **Consumer reads it** — ValueDiagnosticsPanel directly accesses `snapshot.tamiyo.value_function.return_skewness`
- [x] **Display is correct** — Panel renders with sign prefix and `_get_skewness_style()` for coloring
- [x] **Thresholds applied** — Consumer uses 1.0/2.0 absolute value thresholds for color coding

### Test Coverage

| Test Type | File | Test Name | Status |
|-----------|------|-----------|--------|
| Unit (emitter) | N/A | NOT IMPLEMENTED | `[ ]` |
| Unit (aggregator) | N/A | NOT IMPLEMENTED | `[ ]` |
| Integration (end-to-end) | N/A | NOT IMPLEMENTED | `[ ]` |
| Widget (ValueDiagnosticsPanel) | `tests/karn/sanctum/widgets/tamiyo_brain/test_value_diagnostics_panel.py` | Rendering test | `[ ]` |

### Manual Verification Steps

1. **BLOCKED:** Emitter not implemented - cannot verify end-to-end
2. Once implemented: Start training and observe ValueDiagnosticsPanel
3. Skew should hover near 0 for a well-behaved policy
4. Verify color transitions: cyan (|skew| < 1.0) -> yellow (1.0-2.0) -> red (>= 2.0)
5. Positive skew + low P50 = policy sometimes gets lucky but usually fails
6. Negative skew + high P50 = policy usually succeeds but sometimes fails catastrophically

---

## 6. Dependencies

### Upstream (this telemetry depends on)

| Dependency | Type | Notes |
|------------|------|-------|
| Episode completions | event | Requires completed episode returns |
| Return buffer | data | Buffer of recent episode returns (e.g., last 100) |
| Mean and std | computation | Required for standardized skewness calculation |

### Downstream (depends on this telemetry)

| Dependent | Type | Notes |
|-----------|------|-------|
| ValueDiagnosticsPanel display | display | Shows distribution asymmetry |
| Outlier detection | analysis | High skewness indicates outlier-dominated returns |
| Mean reliability | analysis | High skewness = mean is not representative |

---

## 7. History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-03 | Telemetry Audit | Initial creation - documenting planned but unwired metric |

---

## 8. Notes

> **Interpretation Guide:**
> - **Positive skewness (> 0):** Distribution has a right tail. A few episodes achieve very high returns, but most are moderate or low. This could indicate:
>   - Exploration success: Occasionally finds high-reward trajectories
>   - Luck dependence: Performance relies on favorable random events
>
> - **Negative skewness (< 0):** Distribution has a left tail. Most episodes are successful, but a few fail catastrophically. This could indicate:
>   - Fragile policy: Works well most of the time but has failure modes
>   - Edge cases: Certain states trigger poor behavior
>
> **Relationship to Mean:** When skewness is high:
> - Mean is pulled toward the tail
> - Median (P50) better represents typical performance
> - This is why P50 is shown alongside mean in diagnostics
>
> **Implementation Note:** Can use scipy.stats.skew() or compute manually using standardized third moment. Manual computation avoids scipy dependency.
>
> **Wiring Status:** Schema and consumer exist but emitter is completely missing. This is a PLANNED metric awaiting implementation.
