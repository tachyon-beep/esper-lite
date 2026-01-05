# Telemetry Audit: HealthStatusPanel

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo/health_status_panel.py`
**Purpose:** Displays comprehensive training health indicators including advantage statistics, ratio bounds, gradient norms, KL divergence, entropy trends, policy state, value function range, and observation health.

---

## Telemetry Fields Consumed

### Source: TamiyoState (path: `snapshot.tamiyo`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `advantage_mean` | `float` | `0.0` | Displayed as primary advantage value with +/- sign and 3 decimal places |
| `advantage_std` | `float` | `0.0` | Combined with mean as "mean +/- std"; used for health status check |
| `advantage_skewness` | `float` | `nan` | Displayed inline with "sk:" prefix; NaN shows as "---" |
| `advantage_kurtosis` | `float` | `nan` | Displayed inline with "kt:" prefix; NaN shows as "---" |
| `advantage_positive_ratio` | `float` | `nan` | Displayed as percentage with "+:" prefix; healthy range 40-60% |
| `joint_ratio_max` | `float` | `1.0` | Product of per-head ratios for multi-head policy |
| `grad_norm` | `float` | `0.0` | Current gradient norm value |
| `grad_norm_history` | `deque[float]` | empty | Used for sparkline visualization (width=10) |
| `kl_divergence` | `float` | `0.0` | KL divergence between old and new policy |
| `kl_divergence_history` | `deque[float]` | empty | Used for sparkline visualization (width=10) |
| `log_prob_min` | `float` | `nan` | Minimum log probability (NaN predictor) |
| `log_prob_max` | `float` | `nan` | Maximum log probability |
| `entropy` | `float` | `0.0` | Current policy entropy |
| `entropy_velocity` | `float` | `0.0` | Rate of entropy change (d(entropy)/d(batch)) |
| `collapse_risk_score` | `float` | `0.0` | Computed entropy collapse risk (0.0-1.0) |
| `entropy_clip_correlation` | `float` | `0.0` | Pearson correlation between entropy and clip fraction histories |
| `clip_fraction` | `float` | `0.0` | PPO clip fraction |
| `value_mean` | `float` | `0.0` | Mean of value function predictions |
| `value_std` | `float` | `0.0` | Standard deviation of value predictions |
| `value_min` | `float` | `0.0` | Minimum value prediction |
| `value_max` | `float` | `0.0` | Maximum value prediction |
| `initial_value_spread` | `float | None` | `None` | Initial value range for relative threshold calculation |

### Source: ObservationStats (path: `snapshot.observation_stats`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `nan_count` | `int` | `0` | Count of NaN values in observations (critical if > 0) |
| `inf_count` | `int` | `0` | Count of Inf values in observations (critical if > 0) |
| `outlier_pct` | `float` | `0.0` | Percentage of observations outside 3 sigma |
| `normalization_drift` | `float` | `0.0` | How much running mean/std has shifted |

---

## Thresholds and Color Coding

### Advantage Statistics

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `advantage_std` | `< 0.1` (COLLAPSED) | `red bold` | Advantage normalization broken |
| `advantage_std` | `> 3.0` (CRITICAL) | `red bold` | Extreme variance |
| `advantage_std` | `> 2.0` (WARNING) | `yellow` | High variance |
| `advantage_std` | `< 0.5` (LOW_WARNING) | `yellow` | Too little variance |
| `advantage_std` | `0.5 - 2.0` | `cyan` | OK - healthy range |
| `advantage_skewness` | `< -1.0` or `> 2.0` | `red bold` | Critical skew |
| `advantage_skewness` | `< -0.5` or `> 1.0` | `yellow` | Warning skew |
| `advantage_skewness` | `-0.5 to 1.0` | `cyan` | OK |
| `advantage_kurtosis` | `< -2.0` or `> 6.0` | `red bold` | Critical kurtosis |
| `advantage_kurtosis` | `< -1.0` or `> 3.0` | `yellow` | Warning kurtosis |
| `advantage_kurtosis` | `-1.0 to 3.0` | `cyan` | OK |
| `advantage_positive_ratio` | `< 0.2` or `> 0.8` | `red bold` | Severely imbalanced |
| `advantage_positive_ratio` | `< 0.4` or `> 0.6` | `yellow` | Moderately imbalanced |
| `advantage_positive_ratio` | `0.4 - 0.6` | `cyan` | OK - healthy balance |

**Note:** Threshold constants come from `TUIThresholds`:
- `ADVANTAGE_STD_COLLAPSED = 0.1`
- `ADVANTAGE_STD_CRITICAL = 3.0`
- `ADVANTAGE_STD_WARNING = 2.0`
- `ADVANTAGE_STD_LOW_WARNING = 0.5`

### Joint Ratio

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `joint_ratio_max` | `> 3.0` or `< 0.33` | `red bold` | Severe explosion/collapse |
| `joint_ratio_max` | `> 2.0` or `< 0.5` | `yellow` | Elevated but not critical |
| `joint_ratio_max` | `0.5 - 2.0` | `cyan` | OK |

### Gradient Norm

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `grad_norm` | `> 10.0` (CRITICAL) | `red bold` | Gradient explosion |
| `grad_norm` | `> 5.0` (WARNING) | `yellow` | Elevated gradients |
| `grad_norm` | `<= 5.0` | `cyan` | OK |

**Note:** Threshold constants from `TUIThresholds`:
- `GRAD_NORM_CRITICAL = 10.0`
- `GRAD_NORM_WARNING = 5.0`

### KL Divergence

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `kl_divergence` | `> 0.05` | `red bold` | Policy changing too fast |
| `kl_divergence` | `> 0.02` | `yellow` | Elevated but not critical |
| `kl_divergence` | `<= 0.02` | `cyan` | OK |

**Note:** These are hardcoded in `_get_kl_status()`, not from `TUIThresholds` (which has `KL_WARNING=0.015`, `KL_CRITICAL=0.03`).

### Log Probability (NaN Predictor)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `log_prob_min` | `< -100` | `red bold` + "NaN RISK" | Numerical underflow imminent |
| `log_prob_min` | `< -50` | `yellow` + "!" | Action nearly impossible |
| `log_prob_min` | `>= -50` | `cyan` | OK |

### Entropy Trend

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `entropy_velocity` | `< -0.03` | `red bold` [vv] | Rapid decline |
| `entropy_velocity` | `< -0.01` | `yellow` [v] | Gradual decline |
| `entropy_velocity` | `> 0.01` | `green` [^] | Increasing entropy |
| `entropy_velocity` | `abs() < 0.005` | `green` | Stable |
| `entropy_velocity` | else | `dim` [~] | Minor fluctuation |
| `collapse_risk_score` | `> 0.7` | `red bold` [ALERT] | High collapse risk |

**Countdown logic:** When `entropy_velocity < 0` and `entropy > ENTROPY_CRITICAL (0.1)`:
- Computes `batches_to_collapse = distance / abs(velocity)`
- Shows countdown if `< 100 batches`

### Policy State

| Condition | Style | Label |
|-----------|-------|-------|
| `corr < -0.5` AND `entropy < 0.3` AND `clip > 0.25` | `red bold` | COLLAPSE RISK |
| `corr < -0.6` AND `entropy < 0.3` | `yellow` | collapsing |
| `corr < -0.4` AND `clip < 0.15` | `green` | narrowing |
| `abs(corr) < 0.3` | `green` | stable |
| else | `yellow` | drifting |

**Note:** Uses `TUIThresholds.ENTROPY_WARNING = 0.3` and `TUIThresholds.CLIP_WARNING = 0.25`.

### Value Range

| Condition | Style | Meaning |
|-----------|-------|---------|
| `v_range < 0.1` AND `v_std < 0.01` | `red bold` | Value collapse - stuck at constant |
| `CoV > 3.0` (when `abs(v_mean) > 0.1`) | `red bold` | Critical instability |
| `CoV > 2.0` | `yellow` | Warning instability |
| `ratio > 10` (vs initial_value_spread) | `red bold` | Critical explosion |
| `ratio > 5` | `yellow` | Warning explosion |
| `v_range > 1000` OR `abs(v_max) > 10000` | `red bold` | Absolute fallback critical |
| `v_range > 500` OR `abs(v_max) > 5000` | `yellow` | Absolute fallback warning |

**Note:** Coefficient of Variation (CoV) = `v_std / abs(v_mean)`.

### Observation Health

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `nan_count > 0` OR `inf_count > 0` | `red bold` | NaN/Inf in observations |
| `outlier_pct` | `> 0.1` (10%) | `red bold` | Critical outlier percentage |
| `outlier_pct` | `> 0.05` (5%) | `yellow` | Warning outlier percentage |
| `outlier_pct` | `<= 0.05` | `cyan` | OK |
| `normalization_drift` | `> 2.0` | `red bold` | Critical drift (>2 sigma) |
| `normalization_drift` | `> 1.0` | `yellow` | Warning drift (>1 sigma) |
| `normalization_drift` | `<= 1.0` | `cyan` | OK |

---

## Rendering Logic

### Panel Layout

The panel renders as a single Static widget with rich Text content. Each metric occupies one line:

```
Advantage    +0.005+/-0.95 sk:+0.2~ kt:+0.5 +:52%
Ratio Joint  1.234
Grad Norm      0.123 ▁▂▃▄▅▆▇█▇▆↗
KL Diverge   0.0123   ▁▂▃▄▅▆▇█▇▆↗
Log Prob     [-12.3,-0.5]
Entropy D    -0.015/b [v] ~45b
Policy       stable
Value Range  [-2.3,4.5] s=1.23
Obs Health   Out:2.3% Drift:0.45
```

### Sparkline Rendering

- Width: 10 characters (compact for inline display)
- Uses `render_sparkline()` from `sparkline_utils.py`
- Trend arrows: `↗` (rising), `↘` (declining), `→` (stable)
- Trend style depends on metric type:
  - For grad_norm and kl_divergence: "loss" type (rising = bad = red)

### Status Style Mapping

```python
{"ok": "cyan", "warning": "yellow", "critical": "red bold"}
```

### Skewness Hint Symbols

| Skewness Range | Symbol | Meaning |
|----------------|--------|---------|
| `abs() < 0.3` | `~` | Approximately symmetric |
| `> 1.0` | `>>` | Strong right skew |
| `> 0.3` | `>` | Mild right skew |
| `< -1.0` | `<<` | Strong left skew |
| `< -0.3` | `<` | Mild left skew |

### Data Flow

1. Widget receives `SanctumSnapshot` via `update_snapshot()`
2. Stores snapshot and calls `refresh()`
3. `render()` builds `Text` object by:
   - Accessing `snapshot.tamiyo` for all TamiyoState fields
   - Accessing `snapshot.observation_stats` for observation health
   - Computing status for each metric via `_get_*_status()` methods
   - Applying appropriate styles based on status
   - Rendering sparklines for metrics with history
4. Returns styled `Text` for display

### Dependencies

- `TUIThresholds` from `esper.karn.constants` (advantage, entropy, clip, gradient thresholds)
- `render_sparkline`, `detect_trend`, `trend_style` from `sparkline_utils.py`
- `math.isnan()` for NaN detection in optional metrics

---

## Notes

1. **NaN Handling:** Fields like `advantage_skewness`, `advantage_kurtosis`, `advantage_positive_ratio`, `log_prob_min`, `log_prob_max`, and `kl_divergence` use `float("nan")` as default, displayed as "---".

2. **Worst Status Aggregation:** For advantage row, the worst status among `adv_status`, `skew_status`, `kurt_status`, and `adv_pos_status` determines whether a "!" indicator is shown.

3. **Entropy Critical Threshold:** Uses `TUIThresholds.ENTROPY_CRITICAL = 0.1` (from `leyline.DEFAULT_ENTROPY_COLLAPSE_THRESHOLD`).

4. **Entropy Warning Threshold:** Uses `TUIThresholds.ENTROPY_WARNING = 0.3` (from `leyline.DEFAULT_ENTROPY_WARNING_THRESHOLD`).
