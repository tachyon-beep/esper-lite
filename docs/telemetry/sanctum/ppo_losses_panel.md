# Telemetry Audit: PPOLossesPanel

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/ppo_losses_panel.py`
**Purpose:** Displays PPO health metrics with visual gauges (Explained Variance, Entropy, Clip Fraction) in the top section and loss sparklines with trends (Policy Loss, Value Loss, Loss Ratio) in the bottom section.

---

## Telemetry Fields Consumed

### Source: TamiyoState (via `snapshot.tamiyo`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `explained_variance` | `float` | `0.0` | Gauge bar showing value function quality (-1.0 to 1.0 range) |
| `entropy` | `float` | `0.0` | Gauge bar showing policy exploration (0.0 to 2.0 range) |
| `clip_fraction` | `float` | `0.0` | Gauge bar showing PPO clipping rate (0.0 to 0.5 range) |
| `policy_loss` | `float` | `0.0` | Current policy loss value displayed with sparkline |
| `value_loss` | `float` | `0.0` | Current value loss displayed with sparkline |
| `policy_loss_history` | `deque[float]` | `maxlen=10` | Sparkline visualization and trend detection |
| `value_loss_history` | `deque[float]` | `maxlen=10` | Sparkline visualization and trend detection |
| `entropy_velocity` | `float` | `0.0` | Rate of entropy change for collapse prediction |
| `collapse_risk_score` | `float` | `0.0` | Collapse risk indicator (triggers warning title) |

### Source: GradientQualityMetrics (via `snapshot.tamiyo.gradient_quality`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `clip_fraction_positive` | `float` | `0.0` | Directional breakdown showing upward clipping rate |
| `clip_fraction_negative` | `float` | `0.0` | Directional breakdown showing downward clipping rate |

### Source: SanctumSnapshot (via `snapshot`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `current_batch` | `int` | `0` | Determines warmup status (batch < 50) |

---

## Thresholds and Color Coding

### Explained Variance Thresholds (from `TUIThresholds`)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Explained Variance | `>= 0.3` | `cyan` (ok) | Value function learning normally |
| Explained Variance | `0.0 <= x < 0.3` | `yellow` (warning) | Value function weak but learning |
| Explained Variance | `< 0.0` | `red bold` (critical) | Value function useless or harmful |

### Entropy Thresholds (from `TUIThresholds`, sourced from leyline)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Entropy | `>= 0.3` | `cyan` (ok) | Healthy exploration |
| Entropy | `0.1 <= x < 0.3` | `yellow` (warning) | Exploration declining |
| Entropy | `< 0.1` | `red bold` (critical) | Policy collapse imminent |

### Clip Fraction Thresholds (from `TUIThresholds`)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Clip Fraction | `<= 0.25` | `cyan` (ok) | PPO clipping within normal range |
| Clip Fraction | `0.25 < x <= 0.3` | `yellow` (warning) | Elevated clipping |
| Clip Fraction | `> 0.3` | `red bold` (critical) | Excessive clipping |

### Loss Ratio Thresholds (hardcoded in widget)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Lv/Lp Ratio | `0.2 <= x <= 5.0` | `bright_cyan` (ok) | Balanced loss magnitudes |
| Lv/Lp Ratio | `0.1 <= x < 0.2` or `5.0 < x <= 10.0` | `yellow` (warning) | Imbalanced losses |
| Lv/Lp Ratio | `< 0.1` or `> 10.0` | `red bold` (critical) | Severely imbalanced losses |

### Collapse Risk Threshold

| Metric | Threshold | Effect |
|--------|-----------|--------|
| `collapse_risk_score` | `> 0.7` | Border title changes to show "!! COLLAPSE ~Nb" warning |

---

## Rendering Logic

### Panel Structure

The panel is divided into two sections separated by a horizontal line:

1. **Top Section (PPO Gauges):**
   - Three gauge rows for Explained Variance, Entropy, and Clip Fraction
   - Each row: `Label Value [gauge bar] Status`
   - Gauge bar width: 8 characters
   - Clip Fraction row includes directional breakdown: `(arrow_up X.X% arrow_down Y.Y%)`

2. **Bottom Section (Loss Sparklines):**
   - Policy Loss with sparkline (width=5) + value + trend arrow
   - Value Loss with sparkline (width=5) + value + trend arrow
   - Loss Ratio (Lv/Lp) with value + status indicator

### Border Title Logic

| Condition | Border Title |
|-----------|--------------|
| `collapse_risk_score > 0.7` and `entropy_velocity < 0` | `"PPO LOSSES !! COLLAPSE ~Nb"` (batches to collapse) |
| `batch < 50` (WARMUP_BATCHES) | `"PPO LOSSES - WARMING UP [X/50]"` |
| Default | `"PPO LOSSES"` |

### Status Indicators

- During warmup (batch < 50): No status indicators shown
- After warmup: `!` (red bold) for critical, `*` (yellow) for warning

### Trend Detection (via `sparkline_utils.detect_trend`)

- Compares first half average to second half average of values
- Threshold: 5% change from first half mean
- Returns: `arrow_up` (improving), `arrow_down` (declining), `arrow_right` (stable)
- For losses: declining is green (good), rising is red (bad)

### Data Flow

```
SanctumSnapshot
    |
    +-- current_batch (warmup detection)
    |
    +-- tamiyo: TamiyoState
          |
          +-- explained_variance, entropy, clip_fraction (gauges)
          +-- policy_loss, value_loss (sparkline values)
          +-- policy_loss_history, value_loss_history (sparklines)
          +-- entropy_velocity, collapse_risk_score (border title)
          |
          +-- gradient_quality: GradientQualityMetrics
                |
                +-- clip_fraction_positive, clip_fraction_negative (directional)
```

---

## Constants Used

### From Widget (`PPOLossesPanel`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `WARMUP_BATCHES` | `50` | Batch count before showing status indicators |
| `GAUGE_WIDTH` | `8` | Width of gauge bar in characters |

### From `TUIThresholds`

| Constant | Value | Source |
|----------|-------|--------|
| `ENTROPY_CRITICAL` | `0.1` | leyline `DEFAULT_ENTROPY_COLLAPSE_THRESHOLD` |
| `ENTROPY_WARNING` | `0.3` | leyline `DEFAULT_ENTROPY_WARNING_THRESHOLD` |
| `CLIP_WARNING` | `0.25` | karn.constants |
| `CLIP_CRITICAL` | `0.3` | karn.constants |
| `EXPLAINED_VAR_WARNING` | `0.3` | karn.constants |
| `EXPLAINED_VAR_CRITICAL` | `0.0` | karn.constants |

---

## Imports

```python
from esper.karn.constants import TUIThresholds
from .sparkline_utils import detect_trend, render_sparkline, trend_style
```
