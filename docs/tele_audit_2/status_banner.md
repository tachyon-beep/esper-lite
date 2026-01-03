# Telemetry Audit: StatusBanner

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/status_banner.py`
**Purpose:** One-line status summary displaying overall training health with key metrics, warmup spinner, NaN/Inf indicators, and A/B group labeling.

---

## Telemetry Fields Consumed

### Source: SanctumSnapshot (root)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `current_batch` | `int` | `0` | Batch progress display (`Batch:47/100`) and warmup detection (< 50 batches) |
| `max_batches` | `int` | `0` | Batch progress denominator |

### Source: TamiyoState (snapshot.tamiyo)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `ppo_data_received` | `bool` | `False` | Gate for metrics display; shows "Waiting for PPO data..." until True |
| `group_id` | `str \| None` | `None` | A/B testing group label (A/B/C) with color coding |
| `nan_grad_count` | `int` | `0` | NaN gradient indicator (leftmost for F-pattern visibility) |
| `inf_grad_count` | `int` | `0` | Inf gradient indicator |
| `kl_divergence` | `float` | `0.0` | KL metric display with trend arrow and threshold warning |
| `kl_divergence_history` | `deque[float]` | `deque(maxlen=10)` | Trend arrow computation via `detect_trend()` |
| `entropy` | `float` | `0.0` | Critical/warning status detection |
| `explained_variance` | `float` | `0.0` | Critical/warning status detection |
| `advantage_std` | `float` | `0.0` | Critical/warning status detection (both high and low) |
| `clip_fraction` | `float` | `0.0` | Critical/warning status detection |
| `grad_norm` | `float` | `0.0` | Critical/warning status detection |

### Source: InfrastructureMetrics (snapshot.tamiyo.infrastructure)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `compile_enabled` | `bool` | `False` | Border title suffix `[[compiled]]` when True |
| `memory_usage_percent` | `float` (property) | `0.0` | Memory percentage display `[Mem:45%]` with color coding |

---

## Thresholds and Color Coding

### NaN/Inf Gradient Indicators

| Condition | Style | Visual |
|-----------|-------|--------|
| `nan_grad_count > 5` or `inf_grad_count > 5` | `red bold reverse` | Maximum visibility (reverse video) |
| `nan_grad_count > 0` or `inf_grad_count > 0` | `red bold` | Standard alert |

### KL Divergence Status

| Threshold | Style | Meaning |
|-----------|-------|---------|
| `kl > TUIThresholds.KL_CRITICAL` (0.03) | `red bold` | Critical - excessive policy change |
| `kl > TUIThresholds.KL_WARNING` (0.015) | `yellow` | Warning - mild policy drift |
| `kl <= KL_WARNING` | `dim` | OK - healthy policy updates |

### Memory Usage

| Threshold | Style | Meaning |
|-----------|-------|---------|
| `mem_pct > 90` | `red bold` | Critical memory pressure |
| `mem_pct > 75` | `yellow` | Warning - high usage |
| `mem_pct <= 75` | `dim` | Healthy |

### Overall Status (DRL Decision Tree)

Priority order for status determination:

1. **NaN/Inf Detection** (HIGHEST PRIORITY)

| Condition | Status | Label | Style |
|-----------|--------|-------|-------|
| `nan_grad_count > 0` | `critical` | `NaN DETECTED` | `red bold` |
| `inf_grad_count > 0` | `critical` | `Inf DETECTED` | `red bold` |

2. **Warmup Period**

| Condition | Status | Label | Style |
|-----------|--------|-------|-------|
| `current_batch < 50` | `warmup` | `WARMING UP [5/50]` | `cyan` |

3. **Critical Issues** (any triggers critical status)

| Metric | Threshold | Label |
|--------|-----------|-------|
| `entropy` | `< ENTROPY_CRITICAL` (0.1) | `Entropy` |
| `explained_variance` | `<= EXPLAINED_VAR_CRITICAL` (0.0) | `Value` |
| `advantage_std` | `< ADVANTAGE_STD_COLLAPSED` (0.1) | `AdvLow` |
| `advantage_std` | `> ADVANTAGE_STD_CRITICAL` (3.0) | `AdvHigh` |
| `kl_divergence` | `> KL_CRITICAL` (0.03) | `KL` |
| `clip_fraction` | `> CLIP_CRITICAL` (0.3) | `Clip` |
| `grad_norm` | `> GRAD_NORM_CRITICAL` (10.0) | `Grad` |

Display: `FAIL:Primary (+N)` where N is additional issues count.

4. **Warning Issues** (any triggers warning status)

| Metric | Threshold | Label |
|--------|-----------|-------|
| `explained_variance` | `< EXPLAINED_VAR_WARNING` (0.3) | `Value` |
| `entropy` | `< ENTROPY_WARNING` (0.3) | `Entropy` |
| `kl_divergence` | `> KL_WARNING` (0.015) | `KL` |
| `clip_fraction` | `> CLIP_WARNING` (0.25) | `Clip` |
| `advantage_std` | `> ADVANTAGE_STD_WARNING` (2.0) | `AdvHigh` |
| `advantage_std` | `< ADVANTAGE_STD_LOW_WARNING` (0.5) | `AdvLow` |
| `grad_norm` | `> GRAD_NORM_WARNING` (5.0) | `Grad` |

Display: `WARN:Primary (+N)` where N is additional issues count.

5. **Healthy**

| Status | Label | Style |
|--------|-------|-------|
| `ok` | `LEARNING` | `green` |

---

## CSS Classes Applied

| Status | CSS Class | Trigger |
|--------|-----------|---------|
| `status-ok` | Healthy training | No warnings/criticals |
| `status-warning` | Warning detected | Any warning threshold exceeded |
| `status-critical` | Critical issue | Any critical threshold exceeded |
| `status-warmup` | Warmup period | `current_batch < 50` |
| `group-a/b/c` | A/B testing | `group_id` is set |

---

## Trend Arrow Computation

The `_trend_arrow()` method uses `detect_trend()` from schema.py:

| Trend | Arrow | Style | Meaning |
|-------|-------|-------|---------|
| `improving` | `^` | `green` | Metric improving (KL decreasing) |
| `stable` | `->` | `dim` | No significant change |
| `volatile` | `~` | `yellow` | High variance in recent values |
| `warning` | `v` | `red` | Metric worsening (KL increasing) |

Requires minimum 5 history values; uses 10-sample windows for RL-appropriate noise tolerance.

---

## Rendering Logic

### Banner Structure (left to right)

1. **NaN/Inf Indicators** - Leftmost for F-pattern visibility (always first when present)
   - `NaN:3` or `Inf:2` with red styling

2. **A/B Group Label** (if in A/B mode)
   - Color-coded emoji: `A`, `B`, `C`
   - Separator: ` | `

3. **Status Icon**
   - Warmup: Animated braille spinner (`...`)
   - OK: `[check]`
   - Warning: `[!]`
   - Critical: `[X]`

4. **Status Label**
   - `LEARNING`, `WARMING UP [5/50]`, `FAIL:Entropy (+2)`, etc.

5. **Metrics** (only after `ppo_data_received`)
   - `KL:0.008` with trend arrow and threshold warning `!`
   - `Batch:47/100`
   - `[Mem:45%]` (if `memory_usage_percent > 0`)

### Border Title

- Default: `TAMIYO`
- With torch.compile: `TAMIYO [[compiled]]`

### Data Flow

```
SanctumSnapshot
    |
    +-- tamiyo.ppo_data_received --> Gate for metrics display
    +-- tamiyo.nan_grad_count/inf_grad_count --> NaN/Inf indicators
    +-- tamiyo.group_id --> A/B group styling
    +-- tamiyo.kl_divergence --> KL metric display
    +-- tamiyo.kl_divergence_history --> Trend arrow
    +-- tamiyo.entropy/explained_variance/etc --> Status detection
    +-- tamiyo.infrastructure.compile_enabled --> Border title
    +-- tamiyo.infrastructure.memory_usage_percent --> Memory display
    +-- current_batch/max_batches --> Batch progress
```
