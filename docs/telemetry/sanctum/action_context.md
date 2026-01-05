# Telemetry Audit: ActionContext Widget

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo/action_distribution.py`
**Widget:** `ActionContext`
**Purpose:** Consolidated decision context panel combining critic preferences, reward health, returns, action distribution, and sequence.

---

## Overview

The `ActionContext` widget renders a unified view of Tamiyo decision-making context across 5 sections:
1. Critic Preference (Q-values)
2. Reward Signal (health metrics + component breakdown)
3. Returns (episode return history + statistics)
4. Chosen Actions (batch + run distribution bars)
5. Sequence (recent action sequence + pattern warnings)

---

## Data Sources

### Primary: `SanctumSnapshot`
- Accessed via `self._snapshot` (set by `update_snapshot()`)
- Sub-objects: `snapshot.tamiyo` (`TamiyoState`), `snapshot.rewards` (`RewardComponents`)

### Secondary: `RewardHealthData`
- Accessed via `self._reward_health` (set by `update_reward_health()`)
- Computed externally, passed in by `SanctumApp`

---

## Telemetry Fields Consumed

### Section 1: Critic Preference

| Field Path | Type | Default | Usage |
|------------|------|---------|-------|
| `tamiyo.q_germinate` | `float` | `0.0` | Q-value for GERMINATE operation |
| `tamiyo.q_advance` | `float` | `0.0` | Q-value for ADVANCE operation |
| `tamiyo.q_set_alpha` | `float` | `0.0` | Q-value for SET_ALPHA_TARGET operation |
| `tamiyo.q_fossilize` | `float` | `0.0` | Q-value for FOSSILIZE operation |
| `tamiyo.q_prune` | `float` | `0.0` | Q-value for PRUNE operation |
| `tamiyo.q_wait` | `float` | `0.0` | Q-value for WAIT operation |
| `tamiyo.q_variance` | `float` | `0.0` | Variance across Q-values (op conditioning quality) |
| `tamiyo.q_spread` | `float` | `0.0` | max(Q) - min(Q) across operations |

**Rendering Logic:**
- Q-values sorted by value (highest first) with normalized bar visualization
- Bar fill computed via min-max normalization: `(q - q_min) / (q_max - q_min)`
- First/last items marked as "BEST"/"WORST"
- NaN values filtered out; if all NaN, shows "[waiting for data]"

**Color Coding (Q-variance status):**
| Condition | Status | Style |
|-----------|--------|-------|
| `q_variance < 0.01` | `critical` | red bold + X icon |
| `q_variance < 0.1` | `warning` | yellow + `!` marker |
| `q_variance >= 0.1` | `ok` | green + checkmark |
| `isnan(q_variance)` | `ok` | green |

---

### Section 2: Reward Signal

#### From `RewardHealthData` (external)

| Field Path | Type | Default | Usage |
|------------|------|---------|-------|
| `_reward_health.pbrs_fraction` | `float` | `0.0` | PBRS as fraction of total reward |
| `_reward_health.anti_gaming_trigger_rate` | `float` | `0.0` | Fraction of steps with anti-gaming penalties |
| `_reward_health.is_pbrs_healthy` | `property` | computed | `True` if `0.1 <= pbrs_fraction <= 0.4` |
| `_reward_health.is_gaming_healthy` | `property` | computed | `True` if `trigger_rate < 0.05` |
| `_reward_health.hypervolume` | `float` | `0.0` | Pareto hypervolume indicator |

**Thresholds & Color Coding (Line 1):**
| Metric | Healthy Condition | Healthy Style | Unhealthy Style |
|--------|-------------------|---------------|-----------------|
| PBRS fraction | `0.1 <= x <= 0.4` (10-40%) | green + checkmark | red + X |
| Gaming rate | `x < 0.05` (< 5%) | green + checkmark | red + X |
| Hypervolume | (no threshold) | cyan | cyan |

#### From `RewardComponents` (via `snapshot.rewards`)

| Field Path | Type | Default | Usage |
|------------|------|---------|-------|
| `rewards.total` | `float` | `0.0` | Total reward |
| `rewards.bounded_attribution` | `float` | `0.0` | Attribution signal (primary if non-zero) |
| `rewards.base_acc_delta` | `float` | `0.0` | Legacy accuracy delta (fallback if attribution=0) |
| `rewards.compute_rent` | `float` | `0.0` | Cost of active seeds (always <= 0) |
| `rewards.alpha_shock` | `float` | `0.0` | Convex penalty on alpha deltas |
| `rewards.ratio_penalty` | `float` | `0.0` | Penalty for extreme policy ratios |
| `rewards.stage_bonus` | `float` | `0.0` | Bonus for advanced lifecycle stages |
| `rewards.fossilize_terminal_bonus` | `float` | `0.0` | Terminal bonus for fossilization |
| `rewards.hindsight_credit` | `float` | `0.0` | Retroactive credit when beneficiary fossilizes |

**Color Coding (Lines 2-4):**
| Field | Non-Zero Style | Zero Style |
|-------|----------------|------------|
| `total` | green (>= 0), red (< 0) | - |
| `Sig` (bounded_attribution or base_acc_delta) | green (>= 0), red (< 0) | - |
| `compute_rent` | yellow | dim |
| `alpha_shock` | red | dim |
| `ratio_penalty` | red | dim |
| `stage_bonus` | green | dim |
| `fossilize_terminal_bonus` | blue bold | dim |
| `hindsight_credit` | cyan | dim |

---

### Section 3: Returns

| Field Path | Type | Default | Usage |
|------------|------|---------|-------|
| `tamiyo.episode_return_history` | `deque[float]` | empty (maxlen=20) | Episode return history |

**Computed Statistics (from history):**
- **Line 1:** Last 5 returns (most recent first)
- **Line 2:** Percentiles (p10, p50, p90) - requires >= 5 values
- **Line 3:** min, max, mean (mu), std (sigma), trend

**Color Coding:**
| Value | Style |
|-------|-------|
| Return >= 0 | green |
| Return < 0 | red |
| Standard deviation | cyan |

**Spread Warning (p90 - p10):**
| Spread | Icon | Style |
|--------|------|-------|
| `> 50` | double warning | red bold |
| `> 20` | single warning | yellow bold |
| `<= 20` | (none) | - |

**Trend Detection:**
| Condition | Trend | Style |
|-----------|-------|-------|
| `delta > 0.3` (new_mean - old_mean) | rising arrow | green bold |
| `delta < -0.3` | falling arrow | red bold |
| otherwise | horizontal line | dim |

---

### Section 4: Chosen Actions

| Field Path | Type | Default | Usage |
|------------|------|---------|-------|
| `tamiyo.action_counts` | `dict[str, int]` | `{}` | Action counts for current batch |
| `tamiyo.total_actions` | `int` | `0` | Total actions for current batch |
| `tamiyo.cumulative_action_counts` | `dict[str, int]` | `{}` | Action counts for entire run |
| `tamiyo.cumulative_total_actions` | `int` | `0` | Total actions for entire run |

**Action Types Tracked:**
- `GERMINATE`, `SET_ALPHA_TARGET`, `FOSSILIZE`, `PRUNE`, `ADVANCE`, `WAIT`

**Rendering:**
- Two stacked bars: "Batch" (current batch) and "Run" (cumulative)
- Bar width: 28 characters
- Each action segment color-coded (see ACTION_COLORS below)
- Percentages shown below each bar as `G:XX A:XX F:XX P:XX V:XX W:XX`

---

### Section 5: Sequence

| Field Path | Type | Default | Usage |
|------------|------|---------|-------|
| `tamiyo.recent_decisions` | `list[DecisionSnapshot]` | `[]` | Recent decisions (up to 24) |
| `tamiyo.last_action_op` | `str` | `"WAIT"` | Previous operation for context |
| `tamiyo.last_action_success` | `bool` | `True` | Whether previous action executed successfully |

**From `DecisionSnapshot`:**
| Field Path | Type | Default | Usage |
|------------|------|---------|-------|
| `decision.chosen_action` | `str` | - | Action taken (e.g., "GERMINATE") |
| `decision.slot_states` | `dict[str, str]` | `{}` | Slot states at decision time |

**Pattern Detection (`detect_action_patterns()`):**
| Pattern | Condition | Icon | Style |
|---------|-----------|------|-------|
| `STUCK` | 8+ consecutive WAIT with dormant slots available (and no training) | warning | yellow bold reverse |
| `THRASH` | 2+ GERMINATE->PRUNE cycles in last 12 actions | lightning | red bold reverse |
| `ALPHA_OSC` | 4+ SET_ALPHA_TARGET actions in last 12 | arrows | cyan bold reverse |

**Sequence Display:**
- Shows last 12 actions as abbreviated characters (oldest left, newest right)
- Connected with arrows and checkmarks
- When pattern detected, all action chars use pattern color (yellow for STUCK, red for THRASH)

**Last Action Display:**
| Condition | Marker | Style |
|-----------|--------|-------|
| `last_action_success=True` | checkmark | green |
| `last_action_success=False` | X | red bold |

---

## Constants & Configuration

### ACTION_COLORS
```python
ACTION_COLORS = {
    "GERMINATE": "green",
    "SET_ALPHA_TARGET": "cyan",
    "FOSSILIZE": "blue",
    "PRUNE": "red",
    "WAIT": "dim",
    "ADVANCE": "cyan",
}
```

### ACTION_ABBREVS
```python
ACTION_ABBREVS = {
    "GERMINATE": "G",
    "SET_ALPHA_TARGET": "A",
    "FOSSILIZE": "F",
    "PRUNE": "P",
    "WAIT": "W",
    "ADVANCE": "V",  # V for adVance (A taken by SET_ALPHA_TARGET)
}
```

### ACTION_NAMES (for Q-value display)
```python
ACTION_NAMES = {
    "GERMINATE": "GERM",
    "SET_ALPHA_TARGET": "ALPH",
    "FOSSILIZE": "FOSS",
    "PRUNE": "PRUN",
    "WAIT": "WAIT",
    "ADVANCE": "ADVN",
}
```

### Layout Constants
| Constant | Value | Purpose |
|----------|-------|---------|
| `Q_BAR_WIDTH` | 16 | Width of Q-value bar visualization |
| `SEPARATOR_WIDTH` | 38 | Width of horizontal separator lines |

---

## Data Flow

1. **Update Methods:**
   - `update_snapshot(snapshot: SanctumSnapshot)` - Main data update
   - `update_reward_health(data: RewardHealthData)` - Health metrics update

2. **Render Flow:**
   - `render()` calls 5 private methods in order:
     - `_render_critic_preference()` - Section 1
     - `_render_reward_signal()` - Section 2
     - `_render_returns()` - Section 3
     - `_render_action_bars()` - Section 4
     - `_render_action_sequence()` - Section 5

3. **Pattern Detection:**
   - `detect_action_patterns(decisions, slot_states)` - standalone function
   - Called by `_render_action_sequence()` with first 12 decisions

---

## Missing Data Handling

| Section | Condition | Display |
|---------|-----------|---------|
| Critic Preference | `snapshot is None` | "[no data]" dim |
| Critic Preference | All Q-values NaN | "[waiting for data]" dim |
| Reward Signal | `reward_health is None` | "[health pending]" dim |
| Returns | `snapshot is None` | "[no data]" dim |
| Returns | Empty history | Structure preview in dim grey |
| Chosen Actions | `snapshot is None` | "[no data]" dim |
| Chosen Actions | `total_actions == 0` | Empty bar with dim fill |
| Sequence | `snapshot is None` | "[no data]" dim |
| Sequence | No decisions | "[no actions yet]" dim |

---

## Threshold Summary

| Metric | Critical | Warning | Healthy |
|--------|----------|---------|---------|
| Q-variance | < 0.01 | 0.01-0.1 | >= 0.1 |
| PBRS fraction | - | < 10% or > 40% | 10-40% |
| Gaming rate | - | >= 5% | < 5% |
| Return spread (p90-p10) | > 50 | > 20 | <= 20 |
| Return trend delta | - | < -0.3 | > 0.3 |
| STUCK pattern | - | 8+ WAIT with dormant | - |
| THRASH pattern | - | 2+ GERM->PRUNE | - |
| ALPHA_OSC pattern | - | 4+ SET_ALPHA | - |
