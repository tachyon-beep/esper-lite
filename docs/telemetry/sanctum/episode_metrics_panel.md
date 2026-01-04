# Telemetry Audit: EpisodeMetricsPanel

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/episode_metrics_panel.py`

**Purpose:** Displays episode-level training metrics with two distinct rendering modes:
- **Warmup Mode:** Before PPO updates begin, shows collection progress and random policy baseline
- **Training Mode:** After PPO data arrives, shows episode health metrics

---

## 1. Telemetry Fields Consumed

### 1.1 Mode Detection Field

| Field Path | Type | Default | Description |
|------------|------|---------|-------------|
| `snapshot.tamiyo.ppo_data_received` | `bool` | `False` | Determines rendering mode (warmup vs training) |

### 1.2 Warmup Mode Fields

| Field Path | Type | Default | Description |
|------------|------|---------|-------------|
| `snapshot.tamiyo.episode_return_history` | `deque[float]` | Empty deque (maxlen=20) | Episode returns collected during warmup |

### 1.3 Training Mode Fields (via `snapshot.episode_stats`)

| Field Path | Type | Default | Description |
|------------|------|---------|-------------|
| `snapshot.episode_stats.total_episodes` | `int` | `0` | Total episodes completed |
| `snapshot.episode_stats.length_mean` | `float` | `0.0` | Mean episode length (steps) |
| `snapshot.episode_stats.length_std` | `float` | `0.0` | Standard deviation of episode length |
| `snapshot.episode_stats.length_min` | `int` | `0` | Minimum episode length observed |
| `snapshot.episode_stats.length_max` | `int` | `0` | Maximum episode length observed |
| `snapshot.episode_stats.timeout_rate` | `float` | `0.0` | Fraction of episodes hitting max steps (0-1) |
| `snapshot.episode_stats.success_rate` | `float` | `0.0` | Fraction of episodes achieving goal (0-1) |
| `snapshot.episode_stats.early_termination_rate` | `float` | `0.0` | Fraction of episodes terminated early (0-1) |
| `snapshot.episode_stats.steps_per_germinate` | `float` | `0.0` | Average steps between GERMINATE actions |
| `snapshot.episode_stats.steps_per_prune` | `float` | `0.0` | Average steps between PRUNE actions |
| `snapshot.episode_stats.steps_per_fossilize` | `float` | `0.0` | Average steps between FOSSILIZE actions |
| `snapshot.episode_stats.completion_trend` | `str` | `"stable"` | Trend indicator: "improving", "stable", "declining" |

---

## 2. Field Usage in Rendering

### 2.1 Border Title Logic

```python
if not snapshot.tamiyo.ppo_data_received:
    self.border_title = "WARMUP"
else:
    if stats.total_episodes > 0:
        self.border_title = f"EPISODE - {stats.total_episodes} ep"
    else:
        self.border_title = "EPISODE HEALTH"
```

### 2.2 Warmup Mode Rendering (`_render_warmup`)

**Layout (5 lines):**
```
Status       Collecting warmup data...
Baseline     mu:+12.3  sigma:8.2
Episodes     n:45 collected

(waiting for PPO updates...)
```

| Line | Field(s) Used | Rendering Logic |
|------|---------------|-----------------|
| Status | None | Static text "Collecting warmup data..." (cyan) |
| Baseline | `episode_return_history` | If history exists: compute mean/std, display with color coding. Otherwise: "--- (waiting)" |
| Episodes | `episode_return_history` | Display `len(history)` as "n:X collected" or "0" if empty |
| Waiting | None | Static text "(waiting for PPO updates...)" (dim) |

**Baseline Mean Color Logic:**
- Green: `mean >= 0`
- Red: `mean < 0`

### 2.3 Training Mode Rendering (`_render_training`)

**Layout (4 lines):**
```
Length       mu:123  sigma:45   [12-500]
Outcomes     X10%  check75%  circled-x15%
Steps/Act    germ:45  prune:120  foss:200
Trend        improving arrow-up
```

| Line | Field(s) Used | Rendering Logic |
|------|---------------|-----------------|
| Length | `length_mean`, `length_std`, `length_min`, `length_max` | Mean/std always shown; range shown if min or max > 0 |
| Outcomes | `timeout_rate`, `success_rate`, `early_termination_rate` | Each rate formatted as percentage with symbol prefix |
| Steps/Act | `steps_per_germinate`, `steps_per_prune`, `steps_per_fossilize` | All three metrics displayed with abbreviated labels |
| Trend | `completion_trend` | Mapped to text + arrow via `trend_map` |

---

## 3. Thresholds and Color Coding

### 3.1 Rate Style Logic (`_rate_style` method)

The method takes two parameters:
- `rate`: Value between 0 and 1
- `bad_high`: If `True`, high values are bad; if `False`, high values are good

**When `bad_high=True` (timeout_rate, early_termination_rate):**

| Condition | Style | Interpretation |
|-----------|-------|----------------|
| `rate > 0.2` | `red` | Critical - more than 20% |
| `rate > 0.1` | `yellow` | Warning - more than 10% |
| Otherwise | `green` | Healthy - 10% or less |

**When `bad_high=False` (success_rate):**

| Condition | Style | Interpretation |
|-----------|-------|----------------|
| `rate > 0.7` | `green` | Excellent - more than 70% |
| `rate > 0.5` | `yellow` | Moderate - more than 50% |
| `rate < 0.3` | `red` | Critical - less than 30% |
| Otherwise | `dim` | Below average (30-50%) |

### 3.2 Outcome Symbols

| Metric | Symbol | Symbol Style |
|--------|--------|--------------|
| Timeout | X (cross mark) | `red dim` |
| Success | check mark | `green dim` |
| Early Termination | circled-x | `yellow dim` |

### 3.3 Trend Display Mapping

| `completion_trend` Value | Display Text | Style |
|-------------------------|--------------|-------|
| `"improving"` | `"improving arrow-up"` | `green` |
| `"stable"` | `"stable arrow-right"` | `dim` |
| `"declining"` | `"declining arrow-down"` | `red` |
| (default/unknown) | `"stable arrow-right"` | `dim` |

---

## 4. Schema References

### 4.1 EpisodeStats Dataclass

```python
@dataclass
class EpisodeStats:
    # Episode length statistics
    length_mean: float = 0.0
    length_std: float = 0.0
    length_min: int = 0
    length_max: int = 0

    # Outcome tracking (over recent N episodes)
    total_episodes: int = 0
    timeout_count: int = 0
    success_count: int = 0
    early_termination_count: int = 0

    # Derived rates
    timeout_rate: float = 0.0
    success_rate: float = 0.0
    early_termination_rate: float = 0.0

    # Steps per action type
    steps_per_germinate: float = 0.0
    steps_per_prune: float = 0.0
    steps_per_fossilize: float = 0.0

    # Completion trend
    completion_trend: str = "stable"
```

### 4.2 TamiyoState Relevant Fields

```python
@dataclass
class TamiyoState:
    # Episode return tracking
    episode_return_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=20)
    )

    # PPO data received flag
    ppo_data_received: bool = False
```

---

## 5. Telemetry Data Flow

1. **Warmup Phase:**
   - `episode_return_history` is populated by episode completion events
   - `ppo_data_received` remains `False`
   - Panel shows baseline statistics from random policy

2. **Training Phase:**
   - First PPO update sets `ppo_data_received = True`
   - `episode_stats` is populated by episode completion telemetry
   - Panel switches to training metrics display

---

## 6. Column Layout Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `COL1` | `13` | Label column width for alignment |

---

## 7. Notes

- The warmup mode baseline statistics establish the floor that PPO needs to beat (per DRL expert review)
- All numeric fields use `.0f` formatting (no decimal places except baseline mean which uses `.1f`)
- The panel is styled with CSS class `"panel"` and has a dynamic border title
- Rate styles use tri-state color coding with different thresholds based on whether high is good or bad
