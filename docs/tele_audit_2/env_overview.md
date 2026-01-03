# Telemetry Audit: EnvOverview Widget

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/env_overview.py`

**Purpose:** Per-environment overview table displaying a row per environment with metrics, slot states, and status indicators.

---

## 1. Telemetry Fields Consumed

### 1.1 From `SanctumSnapshot`

| Field | Type | Usage |
|-------|------|-------|
| `snapshot.envs` | `dict[int, EnvState]` | Primary data source - iterated to build table rows |
| `snapshot.slot_ids` | `list[str]` | Dynamic column generation for slot cells |
| `snapshot.last_action_env_id` | `int \| None` | Highlights env that received last action with cyan arrow |
| `snapshot.last_action_timestamp` | `datetime \| None` | 5-second hysteresis for action highlight decay |
| `snapshot.aggregate_mean_accuracy` | `float` | Displayed in aggregate row "Sigma" column |
| `snapshot.aggregate_mean_reward` | `float` | Displayed in aggregate row reward column |

### 1.2 From `EnvState` (per environment)

#### Identity & Status

| Field | Type | Default | Column | Usage |
|-------|------|---------|--------|-------|
| `env.env_id` | `int` | - | `Env` | Primary identifier, filter matching |
| `env.status` | `str` | `"initializing"` | `Status` | Status display with icon and color coding |
| `env.reward_mode` | `str \| None` | `None` | `Env` | A/B test cohort pip color |
| `env.rolled_back` | `bool` | `False` | - | Triggers rollback alert row display |
| `env.rollback_reason` | `str` | `""` | - | Shown in rollback alert message |
| `env.last_update` | `datetime \| None` | `None` | `Stale` | Telemetry staleness calculation |

#### Accuracy Metrics

| Field | Type | Default | Column | Usage |
|-------|------|---------|--------|-------|
| `env.host_accuracy` | `float` | `0.0` | `Acc` | Current accuracy with trend arrow |
| `env.best_accuracy` | `float` | `0.0` | `Acc`, `Status` | Color coding (green if at best) |
| `env.accuracy_history` | `deque[float]` | `deque(maxlen=50)` | `Acc^...` | Sparkline generation, trend arrow computation |
| `env.epochs_since_improvement` | `int` | `0` | `Acc`, `Momentum` | Yellow warning if >5 epochs; momentum display |

#### Reward Metrics

| Field | Type | Default | Column | Usage |
|-------|------|---------|--------|-------|
| `env.current_reward` | `float` (property) | `0.0` | `Reward` | Current reward with color coding |
| `env.mean_reward` | `float` (property) | `0.0` | `Reward` | Mean reward in parentheses |
| `env.cumulative_reward` | `float` | `0.0` | `Sum Rwd` | Episode cumulative reward |
| `env.reward_history` | `deque[float]` | `deque(maxlen=50)` | `Rwd^...` | Sparkline generation |

#### Reward Components (from `env.reward_components: RewardComponents`)

| Field | Type | Default | Column | Usage |
|-------|------|---------|--------|-------|
| `reward_components.base_acc_delta` | `float` | `0.0` | `Delta Acc` | Base accuracy delta, colored green/red |
| `reward_components.seed_contribution` | `float` | `0.0` | `Seed Delta` | Seed contribution percentage |
| `reward_components.bounded_attribution` | `float` | `0.0` | `Seed Delta` | Fallback if seed_contribution is 0 |
| `reward_components.compute_rent` | `float` | `0.0` | `Rent` | Compute rent cost (always red) |

#### Host Metrics

| Field | Type | Default | Column | Usage |
|-------|------|---------|--------|-------|
| `env.host_loss` | `float` | `0.0` | `Loss` | Host loss for overfitting detection |
| `env.host_params` | `int` | `0` | - | Used in growth_ratio calculation |
| `env.fossilized_params` | `int` | `0` | - | Used in growth_ratio calculation |
| `env.growth_ratio` | `float` (property) | `1.0` | `Growth` | Model size ratio display |

#### Counterfactual Matrix (from `env.counterfactual_matrix: CounterfactualSnapshot`)

| Field | Type | Default | Column | Usage |
|-------|------|---------|--------|-------|
| `counterfactual_matrix.strategy` | `str` | `"unavailable"` | `CF` | Determines if CF indicator is shown |
| `counterfactual_matrix.slot_ids` | `tuple[str, ...]` | `()` | `CF` | Requires >=2 slots for synergy display |
| `counterfactual_matrix.total_synergy()` | method -> `float` | - | `CF` | Synergy indicator (+/-/.) |

#### Seed State (from `env.seeds: dict[str, SeedState]`)

For each slot in `snapshot.slot_ids`:

| Field | Type | Default | Column | Usage |
|-------|------|---------|--------|-------|
| `seed.stage` | `str` | `"DORMANT"` | `slot_{id}` | Stage abbreviation and color |
| `seed.blueprint_id` | `str \| None` | `None` | `slot_{id}` | Blueprint name (truncated to 6 chars) |
| `seed.alpha` | `float` | `0.0` | `slot_{id}` | Alpha value for BLENDING/HOLDING |
| `seed.epochs_in_stage` | `int` | `0` | `slot_{id}` | Shown as "e{N}" for non-blending stages |
| `seed.has_vanishing` | `bool` | `False` | `slot_{id}` | Yellow down arrow indicator |
| `seed.has_exploding` | `bool` | `False` | `slot_{id}` | Red up arrow indicator |
| `seed.blend_tempo_epochs` | `int` | `5` | `slot_{id}` | Tempo arrows display |
| `seed.alpha_curve` | `str` | `"LINEAR"` | `slot_{id}` | Curve glyph for BLENDING/HOLDING/FOSSILIZED |

#### Action History

| Field | Type | Default | Column | Usage |
|-------|------|---------|--------|-------|
| `env.action_history` | `deque[str]` | `deque(maxlen=10)` | `Last` | Last action taken (abbreviated) |

---

## 2. Column Definitions

| Column Key | Header | Width | Description |
|------------|--------|-------|-------------|
| `env` | `Env` | auto | Env ID with A/B pip and action indicator |
| `acc` | `Acc` | auto | Accuracy with trend arrow and color |
| `cum_rwd` | `Sum Rwd` | auto | Cumulative episode reward |
| `loss` | `Loss` | auto | Host loss for overfitting |
| `cf` | `CF` | auto | Counterfactual synergy indicator |
| `growth` | `Growth` | auto | Growth ratio (model size) |
| `reward` | `Reward` | auto | Current (mean) reward |
| `acc_spark` | `Acc^...` | 8 | Accuracy sparkline |
| `rwd_spark` | `Rwd^...` | 8 | Reward sparkline |
| `delta_acc` | `Delta Acc` | auto | Base accuracy delta component |
| `seed_delta` | `Seed Delta` | auto | Seed contribution component |
| `rent` | `Rent` | auto | Compute rent component |
| `slot_{id}` | `{slot_id}` | auto | Dynamic slot columns |
| `last` | `Last` | auto | Last action abbreviation |
| `momentum` | `Momentum` | auto | Epochs since improvement |
| `status` | `Status` | auto | Status with icon |
| `stale` | `Stale` | auto | Telemetry staleness indicator |

---

## 3. Thresholds and Color Coding Logic

### 3.1 Accuracy (`_format_accuracy`)

| Condition | Color | Trend Arrow |
|-----------|-------|-------------|
| `host_accuracy >= best_accuracy` | green | Computed from history |
| `epochs_since_improvement > 5` | yellow | Computed from history |
| Otherwise | white (default) | Computed from history |

**Trend Arrow Computation** (5-value window):
- `delta > 0.5`: up arrow
- `delta < -0.5`: down arrow
- Otherwise: stable arrow

### 3.2 Cumulative Reward (`_format_cumulative_reward`)

| Condition | Color |
|-----------|-------|
| `cumulative_reward > 0` | green |
| `cumulative_reward < -5` | red |
| Otherwise | white |

### 3.3 Reward (`_format_reward`)

| Condition | Color |
|-----------|-------|
| `current_reward > 0` | green |
| `current_reward < -0.5` | red |
| Otherwise | white |

### 3.4 Host Loss (`_format_host_loss`)

| Condition | Color |
|-----------|-------|
| `loss <= 0` | dim (dash) |
| `loss < 0.1` | green |
| `0.1 <= loss < 0.5` | white |
| `0.5 <= loss < 1.0` | yellow |
| `loss >= 1.0` | red |

### 3.5 Growth Ratio (`_format_growth_ratio`)

Uses leyline constants:
- `DEFAULT_GROWTH_RATIO_GREEN_MAX = 2.0`
- `DEFAULT_GROWTH_RATIO_YELLOW_MAX = 5.0`

| Condition | Color |
|-----------|-------|
| `ratio <= 1.0` | dim |
| `ratio < 2.0` | green |
| `2.0 <= ratio < 5.0` | yellow |
| `ratio >= 5.0` | red |

### 3.6 Counterfactual Synergy (`_format_counterfactual`)

| Condition | Display | Color |
|-----------|---------|-------|
| Unavailable or <2 slots | `.` | dim |
| `synergy > 0.5` | `+` | green |
| `synergy < -0.5` | `-` | bold red |
| Otherwise | `.` | dim |

### 3.7 Momentum Epochs (`_format_momentum_epochs`)

Context-aware coloring based on status:

| Status | Epochs | Color | Prefix |
|--------|--------|-------|--------|
| Any | 0 | green | `+` |
| excellent, healthy | any | green | space |
| initializing | any | dim | space |
| stalled, degraded | <=5 | white | space |
| stalled, degraded | 6-15 | yellow | `!` |
| stalled, degraded | >15 | red | `x` |

### 3.8 Telemetry Staleness (`_format_row_staleness`)

| Condition | Icon | Color | Label |
|-----------|------|-------|-------|
| `last_update is None` | circle | red | BAD |
| `age < 2.0s` | filled circle | green | OK |
| `age <= 5.0s` | half circle | yellow | WARN |
| `age > 5.0s` | empty circle | red | BAD |

### 3.9 Status (`_format_status`)

| Status | Icon | Style |
|--------|------|-------|
| excellent | star | bold green |
| healthy | filled circle | green |
| initializing | half circle | dim |
| stalled | half circle | yellow |
| degraded | empty circle | red |

### 3.10 Slot Cell Stages (`_format_slot_cell`)

Uses `STAGE_COLORS` from leyline:

| Stage | Color |
|-------|-------|
| DORMANT | dim |
| GERMINATED | bright_blue |
| TRAINING | cyan |
| HOLDING | magenta |
| BLENDING | yellow |
| FOSSILIZED | green |
| PRUNED | red |
| EMBARGOED | bright_red |
| RESETTING | dim |

**Gradient Health Indicators:**
- Exploding: red up arrow
- Vanishing: yellow down arrow

**Alpha Curve Glyphs** (from `ALPHA_CURVE_GLYPHS`):
| Curve | Glyph |
|-------|-------|
| LINEAR | diagonal line |
| COSINE | wave |
| SIGMOID_GENTLE | wide arc |
| SIGMOID | narrow arc |
| SIGMOID_SHARP | squared bracket |

**Tempo Arrows:**
| Tempo | Arrows |
|-------|--------|
| <=3 (FAST) | 3 arrows |
| <=5 (STANDARD) | 2 arrows |
| >5 (SLOW) | 1 arrow |

### 3.11 A/B Test Cohort (`_AB_STYLES`)

| Mode | Pip | Color |
|------|-----|-------|
| shaped | filled circle | bright_blue |
| simplified | filled circle | bright_yellow |
| sparse | filled circle | bright_white |

---

## 4. Visual Quieting (Row Dimming)

Rows are dimmed when:
1. Status is "healthy" or "stalled" AND
2. Env is NOT in top 5 by accuracy AND
3. Env does NOT have any FOSSILIZED seeds

---

## 5. Special Row Types

### 5.1 Rollback Alert Row

Triggered when `env.rolled_back = True`. Displays:
- Red background alert message
- Reason mapping:
  - `governor_nan` -> "NaN DETECTED"
  - `governor_lobotomy` -> "LOBOTOMY"
  - `governor_divergence` -> "DIVERGENCE"

### 5.2 Aggregate Row

Shown when multiple envs exist. Computes:
- Mean base_acc_delta across all envs
- Mean compute_rent across all envs
- Mean host_loss across all envs
- Mean growth_ratio across all envs
- Total cumulative_reward across all envs
- Mean epochs_since_improvement across all envs
- Best accuracy from any env
- Mean telemetry age across all envs

### 5.3 Separator Row

Horizontal line (dashes) between data rows and aggregate row.

### 5.4 No Matches Row

Shown when filter matches no environments.

---

## 6. Filtering

Filter matches:
- Numeric input: exact env_id match
- Text input: substring match on env.status

---

## 7. Dependencies

### External Imports
- `textual.widgets.DataTable`
- `textual.widgets.Static`

### Leyline Constants
- `ALPHA_CURVE_GLYPHS`: Curve shape display glyphs
- `DEFAULT_GROWTH_RATIO_GREEN_MAX`: 2.0
- `DEFAULT_GROWTH_RATIO_YELLOW_MAX`: 5.0
- `STAGE_COLORS`: Per-stage color mapping

### Schema Functions
- `make_sparkline()`: Sparkline generation from history values
