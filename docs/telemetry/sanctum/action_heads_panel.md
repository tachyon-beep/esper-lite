# Telemetry Audit: ActionHeadsPanel

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo/action_heads_panel.py`
**Purpose:** Unified visualization of action head health (entropy, gradient norms, PPO ratios) with per-decision output heatmaps and gradient flow diagnostics.

---

## Telemetry Fields Consumed

### Source: TamiyoState (via `SanctumSnapshot.tamiyo`)

#### Per-Head Entropy Fields

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `head_op_entropy` | `float` | `0.0` | Op head entropy - normalized against `HEAD_MAX_ENTROPIES["op"]` |
| `head_slot_entropy` | `float` | `0.0` | Slot head entropy - normalized against `HEAD_MAX_ENTROPIES["slot"]` |
| `head_blueprint_entropy` | `float` | `0.0` | Blueprint head entropy - normalized against `HEAD_MAX_ENTROPIES["blueprint"]` |
| `head_style_entropy` | `float` | `0.0` | Style head entropy - normalized against `HEAD_MAX_ENTROPIES["style"]` |
| `head_tempo_entropy` | `float` | `0.0` | Tempo head entropy - normalized against `HEAD_MAX_ENTROPIES["tempo"]` |
| `head_alpha_target_entropy` | `float` | `0.0` | Alpha target head entropy - normalized against `HEAD_MAX_ENTROPIES["alpha_target"]` |
| `head_alpha_speed_entropy` | `float` | `0.0` | Alpha speed head entropy - normalized against `HEAD_MAX_ENTROPIES["alpha_speed"]` |
| `head_alpha_curve_entropy` | `float` | `0.0` | Alpha curve head entropy - normalized against `HEAD_MAX_ENTROPIES["alpha_curve"]` |

#### Per-Head Gradient Norm Fields

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `head_op_grad_norm` | `float` | `0.0` | Current gradient norm for op head |
| `head_slot_grad_norm` | `float` | `0.0` | Current gradient norm for slot head |
| `head_blueprint_grad_norm` | `float` | `0.0` | Current gradient norm for blueprint head |
| `head_style_grad_norm` | `float` | `0.0` | Current gradient norm for style head |
| `head_tempo_grad_norm` | `float` | `0.0` | Current gradient norm for tempo head |
| `head_alpha_target_grad_norm` | `float` | `0.0` | Current gradient norm for alpha target head |
| `head_alpha_speed_grad_norm` | `float` | `0.0` | Current gradient norm for alpha speed head |
| `head_alpha_curve_grad_norm` | `float` | `0.0` | Current gradient norm for alpha curve head |

#### Per-Head Previous Gradient Norm Fields (for trend detection)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `head_op_grad_norm_prev` | `float` | `0.0` | Previous gradient norm for trend arrow computation |
| `head_slot_grad_norm_prev` | `float` | `0.0` | Previous gradient norm for trend arrow computation |
| `head_blueprint_grad_norm_prev` | `float` | `0.0` | Previous gradient norm for trend arrow computation |
| `head_style_grad_norm_prev` | `float` | `0.0` | Previous gradient norm for trend arrow computation |
| `head_tempo_grad_norm_prev` | `float` | `0.0` | Previous gradient norm for trend arrow computation |
| `head_alpha_target_grad_norm_prev` | `float` | `0.0` | Previous gradient norm for trend arrow computation |
| `head_alpha_speed_grad_norm_prev` | `float` | `0.0` | Previous gradient norm for trend arrow computation |
| `head_alpha_curve_grad_norm_prev` | `float` | `0.0` | Previous gradient norm for trend arrow computation |

#### Per-Head PPO Ratio Fields

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `head_op_ratio_max` | `float` | `1.0` | Max pi_new/pi_old ratio for op head |
| `head_slot_ratio_max` | `float` | `1.0` | Max ratio for slot head |
| `head_blueprint_ratio_max` | `float` | `1.0` | Max ratio for blueprint head |
| `head_style_ratio_max` | `float` | `1.0` | Max ratio for style head |
| `head_tempo_ratio_max` | `float` | `1.0` | Max ratio for tempo head |
| `head_alpha_target_ratio_max` | `float` | `1.0` | Max ratio for alpha target head |
| `head_alpha_speed_ratio_max` | `float` | `1.0` | Max ratio for alpha speed head |
| `head_alpha_curve_ratio_max` | `float` | `1.0` | Max ratio for alpha curve head |

#### Per-Head NaN/Inf Latch Fields

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `head_nan_latch` | `dict[str, bool]` | `{head: False for head in HEAD_NAMES}` | Latched NaN indicators per head (keys: op, slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve) |
| `head_inf_latch` | `dict[str, bool]` | `{head: False for head in HEAD_NAMES}` | Latched Inf indicators per head |

#### Gradient Flow Footer Fields

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `gradient_quality.gradient_cv` | `float` | `0.0` | Gradient coefficient of variation (nested in `GradientQualityMetrics`) |
| `dead_layers` | `int` | `0` | Count of layers with vanishing gradients |
| `exploding_layers` | `int` | `0` | Count of layers with exploding gradients |
| `nan_grad_count` | `int` | `0` | Total NaN gradient count |
| `inf_grad_count` | `int` | `0` | Total Inf gradient count |

### Source: DecisionSnapshot (via `TamiyoState.recent_decisions`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `decision_id` | `str` | `""` | Unique ID for carousel tracking |
| `timestamp` | `datetime` | - | For age calculation and sorting |
| `chosen_action` | `str` | - | Action name (GERMINATE, WAIT, etc.) |
| `chosen_slot` | `str \| None` | `None` | Target slot ID |
| `chosen_blueprint` | `str \| None` | `None` | Blueprint choice |
| `chosen_style` | `str \| None` | `None` | Blending style choice |
| `chosen_tempo` | `str \| None` | `None` | Tempo choice |
| `chosen_alpha_target` | `str \| None` | `None` | Alpha target choice |
| `chosen_alpha_speed` | `str \| None` | `None` | Alpha speed choice |
| `chosen_curve` | `str \| None` | `None` | Alpha curve choice |
| `confidence` | `float` | - | Overall action confidence (fallback) |
| `op_confidence` | `float` | `0.0` | Op head confidence |
| `slot_confidence` | `float` | `0.0` | Slot head confidence |
| `blueprint_confidence` | `float` | `0.0` | Blueprint head confidence |
| `style_confidence` | `float` | `0.0` | Style head confidence |
| `tempo_confidence` | `float` | `0.0` | Tempo head confidence |
| `alpha_target_confidence` | `float` | `0.0` | Alpha target head confidence |
| `alpha_speed_confidence` | `float` | `0.0` | Alpha speed head confidence |
| `curve_confidence` | `float` | `0.0` | Curve head confidence |

### Source: SanctumSnapshot

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `slot_ids` | `list[str]` | `[]` | Used to determine `n_slots` for slot head state classification |

### Source: Leyline Constants

| Constant | Value | Usage |
|----------|-------|-------|
| `HEAD_MAX_ENTROPIES` | `dict[str, float]` | Max entropy per head (`ln(N)` where N = action space size) for normalization |
| `DEFAULT_HOST_LSTM_LAYERS` | `12` | Total layer count for dead/exploding layer display |

---

## Thresholds and Color Coding

### Entropy Thresholds

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Normalized entropy | `> 0.5` | `green` | Healthy exploration |
| Normalized entropy | `> 0.25` | `yellow` | Moderate exploration |
| Normalized entropy | `<= 0.25` | `red` | Policy collapse risk |

### Gradient Norm Thresholds

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Gradient norm | `< 0.01` | `red` | Vanishing gradients (dead) |
| Gradient norm | `< 0.1` | `yellow` | Low gradient signal |
| Gradient norm | `0.1 - 2.0` | `green` | Healthy gradient range |
| Gradient norm | `2.0 - 5.0` | `yellow` | Elevated gradients |
| Gradient norm | `> 5.0` | `red` | Exploding gradients |

### Gradient Trend Thresholds

| Condition | Arrow | Style | Meaning |
|-----------|-------|-------|---------|
| `abs(delta) < 0.01` | `->` | `dim` | Stable |
| `delta > 0` | `up-right arrow` | Context-dependent | Increasing |
| `delta < 0` | `down-right arrow` | Context-dependent | Decreasing |

Trend style is context-sensitive:
- If `grad < 0.1`: increasing is `green`, decreasing is `red` (want gradients to recover)
- If `grad > 2.0`: increasing is `red`, decreasing is `green` (want gradients to reduce)
- Otherwise: `dim` (normal fluctuation)

### PPO Ratio Thresholds

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Ratio | `0.8 - 1.2` | `green` | Within PPO clip range |
| Ratio | `0.5 - 1.5` | `yellow` | Moderate policy shift |
| Ratio | `< 0.5` or `> 1.5` | `red` | Large policy shift |

### Head State Classification

| Condition | Symbol | Style | Meaning |
|-----------|--------|-------|---------|
| `grad < 0.01 AND norm_ent < 0.1` | `circle outline` | `red` | Dead head |
| `grad > 5.0` | `triangle up` | `red bold` | Exploding gradients |
| `norm_ent < 0.1 AND slot head AND n_slots == 1` | `diamond` | `dim` | Expected deterministic |
| `norm_ent < 0.1` (other heads) | `diamond` | `yellow` | Concerning collapse |
| `norm_ent > 0.9 AND grad normal` | `half circle` | `yellow` | Confused (very high entropy) |
| `0.3 <= norm_ent <= 0.7 AND grad normal` | `filled circle` | `green` | Healthy |
| Otherwise | `filled circle` | `dim` | Indeterminate |

### Gradient Flow Footer Thresholds

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Gradient CV | `< 0.5` | `green` + "stable" | Low variance in gradients |
| Gradient CV | `0.5 - 2.0` | `yellow` + "warn" | Moderate gradient variance |
| Gradient CV | `>= 2.0` | `red` + "BAD" | High gradient variance |
| Dead/Exploding layers | `0` | `green` | All layers healthy |
| Dead/Exploding layers | `> 0` | `red` | Layer health issues |
| NaN count | `0` | `dim` | No NaN gradients |
| NaN count | `> 0` | `red bold` | NaN gradients detected |
| Inf count | `0` | `dim` | No Inf gradients |
| Inf count | `> 0` | `red bold` | Inf gradients detected |

### NaN/Inf Latch Indicators

| State | Symbol | Style | Meaning |
|-------|--------|-------|---------|
| Latched (True) | `filled circle` | `red bold` | NaN/Inf detected in head (permanent) |
| Clear (False) | `circle outline` | `dim` | No NaN/Inf detected |

### Decision Age Pip Thresholds

| Age Ratio | Style | Meaning |
|-----------|-------|---------|
| `< 40%` of MAX_DISPLAY_AGE_S (30s) | `green` | Fresh decision |
| `40% - 70%` | `yellow` | Aging |
| `70% - 90%` | `#8B4513` (brown) | Old |
| `>= 90%` | `red` | About to expire |

### Confidence Heat Bars

| Confidence | Bar | Meaning |
|------------|-----|---------|
| `<= 0.0` | `5 empty blocks` | No confidence |
| `< 0.25` | `1 full + 4 empty` | Very low |
| `< 0.5` | `2 full + 3 empty` | Low |
| `< 0.75` | `3 full + 2 empty` | Moderate |
| `< 1.0` | `4 full + 1 empty` | High |
| `>= 1.0` | `5 full blocks` | Maximum |

---

## Rendering Logic

### Overall Structure

The panel renders three main sections separated by horizontal lines:

1. **Head Health Grid** (Rows 0-9)
2. **Decision Table** (Header + 6 decision rows)
3. **Gradient Flow Footer** (Single summary line)

### Head Health Grid Layout

The grid displays 8 action heads in columns with consistent widths:
- **Op**: 7 chars
- **Slot**: 7 chars
- **Blueprint**: 10 chars
- **Style**: 9 chars
- **Tempo**: 9 chars
- **alphaTarget**: 9 chars
- **alphaSpeed**: 9 chars
- **Curve**: 9 chars

Each column has a specific gutter width for visual alignment.

**Row Layout:**
- Row 0: Header labels (right-aligned, dim bold)
- Row 1: Entropy values with differential entropy coefficients (1.3x for sparse heads shown in cyan dim)
- Row 2: Entropy mini-bars (5-char normalized bar)
- Row 3: Gradient values with trend arrows (shifted right so arrow overlaps bar position)
- Row 4: Gradient mini-bars (log-scale)
- Row 5: Ratio values (pi_new/pi_old)
- Row 6: Ratio mini-bars (deviation from 1.0)
- Row 7: State indicators (synthesized from entropy + gradient health)
- Row 8: NaN latch indicators (per-head)
- Row 9: Inf latch indicators (per-head)

### Entropy Coefficient Display

Sparse heads (blueprint, style, tempo, alpha_*) display a coefficient prefix:
- `1.3x` for blueprint, tempo (most sparse)
- `1.2x` for style, alpha_target, alpha_speed, alpha_curve
- No prefix for op, slot (coefficient = 1.0)

### Decision Carousel ("Firehose Model")

The widget maintains up to 6 decisions in a carousel with the following behavior:

1. **Age starts at 0 when added to display** (not from decision's original timestamp)
2. **Always grab the NEWEST decision** when swapping (no backlog queue)
3. **Swap interval**: 5 seconds minimum between swaps
4. **Max display age**: 30 seconds before aging out

The `_display_timestamps` dict tracks when each decision was added for age calculation.

### Decision Table Columns

Each decision row displays:
- **Dec**: Row number with age pip (colored by freshness)
- **Op**: Action abbreviation + confidence heat bar
- **Slot**: Slot ID (first 4 chars) + confidence heat bar
- **Blueprint**: Abbreviated blueprint name + confidence heat bar
- **Style**: Abbreviated style + confidence heat bar
- **Tempo**: Abbreviated tempo + confidence heat bar
- **alphaTarget**: Abbreviated alpha target + confidence heat bar
- **alphaSpeed**: Abbreviated alpha speed + confidence heat bar
- **Curve**: Abbreviated curve + confidence heat bar

Unused fields (e.g., blueprint for WAIT actions) display "-" in dim style.

### Action Abbreviations

| Full Name | Abbreviation | Color |
|-----------|--------------|-------|
| GERMINATE | GERM | green |
| SET_ALPHA_TARGET | ALPH | cyan |
| FOSSILIZE | FOSS | blue |
| PRUNE | PRUN | red |
| WAIT | WAIT | dim |
| ADVANCE | ADVN | cyan |

---

## Data Flow

```
SanctumSnapshot
    |
    +-- tamiyo: TamiyoState
    |       |
    |       +-- head_*_entropy (8 fields)
    |       +-- head_*_grad_norm (8 fields)
    |       +-- head_*_grad_norm_prev (8 fields)
    |       +-- head_*_ratio_max (8 fields)
    |       +-- head_nan_latch (dict)
    |       +-- head_inf_latch (dict)
    |       +-- gradient_quality.gradient_cv
    |       +-- dead_layers, exploding_layers
    |       +-- nan_grad_count, inf_grad_count
    |       +-- recent_decisions (list[DecisionSnapshot])
    |
    +-- slot_ids (for n_slots determination)

Leyline Constants:
    +-- HEAD_MAX_ENTROPIES (for entropy normalization)
    +-- DEFAULT_HOST_LSTM_LAYERS (for layer counts)
```

---

## HEAD_CONFIG Mapping

The widget uses `HEAD_CONFIG` to define column order and properties:

```python
HEAD_CONFIG = [
    (label, entropy_field, grad_norm_field, width, entropy_coef),
    ...
]
```

| Label | Entropy Field | Grad Norm Field | Width | Coef |
|-------|---------------|-----------------|-------|------|
| Op | head_op_entropy | head_op_grad_norm | 7 | 1.0 |
| Slot | head_slot_entropy | head_slot_grad_norm | 7 | 1.0 |
| Blueprint | head_blueprint_entropy | head_blueprint_grad_norm | 10 | 1.3 |
| Style | head_style_entropy | head_style_grad_norm | 9 | 1.2 |
| Tempo | head_tempo_entropy | head_tempo_grad_norm | 9 | 1.3 |
| alphaTarget | head_alpha_target_entropy | head_alpha_target_grad_norm | 9 | 1.2 |
| alphaSpeed | head_alpha_speed_entropy | head_alpha_speed_grad_norm | 9 | 1.2 |
| Curve | head_alpha_curve_entropy | head_alpha_curve_grad_norm | 9 | 1.2 |

The `DISPLAY_TO_LEYLINE_KEY` mapping converts display labels to leyline HEAD_NAMES keys for NaN/Inf latch lookup:

| Display | Leyline Key |
|---------|-------------|
| Op | op |
| Slot | slot |
| Blueprint | blueprint |
| Style | style |
| Tempo | tempo |
| alphaTarget | alpha_target |
| alphaSpeed | alpha_speed |
| Curve | alpha_curve |

---

## Visual Layout Reference

```
+-- ACTION HEADS -------------------------------------------------------------+
|               Op        Slot     Blueprint        Style       Tempo  ...    |
| Entr       0.893       1.000        1.000        0.215       1.000  ...    |
|            ###-        ####         ####         #---        ####   ...    |
| Grad      0.131->     0.135/       0.211->       0.275\      0.192->...    |
|            #---        #---         #---         #---        #---   ...    |
| Ratio      1.023       0.987        1.142        1.312       0.891  ...    |
|            ----        ----         ----         #---        ----   ...    |
| State         *           *            *            o           *   ...    |
| NaN           o           o            o            o           o   ...    |
| Inf           o           o            o            o           o   ...    |
|-----------------------------------------------------------------------------|
|  Dec          Op        Slot     Blueprint       Style       Tempo  ...    |
|   #1   GERM#####   r0c0#####   conv_lt####   LIN_ADD###   STD##--   ...    |
|   #2   WAIT#####           -             -            -          -  ...    |
|   ...                                                                       |
|-----------------------------------------------------------------------------|
| Flow: CV:0.123 stable   Dead:0/12   Exploding:0/12   NaN:0  Inf:0          |
+-----------------------------------------------------------------------------+
```

Legend:
- `#` = filled bar segment
- `-` = empty bar segment
- `*` = healthy state
- `o` = dead/clear state
- `/` = trending up arrow
- `\` = trending down arrow
- `->` = stable arrow
