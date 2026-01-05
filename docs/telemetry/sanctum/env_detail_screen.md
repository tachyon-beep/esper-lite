# Telemetry Audit: EnvDetailScreen

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/env_detail_screen.py`
**Purpose:** Full-screen modal displaying comprehensive diagnostics for a single environment, including per-seed cards with detailed metrics, environment-wide statistics, reward breakdown, and attribution analysis.

---

## Widget Overview

EnvDetailScreen is a `ModalScreen` triggered by 'D' key or Enter on the environment DataTable. It renders:

1. **Header Bar** - Environment summary (ID, status, accuracies, momentum, parameter counts)
2. **Seed Grid** - Per-slot `SeedCard` widgets showing detailed seed state
3. **Metrics Section** - Environment metrics table (accuracy/reward history, action distribution, reward breakdown)
4. **Graveyard Section** - Per-blueprint lifecycle stats (spawns, fossilized, pruned counts)
5. **Attribution Section** - `CounterfactualPanel` and `ShapleyPanel` for causal analysis

---

## Telemetry Fields Consumed

### Source: EnvState (from SanctumSnapshot.envs[env_id])

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `env_id` | `int` | required | Displayed in header bar |
| `status` | `str` | `"initializing"` | Color-coded status in header (excellent/healthy/initializing/stalled/degraded) |
| `best_accuracy` | `float` | `0.0` | Displayed as "Best: X.X%" in header |
| `host_accuracy` | `float` | `0.0` | Displayed as "Current: X.X%" in header |
| `epochs_since_improvement` | `int` | `0` | Momentum indicator in header (red if > threshold) |
| `host_params` | `int` | `0` | Host parameter count in header |
| `fossilized_params` | `int` | `0` | Fossilized params in header and metrics |
| `growth_ratio` | `float` (property) | `1.0` | Model size ratio in header (computed from params) |
| `seeds` | `dict[str, SeedState]` | `{}` | Passed to SeedCard widgets and panels |
| `accuracy_history` | `deque[float]` | maxlen=50 | Sparkline in metrics section |
| `reward_history` | `deque[float]` | maxlen=50 | Sparkline in metrics section |
| `active_seed_count` | `int` | `0` | Displayed in seed counts row |
| `fossilized_count` | `int` | `0` | Displayed in seed counts row |
| `pruned_count` | `int` | `0` | Displayed in seed counts row |
| `action_counts` | `dict[str, int]` | `{}` | Action distribution display |
| `total_actions` | `int` | `0` | Denominator for action % calculation |
| `action_history` | `deque[str]` | maxlen=10 | "Recent Actions" row (last 5 shown) |
| `reward_components` | `RewardComponents` | `RewardComponents()` | Full reward breakdown (see RewardComponents below) |
| `gaming_rate` | `float` (property) | `0.0` | Gaming rate indicator in Signals row |
| `counterfactual_matrix` | `CounterfactualSnapshot` | `CounterfactualSnapshot()` | Passed to CounterfactualPanel |
| `shapley_snapshot` | `ShapleySnapshot` | `ShapleySnapshot()` | Passed to ShapleyPanel |
| `blueprint_spawns` | `dict[str, int]` | `{}` | Graveyard: spawns per blueprint |
| `blueprint_fossilized` | `dict[str, int]` | `{}` | Graveyard: fossilized per blueprint |
| `blueprint_prunes` | `dict[str, int]` | `{}` | Graveyard: pruned per blueprint |

### Source: SeedState (from EnvState.seeds[slot_id])

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `stage` | `str` | `"DORMANT"` | Primary stage indicator with color coding |
| `blueprint_id` | `str \| None` | `None` | Blueprint name displayed in card |
| `seed_params` | `int` | `0` | Formatted with K/M suffix |
| `alpha` | `float` | `0.0` | Alpha value with progress bar (during BLENDING/HOLDING) |
| `alpha_curve` | `str` | `"LINEAR"` | Curve glyph shown during BLENDING/HOLDING/FOSSILIZED |
| `blend_tempo_epochs` | `int` | `5` | Tempo indicator (FAST/STANDARD/SLOW with arrows) |
| `accuracy_delta` | `float` | `0.0` | Accuracy delta with +/- color coding |
| `has_exploding` | `bool` | `False` | Gradient health: "EXPLODING" indicator (red) |
| `has_vanishing` | `bool` | `False` | Gradient health: "VANISHING" indicator (yellow) |
| `grad_ratio` | `float` | `0.0` | Gradient ratio displayed when no issues |
| `epochs_in_stage` | `int` | `0` | Displayed as "Epochs: N" |
| `interaction_sum` | `float` | `0.0` | Synergy indicator (green/red/dim) |
| `boost_received` | `float` | `0.0` | Boost indicator when > threshold |
| `contribution_velocity` | `float` | `0.0` | Trend indicator (improving/declining) |

### Source: RewardComponents (from EnvState.reward_components)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `total` | `float` | `0.0` | Total reward with PBRS fraction |
| `base_acc_delta` | `float` | `0.0` | Accuracy delta signal (Signals row) |
| `compute_rent` | `float` | `0.0` | Compute rent signal (Signals row) |
| `alpha_shock` | `float` | `0.0` | Alpha shock signal (Signals row) |
| `ratio_penalty` | `float` | `0.0` | Ratio penalty signal (Signals row) |
| `bounded_attribution` | `float` | `0.0` | Attribution credit (Credits row) |
| `hindsight_credit` | `float` | `0.0` | Hindsight credit (Credits row) |
| `scaffold_count` | `int` | `0` | Scaffold count shown with hindsight |
| `avg_scaffold_delay` | `float` | `0.0` | Scaffold delay shown with hindsight |
| `stage_bonus` | `float` | `0.0` | Stage bonus credit (Credits row) |
| `fossilize_terminal_bonus` | `float` | `0.0` | Fossilize bonus (Credits row) |
| `blending_warning` | `float` | `0.0` | Blending warning (Warnings row) |
| `holding_warning` | `float` | `0.0` | Holding warning (Warnings row) |

### Source: CounterfactualSnapshot (from EnvState.counterfactual_matrix)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `slot_ids` | `tuple[str, ...]` | `()` | Slot IDs for matrix rendering |
| `configs` | `list[CounterfactualConfig]` | `[]` | Configuration results |
| `strategy` | `str` | `"unavailable"` | Strategy indicator ("full_factorial", "ablation_only", "unavailable") |
| `baseline_accuracy` | `float` (property) | `0.0` | Host-only accuracy |
| `combined_accuracy` | `float` (property) | `0.0` | All-seeds-enabled accuracy |
| `individual_contributions()` | `dict[str, float]` | method | Per-seed solo contributions |
| `pair_contributions()` | `dict[tuple[str, str], float]` | method | Pair contributions |
| `total_synergy()` | `float` | method | Synergy score (combined - expected) |

### Source: ShapleySnapshot (from EnvState.shapley_snapshot)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `slot_ids` | `tuple[str, ...]` | `()` | Slot IDs for attribution |
| `values` | `dict[str, ShapleyEstimate]` | `{}` | Per-slot Shapley values |
| `epoch` | `int` | `0` | Epoch when computed |
| `ranked_slots()` | `list[tuple[str, float]]` | method | Slots ranked by contribution |
| `get_significance(slot_id)` | `bool` | method | Statistical significance check |

### Source: ShapleyEstimate (from ShapleySnapshot.values[slot_id])

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `mean` | `float` | `0.0` | Mean contribution value |
| `std` | `float` | `0.0` | Standard deviation (uncertainty) |
| `n_samples` | `int` | `0` | Number of permutation samples |

---

## Thresholds and Color Coding

### DisplayThresholds (from esper.karn.constants)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| PBRS Fraction | `PBRS_HEALTHY_MIN` (0.1) to `PBRS_HEALTHY_MAX` (0.4) | green checkmark | PBRS is 10-40% of total reward (healthy shaping) |
| PBRS Fraction | outside 0.1-0.4 | yellow warning | PBRS dominating or absent |
| Gaming Rate | < `GAMING_RATE_HEALTHY_MAX` (0.05) | green | < 5% anti-gaming triggers |
| Gaming Rate | >= 0.05 | yellow/red | Excessive anti-gaming penalties |
| Growth Ratio | > `GROWTH_RATIO_WARNING` (1.2) | yellow | >20% parameter growth from seeds |
| Growth Ratio | <= 1.2 | green/dim | Normal parameter overhead |
| Momentum | > `MOMENTUM_STALL_THRESHOLD` (10 epochs) | red | Stalled improvement |
| Momentum | <= 10 epochs | yellow | Moderate time without improvement |
| Momentum | 0 epochs | green | Just improved ("Improving") |
| Interaction Synergy | abs > `INTERACTION_SYNERGY_THRESHOLD` (0.5) | green/red | Significant synergy/interference |
| Boost Received | > `BOOST_RECEIVED_THRESHOLD` (0.1) | cyan | Notable boost from partner |
| Contribution Velocity | abs > `CONTRIBUTION_VELOCITY_EPSILON` (0.01) | green/yellow | Improving/declining trend |
| Blueprint Success Rate | >= `BLUEPRINT_SUCCESS_GREEN` (0.50) | green | >= 50% fossilization rate |
| Blueprint Success Rate | >= `BLUEPRINT_SUCCESS_YELLOW` (0.25) | yellow | 25-50% fossilization rate |
| Blueprint Success Rate | < 0.25 | red | < 25% fossilization rate |

### STAGE_COLORS (from esper.leyline)

| Stage | Color | Rich Markup |
|-------|-------|-------------|
| DORMANT | dim gray | `dim` |
| GERMINATED | bright blue | `bright_blue` |
| TRAINING | cyan | `cyan` |
| HOLDING | magenta | `magenta` |
| BLENDING | yellow | `yellow` |
| FOSSILIZED | green | `green` |
| PRUNED | red | `red` |
| EMBARGOED | bright red | `bright_red` |
| RESETTING | dim gray | `dim` |

### ALPHA_CURVE_GLYPHS (from esper.leyline)

| Curve | Glyph | Visual Meaning |
|-------|-------|----------------|
| LINEAR | `╱` | Straight diagonal |
| COSINE | `∿` | Wave (ease-in/out) |
| SIGMOID_GENTLE | `⌒` | Wide top arc (slow start/end) |
| SIGMOID | `⌢` | Narrow bottom arc (moderate S-curve) |
| SIGMOID_SHARP | `⊐` | Squared bracket (near-step) |

### Status Styles (Header)

| Status | Style | Meaning |
|--------|-------|---------|
| excellent | bold green | Accuracy > 80%, just improved |
| healthy | green | Recently improved, normal operation |
| initializing | dim | Early training, no improvement yet |
| stalled | yellow | >10 epochs without improvement (with hysteresis) |
| degraded | red | Accuracy dropped >1% (with hysteresis) |

### Accuracy Delta Styles (SeedCard)

| Condition | Style | Meaning |
|-----------|-------|---------|
| delta > 0 | green | Seed improving accuracy |
| delta < 0 | red | Seed hurting accuracy |
| delta == 0 or stage in (TRAINING, GERMINATED) | dim italic | No contribution yet |

### Gradient Health Styles (SeedCard)

| Condition | Style | Icon | Meaning |
|-----------|-------|------|---------|
| has_exploding | bold red | `EXPLODING` | Gradient explosion detected |
| has_vanishing | bold yellow | `VANISHING` | Gradient vanishing detected |
| grad_ratio > 0 | green | ratio value | Normal gradient health |
| else | green | `OK` | Default healthy state |

### Blend Tempo Display (SeedCard)

| Tempo Epochs | Name | Arrows | Meaning |
|--------------|------|--------|---------|
| <= 3 | FAST | `▸▸▸` | Fast integration |
| <= 5 | STANDARD | `▸▸` | Standard integration |
| > 5 | SLOW | `▸` | Slow integration |

---

## Rendering Logic

### Header Bar (`_render_header`)

Horizontal summary line containing:
1. **Environment ID**: "Environment N" (bold)
2. **Status**: Color-coded status label (see Status Styles)
3. **Best Accuracy**: "Best: X.X%" (cyan)
4. **Current Accuracy**: "Current: X.X%" (white)
5. **Momentum**: Either "Improving" (green) or "N epochs" with color based on threshold
6. **Parameters**: "Host: NNK +Seed: NNK = X.XXx" showing growth ratio

### Seed Cards (`SeedCard.render`)

Each slot gets a card that renders:
- **Dormant slots**: Gray panel with "DORMANT" label
- **Active slots**: Full card with:
  - Stage (color-coded)
  - Blueprint name
  - Parameter count (formatted K/M)
  - Alpha with progress bar (during BLENDING/HOLDING)
  - Tempo with arrows and curve glyph
  - Accuracy delta (stage-aware - shows "0.0 (learning)" for TRAINING/GERMINATED)
  - Gradient health indicator
  - Epochs in stage
  - Synergy indicator (interaction_sum with boost)
  - Trend indicator (contribution_velocity)

### Metrics Table (`_render_metrics`)

Fixed-structure table with rows (all always visible, dim "--" for empty):
1. **Accuracy History**: Sparkline (40 chars)
2. **Reward History**: Sparkline (40 chars)
3. **Seed Counts**: Active/Fossilized/Pruned counts
4. **Fossilized Params**: Formatted parameter count
5. **Action Distribution**: Per-action percentages with colors
6. **Reward Total**: Total with PBRS fraction indicator
7. **Signals**: Base acc delta, compute rent, alpha shock, ratio penalty + gaming rate
8. **Credits**: Bounded attribution, hindsight credit, stage bonus, fossilize bonus
9. **Warnings**: Blending/holding warnings
10. **Recent Actions**: Last 5 actions joined with " -> "

### Graveyard (`_render_graveyard`)

Per-blueprint lifecycle table:
- Header: Blueprint / spawn / foss / prun / rate
- One row per blueprint seen in spawns, fossilized, or prunes
- Success rate color-coded (green >= 50%, yellow >= 25%, red < 25%)
- "(none)" placeholder when no seeds have spawned

### Attribution Section

Two side-by-side panels:
1. **CounterfactualPanel**: Waterfall visualization of baseline -> individuals -> pairs -> combined with synergy calculation
2. **ShapleyPanel**: Per-slot Shapley values with uncertainty bounds and significance indicators

---

## Data Flow

```
SanctumSnapshot
    └── envs[env_id]: EnvState
            ├── (header fields) → _render_header()
            ├── seeds: dict[slot_id, SeedState] → SeedCard widgets
            │       └── SeedState fields → SeedCard._render_active()
            ├── (metrics fields) → _render_metrics()
            │       └── reward_components: RewardComponents → reward breakdown
            ├── (graveyard fields) → _render_graveyard()
            ├── counterfactual_matrix → CounterfactualPanel
            └── shapley_snapshot → ShapleyPanel
```

### Update Flow

When `update_env_state()` is called:
1. Updates `self._env` reference
2. Re-renders header via `Static.update()`
3. Re-renders metrics via `Static.update()`
4. Updates CounterfactualPanel via `update_matrix()`
5. Updates ShapleyPanel via `update_snapshot()`
6. Updates each SeedCard via `update_seed()`
7. Re-renders graveyard via `Static.update()`

All `NoMatches` exceptions are caught silently (widget may not be mounted yet during initial composition).

---

## Imported Constants and Functions

| Import | Source | Purpose |
|--------|--------|---------|
| `DisplayThresholds` | `esper.karn.constants` | All display thresholds |
| `format_params` | `esper.karn.sanctum.formatting` | Parameter formatting (K/M suffix) |
| `ALPHA_CURVE_GLYPHS` | `esper.leyline` | Curve shape glyphs |
| `STAGE_COLORS` | `esper.leyline` | Stage-to-color mapping |
| `make_sparkline` | `esper.karn.sanctum.schema` | Sparkline generation |
| `CounterfactualPanel` | local widgets | Counterfactual visualization |
| `ShapleyPanel` | local widgets | Shapley value visualization |
