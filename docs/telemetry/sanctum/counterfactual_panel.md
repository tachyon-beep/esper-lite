# Telemetry Audit: CounterfactualPanel

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/counterfactual_panel.py`
**Purpose:** Waterfall visualization of factorial counterfactual analysis showing baseline, individual, pair, and combined seed contributions with synergy/interference detection. Also displays per-seed interaction metrics when available.

---

## Telemetry Fields Consumed

### Source: CounterfactualSnapshot (passed directly to widget)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `strategy` | `str` | `"unavailable"` | Determines render mode: `"unavailable"` shows placeholder, `"full_factorial"` or `"ablation_only"` shows waterfall |
| `configs` | `list[CounterfactualConfig]` | `[]` | Used to check availability (empty = unavailable); contains all 2^n configuration results |
| `slot_ids` | `tuple[str, ...]` | `()` | List of slot IDs for labeling individual/pair contributions; length determines rendering strategy (<=3 inline pairs, >3 top 5) |

### Source: CounterfactualSnapshot (computed properties/methods)

| Property/Method | Type | Usage |
|-----------------|------|-------|
| `baseline_accuracy` | `float` | Accuracy with all seeds disabled; starting point of waterfall |
| `combined_accuracy` | `float` | Accuracy with all seeds enabled; end point of waterfall |
| `individual_contributions()` | `dict[str, float]` | Each seed's solo contribution over baseline; used for individual section |
| `pair_contributions()` | `dict[tuple[str, str], float]` | Each pair's contribution over baseline; used for pairs section |
| `total_synergy()` | `float` | Combined - baseline - sum(individuals); used for synergy/interference display |

### Source: SeedState (from seeds dict passed to widget)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `stage` | `str` | `"DORMANT"` | Filters active seeds (excludes DORMANT, PRUNED, EMBARGOED, RESETTING) for interaction metrics |
| `interaction_sum` | `float` | `0.0` | Summed across active seeds for "Network synergy" metric |
| `boost_received` | `float` | `0.0` | Used to find "Best partner" (seed with highest boost_received > 0.1) |
| `contribution_velocity` | `float` | `0.0` | Used for "Trends" display (> 0.01 = trending up, < -0.01 = trending down) |

---

## Thresholds and Color Coding

### Synergy/Interference Detection

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `total_synergy()` | `< -0.5` (with n_seeds >= 2) | `bold red reverse` | INTERFERENCE DETECTED - seeds are hurting each other |
| `total_synergy()` | `> 0.5` | `bold green` | Synergy - seeds are working together positively |
| `total_synergy()` | `-0.5` to `0.5` | `dim` | Neutral - seeds are independent |

### Pair Synergy Highlighting

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Pair synergy (contrib - ind1 - ind2) | `> 0.5` | `green` | Pair has positive synergy |

### Network Synergy (Seed Dynamics Section)

**Note:** Color thresholds and icon thresholds are independent:

| Metric | Color Threshold | Style | Icon Threshold | Icon |
|--------|-----------------|-------|----------------|------|
| `interaction_sum` total | `> 0.5` | `green` | `> 0` | `▲` |
| `interaction_sum` total | `< -0.5` | `red` | `< 0` | `▼` |
| `interaction_sum` total | `-0.5` to `0.5` | `dim` | `= 0` | `─` |

This means a value of `0.3` would display with `dim` style but an `▲` icon (positive direction, neutral magnitude).

### Best Partner Detection

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `boost_received` | `> 0.1` | `cyan` | Displays as "Best partner: {slot_id} (+{boost}% boost)" |
| `boost_received` | `<= 0.1` | `dim` | Shows "--" placeholder |

### Contribution Velocity (Trends)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `contribution_velocity` | `> 0.01` | `green` with `↗` | Seed contribution trending up |
| `contribution_velocity` | `< -0.01` | `yellow` with `↘` | Seed contribution trending down |

### Delta Display (Bar Lines)

| Metric | Condition | Style | Meaning |
|--------|-----------|-------|---------|
| `delta` | `> 0` | `green` | Positive contribution |
| `delta` | `< 0` | `red` | Negative contribution |
| `delta` | `= 0` | `dim` | No change |

---

## Rendering Logic

### Availability Check
The widget first checks `self._matrix.strategy == "unavailable"` or `not self._matrix.configs`. If either is true, it renders the unavailable state with `--` placeholders maintaining the same visual structure.

### Waterfall Visualization (`_render_waterfall`)

1. **Strategy Indicator**: If `strategy == "ablation_only"`, shows "Live Ablation Analysis" header with italic note about cached baselines.

2. **Baseline Row**: Shows "Baseline (Host only)" with accuracy bar.

3. **Individual Section**: For each slot in `individual_contributions()`, shows contribution bar with delta from baseline.

4. **Pairs Section**:
   - For 2-3 seeds: Shows all pairs inline with synergy highlighting
   - For 4+ seeds: Shows "Top Combinations (by synergy)" - top 5 pairs sorted by pair synergy descending

5. **Combined Section**: Shows "All seeds" with total improvement from baseline.

6. **Synergy Summary**:
   - Expected (sum of solo contributions)
   - Actual improvement
   - Interference warning (loud red reverse if synergy < -0.5 and n_seeds >= 2)
   - Synergy indicator (green if synergy > 0.5)
   - Neutral indicator (dim if -0.5 to 0.5)
   - Note: Single-seed scenarios never show synergy/interference (mathematically 0)

7. **Seed Dynamics Section** (`_render_interaction_metrics`):
   - Always visible to prevent layout shifts
   - Network synergy: Sum of `interaction_sum` across active seeds
   - Best partner: Seed with highest `boost_received` (if > 0.1)
   - Trends: Seeds with positive/negative `contribution_velocity` (up to 3 each)

### Bar Line Rendering (`_make_bar_line`)

Creates visual progress bars:
- Label (20 chars, left-aligned)
- Progress bar (30 chars width): `█` for filled (cyan), `░` for empty (dim)
- Value percentage (5 chars)
- Optional delta with color coding

### Active Seed Filtering

For interaction metrics, seeds are filtered to exclude:
- `DORMANT`
- `PRUNED`
- `EMBARGOED`
- `RESETTING`

This ensures only actively contributing seeds are included in network synergy calculations.

---

## Data Flow

```
SanctumSnapshot
    └── envs[focused_env_id]
            └── counterfactual_matrix: CounterfactualSnapshot
            └── seeds: dict[str, SeedState]
                    ↓
            CounterfactualPanel.__init__(matrix, seeds)
                    ↓
            render() → Panel
```

The widget receives:
1. `CounterfactualSnapshot` directly (not from SanctumSnapshot path)
2. Optional `seeds` dict for interaction metrics

Updates occur via `update_matrix(matrix, seeds)` method.

---

## Panel Styling

| Condition | Border Style | Title |
|-----------|--------------|-------|
| Available data | `cyan` | "Counterfactual Analysis" |
| Unavailable | `dim` | "Counterfactual Analysis" |
