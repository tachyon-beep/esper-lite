# Telemetry Audit: ShapleyPanel

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/shapley_panel.py`
**Purpose:** Visualizes Shapley value attribution for slot contributions, showing per-slot marginal contributions with uncertainty bounds and statistical significance indicators.

---

## Telemetry Fields Consumed

### Source: ShapleySnapshot (via constructor `snapshot` parameter)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `values` | `dict[str, ShapleyEstimate]` | `{}` | Primary data - maps slot_id to Shapley estimate. If empty, renders "unavailable" state |
| `epoch` | `int` | `0` | Displayed in header as "[Epoch N]" or "Latest" if epoch is 0 |
| `slot_ids` | `tuple[str, ...]` | `()` | Not directly accessed (uses `values.keys()` instead) |
| `timestamp` | `datetime \| None` | `None` | Not displayed in current widget |

### Source: ShapleySnapshot.ranked_slots() method

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| Returns `list[tuple[str, float]]` | - | - | Slots ranked by mean contribution (descending) for display order |

### Source: ShapleySnapshot.get_significance() method

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| Returns `bool` | - | - | Checks if slot contribution is statistically significant at 95% CI (z=1.96) |

### Source: ShapleyEstimate (nested in ShapleySnapshot.values)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `mean` | `float` | `0.0` | Primary value displayed; determines color coding and ranking |
| `std` | `float` | `0.0` | Displayed as uncertainty bound (e.g., "+/-0.123") |
| `n_samples` | `int` | `0` | Not displayed in current widget |

### Source: SeedState (via constructor `seeds` parameter)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `stage` | `str` | `"DORMANT"` | Used to color-code slot names by lifecycle stage |

---

## Thresholds and Color Coding

### Shapley Value Direction (mean contribution)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| `mean` | `> 0.01` | `green` | Positive contribution to ensemble |
| `mean` | `< -0.01` | `red` | Negative contribution (hurting performance) |
| `mean` | `-0.01 <= mean <= 0.01` | `dim` | Negligible contribution |

### Statistical Significance (computed by `get_significance()`)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Significant + positive mean | `abs(mean) > 1.96 * std` | `bold green` | Statistically significant positive contributor |
| Significant + negative mean | `abs(mean) > 1.96 * std` | `bold red` | Statistically significant negative contributor |
| Not significant | `abs(mean) <= 1.96 * std` | `dim` | Contribution within noise bounds |

### Significance Indicator Characters

| Character | Condition | Meaning |
|-----------|-----------|---------|
| `*` (star) | `is_significant == True` | 95% CI excludes zero |
| `o` (circle) | `is_significant == False` | Cannot distinguish from zero |

### Seed Stage Color Coding (for slot names)

| Stage | Style | Description |
|-------|-------|-------------|
| `DORMANT` | `dim` | Inactive slot |
| `GERMINATED` | `yellow` | Newly created module |
| `TRAINING` | `cyan` | Learning from host errors |
| `HOLDING` | `magenta` | Held at current alpha |
| `BLENDING` | `blue` | Being integrated into host |
| `FOSSILIZED` | `green bold` | Permanently fused |
| `PRUNED` | `red dim` | Removed due to poor performance |
| `EMBARGOED` | `red` | Temporarily blocked |
| `RESETTING` | `yellow dim` | Being reset |
| (unknown) | `white` | Fallback for unrecognized stages |

---

## Rendering Logic

### Unavailable State

When `self._snapshot.values` is empty (falsy):
- Displays "Shapley values unavailable" message
- Explains that values are "Computed at episode boundaries when 2+ seeds are active"
- Panel border styled as `dim`

### Active State

When Shapley values are present:
1. **Header**: Shows epoch context as "[Epoch N]" or "[Latest]" if epoch is 0
2. **Column Headers**: "Slot", "Shapley", "+/-Std", "Sig"
3. **Per-Slot Rows** (ranked by mean, descending):
   - Slot ID: Left-aligned, colored by seed stage
   - Mean: Right-aligned, signed format (+/-), colored by direction
   - Std: Right-aligned with +/- prefix, styled dim
   - Significance: Star (*) or circle (o) with appropriate color
4. **Summary**: Total contribution (sum of all means)
5. **Legend**: Explains significance symbols
6. Panel border styled as `cyan`

### Value Formatting

- For `|mean| < 1`: Format as `+/-X.XXX` (3 decimal places)
- For `|mean| >= 1`: Format as `+/-X.X` (1 decimal place)
- Same logic applies to `std` formatting

---

## Data Flow

```
SanctumSnapshot
    |
    +-- EnvState[focused_env_id]
            |
            +-- shapley_snapshot: ShapleySnapshot
            |       |
            |       +-- values: dict[str, ShapleyEstimate]
            |       |       |
            |       |       +-- [slot_id]: ShapleyEstimate
            |       |               |
            |       |               +-- mean: float
            |       |               +-- std: float
            |       |
            |       +-- epoch: int
            |       +-- ranked_slots() -> list[tuple[str, float]]
            |       +-- get_significance(slot_id) -> bool
            |
            +-- seeds: dict[str, SeedState]
                    |
                    +-- [slot_id]: SeedState
                            |
                            +-- stage: str
```

---

## Notes

1. **Shapley Computation Timing**: Values are computed at episode boundaries via permutation sampling, so they may lag behind real-time slot states during episodes.

2. **Minimum Seeds Required**: Widget explicitly notes that 2+ seeds must be active for Shapley computation (meaningful marginal contribution requires alternatives).

3. **Significance Test**: Uses z=1.96 (95% confidence interval). The test checks if `|mean| > 1.96 * std` with special case for zero std: if `std == 0`, significance is `mean != 0`.

4. **Stage Lookup**: If a slot exists in Shapley values but not in seeds dict, the stage defaults to "DORMANT" for styling purposes.
