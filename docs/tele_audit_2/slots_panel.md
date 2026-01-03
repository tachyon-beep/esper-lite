# Telemetry Audit: SlotsPanel

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo_brain/slots_panel.py`
**Purpose:** Displays slot stage distribution across all environments with proportional bars, plus seed lifecycle aggregate statistics (fossilization, pruning, germination rates and trends).

---

## Telemetry Fields Consumed

### Source: SanctumSnapshot (root)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `envs` | `dict[int, EnvState]` | `{}` | Used to count number of environments for border title display (`len(snapshot.envs)`) |
| `total_slots` | `int` | `0` | Total slot count displayed in border title and used for zero-check |
| `slot_stage_counts` | `dict[str, int]` | `{}` | Stage distribution: keys are stage names (DORMANT, GERMINATED, etc.), values are counts across all envs |
| `seed_lifecycle` | `SeedLifecycleStats` | (see below) | Aggregate lifecycle metrics for bottom section |

### Source: SeedLifecycleStats (via `snapshot.seed_lifecycle`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `active_count` | `int` | `0` | Current active (non-terminal) seed count, displayed as `Active:{active_count}/{total_slots}` |
| `total_slots` | `int` | `0` | Total slots denominator for active ratio |
| `fossilize_count` | `int` | `0` | Cumulative fossilizations, displayed as `Foss:{count}` |
| `prune_count` | `int` | `0` | Cumulative prunes, displayed as `Prune:{count}` with conditional styling |
| `germination_count` | `int` | `0` | Cumulative germinations, displayed as `Germ:{count}` |
| `germination_rate` | `float` | `0.0` | Germinations per episode, displayed as `{rate:.1f}/ep` |
| `prune_rate` | `float` | `0.0` | Prunes per episode, displayed as `{rate:.1f}/ep` |
| `fossilize_rate` | `float` | `0.0` | Fossilizations per episode, displayed as `{rate:.2f}/ep` |
| `germination_trend` | `str` | `"stable"` | Trend indicator for germination rate: `"rising"`, `"stable"`, or `"falling"` |
| `prune_trend` | `str` | `"stable"` | Trend indicator for prune rate |
| `fossilize_trend` | `str` | `"stable"` | Trend indicator for fossilization rate |
| `blend_success_rate` | `float` | `0.0` | Ratio of fossilized/(fossilized+pruned), displayed as percentage |
| `avg_lifespan_epochs` | `float` | `0.0` | Mean epochs a seed lives before terminal state, displayed as `Lifespan: mu{epochs:.0f} eps` |

### Source: Leyline Constants

| Constant | Type | Value | Usage |
|----------|------|-------|-------|
| `STAGE_COLORS` | `dict[str, str]` | See below | Maps stage names to Rich color styles for bars and counts |
| `STAGE_ABBREVIATIONS` | `dict[str, str]` | See below | Maps stage names to 4-5 char abbreviations |

**STAGE_COLORS values:**
- `DORMANT`: `"dim"`
- `GERMINATED`: `"bright_blue"`
- `TRAINING`: `"cyan"`
- `HOLDING`: `"magenta"`
- `BLENDING`: `"yellow"`
- `FOSSILIZED`: `"green"`

**STAGE_ABBREVIATIONS values:**
- `DORMANT`: `"Dorm"` (displayed as `"DORM"`)
- `GERMINATED`: `"Germ"` (displayed as `"GERM"`)
- `TRAINING`: `"Train"` (displayed as `"TRAIN"`)
- `HOLDING`: `"Hold"` (displayed as `"HOLD"`)
- `BLENDING`: `"Blend"` (displayed as `"BLEND"`)
- `FOSSILIZED`: `"Foss"` (displayed as `"FOSS"`)

---

## Thresholds and Color Coding

### Stage Distribution Section

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Stage count | `count > 0` | Stage's `STAGE_COLORS` value | Active stage with slots in it |
| Stage count | `count == 0` | `"dim"` | No slots in this stage |
| Bar character | `stage == "DORMANT"` | `"○"` (hollow) | Dormant slots use hollow circles |
| Bar character | `stage != "DORMANT"` | `"●"` (filled) | Active stages use filled circles |
| Zero bar | `bar_width == 0` | `"·"` (dim) | Placeholder dot when no slots in stage |

### Lifecycle Section

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| Prune count | `prune_count > fossilize_count` | `"red"` | More prunes than fossilizations (unhealthy) |
| Prune count | `prune_count <= fossilize_count` | `"dim"` | Normal pruning ratio |
| Active count | always | `"cyan"` | Active slot display |
| Fossilize count | always | `"blue"` | Fossilization count |
| Germination count | always | `"green"` | Germination count |
| Germination trend | `"rising"` | `"green"` + `"up-right arrow"` | Germination rate increasing |
| Germination trend | `"stable"` | `"dim"` + `"right arrow"` | Germination rate stable |
| Germination trend | `"falling"` | `"red"` + `"down-right arrow"` | Germination rate decreasing |
| Prune trend | `"rising"` | `"green"` + `"up-right arrow"` | Prune rate increasing (note: same color logic) |
| Prune trend | `"falling"` | `"red"` + `"down-right arrow"` | Prune rate decreasing |
| Fossilize trend | same pattern | same | Same trend indicator logic |
| Blend success rate | `rate >= 70%` | `"green"` | Excellent blend success |
| Blend success rate | `50% <= rate < 70%` | `"yellow"` | Moderate blend success |
| Blend success rate | `rate < 50%` | `"red"` | Poor blend success (more failures than successes) |
| Avg lifespan | always | `"cyan"` | Lifespan display |

---

## Rendering Logic

### Border Title
The panel title dynamically shows the total slot count and environment count:
```
CURRENT SLOTS - {total_slots} across {n_envs} envs
```

### Stage Distribution (Top Section)
For each stage in order `[DORMANT, GERMINATED, TRAINING, BLENDING, HOLDING, FOSSILIZED]`:
1. Display 5-char left-aligned abbreviation (dim)
2. Display 3-char right-aligned count (stage color if >0, dim if 0)
3. Display proportional bar (max width 24 chars):
   - Bar width = `(count / total) * 24`
   - Uses `"○"` for DORMANT, `"●"` for others
   - Shows `"·"` placeholder if count is 0

### Separator
40-character dim horizontal line (`"─" * 40`)

### Lifecycle Metrics (Bottom Section)
**Line 1 - Cumulative counts:**
```
Active:{active}/{total}  Foss:{foss}  Prune:{prune}  Germ:{germ}
```

**Line 2 - Per-episode rates with trend arrows:**
```
Germ{arrow}{rate:.1f}/ep  Prune{arrow}{rate:.1f}/ep  Foss{arrow}{rate:.2f}/ep
```
Where `{arrow}` is `"up-right"` (green), `"right"` (dim), or `"down-right"` (red).

**Line 3 - Quality metrics:**
```
Lifespan: mu{epochs:.0f} eps  Blend:{rate:.0f}% success
```

### No Data States
- `snapshot is None`: Shows `"[no data]"` (dim)
- `total_slots == 0`: Shows `"[no environments]"` (dim)

---

## Data Flow

1. `SanctumSnapshot` is passed to `update_snapshot()` method
2. Method caches snapshot reference and updates border title
3. On `render()` call:
   - Reads `slot_stage_counts` for stage distribution bars
   - Reads `seed_lifecycle` for aggregate lifecycle metrics
   - Uses `STAGE_COLORS` and `STAGE_ABBREVIATIONS` from leyline for styling
4. Returns `Rich.Text` object with all formatted content
