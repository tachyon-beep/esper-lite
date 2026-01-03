# Telemetry Audit: Scoreboard Widget

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/scoreboard.py`
**Purpose:** Two-panel leaderboard showing top 5 environments by peak accuracy (Best Runs) and bottom 5 runs with most regression from peak (Worst Trajectory).

---

## Telemetry Fields Consumed

### Source: SanctumSnapshot (root)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `best_runs` | `list[BestRunRecord]` | `[]` | Primary data source for both panels; sorted by peak_accuracy (descending) for Best Runs, by trajectory delta (ascending) for Worst Trajectory |
| `cumulative_fossilized` | `int` | `0` | Displayed in stats header as "Foss: N" |
| `cumulative_pruned` | `int` | `0` | Displayed in stats header as "Prune: N" |
| `envs` | `dict[int, EnvState]` | `{}` | Fallback for computing global_best and mean_best when `best_runs` is empty |

### Source: EnvState (via `snapshot.envs.values()`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `best_accuracy` | `float` | `0.0` | Fallback computation of global best and mean best when no BestRunRecord entries exist |

### Source: BestRunRecord (via `snapshot.best_runs`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `record_id` | `str` | `""` | Row key for DataTable; used for pin toggle targeting |
| `episode` | `int` | N/A | Displayed as "Ep" column (1-indexed: `episode + 1`) |
| `epoch` | `int` | `0` | Displayed as "@" column with color coding based on when peak was achieved |
| `peak_accuracy` | `float` | N/A | Displayed as "Peak" column; used for sorting Best Runs and computing trajectory delta |
| `final_accuracy` | `float` | N/A | Used to compute trajectory delta (`final - peak`); displayed in "Traj" column with arrow |
| `growth_ratio` | `float` | `1.0` | Displayed as "Grw" column showing model size ratio |
| `seeds` | `dict[str, SeedState]` | `{}` | Displayed as "Seeds" column; counts by stage (blending/holding/fossilized) |
| `pinned` | `bool` | `False` | Pinned count shown in panel title as "BEST RUNS (N pin)" |

### Source: SeedState (via `record.seeds.values()`)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `stage` | `str` | `"DORMANT"` | Counted to produce seed status display (B/H/F counts) |

---

## Thresholds and Color Coding

### Epoch Color Coding (`_format_epoch`)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| epoch | < 25 | `[green]` | Early peak - lots of room to grow |
| epoch | 25-49 | plain (white) | Mid peak |
| epoch | 50-64 | `[yellow]` | Late peak |
| epoch | >= 65 | `[red]` | Very late peak - near max epochs |

### Trajectory Color Coding (`_format_trajectory`)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| delta (final - peak) | > +0.5% | `[green]` + `arrow_up_right` | Still climbing |
| delta | -1.0% to +0.5% | `[dim]` + `arrow_right` | Held steady |
| delta | -2.0% to -1.0% | `[dim]` + `arrow_down_right` | Small regression |
| delta | -5.0% to -2.0% | `[yellow]` + `arrow_down_right` | Moderate regression |
| delta | < -5.0% | `[red]` + `arrow_down_right` | Severe regression |

### Growth Ratio Color Coding (`_format_growth_ratio`)

| Metric | Threshold | Style | Meaning |
|--------|-----------|-------|---------|
| growth_ratio | <= 1.0 | `[dim]1.00x` | No growth (no fossilized params) |
| growth_ratio | 1.0 < x < 1.1 | `[cyan]` | Modest growth (<10% overhead) |
| growth_ratio | >= 1.1 | `[bold cyan]` | Significant growth (>=10% overhead) |

### Worst Trajectory Filter

| Metric | Threshold | Effect |
|--------|-----------|--------|
| trajectory delta | < -0.5% | Only runs with at least 0.5% regression are included in Worst Trajectory panel |

### Peak Accuracy Color Coding

| Panel | Style | Meaning |
|-------|-------|---------|
| Best Runs | `[bold green]` | Peak accuracy (best runs) |
| Worst Trajectory | `[yellow]` | Peak accuracy (regressed runs) |

### Seed Stage Colors (from `leyline.STAGE_COLORS`)

| Stage | Color | Usage |
|-------|-------|-------|
| BLENDING | `yellow` | First position in B/H/F count |
| HOLDING | `magenta` | Second position in B/H/F count |
| FOSSILIZED | `green` | Third position in B/H/F count |

---

## Rendering Logic

### Stats Header (`_refresh_stats`)

1. Computes `global_best` (max peak_accuracy across all best_runs)
2. Computes `mean_best` (average peak_accuracy across all best_runs)
3. Fallback: if no best_runs, uses `best_accuracy` from all EnvState entries
4. Displays: `Best: {global_best}%  Mean: {mean_best}%  Foss: {total_fossilized}  Prune: {total_pruned}`
5. Updates panel title with pinned count if any records are pinned

### Best Runs Table (`_refresh_table`)

1. Sorts `best_runs` by `peak_accuracy` descending
2. Takes top 5 entries
3. For each record, displays:
   - Rank (1-5)
   - Episode number (1-indexed)
   - Epoch when peak achieved (color-coded)
   - Peak accuracy (bold green)
   - Trajectory arrow + final accuracy (color-coded)
   - Growth ratio (color-coded)
   - Seed counts as "B/H/F" format

### Worst Trajectory Table (`_refresh_bottom_table`)

1. Filters `best_runs` to only those with trajectory delta < -0.5%
2. Sorts by trajectory delta ascending (worst regression first)
3. Takes bottom 5 entries
4. Same column format as Best Runs, but peak accuracy in yellow

### Data Flow

```
SanctumSnapshot
    |
    +-- best_runs: list[BestRunRecord]
    |       |
    |       +-- record_id, episode, epoch, peak_accuracy, final_accuracy
    |       +-- growth_ratio, pinned
    |       +-- seeds: dict[str, SeedState]
    |               |
    |               +-- stage (counted for B/H/F display)
    |
    +-- cumulative_fossilized, cumulative_pruned (stats header)
    |
    +-- envs: dict[int, EnvState] (fallback for best_accuracy)
```

### Interactive Features

| Event | Handler | Effect |
|-------|---------|--------|
| Row selected | `on_data_table_row_selected` | Posts `BestRunSelected` message with record |
| Pin toggle (`p` key) | `request_pin_toggle` | Posts `BestRunPinToggled` message with record_id |

### State Preservation

- Cursor position and scroll position are saved before table refresh
- Restored after refresh to maintain user context during live updates
