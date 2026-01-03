# Telemetry Audit: HistoricalEnvDetail

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/historical_env_detail.py`

**Purpose:** Modal for viewing frozen environment state from Best Runs leaderboard. Shows the complete environment state as it was when the env achieved its peak accuracy. This is a static snapshot (no live updates).

**Triggered by:** Left-clicking a row in the Best Runs scoreboard.

---

## 1. Telemetry Fields Consumed from BestRunRecord

The widget receives a `BestRunRecord` instance and consumes the following fields:

### 1.1 Header Bar Fields (`_render_header`)

| Field | Type | Path | Default | Usage |
|-------|------|------|---------|-------|
| `episode` | `int` | `record.episode` | Required | Displayed as "Episode {episode + 1}" (1-indexed for display) |
| `env_id` | `int` | `record.env_id` | Required | Displayed as "Env {env_id}" in dim style |
| `peak_accuracy` | `float` | `record.peak_accuracy` | Required | Hero metric displayed as "Peak: {peak_accuracy:.1f}%" in bold green |
| `final_accuracy` | `float` | `record.final_accuracy` | Required | Displayed as "Final: {final_accuracy:.1f}%" in cyan; only shown if differs from peak_accuracy |
| `growth_ratio` | `float` | `record.growth_ratio` | `1.0` | Displayed as "Growth: {growth_ratio:.2f}x"; color-coded based on threshold |
| `reward_mode` | `str \| None` | `record.reward_mode` | `None` | A/B cohort indicator; displayed as "Cohort {reward_mode}" with color based on value |
| `pinned` | `bool` | `record.pinned` | `False` | Displayed as "PINNED" in bold cyan if True |
| `host_params` | `int` | `record.host_params` | `0` | Formatted via `format_params()` and displayed as "Host: {formatted}" |
| `fossilized_count` | `int` | `record.fossilized_count` | `0` | Displayed as "Fossilized: {count}" in green |
| `pruned_count` | `int` | `record.pruned_count` | `0` | Displayed as "Pruned: {count}" in red |

### 1.2 Seed Grid Fields (`compose`)

| Field | Type | Path | Default | Usage |
|-------|------|------|---------|-------|
| `slot_ids` | `list[str]` | `record.slot_ids` | `[]` | List of all slot IDs; falls back to `sorted(record.seeds.keys())` if empty |
| `seeds` | `dict[str, SeedState]` | `record.seeds` | `{}` | Maps slot_id to SeedState; passed to `SeedCard` widget for each slot |

### 1.3 Metrics Section Fields (`_render_metrics`)

| Field | Type | Path | Default | Usage |
|-------|------|------|---------|-------|
| `accuracy_history` | `list[float]` | `record.accuracy_history` | `[]` | Rendered as sparkline (40 chars wide) via `make_sparkline()` |
| `reward_history` | `list[float]` | `record.reward_history` | `[]` | Rendered as sparkline (40 chars wide) via `make_sparkline()` |
| `seeds` | `dict[str, SeedState]` | `record.seeds` | `{}` | Used to compute active seed count (excluding DORMANT, FOSSILIZED, PRUNED stages) |
| `fossilized_count` | `int` | `record.fossilized_count` | `0` | Displayed in "Seed Counts" row in green |
| `pruned_count` | `int` | `record.pruned_count` | `0` | Displayed in "Seed Counts" row in red |
| `reward_components` | `RewardComponents \| None` | `record.reward_components` | `None` | Full reward breakdown (see sub-fields below) |
| `action_history` | `list[str]` | `record.action_history` | `[]` | Last 5 actions displayed as "action1 -> action2 -> ..." |

#### 1.3.1 RewardComponents Sub-fields

| Field | Type | Path | Default | Usage |
|-------|------|------|---------|-------|
| `total` | `float` | `rc.total` | `0.0` | Primary reward display; bold green if >= 0, bold red if < 0 |
| `stage_bonus` | `float` | `rc.stage_bonus` | `0.0` | Used to compute PBRS fraction: `abs(stage_bonus) / abs(total)` |
| `base_acc_delta` | `float` | `rc.base_acc_delta` | `0.0` | Displayed in Signals row; green if > 0, red if < 0 |
| `compute_rent` | `float` | `rc.compute_rent` | `0.0` | Displayed in Signals row in red |
| `alpha_shock` | `float` | `rc.alpha_shock` | `0.0` | Displayed in Signals row in red |
| `ratio_penalty` | `float` | `rc.ratio_penalty` | `0.0` | Displayed in Signals row in red |
| `bounded_attribution` | `float` | `rc.bounded_attribution` | `0.0` | Displayed in Credits row; green if > 0, red if < 0 |
| `hindsight_credit` | `float` | `rc.hindsight_credit` | `0.0` | Displayed in Credits row in blue |
| `scaffold_count` | `int` | `rc.scaffold_count` | `0` | Appended to hindsight_credit display as "({count}x, {delay:.1f}e)" |
| `avg_scaffold_delay` | `float` | `rc.avg_scaffold_delay` | `0.0` | Appended to hindsight_credit display |
| `fossilize_terminal_bonus` | `float` | `rc.fossilize_terminal_bonus` | `0.0` | Displayed in Credits row in blue as "Foss: {value}" |
| `blending_warning` | `float` | `rc.blending_warning` | `0.0` | Displayed in Warnings row in yellow if < 0 |
| `holding_warning` | `float` | `rc.holding_warning` | `0.0` | Displayed in Warnings row in yellow if < 0 |

### 1.4 Graveyard Section Fields (`_render_graveyard`)

| Field | Type | Path | Default | Usage |
|-------|------|------|---------|-------|
| `blueprint_spawns` | `dict[str, int]` | `record.blueprint_spawns` | `{}` | Per-blueprint spawn count; displayed in cyan |
| `blueprint_fossilized` | `dict[str, int]` | `record.blueprint_fossilized` | `{}` | Per-blueprint fossilized count; displayed in green |
| `blueprint_prunes` | `dict[str, int]` | `record.blueprint_prunes` | `{}` | Per-blueprint prune count; displayed in red |

### 1.5 Counterfactual Section Fields (`compose`)

| Field | Type | Path | Default | Usage |
|-------|------|------|---------|-------|
| `counterfactual_matrix` | `CounterfactualSnapshot \| None` | `record.counterfactual_matrix` | `None` | Passed to `CounterfactualPanel`; defaults to empty snapshot with strategy="unavailable" |
| `seeds` | `dict[str, SeedState]` | `record.seeds` | `{}` | Also passed to `CounterfactualPanel` for seed context |

### 1.6 Shapley Section Fields (`compose`)

| Field | Type | Path | Default | Usage |
|-------|------|------|---------|-------|
| `shapley_snapshot` | `ShapleySnapshot \| None` | `record.shapley_snapshot` | `None` | Passed to `ShapleyPanel`; defaults to empty `ShapleySnapshot()` |
| `seeds` | `dict[str, SeedState]` | `record.seeds` | `{}` | Also passed to `ShapleyPanel` for seed context |

### 1.7 Footer Fields (`compose`)

| Field | Type | Path | Default | Usage |
|-------|------|------|---------|-------|
| `pinned` | `bool` | `record.pinned` | `False` | Displayed as "Pinned" or "Not pinned (right-click to pin)" |

---

## 2. Thresholds and Color Coding Logic

### 2.1 Growth Ratio (Header)

**Source:** `DisplayThresholds.GROWTH_RATIO_WARNING = 1.2`

| Condition | Style | Meaning |
|-----------|-------|---------|
| `growth > GROWTH_RATIO_WARNING` (1.2) | `yellow` | Model has grown >20% from fossilized seeds |
| `growth > 1.0` (but <= 1.2) | `green` | Normal growth |
| `growth <= 1.0` | `dim` | No growth (displayed as "1.00x") |

### 2.2 Reward Mode / Cohort (Header)

| Condition | Style |
|-----------|-------|
| `reward_mode == "A"` | `cyan` |
| `reward_mode != "A"` (e.g., "B") | `magenta` |

### 2.3 PBRS Health (Metrics - Reward Total)

**Source:** `DisplayThresholds.PBRS_HEALTHY_MIN = 0.1`, `DisplayThresholds.PBRS_HEALTHY_MAX = 0.4`

| Condition | Icon | Style |
|-----------|------|-------|
| `0.1 <= pbrs_fraction <= 0.4` | `checkmark` | `green` |
| `pbrs_fraction > 0.4` or `< 0.1` | `warning` | `yellow` |
| `pbrs_fraction == 0` | (no icon) | N/A |

### 2.4 Reward Components Styling

| Component | Condition | Style |
|-----------|-----------|-------|
| `total` | `>= 0` | `bold green` |
| `total` | `< 0` | `bold red` |
| `base_acc_delta` | `> 0` | `green` |
| `base_acc_delta` | `< 0` | `red` |
| `compute_rent` | any non-zero | `red` |
| `alpha_shock` | any non-zero | `red` |
| `ratio_penalty` | any non-zero | `red` |
| `bounded_attribution` | `> 0` | `green` |
| `bounded_attribution` | `< 0` | `red` |
| `hindsight_credit` | any non-zero | `blue` |
| `stage_bonus` | any non-zero | `blue` |
| `fossilize_terminal_bonus` | any non-zero | `blue` |
| `blending_warning` | `< 0` | `yellow` |
| `holding_warning` | `< 0` | `yellow` |

### 2.5 Blueprint Success Rate (Graveyard)

**Source:** `DisplayThresholds.BLUEPRINT_SUCCESS_GREEN = 0.50`, `DisplayThresholds.BLUEPRINT_SUCCESS_YELLOW = 0.25`

| Condition | Style | Meaning |
|-----------|-------|---------|
| `success_rate >= 0.50` | `green` | 50%+ of terminated seeds fossilized |
| `success_rate >= 0.25` | `yellow` | 25-50% fossilized |
| `success_rate < 0.25` | `red` | <25% fossilized |
| No terminated seeds | `dim` | Display "--" |

---

## 3. Rendering Details

### 3.1 Sparkline Generation

Uses `make_sparkline(values, width=40)` from `esper.karn.sanctum.schema`:
- Returns Unicode block characters representing value distribution
- Empty placeholder "dash" characters for missing data

### 3.2 Active Seed Calculation

```python
active_count = len([
    s for s in record.seeds.values()
    if s and s.stage not in ("DORMANT", "FOSSILIZED", "PRUNED")
])
```

### 3.3 Metrics Not Tracked in Historical Records

The following rows always display "--" placeholder:
- **Fossilized Params** - Not captured in BestRunRecord
- **Action Distribution** - Not captured in BestRunRecord
- **Gaming Rate** - Not captured in BestRunRecord (shown as "Gaming: --")

### 3.4 Recent Actions Display

Shows last 5 actions from `action_history` joined with " -> " separator.

---

## 4. Delegated Rendering

### 4.1 SeedCard Widget

Each seed in `slot_ids` is rendered via `SeedCard(seed, slot_id)` from `env_detail_screen.py`. The seed may be `None` for DORMANT slots.

### 4.2 CounterfactualPanel Widget

Receives:
- `counterfactual_matrix`: The `CounterfactualSnapshot` or default unavailable snapshot
- `seeds`: The seed state dict for context

### 4.3 ShapleyPanel Widget

Receives:
- `shapley_snapshot`: The `ShapleySnapshot` or default empty snapshot
- `seeds`: The seed state dict for context

---

## 5. Data Flow Summary

```
BestRunRecord (captured at peak accuracy)
    |
    +-- Header: episode, env_id, peak_accuracy, final_accuracy, growth_ratio,
    |           reward_mode, pinned, host_params, fossilized_count, pruned_count
    |
    +-- Seed Grid: slot_ids, seeds -> SeedCard widgets
    |
    +-- Metrics: accuracy_history, reward_history, seeds (for active count),
    |            fossilized_count, pruned_count, reward_components, action_history
    |
    +-- Graveyard: blueprint_spawns, blueprint_fossilized, blueprint_prunes
    |
    +-- Counterfactual: counterfactual_matrix, seeds -> CounterfactualPanel
    |
    +-- Shapley: shapley_snapshot, seeds -> ShapleyPanel
    |
    +-- Footer: pinned
```

---

## 6. BestRunRecord Schema Reference

From `esper/karn/sanctum/schema.py` (lines 1226-1274):

```python
@dataclass
class BestRunRecord:
    env_id: int
    episode: int
    peak_accuracy: float
    final_accuracy: float
    epoch: int = 0
    seeds: dict[str, SeedState] = field(default_factory=dict)
    slot_ids: list[str] = field(default_factory=list)
    growth_ratio: float = 1.0
    record_id: str = ""
    pinned: bool = False
    reward_components: RewardComponents | None = None
    counterfactual_matrix: CounterfactualSnapshot | None = None
    shapley_snapshot: ShapleySnapshot | None = None
    action_history: list[str] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)
    host_loss: float = 0.0
    host_params: int = 0
    fossilized_count: int = 0
    pruned_count: int = 0
    reward_mode: str | None = None
    blueprint_spawns: dict[str, int] = field(default_factory=dict)
    blueprint_fossilized: dict[str, int] = field(default_factory=dict)
    blueprint_prunes: dict[str, int] = field(default_factory=dict)
```
