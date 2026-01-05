# Telemetry Audit: NarrativePanel

**File:** `/home/john/esper-lite/src/esper/karn/sanctum/widgets/tamiyo/narrative_panel.py`
**Purpose:** Now/Why/Next guidance panel (replaces StatusBanner) combining a compact health summary with contextual drivers and a “what to watch next” hint.

---

## Overview

The `NarrativePanel` renders three sections:

1. **NOW**: Overall status label + key metrics (warmup spinner, NaN/Inf, KL, round progress, memory, A/B cohort tag).
2. **WHY**: Top drivers behind the current state (derived from snapshot + PPO/obs/episode signals).
3. **NEXT**: Operator hint for likely recovery trigger / what to monitor.

---

## Telemetry Fields Consumed

### Source: SanctumSnapshot (root)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `connected` | `bool` | `False` | Immediate critical status when disconnected |
| `training_thread_alive` | `bool \| None` | `None` | Immediate critical status when `False` |
| `staleness_seconds` | `float` | `inf` | “SLOW/STALE” status based on thresholds |
| `current_batch` | `int` | `0` | Warmup detection and round progress display |
| `max_batches` | `int` | `0` | Round progress denominator |
| `envs` | `dict[int, EnvState]` | `{}` | Stall rate for context line + WHY drivers |
| `total_slots` | `int` | `0` | Empty-slot pressure indicator |
| `active_slots` | `int` | `0` | Empty-slot pressure indicator |
| `aggregate_mean_accuracy` | `float` | `0.0` | Mean accuracy context + WHY drivers |
| `mean_accuracy_history` | `deque[float]` | `deque(maxlen=50)` | Trend hinting |
| `observation_stats` | `ObservationStats` | defaults | Obs health drivers (NaN/Inf/outliers/near-clip/drift) |
| `episode_stats` | `EpisodeStats` | defaults | Episode outcomes drivers (timeouts/success trends) |

### Source: TamiyoState (snapshot.tamiyo)

| Field | Type | Default | Usage |
|-------|------|---------|-------|
| `ppo_data_received` | `bool` | `False` | Gates metric rendering; shows “WAITING” until True |
| `group_id` | `str \| None` | `None` | A/B cohort tag in border title + NOW line |
| `nan_grad_count` | `int` | `0` | Highest-priority critical indicator (leftmost) |
| `inf_grad_count` | `int` | `0` | Highest-priority critical indicator (leftmost) |
| `entropy` | `float` | `0.0` | Overall status detection (critical/warning) |
| `explained_variance` | `float` | `0.0` | Overall status detection (value health) |
| `advantage_std` | `float` | `0.0` | Overall status detection (collapsed/exploding learning signal) |
| `kl_divergence` | `float` | `0.0` | NOW metric + status detection |
| `kl_divergence_history` | `deque[float]` | `deque(maxlen=10)` | Trend arrow |
| `clip_fraction` | `float` | `0.0` | Status drivers and policy-state coloring |
| `grad_norm` | `float` | `0.0` | Status drivers |
| `total_actions` | `int` | `0` | Context line “op mix” ratios |
| `action_counts` | `dict[str, int]` | `{}` | Context line “op mix” ratios |
| `infrastructure.compile_enabled` | `bool` | `False` | `[[compiled]]` suffix in border title |
| `infrastructure.memory_usage_percent` | `float` (property) | `0.0` | `[Mem:X%]` metric with thresholds |

---

## Thresholds and Color Coding

### Warmup

Warmup is defined as `current_batch < 50` and renders a spinner + label:
`WARMING UP [round {current_batch}/50]`.

### NaN/Inf (highest priority)

| Condition | Style | Visual |
|-----------|-------|--------|
| `nan_grad_count > 5` or `inf_grad_count > 5` | `red bold reverse` | Maximum visibility |
| `nan_grad_count > 0` or `inf_grad_count > 0` | `red bold` | Standard alert |

### KL divergence

Uses `TUIThresholds.KL_WARNING` / `TUIThresholds.KL_CRITICAL` and appends `!` when above warning.

### Memory usage percent

| Threshold | Style |
|-----------|-------|
| `> 90%` | `red bold` |
| `> 75%` | `yellow` |
| otherwise | `dim` |

---

## Border Title

- Default: `NARRATIVE`
- With group id: `NARRATIVE {A|B|C}`
- With torch.compile: `NARRATIVE … [[compiled]]`
