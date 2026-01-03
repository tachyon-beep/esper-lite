# Telemetry Audit: RewardHealthPanel

**File:** `src/esper/karn/sanctum/widgets/reward_health.py`
**Audit Date:** 2026-01-03

## Overview

The RewardHealthPanel displays DRL Expert recommended metrics for monitoring reward signal health during training. It provides a compact single-line display with color-coded health indicators.

---

## RewardHealthData Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pbrs_fraction` | `float` | `0.0` | Ratio of PBRS (Potential-Based Reward Shaping) to total reward: `\|PBRS\| / \|total_reward\|` |
| `anti_gaming_trigger_rate` | `float` | `0.0` | Fraction of steps where anti-gaming penalties were applied |
| `ev_explained` | `float` | `0.0` | Value function explained variance (how well value function predicts returns) |
| `hypervolume` | `float` | `0.0` | Pareto hypervolume indicator for multi-objective optimization |

---

## Health Threshold Properties

| Property | Condition | Healthy Range | Rationale |
|----------|-----------|---------------|-----------|
| `is_pbrs_healthy` | `0.1 <= pbrs_fraction <= 0.4` | 10-40% | DRL Expert recommendation: PBRS should contribute meaningfully but not dominate |
| `is_gaming_healthy` | `anti_gaming_trigger_rate < 0.05` | <5% | Anti-gaming penalties should be infrequent edge cases |
| `is_ev_healthy` | `ev_explained > 0.5` | >0.5 | Value function should explain majority of return variance |

**Note:** `hypervolume` has no health threshold; it is expected to increase monotonically over training.

---

## Rendering Behavior

### Display Format

Single-line compact format:
```
PBRS:25% Game:2% EV:0.72 HV:1.2
```

### Color Coding

| Metric | Healthy Color | Unhealthy Color | Notes |
|--------|---------------|-----------------|-------|
| PBRS fraction | `green` | `red` | Binary healthy/unhealthy based on 10-40% range |
| Anti-gaming rate | `green` | `red` | Binary healthy/unhealthy based on <5% threshold |
| EV explained | `green` | `yellow` | Uses `yellow` for unhealthy (warning rather than error) |
| Hypervolume | `cyan` | N/A | Always cyan; no health threshold defined |

### Formatting

| Field | Format Specifier | Example Output |
|-------|------------------|----------------|
| `pbrs_fraction` | `:.0%` | `25%` |
| `anti_gaming_trigger_rate` | `:.0%` | `2%` |
| `ev_explained` | `:.2f` | `0.72` |
| `hypervolume` | `:.1f` | `1.2` |

---

## Widget Behavior

- **Border Title:** `"REWARD HEALTH"`
- **Initialization:** Accepts optional `RewardHealthData`; defaults to zero-initialized instance
- **Update Method:** `update_data(data: RewardHealthData)` replaces internal data and triggers refresh

---

## Data Source

The audit does not identify where `RewardHealthData` is populated. This requires tracing:
- PBRS reward decomposition from reward system
- Anti-gaming penalty tracking from reward shaping
- Explained variance calculation from PPO/value function training
- Hypervolume computation from multi-objective optimizer

---

## Observations

1. **No hypervolume health threshold:** The widget tracks hypervolume but does not define what constitutes healthy hypervolume. This is appropriate since hypervolume should increase monotonically rather than stay within a range.

2. **EV uses yellow for unhealthy:** Unlike PBRS and gaming rate which use red for unhealthy states, EV explained uses yellow, treating low explained variance as a warning rather than an error.

3. **All defaults are zero:** This means an uninitialized panel will show all metrics as unhealthy (PBRS 0% is outside 10-40%, EV 0.0 is below 0.5), which is appropriate fail-safe behavior.
