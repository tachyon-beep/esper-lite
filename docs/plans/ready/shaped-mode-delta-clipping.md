# SHAPED Mode Delta Clipping

## Metadata

| Field | Value |
|-------|-------|
| Status | Ready |
| Priority | High |
| Created | 2026-01-10 |
| Reviewed By | drl-expert (2026-01-10) |

## Problem Summary

SHAPED mode has **unbounded reward inflation** for long-lived seeds. The issue was identified through telemetry analysis of training run `telemetry_2026-01-10_174336`:

| Episode | Final Accuracy | Seed Contribution | Episode Reward |
|---------|----------------|-------------------|----------------|
| 848     | 24.15%         | 15-18%            | 760            |
| 899     | 22.95%         | 13-16%            | 744            |
| 979     | 23.9%          | 0 (no seed!)      | 719            |

### Root Cause

In SHAPED mode, the reward uses **cumulative** `seed_contribution`:

```python
# Line 423-427 in contribution.py
attributed = _compute_attributed_value(
    progress=progress,                    # val_acc - acc_at_germination (grows over time)
    seed_contribution=seed_contribution,  # val_acc - baseline_with_seed_disabled (cumulative!)
    formula=config.attribution_formula,
)
```

As the host+seed improve together:
1. `progress` grows cumulatively
2. `seed_contribution` grows cumulatively
3. `attributed` grows without bound
4. Each step's reward compounds, even though the **per-step improvement is marginal**

ESCROW mode solves this with `escrow_delta_clip=2.0` which clips per-step changes to ±2.0.

### DRL Expert Analysis

The specialist confirmed (2026-01-10):

> "The problem isn't that the 18% counterfactual measurement is wrong - it's that using cumulative progress in the reward formula causes credit to compound over time. A seed maintaining a constant 5% contribution gets progressively higher rewards as the episode progresses because `progress` keeps growing. This violates the RL principle of temporal credit assignment."

## Proposed Solution

Add `shaped_delta_clip` parameter to `ContributionRewardConfig` and track previous attribution for delta calculation in SHAPED mode.

### Design Rationale

**Why delta-clipping instead of value-clipping?**

1. **Temporal credit**: Per-step rewards should reflect per-step contribution, not accumulated history
2. **PPO stability**: Large reward variance destabilizes PPO's trust region (KL constraint)
3. **Consistency**: Matches ESCROW mode's proven approach (`escrow_delta_clip=2.0`)

**Why not use ESCROW mode everywhere?**

ESCROW requires additional state (`stable_val_acc` window, `escrow_credit_prev`) and is more complex. SHAPED mode is simpler for quick experiments and debugging. Both should be viable.

## Implementation Plan

### Phase 1: Add Configuration Parameter

**File:** `src/esper/simic/rewards/contribution.py`

Add to `ContributionRewardConfig`:

```python
# === SHAPED Mode Parameters ===
# Per-step cap on attribution delta (mirrors ESCROW's escrow_delta_clip).
# Clips large reward swings from cumulative seed_contribution inflation.
# Set to 0.0 to disable (legacy behavior, not recommended).
shaped_delta_clip: float = 2.0

# Previous bounded_attribution value for delta calculation.
# Only used when shaped_delta_clip > 0 and return_components=True.
# The caller is responsible for tracking and passing this value.
```

### Phase 2: Add Delta Tracking to RewardComponentsTelemetry

**File:** `src/esper/simic/rewards/reward_telemetry.py`

Add field:

```python
shaped_delta: float = 0.0
shaped_prev_attribution: float = 0.0
shaped_clipped: bool = False
```

### Phase 3: Implement Delta Clipping in SHAPED Mode

**File:** `src/esper/simic/rewards/contribution.py`

After computing `bounded_attribution` in the non-ESCROW branch (around line 436), add:

```python
# SHAPED mode delta clipping (mirrors ESCROW's escrow_delta_clip)
if config.shaped_delta_clip > 0 and return_components:
    shaped_delta = bounded_attribution - shaped_prev_attribution
    if abs(shaped_delta) > config.shaped_delta_clip:
        shaped_delta = math.copysign(config.shaped_delta_clip, shaped_delta)
        bounded_attribution = shaped_prev_attribution + shaped_delta
        if components:
            components.shaped_clipped = True
    if components:
        components.shaped_delta = shaped_delta
        components.shaped_prev_attribution = shaped_prev_attribution
```

### Phase 4: Add Function Parameter

**File:** `src/esper/simic/rewards/contribution.py`

Add parameter to `compute_contribution_reward()`:

```python
def compute_contribution_reward(
    ...
    shaped_prev_attribution: float = 0.0,  # NEW: For SHAPED mode delta clipping
) -> float | tuple[float, RewardComponentsTelemetry]:
```

### Phase 5: Wire Through Vectorized Trainer

**File:** `src/esper/simic/training/vectorized_trainer.py`

Track `shaped_prev_attribution` per-env in the training loop, similar to how `escrow_credit_prev` is tracked for ESCROW mode.

### Phase 6: Add Tests

**File:** `tests/simic/rewards/test_shaped_delta_clip.py`

Test cases:
1. `test_shaped_delta_clip_bounds_large_jumps` - verify clipping works
2. `test_shaped_delta_clip_disabled_by_zero` - verify 0.0 disables
3. `test_shaped_delta_clip_negative_clipping` - verify negative deltas clipped
4. `test_shaped_vs_escrow_similar_bounds` - verify both modes have bounded variance

## Verification Plan

### Unit Tests

```bash
uv run pytest tests/simic/rewards/test_shaped_delta_clip.py -v
```

### Integration Test

Run short training and verify:
1. Per-step rewards bounded (no 15+ per-step values)
2. Episode rewards don't explode for long-lived seeds
3. Telemetry shows `shaped_clipped=True` events when clipping activates

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --preset cifar10 --task cifar10 --rounds 5 --envs 2 \
    --episode-length 20 --reward-mode shaped --no-tui
```

### Telemetry Query

After fix, verify bounded variance:

```sql
SELECT
    epoch,
    AVG(bounded_attribution) as avg_attr,
    MAX(bounded_attribution) as max_attr,
    AVG(shaped_delta) as avg_delta,
    MAX(ABS(shaped_delta)) as max_delta,
    SUM(CASE WHEN shaped_clipped THEN 1 ELSE 0 END) as clip_count
FROM rewards
WHERE run_dir = '<new_run>' AND reward_mode = 'shaped'
GROUP BY epoch ORDER BY epoch
```

## Rollback Plan

Set `shaped_delta_clip=0.0` in config or CLI to disable clipping and restore legacy behavior.

## Related Work

- ESCROW mode already implements this via `escrow_delta_clip` (lines 407-411)
- DRL Expert review confirmed this approach aligns with reward shaping best practices (Ng et al., 1999)
