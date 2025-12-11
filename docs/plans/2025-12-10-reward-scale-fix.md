# Reward Scale and Value Function Stability Fix

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix reward scaling issues causing value function collapse (value_loss=155.99, EV<0.1) and prevent zero-contribution fossilization.

**Problem Summary:**
- Per-step rewards: +50 to +80 (stable range: -10 to +10)
- Episode returns: ~1589 (stable range: ~500)
- Value loss: 155.99 (healthy: 0.1-10)
- Seeds fossilizing with +0.00% Δacc but +50 attribution rewards

**Root Cause:** `contribution_weight=3.0` with geometric mean producing ~50+ per step attribution rewards, 5-8x too high for stable PPO.

**Tech Stack:** Python, pytest

**DRL Expert Sign-Off:** ✅ APPROVED (2025-12-10)
- Task 3 removed (RewardNormalizer already implemented in vectorized.py)
- Test file path corrected

---

## Task 1: Reduce Contribution Weights

**Files:**
- Modify: `src/esper/simic/rewards.py`

**Step 1: Update ContributionRewardConfig defaults**

Change lines 129 and 133:
```python
# Primary signal: seed contribution weight
contribution_weight: float = 3.0

# Proxy signal for pre-blending stages (when counterfactual unavailable)
# Lower weight than contribution_weight since it's noisier and conflated with host drift
proxy_contribution_weight: float = 1.0
```

To:
```python
# Primary signal: seed contribution weight
# Reduced from 3.0 to 1.0 for stable PPO (per-step rewards should be in [-10, +10])
contribution_weight: float = 1.0

# Proxy signal for pre-blending stages (when counterfactual unavailable)
# Proportionally reduced from 1.0 to 0.3 (maintains 3:1 ratio with contribution_weight)
proxy_contribution_weight: float = 0.3
```

**Step 2: Verify import works**

Run: `python -c "from esper.simic.rewards import ContributionRewardConfig; c = ContributionRewardConfig(); print(f'contribution_weight={c.contribution_weight}, proxy={c.proxy_contribution_weight}')"`

Expected: `contribution_weight=1.0, proxy=0.3`

**Step 3: Commit**

```bash
git add src/esper/simic/rewards.py && git commit -m "fix(rewards): reduce contribution weights for stable PPO

contribution_weight: 3.0 → 1.0
proxy_contribution_weight: 1.0 → 0.3

Per-step rewards were 50-80 (5-8x too high for PPO stability).
This brings them to ~15-25 range. Existing RewardNormalizer clips to [-10, +10]."
```

---

## Task 2: Add Minimum Contribution Threshold for Fossilization

**Files:**
- Modify: `src/esper/kasmina/slot.py`

**Step 1: Add MIN_FOSSILIZE_CONTRIBUTION constant**

Add after the existing gate-related constants (around line 300, near QualityGates class):

```python
# Minimum counterfactual contribution required for fossilization.
# Prevents "free rider" seeds that provide negligible value from becoming permanent.
# DRL rationale: seeds must provide economically significant contribution to justify
# their parameter cost. A 1% threshold ensures measurable causal impact.
MIN_FOSSILIZE_CONTRIBUTION = 1.0  # 1% minimum causal contribution
```

**Step 2: Update _check_g5() to enforce minimum threshold**

Change lines 510-514 in `_check_g5()`:
```python
# Check contribution is positive
if contribution > 0:
    checks_passed.append(f"positive_contribution_{contribution:.2f}%")
else:
    checks_failed.append("non_positive_contribution")
```

To:
```python
# Check contribution meets minimum threshold
# Prevents zero/negligible contribution seeds from fossilizing
if contribution >= MIN_FOSSILIZE_CONTRIBUTION:
    checks_passed.append(f"sufficient_contribution_{contribution:.2f}%")
else:
    checks_failed.append(
        f"insufficient_contribution_{contribution:.2f}%_below_{MIN_FOSSILIZE_CONTRIBUTION}%"
    )
```

**Step 3: Verify import works**

Run: `python -c "from esper.kasmina.slot import QualityGates, MIN_FOSSILIZE_CONTRIBUTION; print(f'MIN_FOSSILIZE_CONTRIBUTION={MIN_FOSSILIZE_CONTRIBUTION}')"`

Expected: `MIN_FOSSILIZE_CONTRIBUTION=1.0`

**Step 4: Commit**

```bash
git add src/esper/kasmina/slot.py && git commit -m "fix(kasmina): require minimum 1% contribution for fossilization

G5 gate now requires counterfactual_contribution >= 1.0% (was > 0).
Prevents 'free rider' seeds with negligible impact from becoming permanent.

DRL rationale: seeds must provide economically significant contribution
to justify their parameter cost."
```

---

## Task 3: Add Tests for Minimum Contribution Threshold

**Files:**
- Create: `tests/kasmina/test_g5_fossilization.py`

**Step 1: Create test file with G5 threshold tests**

```python
"""Tests for G5 fossilization gate minimum contribution threshold."""

import pytest
from esper.kasmina.slot import (
    QualityGates,
    MIN_FOSSILIZE_CONTRIBUTION,
    SeedState,
    SeedMetrics,
)
from esper.leyline import SeedStage


def create_probationary_state(contribution: float, healthy: bool = True) -> SeedState:
    """Create a SeedState in PROBATIONARY stage with given contribution.

    SeedState is a dataclass with required fields seed_id and blueprint_id.
    SeedMetrics.counterfactual_contribution is a settable field.
    """
    metrics = SeedMetrics()
    metrics.counterfactual_contribution = contribution

    state = SeedState(
        seed_id="test-seed",
        blueprint_id="test-blueprint",
        stage=SeedStage.PROBATIONARY,
        metrics=metrics,
        is_healthy=healthy,
    )
    return state


class TestG5MinimumContribution:
    """Test G5 gate enforces minimum contribution threshold."""

    def test_g5_rejects_below_threshold(self):
        """G5 gate should reject seeds with contribution below 1%."""
        gates = QualityGates()
        state = create_probationary_state(contribution=0.5)  # Below 1%

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert any("insufficient_contribution" in c for c in result.checks_failed)

    def test_g5_rejects_zero_contribution(self):
        """G5 gate should reject seeds with zero contribution."""
        gates = QualityGates()
        state = create_probationary_state(contribution=0.0)

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert any("insufficient_contribution" in c for c in result.checks_failed)

    def test_g5_rejects_negative_contribution(self):
        """G5 gate should reject seeds with negative contribution."""
        gates = QualityGates()
        state = create_probationary_state(contribution=-5.0)

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert any("insufficient_contribution" in c for c in result.checks_failed)

    def test_g5_accepts_at_threshold(self):
        """G5 gate should accept seeds with contribution exactly at 1%."""
        gates = QualityGates()
        state = create_probationary_state(contribution=1.0)

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert result.passed
        assert any("sufficient_contribution" in c for c in result.checks_passed)

    def test_g5_accepts_above_threshold(self):
        """G5 gate should accept seeds with contribution above 1%."""
        gates = QualityGates()
        state = create_probationary_state(contribution=5.0)

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert result.passed
        assert any("sufficient_contribution" in c for c in result.checks_passed)

    def test_min_fossilize_contribution_constant(self):
        """MIN_FOSSILIZE_CONTRIBUTION should be 1.0%."""
        assert MIN_FOSSILIZE_CONTRIBUTION == 1.0
```

**Step 2: Run tests**

Run: `pytest tests/kasmina/test_g5_fossilization.py -v --tb=short`

Expected: All 6 tests PASS

**Step 3: Commit**

```bash
git add tests/kasmina/test_g5_fossilization.py && git commit -m "test(kasmina): add tests for G5 minimum contribution threshold

Verifies that seeds with <1% contribution cannot fossilize."
```

---

## Task 4: Final Verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`

Expected: All tests PASS

**Step 2: Verify reward scale impact**

Run a quick sanity check:
```python
python -c "
from esper.simic.rewards import ContributionRewardConfig, compute_contribution_reward
from esper.simic.rewards import SeedInfo
from esper.leyline.actions import ActionEnum

config = ContributionRewardConfig()
info = SeedInfo(stage=4, epochs_in_stage=5, total_params=1000, alpha=0.8)

# Simulate a typical high-contribution step
reward = compute_contribution_reward(
    action=ActionEnum.WAIT,
    seed_contribution=0.15,  # 15% contribution
    val_acc=70.0,
    seed_info=info,
    epoch=10,
    max_epochs=25,
    counterfactual_available=True,
    config=config,
)
print(f'Reward for 15% contribution: {reward.total:.2f}')
print(f'  Attribution component: {reward.components.get(\"bounded_attribution\", 0):.2f}')
"
```

Expected: Total reward in range 5-20 (down from 50-80)

**Step 3: Verify existing RewardNormalizer is active**

Run: `grep -n "RewardNormalizer" src/esper/simic/vectorized.py | head -5`

Expected: Shows RewardNormalizer instantiation and usage (already implemented)

**Step 4: Verify no import errors**

Run: `python -c "from esper.simic import *; from esper.kasmina import *; print('All imports OK')"`

Expected: `All imports OK`

---

## Summary

| Task | Files Changed | Description |
|------|---------------|-------------|
| 1 | rewards.py | Reduce contribution_weight 3.0→1.0, proxy 1.0→0.3 |
| 2 | slot.py | Add MIN_FOSSILIZE_CONTRIBUTION=1.0% threshold |
| 3 | test_g5_fossilization.py | Add tests for new threshold |
| 4 | - | Final verification |

**Note:** RewardNormalizer is already implemented in vectorized.py (lines 346, 1149) and clips to [-10, +10].

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Per-step reward (raw) | +50-80 | +15-25 |
| Per-step reward (normalized) | N/A | [-10, +10] |
| Episode return | ~1589 | ~200-400 |
| Value loss | 155.99 | ~5-20 |
| Explained variance | <0.1 | >0.3 |
| Zero-contrib fossilization | Allowed | Blocked (<1%) |

---

## Monitoring After Deployment

Track these metrics over the next 5-15 PPO updates:

**Healthy trajectory:**
- Updates 1-3: value_loss 155 → 30-50, EV → 0.15-0.25
- Updates 4-10: value_loss → 5-15, EV → 0.4-0.6
- Updates 10+: Steady state

**Red flags requiring intervention:**
- Value loss increases for 3+ consecutive updates
- Explained variance goes negative
- KL divergence > 0.05 consistently
- Clip fraction > 0.3 consistently

---

## DRL Expert Notes

1. **PBRS becomes more prominent** after weight reduction - desirable for lifecycle progression
2. **Compute rent** becomes proportionally larger but still appropriately small
3. **Terminal bonus** increases from ~5-7% to ~15-25% of attribution - appropriate for end-of-episode signal
4. **Recovery timeline:** 5-15 PPO updates to return to healthy metrics
