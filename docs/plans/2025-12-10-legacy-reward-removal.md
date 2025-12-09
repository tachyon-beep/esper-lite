# Legacy Reward Code Removal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `RewardConfig`, `compute_shaped_reward`, and associated helper functions that are no longer called by any production code.

**Architecture:** Complete deletion of legacy reward system. All production code has been migrated to `compute_contribution_reward` with proxy signal support. No backwards compatibility needed per CLAUDE.md policy.

**Tech Stack:** Python, pytest

---

## Task 1: Delete Legacy Test File

**Files:**
- Delete: `tests/test_simic_fossilize_shaping.py`

**Step 1: Delete the test file**

```bash
rm tests/test_simic_fossilize_shaping.py
```

**Step 2: Verify deletion**

Run: `ls tests/test_simic_fossilize_shaping.py 2>&1`
Expected: `ls: cannot access 'tests/test_simic_fossilize_shaping.py': No such file or directory`

**Step 3: Commit**

```bash
git add -A && git commit -m "test: remove legacy fossilize shaping tests

These tests covered _advance_shaping() with RewardConfig, both
of which are being removed. Fossilize shaping for the new reward
system is tested via _contribution_fossilize_shaping."
```

---

## Task 2: Remove Legacy Strategy from Test Helpers

**Files:**
- Modify: `tests/strategies.py:314-328` (delete `reward_configs` function)
- Modify: `tests/strategies.py:456` (remove from `__all__`)

**Step 1: Delete the reward_configs strategy function**

Delete lines 314-328:
```python
@st.composite
def reward_configs(draw):
    """Generate RewardConfig instances with valid hyperparameters.

    All weights are randomized within reasonable bounds.
    """
    from esper.simic.rewards import RewardConfig

    return RewardConfig(
        acc_delta_weight=draw(bounded_floats(0.0, 2.0)),
        training_bonus=draw(bounded_floats(0.0, 1.0)),
        blending_bonus=draw(bounded_floats(0.0, 1.0)),
        fossilized_bonus=draw(bounded_floats(0.0, 2.0)),
        stage_improvement_weight=draw(bounded_floats(0.0, 0.5)),
    )
```

**Step 2: Remove from __all__**

On line 456, delete:
```python
    "reward_configs",
```

**Step 3: Run tests to verify strategies still work**

Run: `pytest tests/ -k "strategies" -v --tb=short`
Expected: PASS (no tests directly depend on reward_configs)

**Step 4: Commit**

```bash
git add tests/strategies.py && git commit -m "test: remove legacy reward_configs strategy

No longer needed after migration to compute_contribution_reward."
```

---

## Task 3: Remove Legacy Exports from simic/__init__.py

**Files:**
- Modify: `src/esper/simic/__init__.py:47-61` (imports section)
- Modify: `src/esper/simic/__init__.py:142-156` (__all__ section)

**Step 1: Update imports section**

Change lines 47-61 from:
```python
# Rewards
from esper.simic.rewards import (
    RewardConfig,
    LossRewardConfig,
    SeedInfo,
    compute_shaped_reward,
    compute_potential,
    compute_pbrs_bonus,
    compute_pbrs_stage_bonus,
    compute_loss_reward,
    compute_seed_potential,
    get_intervention_cost,
    STAGE_TRAINING,
    STAGE_BLENDING,
    STAGE_FOSSILIZED,
)
```

To:
```python
# Rewards
from esper.simic.rewards import (
    LossRewardConfig,
    ContributionRewardConfig,
    SeedInfo,
    compute_contribution_reward,
    compute_potential,
    compute_pbrs_bonus,
    compute_pbrs_stage_bonus,
    compute_loss_reward,
    compute_seed_potential,
    get_intervention_cost,
    STAGE_TRAINING,
    STAGE_BLENDING,
    STAGE_FOSSILIZED,
)
```

**Step 2: Update __all__ section**

Change lines 142-156 from:
```python
    # Rewards
    "RewardConfig",
    "LossRewardConfig",
    "SeedInfo",
    "compute_shaped_reward",
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_pbrs_stage_bonus",
    "compute_loss_reward",
    "compute_seed_potential",
    "get_intervention_cost",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
```

To:
```python
    # Rewards
    "LossRewardConfig",
    "ContributionRewardConfig",
    "SeedInfo",
    "compute_contribution_reward",
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_pbrs_stage_bonus",
    "compute_loss_reward",
    "compute_seed_potential",
    "get_intervention_cost",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
```

**Step 3: Verify imports work**

Run: `python -c "from esper.simic import *; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/simic/__init__.py && git commit -m "refactor(simic): remove legacy reward exports

Replace RewardConfig and compute_shaped_reward with
ContributionRewardConfig and compute_contribution_reward."
```

---

## Task 4: Update Comment in episodes.py

**Files:**
- Modify: `src/esper/simic/episodes.py:221`

**Step 1: Update the comment**

Change line 221 from:
```python
    # Computed reward (populated by compute_shaped_reward from simic.rewards)
```

To:
```python
    # Computed reward (populated by compute_contribution_reward from simic.rewards)
```

**Step 2: Commit**

```bash
git add src/esper/simic/episodes.py && git commit -m "docs: update reward function reference in episodes.py"
```

---

## Task 5: Remove Legacy Code from rewards.py (Part 1 - RewardConfig)

**Files:**
- Modify: `src/esper/simic/rewards.py:108-237` (delete RewardConfig class)

**Step 1: Delete RewardConfig class**

Delete lines 108-237 (the entire `@dataclass class RewardConfig:` including all its docstrings and fields).

The section to delete starts with:
```python
@dataclass(slots=True)
class RewardConfig:
    """Configuration for reward computation.
```

And ends just before:
```python
# =============================================================================
# Loss-Primary Reward Configuration (Phase 2)
# =============================================================================
```

**Step 2: Run import test**

Run: `python -c "from esper.simic.rewards import ContributionRewardConfig; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/esper/simic/rewards.py && git commit -m "refactor(rewards): remove legacy RewardConfig class

~130 lines removed. All callers migrated to ContributionRewardConfig."
```

---

## Task 6: Remove Legacy Code from rewards.py (Part 2 - Helper Functions)

**Files:**
- Modify: `src/esper/simic/rewards.py` (delete helper functions after RewardConfig removal)

Note: After Task 5, line numbers will have shifted. Use function names to locate.

**Step 1: Delete _germinate_shaping function**

Delete the entire `def _germinate_shaping(...)` function (~42 lines).

**Step 2: Delete _advance_shaping function**

Delete the entire `def _advance_shaping(...)` function (~43 lines).

**Step 3: Delete _cull_shaping function**

Delete the entire `def _cull_shaping(...)` function (~95 lines).

**Step 4: Delete _wait_shaping function**

Delete the entire `def _wait_shaping(...)` function (~46 lines).

**Step 5: Run import test**

Run: `python -c "from esper.simic.rewards import compute_contribution_reward; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/esper/simic/rewards.py && git commit -m "refactor(rewards): remove legacy shaping helper functions

Removed:
- _germinate_shaping
- _advance_shaping
- _cull_shaping
- _wait_shaping

~226 lines removed. All shaping now handled by compute_contribution_reward."
```

---

## Task 7: Remove Legacy Code from rewards.py (Part 3 - Main Function)

**Files:**
- Modify: `src/esper/simic/rewards.py` (delete compute_shaped_reward and _DEFAULT_CONFIG)

Note: After Tasks 5-6, line numbers will have shifted. Use function names to locate.

**Step 1: Delete compute_shaped_reward function**

Delete the entire `def compute_shaped_reward(...)` function (~162 lines).

**Step 2: Delete _DEFAULT_CONFIG singleton**

Delete the line:
```python
_DEFAULT_CONFIG = RewardConfig()
```

**Step 3: Run import test**

Run: `python -c "from esper.simic.rewards import compute_contribution_reward, SeedInfo; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/esper/simic/rewards.py && git commit -m "refactor(rewards): remove compute_shaped_reward function

~163 lines removed. compute_contribution_reward is now the sole
reward function for seed lifecycle control."
```

---

## Task 8: Update rewards.py Exports and Docstring

**Files:**
- Modify: `src/esper/simic/rewards.py` (update __all__ and module docstring)

**Step 1: Update module docstring (lines 14-26)**

Change from:
```python
Usage:
    from esper.simic.rewards import compute_shaped_reward, RewardConfig

    reward = compute_shaped_reward(
        action=ActionEnum.GERMINATE_CONV,  # Pass Enum member
        acc_delta=0.5,
        val_acc=65.0,
        seed_info=SeedInfo(...),
        epoch=10,
        max_epochs=25,
        action_enum=ActionEnum, # Required context
    )
```

To:
```python
Usage:
    from esper.simic.rewards import compute_contribution_reward, SeedInfo

    reward = compute_contribution_reward(
        action=ActionEnum.GERMINATE_CONV,
        seed_contribution=0.5,  # From counterfactual validation
        val_acc=65.0,
        seed_info=SeedInfo(...),
        epoch=10,
        max_epochs=25,
    )
```

**Step 2: Update __all__ list**

Remove these entries:
```python
    "RewardConfig",
    "compute_shaped_reward",  # Legacy: uses acc_delta (conflated signal)
```

The __all__ should now be:
```python
__all__ = [
    # Config classes
    "LossRewardConfig",
    "ContributionRewardConfig",
    # Seed info
    "SeedInfo",
    # Reward functions
    "compute_contribution_reward",
    "compute_loss_reward",
    # PBRS utilities
    "compute_potential",
    "compute_pbrs_bonus",
    "compute_pbrs_stage_bonus",
    "compute_seed_potential",
    # Intervention costs
    "get_intervention_cost",
    "INTERVENTION_COSTS_BY_NAME",
    # Stage constants and PBRS configuration
    "DEFAULT_GAMMA",
    "STAGE_POTENTIALS",
    "STAGE_TRAINING",
    "STAGE_BLENDING",
    "STAGE_FOSSILIZED",
    "STAGE_SHADOWING",
    "STAGE_PROBATIONARY",
]
```

**Step 3: Run full test suite**

Run: `pytest tests/ -k "reward" -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/simic/rewards.py && git commit -m "docs(rewards): update module docstring and exports

Updated usage example to show compute_contribution_reward.
Removed legacy exports from __all__."
```

---

## Task 9: Final Verification

**Files:**
- None (verification only)

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Verify no import errors in production code**

Run: `python -c "from esper.simic import *; from esper.simic.training import *; from esper.simic.vectorized import *; print('All imports OK')"`
Expected: `All imports OK`

**Step 3: Verify no references to removed code**

Run: `grep -r "compute_shaped_reward\|RewardConfig" src/ --include="*.py" | grep -v "__pycache__"`
Expected: No output (no remaining references)

Run: `grep -r "_DEFAULT_CONFIG\|_germinate_shaping\|_advance_shaping\|_cull_shaping\|_wait_shaping" src/ --include="*.py" | grep -v "__pycache__"`
Expected: No output (no remaining references)

**Step 4: Final commit (if any cleanup needed)**

```bash
git status
# If clean, no commit needed
```

---

## Summary

| Task | Lines Removed | Description |
|------|---------------|-------------|
| 1 | 102 | Delete test_simic_fossilize_shaping.py |
| 2 | 15 | Remove reward_configs strategy |
| 3 | 4 | Update simic/__init__.py exports |
| 4 | 0 | Update comment in episodes.py |
| 5 | 130 | Delete RewardConfig class |
| 6 | 226 | Delete helper functions |
| 7 | 163 | Delete compute_shaped_reward |
| 8 | 3 | Update __all__ and docstring |
| **Total** | **~643** | Legacy code removed |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| External code depends on RewardConfig | Grep confirmed no external production usage |
| Tests fail after removal | All legacy-dependent tests identified and marked for deletion first |
| Import cycles | Each task verifies imports work before committing |
| Documentation outdated | Module docstring updated in Task 8 |
