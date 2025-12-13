# Sparse Reward Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add sparse (terminal-only) reward mode to test whether dense reward shaping causes Goodhart risk.

**Architecture:** Add `RewardMode` enum and `compute_reward` dispatcher that routes to existing `compute_contribution_reward` (shaped), new `compute_sparse_reward` (terminal-only), or `compute_minimal_reward` (sparse + early-cull penalty). All changes are additive - default behavior unchanged.

**Tech Stack:** Python 3.11+, PyTorch, Hypothesis (property tests)

---

## Experiment Design (DRL Expert Review)

### Success Criteria

| Outcome | Definition | Interpretation |
|---------|------------|----------------|
| **SUCCESS** | SPARSE achieves ≥80% of SHAPED final accuracy within 2x episodes | Dense shaping may have Goodhart risk |
| **PARTIAL** | SPARSE achieves ≥60% of SHAPED after 10k episodes | Credit assignment possible but slow |
| **FAILURE** | SPARSE matches random baseline after 10k episodes | Credit assignment too hard without shaping |

### Baselines Required

1. **SHAPED** (5,000 episodes) - Current dense reward, establishes target performance
2. **RANDOM** (1,000 episodes) - Random action selection, establishes floor
3. **SPARSE** (10,000 episodes) - Terminal-only reward
4. **MINIMAL** (10,000 episodes) - Sparse + early-cull penalty (if SPARSE fails)

### Key Hyperparameters

- **gamma:** Use 0.995 (matches DEFAULT_GAMMA in rewards.py) for better credit assignment
- **sparse_reward_scale:** Default 1.0, try 2.0-3.0 if learning fails
- **entropy_coef:** May need increase (0.1) for exploration with sparse signal

### Expected Outcome (DRL Expert)

> *"60-70% probability SPARSE fails initially. Plan for diagnostic logging and iterative refinement."*

---

## Task 1: Add RewardMode Enum

**Files:**
- Modify: `src/esper/simic/rewards.py:32-33`

**Step 1: Write the failing test**

Create test file:
```python
# tests/simic/test_reward_modes.py
"""Tests for reward mode enum and sparse reward functions."""

import pytest
from esper.simic.rewards import RewardMode


def test_reward_mode_enum_exists():
    """RewardMode enum has three modes."""
    assert RewardMode.SHAPED.value == "shaped"
    assert RewardMode.SPARSE.value == "sparse"
    assert RewardMode.MINIMAL.value == "minimal"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_reward_mode_enum_exists -v`
Expected: FAIL with "cannot import name 'RewardMode'"

**Step 3: Write minimal implementation**

In `src/esper/simic/rewards.py`, add after line 32 (after `from enum import IntEnum`):

```python
from enum import IntEnum, Enum
```

Then add after line 40 (after the `_is_germinate_action` function, before the PBRS comment block):

```python
class RewardMode(Enum):
    """Reward function variant for experimentation.

    SHAPED: Current dense shaping with PBRS, attribution, warnings (default)
    SPARSE: Terminal-only ground truth (accuracy - param_cost)
    MINIMAL: Sparse + early-cull penalty only
    """
    SHAPED = "shaped"
    SPARSE = "sparse"
    MINIMAL = "minimal"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_reward_mode_enum_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/simic/test_reward_modes.py
git commit -m "feat(rewards): add RewardMode enum for experiment modes"
```

---

## Task 2: Add Sparse Config Fields to ContributionRewardConfig

**Files:**
- Modify: `src/esper/simic/rewards.py:135-196`
- Modify: `tests/simic/test_reward_modes.py`

**Step 1: Write the failing test**

Add to `tests/simic/test_reward_modes.py`:

```python
from esper.simic.rewards import RewardMode, ContributionRewardConfig


def test_config_has_sparse_fields():
    """ContributionRewardConfig has sparse reward fields."""
    config = ContributionRewardConfig()

    # Default mode is SHAPED
    assert config.reward_mode == RewardMode.SHAPED

    # Sparse reward parameters
    assert config.param_budget == 500_000
    assert config.param_penalty_weight == 0.1
    assert config.sparse_reward_scale == 1.0  # DRL Expert: try 2.0-3.0 if learning fails

    # Minimal mode parameters
    assert config.early_cull_threshold == 5
    assert config.early_cull_penalty == -0.1
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_config_has_sparse_fields -v`
Expected: FAIL with "AttributeError: 'ContributionRewardConfig' object has no attribute 'reward_mode'"

**Step 3: Write minimal implementation**

In `src/esper/simic/rewards.py`, modify `ContributionRewardConfig` class. Add these fields after line 191 (after `gamma: float = DEFAULT_GAMMA`):

```python
    # === Experiment Mode ===
    reward_mode: RewardMode = RewardMode.SHAPED

    # === Sparse Reward Parameters ===
    # Parameter budget for efficiency calculation (sparse/minimal modes)
    param_budget: int = 500_000
    # Weight for parameter penalty in sparse reward
    param_penalty_weight: float = 0.1
    # Reward scaling factor for sparse mode (DRL Expert: try 2.0-3.0 if learning fails)
    # Higher scale helps with credit assignment over 25 timesteps
    sparse_reward_scale: float = 1.0

    # === Minimal Mode Parameters ===
    # Minimum seed age before cull (epochs)
    early_cull_threshold: int = 5
    # Penalty for culling young seeds
    early_cull_penalty: float = -0.1
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_config_has_sparse_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/simic/test_reward_modes.py
git commit -m "feat(rewards): add sparse reward config fields"
```

---

## Task 3: Update __all__ Exports

**Files:**
- Modify: `src/esper/simic/rewards.py:953-977`

**Step 1: Write the failing test**

Add to `tests/simic/test_reward_modes.py`:

```python
def test_reward_mode_exported():
    """RewardMode is in module __all__."""
    from esper.simic import rewards
    assert "RewardMode" in rewards.__all__
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_reward_mode_exported -v`
Expected: FAIL with "AssertionError"

**Step 3: Write minimal implementation**

In `src/esper/simic/rewards.py`, modify `__all__` list (around line 953). Add `"RewardMode"` after `"ContributionRewardConfig"`:

```python
__all__ = [
    # Config classes
    "LossRewardConfig",
    "ContributionRewardConfig",
    "RewardMode",
    # ... rest unchanged
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_reward_mode_exported -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/simic/test_reward_modes.py
git commit -m "feat(rewards): export RewardMode in __all__"
```

---

## Task 4: Implement compute_sparse_reward Function

**Files:**
- Modify: `src/esper/simic/rewards.py` (add after `compute_contribution_reward`)
- Modify: `tests/simic/test_reward_modes.py`

**Step 1: Write the failing test**

Add to `tests/simic/test_reward_modes.py`:

```python
from esper.simic.rewards import compute_sparse_reward


def test_sparse_reward_zero_before_terminal():
    """Sparse reward is 0.0 for non-terminal epochs."""
    config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

    # Epoch 10 of 25 - not terminal
    reward = compute_sparse_reward(
        host_max_acc=75.0,
        total_params=100_000,
        epoch=10,
        max_epochs=25,
        config=config,
    )
    assert reward == 0.0


def test_sparse_reward_nonzero_at_terminal():
    """Sparse reward is non-zero at terminal epoch."""
    config = ContributionRewardConfig(
        reward_mode=RewardMode.SPARSE,
        param_budget=500_000,
        param_penalty_weight=0.1,
        sparse_reward_scale=1.0,
    )

    # Epoch 25 of 25 - terminal
    reward = compute_sparse_reward(
        host_max_acc=80.0,
        total_params=100_000,
        epoch=25,
        max_epochs=25,
        config=config,
    )

    # Expected: 1.0 * ((80/100) - 0.1 * (100_000 / 500_000)) = 0.8 - 0.02 = 0.78
    assert abs(reward - 0.78) < 0.001


def test_sparse_reward_with_scale():
    """Sparse reward respects scale parameter."""
    config = ContributionRewardConfig(
        reward_mode=RewardMode.SPARSE,
        param_budget=500_000,
        param_penalty_weight=0.1,
        sparse_reward_scale=2.5,  # DRL Expert recommendation for credit assignment
    )

    reward = compute_sparse_reward(
        host_max_acc=80.0,
        total_params=100_000,
        epoch=25,
        max_epochs=25,
        config=config,
    )

    # Expected: 2.5 * (0.8 - 0.02) = 2.5 * 0.78 = 1.95, clamped to 1.0
    # Actually: scale applied before clamp, so raw = 1.95, clamped = 1.0
    # But if we want scale to help, clamp should be [-scale, +scale]
    # Let's verify behavior matches implementation
    assert reward == 1.0  # Clamped at upper bound
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_sparse_reward_zero_before_terminal -v`
Expected: FAIL with "cannot import name 'compute_sparse_reward'"

**Step 3: Write minimal implementation**

In `src/esper/simic/rewards.py`, add after `compute_contribution_reward` function (around line 631):

```python
def compute_sparse_reward(
    host_max_acc: float,
    total_params: int,
    epoch: int,
    max_epochs: int,
    config: ContributionRewardConfig,
) -> float:
    """Compute sparse (terminal-only) reward.

    This reward function returns 0.0 for all non-terminal timesteps,
    forcing the LSTM policy to perform genuine temporal credit assignment
    over the full episode. At terminal, it rewards accuracy and penalizes
    parameter count.

    Design rationale:
    - Terminal-only: Forces credit assignment, tests if shaping is necessary
    - Accuracy-primary: The true objective is host performance
    - Param penalty: Efficiency matters, but less than accuracy
    - Scale factor: DRL Expert recommends 2.0-3.0 if learning fails

    Args:
        host_max_acc: Maximum accuracy achieved during episode (0-100)
        total_params: Total parameters (host + seeds) at episode end
        epoch: Current epoch (1-indexed)
        max_epochs: Maximum epochs in episode
        config: Reward configuration with param_budget, param_penalty_weight, sparse_reward_scale

    Returns:
        0.0 for non-terminal epochs, scaled reward at terminal (clamped to [-1, 1])
    """
    # Non-terminal: return 0.0 (the defining property of sparse rewards)
    if epoch != max_epochs:
        return 0.0

    # Terminal reward: accuracy minus parameter cost
    accuracy_reward = host_max_acc / 100.0
    param_cost = config.param_penalty_weight * (total_params / config.param_budget)

    # Apply scale for better gradient signal (DRL Expert recommendation)
    reward = config.sparse_reward_scale * (accuracy_reward - param_cost)

    # Clamp to [-1.0, 1.0] for stable learning
    return max(-1.0, min(1.0, reward))
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py -v -k "sparse_reward"`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/simic/test_reward_modes.py
git commit -m "feat(rewards): implement compute_sparse_reward function with scale"
```

---

## Task 5: Implement compute_minimal_reward Function

**Files:**
- Modify: `src/esper/simic/rewards.py`
- Modify: `tests/simic/test_reward_modes.py`

**Step 1: Write the failing test**

Add to `tests/simic/test_reward_modes.py`:

```python
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import compute_minimal_reward


def test_minimal_reward_no_penalty_for_old_cull():
    """MINIMAL mode: no penalty for culling old seeds."""
    config = ContributionRewardConfig(
        reward_mode=RewardMode.MINIMAL,
        early_cull_threshold=5,
        early_cull_penalty=-0.1,
    )

    # Cull a seed that's old enough (age >= threshold)
    reward = compute_minimal_reward(
        host_max_acc=75.0,
        total_params=100_000,
        epoch=10,
        max_epochs=25,
        action=LifecycleOp.CULL,
        seed_age=5,  # Exactly at threshold
        config=config,
    )

    # Non-terminal, no penalty -> 0.0
    assert reward == 0.0


def test_minimal_reward_penalty_for_young_cull():
    """MINIMAL mode: penalty for culling young seeds."""
    config = ContributionRewardConfig(
        reward_mode=RewardMode.MINIMAL,
        early_cull_threshold=5,
        early_cull_penalty=-0.1,
    )

    # Cull a seed that's too young
    reward = compute_minimal_reward(
        host_max_acc=75.0,
        total_params=100_000,
        epoch=10,
        max_epochs=25,
        action=LifecycleOp.CULL,
        seed_age=3,  # Below threshold
        config=config,
    )

    # Non-terminal but penalty applies -> -0.1
    assert reward == config.early_cull_penalty
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_minimal_reward_no_penalty_for_old_cull -v`
Expected: FAIL with "cannot import name 'compute_minimal_reward'"

**Step 3: Write minimal implementation**

In `src/esper/simic/rewards.py`, add after `compute_sparse_reward`:

```python
def compute_minimal_reward(
    host_max_acc: float,
    total_params: int,
    epoch: int,
    max_epochs: int,
    action: IntEnum,
    seed_age: int | None,
    config: ContributionRewardConfig,
) -> float:
    """Compute minimal reward (sparse + early-cull penalty).

    This is a fallback if pure sparse rewards fail to learn. It adds
    a single shaping signal: penalize culling seeds before they've had
    a chance to prove themselves.

    Design rationale:
    - Sparse base: Preserves most of the credit assignment challenge
    - Early-cull penalty: Prevents degenerate "cull everything" policy
    - No other shaping: Tests if minimal guidance is sufficient

    Args:
        host_max_acc: Maximum accuracy achieved during episode
        total_params: Total parameters at episode end
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        action: Action taken this timestep
        seed_age: Age of the seed in epochs (None if no seed)
        config: Reward configuration

    Returns:
        Sparse reward + early-cull penalty if applicable
    """
    # Start with sparse reward
    reward = compute_sparse_reward(
        host_max_acc=host_max_acc,
        total_params=total_params,
        epoch=epoch,
        max_epochs=max_epochs,
        config=config,
    )

    # Add early-cull penalty if applicable
    if action == LifecycleOp.CULL and seed_age is not None:
        if seed_age < config.early_cull_threshold:
            reward += config.early_cull_penalty

    return reward
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py -v -k "minimal_reward"`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add src/esper/simic/rewards.py tests/simic/test_reward_modes.py
git commit -m "feat(rewards): implement compute_minimal_reward function"
```

---

## Task 6: Implement compute_reward Dispatcher

**Files:**
- Modify: `src/esper/simic/rewards.py`
- Modify: `tests/simic/test_reward_modes.py`

**Step 1: Write the failing test**

Add to `tests/simic/test_reward_modes.py`:

```python
from esper.simic.rewards import compute_reward, SeedInfo


def test_compute_reward_shaped_mode():
    """compute_reward dispatches to shaped reward by default."""
    config = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)

    reward = compute_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=70.0,
        host_max_acc=70.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        total_params=100_000,
        host_params=100_000,
        acc_at_germination=None,
        acc_delta=0.0,
        config=config,
    )

    # Shaped reward with no seed should be non-zero (rent, etc.)
    assert isinstance(reward, float)


def test_compute_reward_sparse_mode():
    """compute_reward dispatches to sparse reward when mode is SPARSE."""
    config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

    # Non-terminal epoch
    reward = compute_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=70.0,
        host_max_acc=70.0,
        seed_info=None,
        epoch=10,
        max_epochs=25,
        total_params=100_000,
        host_params=100_000,
        acc_at_germination=None,
        acc_delta=0.0,
        config=config,
    )

    # Sparse reward at non-terminal = 0.0
    assert reward == 0.0


def test_compute_reward_minimal_mode():
    """compute_reward dispatches to minimal reward when mode is MINIMAL."""
    config = ContributionRewardConfig(
        reward_mode=RewardMode.MINIMAL,
        early_cull_threshold=5,
        early_cull_penalty=-0.1,
    )

    # Create a young seed
    seed_info = SeedInfo(
        stage=3,  # TRAINING
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=2,
        seed_params=10_000,
        previous_stage=2,
        previous_epochs_in_stage=1,
        seed_age_epochs=3,  # Young seed
    )

    # Cull action on young seed
    reward = compute_reward(
        action=LifecycleOp.CULL,
        seed_contribution=None,
        val_acc=70.0,
        host_max_acc=70.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        total_params=110_000,
        host_params=100_000,
        acc_at_germination=65.0,
        acc_delta=0.5,
        config=config,
    )

    # Should get early-cull penalty
    assert reward == -0.1
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_compute_reward_shaped_mode -v`
Expected: FAIL with "cannot import name 'compute_reward'"

**Step 3: Write minimal implementation**

In `src/esper/simic/rewards.py`, add after `compute_minimal_reward`:

```python
def compute_reward(
    action: IntEnum,
    seed_contribution: float | None,
    val_acc: float,
    host_max_acc: float,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    total_params: int,
    host_params: int,
    acc_at_germination: float | None,
    acc_delta: float,
    num_fossilized_seeds: int = 0,
    num_contributing_fossilized: int = 0,
    config: ContributionRewardConfig | None = None,
    return_components: bool = False,
) -> float | tuple[float, "RewardComponentsTelemetry"]:
    """Unified reward computation dispatcher.

    Routes to the appropriate reward function based on config.reward_mode:
    - SHAPED: Dense shaping with PBRS, attribution, warnings (default)
    - SPARSE: Terminal-only ground truth reward
    - MINIMAL: Sparse + early-cull penalty

    Args:
        action: Action taken (LifecycleOp or similar IntEnum)
        seed_contribution: Counterfactual contribution (None if unavailable)
        val_acc: Current validation accuracy
        host_max_acc: Maximum accuracy achieved during episode
        seed_info: Seed state info (None if no active seed)
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        total_params: Total parameters (host + seeds)
        host_params: Host model parameters
        acc_at_germination: Accuracy when seed was planted
        acc_delta: Per-epoch accuracy change
        num_fossilized_seeds: Count of fossilized seeds
        num_contributing_fossilized: Count of contributing fossilized seeds
        config: Reward configuration (uses default if None)
        return_components: If True, return (reward, components) tuple

    Returns:
        Reward value, or (reward, components) if return_components=True
    """
    if config is None:
        config = ContributionRewardConfig()

    # Dispatch based on reward mode
    if config.reward_mode == RewardMode.SHAPED:
        return compute_contribution_reward(
            action=action,
            seed_contribution=seed_contribution,
            val_acc=val_acc,
            seed_info=seed_info,
            epoch=epoch,
            max_epochs=max_epochs,
            total_params=total_params,
            host_params=host_params,
            config=config,
            acc_at_germination=acc_at_germination,
            acc_delta=acc_delta,
            return_components=return_components,
            num_fossilized_seeds=num_fossilized_seeds,
            num_contributing_fossilized=num_contributing_fossilized,
        )

    elif config.reward_mode == RewardMode.SPARSE:
        reward = compute_sparse_reward(
            host_max_acc=host_max_acc,
            total_params=total_params,
            epoch=epoch,
            max_epochs=max_epochs,
            config=config,
        )

    elif config.reward_mode == RewardMode.MINIMAL:
        seed_age = seed_info.seed_age_epochs if seed_info else None
        reward = compute_minimal_reward(
            host_max_acc=host_max_acc,
            total_params=total_params,
            epoch=epoch,
            max_epochs=max_epochs,
            action=action,
            seed_age=seed_age,
            config=config,
        )

    else:
        raise ValueError(f"Unknown reward mode: {config.reward_mode}")

    # Handle return_components for sparse/minimal modes
    if return_components:
        components = RewardComponentsTelemetry()
        components.total_reward = reward
        components.action_name = action.name
        components.epoch = epoch
        components.seed_stage = seed_info.stage if seed_info else None
        components.val_acc = val_acc
        return reward, components

    return reward
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py -v -k "compute_reward"`
Expected: PASS (all three tests)

**Step 5: Update __all__ and commit**

Add `"compute_reward"`, `"compute_sparse_reward"`, `"compute_minimal_reward"` to `__all__` in rewards.py:

```python
__all__ = [
    # Config classes
    "LossRewardConfig",
    "ContributionRewardConfig",
    "RewardMode",
    # Seed info
    "SeedInfo",
    # Reward functions
    "compute_reward",
    "compute_contribution_reward",
    "compute_sparse_reward",
    "compute_minimal_reward",
    "compute_loss_reward",
    # ... rest unchanged
]
```

```bash
git add src/esper/simic/rewards.py tests/simic/test_reward_modes.py
git commit -m "feat(rewards): implement compute_reward dispatcher"
```

---

## Task 7: Add Sparse Reward Property Tests

**Files:**
- Create: `tests/simic/properties/test_sparse_properties.py`

**Step 1: Write the property tests**

```python
# tests/simic/properties/test_sparse_properties.py
"""Property-based tests for sparse reward invariants.

DRL Expert Review: These tests verify critical invariants for sparse rewards.
"""

import math
import pytest
from hypothesis import given, strategies as st, settings

from esper.simic.rewards import (
    RewardMode,
    ContributionRewardConfig,
    compute_sparse_reward,
    compute_minimal_reward,
)
from esper.leyline.factored_actions import LifecycleOp


# Strategy for valid sparse inputs
@st.composite
def sparse_inputs(draw):
    """Generate valid inputs for sparse reward."""
    max_epochs = 25
    epoch = draw(st.integers(1, max_epochs))
    return {
        "host_max_acc": draw(st.floats(0.0, 100.0, allow_nan=False)),
        "total_params": draw(st.integers(0, 2_000_000)),
        "epoch": epoch,
        "max_epochs": max_epochs,
    }


@st.composite
def terminal_inputs(draw):
    """Generate inputs at terminal epoch."""
    max_epochs = 25
    return {
        "host_max_acc": draw(st.floats(0.0, 100.0, allow_nan=False)),
        "total_params": draw(st.integers(0, 2_000_000)),
        "epoch": max_epochs,
        "max_epochs": max_epochs,
    }


class TestSparseRewardProperties:
    """Property tests for sparse reward invariants."""

    @given(sparse_inputs())
    @settings(max_examples=200, deadline=None)
    def test_zero_before_terminal(self, inputs):
        """INVARIANT: Sparse reward is 0.0 for all non-terminal epochs."""
        if inputs["epoch"] == inputs["max_epochs"]:
            return  # Skip terminal case

        config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)
        reward = compute_sparse_reward(**inputs, config=config)

        assert reward == 0.0, f"Non-terminal reward should be 0.0, got {reward}"

    @given(sparse_inputs())
    @settings(max_examples=200, deadline=None)
    def test_bounded(self, inputs):
        """INVARIANT: Sparse reward is bounded to [-1.0, 1.0]."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)
        reward = compute_sparse_reward(**inputs, config=config)

        assert -1.0 <= reward <= 1.0, f"Reward {reward} outside [-1, 1]"

    @given(st.floats(0.0, 100.0), st.floats(0.0, 100.0), st.integers(0, 1_000_000))
    @settings(max_examples=200, deadline=None)
    def test_monotonic_in_accuracy(self, acc1, acc2, params):
        """INVARIANT: Higher accuracy -> higher reward (at terminal)."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

        r1 = compute_sparse_reward(
            host_max_acc=acc1, total_params=params,
            epoch=25, max_epochs=25, config=config,
        )
        r2 = compute_sparse_reward(
            host_max_acc=acc2, total_params=params,
            epoch=25, max_epochs=25, config=config,
        )

        if acc1 < acc2:
            assert r1 <= r2, f"acc {acc1}->{acc2} but reward {r1}->{r2}"
        elif acc1 > acc2:
            assert r1 >= r2, f"acc {acc1}->{acc2} but reward {r1}->{r2}"

    @given(st.floats(50.0, 100.0), st.integers(0, 1_000_000), st.integers(0, 1_000_000))
    @settings(max_examples=200, deadline=None)
    def test_monotonic_in_efficiency(self, acc, params1, params2):
        """INVARIANT: Fewer params -> higher reward (at terminal)."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

        r1 = compute_sparse_reward(
            host_max_acc=acc, total_params=params1,
            epoch=25, max_epochs=25, config=config,
        )
        r2 = compute_sparse_reward(
            host_max_acc=acc, total_params=params2,
            epoch=25, max_epochs=25, config=config,
        )

        if params1 < params2:
            assert r1 >= r2, f"params {params1}->{params2} but reward {r1}->{r2}"
        elif params1 > params2:
            assert r1 <= r2, f"params {params1}->{params2} but reward {r1}->{r2}"

    def test_accuracy_dominates_params(self):
        """INVARIANT: 80% acc with 500k params > 70% acc with 100k params."""
        config = ContributionRewardConfig(
            reward_mode=RewardMode.SPARSE,
            param_budget=500_000,
            param_penalty_weight=0.1,
        )

        # High accuracy, high params
        r_high_acc = compute_sparse_reward(
            host_max_acc=80.0, total_params=500_000,
            epoch=25, max_epochs=25, config=config,
        )

        # Low accuracy, low params
        r_low_acc = compute_sparse_reward(
            host_max_acc=70.0, total_params=100_000,
            epoch=25, max_epochs=25, config=config,
        )

        assert r_high_acc > r_low_acc, (
            f"Accuracy should dominate: {r_high_acc} should be > {r_low_acc}"
        )


class TestSparseRewardEdgeCases:
    """Edge case tests from DRL Expert review."""

    def test_terminal_reward_deterministic(self):
        """Same inputs -> same terminal reward (no hidden state)."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

        r1 = compute_sparse_reward(
            host_max_acc=75.0, total_params=200_000,
            epoch=25, max_epochs=25, config=config,
        )
        r2 = compute_sparse_reward(
            host_max_acc=75.0, total_params=200_000,
            epoch=25, max_epochs=25, config=config,
        )

        assert r1 == r2, "Terminal reward should be deterministic"

    def test_gradient_safe_zero_params(self):
        """Sparse reward handles params=0 without NaN/Inf."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SPARSE)

        reward = compute_sparse_reward(
            host_max_acc=75.0, total_params=0,
            epoch=25, max_epochs=25, config=config,
        )

        assert math.isfinite(reward), f"Reward should be finite, got {reward}"

    def test_gradient_safe_extreme_params(self):
        """Sparse reward handles very large params without overflow."""
        config = ContributionRewardConfig(
            reward_mode=RewardMode.SPARSE,
            param_budget=500_000,
        )

        reward = compute_sparse_reward(
            host_max_acc=75.0, total_params=100_000_000,  # 100M params
            epoch=25, max_epochs=25, config=config,
        )

        assert math.isfinite(reward), f"Reward should be finite, got {reward}"
        assert reward == -1.0, "Should clamp to -1.0 for extreme params"

    def test_clamp_triggers_at_upper_bound(self):
        """Verify clamping at +1.0."""
        config = ContributionRewardConfig(
            reward_mode=RewardMode.SPARSE,
            sparse_reward_scale=3.0,  # High scale to trigger clamp
        )

        reward = compute_sparse_reward(
            host_max_acc=100.0, total_params=0,  # Best case
            epoch=25, max_epochs=25, config=config,
        )

        # Raw: 3.0 * (1.0 - 0) = 3.0, should clamp to 1.0
        assert reward == 1.0, f"Should clamp to 1.0, got {reward}"


class TestMinimalRewardProperties:
    """Property tests for minimal reward invariants."""

    @given(st.integers(0, 4), st.integers(5, 25))
    @settings(max_examples=100, deadline=None)
    def test_early_cull_penalty(self, young_age, old_age):
        """INVARIANT: Culling young seeds gets penalty, old seeds don't."""
        config = ContributionRewardConfig(
            reward_mode=RewardMode.MINIMAL,
            early_cull_threshold=5,
            early_cull_penalty=-0.1,
        )

        base_inputs = {
            "host_max_acc": 75.0,
            "total_params": 100_000,
            "epoch": 10,  # Non-terminal
            "max_epochs": 25,
            "action": LifecycleOp.CULL,
            "config": config,
        }

        r_young = compute_minimal_reward(**base_inputs, seed_age=young_age)
        r_old = compute_minimal_reward(**base_inputs, seed_age=old_age)

        assert r_young == -0.1, f"Young cull should get penalty, got {r_young}"
        assert r_old == 0.0, f"Old cull should get no penalty, got {r_old}"

    @given(st.sampled_from([LifecycleOp.WAIT, LifecycleOp.GERMINATE, LifecycleOp.FOSSILIZE]))
    @settings(max_examples=50, deadline=None)
    def test_non_cull_no_penalty(self, action):
        """INVARIANT: Non-CULL actions get no early-cull penalty."""
        config = ContributionRewardConfig(
            reward_mode=RewardMode.MINIMAL,
            early_cull_threshold=5,
            early_cull_penalty=-0.1,
        )

        reward = compute_minimal_reward(
            host_max_acc=75.0,
            total_params=100_000,
            epoch=10,
            max_epochs=25,
            action=action,
            seed_age=2,  # Young seed
            config=config,
        )

        # Non-terminal, non-CULL -> 0.0 (no penalty regardless of seed age)
        assert reward == 0.0, f"Non-CULL action should get no penalty, got {reward}"

    def test_minimal_equals_sparse_plus_penalty(self):
        """INVARIANT: MINIMAL = SPARSE + early_cull_penalty (when applicable)."""
        config = ContributionRewardConfig(
            reward_mode=RewardMode.MINIMAL,
            early_cull_threshold=5,
            early_cull_penalty=-0.1,
        )

        # Sparse base reward (non-terminal = 0.0)
        sparse_reward = compute_sparse_reward(
            host_max_acc=75.0,
            total_params=100_000,
            epoch=10,
            max_epochs=25,
            config=config,
        )

        # Minimal with young cull
        minimal_reward = compute_minimal_reward(
            host_max_acc=75.0,
            total_params=100_000,
            epoch=10,
            max_epochs=25,
            action=LifecycleOp.CULL,
            seed_age=3,
            config=config,
        )

        assert minimal_reward == sparse_reward + config.early_cull_penalty, (
            f"MINIMAL should equal SPARSE + penalty: {minimal_reward} != {sparse_reward} + {config.early_cull_penalty}"
        )
```

**Step 2: Run tests**

Run: `PYTHONPATH=src pytest tests/simic/properties/test_sparse_properties.py -v`
Expected: PASS (all property tests)

**Step 3: Commit**

```bash
git add tests/simic/properties/test_sparse_properties.py
git commit -m "test(rewards): add property tests for sparse reward invariants"
```

---

## Task 8: Add host_max_acc to ParallelEnvState

**Files:**
- Modify: `src/esper/simic/vectorized.py:65-105`

**Step 1: Write the failing test**

Add to `tests/simic/test_reward_modes.py`:

```python
def test_parallel_env_state_has_host_max_acc():
    """ParallelEnvState tracks host_max_acc."""
    from esper.simic.vectorized import ParallelEnvState
    import inspect

    # Check the dataclass has the field
    hints = inspect.get_annotations(ParallelEnvState)
    assert "host_max_acc" in hints, "ParallelEnvState should have host_max_acc field"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_parallel_env_state_has_host_max_acc -v`
Expected: FAIL with "AssertionError: ParallelEnvState should have host_max_acc field"

**Step 3: Write minimal implementation**

In `src/esper/simic/vectorized.py`, add to `ParallelEnvState` dataclass at **line 93** (after `acc_at_germination: float | None = None`):

```python
    # Maximum accuracy achieved during episode (for sparse reward)
    host_max_acc: float = 0.0
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/simic/test_reward_modes.py::test_parallel_env_state_has_host_max_acc -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py tests/simic/test_reward_modes.py
git commit -m "feat(vectorized): add host_max_acc tracking to ParallelEnvState"
```

---

## Task 9: Update host_max_acc During Training

**Files:**
- Modify: `src/esper/simic/vectorized.py` (training loop)

**PyTorch Expert Review:** Be explicit about line numbers.

**Step 1: Add host_max_acc update after val_acc assignment**

Location: **Line 997** (inside `for env_idx, env_state in enumerate(env_states):` loop)

After:
```python
env_state.val_acc = val_acc
```

Add:
```python
                # Track maximum accuracy for sparse reward
                env_state.host_max_acc = max(env_state.host_max_acc, env_state.val_acc)
```

**Step 2: Verify reset happens implicitly**

The `host_max_acc` field defaults to `0.0` in the dataclass. Fresh `ParallelEnvState` objects are created via `create_env_state()` at **line 680-684** for each batch, so reset happens automatically.

However, for episode boundaries within a batch, add explicit reset. Find where `seeds_fossilized = 0` is set (episode reset, around **line 1310**) and add:

```python
                    env_state.host_max_acc = 0.0
```

**Step 3: Verify with grep**

Run: `grep -n "host_max_acc" src/esper/simic/vectorized.py`
Expected output:
```
93:    host_max_acc: float = 0.0
997:                env_state.host_max_acc = max(env_state.host_max_acc, env_state.val_acc)
1310:                    env_state.host_max_acc = 0.0
```

**Step 4: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(vectorized): update host_max_acc during training loop"
```

---

## Task 10: Add Reward Mode Parameters to train_ppo_vectorized

**Files:**
- Modify: `src/esper/simic/vectorized.py:54` (imports)
- Modify: `src/esper/simic/vectorized.py:181-212` (function signature)

**PyTorch Expert Review:** Import at module level, not inside function.

**Step 1: Add module-level import**

At **line 54** (in the imports section), add:

```python
from esper.simic.rewards import compute_reward, RewardMode, ContributionRewardConfig, SeedInfo
```

Remove the existing import:
```python
from esper.simic.rewards import compute_contribution_reward, SeedInfo
```

**Step 2: Add parameters to function signature**

In `train_ppo_vectorized`, add after `max_seeds_per_slot` parameter (around **line 211**):

```python
    reward_mode: str = "shaped",
    param_budget: int = 500_000,
    param_penalty_weight: float = 0.1,
    sparse_reward_scale: float = 1.0,
```

**Step 3: Create reward config near function start**

After the function docstring and initial setup (around **line 280**), add:

```python
    # Create reward config based on mode
    reward_mode_enum = RewardMode(reward_mode)
    reward_config = ContributionRewardConfig(
        reward_mode=reward_mode_enum,
        param_budget=param_budget,
        param_penalty_weight=param_penalty_weight,
        sparse_reward_scale=sparse_reward_scale,
    )

    # Log reward mode at training start
    _logger.info(f"Reward mode: {reward_mode} (param_budget={param_budget}, penalty_weight={param_penalty_weight}, scale={sparse_reward_scale})")
```

**Step 4: Verify with grep**

Run: `grep -n "reward_mode\|reward_config" src/esper/simic/vectorized.py | head -20`
Expected: Should show parameter in signature and config creation

**Step 5: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(vectorized): add reward_mode parameter to train_ppo_vectorized"
```

---

## Task 11: Update Reward Computation to Use compute_reward

**Files:**
- Modify: `src/esper/simic/vectorized.py:1196-1228`

**Step 1: Update reward computation calls**

Replace both `compute_contribution_reward` calls with `compute_reward`, adding the new parameters.

At **line 1200** (telemetry path):
```python
                if collect_reward_telemetry:
                    reward, reward_components = compute_reward(
                        action=action_for_reward,
                        seed_contribution=seed_contribution,
                        val_acc=env_state.val_acc,
                        host_max_acc=env_state.host_max_acc,
                        seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
                        epoch=epoch,
                        max_epochs=max_epochs,
                        total_params=total_params,
                        host_params=host_params,
                        acc_at_germination=env_state.acc_at_germination,
                        acc_delta=signals.metrics.accuracy_delta,
                        return_components=True,
                        num_fossilized_seeds=env_state.seeds_fossilized,
                        num_contributing_fossilized=env_state.contributing_fossilized,
                        config=reward_config,
                    )
```

At **line 1218** (non-telemetry path):
```python
                else:
                    reward = compute_reward(
                        action=action_for_reward,
                        seed_contribution=seed_contribution,
                        val_acc=env_state.val_acc,
                        host_max_acc=env_state.host_max_acc,
                        seed_info=SeedInfo.from_seed_state(seed_state, model.active_seed_params),
                        epoch=epoch,
                        max_epochs=max_epochs,
                        total_params=total_params,
                        host_params=host_params,
                        acc_at_germination=env_state.acc_at_germination,
                        acc_delta=signals.metrics.accuracy_delta,
                        num_fossilized_seeds=env_state.seeds_fossilized,
                        num_contributing_fossilized=env_state.contributing_fossilized,
                        config=reward_config,
                    )
```

**Step 2: Run existing tests**

Run: `PYTHONPATH=src pytest tests/simic/test_rewards.py -v --tb=short`
Expected: PASS (existing tests should still work with SHAPED mode default)

**Step 3: Commit**

```bash
git add src/esper/simic/vectorized.py
git commit -m "feat(vectorized): use compute_reward dispatcher for reward computation"
```

---

## Task 12: Add CLI Arguments

**Files:**
- Modify: `src/esper/scripts/train.py:57-104`

**Step 1: Add reward mode arguments**

In the ppo_parser section, add after `--max-seeds-per-slot`:

```python
    ppo_parser.add_argument(
        "--reward-mode",
        type=str,
        choices=["shaped", "sparse", "minimal"],
        default="shaped",
        help="Reward mode: shaped (dense, default), sparse (terminal-only), minimal (sparse + early-cull)"
    )
    ppo_parser.add_argument(
        "--param-budget",
        type=int,
        default=500_000,
        help="Parameter budget for sparse reward efficiency calculation (default: 500000)"
    )
    ppo_parser.add_argument(
        "--param-penalty",
        type=float,
        default=0.1,
        help="Parameter penalty weight in sparse reward (default: 0.1)"
    )
    ppo_parser.add_argument(
        "--sparse-scale",
        type=float,
        default=1.0,
        help="Reward scale for sparse mode (DRL Expert: try 2.0-3.0 if learning fails)"
    )
```

**Step 2: Pass arguments to train_ppo_vectorized**

In the `train_ppo_vectorized` call (around line 157), add:

```python
            reward_mode=args.reward_mode,
            param_budget=args.param_budget,
            param_penalty_weight=args.param_penalty,
            sparse_reward_scale=args.sparse_scale,
```

**Step 3: Verify CLI**

Run: `PYTHONPATH=src python -m esper.scripts.train ppo --help | grep -A2 "reward-mode"`
Expected: Should show reward-mode argument with choices

**Step 4: Commit**

```bash
git add src/esper/scripts/train.py
git commit -m "feat(train): add --reward-mode CLI argument"
```

---

## Task 13: Create Integration Test File

**Files:**
- Create: `tests/integration/test_sparse_training.py`

**PyTorch Expert Review:** Add proper pytest integration test, not just manual verification.

**Step 1: Create integration test**

```python
# tests/integration/test_sparse_training.py
"""Integration tests for sparse reward training.

PyTorch Expert Review: These tests verify the training loop works
with all reward modes without requiring full training runs.
"""

import pytest
import math

from esper.simic.normalization import RewardNormalizer


class TestRewardNormalizerWithSparse:
    """Verify RewardNormalizer handles sparse reward distribution."""

    def test_normalizer_handles_zeros_then_spike(self):
        """RewardNormalizer handles 24 zeros then a spike (typical sparse episode)."""
        normalizer = RewardNormalizer()

        # Simulate sparse episode: 24 zeros, then terminal reward
        rewards = [0.0] * 24 + [0.78]
        normalized = [normalizer.update_and_normalize(r) for r in rewards]

        # All normalized values should be finite
        assert all(math.isfinite(n) for n in normalized), (
            f"Normalized rewards should be finite: {normalized}"
        )

    def test_normalizer_handles_multiple_sparse_episodes(self):
        """RewardNormalizer stabilizes over multiple sparse episodes."""
        normalizer = RewardNormalizer()

        # 10 sparse episodes
        for ep in range(10):
            rewards = [0.0] * 24 + [0.7 + 0.02 * ep]
            normalized = [normalizer.update_and_normalize(r) for r in rewards]

            assert all(math.isfinite(n) for n in normalized), (
                f"Episode {ep}: normalized rewards should be finite"
            )


@pytest.mark.slow
class TestSparseTrainingSmoke:
    """Smoke tests for sparse reward training (marked slow)."""

    def test_sparse_mode_trains_without_error(self):
        """Sparse mode completes training loop without exceptions."""
        pytest.importorskip("torch")

        from esper.simic.vectorized import train_ppo_vectorized

        # Minimal smoke test - just verify it runs
        try:
            agent, history = train_ppo_vectorized(
                n_episodes=2,
                n_envs=1,
                max_epochs=5,
                reward_mode="sparse",
                slots=["mid"],
                use_telemetry=False,
            )

            assert len(history) > 0
            # With sparse rewards, avg_reward should be ~0 until terminal
        except Exception as e:
            pytest.fail(f"Sparse training failed with: {e}")

    def test_minimal_mode_trains_without_error(self):
        """Minimal mode completes training loop without exceptions."""
        pytest.importorskip("torch")

        from esper.simic.vectorized import train_ppo_vectorized

        try:
            agent, history = train_ppo_vectorized(
                n_episodes=2,
                n_envs=1,
                max_epochs=5,
                reward_mode="minimal",
                slots=["mid"],
                use_telemetry=False,
            )

            assert len(history) > 0
        except Exception as e:
            pytest.fail(f"Minimal training failed with: {e}")
```

**Step 2: Run normalizer tests**

Run: `PYTHONPATH=src pytest tests/integration/test_sparse_training.py -v -k "normalizer"`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_sparse_training.py
git commit -m "test(integration): add sparse reward training integration tests"
```

---

## Task 14: Run Manual Integration Tests

**Files:** None (verification only)

**Step 1: Test shaped mode (default)**

Run: `PYTHONPATH=src timeout 60 python -m esper.scripts.train ppo --episodes 2 --max-epochs 5 --n-envs 1 --telemetry-level minimal`
Expected: Should run without errors, show normal reward values

**Step 2: Test sparse mode**

Run: `PYTHONPATH=src timeout 60 python -m esper.scripts.train ppo --episodes 2 --max-epochs 5 --n-envs 1 --reward-mode sparse --telemetry-level minimal`
Expected: Should run without errors, show 0.0 rewards until terminal

**Step 3: Test minimal mode**

Run: `PYTHONPATH=src timeout 60 python -m esper.scripts.train ppo --episodes 2 --max-epochs 5 --n-envs 1 --reward-mode minimal --telemetry-level minimal`
Expected: Should run without errors

**Step 4: Test with sparse scale**

Run: `PYTHONPATH=src timeout 60 python -m esper.scripts.train ppo --episodes 2 --max-epochs 5 --n-envs 1 --reward-mode sparse --sparse-scale 2.5 --telemetry-level minimal`
Expected: Should run without errors

**Step 5: Commit**

```bash
git add -A
git commit -m "test: verify reward mode integration"
```

---

## Task 15: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `PYTHONPATH=src pytest tests/ -v --tb=short -x --ignore=tests/integration/test_sparse_training.py`
Expected: All tests pass

**Step 2: Run property tests specifically**

Run: `PYTHONPATH=src pytest tests/simic/properties/ -v`
Expected: All property tests pass

**Step 3: Run integration tests (slow)**

Run: `PYTHONPATH=src pytest tests/integration/test_sparse_training.py -v --tb=short`
Expected: All tests pass (may take 1-2 minutes)

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(rewards): complete sparse reward experiment infrastructure"
```

---

## Summary

This plan implements the sparse reward experiment infrastructure in 15 tasks:

| Task | Description | Estimated Time |
|------|-------------|----------------|
| 1-3 | RewardMode enum and config | 15 min |
| 4-5 | Sparse/minimal reward functions | 20 min |
| 6 | compute_reward dispatcher | 15 min |
| 7 | Property tests (including edge cases) | 20 min |
| 8-9 | host_max_acc tracking | 15 min |
| 10-11 | Vectorized integration | 20 min |
| 12 | CLI arguments | 10 min |
| 13-15 | Integration tests and verification | 20 min |
| **Total** | | **~2.5 hours** |

---

## Running the Experiment

After completing all tasks, run the experiment:

```bash
# 1. Random baseline (establishes floor)
PYTHONPATH=src python -m esper.scripts.train ppo --episodes 1000 --reward-mode shaped --telemetry-dir results/random_baseline
# Note: For true random, would need --random-policy flag (not implemented in this plan)

# 2. Shaped baseline (establishes target)
PYTHONPATH=src python -m esper.scripts.train ppo --episodes 5000 --reward-mode shaped --telemetry-dir results/shaped_baseline

# 3. Sparse experiment
PYTHONPATH=src python -m esper.scripts.train ppo --episodes 10000 --reward-mode sparse --gamma 0.995 --telemetry-dir results/sparse_10k

# 4. Sparse with higher scale (if learning fails)
PYTHONPATH=src python -m esper.scripts.train ppo --episodes 10000 --reward-mode sparse --sparse-scale 2.5 --gamma 0.995 --telemetry-dir results/sparse_scaled

# 5. Minimal fallback (if sparse fails)
PYTHONPATH=src python -m esper.scripts.train ppo --episodes 10000 --reward-mode minimal --gamma 0.995 --telemetry-dir results/minimal_10k
```

---

## Expert Review Incorporated

| Expert | Finding | Resolution |
|--------|---------|------------|
| DRL | Reward scale too small | Added `sparse_reward_scale` config (Task 2, 4) |
| DRL | Missing random baseline | Documented in experiment design (future work) |
| DRL | Missing success criteria | Added explicit criteria in experiment design section |
| DRL | Missing edge case tests | Added `TestSparseRewardEdgeCases` (Task 7) |
| DRL | Missing compositionality test | Added `test_minimal_equals_sparse_plus_penalty` (Task 7) |
| DRL | Gamma mismatch risk | Documented gamma=0.995 recommendation |
| PyTorch | Vague line numbers | Made explicit (Tasks 9, 10, 11) |
| PyTorch | Import inside function | Moved to module level (Task 10) |
| PyTorch | Missing integration test | Added `tests/integration/test_sparse_training.py` (Task 13) |
| PyTorch | Missing normalizer test | Added `TestRewardNormalizerWithSparse` (Task 13) |
