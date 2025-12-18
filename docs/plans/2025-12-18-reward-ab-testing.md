# Reward A/B Testing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A/B test simplified (3-component) reward vs current (7-component) shaped reward with split environments

**Architecture:** Add `RewardMode.SIMPLIFIED` with only PBRS + intervention cost + terminal bonus. Enable per-environment reward mode selection so 4 envs run SIMPLIFIED and 4 run SHAPED simultaneously, sharing the same policy network for fair comparison.

**Tech Stack:** Python 3.11+, PyTorch, existing Simic/Leyline infrastructure

**DRL Expert Recommendation:** The current 7-component reward creates conflicting gradients and an unlearnable landscape. The simplified 3-component reward preserves PBRS guarantees while letting the LSTM do temporal credit assignment.

---

## Task 1: Add RewardMode.SIMPLIFIED Enum Value

**Files:**
- Modify: `src/esper/simic/rewards/rewards.py:44-54`

**Step 1: Write the failing test**

Create test file:
```python
# tests/simic/test_reward_simplified.py
"""Tests for SIMPLIFIED reward mode."""
import pytest
from esper.simic.rewards import RewardMode


def test_simplified_mode_exists():
    """RewardMode.SIMPLIFIED should be a valid enum member."""
    assert hasattr(RewardMode, "SIMPLIFIED")
    assert RewardMode.SIMPLIFIED.value == "simplified"


def test_simplified_mode_string_conversion():
    """SIMPLIFIED mode should round-trip through string."""
    mode = RewardMode("simplified")
    assert mode == RewardMode.SIMPLIFIED
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_simplified.py::test_simplified_mode_exists -v`
Expected: FAIL with `AttributeError: SIMPLIFIED`

**Step 3: Add SIMPLIFIED to RewardMode enum**

In `src/esper/simic/rewards/rewards.py`, modify the RewardMode enum:

```python
class RewardMode(Enum):
    """Reward function variant for experimentation.

    SHAPED: Current dense shaping with PBRS, attribution, warnings (default)
    SPARSE: Terminal-only ground truth (accuracy - param_cost)
    MINIMAL: Sparse + early-cull penalty only
    SIMPLIFIED: DRL Expert recommended - PBRS + intervention cost + terminal only
    """
    SHAPED = "shaped"
    SPARSE = "sparse"
    MINIMAL = "minimal"
    SIMPLIFIED = "simplified"
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_simplified.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/rewards.py tests/simic/test_reward_simplified.py
git commit -m "feat(simic): add RewardMode.SIMPLIFIED enum value"
```

---

## Task 2: Implement compute_simplified_reward Function

**Files:**
- Modify: `src/esper/simic/rewards/rewards.py` (add function after `compute_minimal_reward`)
- Test: `tests/simic/test_reward_simplified.py`

**Step 1: Write the failing test**

Append to `tests/simic/test_reward_simplified.py`:

```python
from esper.simic.rewards import (
    compute_simplified_reward,
    ContributionRewardConfig,
    SeedInfo,
    STAGE_TRAINING,
    STAGE_PROBATIONARY,
)
from esper.leyline.factored_actions import LifecycleOp


class TestComputeSimplifiedReward:
    """Tests for the simplified 3-component reward."""

    def test_non_terminal_returns_pbrs_plus_cost(self):
        """Non-terminal steps: PBRS + intervention cost only."""
        config = ContributionRewardConfig()
        seed_info = SeedInfo(
            stage=STAGE_TRAINING,
            improvement_since_stage_start=0.5,
            total_improvement=1.0,
            epochs_in_stage=3,
            seed_params=1000,
            previous_stage=STAGE_TRAINING,
            previous_epochs_in_stage=2,
            seed_age_epochs=5,
        )

        # WAIT action: only PBRS, no intervention cost
        reward_wait = compute_simplified_reward(
            action=LifecycleOp.WAIT,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            val_acc=65.0,
            num_contributing_fossilized=0,
            config=config,
        )

        # GERMINATE action: PBRS + intervention cost
        reward_germinate = compute_simplified_reward(
            action=LifecycleOp.GERMINATE,
            seed_info=None,  # No seed when germinating
            epoch=10,
            max_epochs=25,
            val_acc=65.0,
            num_contributing_fossilized=0,
            config=config,
        )

        # Germinate should have small negative cost
        assert reward_germinate < 0.0  # Intervention cost

    def test_terminal_includes_accuracy_and_fossilize_bonus(self):
        """Terminal step: PBRS + cost + accuracy + fossilize bonus."""
        config = ContributionRewardConfig()

        # Terminal with 2 contributing fossilized seeds
        reward = compute_simplified_reward(
            action=LifecycleOp.WAIT,
            seed_info=None,
            epoch=25,
            max_epochs=25,
            val_acc=75.0,
            num_contributing_fossilized=2,
            config=config,
        )

        # Should have: accuracy bonus (75/100 * 3 = 2.25) + fossilize bonus (2 * 2 = 4)
        # Total terminal ~ 6.25
        assert reward > 5.0
        assert reward < 8.0

    def test_no_attribution_component(self):
        """SIMPLIFIED should NOT include bounded_attribution."""
        config = ContributionRewardConfig()
        seed_info = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=5.0,  # High improvement
            total_improvement=10.0,
            epochs_in_stage=3,
            seed_params=1000,
            previous_stage=STAGE_PROBATIONARY,
            previous_epochs_in_stage=2,
            seed_age_epochs=15,
        )

        # With SHAPED, high improvement would give large attribution reward
        # With SIMPLIFIED, there should be NO attribution component
        reward = compute_simplified_reward(
            action=LifecycleOp.WAIT,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            val_acc=70.0,
            num_contributing_fossilized=0,
            config=config,
        )

        # Reward should be small (just PBRS epoch progress)
        # Not inflated by attribution
        assert abs(reward) < 1.0

    def test_no_warning_components(self):
        """SIMPLIFIED should NOT include blending_warning or probation_warning."""
        config = ContributionRewardConfig()
        seed_info = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=-5.0,  # Negative (would trigger warnings)
            total_improvement=-3.0,
            epochs_in_stage=5,  # Long time (would trigger probation_warning)
            seed_params=1000,
            previous_stage=STAGE_PROBATIONARY,
            previous_epochs_in_stage=4,
            seed_age_epochs=20,
        )

        # With SHAPED, this would trigger severe probation_warning (-9.0 or worse)
        # With SIMPLIFIED, no warning penalties
        reward = compute_simplified_reward(
            action=LifecycleOp.WAIT,
            seed_info=seed_info,
            epoch=20,
            max_epochs=25,
            val_acc=60.0,
            num_contributing_fossilized=0,
            config=config,
        )

        # Should NOT have the -9.0 probation_warning
        assert reward > -2.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_simplified.py::TestComputeSimplifiedReward -v`
Expected: FAIL with `ImportError: cannot import name 'compute_simplified_reward'`

**Step 3: Implement compute_simplified_reward**

Add after `compute_minimal_reward` (around line 837) in `src/esper/simic/rewards/rewards.py`:

```python
def compute_simplified_reward(
    action: LifecycleOp,
    seed_info: SeedInfo | None,
    epoch: int,
    max_epochs: int,
    val_acc: float,
    num_contributing_fossilized: int,
    config: ContributionRewardConfig | None = None,
) -> float:
    """Compute simplified 3-component reward (DRL Expert recommended).

    This reward function addresses the "unlearnable landscape" problem by
    removing conflicting components. Only three signals remain:

    1. PBRS stage progression (preserves optimal policy per Ng et al., 1999)
    2. Uniform intervention cost (small friction on non-WAIT actions)
    3. Terminal bonus (accuracy + fossilize count, scaled for 25-step credit)

    Removed vs SHAPED:
    - bounded_attribution (replace with terminal accuracy)
    - blending_warning (let terminal handle bad seeds)
    - probation_warning (let PBRS + terminal handle pacing)
    - ratio_penalty / attribution_discount (address via environment, not reward)
    - compute_rent (simplify - not critical for learning)

    Args:
        action: Action taken (LifecycleOp enum member)
        seed_info: Seed state info (None if no active seed)
        epoch: Current epoch
        max_epochs: Maximum epochs in episode
        val_acc: Current validation accuracy
        num_contributing_fossilized: Count of fossilized seeds with meaningful contribution
        config: Reward configuration (uses default if None)

    Returns:
        Simplified reward value
    """
    if config is None:
        config = _DEFAULT_CONTRIBUTION_CONFIG

    reward = 0.0

    # === 1. PBRS: Stage Progression ===
    # This is the ONLY shaping that preserves optimal policy guarantees
    if seed_info is not None:
        reward += _contribution_pbrs_bonus(seed_info, config)

    # === 2. Intervention Cost ===
    # Uniform small negative cost for any non-WAIT action
    # Prevents "action spam" without creating complex penalty landscape
    if action != LifecycleOp.WAIT:
        reward -= 0.01

    # === 3. Terminal Bonus ===
    # Scaled for 25-step credit assignment (DRL Expert recommendation)
    if epoch == max_epochs:
        # Accuracy component: [0, 3] range
        accuracy_bonus = (val_acc / 100.0) * 3.0
        # Fossilize component: [0, 6] for 3 slots max
        fossilize_bonus = num_contributing_fossilized * 2.0
        reward += accuracy_bonus + fossilize_bonus

    return reward
```

**Step 4: Add to exports**

In `src/esper/simic/rewards/rewards.py`, add to `__all__` list (around line 1424):

```python
__all__ = [
    # ... existing exports ...
    "compute_simplified_reward",
    # ...
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_simplified.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/rewards/rewards.py tests/simic/test_reward_simplified.py
git commit -m "feat(simic): implement compute_simplified_reward for A/B testing"
```

---

## Task 3: Wire SIMPLIFIED into compute_reward Dispatcher

**Files:**
- Modify: `src/esper/simic/rewards/rewards.py:887-927` (compute_reward function)
- Test: `tests/simic/test_reward_simplified.py`

**Step 1: Write the failing test**

Append to `tests/simic/test_reward_simplified.py`:

```python
from esper.simic.rewards import compute_reward


class TestComputeRewardDispatcher:
    """Test that compute_reward dispatches to SIMPLIFIED correctly."""

    def test_simplified_mode_dispatches(self):
        """compute_reward with SIMPLIFIED mode should use simplified logic."""
        config = ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED)
        seed_info = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=-5.0,  # Would trigger warnings in SHAPED
            total_improvement=-3.0,
            epochs_in_stage=5,
            seed_params=1000,
            previous_stage=STAGE_PROBATIONARY,
            previous_epochs_in_stage=4,
            seed_age_epochs=20,
        )

        # Call through dispatcher
        reward = compute_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=5.0,  # Would give attribution in SHAPED
            val_acc=60.0,
            host_max_acc=65.0,
            seed_info=seed_info,
            epoch=20,
            max_epochs=25,
            total_params=100000,
            host_params=90000,
            acc_at_germination=55.0,
            acc_delta=0.5,
            num_fossilized_seeds=1,
            num_contributing_fossilized=1,
            config=config,
        )

        # Should NOT have probation_warning (-9.0) or attribution (+5.0)
        # Should be small (just PBRS)
        assert -2.0 < reward < 2.0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_simplified.py::TestComputeRewardDispatcher -v`
Expected: FAIL (dispatch not implemented, falls through to error)

**Step 3: Add SIMPLIFIED dispatch to compute_reward**

In `src/esper/simic/rewards/rewards.py`, modify `compute_reward` function (around line 887-927):

```python
def compute_reward(
    action: LifecycleOp,
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
    - SIMPLIFIED: PBRS + intervention cost + terminal (DRL Expert recommended)
    ...
    """
    if config is None:
        config = ContributionRewardConfig()

    # Dispatch based on reward mode
    if config.reward_mode == RewardMode.SHAPED:
        return compute_contribution_reward(
            # ... existing code ...
        )

    elif config.reward_mode == RewardMode.SPARSE:
        reward = compute_sparse_reward(
            # ... existing code ...
        )

    elif config.reward_mode == RewardMode.MINIMAL:
        seed_age = seed_info.seed_age_epochs if seed_info else None
        reward = compute_minimal_reward(
            # ... existing code ...
        )

    elif config.reward_mode == RewardMode.SIMPLIFIED:
        reward = compute_simplified_reward(
            action=action,
            seed_info=seed_info,
            epoch=epoch,
            max_epochs=max_epochs,
            val_acc=val_acc,
            num_contributing_fossilized=num_contributing_fossilized,
            config=config,
        )

    else:
        raise ValueError(f"Unknown reward mode: {config.reward_mode}")

    # Handle return_components for sparse/minimal/simplified modes
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

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_simplified.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/rewards/rewards.py tests/simic/test_reward_simplified.py
git commit -m "feat(simic): wire SIMPLIFIED mode into compute_reward dispatcher"
```

---

## Task 4: Add Per-Environment Reward Mode Support

**Files:**
- Modify: `src/esper/simic/training/vectorized.py`
- Modify: `src/esper/simic/training/config.py`
- Test: `tests/simic/test_reward_simplified.py`

**Step 1: Write the failing test**

Append to `tests/simic/test_reward_simplified.py`:

```python
from esper.simic.training.config import TrainingConfig


class TestABTestingConfig:
    """Test A/B testing configuration."""

    def test_ab_reward_modes_field_exists(self):
        """TrainingConfig should have ab_reward_modes field."""
        config = TrainingConfig()
        assert hasattr(config, "ab_reward_modes")
        # Default should be None (all envs use reward_mode)
        assert config.ab_reward_modes is None

    def test_ab_reward_modes_splits_envs(self):
        """ab_reward_modes should specify per-env reward modes."""
        config = TrainingConfig(
            n_envs=8,
            ab_reward_modes=["shaped", "shaped", "shaped", "shaped",
                            "simplified", "simplified", "simplified", "simplified"],
        )
        assert len(config.ab_reward_modes) == 8
        assert config.ab_reward_modes[0] == "shaped"
        assert config.ab_reward_modes[4] == "simplified"

    def test_ab_reward_modes_validation(self):
        """ab_reward_modes length must match n_envs."""
        with pytest.raises(ValueError, match="ab_reward_modes.*must match.*n_envs"):
            TrainingConfig(
                n_envs=8,
                ab_reward_modes=["shaped", "simplified"],  # Wrong length
            )
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_simplified.py::TestABTestingConfig -v`
Expected: FAIL with `AttributeError: ab_reward_modes`

**Step 3: Add ab_reward_modes to TrainingConfig**

In `src/esper/simic/training/config.py`, add field to `TrainingConfig`:

```python
@dataclass(slots=True)
class TrainingConfig:
    # ... existing fields ...

    # === A/B Testing ===
    # Per-environment reward mode override for A/B testing
    # If None, all envs use reward_mode. If list, must match n_envs length.
    # Example: ["shaped"]*4 + ["simplified"]*4 for 8-env A/B test
    ab_reward_modes: list[str] | None = None

    # ... rest of class ...
```

Add validation in `_validate()`:

```python
def _validate(self) -> None:
    # ... existing validation ...

    # A/B testing validation
    if self.ab_reward_modes is not None:
        if len(self.ab_reward_modes) != self.n_envs:
            raise ValueError(
                f"ab_reward_modes length ({len(self.ab_reward_modes)}) "
                f"must match n_envs ({self.n_envs})"
            )
        # Validate each mode is valid
        valid_modes = {m.value for m in RewardMode}
        for i, mode in enumerate(self.ab_reward_modes):
            if mode not in valid_modes:
                raise ValueError(
                    f"ab_reward_modes[{i}] = '{mode}' is not a valid RewardMode. "
                    f"Valid modes: {sorted(valid_modes)}"
                )
```

Add to `to_train_kwargs()`:

```python
def to_train_kwargs(self) -> dict[str, Any]:
    return {
        # ... existing fields ...
        "ab_reward_modes": self.ab_reward_modes,
    }
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/simic/test_reward_simplified.py::TestABTestingConfig -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/simic/training/config.py tests/simic/test_reward_simplified.py
git commit -m "feat(simic): add ab_reward_modes config for A/B testing"
```

---

## Task 5: Wire Per-Environment Reward Modes in Vectorized Training

**Files:**
- Modify: `src/esper/simic/training/vectorized.py:385-460` (function signature and setup)
- Modify: `src/esper/simic/training/vectorized.py:1850-1900` (reward computation)

**Step 1: Add ab_reward_modes parameter to train_ppo_vectorized**

In `src/esper/simic/training/vectorized.py`, add parameter:

```python
def train_ppo_vectorized(
    # ... existing params ...
    reward_family: str = "contribution",
    ab_reward_modes: list[str] | None = None,  # ADD THIS
    quiet_analytics: bool = False,
    # ...
) -> tuple[PPOAgent, list[dict]]:
```

**Step 2: Create per-env reward configs in setup**

After the reward_config initialization (around line 543), add:

```python
    # Per-environment reward configs for A/B testing
    if ab_reward_modes is not None:
        if len(ab_reward_modes) != n_envs:
            raise ValueError(
                f"ab_reward_modes length ({len(ab_reward_modes)}) must match n_envs ({n_envs})"
            )
        env_reward_configs = []
        for env_idx, mode_str in enumerate(ab_reward_modes):
            env_mode = RewardMode(mode_str)
            env_config = ContributionRewardConfig(
                reward_mode=env_mode,
                param_budget=param_budget,
                param_penalty_weight=param_penalty_weight,
                sparse_reward_scale=sparse_reward_scale,
            )
            env_reward_configs.append(env_config)
        _logger.info(
            "A/B testing enabled: %s",
            {mode: ab_reward_modes.count(mode) for mode in set(ab_reward_modes)}
        )
    else:
        env_reward_configs = [reward_config] * n_envs
```

**Step 3: Use per-env config in reward computation**

In the reward computation section (around line 1872), change:

```python
# Before:
config=reward_config,

# After:
config=env_reward_configs[env_idx],
```

Apply this change to both the `need_reward_components` branch and the else branch.

**Step 4: Add A/B group tracking to telemetry**

In the telemetry emission (around line 2027), add:

```python
hub.emit(TelemetryEvent(
    event_type=TelemetryEventType.REWARD_COMPUTED,
    # ... existing fields ...
    data={
        "env_id": env_idx,
        "episode": episodes_completed + env_idx,
        "ab_group": env_reward_configs[env_idx].reward_mode.value,  # ADD THIS
        **reward_components.to_dict(),
    },
    # ...
))
```

**Step 5: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/simic/ -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "feat(simic): wire per-environment reward modes for A/B testing"
```

---

## Task 6: Add CLI Support for A/B Testing

**Files:**
- Modify: `src/esper/scripts/train.py`
- Test: Manual CLI test

**Step 1: Add --ab-test flag to CLI**

In `src/esper/scripts/train.py`, in the PPO argument group:

```python
ppo_parser.add_argument(
    "--ab-test",
    type=str,
    choices=["shaped-vs-simplified", "shaped-vs-sparse"],
    default=None,
    help="Run A/B test: split envs between two reward modes (requires even n_envs)",
)
```

**Step 2: Process --ab-test in main**

Before calling `train_ppo_vectorized`:

```python
# Handle A/B testing
ab_reward_modes = None
if args.ab_test:
    if config.n_envs % 2 != 0:
        raise ValueError("--ab-test requires even number of envs")
    half = config.n_envs // 2
    if args.ab_test == "shaped-vs-simplified":
        ab_reward_modes = ["shaped"] * half + ["simplified"] * half
    elif args.ab_test == "shaped-vs-sparse":
        ab_reward_modes = ["shaped"] * half + ["sparse"] * half
    print(f"[A/B Test] {half} envs SHAPED vs {half} envs {args.ab_test.split('-vs-')[1].upper()}")
```

Pass to training:

```python
agent, history = train_ppo_vectorized(
    # ... existing args ...
    ab_reward_modes=ab_reward_modes,
)
```

**Step 3: Manual test**

Run: `PYTHONPATH=src uv run python -m esper.scripts.train ppo --preset cifar10 --n-envs 8 --ab-test shaped-vs-simplified --episodes 2`

Expected: Startup message shows A/B split, training runs without error

**Step 4: Commit**

```bash
git add src/esper/scripts/train.py
git commit -m "feat(cli): add --ab-test flag for reward A/B testing"
```

---

## Task 7: Add A/B Telemetry Summary

**Files:**
- Modify: `src/esper/simic/training/vectorized.py` (end-of-training summary)

**Step 1: Add A/B comparison at training end**

After the training loop completes (around the final summary section), add:

```python
# A/B Test Summary
if ab_reward_modes is not None:
    print("\n" + "=" * 60)
    print("A/B TEST RESULTS")
    print("=" * 60)

    # Group episodes by reward mode
    from collections import defaultdict
    ab_groups = defaultdict(list)
    for ep_idx, ep_data in enumerate(episode_history):
        env_idx = ep_idx % n_envs
        mode = env_reward_configs[env_idx].reward_mode.value
        ab_groups[mode].append(ep_data)

    for mode, episodes in sorted(ab_groups.items()):
        rewards = [ep.get("episode_reward", 0) for ep in episodes]
        accuracies = [ep.get("final_accuracy", 0) for ep in episodes]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
        print(f"\n{mode.upper()} ({len(episodes)} episodes):")
        print(f"  Avg Episode Reward: {avg_reward:.2f}")
        print(f"  Avg Final Accuracy: {avg_acc:.2f}%")
        print(f"  Reward Range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    print("=" * 60)
```

**Step 2: Commit**

```bash
git add src/esper/simic/training/vectorized.py
git commit -m "feat(simic): add A/B test summary at training end"
```

---

## Task 8: Integration Test

**Files:**
- Create: `tests/integration/test_ab_reward_testing.py`

**Step 1: Write integration test**

```python
"""Integration test for reward A/B testing."""
import pytest
import torch

from esper.simic.training.config import TrainingConfig
from esper.simic.rewards import RewardMode


@pytest.mark.slow
def test_ab_testing_runs_without_error():
    """A/B testing with split reward modes should complete without error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for vectorized training")

    config = TrainingConfig(
        n_envs=4,
        n_episodes=2,  # Short test
        max_epochs=5,  # Short episodes
        ab_reward_modes=["shaped", "shaped", "simplified", "simplified"],
    )

    # Verify config is valid
    assert config.ab_reward_modes is not None
    assert len(config.ab_reward_modes) == 4

    # Note: Full training test would be too slow for CI
    # This just validates config construction and validation
```

**Step 2: Run test**

Run: `PYTHONPATH=src uv run pytest tests/integration/test_ab_reward_testing.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_ab_reward_testing.py
git commit -m "test(integration): add A/B reward testing validation"
```

---

## Task 9: Update Exports and Documentation

**Files:**
- Modify: `src/esper/simic/rewards/__init__.py`

**Step 1: Add compute_simplified_reward to exports**

```python
from .rewards import (
    # ... existing exports ...
    compute_simplified_reward,
)

__all__ = [
    # ... existing exports ...
    "compute_simplified_reward",
]
```

**Step 2: Commit**

```bash
git add src/esper/simic/rewards/__init__.py
git commit -m "chore(simic): export compute_simplified_reward"
```

---

## Task 10: Final Verification

**Step 1: Run full test suite**

```bash
PYTHONPATH=src uv run pytest tests/ -v --ignore=tests/integration
```

Expected: All tests pass

**Step 2: Run a short A/B test**

```bash
PYTHONPATH=src uv run python -m esper.scripts.train ppo \
    --preset cifar10 \
    --n-envs 8 \
    --episodes 10 \
    --ab-test shaped-vs-simplified \
    --device cuda:0
```

Expected: Training completes, A/B summary shows both groups

**Step 3: Final commit with verification**

```bash
git add -A
git commit -m "feat(simic): complete A/B reward testing infrastructure

- Add RewardMode.SIMPLIFIED (3-component: PBRS + cost + terminal)
- Add compute_simplified_reward() function
- Add per-environment reward mode support
- Add --ab-test CLI flag (shaped-vs-simplified, shaped-vs-sparse)
- Add A/B telemetry and end-of-training summary

DRL Expert recommendation: SIMPLIFIED removes conflicting reward
components that create unlearnable gradients. Test against SHAPED
to validate learning improvement.
"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Add SIMPLIFIED enum | rewards.py |
| 2 | Implement compute_simplified_reward | rewards.py |
| 3 | Wire dispatcher | rewards.py |
| 4 | Add ab_reward_modes config | config.py |
| 5 | Per-env reward in vectorized | vectorized.py |
| 6 | CLI --ab-test flag | train.py |
| 7 | A/B summary telemetry | vectorized.py |
| 8 | Integration test | test_ab_reward_testing.py |
| 9 | Exports | __init__.py |
| 10 | Final verification | - |

**Recommended PPO settings for A/B test:**
```bash
--entropy-coef 0.05   # Higher for sparse-ish SIMPLIFIED
--episodes 100        # Enough to see divergence
--n-envs 8            # 4 vs 4 split
```
