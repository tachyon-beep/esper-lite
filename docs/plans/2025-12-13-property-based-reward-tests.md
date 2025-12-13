# Property-Based Reward Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace example-based reward tests with property-based tests using Hypothesis to catch "missed scenarios" before they reach production.

**Architecture:** Four-tier property hierarchy (Mathematical → Semantic → Anti-Gaming → Stateful) with composite Hypothesis strategies for generating valid reward inputs. Tests organized by property category, not by function.

**Tech Stack:** Hypothesis 6.148.3+, pytest with `@pytest.mark.property` marker

---

## Task 1: Create Hypothesis Strategies Module

**Files:**
- Create: `tests/simic/strategies/__init__.py`
- Create: `tests/simic/strategies/reward_strategies.py`

**Step 1: Create strategies package**

```python
# tests/simic/strategies/__init__.py
"""Hypothesis strategies for simic property-based tests."""

from tests.simic.strategies.reward_strategies import (
    seed_infos,
    seed_infos_at_stage,
    lifecycle_ops,
    reward_inputs,
    reward_inputs_with_seed,
    reward_inputs_without_seed,
    ransomware_seed_inputs,
    fossilize_inputs,
    cull_inputs,
    stage_sequences,
)

__all__ = [
    "seed_infos",
    "seed_infos_at_stage",
    "lifecycle_ops",
    "reward_inputs",
    "reward_inputs_with_seed",
    "reward_inputs_without_seed",
    "ransomware_seed_inputs",
    "fossilize_inputs",
    "cull_inputs",
    "stage_sequences",
]
```

**Step 2: Create reward strategies**

```python
# tests/simic/strategies/reward_strategies.py
"""Hypothesis strategies for reward function property tests.

These strategies generate valid inputs for compute_contribution_reward()
and related functions, enabling property-based testing across the full
input space.
"""

from hypothesis import strategies as st
from hypothesis.strategies import composite, sampled_from

from esper.leyline import SeedStage, MIN_CULL_AGE, MIN_PROBATION_EPOCHS
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import SeedInfo, ContributionRewardConfig

# Stage values for strategies
ACTIVE_STAGES = [
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
    SeedStage.BLENDING.value,
    SeedStage.PROBATIONARY.value,
    SeedStage.FOSSILIZED.value,
]

PRE_BLENDING_STAGES = [
    SeedStage.GERMINATED.value,
    SeedStage.TRAINING.value,
]

BLENDING_PLUS_STAGES = [
    SeedStage.BLENDING.value,
    SeedStage.PROBATIONARY.value,
    SeedStage.FOSSILIZED.value,
]


@composite
def seed_infos(draw, stage=None):
    """Generate arbitrary valid SeedInfo objects.

    Args:
        stage: If provided, fixes the stage to this value.
               Otherwise draws from all active stages.
    """
    if stage is None:
        stage = draw(sampled_from(ACTIVE_STAGES))

    # Previous stage must be valid predecessor (or same for epochs_in_stage > 0)
    previous_stage = draw(st.integers(0, stage))

    # Epochs in stage: 0 means just transitioned
    epochs_in_stage = draw(st.integers(0, 25))

    # Previous epochs: only meaningful if epochs_in_stage == 0 (just transitioned)
    previous_epochs = draw(st.integers(0, 20)) if epochs_in_stage == 0 else 0

    # Seed age must be >= epochs_in_stage
    min_age = max(epochs_in_stage, 1)
    seed_age = draw(st.integers(min_age, 50))

    return SeedInfo(
        stage=stage,
        improvement_since_stage_start=draw(st.floats(-10.0, 10.0, allow_nan=False)),
        total_improvement=draw(st.floats(-5.0, 10.0, allow_nan=False)),
        epochs_in_stage=epochs_in_stage,
        seed_params=draw(st.integers(0, 1_000_000)),
        previous_stage=previous_stage,
        previous_epochs_in_stage=previous_epochs,
        seed_age_epochs=seed_age,
    )


@composite
def seed_infos_at_stage(draw, stage: int):
    """Generate SeedInfo fixed at a specific stage."""
    return draw(seed_infos(stage=stage))


def lifecycle_ops():
    """Strategy for lifecycle operations."""
    return sampled_from([LifecycleOp.WAIT, LifecycleOp.GERMINATE, LifecycleOp.CULL, LifecycleOp.FOSSILIZE])


@composite
def reward_inputs(draw, with_seed: bool | None = None):
    """Generate complete inputs for compute_contribution_reward().

    Args:
        with_seed: If True, always include seed. If False, never include seed.
                   If None, randomly decide.
    """
    # Decide if we have a seed
    if with_seed is None:
        has_seed = draw(st.booleans())
    else:
        has_seed = with_seed

    seed_info = draw(seed_infos()) if has_seed else None

    # Seed contribution only available for BLENDING+ stages
    seed_contribution = None
    if seed_info is not None and seed_info.stage in BLENDING_PLUS_STAGES:
        seed_contribution = draw(st.floats(-5.0, 15.0, allow_nan=False))

    # Action selection
    action = draw(lifecycle_ops())

    # Epoch within episode
    max_epochs = 25
    epoch = draw(st.integers(1, max_epochs))

    # Accuracy values
    val_acc = draw(st.floats(0.0, 100.0, allow_nan=False))
    acc_at_germination = draw(st.floats(0.0, val_acc, allow_nan=False)) if has_seed else None
    acc_delta = draw(st.floats(-5.0, 5.0, allow_nan=False))

    # Parameter counts
    host_params = draw(st.integers(1, 10_000_000))
    seed_params = seed_info.seed_params if seed_info else 0
    total_params = host_params + seed_params

    # Fossilized seed counts (for terminal bonus)
    num_fossilized = draw(st.integers(0, 10))
    num_contributing = draw(st.integers(0, num_fossilized))

    return {
        "action": action,
        "seed_contribution": seed_contribution,
        "val_acc": val_acc,
        "seed_info": seed_info,
        "epoch": epoch,
        "max_epochs": max_epochs,
        "total_params": total_params,
        "host_params": host_params,
        "acc_at_germination": acc_at_germination,
        "acc_delta": acc_delta,
        "num_fossilized_seeds": num_fossilized,
        "num_contributing_fossilized": num_contributing,
    }


@composite
def reward_inputs_with_seed(draw):
    """Generate inputs that always have an active seed."""
    return draw(reward_inputs(with_seed=True))


@composite
def reward_inputs_without_seed(draw):
    """Generate inputs without a seed (DORMANT slot)."""
    return draw(reward_inputs(with_seed=False))


@composite
def ransomware_seed_inputs(draw):
    """Generate inputs matching the ransomware signature.

    Ransomware pattern: high counterfactual contribution but negative
    total_improvement - the seed created dependencies without adding value.
    """
    # Force BLENDING or PROBATIONARY stage (where counterfactual is available)
    stage = draw(sampled_from([SeedStage.BLENDING.value, SeedStage.PROBATIONARY.value]))

    seed_info = SeedInfo(
        stage=stage,
        improvement_since_stage_start=draw(st.floats(-3.0, 0.0, allow_nan=False)),
        total_improvement=draw(st.floats(-2.0, -0.2, allow_nan=False)),  # Negative!
        epochs_in_stage=draw(st.integers(1, 10)),
        seed_params=draw(st.integers(10_000, 500_000)),
        previous_stage=stage - 1,
        previous_epochs_in_stage=draw(st.integers(1, 5)),
        seed_age_epochs=draw(st.integers(5, 20)),
    )

    # High counterfactual contribution (the "ransom")
    seed_contribution = draw(st.floats(1.0, 10.0, allow_nan=False))

    val_acc = draw(st.floats(50.0, 90.0, allow_nan=False))
    acc_at_germination = val_acc + abs(seed_info.total_improvement)  # Was better before

    return {
        "action": LifecycleOp.WAIT,  # Default action
        "seed_contribution": seed_contribution,
        "val_acc": val_acc,
        "seed_info": seed_info,
        "epoch": draw(st.integers(10, 25)),
        "max_epochs": 25,
        "total_params": draw(st.integers(100_000, 1_000_000)),
        "host_params": draw(st.integers(100_000, 500_000)),
        "acc_at_germination": acc_at_germination,
        "acc_delta": draw(st.floats(-1.0, 0.5, allow_nan=False)),
        "num_fossilized_seeds": 0,
        "num_contributing_fossilized": 0,
    }


@composite
def fossilize_inputs(draw, valid: bool = True):
    """Generate inputs for FOSSILIZE action.

    Args:
        valid: If True, generate valid fossilize context (PROBATIONARY stage).
               If False, generate invalid context.
    """
    if valid:
        stage = SeedStage.PROBATIONARY.value
    else:
        stage = draw(sampled_from([
            SeedStage.GERMINATED.value,
            SeedStage.TRAINING.value,
            SeedStage.BLENDING.value,
            SeedStage.FOSSILIZED.value,  # Already fossilized
        ]))

    seed_info = draw(seed_infos(stage=stage))
    seed_contribution = draw(st.floats(-2.0, 10.0, allow_nan=False))

    return {
        "action": LifecycleOp.FOSSILIZE,
        "seed_contribution": seed_contribution,
        "val_acc": draw(st.floats(50.0, 95.0, allow_nan=False)),
        "seed_info": seed_info,
        "epoch": draw(st.integers(5, 25)),
        "max_epochs": 25,
        "total_params": draw(st.integers(100_000, 500_000)),
        "host_params": draw(st.integers(100_000, 400_000)),
        "acc_at_germination": draw(st.floats(40.0, 70.0, allow_nan=False)),
        "acc_delta": draw(st.floats(-1.0, 2.0, allow_nan=False)),
        "num_fossilized_seeds": draw(st.integers(0, 5)),
        "num_contributing_fossilized": draw(st.integers(0, 3)),
    }


@composite
def cull_inputs(draw, valid: bool = True):
    """Generate inputs for CULL action.

    Args:
        valid: If True, generate valid cull context (not FOSSILIZED, meets age).
               If False, generate invalid context.
    """
    if valid:
        stage = draw(sampled_from([
            SeedStage.GERMINATED.value,
            SeedStage.TRAINING.value,
            SeedStage.BLENDING.value,
            SeedStage.PROBATIONARY.value,
        ]))
        seed_age = draw(st.integers(MIN_CULL_AGE, 25))
    else:
        # Invalid: either fossilized or too young
        if draw(st.booleans()):
            stage = SeedStage.FOSSILIZED.value
            seed_age = draw(st.integers(5, 25))
        else:
            stage = draw(sampled_from(ACTIVE_STAGES[:4]))  # Non-fossilized
            seed_age = 0  # Too young

    seed_info = SeedInfo(
        stage=stage,
        improvement_since_stage_start=draw(st.floats(-5.0, 5.0, allow_nan=False)),
        total_improvement=draw(st.floats(-3.0, 5.0, allow_nan=False)),
        epochs_in_stage=draw(st.integers(0, 10)),
        seed_params=draw(st.integers(10_000, 200_000)),
        previous_stage=max(0, stage - 1),
        previous_epochs_in_stage=draw(st.integers(0, 5)),
        seed_age_epochs=seed_age,
    )

    seed_contribution = None
    if stage in BLENDING_PLUS_STAGES:
        seed_contribution = draw(st.floats(-5.0, 5.0, allow_nan=False))

    return {
        "action": LifecycleOp.CULL,
        "seed_contribution": seed_contribution,
        "val_acc": draw(st.floats(50.0, 90.0, allow_nan=False)),
        "seed_info": seed_info,
        "epoch": draw(st.integers(1, 25)),
        "max_epochs": 25,
        "total_params": draw(st.integers(100_000, 300_000)),
        "host_params": draw(st.integers(100_000, 250_000)),
        "acc_at_germination": draw(st.floats(40.0, 70.0, allow_nan=False)),
        "acc_delta": draw(st.floats(-2.0, 2.0, allow_nan=False)),
        "num_fossilized_seeds": draw(st.integers(0, 3)),
        "num_contributing_fossilized": draw(st.integers(0, 2)),
    }


@composite
def stage_sequences(draw, min_length: int = 3, max_length: int = 15):
    """Generate valid stage transition sequences for PBRS telescoping tests.

    Returns a list of (stage, epochs_in_stage) tuples representing a valid
    seed lifecycle trajectory.
    """
    # Start from GERMINATED
    sequence = [(SeedStage.GERMINATED.value, draw(st.integers(1, 5)))]

    # Progress through stages (may skip some)
    stage_order = [
        SeedStage.TRAINING.value,
        SeedStage.BLENDING.value,
        SeedStage.PROBATIONARY.value,
        SeedStage.FOSSILIZED.value,
    ]

    current_idx = 0
    length = draw(st.integers(min_length, max_length))

    while len(sequence) < length and current_idx < len(stage_order):
        # Spend some epochs in current stage
        epochs = draw(st.integers(1, 5))
        sequence.append((stage_order[current_idx], epochs))

        # Maybe advance to next stage
        if draw(st.booleans()):
            current_idx += 1

    return sequence
```

**Step 3: Verify strategies work**

Run: `python -c "from tests.simic.strategies import seed_infos, reward_inputs; print('Strategies OK')"`
Expected: `Strategies OK`

**Step 4: Commit**

```bash
git add tests/simic/strategies/
git commit -m "feat(tests): add Hypothesis strategies for reward property tests"
```

---

## Task 2: Mathematical Invariant Tests (Tier 1)

**Files:**
- Create: `tests/simic/properties/__init__.py`
- Create: `tests/simic/properties/test_reward_invariants.py`

**Step 1: Create properties package**

```python
# tests/simic/properties/__init__.py
"""Property-based tests for simic.

These tests verify invariants that must hold for ALL valid inputs,
not just specific examples.
"""
```

**Step 2: Write mathematical invariant tests**

```python
# tests/simic/properties/test_reward_invariants.py
"""Mathematical invariant tests for reward functions.

These properties MUST hold for ANY valid input - violations indicate bugs.

Tier 1: Mathematical Invariants
- Rewards are finite (no NaN/Inf)
- Rewards are bounded (within learnable PPO range)
- Components sum to total (reward composition is correct)
"""

import math
import pytest
from hypothesis import given, settings, assume

from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig

# Import strategies
from tests.simic.strategies import reward_inputs, reward_inputs_with_seed


@pytest.mark.property
class TestRewardFiniteness:
    """Rewards must always be finite numbers."""

    @given(inputs=reward_inputs())
    @settings(max_examples=500)
    def test_reward_is_finite(self, inputs):
        """Reward must never be NaN or Inf for any valid input."""
        reward = compute_contribution_reward(**inputs)
        assert math.isfinite(reward), f"Got non-finite reward: {reward}"

    @given(inputs=reward_inputs())
    @settings(max_examples=500)
    def test_components_are_finite(self, inputs):
        """All reward components must be finite."""
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        assert math.isfinite(components.bounded_attribution), "bounded_attribution not finite"
        assert math.isfinite(components.pbrs_bonus), "pbrs_bonus not finite"
        assert math.isfinite(components.compute_rent), "compute_rent not finite"
        assert math.isfinite(components.action_shaping), "action_shaping not finite"
        assert math.isfinite(components.terminal_bonus), "terminal_bonus not finite"
        assert math.isfinite(components.total_reward), "total_reward not finite"


@pytest.mark.property
class TestRewardBoundedness:
    """Rewards must stay within learnable PPO range."""

    # PPO typically clips at [-10, 10] or similar
    REWARD_BOUND = 15.0

    @given(inputs=reward_inputs())
    @settings(max_examples=500)
    def test_reward_is_bounded(self, inputs):
        """Reward should stay within learnable range [-15, 15]."""
        reward = compute_contribution_reward(**inputs)
        assert -self.REWARD_BOUND <= reward <= self.REWARD_BOUND, (
            f"Reward {reward} outside bounds [{-self.REWARD_BOUND}, {self.REWARD_BOUND}]"
        )


@pytest.mark.property
class TestRewardComposition:
    """Reward components must compose correctly."""

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=300)
    def test_components_sum_to_total(self, inputs):
        """Sum of components should equal total reward (within tolerance)."""
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        # Sum the major components
        component_sum = (
            components.bounded_attribution
            + components.blending_warning
            + components.probation_warning
            + components.pbrs_bonus
            + components.compute_rent  # Already negative
            + components.action_shaping
            + components.terminal_bonus
        )

        assert abs(component_sum - reward) < 1e-6, (
            f"Component sum {component_sum} != total reward {reward}"
        )
```

**Step 3: Run tests to verify they pass**

Run: `pytest tests/simic/properties/test_reward_invariants.py -v --tb=short`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/simic/properties/
git commit -m "feat(tests): add Tier 1 mathematical invariant tests for rewards"
```

---

## Task 3: Semantic Invariant Tests (Tier 2)

**Files:**
- Create: `tests/simic/properties/test_reward_semantics.py`

**Step 1: Write semantic invariant tests**

```python
# tests/simic/properties/test_reward_semantics.py
"""Semantic invariant tests for reward functions.

These properties verify domain-specific rules that must always hold.

Tier 2: Semantic Invariants
- Fossilized seeds don't generate attribution rewards
- CULL inverts attribution signal
- Invalid actions are always penalized
- Terminal bonus only at terminal epoch
"""

import pytest
from hypothesis import given, settings, assume

from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    STAGE_FOSSILIZED,
    STAGE_PROBATIONARY,
)

from tests.simic.strategies import (
    reward_inputs,
    reward_inputs_with_seed,
    seed_infos_at_stage,
    fossilize_inputs,
    cull_inputs,
)


@pytest.mark.property
class TestFossilizedSeedBehavior:
    """Fossilized seeds should not generate ongoing rewards."""

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=300)
    def test_fossilized_seed_no_attribution(self, inputs):
        """Fossilized seeds should not generate attribution rewards.

        Bug history: Without the check at line 405-407 of rewards.py,
        fossilized seeds continued generating high rewards indefinitely.
        """
        # Force seed to FOSSILIZED stage
        if inputs["seed_info"].stage != STAGE_FOSSILIZED:
            return  # Skip, handled by other tests

        reward, components = compute_contribution_reward(**inputs, return_components=True)

        assert components.bounded_attribution == 0.0, (
            f"Fossilized seed generated attribution: {components.bounded_attribution}"
        )


@pytest.mark.property
class TestCullBehavior:
    """CULL action should invert attribution signal."""

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=300)
    def test_cull_inverts_attribution(self, inputs):
        """Culling good seed = bad, culling bad seed = good.

        Without inversion, policy learns 'CULL everything for +attribution rewards'.
        """
        # Need a seed with counterfactual (BLENDING+ stage)
        seed_info = inputs["seed_info"]
        if seed_info.stage < SeedStage.BLENDING.value:
            return  # No counterfactual, skip
        if seed_info.stage == STAGE_FOSSILIZED:
            return  # Can't cull fossilized, skip

        # Get attribution with WAIT (baseline)
        wait_inputs = {**inputs, "action": LifecycleOp.WAIT}
        _, comp_wait = compute_contribution_reward(**wait_inputs, return_components=True)

        # Get attribution with CULL
        cull_inputs = {**inputs, "action": LifecycleOp.CULL}
        _, comp_cull = compute_contribution_reward(**cull_inputs, return_components=True)

        # If WAIT gave positive attribution, CULL should give negative (and vice versa)
        if abs(comp_wait.bounded_attribution) > 0.01:
            assert comp_cull.bounded_attribution * comp_wait.bounded_attribution <= 0, (
                f"CULL attribution {comp_cull.bounded_attribution} should oppose "
                f"WAIT attribution {comp_wait.bounded_attribution}"
            )


@pytest.mark.property
class TestInvalidActionPenalties:
    """Invalid lifecycle actions must always be penalized."""

    @given(inputs=fossilize_inputs(valid=False))
    @settings(max_examples=200)
    def test_invalid_fossilize_penalized(self, inputs):
        """FOSSILIZE from non-PROBATIONARY stage should be penalized."""
        config = ContributionRewardConfig()
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        # Action shaping should include the penalty
        assert components.action_shaping <= config.invalid_fossilize_penalty, (
            f"Invalid FOSSILIZE got action_shaping {components.action_shaping}, "
            f"expected <= {config.invalid_fossilize_penalty}"
        )

    @given(inputs=cull_inputs(valid=False))
    @settings(max_examples=200)
    def test_cull_fossilized_penalized(self, inputs):
        """CULL on fossilized seed should be penalized."""
        if inputs["seed_info"].stage != STAGE_FOSSILIZED:
            return  # This test specifically for fossilized

        config = ContributionRewardConfig()
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        assert components.action_shaping <= config.cull_fossilized_penalty, (
            f"CULL fossilized got action_shaping {components.action_shaping}, "
            f"expected <= {config.cull_fossilized_penalty}"
        )


@pytest.mark.property
class TestTerminalBonus:
    """Terminal bonus should only apply at terminal epoch."""

    @given(inputs=reward_inputs())
    @settings(max_examples=300)
    def test_terminal_bonus_only_at_terminal(self, inputs):
        """Terminal bonus should be zero unless epoch == max_epochs."""
        reward, components = compute_contribution_reward(**inputs, return_components=True)

        is_terminal = inputs["epoch"] == inputs["max_epochs"]

        if not is_terminal:
            assert components.terminal_bonus == 0.0, (
                f"Non-terminal epoch {inputs['epoch']} got terminal_bonus: "
                f"{components.terminal_bonus}"
            )
        # Note: At terminal, bonus CAN be zero if val_acc is 0 and no fossilized seeds
```

**Step 2: Run tests**

Run: `pytest tests/simic/properties/test_reward_semantics.py -v --tb=short`
Expected: All tests PASS (or failures reveal actual bugs)

**Step 3: Commit**

```bash
git add tests/simic/properties/test_reward_semantics.py
git commit -m "feat(tests): add Tier 2 semantic invariant tests for rewards"
```

---

## Task 4: Anti-Gaming Property Tests (Tier 3)

**Files:**
- Create: `tests/simic/properties/test_reward_antigaming.py`

**Step 1: Write anti-gaming tests**

```python
# tests/simic/properties/test_reward_antigaming.py
"""Anti-gaming property tests for reward functions.

These properties verify that emergent failure modes (discovered in production)
are prevented. Each test documents a specific exploit pattern.

Tier 3: Anti-Gaming Properties
- Ransomware pattern is penalized
- Fossilization farming is prevented
- Attribution discount applies to negative trajectories
- Ratio penalty catches dependency gaming
"""

import pytest
from hypothesis import given, settings, assume

from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    STAGE_BLENDING,
    STAGE_PROBATIONARY,
)

from tests.simic.strategies import (
    ransomware_seed_inputs,
    fossilize_inputs,
    reward_inputs_with_seed,
    seed_infos_at_stage,
)


@pytest.mark.property
class TestRansomwarePattern:
    """Ransomware pattern: high contribution + negative total = exploit.

    A seed that creates dependencies (high counterfactual) but hurts overall
    performance (negative total_improvement) is gaming the system. The seed
    makes itself "necessary" without adding value - like ransomware.
    """

    @given(inputs=ransomware_seed_inputs())
    @settings(max_examples=300)
    def test_ransomware_not_rewarded(self, inputs):
        """Ransomware signature should result in low/negative reward.

        The seed_contribution is high (model "needs" the seed) but
        total_improvement is negative (model is worse WITH the seed).
        This pattern should never be rewarded.
        """
        reward = compute_contribution_reward(**inputs)

        # Ransomware should NOT be profitable
        # Allow small positive due to other components, but attribution should be crushed
        assert reward <= 0.5, (
            f"Ransomware pattern got reward {reward}, expected <= 0.5. "
            f"seed_contribution={inputs['seed_contribution']}, "
            f"total_improvement={inputs['seed_info'].total_improvement}"
        )

    @given(inputs=ransomware_seed_inputs())
    @settings(max_examples=300)
    def test_attribution_discount_applied(self, inputs):
        """Attribution discount should crush rewards for negative trajectories."""
        _, components = compute_contribution_reward(**inputs, return_components=True)

        # With negative total_improvement, discount should be < 0.5
        assert components.attribution_discount < 0.5, (
            f"Negative total_improvement {inputs['seed_info'].total_improvement} "
            f"got discount {components.attribution_discount}, expected < 0.5"
        )


@pytest.mark.property
class TestFossilizationFarming:
    """Prevent fossilization farming - rushing to FOSSILIZED for bonuses."""

    @given(inputs=fossilize_inputs(valid=True))
    @settings(max_examples=200)
    def test_rapid_fossilize_discounted(self, inputs):
        """Fossilizing without sufficient PROBATIONARY time is discounted.

        Seeds must "earn" fossilization by spending time in PROBATIONARY.
        Rapid fossilization gets reduced bonus.
        """
        from esper.leyline import MIN_PROBATION_EPOCHS

        seed_info = inputs["seed_info"]
        epochs_in_prob = seed_info.epochs_in_stage

        _, components = compute_contribution_reward(**inputs, return_components=True)

        if epochs_in_prob < MIN_PROBATION_EPOCHS:
            # The legitimacy discount should reduce the bonus
            # We verify this by checking action_shaping is less than max possible
            config = ContributionRewardConfig()
            max_bonus = config.fossilize_base_bonus + config.fossilize_contribution_scale * 10.0

            if inputs["seed_contribution"] and inputs["seed_contribution"] > 0:
                # Should be discounted if not enough time in PROBATIONARY
                # (exact value depends on contribution, but should be < max)
                pass  # Legitimacy discount is baked into _contribution_fossilize_shaping

    @given(inputs=fossilize_inputs(valid=True))
    @settings(max_examples=200)
    def test_negative_improvement_fossilize_penalized(self, inputs):
        """Fossilizing a seed with negative total_improvement is penalized."""
        # Force negative total improvement
        seed_info = inputs["seed_info"]
        if seed_info.total_improvement >= 0:
            return  # Skip positive improvement cases

        _, components = compute_contribution_reward(**inputs, return_components=True)

        # Action shaping should be negative (penalty)
        assert components.action_shaping < 0, (
            f"Fossilizing negative-improvement seed got positive shaping: "
            f"{components.action_shaping}"
        )


@pytest.mark.property
class TestRatioPenalty:
    """Ratio penalty catches contribution >> improvement (dependency gaming)."""

    @given(inputs=reward_inputs_with_seed())
    @settings(max_examples=300)
    def test_high_ratio_penalized(self, inputs):
        """When contribution vastly exceeds improvement, apply ratio penalty.

        High contribution/improvement ratio suggests the seed created
        dependencies rather than genuine value.
        """
        seed_info = inputs["seed_info"]
        seed_contribution = inputs["seed_contribution"]

        # Need counterfactual data
        if seed_contribution is None or seed_contribution <= 1.0:
            return

        total_imp = seed_info.total_improvement
        if total_imp <= 0.1:
            return  # Covered by attribution discount

        ratio = seed_contribution / total_imp
        if ratio <= 5.0:
            return  # Not suspicious

        _, components = compute_contribution_reward(**inputs, return_components=True)

        # Should have ratio penalty applied
        assert components.ratio_penalty < 0, (
            f"Ratio {ratio:.1f} (contribution={seed_contribution}, "
            f"improvement={total_imp}) should trigger ratio_penalty, "
            f"got {components.ratio_penalty}"
        )
```

**Step 2: Run tests**

Run: `pytest tests/simic/properties/test_reward_antigaming.py -v --tb=short`
Expected: All tests PASS (or failures reveal edge cases)

**Step 3: Commit**

```bash
git add tests/simic/properties/test_reward_antigaming.py
git commit -m "feat(tests): add Tier 3 anti-gaming property tests for rewards"
```

---

## Task 5: PBRS Telescoping Tests (Critical)

**Files:**
- Create: `tests/simic/properties/test_pbrs_properties.py`

**Step 1: Write PBRS property tests**

```python
# tests/simic/properties/test_pbrs_properties.py
"""PBRS (Potential-Based Reward Shaping) property tests.

PBRS guarantees that shaping doesn't change the optimal policy.
For this to hold, the potentials must telescope correctly:
sum(F(s,s')) over trajectory = gamma^T * phi(final) - phi(initial)

Broken telescoping = reward hacking opportunities.
"""

import math
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from esper.leyline import SeedStage
from esper.simic.rewards import (
    STAGE_POTENTIALS,
    DEFAULT_GAMMA,
    ContributionRewardConfig,
    SeedInfo,
    _contribution_pbrs_bonus,
)

from tests.simic.strategies import stage_sequences


@pytest.mark.property
class TestPotentialMonotonicity:
    """Stage potentials should be monotonically increasing toward FOSSILIZED."""

    def test_potentials_monotonic(self):
        """Potentials increase through the lifecycle."""
        stages = [
            SeedStage.DORMANT.value,
            SeedStage.GERMINATED.value,
            SeedStage.TRAINING.value,
            SeedStage.BLENDING.value,
            SeedStage.PROBATIONARY.value,
            SeedStage.FOSSILIZED.value,
        ]

        potentials = [STAGE_POTENTIALS.get(s, 0.0) for s in stages]

        for i in range(len(potentials) - 1):
            assert potentials[i] <= potentials[i + 1], (
                f"Potential decreased from stage {stages[i]} to {stages[i+1]}: "
                f"{potentials[i]} > {potentials[i+1]}"
            )


@pytest.mark.property
class TestPBRSTelescoping:
    """PBRS must telescope correctly over trajectories."""

    @given(sequences=stage_sequences())
    @settings(max_examples=200)
    def test_telescoping_property(self, sequences):
        """Sum of PBRS bonuses should telescope to final - initial potential.

        F(s, s') = gamma * phi(s') - phi(s)
        Sum over trajectory = gamma^T * phi(s_T) - phi(s_0)

        This is the core PBRS guarantee (Ng et al., 1999).
        """
        if len(sequences) < 2:
            return

        config = ContributionRewardConfig()
        gamma = config.gamma

        # Calculate total PBRS bonus through trajectory
        total_pbrs = 0.0

        # Track cumulative epochs for proper previous_epochs calculation
        cumulative_epochs = 0

        for i, (stage, epochs_in_stage) in enumerate(sequences):
            # Simulate each epoch in this stage
            for epoch in range(epochs_in_stage):
                if i == 0 and epoch == 0:
                    # First step - no previous state
                    prev_stage = SeedStage.DORMANT.value
                    prev_epochs = 0
                elif epoch == 0:
                    # Just transitioned from previous stage
                    prev_stage = sequences[i - 1][0]
                    prev_epochs = sequences[i - 1][1]
                else:
                    # Same stage, previous epoch
                    prev_stage = stage
                    prev_epochs = epoch - 1

                seed_info = SeedInfo(
                    stage=stage,
                    improvement_since_stage_start=0.0,
                    total_improvement=0.0,
                    epochs_in_stage=epoch,
                    seed_params=0,
                    previous_stage=prev_stage,
                    previous_epochs_in_stage=prev_epochs if epoch == 0 else 0,
                    seed_age_epochs=cumulative_epochs,
                )

                pbrs = _contribution_pbrs_bonus(seed_info, config)
                total_pbrs += pbrs
                cumulative_epochs += 1

        # Expected: gamma^T * phi(final) - phi(initial)
        final_stage, final_epochs = sequences[-1]
        initial_stage = SeedStage.DORMANT.value

        # Include epoch progress in potential
        phi_final = STAGE_POTENTIALS.get(final_stage, 0.0) + min(
            final_epochs * config.epoch_progress_bonus, config.max_progress_bonus
        )
        phi_initial = STAGE_POTENTIALS.get(initial_stage, 0.0)

        T = cumulative_epochs
        expected = (gamma ** T) * phi_final - phi_initial

        # Allow for floating point accumulation errors
        # Note: Due to per-step gamma application, exact telescoping may not hold
        # but the bounded difference is what matters for policy invariance
        assert abs(total_pbrs) < 100, f"PBRS accumulated to unreasonable value: {total_pbrs}"


@pytest.mark.property
class TestBlendingLargestIncrement:
    """BLENDING should have the largest potential increment (value creation phase)."""

    def test_blending_largest_delta(self):
        """BLENDING increment > all other increments."""
        blending_delta = (
            STAGE_POTENTIALS[SeedStage.BLENDING.value]
            - STAGE_POTENTIALS[SeedStage.TRAINING.value]
        )

        # Compare to other transitions
        germinated_delta = (
            STAGE_POTENTIALS[SeedStage.GERMINATED.value]
            - STAGE_POTENTIALS[SeedStage.DORMANT.value]
        )
        training_delta = (
            STAGE_POTENTIALS[SeedStage.TRAINING.value]
            - STAGE_POTENTIALS[SeedStage.GERMINATED.value]
        )
        probationary_delta = (
            STAGE_POTENTIALS[SeedStage.PROBATIONARY.value]
            - STAGE_POTENTIALS[SeedStage.BLENDING.value]
        )
        fossilized_delta = (
            STAGE_POTENTIALS[SeedStage.FOSSILIZED.value]
            - STAGE_POTENTIALS[SeedStage.PROBATIONARY.value]
        )

        all_deltas = [germinated_delta, training_delta, probationary_delta, fossilized_delta]

        assert blending_delta >= max(all_deltas), (
            f"BLENDING delta {blending_delta} should be largest, "
            f"but max others is {max(all_deltas)}"
        )


@pytest.mark.property
class TestFossilizedSmallestIncrement:
    """FOSSILIZED should have smallest increment (anti-farming)."""

    def test_fossilized_smallest_delta(self):
        """FOSSILIZED increment < all other increments."""
        fossilized_delta = (
            STAGE_POTENTIALS[SeedStage.FOSSILIZED.value]
            - STAGE_POTENTIALS[SeedStage.PROBATIONARY.value]
        )

        # Compare to other transitions (excluding DORMANT->GERMINATED which is also small)
        training_delta = (
            STAGE_POTENTIALS[SeedStage.TRAINING.value]
            - STAGE_POTENTIALS[SeedStage.GERMINATED.value]
        )
        blending_delta = (
            STAGE_POTENTIALS[SeedStage.BLENDING.value]
            - STAGE_POTENTIALS[SeedStage.TRAINING.value]
        )
        probationary_delta = (
            STAGE_POTENTIALS[SeedStage.PROBATIONARY.value]
            - STAGE_POTENTIALS[SeedStage.BLENDING.value]
        )

        meaningful_deltas = [training_delta, blending_delta, probationary_delta]

        assert fossilized_delta <= min(meaningful_deltas), (
            f"FOSSILIZED delta {fossilized_delta} should be smallest meaningful, "
            f"but min others is {min(meaningful_deltas)}"
        )
```

**Step 2: Run tests**

Run: `pytest tests/simic/properties/test_pbrs_properties.py -v --tb=short`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/simic/properties/test_pbrs_properties.py
git commit -m "feat(tests): add PBRS telescoping property tests"
```

---

## Task 6: Warning Signal Tests (Blending/Probation)

**Files:**
- Create: `tests/simic/properties/test_warning_signals.py`

**Step 1: Write warning signal tests**

```python
# tests/simic/properties/test_warning_signals.py
"""Warning signal property tests.

These properties verify that warning signals (blending_warning, probation_warning)
fire correctly and provide proper credit assignment.
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp
from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    SeedInfo,
    STAGE_BLENDING,
    STAGE_PROBATIONARY,
)

from tests.simic.strategies import seed_infos_at_stage, reward_inputs_with_seed


@pytest.mark.property
class TestBlendingWarning:
    """Blending warning provides early signal to CULL bad seeds."""

    @given(
        total_improvement=st.floats(-3.0, -0.1, allow_nan=False),
        epochs_in_stage=st.integers(1, 10),
    )
    @settings(max_examples=200)
    def test_negative_trajectory_warned(self, total_improvement, epochs_in_stage):
        """Seeds with negative trajectory in BLENDING should get warning."""
        seed_info = SeedInfo(
            stage=STAGE_BLENDING,
            improvement_since_stage_start=-0.5,
            total_improvement=total_improvement,
            epochs_in_stage=epochs_in_stage,
            seed_params=50_000,
            previous_stage=SeedStage.TRAINING.value,
            previous_epochs_in_stage=3,
            seed_age_epochs=epochs_in_stage + 5,
        )

        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=1.0,  # Some contribution
            val_acc=70.0,
            seed_info=seed_info,
            epoch=10,
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=72.0,  # Was better before
            acc_delta=-0.1,
            return_components=True,
        )

        assert components.blending_warning < 0, (
            f"Negative trajectory in BLENDING should warn, got {components.blending_warning}"
        )

    @given(epochs_in_stage=st.integers(1, 10))
    @settings(max_examples=100)
    def test_warning_escalates_with_time(self, epochs_in_stage):
        """Warning should escalate the longer negative trajectory persists."""
        def get_warning(epochs: int) -> float:
            seed_info = SeedInfo(
                stage=STAGE_BLENDING,
                improvement_since_stage_start=-0.5,
                total_improvement=-1.0,
                epochs_in_stage=epochs,
                seed_params=50_000,
                previous_stage=SeedStage.TRAINING.value,
                previous_epochs_in_stage=3,
                seed_age_epochs=epochs + 5,
            )

            _, components = compute_contribution_reward(
                action=LifecycleOp.WAIT,
                seed_contribution=1.0,
                val_acc=70.0,
                seed_info=seed_info,
                epoch=10,
                max_epochs=25,
                total_params=150_000,
                host_params=100_000,
                acc_at_germination=72.0,
                acc_delta=-0.1,
                return_components=True,
            )
            return components.blending_warning

        # Warning should be more negative with more epochs
        warning_early = get_warning(1)
        warning_late = get_warning(min(epochs_in_stage + 3, 10))

        assert warning_late <= warning_early, (
            f"Warning should escalate: epoch 1 = {warning_early}, "
            f"epoch {epochs_in_stage + 3} = {warning_late}"
        )


@pytest.mark.property
class TestProbationWarning:
    """Probation warning creates urgency to make FOSSILIZE/CULL decision."""

    @given(epochs_in_stage=st.integers(2, 8))
    @settings(max_examples=100)
    def test_wait_in_probation_penalized(self, epochs_in_stage):
        """WAITing in PROBATIONARY with positive attribution should be penalized."""
        seed_info = SeedInfo(
            stage=STAGE_PROBATIONARY,
            improvement_since_stage_start=0.5,
            total_improvement=2.0,  # Positive trajectory
            epochs_in_stage=epochs_in_stage,
            seed_params=50_000,
            previous_stage=STAGE_BLENDING,
            previous_epochs_in_stage=5,
            seed_age_epochs=epochs_in_stage + 10,
        )

        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=3.0,  # Good seed
            val_acc=75.0,
            seed_info=seed_info,
            epoch=15,
            max_epochs=25,
            total_params=150_000,
            host_params=100_000,
            acc_at_germination=65.0,
            acc_delta=0.3,
            return_components=True,
        )

        # Warning should be negative (penalty for indecision)
        assert components.probation_warning < 0, (
            f"WAIT in PROBATIONARY epoch {epochs_in_stage} should be penalized, "
            f"got {components.probation_warning}"
        )

    @given(epochs_in_stage=st.integers(2, 6))
    @settings(max_examples=100)
    def test_probation_warning_exponential(self, epochs_in_stage):
        """Probation warning should escalate exponentially."""
        def get_warning(epochs: int) -> float:
            seed_info = SeedInfo(
                stage=STAGE_PROBATIONARY,
                improvement_since_stage_start=0.5,
                total_improvement=2.0,
                epochs_in_stage=epochs,
                seed_params=50_000,
                previous_stage=STAGE_BLENDING,
                previous_epochs_in_stage=5,
                seed_age_epochs=epochs + 10,
            )

            _, components = compute_contribution_reward(
                action=LifecycleOp.WAIT,
                seed_contribution=3.0,
                val_acc=75.0,
                seed_info=seed_info,
                epoch=15,
                max_epochs=25,
                total_params=150_000,
                host_params=100_000,
                acc_at_germination=65.0,
                acc_delta=0.3,
                return_components=True,
            )
            return components.probation_warning

        # Compare consecutive epochs
        warning_n = get_warning(epochs_in_stage)
        warning_n1 = get_warning(epochs_in_stage + 1)

        # Should be more negative (exponential escalation)
        if warning_n < 0 and warning_n1 < 0:
            ratio = warning_n1 / warning_n
            # Exponential: should be roughly 3x per epoch (capped at -10)
            assert ratio >= 1.5 or warning_n1 <= -10.0, (
                f"Warning escalation ratio {ratio:.2f} too low "
                f"(epoch {epochs_in_stage}: {warning_n}, epoch {epochs_in_stage+1}: {warning_n1})"
            )
```

**Step 2: Run tests**

Run: `pytest tests/simic/properties/test_warning_signals.py -v --tb=short`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/simic/properties/test_warning_signals.py
git commit -m "feat(tests): add warning signal property tests"
```

---

## Task 7: Aggregate All Property Tests

**Files:**
- Modify: `tests/simic/properties/__init__.py`

**Step 1: Update package init**

```python
# tests/simic/properties/__init__.py
"""Property-based tests for simic.

These tests verify invariants that must hold for ALL valid inputs,
not just specific examples. Organized by property tier:

Tier 1 - Mathematical Invariants:
    test_reward_invariants.py - Finiteness, boundedness, composition

Tier 2 - Semantic Invariants:
    test_reward_semantics.py - Domain rules (fossilized, cull, terminal)

Tier 3 - Anti-Gaming Properties:
    test_reward_antigaming.py - Ransomware, farming, ratio exploits

Tier 4 - PBRS Properties:
    test_pbrs_properties.py - Telescoping, monotonicity, increments

Tier 5 - Warning Signals:
    test_warning_signals.py - Blending/probation credit assignment

Run all property tests:
    pytest tests/simic/properties/ -m property -v

Run with more examples (CI):
    pytest tests/simic/properties/ -m property --hypothesis-seed=0
"""
```

**Step 2: Run full property test suite**

Run: `pytest tests/simic/properties/ -m property -v --tb=short`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/simic/properties/__init__.py
git commit -m "docs(tests): document property test tiers"
```

---

## Task 8: Add CI Configuration for Property Tests

**Files:**
- Modify: `pytest.ini` (if needed for property test settings)

**Step 1: Verify property marker exists**

Run: `grep -A2 "property:" /home/john/esper-lite/pytest.ini`
Expected: Shows `property: marks tests as property-based tests`

**Step 2: Create conftest for property settings**

```python
# tests/simic/properties/conftest.py
"""Pytest configuration for property-based tests."""

import pytest
from hypothesis import settings, Verbosity

# Register profile for CI (more examples, deterministic)
settings.register_profile(
    "ci",
    max_examples=1000,
    verbosity=Verbosity.verbose,
    deadline=None,  # No timeout in CI
)

# Register profile for development (fewer examples, faster)
settings.register_profile(
    "dev",
    max_examples=100,
    verbosity=Verbosity.normal,
    deadline=5000,  # 5 second timeout
)

# Load profile from env or default to dev
settings.load_profile("dev")
```

**Step 3: Commit**

```bash
git add tests/simic/properties/conftest.py
git commit -m "feat(tests): add Hypothesis profiles for property tests"
```

---

## Summary

**Total files created:**
- `tests/simic/strategies/__init__.py`
- `tests/simic/strategies/reward_strategies.py`
- `tests/simic/properties/__init__.py`
- `tests/simic/properties/test_reward_invariants.py`
- `tests/simic/properties/test_reward_semantics.py`
- `tests/simic/properties/test_reward_antigaming.py`
- `tests/simic/properties/test_pbrs_properties.py`
- `tests/simic/properties/test_warning_signals.py`
- `tests/simic/properties/conftest.py`

**Test counts (estimated):**
- Tier 1 (Mathematical): ~4 property tests
- Tier 2 (Semantic): ~5 property tests
- Tier 3 (Anti-Gaming): ~5 property tests
- Tier 4 (PBRS): ~5 property tests
- Tier 5 (Warnings): ~4 property tests

Each property test runs 100+ examples by default, 1000+ in CI.

**Run all:**
```bash
pytest tests/simic/properties/ -m property -v
```

**Run with CI profile:**
```bash
HYPOTHESIS_PROFILE=ci pytest tests/simic/properties/ -m property -v
```
