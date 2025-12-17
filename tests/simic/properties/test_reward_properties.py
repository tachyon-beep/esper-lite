"""Property-based tests for reward computation.

Tests mathematical invariants that must hold for ALL inputs:
- Bounds: rewards must be bounded
- Monotonicity: better performance → better reward
- Consistency: same input → same output
- Conservation: potential-based shaping preserves optimal policy
"""

import math
from hypothesis import given, assume, settings
from hypothesis import strategies as st
from tests.strategies import (
    bounded_floats,
    accuracies,
    action_members,
    seed_infos,
    seed_stages,
)

from esper.simic.rewards import (
    compute_contribution_reward,
    compute_potential,
    compute_pbrs_bonus,
    compute_seed_potential,
    get_intervention_cost,
)


class TestRewardBounds:
    """Test that rewards are bounded for all inputs."""

    @given(
        action=action_members(),
        acc_delta=bounded_floats(-10.0, 10.0),
        val_acc=accuracies(),
        seed_info=st.one_of(seed_infos(), st.none()),
        epoch=st.integers(0, 1000),
        max_epochs=st.integers(1, 1000),
    )
    @settings(max_examples=200)
    def test_reward_bounded(self, action, acc_delta, val_acc, seed_info, epoch, max_epochs):
        """Property: Reward must be bounded regardless of inputs.

        This prevents reward explosion that destabilizes training.
        """
        assume(epoch <= max_epochs)  # Invariant: epoch <= max_epochs

        reward = compute_contribution_reward(
            action=action,
            seed_contribution=None,  # Use proxy signal path
            val_acc=val_acc,
            seed_info=seed_info,
            epoch=epoch,
            max_epochs=max_epochs,
            acc_delta=acc_delta,
        )

        # Reward should be bounded (conservative bounds for safety)
        # PBRS bonuses can add headroom above legacy limits; keep a generous cap.
        assert -40.0 < reward < 40.0, f"Reward {reward} out of bounds"

    @given(
        action=action_members(),
        acc_delta=bounded_floats(-10.0, 10.0),
        val_acc=accuracies(),
        epoch=st.integers(0, 100),
        max_epochs=st.integers(1, 100),
    )
    def test_reward_finite(self, action, acc_delta, val_acc, epoch, max_epochs):
        """Property: Reward must be finite (no NaN, no Inf).

        NaN/Inf values crash gradient descent.
        """
        assume(epoch <= max_epochs)

        reward = compute_contribution_reward(
            action=action,
            seed_contribution=None,
            val_acc=val_acc,
            seed_info=None,
            epoch=epoch,
            max_epochs=max_epochs,
            acc_delta=acc_delta,
        )

        assert not math.isnan(reward), "Reward is NaN"
        assert not math.isinf(reward), "Reward is Inf"


class TestRewardMonotonicity:
    """Test that better performance → better reward (proxy signal path)."""

    delta_pairs = bounded_floats(0.01, 4.9).flatmap(
        lambda d1: st.tuples(st.just(d1), bounded_floats(min_value=d1 + 0.01, max_value=5.0))
    )

    @given(
        action=action_members(),
        deltas=delta_pairs,
        val_acc=accuracies(),
        epoch=st.integers(0, 100),
        max_epochs=st.integers(1, 100),
    )
    def test_higher_acc_delta_better_reward(self, action, deltas, val_acc, epoch, max_epochs):
        """Property: Higher accuracy improvement → higher reward.

        This is the core signal for the RL agent (proxy signal path).
        Note: Only positive deltas are rewarded in proxy path.
        """
        assume(epoch <= max_epochs)
        acc_delta1, acc_delta2 = deltas

        r1 = compute_contribution_reward(
            action, None, val_acc, None, epoch, max_epochs, acc_delta=acc_delta1
        )
        r2 = compute_contribution_reward(
            action, None, val_acc, None, epoch, max_epochs, acc_delta=acc_delta2
        )

        # Higher accuracy improvement should give higher reward
        # (allow small epsilon for floating point)
        assert r2 >= r1 - 1e-6, f"acc_delta {acc_delta2} should give >= reward than {acc_delta1}"


class TestPotentialBasedShaping:
    """Test that potential-based reward shaping preserves optimal policy."""

    @given(
        val_acc=accuracies(),
        epoch=st.integers(0, 100),
        max_epochs=st.integers(1, 100),
    )
    def test_potential_bounded(self, val_acc, epoch, max_epochs):
        """Property: Potential function must be bounded."""
        assume(epoch <= max_epochs)

        phi = compute_potential(val_acc, epoch, max_epochs)

        # Potential should be bounded
        assert -100.0 < phi < 100.0

    @given(
        phi_prev=bounded_floats(0.0, 10.0),
        phi_next=bounded_floats(0.0, 10.0),
        gamma=bounded_floats(0.9, 1.0),
    )
    def test_pbrs_bonus_bounded(self, phi_prev, phi_next, gamma):
        """Property: PBRS bonus must be bounded."""
        bonus = compute_pbrs_bonus(phi_prev, phi_next, gamma)

        # Bonus = gamma * phi_next - phi_prev
        # Given phi in [0, 10], bonus in [-10, 10]
        assert -15.0 < bonus < 15.0


class TestInterventionCosts:
    """Test intervention costs are consistent."""

    @given(action=action_members())
    def test_intervention_cost_non_positive(self, action):
        """Property: Intervention costs should be <= 0 (discourage unnecessary actions)."""
        cost = get_intervention_cost(action)

        assert cost <= 0.0, "Intervention cost should be non-positive"

    def test_wait_has_zero_cost(self):
        """Property: WAIT action should have zero cost."""
        from enum import IntEnum
        TestAction = IntEnum("TestAction", {"WAIT": 0, "FOSSILIZE": 1, "CULL": 2})
        cost = get_intervention_cost(TestAction.WAIT)

        assert cost == 0.0


class TestSeedPotential:
    """Test seed potential computation."""

    @given(
        has_active=st.booleans(),
        seed_stage=seed_stages(),
        epochs_in_stage=st.integers(0, 100),
    )
    def test_seed_potential_non_negative(self, has_active, seed_stage, epochs_in_stage):
        """Property: Seed potential must be >= 0."""
        obs = {
            'has_active_seed': int(has_active),
            'seed_stage': seed_stage,
            'seed_epochs_in_stage': epochs_in_stage,
        }

        potential = compute_seed_potential(obs)

        assert potential >= 0.0, "Seed potential must be non-negative"

    def test_no_seed_zero_potential(self):
        """Property: No active seed → zero potential."""
        obs = {'has_active_seed': 0, 'seed_stage': 0, 'seed_epochs_in_stage': 0}

        potential = compute_seed_potential(obs)

        assert potential == 0.0
