# tests/simic/properties/test_sparse_properties.py
"""Property-based tests for sparse reward invariants.

DRL Expert Review: These tests verify critical invariants for sparse rewards.
"""

import math
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

        # H10 FIX: Base reward clamped to [-1, 1] BEFORE scaling
        # base = 1.0 (already in [-1, 1]), scaled = 3.0 * 1.0 = 3.0
        # Final reward in [-scale, scale] = [-3.0, 3.0]
        assert reward == 3.0, f"Scale should be effective, got {reward}"


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
