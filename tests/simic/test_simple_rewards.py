"""Tests for simplified multi-slot reward function.

Design principle: contribution good, bloat bad, that's it.
Let the agent learn everything else.
"""
import math
import pytest


def test_positive_contribution_gives_positive_reward():
    """Seeds that help should get positive reward."""
    from esper.simic.simple_rewards import compute_simple_reward

    reward = compute_simple_reward(
        seed_contributions={"mid": 5.0},  # 5% improvement from seed
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )

    assert reward > 0, "Positive contribution should give positive reward"


def test_negative_contribution_gives_negative_reward():
    """Seeds that hurt should get negative reward."""
    from esper.simic.simple_rewards import compute_simple_reward

    reward = compute_simple_reward(
        seed_contributions={"mid": -3.0},  # Seed hurts by 3%
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )

    assert reward < 0, "Negative contribution should give negative reward"


def test_bloat_is_penalized():
    """More parameters = more penalty."""
    from esper.simic.simple_rewards import compute_simple_reward

    # Same contribution, different sizes
    reward_small = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,  # No bloat
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )

    reward_bloated = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=200_000,  # 2x bloat
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )

    assert reward_small > reward_bloated, "Bloat should reduce reward"


def test_multiple_slots_sum_contributions():
    """Multiple active slots should sum their contributions."""
    from esper.simic.simple_rewards import compute_simple_reward

    reward_single = compute_simple_reward(
        seed_contributions={"mid": 3.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )

    reward_multi = compute_simple_reward(
        seed_contributions={"early": 1.0, "mid": 1.0, "late": 1.0},  # Same total
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )

    assert abs(reward_single - reward_multi) < 0.01, "Same total contribution = same reward"


def test_terminal_bonus_for_accuracy():
    """Terminal state should add accuracy bonus."""
    from esper.simic.simple_rewards import compute_simple_reward

    reward_mid = compute_simple_reward(
        seed_contributions={},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=80.0,
    )

    reward_terminal = compute_simple_reward(
        seed_contributions={},
        total_params=100_000,
        host_params=100_000,
        is_terminal=True,
        val_acc=80.0,
    )

    assert reward_terminal > reward_mid, "Terminal should have accuracy bonus"


def test_none_contributions_ignored():
    """None contributions (no counterfactual available) should be ignored."""
    from esper.simic.simple_rewards import compute_simple_reward

    reward = compute_simple_reward(
        seed_contributions={"early": None, "mid": 2.0, "late": None},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )

    # Should only count the mid slot's 2.0 contribution
    reward_explicit = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
    )

    assert abs(reward - reward_explicit) < 0.01, "None contributions should be ignored"


def test_zero_params_no_crash():
    """Should handle edge case of zero params gracefully."""
    from esper.simic.simple_rewards import compute_simple_reward

    # Should not crash
    reward = compute_simple_reward(
        seed_contributions={"mid": 1.0},
        total_params=0,
        host_params=0,
        is_terminal=False,
        val_acc=70.0,
    )

    assert reward == 1.0, "Zero params should give no rent penalty"


def test_no_budget_pressure_when_not_germinating():
    """Budget pressure should only apply on GERMINATE actions."""
    from esper.simic.simple_rewards import compute_simple_reward

    # Non-GERMINATE action with high utilization - no penalty
    reward_no_germinate = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
        seed_utilization=0.8,  # High utilization
        action_is_germinate=False,  # Not germinating
    )

    # GERMINATE action with zero utilization - no penalty
    reward_zero_util = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
        seed_utilization=0.0,  # Zero utilization
        action_is_germinate=True,  # Germinating
    )

    # Should be equal (no budget pressure in either case)
    assert abs(reward_no_germinate - reward_zero_util) < 0.01, "No penalty without germinate or with zero utilization"


def test_budget_pressure_increases_quadratically():
    """Budget pressure should increase quadratically with utilization."""
    from esper.simic.simple_rewards import compute_simple_reward

    # GERMINATE at low utilization
    reward_low = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
        seed_utilization=0.2,  # 20% full
        action_is_germinate=True,
    )

    # GERMINATE at medium utilization
    reward_mid = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
        seed_utilization=0.5,  # 50% full
        action_is_germinate=True,
    )

    # GERMINATE at high utilization
    reward_high = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
        seed_utilization=0.9,  # 90% full
        action_is_germinate=True,
    )

    # Rewards should decrease as utilization increases
    assert reward_low > reward_mid, "Higher utilization should give lower reward"
    assert reward_mid > reward_high, "Highest utilization should give lowest reward"

    # Check quadratic behavior: penalty difference should be larger at high utilization
    penalty_low = reward_low - reward_mid
    penalty_high = reward_mid - reward_high
    assert penalty_high > penalty_low, "Quadratic: penalty should increase faster at high utilization"


def test_budget_pressure_scale_parameter():
    """budget_pressure_scale should control penalty magnitude."""
    from esper.simic.simple_rewards import compute_simple_reward

    # Default scale
    reward_default = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
        seed_utilization=0.5,
        action_is_germinate=True,
        budget_pressure_scale=0.1,  # Default
    )

    # Higher scale (more penalty)
    reward_high_scale = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
        seed_utilization=0.5,
        action_is_germinate=True,
        budget_pressure_scale=0.5,  # 5x larger
    )

    # Zero scale (no penalty)
    reward_zero_scale = compute_simple_reward(
        seed_contributions={"mid": 2.0},
        total_params=100_000,
        host_params=100_000,
        is_terminal=False,
        val_acc=70.0,
        seed_utilization=0.5,
        action_is_germinate=True,
        budget_pressure_scale=0.0,  # No penalty
    )

    assert reward_high_scale < reward_default, "Higher scale should give more penalty"
    assert reward_zero_scale > reward_default, "Zero scale should give no penalty"
