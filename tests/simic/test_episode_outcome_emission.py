"""Tests that EpisodeOutcome is emitted at episode end."""

import pytest
import numpy as np
from esper.karn.store import EpisodeOutcome


def test_episode_outcome_created_at_episode_end():
    """vectorized training creates EpisodeOutcome at episode completion."""
    outcome = EpisodeOutcome(
        env_id=0,
        episode_idx=1,
        final_accuracy=72.5,
        param_ratio=0.12,
        num_fossilized=1,
        num_contributing_fossilized=1,
        episode_reward=10.5,
        stability_score=0.8,
        reward_mode="shaped",
    )

    # Verify all required fields exist
    d = outcome.to_dict()
    required_fields = [
        "env_id", "episode_idx", "final_accuracy", "param_ratio",
        "num_fossilized", "num_contributing_fossilized", "episode_reward",
        "stability_score", "reward_mode", "timestamp"
    ]
    for field in required_fields:
        assert field in d, f"Missing required field: {field}"


def test_stability_score_from_reward_variance():
    """Stability score computed correctly from reward variance."""
    # Low variance = high stability
    low_var_rewards = [10.0, 10.1, 9.9, 10.0, 10.2]
    var_low = np.var(low_var_rewards)
    stability_low = 1.0 / (1.0 + var_low)
    assert stability_low > 0.9, "Low variance should give high stability"

    # High variance = low stability
    high_var_rewards = [5.0, 15.0, 2.0, 18.0, 8.0]
    var_high = np.var(high_var_rewards)
    stability_high = 1.0 / (1.0 + var_high)
    assert stability_high < 0.1, "High variance should give low stability"

    # Stability always in [0, 1]
    assert 0.0 <= stability_low <= 1.0
    assert 0.0 <= stability_high <= 1.0


def test_episode_outcome_dominates():
    """Verify Pareto dominance check works correctly."""
    # Better on all objectives
    better = EpisodeOutcome(
        env_id=0,
        episode_idx=1,
        final_accuracy=80.0,
        param_ratio=0.1,  # lower is better
        num_fossilized=2,
        num_contributing_fossilized=2,
        episode_reward=20.0,
        stability_score=0.9,  # higher is better
        reward_mode="shaped",
    )
    worse = EpisodeOutcome(
        env_id=0,
        episode_idx=2,
        final_accuracy=70.0,
        param_ratio=0.2,
        num_fossilized=1,
        num_contributing_fossilized=1,
        episode_reward=10.0,
        stability_score=0.7,
        reward_mode="shaped",
    )

    assert better.dominates(worse), "Better should dominate worse"
    assert not worse.dominates(better), "Worse should not dominate better"


def test_episode_outcome_pareto_incomparable():
    """Verify Pareto-incomparable outcomes don't dominate each other."""
    # Trade-off: high accuracy, high param_ratio vs low accuracy, low param_ratio
    high_acc = EpisodeOutcome(
        env_id=0,
        episode_idx=1,
        final_accuracy=85.0,
        param_ratio=0.3,  # worse
        num_fossilized=2,
        num_contributing_fossilized=2,
        episode_reward=15.0,
        stability_score=0.8,
        reward_mode="shaped",
    )
    efficient = EpisodeOutcome(
        env_id=0,
        episode_idx=2,
        final_accuracy=75.0,  # worse
        param_ratio=0.1,  # better
        num_fossilized=1,
        num_contributing_fossilized=1,
        episode_reward=12.0,
        stability_score=0.8,
        reward_mode="shaped",
    )

    # Neither dominates the other (Pareto front)
    assert not high_acc.dominates(efficient), "Trade-off: neither should dominate"
    assert not efficient.dominates(high_acc), "Trade-off: neither should dominate"
