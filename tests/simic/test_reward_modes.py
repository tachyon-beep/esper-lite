"""Tests for reward mode enum and sparse reward functions."""

import pytest
from esper.simic.rewards import RewardMode, ContributionRewardConfig


def test_reward_mode_enum_exists():
    """RewardMode enum has three modes."""
    assert RewardMode.SHAPED.value == "shaped"
    assert RewardMode.SPARSE.value == "sparse"
    assert RewardMode.MINIMAL.value == "minimal"


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


def test_reward_mode_exported():
    """RewardMode is in module __all__."""
    from esper.simic import rewards
    assert "RewardMode" in rewards.__all__
