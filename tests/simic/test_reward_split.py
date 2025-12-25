"""Tests for TrainingConfig.with_reward_split() factory method."""

import pytest

from esper.simic.rewards import RewardMode
from esper.simic.training.config import TrainingConfig


class TestWithRewardSplit:
    """Test the with_reward_split() factory method."""

    def test_with_reward_split_basic(self):
        """Basic 2-mode split cycles correctly."""
        cfg = TrainingConfig.with_reward_split(4, [RewardMode.SHAPED, RewardMode.SIMPLIFIED])
        assert cfg.reward_mode_per_env == (
            RewardMode.SHAPED, RewardMode.SIMPLIFIED,
            RewardMode.SHAPED, RewardMode.SIMPLIFIED,
        )

    def test_with_reward_split_single_mode(self):
        """Single mode fills all environments."""
        cfg = TrainingConfig.with_reward_split(3, [RewardMode.SHAPED])
        assert cfg.reward_mode_per_env == (RewardMode.SHAPED,) * 3

    def test_with_reward_split_empty_raises(self):
        """Empty modes list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            TrainingConfig.with_reward_split(4, [])

    def test_with_reward_split_three_modes(self):
        """3-mode A/B/C split cycles correctly."""
        cfg = TrainingConfig.with_reward_split(
            6, [RewardMode.SHAPED, RewardMode.SIMPLIFIED, RewardMode.SPARSE]
        )
        assert cfg.reward_mode_per_env == (
            RewardMode.SHAPED, RewardMode.SIMPLIFIED, RewardMode.SPARSE,
            RewardMode.SHAPED, RewardMode.SIMPLIFIED, RewardMode.SPARSE,
        )

    def test_with_reward_split_uneven_distribution(self):
        """Modes cycle when num_envs is not evenly divisible."""
        cfg = TrainingConfig.with_reward_split(5, [RewardMode.SHAPED, RewardMode.SIMPLIFIED])
        assert cfg.reward_mode_per_env == (
            RewardMode.SHAPED, RewardMode.SIMPLIFIED,
            RewardMode.SHAPED, RewardMode.SIMPLIFIED,
            RewardMode.SHAPED,
        )

    def test_with_reward_split_sets_n_envs(self):
        """Factory method also sets n_envs correctly."""
        cfg = TrainingConfig.with_reward_split(8, [RewardMode.SPARSE])
        assert cfg.n_envs == 8
        assert len(cfg.reward_mode_per_env) == 8

    def test_with_reward_split_tuple_input(self):
        """Factory accepts tuple as modes input."""
        cfg = TrainingConfig.with_reward_split(
            4, (RewardMode.SHAPED, RewardMode.SIMPLIFIED)
        )
        assert cfg.reward_mode_per_env == (
            RewardMode.SHAPED, RewardMode.SIMPLIFIED,
            RewardMode.SHAPED, RewardMode.SIMPLIFIED,
        )
