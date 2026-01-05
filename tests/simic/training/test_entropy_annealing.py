"""Tests for entropy annealing calculation and semantics.

Verifies that:
1. _calculate_entropy_anneal_steps correctly converts env-episodes to PPO update steps
2. The conversion is consistent with CLI documentation
3. Edge cases (zero, single env) are handled correctly
"""

import math

import pytest

from esper.simic.training.vectorized import _calculate_entropy_anneal_steps


class TestEntropyAnnealStepsCalculation:
    """Test the _calculate_entropy_anneal_steps conversion function."""

    def test_basic_conversion(self):
        """100 episodes / 4 envs = 25 batches * 1 update = 25 steps."""
        result = _calculate_entropy_anneal_steps(
            entropy_anneal_episodes=100,
            n_envs=4,
            ppo_updates_per_batch=1,
        )
        assert result == 25

    def test_conversion_with_multiple_updates_per_batch(self):
        """100 episodes / 4 envs = 25 batches * 2 updates = 50 steps."""
        result = _calculate_entropy_anneal_steps(
            entropy_anneal_episodes=100,
            n_envs=4,
            ppo_updates_per_batch=2,
        )
        assert result == 50

    def test_single_env_equals_episodes(self):
        """With 1 env, batches = episodes."""
        result = _calculate_entropy_anneal_steps(
            entropy_anneal_episodes=100,
            n_envs=1,
            ppo_updates_per_batch=1,
        )
        assert result == 100

    def test_zero_episodes_returns_zero(self):
        """0 episodes = disabled annealing."""
        result = _calculate_entropy_anneal_steps(
            entropy_anneal_episodes=0,
            n_envs=4,
            ppo_updates_per_batch=1,
        )
        assert result == 0

    def test_negative_episodes_returns_zero(self):
        """Negative episodes treated as disabled."""
        result = _calculate_entropy_anneal_steps(
            entropy_anneal_episodes=-10,
            n_envs=4,
            ppo_updates_per_batch=1,
        )
        assert result == 0

    def test_ceiling_division_rounds_up(self):
        """Non-divisible episodes should round up (ceil)."""
        # 10 episodes / 3 envs = ceil(3.33) = 4 batches
        result = _calculate_entropy_anneal_steps(
            entropy_anneal_episodes=10,
            n_envs=3,
            ppo_updates_per_batch=1,
        )
        assert result == 4
        assert result == math.ceil(10 / 3)

    def test_invalid_n_envs_raises(self):
        """n_envs must be positive when annealing is enabled."""
        with pytest.raises(ValueError, match="n_envs must be positive"):
            _calculate_entropy_anneal_steps(
                entropy_anneal_episodes=100,
                n_envs=0,
                ppo_updates_per_batch=1,
            )

    def test_zero_updates_per_batch_defaults_to_one(self):
        """ppo_updates_per_batch=0 should be treated as 1."""
        result = _calculate_entropy_anneal_steps(
            entropy_anneal_episodes=100,
            n_envs=4,
            ppo_updates_per_batch=0,
        )
        # 100/4 = 25 batches * max(1, 0) = 25 steps
        assert result == 25


class TestEntropyAnnealingSemanticsDocumentation:
    """Tests that document and verify the semantic meaning of entropy annealing.

    These tests serve as executable documentation of the intended behavior.
    """

    def test_same_total_episodes_different_parallelism(self):
        """Same --entropy-anneal-episodes with different --envs produces same total experience.

        This documents that the semantics are:
        - "Total env-episodes" means the total across all envs
        - More envs = fewer batches, but same total experience
        """
        # 100 total env-episodes
        steps_4_envs = _calculate_entropy_anneal_steps(100, n_envs=4, ppo_updates_per_batch=1)
        steps_8_envs = _calculate_entropy_anneal_steps(100, n_envs=8, ppo_updates_per_batch=1)
        steps_1_env = _calculate_entropy_anneal_steps(100, n_envs=1, ppo_updates_per_batch=1)

        # Different number of PPO updates...
        assert steps_4_envs == 25
        assert steps_8_envs == 13  # ceil(100/8) = 13
        assert steps_1_env == 100

        # ...but approximately same total env-episodes
        # (exact for 4 envs, slightly more for 8 due to ceiling)
        total_episodes_4 = steps_4_envs * 4  # 25 * 4 = 100
        total_episodes_8 = steps_8_envs * 8  # 13 * 8 = 104 (ceil overshoot)
        total_episodes_1 = steps_1_env * 1  # 100 * 1 = 100

        assert total_episodes_4 == 100
        assert total_episodes_8 >= 100  # Ceiling rounds up
        assert total_episodes_1 == 100

    def test_cli_help_example_is_accurate(self):
        """Verify the CLI help text example: 'With K envs, produces ceil(N/K) PPO batches'.

        CLI help says:
        --entropy-anneal-episodes N
            Total env-episodes over which to anneal entropy coefficient.
            With K envs, this produces ceil(N/K) PPO batches of annealing.
        """
        N = 100
        K = 4

        result = _calculate_entropy_anneal_steps(
            entropy_anneal_episodes=N,
            n_envs=K,
            ppo_updates_per_batch=1,
        )

        expected_batches = math.ceil(N / K)
        assert result == expected_batches, (
            f"CLI help says ceil({N}/{K}) = {expected_batches} batches, "
            f"but got {result}"
        )
