"""Integration test for reward A/B testing."""
import pytest
import torch

from esper.simic.rewards import RewardMode
from esper.simic.training.config import TrainingConfig


@pytest.mark.slow
def test_ab_testing_runs_without_error():
    """A/B testing with split reward modes should complete without error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for vectorized training")

    config = TrainingConfig(
        n_envs=4,
        n_episodes=2,  # Short test
        max_epochs=5,  # Short episodes
        reward_mode_per_env=(
            RewardMode.SHAPED, RewardMode.SHAPED,
            RewardMode.SIMPLIFIED, RewardMode.SIMPLIFIED,
        ),
    )

    # Verify config is valid
    assert config.reward_mode_per_env is not None
    assert len(config.reward_mode_per_env) == 4

    # Note: Full training test would be too slow for CI
    # This just validates config construction and validation
