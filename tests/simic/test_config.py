"""Tests for TrainingConfig."""

import inspect

from esper.simic.config import TrainingConfig
from esper.simic.vectorized import train_ppo_vectorized


def test_default_gamma_gae_lambda():
    """Default gamma should be optimized for 25-epoch episodes."""
    config = TrainingConfig()

    assert config.gamma == 0.995


def test_to_ppo_kwargs():
    """to_ppo_kwargs should include vectorized training parameters."""
    config = TrainingConfig(entropy_anneal_episodes=8, n_envs=4, ppo_updates_per_batch=2)
    ppo_kwargs = config.to_ppo_kwargs()

    assert ppo_kwargs.get("num_envs") == config.n_envs
    assert ppo_kwargs.get("max_steps_per_env") == config.max_epochs
    assert ppo_kwargs.get("gamma") == 0.995
    assert ppo_kwargs.get("entropy_anneal_steps") == 4


def test_to_train_kwargs_is_subset_of_vectorized_signature():
    """Config → train kwargs must stay in sync with train_ppo_vectorized signature."""
    signature = inspect.signature(train_ppo_vectorized)
    allowed = set(signature.parameters)

    config = TrainingConfig()
    kwargs = set(config.to_train_kwargs())

    assert kwargs <= allowed
