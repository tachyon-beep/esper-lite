"""Tests for TrainingConfig."""

import pytest

from esper.simic.config import TrainingConfig


def test_default_gamma_gae_lambda():
    """Default gamma/gae_lambda should be optimized for 25-epoch episodes."""
    config = TrainingConfig()

    # Long-horizon credit assignment (gamma^25 ~ 0.88)
    assert config.gamma == 0.995
    assert config.gae_lambda == 0.97


def test_to_ppo_kwargs():
    """to_ppo_kwargs should include vectorized training parameters."""
    config = TrainingConfig()
    ppo_kwargs = config.to_ppo_kwargs()

    assert ppo_kwargs.get("num_envs") == config.n_envs
    assert ppo_kwargs.get("max_steps_per_env") == config.max_epochs
    assert ppo_kwargs.get("gamma") == 0.995
    assert ppo_kwargs.get("gae_lambda") == 0.97
