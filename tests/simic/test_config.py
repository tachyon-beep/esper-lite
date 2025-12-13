"""Tests for TrainingConfig."""

import pytest

from esper.simic.config import TrainingConfig


def test_tamiyo_preset():
    """for_tamiyo preset should have long-horizon hyperparameters."""
    config = TrainingConfig.for_tamiyo()

    # Long-horizon credit assignment
    assert config.gamma == 0.995
    assert config.gae_lambda == 0.97

    # Tamiyo mode enabled
    assert config.tamiyo is True
    assert config.recurrent is False

    # Entropy schedule
    assert config.entropy_coef_start == 0.05
    assert config.entropy_coef_end == 0.005


def test_tamiyo_preset_num_envs():
    """for_tamiyo preset should configure parallel environments."""
    config = TrainingConfig.for_tamiyo()

    assert config.n_envs == 4
    assert config.max_epochs == 25


def test_tamiyo_preset_lstm_config():
    """for_tamiyo preset should configure LSTM parameters."""
    config = TrainingConfig.for_tamiyo()

    assert config.lstm_hidden_dim == 128
    assert config.chunk_length == 25  # Matches max_epochs


def test_tamiyo_to_ppo_kwargs():
    """to_ppo_kwargs should include tamiyo-specific parameters."""
    config = TrainingConfig.for_tamiyo()
    ppo_kwargs = config.to_ppo_kwargs()

    assert ppo_kwargs.get("tamiyo") is True
    assert ppo_kwargs.get("num_envs") == 4
    assert ppo_kwargs.get("max_steps_per_env") == 25
