"""Tests for TrainingConfig dataclass."""

import inspect

import pytest

from esper.simic.config import TrainingConfig
from esper.simic.vectorized import train_ppo_vectorized


class TestTrainingConfigDefaults:
    """Tests for TrainingConfig default values."""

    def test_default_config_creates_successfully(self):
        """Default TrainingConfig should initialize without errors."""
        config = TrainingConfig()
        assert config.lstm_hidden_dim == 128
        assert config.chunk_length == config.max_epochs  # Auto-matched
        assert config.reward_family.value == "contribution"
        assert config.reward_mode.value == "shaped"

    def test_default_gamma_gae_lambda(self):
        """Default gamma should be optimized for 25-epoch episodes."""
        config = TrainingConfig()
        assert config.gamma == 0.995

    def test_chunk_length_auto_matches_max_epochs(self):
        """chunk_length=None should auto-match to max_epochs."""
        config = TrainingConfig(max_epochs=50)
        assert config.chunk_length == 50

    def test_chunk_length_explicit_value_preserved(self):
        """Explicit chunk_length should not be overridden."""
        config = TrainingConfig(max_epochs=25, chunk_length=25)
        assert config.chunk_length == 25


class TestTrainingConfigConversion:
    """Tests for TrainingConfig to kwargs conversion methods."""

    def test_lstm_config_to_ppo_kwargs(self):
        """LSTM params should flow to PPOAgent kwargs."""
        config = TrainingConfig(
            lstm_hidden_dim=256,
            chunk_length=25,
            entropy_anneal_episodes=8,
            n_envs=4,
            ppo_updates_per_batch=2,
        )
        kwargs = config.to_ppo_kwargs()
        assert kwargs["lstm_hidden_dim"] == 256
        assert kwargs["chunk_length"] == 25
        # ceil(episodes / n_envs) * updates_per_batch
        assert kwargs["entropy_anneal_steps"] == 4

    def test_to_ppo_kwargs_includes_vectorized_params(self):
        """to_ppo_kwargs should include vectorized training parameters."""
        config = TrainingConfig(entropy_anneal_episodes=8, n_envs=4, ppo_updates_per_batch=2)
        ppo_kwargs = config.to_ppo_kwargs()

        assert ppo_kwargs.get("num_envs") == config.n_envs
        assert ppo_kwargs.get("max_steps_per_env") == config.max_epochs
        assert ppo_kwargs.get("gamma") == 0.995
        assert ppo_kwargs.get("entropy_anneal_steps") == 4

    def test_lstm_config_to_train_kwargs(self):
        """LSTM params should flow to train_ppo_vectorized kwargs."""
        config = TrainingConfig()
        kwargs = config.to_train_kwargs()
        assert "lstm_hidden_dim" in kwargs
        assert "chunk_length" in kwargs
        assert kwargs["reward_family"] == "contribution"
        assert kwargs["reward_mode"] == "shaped"

    def test_to_train_kwargs_is_subset_of_vectorized_signature(self):
        """Config → train kwargs must stay in sync with train_ppo_vectorized signature."""
        signature = inspect.signature(train_ppo_vectorized)
        allowed = set(signature.parameters)

        config = TrainingConfig()
        kwargs = set(config.to_train_kwargs())

        assert kwargs <= allowed


class TestTrainingConfigSummary:
    """Tests for TrainingConfig summary display."""

    def test_summary_includes_lstm_info(self):
        """Summary should show LSTM info."""
        config = TrainingConfig(lstm_hidden_dim=256, chunk_length=25)
        summary = config.summary()
        assert "LSTM" in summary
        assert "hidden=256" in summary
        assert "chunk=25" in summary


class TestTrainingConfigSerialization:
    """Tests for TrainingConfig serialization."""

    def test_from_dict_rejects_unknown_keys(self):
        """Unknown keys should hard-fail to avoid drift."""
        with pytest.raises(ValueError):
            TrainingConfig.from_dict({"lr": 1e-4, "unknown": 1})

    def test_to_dict_roundtrip_preserves_enums(self):
        """Enum fields should serialize to values and back."""
        original = TrainingConfig(reward_family="loss", reward_mode="shaped", slots=["r0c0", "r0c2"])
        loaded = TrainingConfig.from_dict(original.to_dict())
        assert loaded.reward_family.value == "loss"
        assert loaded.slots == ["r0c0", "r0c2"]
