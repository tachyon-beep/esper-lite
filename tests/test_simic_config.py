"""Tests for TrainingConfig dataclass."""

import pytest

from esper.simic.config import TrainingConfig


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config_creates_successfully(self):
        """Default TrainingConfig should initialize without errors."""
        config = TrainingConfig()
        assert config.lstm_hidden_dim == 128
        assert config.chunk_length == config.max_epochs  # Auto-matched

    def test_chunk_length_auto_matches_max_epochs(self):
        """chunk_length=None should auto-match to max_epochs."""
        config = TrainingConfig(max_epochs=50)
        assert config.chunk_length == 50

    def test_chunk_length_explicit_value_preserved(self):
        """Explicit chunk_length should not be overridden."""
        config = TrainingConfig(max_epochs=25, chunk_length=16)
        assert config.chunk_length == 16

    def test_lstm_config_to_ppo_kwargs(self):
        """LSTM params should flow to PPOAgent kwargs."""
        config = TrainingConfig(lstm_hidden_dim=256, chunk_length=16)
        kwargs = config.to_ppo_kwargs()
        assert kwargs["lstm_hidden_dim"] == 256
        assert kwargs["chunk_length"] == 16
        # recurrent key should no longer exist
        assert "recurrent" not in kwargs

    def test_lstm_config_to_train_kwargs(self):
        """LSTM params should flow to train_ppo_vectorized kwargs."""
        config = TrainingConfig()
        kwargs = config.to_train_kwargs()
        assert "lstm_hidden_dim" in kwargs
        assert "chunk_length" in kwargs
        # recurrent key should no longer exist
        assert "recurrent" not in kwargs

    def test_summary_includes_lstm_info(self):
        """Summary should show LSTM info."""
        config = TrainingConfig(lstm_hidden_dim=256, chunk_length=30)
        summary = config.summary()
        assert "LSTM" in summary
        assert "hidden=256" in summary
        assert "chunk=30" in summary
