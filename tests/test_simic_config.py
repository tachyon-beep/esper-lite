"""Tests for TrainingConfig dataclass."""

import pytest
import warnings

from esper.simic.config import TrainingConfig


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config_creates_successfully(self):
        """Default TrainingConfig should initialize without errors."""
        config = TrainingConfig()
        assert config.recurrent is False
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

    def test_chunk_length_warning_when_less_than_max_epochs(self):
        """Should warn when chunk_length < max_epochs with recurrent=True."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = TrainingConfig(recurrent=True, max_epochs=25, chunk_length=10)
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "Mid-episode chunking will occur" in str(w[0].message)

    def test_no_warning_when_chunk_length_matches_max_epochs(self):
        """No warning when chunk_length >= max_epochs."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = TrainingConfig(recurrent=True, max_epochs=25, chunk_length=25)
            # Filter out the lr warning
            chunk_warnings = [x for x in w if "chunk_length" in str(x.message)]
            assert len(chunk_warnings) == 0

    def test_no_warning_when_recurrent_false(self):
        """No chunk_length warning when recurrent=False."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = TrainingConfig(recurrent=False, max_epochs=25, chunk_length=10)
            chunk_warnings = [x for x in w if "chunk_length" in str(x.message)]
            assert len(chunk_warnings) == 0

    def test_recurrent_config_to_ppo_kwargs(self):
        """Recurrent params should flow to PPOAgent kwargs."""
        config = TrainingConfig(recurrent=True, lstm_hidden_dim=256, chunk_length=16)
        kwargs = config.to_ppo_kwargs()
        assert kwargs["recurrent"] is True
        assert kwargs["lstm_hidden_dim"] == 256
        assert kwargs["chunk_length"] == 16

    def test_recurrent_config_to_train_kwargs(self):
        """Recurrent params should flow to train_ppo_vectorized kwargs."""
        config = TrainingConfig(recurrent=True)
        kwargs = config.to_train_kwargs()
        assert kwargs["recurrent"] is True
        assert "lstm_hidden_dim" in kwargs
        assert "chunk_length" in kwargs

    def test_summary_includes_recurrent_info(self):
        """Summary should show LSTM info when recurrent=True."""
        config = TrainingConfig(recurrent=True, lstm_hidden_dim=256, chunk_length=30)
        summary = config.summary()
        assert "LSTM" in summary
        assert "hidden=256" in summary
        assert "chunk=30" in summary

    def test_summary_shows_mlp_when_not_recurrent(self):
        """Summary should show MLP when recurrent=False."""
        config = TrainingConfig(recurrent=False)
        summary = config.summary()
        assert "MLP" in summary
        # Should not show hidden/chunk details
        assert "hidden=" not in summary
