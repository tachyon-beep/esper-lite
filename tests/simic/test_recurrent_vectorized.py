"""Tests for recurrent policy integration with vectorized training."""

import torch
from unittest.mock import MagicMock

from esper.simic.vectorized import ParallelEnvState


class TestParallelEnvStateRecurrence:
    """Tests for LSTM hidden state tracking in ParallelEnvState."""

    def test_lstm_hidden_field_exists(self):
        """ParallelEnvState should have lstm_hidden field."""
        env_state = ParallelEnvState(
            model=MagicMock(),
            host_optimizer=MagicMock(),
            signal_tracker=MagicMock(),
            governor=MagicMock(),
        )
        # Direct attribute access (no hasattr)
        assert env_state.lstm_hidden is None  # Default

    def test_lstm_hidden_can_store_tuple(self):
        """lstm_hidden should accept (h, c) tuple."""
        env_state = ParallelEnvState(
            model=MagicMock(),
            host_optimizer=MagicMock(),
            signal_tracker=MagicMock(),
            governor=MagicMock(),
        )
        h = torch.zeros(1, 1, 128)
        c = torch.zeros(1, 1, 128)
        env_state.lstm_hidden = (h, c)

        assert env_state.lstm_hidden is not None
        assert torch.equal(env_state.lstm_hidden[0], h)
        assert torch.equal(env_state.lstm_hidden[1], c)

    def test_lstm_hidden_reset_to_none(self):
        """lstm_hidden should be resettable to None for episode boundaries."""
        env_state = ParallelEnvState(
            model=MagicMock(),
            host_optimizer=MagicMock(),
            signal_tracker=MagicMock(),
            governor=MagicMock(),
        )
        env_state.lstm_hidden = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
        env_state.lstm_hidden = None  # Reset on episode end

        assert env_state.lstm_hidden is None
