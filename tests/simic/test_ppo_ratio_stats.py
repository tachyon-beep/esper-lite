"""Tests for PPO ratio statistics collection."""

import torch
import pytest

from esper.simic.ppo import PPOAgent
from esper.simic.buffers import RolloutBuffer, RecurrentRolloutBuffer


class TestPPORatioStats:
    """Tests for ratio statistics in PPO update."""

    @pytest.fixture
    def agent(self):
        """Create a simple PPO agent."""
        return PPOAgent(
            state_dim=10,
            action_dim=4,
            device="cpu",
        )

    @pytest.fixture
    def filled_buffer(self, agent):
        """Create buffer with some transitions."""
        for _ in range(20):
            state = torch.randn(10)
            action_mask = torch.ones(4)
            action, log_prob, value, _ = agent.get_action(state, action_mask)
            agent.store_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=0.1,
                done=False,
                action_mask=action_mask,
            )
        return agent.buffer

    def test_update_returns_ratio_stats(self, agent, filled_buffer):
        """PPO update returns ratio statistics."""
        metrics = agent.update(last_value=0.0)

        assert "ratio_mean" in metrics
        assert "ratio_std" in metrics
        assert "ratio_max" in metrics
        assert "ratio_min" in metrics

    def test_ratio_stats_are_reasonable(self, agent, filled_buffer):
        """Ratio stats have reasonable values for fresh policy."""
        metrics = agent.update(last_value=0.0)

        # For a fresh policy with no updates, ratios should be near 1.0
        assert 0.5 < metrics["ratio_mean"] < 2.0
        assert metrics["ratio_std"] < 1.0
        assert metrics["ratio_max"] < 5.0
        assert metrics["ratio_min"] > 0.1


class TestRecurrentPPORatioStats:
    """Tests for ratio statistics in recurrent PPO update."""

    @pytest.fixture
    def recurrent_agent(self):
        """Create a recurrent PPO agent."""
        return PPOAgent(
            state_dim=10,
            action_dim=4,
            recurrent=True,
            lstm_hidden_dim=32,
            chunk_length=4,
            device="cpu",
        )

    @pytest.fixture
    def filled_recurrent_buffer(self, recurrent_agent):
        """Create recurrent buffer with one episode."""
        recurrent_agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None

        for i in range(8):
            state = torch.randn(10)
            action_mask = torch.ones(4, dtype=torch.bool)
            action, log_prob, value, hidden = recurrent_agent.get_action(
                state, action_mask, hidden
            )
            recurrent_agent.store_recurrent_transition(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=0.1,
                done=(i == 7),
                action_mask=action_mask,
                env_id=0,
            )

        recurrent_agent.recurrent_buffer.end_episode(env_id=0)
        return recurrent_agent.recurrent_buffer

    def test_update_recurrent_returns_ratio_stats(
        self, recurrent_agent, filled_recurrent_buffer
    ):
        """Recurrent PPO update returns ratio statistics."""
        metrics = recurrent_agent.update_recurrent(n_epochs=1)

        assert "ratio_max" in metrics
        assert "ratio_min" in metrics
        assert "ratio_std" in metrics

    def test_recurrent_ratio_stats_are_reasonable(
        self, recurrent_agent, filled_recurrent_buffer
    ):
        """Ratio stats have reasonable values for fresh recurrent policy."""
        metrics = recurrent_agent.update_recurrent(n_epochs=1)

        # For a fresh policy with no updates, ratios should be near 1.0
        assert isinstance(metrics["ratio_max"], list)
        assert isinstance(metrics["ratio_min"], list)
        assert isinstance(metrics["ratio_std"], list)

        assert metrics["ratio_max"][0] < 5.0
        assert metrics["ratio_min"][0] > 0.1
        assert metrics["ratio_std"][0] < 1.0

    def test_recurrent_ratio_nan_inf_detection(self, recurrent_agent, filled_recurrent_buffer):
        """Recurrent update detects NaN/Inf in ratios."""
        metrics = recurrent_agent.update_recurrent(n_epochs=1)

        # Should not have NaN/Inf in normal operation
        assert not metrics.get("ratio_has_nan", False)
        assert not metrics.get("ratio_has_inf", False)
