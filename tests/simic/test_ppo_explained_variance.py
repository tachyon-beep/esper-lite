"""Tests for explained variance in PPO update."""

import torch
import pytest

from esper.simic.ppo import PPOAgent


class TestPPOExplainedVariance:
    """Tests for explained variance calculation."""

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

    def test_update_returns_explained_variance(self, agent, filled_buffer):
        """PPO update returns explained_variance metric."""
        metrics = agent.update(last_value=0.0)
        assert "explained_variance" in metrics

    def test_explained_variance_is_reasonable(self, agent, filled_buffer):
        """Explained variance is in valid range."""
        metrics = agent.update(last_value=0.0)
        # Can be negative (worse than mean) or up to 1.0 (perfect)
        assert -2.0 < metrics["explained_variance"] < 1.0


class TestRecurrentPPOExplainedVariance:
    """Tests for explained variance in recurrent PPO update."""

    @pytest.fixture
    def recurrent_agent(self):
        """Create a recurrent PPO agent."""
        return PPOAgent(
            state_dim=10,
            action_dim=4,
            recurrent=True,
            lstm_hidden_dim=32,
            chunk_length=8,
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

    def test_update_recurrent_returns_explained_variance(
        self, recurrent_agent, filled_recurrent_buffer
    ):
        """Recurrent PPO update returns explained_variance metric."""
        metrics = recurrent_agent.update_recurrent(n_epochs=1)
        assert "explained_variance" in metrics

    def test_recurrent_explained_variance_is_reasonable(
        self, recurrent_agent, filled_recurrent_buffer
    ):
        """Explained variance is in valid range for recurrent PPO."""
        metrics = recurrent_agent.update_recurrent(n_epochs=1)
        # Clamped to [-1, 1] in implementation
        assert -1.0 <= metrics["explained_variance"] <= 1.0

    def test_recurrent_explained_variance_excludes_padding(self, recurrent_agent):
        """Explained variance computation excludes padded timesteps."""
        # Create a short episode (shorter than chunk_length)
        recurrent_agent.recurrent_buffer.start_episode(env_id=0)
        hidden = None

        for i in range(3):  # Only 3 steps, chunk_length=8
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
                reward=1.0,  # Non-zero to ensure variance
                done=(i == 2),
                action_mask=action_mask,
                env_id=0,
            )

        recurrent_agent.recurrent_buffer.end_episode(env_id=0)

        # Should not crash and should return valid EV
        metrics = recurrent_agent.update_recurrent(n_epochs=1)
        assert "explained_variance" in metrics
        assert -1.0 <= metrics["explained_variance"] <= 1.0
