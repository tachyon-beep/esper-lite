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
            action, log_prob, value = agent.get_action(state, action_mask)
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


