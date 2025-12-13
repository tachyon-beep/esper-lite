"""Tests for PPO ratio statistics collection."""

import torch
import pytest

from esper.simic.ppo import PPOAgent
from esper.simic.buffers import RolloutBuffer


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


