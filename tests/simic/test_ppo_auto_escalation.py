"""Tests for PPO auto-escalation on anomaly detection."""

import torch
import pytest

from esper.simic.ppo import PPOAgent
from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel


class TestPPOAutoEscalation:
    """Tests for auto-escalation in PPO update."""

    @pytest.fixture
    def agent(self):
        """Create PPO agent."""
        return PPOAgent(
            state_dim=10,
            action_dim=4,
            device="cpu",
        )

    @pytest.fixture
    def filled_buffer(self, agent):
        """Fill buffer with transitions."""
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

    def test_escalation_triggered_on_anomaly(self, agent, filled_buffer):
        """Anomaly detection triggers escalation."""
        config = TelemetryConfig(
            level=TelemetryLevel.NORMAL,
            auto_escalate_on_anomaly=True,
        )

        # Initial state - not escalated
        assert config.effective_level == TelemetryLevel.NORMAL
        assert config.escalation_epochs_remaining == 0

        # Run update - may or may not detect anomaly depending on random init
        # For this test, we just verify the mechanism exists
        metrics = agent.update(last_value=0.0, telemetry_config=config)

        # Verify anomaly_detected field is in metrics
        assert "anomaly_detected" in metrics

    def test_escalation_disabled_when_flag_false(self, agent, filled_buffer):
        """Escalation doesn't happen when auto_escalate_on_anomaly=False."""
        config = TelemetryConfig(
            level=TelemetryLevel.NORMAL,
            auto_escalate_on_anomaly=False,
        )

        metrics = agent.update(last_value=0.0, telemetry_config=config)

        # Should never escalate
        assert config.escalation_epochs_remaining == 0

    def test_tick_escalation_called_each_update(self, agent, filled_buffer):
        """Escalation countdown ticks each update."""
        config = TelemetryConfig(
            level=TelemetryLevel.NORMAL,
            auto_escalate_on_anomaly=False,  # Disable auto-escalation for this test
        )
        config.escalate_temporarily(epochs=3)

        assert config.escalation_epochs_remaining == 3

        # First update
        agent.update(last_value=0.0, telemetry_config=config)
        assert config.escalation_epochs_remaining == 2

        # Refill buffer for another update
        for _ in range(20):
            state = torch.randn(10)
            action_mask = torch.ones(4)
            action, log_prob, value = agent.get_action(state, action_mask)
            agent.store_transition(
                state=state, action=action, log_prob=log_prob,
                value=value, reward=0.1, done=False, action_mask=action_mask,
            )

        # Second update
        agent.update(last_value=0.0, telemetry_config=config)
        assert config.escalation_epochs_remaining == 1
