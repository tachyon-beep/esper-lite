"""Integration tests for PPO telemetry."""

import torch
import pytest

from esper.simic.ppo import PPOAgent
from esper.simic.telemetry_config import TelemetryConfig, TelemetryLevel


class TestPPOTelemetryIntegration:
    """Tests for telemetry integration in PPO."""

    @pytest.fixture
    def agent_with_telemetry(self):
        """Create PPO agent with telemetry config."""
        return PPOAgent(
            state_dim=10,
            action_dim=4,
            device="cpu",
        )

    @pytest.fixture
    def filled_buffer(self, agent_with_telemetry):
        """Fill buffer with transitions."""
        agent = agent_with_telemetry
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

    def test_update_returns_ppo_health_telemetry(
        self, agent_with_telemetry, filled_buffer
    ):
        """PPO update returns structured telemetry."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        metrics = agent_with_telemetry.update(
            last_value=0.0,
            telemetry_config=config,
        )

        # Should have all Ops Normal metrics
        assert "ratio_mean" in metrics
        assert "ratio_max" in metrics
        assert "explained_variance" in metrics
        assert "policy_loss" in metrics

    def test_debug_level_adds_extra_diagnostics(
        self, agent_with_telemetry, filled_buffer
    ):
        """DEBUG level adds extra diagnostic info beyond NORMAL level."""
        config = TelemetryConfig(level=TelemetryLevel.DEBUG)
        metrics = agent_with_telemetry.update(
            last_value=0.0,
            telemetry_config=config,
        )

        # Should have NORMAL level metrics
        assert "ratio_mean" in metrics
        assert "explained_variance" in metrics

        # Should ALSO have DEBUG-specific metrics
        assert "debug_gradient_stats" in metrics
        assert "debug_numerical_stability" in metrics

    def test_debug_level_collects_layer_gradients(
        self, agent_with_telemetry, filled_buffer
    ):
        """DEBUG level collects per-layer gradient statistics."""
        config = TelemetryConfig(level=TelemetryLevel.DEBUG)
        metrics = agent_with_telemetry.update(
            last_value=0.0,
            telemetry_config=config,
        )

        # Should have debug-specific metrics
        assert "debug_gradient_stats" in metrics
        assert "debug_numerical_stability" in metrics

    def test_normal_level_skips_debug_collection(
        self, agent_with_telemetry, filled_buffer
    ):
        """NORMAL level does not collect expensive debug telemetry."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        metrics = agent_with_telemetry.update(
            last_value=0.0,
            telemetry_config=config,
        )

        # Should NOT have debug-specific metrics
        assert "debug_gradient_stats" not in metrics
        assert "debug_numerical_stability" not in metrics
