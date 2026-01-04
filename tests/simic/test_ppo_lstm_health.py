"""Tests for LSTM health monitoring in PPO agent.

TELE-340: LSTM hidden state health metrics should be computed and returned
during PPO update() calls. This enables monitoring of LSTM numerical stability
during training.
"""

import pytest
import torch

from esper.leyline import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)
from esper.leyline.slot_config import SlotConfig
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy import create_policy
from esper.tamiyo.policy.features import get_feature_size


@pytest.fixture
def ppo_agent():
    """Create a minimal PPO agent for testing."""
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    return PPOAgent(
        policy=policy,
        slot_config=slot_config,
        device="cpu",
        num_envs=2,
        max_steps_per_env=8,
        target_kl=None,  # Disable early stopping for predictable test behavior
    )


def _fill_buffer_with_rollout(agent: PPOAgent) -> None:
    """Fill the agent's buffer with a minimal rollout for testing."""
    device = torch.device(agent.device)
    state_dim = get_feature_size(agent.slot_config)
    hidden = agent.policy.network.get_initial_hidden(1, device)

    for env_id in range(agent.num_envs):
        agent.buffer.start_episode(env_id)
        for step in range(agent.max_steps_per_env):
            state = torch.randn(1, state_dim, device=device)
            masks = {
                "slot": torch.ones(1, agent.slot_config.num_slots, dtype=torch.bool, device=device),
                "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
                "style": torch.ones(1, NUM_STYLES, dtype=torch.bool, device=device),
                "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool, device=device),
                "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
                "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
                "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
                "op": torch.ones(1, NUM_OPS, dtype=torch.bool, device=device),
            }
            pre_hidden = hidden
            bp_indices = torch.zeros(1, agent.slot_config.num_slots, dtype=torch.long, device=device)
            result = agent.policy.network.get_action(
                state,
                bp_indices,
                hidden,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )
            hidden = result.hidden

            agent.buffer.add(
                env_id=env_id,
                state=state.squeeze(0),
                slot_action=result.actions["slot"].item(),
                blueprint_action=result.actions["blueprint"].item(),
                style_action=result.actions["style"].item(),
                tempo_action=result.actions["tempo"].item(),
                alpha_target_action=result.actions["alpha_target"].item(),
                alpha_speed_action=result.actions["alpha_speed"].item(),
                alpha_curve_action=result.actions["alpha_curve"].item(),
                op_action=result.actions["op"].item(),
                effective_op_action=result.actions["op"].item(),
                slot_log_prob=result.log_probs["slot"].item(),
                blueprint_log_prob=result.log_probs["blueprint"].item(),
                style_log_prob=result.log_probs["style"].item(),
                tempo_log_prob=result.log_probs["tempo"].item(),
                alpha_target_log_prob=result.log_probs["alpha_target"].item(),
                alpha_speed_log_prob=result.log_probs["alpha_speed"].item(),
                alpha_curve_log_prob=result.log_probs["alpha_curve"].item(),
                op_log_prob=result.log_probs["op"].item(),
                value=result.values.item(),
                reward=1.0,
                done=step == agent.max_steps_per_env - 1,
                truncated=False,
                slot_mask=masks["slot"].squeeze(0),
                blueprint_mask=masks["blueprint"].squeeze(0),
                style_mask=masks["style"].squeeze(0),
                tempo_mask=masks["tempo"].squeeze(0),
                alpha_target_mask=masks["alpha_target"].squeeze(0),
                alpha_speed_mask=masks["alpha_speed"].squeeze(0),
                alpha_curve_mask=masks["alpha_curve"].squeeze(0),
                op_mask=masks["op"].squeeze(0),
                hidden_h=pre_hidden[0],
                hidden_c=pre_hidden[1],
                bootstrap_value=0.0,
                blueprint_indices=bp_indices.squeeze(0),
            )
        agent.buffer.end_episode(env_id)


class TestTELE340LstmHealthWiring:
    """TELE-340: LSTM health metrics should be computed during PPO update."""

    def test_lstm_health_in_update_metrics(self, ppo_agent):
        """TELE-340: update() should return LSTM health metrics."""
        # Collect rollout
        _fill_buffer_with_rollout(ppo_agent)

        # Run update
        metrics = ppo_agent.update()

        # LSTM health metrics should be present
        assert "lstm_h_rms" in metrics, "lstm_h_rms should be in update metrics"
        assert "lstm_c_rms" in metrics, "lstm_c_rms should be in update metrics"
        assert "lstm_h_env_rms_mean" in metrics, "lstm_h_env_rms_mean should be in update metrics"
        assert "lstm_h_env_rms_max" in metrics, "lstm_h_env_rms_max should be in update metrics"
        assert "lstm_c_env_rms_mean" in metrics, "lstm_c_env_rms_mean should be in update metrics"
        assert "lstm_c_env_rms_max" in metrics, "lstm_c_env_rms_max should be in update metrics"
        assert metrics["lstm_h_rms"] is not None, "lstm_h_rms should not be None"
        assert metrics["lstm_c_rms"] is not None, "lstm_c_rms should not be None"
        assert isinstance(metrics["lstm_h_rms"], float), "lstm_h_rms should be a float"
        assert isinstance(metrics["lstm_c_rms"], float), "lstm_c_rms should be a float"
        # RMS magnitudes should be non-negative
        assert metrics["lstm_h_rms"] >= 0, "lstm_h_rms should be non-negative"
        assert metrics["lstm_c_rms"] >= 0, "lstm_c_rms should be non-negative"

    def test_lstm_health_max_values(self, ppo_agent):
        """TELE-340: LSTM health should include max values for spike detection."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        # Max values should be present
        assert "lstm_h_max" in metrics, "lstm_h_max should be in update metrics"
        assert "lstm_c_max" in metrics, "lstm_c_max should be in update metrics"
        assert isinstance(metrics["lstm_h_max"], float), "lstm_h_max should be a float"
        assert isinstance(metrics["lstm_c_max"], float), "lstm_c_max should be a float"
        # Max values should be non-negative (abs values)
        assert metrics["lstm_h_max"] >= 0, "lstm_h_max should be non-negative"
        assert metrics["lstm_c_max"] >= 0, "lstm_c_max should be non-negative"

    def test_lstm_health_boolean_flags(self, ppo_agent):
        """TELE-340: LSTM health should include boolean flags for NaN/Inf detection."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        # Boolean flags should be present and False for healthy state
        assert "lstm_has_nan" in metrics, "lstm_has_nan should be in update metrics"
        assert "lstm_has_inf" in metrics, "lstm_has_inf should be in update metrics"
        assert metrics["lstm_has_nan"] is False, "lstm_has_nan should be False for healthy state"
        assert metrics["lstm_has_inf"] is False, "lstm_has_inf should be False for healthy state"

    def test_lstm_health_none_when_empty_buffer(self, ppo_agent):
        """TELE-340: When buffer is empty, update should return empty dict."""
        # Don't fill buffer - leave it empty
        metrics = ppo_agent.update()

        # Empty buffer returns empty dict
        assert metrics == {}, "Empty buffer should return empty metrics dict"
