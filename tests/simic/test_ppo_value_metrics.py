"""Tests for value function metrics in PPO update.

TELE-220 to TELE-228: Value function diagnostic metrics should be computed
and returned during PPO update() calls. These metrics enable monitoring of
value function health and training progress.
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


def _fill_buffer_with_rollout(agent: PPOAgent, varied_rewards: bool = False) -> None:
    """Fill the agent's buffer with a minimal rollout for testing.

    Args:
        agent: PPO agent with buffer to fill
        varied_rewards: If True, use varied rewards to create non-trivial statistics
    """
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

            # Varied rewards create non-trivial statistics for percentile tests
            if varied_rewards:
                reward = float(step + env_id)  # 0, 1, 2, ... varying rewards
            else:
                reward = 1.0

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
                reward=reward,
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


class TestTELE220to228ValueFunctionMetrics:
    """TELE-220 to TELE-228: Value function metrics in PPO update."""

    def test_value_function_metrics_in_update_result(self, ppo_agent):
        """update() should return all 9 value function metrics."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        # All 9 metrics should be present (TELE-220 to TELE-228)
        assert "v_return_correlation" in metrics, "TELE-220: v_return_correlation missing"
        assert "td_error_mean" in metrics, "TELE-221: td_error_mean missing"
        assert "td_error_std" in metrics, "TELE-222: td_error_std missing"
        assert "bellman_error" in metrics, "TELE-223: bellman_error missing"
        assert "return_p10" in metrics, "TELE-224: return_p10 missing"
        assert "return_p50" in metrics, "TELE-225: return_p50 missing"
        assert "return_p90" in metrics, "TELE-226: return_p90 missing"
        assert "return_variance" in metrics, "TELE-227: return_variance missing"
        assert "return_skewness" in metrics, "TELE-228: return_skewness missing"

    def test_v_return_correlation_in_valid_range(self, ppo_agent):
        """V-return correlation should be in [-1, 1]."""
        _fill_buffer_with_rollout(ppo_agent, varied_rewards=True)

        metrics = ppo_agent.update()

        assert -1.0 <= metrics["v_return_correlation"] <= 1.0, \
            f"v_return_correlation {metrics['v_return_correlation']} out of [-1, 1]"

    def test_return_percentiles_ordered(self, ppo_agent):
        """Return percentiles should be ordered: p10 <= p50 <= p90."""
        _fill_buffer_with_rollout(ppo_agent, varied_rewards=True)

        metrics = ppo_agent.update()

        assert metrics["return_p10"] <= metrics["return_p50"], \
            f"p10 ({metrics['return_p10']}) > p50 ({metrics['return_p50']})"
        assert metrics["return_p50"] <= metrics["return_p90"], \
            f"p50 ({metrics['return_p50']}) > p90 ({metrics['return_p90']})"

    def test_td_error_mean_is_finite(self, ppo_agent):
        """TD error mean should be finite."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        import math
        assert math.isfinite(metrics["td_error_mean"]), \
            f"td_error_mean is not finite: {metrics['td_error_mean']}"

    def test_bellman_error_non_negative(self, ppo_agent):
        """Bellman error (mean absolute TD error) should be non-negative."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        assert metrics["bellman_error"] >= 0, \
            f"bellman_error should be non-negative: {metrics['bellman_error']}"

    def test_return_variance_non_negative(self, ppo_agent):
        """Return variance should be non-negative."""
        _fill_buffer_with_rollout(ppo_agent)

        metrics = ppo_agent.update()

        assert metrics["return_variance"] >= 0, \
            f"return_variance should be non-negative: {metrics['return_variance']}"

    def test_empty_buffer_returns_empty_dict(self, ppo_agent):
        """When buffer is empty, update should return empty dict."""
        # Don't fill buffer - leave it empty
        metrics = ppo_agent.update()

        assert metrics == {}, "Empty buffer should return empty metrics dict"
