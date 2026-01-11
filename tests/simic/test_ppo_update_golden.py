"""Golden-value PPO update metrics for refactor drift detection."""

from __future__ import annotations

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


def _build_agent() -> tuple[PPOAgent, SlotConfig]:
    torch.manual_seed(123)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=4,
        chunk_length=4,
        device="cpu",
        target_kl=None,
        recurrent_n_epochs=1,
    )
    return agent, slot_config


def _fill_buffer(agent: PPOAgent, slot_config: SlotConfig, log_prob_offset: float) -> None:
    state_dim = get_feature_size(slot_config)
    device = torch.device(agent.device)
    base_state = torch.linspace(-1.0, 1.0, steps=state_dim, device=device)

    masks = {
        "slot": torch.ones(1, slot_config.num_slots, dtype=torch.bool, device=device),
        "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
        "style": torch.ones(1, NUM_STYLES, dtype=torch.bool, device=device),
        "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool, device=device),
        "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
        "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
        "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
        "op": torch.ones(1, NUM_OPS, dtype=torch.bool, device=device),
    }

    hidden = agent.policy.network.get_initial_hidden(1, device)
    agent.buffer.start_episode(0)

    rewards = [0.2, -0.1, 0.3, 0.0]
    for step, reward in enumerate(rewards):
        state = (base_state + step * 0.01).unsqueeze(0)
        pre_hidden = hidden
        bp_indices = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=device)
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
            env_id=0,
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
            slot_log_prob=result.log_probs["slot"].item() + log_prob_offset,
            blueprint_log_prob=result.log_probs["blueprint"].item() + log_prob_offset,
            style_log_prob=result.log_probs["style"].item() + log_prob_offset,
            tempo_log_prob=result.log_probs["tempo"].item() + log_prob_offset,
            alpha_target_log_prob=result.log_probs["alpha_target"].item() + log_prob_offset,
            alpha_speed_log_prob=result.log_probs["alpha_speed"].item() + log_prob_offset,
            alpha_curve_log_prob=result.log_probs["alpha_curve"].item() + log_prob_offset,
            op_log_prob=result.log_probs["op"].item() + log_prob_offset,
            value=result.values.item(),
            reward=reward,
            done=step == len(rewards) - 1,
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
    agent.buffer.end_episode(0)


def test_ppo_update_golden_metrics() -> None:
    agent, slot_config = _build_agent()
    _fill_buffer(agent, slot_config, log_prob_offset=0.1)
    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True
    assert metrics["finiteness_gate_skip_count"] == 0

    # Phase 4: ResidualLSTM hidden state shape fix (hidden=None now creates correct [batch, hidden])
    assert metrics["policy_loss"] == pytest.approx(-2.1166460514068604, abs=1e-6)
    assert metrics["value_loss"] == pytest.approx(0.03467179462313652, abs=1e-6)
    assert metrics["entropy"] == pytest.approx(9.399517059326172, abs=1e-6)
    assert metrics["approx_kl"] == pytest.approx(0.004860731307417154, abs=1e-6)
    assert metrics["clip_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["ratio_mean"] == pytest.approx(0.4489768147468567, abs=1e-6)
    assert metrics["ratio_max"] == pytest.approx(0.4493289887905121, abs=1e-6)
    assert metrics["ratio_min"] == pytest.approx(0.4479205310344696, abs=1e-6)
    assert metrics["ratio_std"] == pytest.approx(0.0007041990756988525, abs=1e-6)
