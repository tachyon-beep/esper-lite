"""P0-1: the op-conditioned q_head is TRAINED (not dead telemetry).

After one PPO update, BOTH the op-INDEPENDENT state_value_head AND the op-conditioned
q_head must receive nonzero gradient — the q_head via the small detached Q-aux
regression toward the same normalized-returns target as V(s). A never-trained Q head
would emit init noise (a forbidden dead telemetry component).
"""

from __future__ import annotations

import torch

from esper.leyline import (
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
)
from esper.leyline.slot_config import SlotConfig
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy import create_policy
from esper.tamiyo.policy.features import get_feature_size


def _build_agent(recurrent_n_epochs: int = 1) -> tuple[PPOAgent, SlotConfig]:
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
        recurrent_n_epochs=recurrent_n_epochs,
    )
    return agent, slot_config


def _fill_buffer(agent: PPOAgent, slot_config: SlotConfig) -> None:
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
            state, bp_indices, hidden,
            slot_mask=masks["slot"], blueprint_mask=masks["blueprint"],
            style_mask=masks["style"], tempo_mask=masks["tempo"],
            alpha_target_mask=masks["alpha_target"], alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"], op_mask=masks["op"],
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


def test_q_head_trained_aux() -> None:
    """After one update, q_head AND state_value_head params received nonzero grad."""
    agent, slot_config = _build_agent()
    _fill_buffer(agent, slot_config)

    metrics = agent.update(clear_buffer=True)
    assert metrics["ppo_update_performed"] is True

    # Per-head gradient-state telemetry captures the BACKWARD grad state before the
    # optimizer step. Both value-like heads must report a real (finite) gradient,
    # never "missing"/"not_learnable" -- "value" is V(s) (state_value_head), "q" is the
    # op-conditioned q_head trained by the detached Q-aux regression.
    grad_states = metrics["head_gradient_states"]
    assert grad_states["value"][0] in ("finite", "nonfinite"), grad_states["value"]
    assert grad_states["q"][0] in ("finite", "nonfinite"), grad_states["q"]

    # And the recorded grad norms must be finite and > 0 for both value-like heads.
    grad_norms = metrics["head_grad_norms"]
    v_norm = grad_norms["value"][0]
    q_norm = grad_norms["q"][0]
    assert v_norm > 0.0 and torch.isfinite(torch.tensor(v_norm)), f"V grad norm = {v_norm}"
    assert q_norm > 0.0 and torch.isfinite(torch.tensor(q_norm)), f"Q grad norm = {q_norm}"

    # The Q-aux loss telemetry must be present and finite (the head is being trained).
    assert "q_aux_loss" in metrics
    assert torch.isfinite(torch.as_tensor(metrics["q_aux_loss"])).all()
