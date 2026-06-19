"""Q-value finiteness gate (P1-b) and PPO q_value contract (P2-a).

P1-b: a non-finite q_value must trip the SAME skip-epoch finiteness gate as a
non-finite value, so the optimizer never steps on a corrupted Q-aux loss. Before
the fix the fused gate reduced over new/old log_probs and ``values`` but NOT
``q_values``, so a non-finite q reached backward()/optimizer.step().

P2-a: PPO genuinely requires EvalResult.q_value (P0-1 always builds a q_head).
If a policy returns q_value=None, PPO must fail loud at the boundary with a clear
AssertionError rather than crashing opaquely at ``q_values[valid_mask]``.
"""

from __future__ import annotations

import dataclasses

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


def _snapshot_params(agent: PPOAgent) -> list[torch.Tensor]:
    return [p.detach().clone() for p in agent.policy.network.parameters()]


def _params_unchanged(agent: PPOAgent, snapshot: list[torch.Tensor]) -> bool:
    return all(
        torch.equal(p.detach(), s)
        for p, s in zip(agent.policy.network.parameters(), snapshot, strict=True)
    )


def test_nonfinite_q_value_skips_optimizer_step() -> None:
    """P1-b: a non-finite q_value trips the finiteness gate; the optimizer never steps.

    Before the gate fix, q_values were excluded from the fused finiteness reduction,
    so a NaN q fed q_aux_loss -> total_loss -> backward()/optimizer.step(), corrupting
    every parameter. The gate must skip the epoch (ppo_update_performed=False, params
    unchanged) on the SAME path as a non-finite value.
    """
    agent, slot_config = _build_agent()
    _fill_buffer(agent, slot_config)

    real_eval = agent.policy.evaluate_actions

    def poisoned_eval(*args, **kwargs):
        result = real_eval(*args, **kwargs)
        bad_q = result.q_value.clone()
        bad_q[..., 0] = float("nan")  # inject NaN into the op-conditioned Q-aux head
        return dataclasses.replace(result, q_value=bad_q)

    agent.policy.evaluate_actions = poisoned_eval  # type: ignore[method-assign]

    snapshot = _snapshot_params(agent)
    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is False, (
        "Non-finite q_value must trip the finiteness gate and skip the optimizer step."
    )
    assert metrics["finiteness_gate_skip_count"] >= 1
    assert _params_unchanged(agent, snapshot), (
        "Optimizer stepped on a NaN q_value: parameters were mutated."
    )


def test_finite_q_value_steps_optimizer() -> None:
    """Control: with a finite q_value the gate passes and the optimizer steps."""
    agent, slot_config = _build_agent()
    _fill_buffer(agent, slot_config)

    snapshot = _snapshot_params(agent)
    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True
    assert not _params_unchanged(agent, snapshot), (
        "A healthy update must mutate parameters (optimizer stepped)."
    )


def test_none_q_value_raises_clear_assertion() -> None:
    """P2-a: PPO requires q_value; a None must fail loud at the boundary."""
    agent, slot_config = _build_agent()
    _fill_buffer(agent, slot_config)

    real_eval = agent.policy.evaluate_actions

    def none_q_eval(*args, **kwargs):
        result = real_eval(*args, **kwargs)
        return dataclasses.replace(result, q_value=None)

    agent.policy.evaluate_actions = none_q_eval  # type: ignore[method-assign]

    with pytest.raises(AssertionError, match="PPO requires a q_value"):
        agent.update(clear_buffer=True)
