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
    """Fill the rollout buffer with a fixed, deterministic 4-step episode.

    Load-bearing for the anchored-reference-pass goldens: this fill emits NO fresh
    counterfactual-contribution measurements, so the aux contribution-predictor loss is
    multiplied to zero (and contributes zero gradient). That is why the goldens are stable
    under the production default (enable_contribution_aux=True) despite the aux Dropout
    drawing RNG in the no_grad anchor — the draw is annihilated by the zero aux mask and so
    cannot make these goldens RNG-order-dependent. A future change that emits fresh-
    contribution timesteps here would make the aux gradient (and thus the goldens) depend on
    the anchor-vs-epoch dropout ordering; regenerate and re-pin if that happens.
    """
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


# Golden PPO-update metrics keyed by recurrent_n_epochs (K).
#
# epoch-0 ratio==1.0 by construction (anchored reference pass); K=4 goldens capture
# intended multi-epoch drift. Re-baselined 2026-06-18 for the value-head init gain
# 0.01->0.1 change (b1935d39): the larger init shifts V(s) -> advantages, moving the
# advantage-dependent metrics (policy_loss/value_loss, and the post-update entropy/kl).
# At K=1 the anchored epoch-0 ratio is still exactly 1.0 (kl/clip/ratio_* unchanged);
# only policy_loss/value_loss move. At K=4 the epoch-0 reference pass still yields
# ratio==1.0, but epochs 1-3 measure real pi_theta_k/pi_theta_0 drift, so the aggregated
# metrics (mean over epochs) diverge from K=1: approx_kl, clip_fraction, and the ratio_*
# spread all become non-degenerate.
_GOLDENS: dict[int, dict[str, float]] = {
    1: {
        "policy_loss": -2.378634214401245,
        "value_loss": 0.03463732823729515,
        "entropy": 9.399517059326172,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "ratio_mean": 0.9999998807907104,
        "ratio_max": 1.0,
        "ratio_min": 0.9999997615814209,
        "ratio_std": 9.733398087519163e-08,
    },
    4: {
        "policy_loss": -2.8116531372070312,
        "value_loss": 0.01667206548154354,
        "entropy": 9.360645294189453,
        "approx_kl": 0.010239769704639912,
        "clip_fraction": 0.5,
        "ratio_mean": 1.2072563171386719,
        "ratio_max": 3.147684097290039,
        "ratio_min": 0.4454832971096039,
        "ratio_std": 0.538439929485321,
    },
}


@pytest.mark.parametrize("recurrent_n_epochs", [1, 4])
def test_ppo_update_golden_metrics(recurrent_n_epochs: int) -> None:
    agent, slot_config = _build_agent(recurrent_n_epochs=recurrent_n_epochs)
    _fill_buffer(agent, slot_config)
    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True
    assert metrics["finiteness_gate_skip_count"] == 0

    golden = _GOLDENS[recurrent_n_epochs]
    keys = (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "ratio_mean",
        "ratio_max",
        "ratio_min",
        "ratio_std",
    )
    for key in keys:
        assert metrics[key] == pytest.approx(golden[key], abs=1e-6), key
