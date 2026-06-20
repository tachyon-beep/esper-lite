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
# intended multi-epoch drift. At K=4 the epoch-0 reference pass yields ratio==1.0, but
# epochs 1-3 measure real pi_theta_k/pi_theta_0 drift, so the aggregated metrics (mean
# over epochs) diverge from K=1: approx_kl, clip_fraction, and the ratio_* spread all
# become non-degenerate.
#
# RE-BASELINED 2026-06-17 (P0-1: op-INDEPENDENT V(s) baseline). The op-conditioned
# value_head Q(s, op) was split into a NEW op-independent state_value_head (the PPO
# baseline V(s)) plus a renamed q_head (telemetry/aux). This legitimately shifts these
# goldens for two compounding reasons:
#   1. The PPO baseline is now V(s) not Q(s, sampled_op) -> different advantages ->
#      different policy_loss/value_loss.
#   2. Adding state_value_head changes the orthogonal-init RNG draw sequence in
#      _init_weights, so every downstream head initializes from a different slice of
#      the RNG stream (entropy and absolute losses move accordingly).
# These were NOT re-pinned by relaxing assertions: they are the deterministic output of
# the new architecture, regenerated and verified stable across repeated runs. (K=1 ratio
# is now exactly 1.0 -- the new V(s) head leaves the epoch-0 anchored ratio identity
# cleaner than the prior 0.9999998 float residue.)
_GOLDENS: dict[int, dict[str, float]] = {
    1: {
        "policy_loss": -1.1549229621887207,
        "value_loss": 0.025900892913341522,
        "entropy": 6.99962043762207,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "ratio_mean": 1.0,
        "ratio_max": 1.0,
        "ratio_min": 1.0,
        "ratio_std": 0.0,
    },
    4: {
        "policy_loss": -1.3849077224731445,
        "value_loss": 0.01459794957190752,
        "entropy": 6.966041088104248,
        "approx_kl": 0.007933689281344414,
        "clip_fraction": 0.3125,
        "ratio_mean": 1.294655680656433,
        "ratio_max": 3.2952778339385986,
        "ratio_min": 0.8902816772460938,
        "ratio_std": 0.4891107678413391,
    },
}

# GitHub's CPU-only PyTorch wheel and local CUDA-capable builds differ slightly
# in deterministic CPU math here. Keep the tolerance tight enough to catch
# refactor drift while allowing runner-level floating-point variation.
_GOLDEN_ABS_TOLERANCE = 2e-4


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
        assert metrics[key] == pytest.approx(
            golden[key],
            abs=_GOLDEN_ABS_TOLERANCE,
        ), key
