"""Acceptance #5 (HEADLINE): pre-update explained_variance lifts off 0 at K=4 vs K=1.

CI-gated (``-m integration``). A short, deterministic, fixed-seed paired run isolates the
recurrent-epoch count K as the ONLY variable: both legs share the seed, the fixed
states+rewards, and the same gain=0.01 value-head init. Each cycle re-scores the SAME fixed
rollout with the current network so ``data["values"]`` (the quantity pre-update EV is computed
against, ppo_agent.py:597) reflects the freshly-trained critic. Scope note: rewards are fixed
per step and action-independent, so this isolates the critic-FIT mechanism (more grad steps
-> lower TD error -> higher EV) on a stationary target; it deliberately does NOT exercise
non-stationary data or exploration. Because EV reads against the stored values, a frozen/
replayed buffer (stale stored values) cannot show the liftoff under this EV definition.

The mechanism (spec 2026-06-17): K=1 gives the critic one gradient step per rollout, too few
to fit a high-variance return target, so EV stays pinned ~0. K=4 gives 4 steps per rollout
(anchored reference pass keeps the multi-epoch update mathematically exact), so the critic
fits and successive rollouts' stored values track returns -> EV climbs off 0 toward 0.3-0.8.

Empirical bound captured 2026-06-17 (CPU/FP32, seed 2024, 40 cycles): K=4 final EV ~0.47,
K=1 final EV ~0.09. Thresholds below carry margin against that observation.
"""
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

_N_STEPS = 12
# High-variance reward sequence -> returns with real spread (spec: returns std 7-13).
_REWARDS = [4.0, -3.0, 6.0, -5.0, 2.0, -4.0, 5.0, -2.0, 3.0, -6.0, 4.0, -1.0]
_CYCLES = 40
_SEED = 2024


def _build_agent(recurrent_n_epochs: int) -> tuple[PPOAgent, SlotConfig]:
    torch.manual_seed(_SEED)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm", slot_config=slot_config, device="cpu", compile_mode="off"
    )
    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=_N_STEPS,
        chunk_length=_N_STEPS,
        device="cpu",
        target_kl=None,
        recurrent_n_epochs=recurrent_n_epochs,
    )
    return agent, slot_config


def _regenerate_rollout(agent: PPOAgent, slot_config: SlotConfig) -> None:
    """Score a fixed state/reward sequence with the CURRENT network and fill the buffer.

    Values come from the current critic, so as it trains, successive rollouts' stored
    values improve -> pre-update EV reflects the training. Actions do not affect returns
    (rewards are fixed per step), so K is the only thing that differs between legs.
    """
    state_dim = get_feature_size(slot_config)
    device = torch.device(agent.device)
    base = torch.linspace(-1.0, 1.0, steps=state_dim, device=device)
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
    for step, reward in enumerate(_REWARDS):
        state = (base + step * 0.05).unsqueeze(0)
        pre_hidden = hidden
        bp = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=device)
        r = agent.policy.network.get_action(
            state, bp, hidden,
            slot_mask=masks["slot"], blueprint_mask=masks["blueprint"],
            style_mask=masks["style"], tempo_mask=masks["tempo"],
            alpha_target_mask=masks["alpha_target"], alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"], op_mask=masks["op"],
        )
        hidden = r.hidden
        agent.buffer.add(
            env_id=0, state=state.squeeze(0),
            slot_action=r.actions["slot"].item(), blueprint_action=r.actions["blueprint"].item(),
            style_action=r.actions["style"].item(), tempo_action=r.actions["tempo"].item(),
            alpha_target_action=r.actions["alpha_target"].item(),
            alpha_speed_action=r.actions["alpha_speed"].item(),
            alpha_curve_action=r.actions["alpha_curve"].item(), op_action=r.actions["op"].item(),
            effective_op_action=r.actions["op"].item(),
            slot_log_prob=r.log_probs["slot"].item(), blueprint_log_prob=r.log_probs["blueprint"].item(),
            style_log_prob=r.log_probs["style"].item(), tempo_log_prob=r.log_probs["tempo"].item(),
            alpha_target_log_prob=r.log_probs["alpha_target"].item(),
            alpha_speed_log_prob=r.log_probs["alpha_speed"].item(),
            alpha_curve_log_prob=r.log_probs["alpha_curve"].item(), op_log_prob=r.log_probs["op"].item(),
            value=r.values.item(), reward=reward, done=step == len(_REWARDS) - 1, truncated=False,
            slot_mask=masks["slot"].squeeze(0), blueprint_mask=masks["blueprint"].squeeze(0),
            style_mask=masks["style"].squeeze(0), tempo_mask=masks["tempo"].squeeze(0),
            alpha_target_mask=masks["alpha_target"].squeeze(0),
            alpha_speed_mask=masks["alpha_speed"].squeeze(0),
            alpha_curve_mask=masks["alpha_curve"].squeeze(0), op_mask=masks["op"].squeeze(0),
            hidden_h=pre_hidden[0], hidden_c=pre_hidden[1], bootstrap_value=0.0,
            blueprint_indices=bp.squeeze(0),
        )
    agent.buffer.end_episode(0)


def _ev_trajectory(recurrent_n_epochs: int) -> list[float]:
    agent, slot_config = _build_agent(recurrent_n_epochs)
    traj: list[float] = []
    for _ in range(_CYCLES):
        _regenerate_rollout(agent, slot_config)
        metrics = agent.update(clear_buffer=True)
        traj.append(float(metrics["explained_variance"]))
    return traj


@pytest.mark.integration
def test_explained_variance_lifts_off_zero_at_k4_vs_k1() -> None:
    """K=4 lifts pre-update EV off ~0 past 0.3; K=1 stays pinned. (Acceptance #5.)"""
    ev_k1 = _ev_trajectory(1)
    ev_k4 = _ev_trajectory(4)
    final_k1, final_k4 = ev_k1[-1], ev_k4[-1]
    early_k4 = ev_k4[5]

    # 1. K=4 EV lifts off 0 past the spec's 0.3 target (observed ~0.47).
    assert final_k4 > 0.30, (
        f"K=4 EV failed to lift off 0 (final={final_k4:.4f}, traj tail={ev_k4[-5:]}). "
        "The multi-epoch critic fix did not raise explained_variance; investigate the "
        "value-target bootstrap (single-sample Q(s',op'), P0-1) or the gain=0.1 "
        "value-head-init secondary lever before concluding."
    )
    # 2. K=4 clears K=1 by a wide margin (observed ~5.3x) -> K is the lever.
    assert final_k4 > 3.0 * final_k1, (
        f"K=4 EV ({final_k4:.4f}) did not clearly exceed K=1 EV ({final_k1:.4f}); "
        "more gradient steps per rollout is not lifting EV as the spec predicts."
    )
    # 3. K=1 stays low (the dead-critic baseline the spec describes; observed ~0.09).
    assert final_k1 < 0.20, (
        f"K=1 EV ({final_k1:.4f}) unexpectedly high — the paired baseline is not isolating K."
    )
    # 4. K=4 genuinely CLIMBS (not high-at-init); EV rises from early to final.
    assert final_k4 > early_k4 + 0.10, (
        f"K=4 EV did not climb over the run (early={early_k4:.4f}, final={final_k4:.4f})."
    )
