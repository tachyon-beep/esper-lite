"""EV-stab Stage 2: ON-leg wiring for the HRA value decomposition (§4.5).

These tests exercise the parts the OFF-leg golden CANNOT reach: the cf value head in
the optimizer, the head-only V_cf loss term, the per-stream normalizers + EV keys, the
trunk-detach gradient isolation (the §2 provable-by-construction safety claim), and the
v3 checkpoint round-trip (three normalizers + flag-mismatch error).

The OFF-leg byte-identity gate lives in tests/simic/test_ppo_update_golden.py (unchanged)
and tests/simic/agent/test_rollout_buffer_cf_gae.py.
"""

from __future__ import annotations

from pathlib import Path

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
    VALUE_HEAD_SCHEMA_VERSION,
)
from esper.leyline.slot_config import SlotConfig
from esper.simic.agent import PPOAgent
from esper.simic.agent.ppo_agent import CHECKPOINT_VERSION
from esper.simic.rewards.partition import COMPONENT_TERMS
from esper.tamiyo.policy import create_policy
from esper.tamiyo.policy.features import get_feature_size


def _build_agent(
    *, hra: bool, recurrent_n_epochs: int = 1, return_variance_telemetry: bool = False
) -> tuple[PPOAgent, SlotConfig]:
    torch.manual_seed(123)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
        hra_value_decomposition=hra,
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
        hra_value_decomposition=hra,
        return_variance_telemetry=return_variance_telemetry,
    )
    return agent, slot_config


def _fill_buffer(
    agent: PPOAgent, slot_config: SlotConfig, *, hra: bool, with_components: bool = False
) -> None:
    """Deterministic 4-step episode. On the ON leg a deterministic r_cf stream + the
    network's own V_cf are threaded into the buffer (mirrors the rollout contract)."""
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
    r_cf_stream = [0.05, -0.02, 0.1, 0.0]
    for step, (reward, r_cf) in enumerate(zip(rewards, r_cf_stream)):
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
        if hra:
            assert result.cf_value is not None
            cf_value = result.cf_value.item()
            cf_kwargs = {"cf_value": cf_value, "cf_bootstrap_value": 0.0, "r_cf_norm": r_cf}
        else:
            assert result.cf_value is None
            cf_kwargs = {}
        if with_components:
            # Minimal signed decomposition that sums to reward: cf = r_cf stream,
            # residual = reward - r_cf, all other additends 0 (mirrors the rollout feed).
            comp = dict.fromkeys(COMPONENT_TERMS, 0.0)
            comp["bounded_attribution"] = r_cf
            comp["residual"] = reward - r_cf
            cf_kwargs = {**cf_kwargs, "component_additends": comp}
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
            **cf_kwargs,
        )
    agent.buffer.end_episode(0)


# ---------------------------------------------------------------------------
# Conditional construction (true bypass)
# ---------------------------------------------------------------------------

def test_off_leg_builds_no_cf_head_or_normalizers() -> None:
    agent, _ = _build_agent(hra=False)
    assert agent.policy.network.cf_value_head is None
    assert agent.value_main_normalizer is None
    assert agent.cf_value_normalizer is None


def test_on_leg_builds_cf_head_and_two_extra_normalizers() -> None:
    agent, _ = _build_agent(hra=True)
    assert agent.policy.network.cf_value_head is not None
    assert agent.value_main_normalizer is not None
    assert agent.cf_value_normalizer is not None
    # The total normalizer is untouched and distinct from the per-stream ones.
    assert agent.value_normalizer is not agent.value_main_normalizer
    assert agent.value_normalizer is not agent.cf_value_normalizer


# ---------------------------------------------------------------------------
# Optimizer critic-group registration (§7 head registration)
# ---------------------------------------------------------------------------

def test_cf_head_params_in_optimizer_iff_on_with_weight_decay() -> None:
    # weight_decay > 0 takes the explicit-groups branch where critic_params is enumerated.
    on = _build_agent_with_wd(hra=True, weight_decay=0.01)
    off = _build_agent_with_wd(hra=False, weight_decay=0.01)

    cf_params = {id(p) for p in on.policy.network.cf_value_head.parameters()}
    critic_group = next(g for g in on.optimizer.param_groups if g["name"] == "critic")
    critic_ids = {id(p) for p in critic_group["params"]}
    assert cf_params, "cf head must have parameters"
    assert cf_params <= critic_ids, "cf_value_head params must be in the critic optimizer group (ON)"

    # OFF leg: no cf head, so no cf params anywhere; critic group is V_main + q_head only.
    assert off.policy.network.cf_value_head is None


def _build_agent_with_wd(*, hra: bool, weight_decay: float) -> PPOAgent:
    torch.manual_seed(123)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
        hra_value_decomposition=hra,
    )
    return PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=1,
        max_steps_per_env=4,
        chunk_length=4,
        device="cpu",
        target_kl=None,
        recurrent_n_epochs=1,
        weight_decay=weight_decay,
        hra_value_decomposition=hra,
    )


# ---------------------------------------------------------------------------
# Head-only gradient isolation (the §2 provable-by-construction safety claim)
# ---------------------------------------------------------------------------

def test_cf_value_is_trunk_detached_head_only_critic() -> None:
    agent, slot_config = _build_agent(hra=True)
    net = agent.policy.network
    state_dim = get_feature_size(slot_config)
    state = torch.randn(2, 3, state_dim)
    bp = torch.zeros(2, 3, slot_config.num_slots, dtype=torch.long)
    hidden = net.get_initial_hidden(2, torch.device("cpu"))
    out = net.forward(state, bp, hidden)

    net.zero_grad()
    out["cf_value"].sum().backward()

    lstm_grad = sum(
        p.grad.abs().sum().item() for p in net.lstm.parameters() if p.grad is not None
    )
    cf_head_grad = sum(
        p.grad.abs().sum().item()
        for p in net.cf_value_head.parameters()
        if p.grad is not None
    )
    assert lstm_grad == 0.0, "V_cf must add NO gradient to the LSTM trunk (head-only/detached)"
    assert cf_head_grad > 0.0, "the cf head itself must receive gradient"


# ---------------------------------------------------------------------------
# End-to-end ON update: cf loss term + per-stream EV keys + nonzero cf-head grad
# ---------------------------------------------------------------------------

def test_on_leg_update_emits_cf_loss_and_per_stream_ev() -> None:
    agent, slot_config = _build_agent(hra=True)
    _fill_buffer(agent, slot_config, hra=True)
    # Snapshot cf-head weights to prove the head actually trains under one update.
    before = [p.detach().clone() for p in agent.policy.network.cf_value_head.parameters()]
    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True
    assert metrics["finiteness_gate_skip_count"] == 0
    assert metrics["cf_value_loss"] > 0.0
    # Per-stream EV keys + EV-stab Stage 0 GATE metrics present and finite on the ON leg.
    for key in ("ev_main", "ev_cf", "ev_sum", "cov_rcf_return_share", "r_main_cov"):
        assert key in metrics
        assert torch.isfinite(torch.tensor(metrics[key]))
    # The cf head moved (its loss backprops into its own params).
    after = list(agent.policy.network.cf_value_head.parameters())
    moved = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert moved, "cf_value_head weights must change after an ON update"


def test_on_leg_update_emits_per_stream_target_scales() -> None:
    """Stage-2 §9 diagnostic scalars (gate doc): the per-stream normalizer scales, so a
    reader can tell 'critic got better' from 'target got trivially easy'. ON-leg-only;
    post-update() scale, matching the total value_target_scale convention."""
    agent, slot_config = _build_agent(hra=True)
    _fill_buffer(agent, slot_config, hra=True)
    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True
    for key in ("value_main_target_scale", "cf_value_target_scale"):
        assert key in metrics
        value = torch.tensor(metrics[key])
        assert torch.isfinite(value)
        assert value > 0.0  # a normalizer scale is strictly positive


def test_off_leg_update_emits_no_cf_or_ev_keys() -> None:
    # OFF-leg byte-identity: the returned metrics dict is a contract site and must gain NO
    # new keys (cf_value_loss / ev_main / ev_cf / ev_sum are ON-leg-only). The dual_ab
    # OFF-config training path relies on the strict per-key reducer whitelist, so an
    # unexpected key there is a hard KeyError, not a silent extra.
    agent, slot_config = _build_agent(hra=False)
    _fill_buffer(agent, slot_config, hra=False)
    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True
    for key in (
        "cf_value_loss",
        "ev_main",
        "ev_cf",
        "ev_sum",
        "cov_rcf_return_share",
        "r_main_cov",
        # Stage-2 §9 scalars (S7 emission) are ON-leg-only too.
        "value_main_target_scale",
        "cf_value_target_scale",
    ):
        assert key not in metrics


_RETURN_VAR_KEYS = (
    "return_var_cf_share",
    "return_var_main_share",
    "return_var_residual_share",
)


def test_return_variance_gate_emitted_when_flag_on() -> None:
    """The value-free Stage-0 gate is emitted from update() when the flag is ON — and on
    the HRA-OFF leg (it is the CONTROL-run gate, independent of hra_value_decomposition)."""
    agent, slot_config = _build_agent(hra=False, return_variance_telemetry=True)
    _fill_buffer(agent, slot_config, hra=False, with_components=True)
    metrics = agent.update(clear_buffer=True)

    assert metrics["ppo_update_performed"] is True
    for key in _RETURN_VAR_KEYS:
        assert key in metrics, f"missing value-free gate key {key!r}"
        assert torch.isfinite(torch.tensor(metrics[key])).all()


def test_return_variance_gate_absent_when_flag_off() -> None:
    """Flag OFF (default): the gate keys are absent, so the metrics-dict contract and the
    strict reducer whitelist are byte-identical to the pre-Stage-0 baseline."""
    agent, slot_config = _build_agent(hra=False)  # return_variance_telemetry defaults OFF
    _fill_buffer(agent, slot_config, hra=False)
    metrics = agent.update(clear_buffer=True)

    for key in _RETURN_VAR_KEYS:
        assert key not in metrics


# ---------------------------------------------------------------------------
# ON-leg regression anchor (deterministic; mirrors the golden posture)
# ---------------------------------------------------------------------------

# Pinned 2026-06-24 from the deterministic ON build (seed 123, the _fill_buffer above).
# These are the new-architecture deterministic output; regenerate + re-pin only on an
# intentional ON-leg change. Tolerance mirrors the golden file's strict band.
_ON_GOLDENS: dict[int, dict[str, float]] = {
    1: {
        "policy_loss": -2.0044755935668945,
        "value_loss": 0.007826905697584152,
        "cf_value_loss": 0.009760173037648201,
        "entropy": 6.799691200256348,
    },
    4: {
        "policy_loss": -2.425248146057129,
        "value_loss": 0.005328205414116383,
        "cf_value_loss": 0.0037804325111210346,
        "entropy": 6.764498710632324,
    },
}


@pytest.mark.parametrize("recurrent_n_epochs", [1, 4])
def test_on_leg_golden_metrics(recurrent_n_epochs: int) -> None:
    agent, slot_config = _build_agent(hra=True, recurrent_n_epochs=recurrent_n_epochs)
    _fill_buffer(agent, slot_config, hra=True)
    metrics = agent.update(clear_buffer=True)
    golden = _ON_GOLDENS[recurrent_n_epochs]
    for key, expected in golden.items():
        assert metrics[key] == pytest.approx(expected, abs=1e-6), key


# ---------------------------------------------------------------------------
# Checkpoint round-trip (v3 three normalizers + flag mismatch)
# ---------------------------------------------------------------------------

def test_on_checkpoint_roundtrip_restores_three_normalizers_and_cf_head(
    tmp_path: Path,
) -> None:
    agent, slot_config = _build_agent(hra=True)
    _fill_buffer(agent, slot_config, hra=True)
    agent.update(clear_buffer=True)  # populate all three normalizer running stats

    path = tmp_path / "agent_on.pt"
    agent.save(path)

    loaded = PPOAgent.load(path, device="cpu")
    assert loaded.hra_value_decomposition is True
    assert loaded.policy.network.cf_value_head is not None
    assert loaded.value_main_normalizer is not None
    assert loaded.cf_value_normalizer is not None
    # Normalizer running stats restored.
    assert torch.allclose(
        loaded.value_main_normalizer.var, agent.value_main_normalizer.var
    )
    assert torch.allclose(loaded.cf_value_normalizer.var, agent.cf_value_normalizer.var)
    # cf head weights restored bit-exact.
    for orig_p, load_p in zip(
        agent.policy.network.cf_value_head.parameters(),
        loaded.policy.network.cf_value_head.parameters(),
    ):
        assert torch.equal(orig_p, load_p)


def test_off_checkpoint_roundtrip_has_no_cf_state(tmp_path: Path) -> None:
    agent, _ = _build_agent(hra=False)
    path = tmp_path / "agent_off.pt"
    agent.save(path)
    loaded = PPOAgent.load(path, device="cpu")
    assert loaded.hra_value_decomposition is False
    assert loaded.policy.network.cf_value_head is None
    assert loaded.value_main_normalizer is None
    assert loaded.cf_value_normalizer is None


def test_loading_on_checkpoint_into_off_build_raises_descriptive_error(
    tmp_path: Path,
) -> None:
    agent, _ = _build_agent(hra=True)
    path = tmp_path / "agent_on.pt"
    agent.save(path)

    # Tamper the recorded flag to OFF so the rebuilt network drops the cf head while the
    # saved network_state_dict still carries cf_value_head.* keys -> strict load fails.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["config"]["hra_value_decomposition"] = False
    with pytest.raises(RuntimeError):
        PPOAgent.load_from_checkpoint_dict(checkpoint, device="cpu")


def test_checkpoint_versions_bumped_to_three() -> None:
    assert CHECKPOINT_VERSION == 3
    assert VALUE_HEAD_SCHEMA_VERSION == 3


# ---------------------------------------------------------------------------
# Vectorized training aggregator: the ON-leg metric keys must have a declared
# reducer (the aggregator hard-fails on any undeclared key). This guards the
# train_ppo_vectorized ON-leg path without spinning up a full training run.
# ---------------------------------------------------------------------------

def test_aggregator_reduces_on_leg_metric_keys() -> None:
    from esper.simic.training.vectorized import _aggregate_ppo_metrics

    update_metrics = [
        {
            "cf_value_loss": 0.2,
            "ev_main": 0.4,
            "ev_cf": 0.1,
            "ev_sum": 0.3,
            "cov_rcf_return_share": 0.5,
            "r_main_cov": 0.8,
        },
        {
            "cf_value_loss": 0.4,
            "ev_main": 0.6,
            "ev_cf": 0.3,
            "ev_sum": 0.5,
            "cov_rcf_return_share": 0.7,
            "r_main_cov": 1.0,
        },
    ]
    aggregated = _aggregate_ppo_metrics(update_metrics)
    assert aggregated["cf_value_loss"] == pytest.approx(0.3)
    assert aggregated["ev_main"] == pytest.approx(0.5)
    assert aggregated["ev_cf"] == pytest.approx(0.2)
    assert aggregated["ev_sum"] == pytest.approx(0.4)
    assert aggregated["cov_rcf_return_share"] == pytest.approx(0.6)
    assert aggregated["r_main_cov"] == pytest.approx(0.9)
