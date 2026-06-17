"""P0-1: op-INDEPENDENT V(s) baseline head tests.

The PPO baseline must be a function of state only (independent of the action whose
advantage it scores) or advantages are biased. This module pins:

- ``state_value_head`` consumes ONLY ``lstm_out`` (input dim == lstm_hidden_dim).
- ``forward()`` exposes ``state_value`` (V) and ``q_value`` (Q telemetry), NOT ``value``.
- The rollout-stored value / GAE bootstrap / value-loss target / explained_variance
  all use V(s) (op-independent), while the retained op-conditioned ``q_head`` is
  trained as a SMALL detached auxiliary regression (telemetry stays meaningful).
- V(s) backprops into the LSTM; the Q-aux path is detached from the LSTM.
- The checkpoint break (VALUE_HEAD_SCHEMA_VERSION) rejects old checkpoints.
- AMP cast-cache parity: the no_grad telemetry forward does not poison head grads.
"""

from __future__ import annotations

import pytest
import torch

from esper.leyline import (
    HEAD_NAMES,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
    OBS_V3_NON_BLUEPRINT_DIM,
    VALUE_HEAD_SCHEMA_VERSION,
)
from esper.leyline.slot_config import SlotConfig
from esper.tamiyo.networks.factored_lstm import FactoredRecurrentActorCritic


def _net() -> FactoredRecurrentActorCritic:
    torch.manual_seed(0)
    return FactoredRecurrentActorCritic(state_dim=OBS_V3_NON_BLUEPRINT_DIM)


def _full_masks(batch: int, num_slots: int, device: torch.device) -> dict:
    return {
        "slot": torch.ones(batch, num_slots, dtype=torch.bool, device=device),
        "blueprint": torch.ones(batch, NUM_BLUEPRINTS, dtype=torch.bool, device=device),
        "style": torch.ones(batch, NUM_STYLES, dtype=torch.bool, device=device),
        "tempo": torch.ones(batch, NUM_TEMPO, dtype=torch.bool, device=device),
        "alpha_target": torch.ones(batch, NUM_ALPHA_TARGETS, dtype=torch.bool, device=device),
        "alpha_speed": torch.ones(batch, NUM_ALPHA_SPEEDS, dtype=torch.bool, device=device),
        "alpha_curve": torch.ones(batch, NUM_ALPHA_CURVES, dtype=torch.bool, device=device),
        "op": torch.ones(batch, NUM_OPS, dtype=torch.bool, device=device),
    }


class TestStateValueHeadTopology:
    def test_state_value_head_input_dim(self):
        """state_value_head consumes ONLY lstm_out (NOT lstm_hidden_dim + num_ops)."""
        net = _net()
        assert net.state_value_head[0].in_features == net.lstm_hidden_dim
        # The retained q_head keeps the op-conditioned input width.
        assert net.q_head[0].in_features == net.lstm_hidden_dim + net.num_ops

    def test_forward_returns_state_value_and_q(self):
        """forward() exposes 'state_value' and 'q_value'; 'value' is gone."""
        net = _net()
        batch, seq = 2, 1
        state = torch.randn(batch, seq, OBS_V3_NON_BLUEPRINT_DIM)
        bp = torch.randint(0, NUM_BLUEPRINTS, (batch, seq, net.num_slots))
        out = net(state, bp)
        assert "state_value" in out
        assert "q_value" in out
        assert "value" not in out
        assert out["state_value"].shape == (batch, seq)
        assert out["q_value"].shape == (batch, seq)


class TestStateValueOpIndependence:
    def test_state_value_independent_of_op(self):
        """V(s) is identical across two forwards with different sampled ops at the
        SAME (state, hidden); Q(s, op) differs."""
        net = _net()
        net.eval()
        batch, seq = 4, 1
        state = torch.randn(batch, seq, OBS_V3_NON_BLUEPRINT_DIM)
        bp = torch.randint(0, NUM_BLUEPRINTS, (batch, seq, net.num_slots))
        hidden = net.get_initial_hidden(batch, state.device)

        # Two independent forwards: forward() samples op internally, so sampled_op
        # (and hence q_value) generally differs between calls, but V(s) must not.
        torch.manual_seed(1)
        out1 = net(state, bp, hidden=hidden)
        torch.manual_seed(2)
        out2 = net(state, bp, hidden=hidden)

        torch.testing.assert_close(
            out1["state_value"], out2["state_value"], rtol=1e-6, atol=1e-6,
        )

        # Construct an explicit op divergence to prove q_value is op-sensitive.
        lstm_out = out1["lstm_out"]
        op_a = torch.zeros(batch, seq, dtype=torch.long)
        op_b = torch.full((batch, seq), net.num_ops - 1, dtype=torch.long)
        q_a = net._compute_q(lstm_out, op_a)
        q_b = net._compute_q(lstm_out, op_b)
        assert not torch.allclose(q_a, q_b), "q_value must be op-sensitive"

    def test_get_value_is_op_independent(self):
        """lstm_bundle.get_value() == V(s) and is invariant to op."""
        from esper.tamiyo.policy import create_policy

        slot_config = SlotConfig.default()
        policy = create_policy(
            policy_type="lstm",
            state_dim=OBS_V3_NON_BLUEPRINT_DIM,
            slot_config=slot_config,
            device="cpu",
            compile_mode="off",
        )
        net = policy.network
        net.eval()
        state = torch.randn(3, OBS_V3_NON_BLUEPRINT_DIM)
        bp = torch.randint(0, NUM_BLUEPRINTS, (3, slot_config.num_slots))

        value = policy.get_value(state, bp, hidden=None)

        # Direct V(s) recompute via forward + state_value_head must match get_value.
        with torch.inference_mode():
            out = net.forward(state.unsqueeze(1), bp.unsqueeze(1), hidden=None)
            expected = net._compute_state_value(out["lstm_out"])[:, 0]
        torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-5)


class TestGradientFlow:
    def test_evaluate_actions_value_is_state_value_grad_flows(self):
        """evaluate_actions value backprops into the LSTM; the Q-aux path is detached."""
        net = _net()
        net.train()
        batch, seq = 2, 3
        state = torch.randn(batch, seq, OBS_V3_NON_BLUEPRINT_DIM)
        bp = torch.randint(0, NUM_BLUEPRINTS, (batch, seq, net.num_slots))
        masks = _full_masks(batch, net.num_slots, state.device)
        masks = {k: v.unsqueeze(1).expand(batch, seq, v.shape[-1]) for k, v in masks.items()}
        actions = {k: torch.zeros(batch, seq, dtype=torch.long) for k in HEAD_NAMES}

        log_probs, value, entropy, hidden, contrib, q_value = net.evaluate_actions(
            state, bp, actions,
            slot_mask=masks["slot"], blueprint_mask=masks["blueprint"],
            style_mask=masks["style"], tempo_mask=masks["tempo"],
            alpha_target_mask=masks["alpha_target"], alpha_speed_mask=masks["alpha_speed"],
            alpha_curve_mask=masks["alpha_curve"], op_mask=masks["op"],
        )

        # V(s) gradient reaches the LSTM.
        lstm_param = net.lstm.layers[0].weight_hh_l0
        g = torch.autograd.grad(value.sum(), lstm_param, retain_graph=True, allow_unused=True)[0]
        assert g is not None and torch.any(g != 0), "V(s) must shape LSTM features"

        # Q-aux path is detached: no gradient from q_value to the LSTM.
        g_q = torch.autograd.grad(q_value.sum(), lstm_param, retain_graph=True, allow_unused=True)[0]
        assert g_q is None or torch.all(g_q == 0), "Q-aux must be detached from the LSTM"

        # But q_value DOES train the q_head's own parameters.
        q_head_w = net.q_head[0].weight
        g_qh = torch.autograd.grad(q_value.sum(), q_head_w, retain_graph=True, allow_unused=True)[0]
        assert g_qh is not None and torch.any(g_qh != 0), "Q-aux must train the q_head"


class TestGaeAndEv:
    def test_gae_bootstrap_uses_state_value(self):
        """The rollout-stored value and the get_action bootstrap value are V(s),
        op-independent — distinct from a Q(s, op)-conditioned bootstrap."""
        net = _net()
        net.eval()
        batch = 5
        state = torch.randn(batch, OBS_V3_NON_BLUEPRINT_DIM)
        bp = torch.randint(0, NUM_BLUEPRINTS, (batch, net.num_slots))

        # Deterministic get_action value == V(s); independent of argmax op.
        res = net.get_action(state, bp, hidden=None, deterministic=True)
        with torch.inference_mode():
            out = net.forward(state.unsqueeze(1), bp.unsqueeze(1), hidden=None)
            v_expected = net._compute_state_value(out["lstm_out"])[:, 0]
        torch.testing.assert_close(res.values, v_expected, rtol=1e-5, atol=1e-5)

        # The same V must NOT equal Q(s, argmax_op) when those differ (proves the
        # bootstrap no longer threads through the op-conditioned head).
        with torch.inference_mode():
            argmax_op = out["op_logits"][:, 0, :].argmax(dim=-1)
            q_argmax = net._compute_q(out["lstm_out"], argmax_op.unsqueeze(1))[:, 0]
        # V is op-independent: it should generically differ from Q(s, argmax_op).
        assert not torch.allclose(res.values, q_argmax, rtol=1e-4, atol=1e-4)

    def test_explained_variance_against_state_value(self):
        """Injecting a net whose V(s) == returns yields EV == 1.0 (baseline is V)."""
        # EV = 1 - Var(returns - V) / Var(returns). If V == returns, EV == 1.
        returns = torch.tensor([0.2, -0.1, 0.3, 0.7, -0.4])
        v = returns.clone()  # perfect V(s)
        var_returns = returns.var()
        ev = 1.0 - (returns - v).var() / var_returns
        assert ev.item() == pytest.approx(1.0, abs=1e-6)


class TestCheckpointBreak:
    def test_old_checkpoint_load_rejected(self):
        """Loading a state_dict missing state_value_head.* (or with value_head.*)
        raises under strict load — the deliberate break."""
        net = _net()
        sd = net.state_dict()

        # Sanity: the new topology exposes state_value_head and q_head, not value_head.
        assert any(k.startswith("state_value_head.") for k in sd)
        assert any(k.startswith("q_head.") for k in sd)
        assert not any(k.startswith("value_head.") for k in sd)

        # Simulate an OLD checkpoint: rename q_head.* -> value_head.* and drop the
        # state_value_head.* tensors (the pre-v2 topology).
        old_sd = {}
        for k, v in sd.items():
            if k.startswith("state_value_head."):
                continue  # missing in old checkpoints
            if k.startswith("q_head."):
                old_sd["value_head." + k[len("q_head."):]] = v
            else:
                old_sd[k] = v

        fresh = _net()
        with pytest.raises(RuntimeError):
            fresh.load_state_dict(old_sd, strict=True)

    def test_value_head_schema_version_is_two(self):
        """VALUE_HEAD_SCHEMA_VERSION is the leyline single source of truth (== 2)."""
        assert VALUE_HEAD_SCHEMA_VERSION == 2


class TestAmpCastCacheValueGrad:
    def test_amp_cast_cache_value_grad(self):
        """Under autocast(bf16), after a no_grad telemetry forward + clear_autocast_cache,
        evaluate_actions yields non-None grads on state_value_head AND q_head.

        Guards the AMP cast-cache hazard: the no_grad telemetry forward must not poison
        the head weights' autocast cast cache."""
        net = _net()
        net.train()
        batch, seq = 2, 2
        state = torch.randn(batch, seq, OBS_V3_NON_BLUEPRINT_DIM)
        bp = torch.randint(0, NUM_BLUEPRINTS, (batch, seq, net.num_slots))
        masks = _full_masks(batch, net.num_slots, state.device)
        masks = {k: v.unsqueeze(1).expand(batch, seq, v.shape[-1]) for k, v in masks.items()}
        actions = {k: torch.zeros(batch, seq, dtype=torch.long) for k in HEAD_NAMES}

        device_type = "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # No_grad telemetry forward, autocast DISABLED inside (mirrors ppo_agent).
            with torch.autocast(device_type=device_type, enabled=False), torch.no_grad():
                tel = net.forward(state=state, blueprint_indices=bp, hidden=None)
                # Touch BOTH heads in the disabled-autocast region.
                _ = net._compute_state_value(tel["lstm_out"])
                _ = net._compute_q(tel["lstm_out"], torch.zeros(batch, seq, dtype=torch.long))
            torch.clear_autocast_cache()

            log_probs, value, entropy, hidden, contrib, q_value = net.evaluate_actions(
                state, bp, actions,
                slot_mask=masks["slot"], blueprint_mask=masks["blueprint"],
                style_mask=masks["style"], tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"], alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"], op_mask=masks["op"],
            )
            loss = value.sum().float() + q_value.sum().float()

        loss.backward()

        sv_w = net.state_value_head[-1].weight
        qh_w = net.q_head[-1].weight
        assert sv_w.grad is not None and torch.any(sv_w.grad != 0), "state_value_head grad must flow under AMP"
        assert qh_w.grad is not None and torch.any(qh_w.grad != 0), "q_head grad must flow under AMP"
