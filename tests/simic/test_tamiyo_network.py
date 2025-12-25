"""Tests for FactoredRecurrentActorCritic - Tamiyo's neural network."""

import pytest
import torch

from esper.leyline import (
    GerminationStyle,
    LifecycleOp,
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)
from esper.tamiyo.policy.action_masks import InvalidStateMachineError
from esper.tamiyo.networks import FactoredRecurrentActorCritic


class TestFactoredRecurrentActorCritic:
    """Tests for the factored recurrent network."""

    def test_forward_returns_all_heads(self):
        """Forward pass must return logits for all factored heads."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 1, 50)  # [batch, seq, state_dim]

        output = net(state)

        assert "slot_logits" in output
        assert "blueprint_logits" in output
        assert "style_logits" in output
        assert "tempo_logits" in output
        assert "alpha_target_logits" in output
        assert "alpha_speed_logits" in output
        assert "alpha_curve_logits" in output
        assert "op_logits" in output
        assert "value" in output
        assert "hidden" in output

        # Check shapes
        assert output["slot_logits"].shape == (2, 1, 3)  # NUM_SLOTS=3
        assert output["blueprint_logits"].shape == (2, 1, NUM_BLUEPRINTS)
        assert output["style_logits"].shape == (2, 1, NUM_STYLES)
        assert output["tempo_logits"].shape == (2, 1, NUM_TEMPO)
        assert output["alpha_target_logits"].shape == (2, 1, NUM_ALPHA_TARGETS)
        assert output["alpha_speed_logits"].shape == (2, 1, NUM_ALPHA_SPEEDS)
        assert output["alpha_curve_logits"].shape == (2, 1, NUM_ALPHA_CURVES)
        assert output["op_logits"].shape == (2, 1, NUM_OPS)
        assert output["value"].shape == (2, 1)

    def test_hidden_state_propagates(self):
        """LSTM hidden state must propagate across time steps."""
        net = FactoredRecurrentActorCritic(state_dim=50, lstm_hidden_dim=64)

        batch_size = 2
        hidden = net.get_initial_hidden(batch_size, torch.device("cpu"))

        # First step
        state1 = torch.randn(batch_size, 1, 50)
        output1 = net(state1, hidden=hidden)

        # Second step with updated hidden
        state2 = torch.randn(batch_size, 1, 50)
        output2 = net(state2, hidden=output1["hidden"])

        # Hidden states should be different after processing different inputs
        h1, c1 = output1["hidden"]
        h2, c2 = output2["hidden"]
        assert not torch.allclose(h1, h2), "Hidden states should change between steps"

    def test_masks_applied_correctly(self):
        """Invalid actions must have large negative logits after masking."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(1, 1, 50)

        # Mask out slot 0 and 2, only slot 1 valid
        slot_mask = torch.tensor([[[False, True, False]]])
        op_mask = torch.ones(1, 1, NUM_OPS, dtype=torch.bool)

        output = net(
            state,
            slot_mask=slot_mask,
            blueprint_mask=None,
            style_mask=None,
            op_mask=op_mask,
        )

        slot_logits = output["slot_logits"][0, 0]
        # Masked actions should have large negative value (not -inf for FP16 safety)
        assert slot_logits[0] < -1000, "Masked action should have large negative logit"
        assert slot_logits[1] > -1000, "Valid action should have normal logit"
        assert slot_logits[2] < -1000, "Masked action should have large negative logit"

    def test_per_head_log_probs(self):
        """evaluate_actions must return per-head log probs."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 5, 50)  # [batch, seq, state_dim]
        actions = {
            "slot": torch.zeros(2, 5, dtype=torch.long),
            "blueprint": torch.zeros(2, 5, dtype=torch.long),
            "style": torch.zeros(2, 5, dtype=torch.long),
            "tempo": torch.zeros(2, 5, dtype=torch.long),
            "alpha_target": torch.zeros(2, 5, dtype=torch.long),
            "alpha_speed": torch.zeros(2, 5, dtype=torch.long),
            "alpha_curve": torch.zeros(2, 5, dtype=torch.long),
            "op": torch.zeros(2, 5, dtype=torch.long),
        }

        log_probs, values, entropy, hidden = net.evaluate_actions(state, actions)

        # Per-head log probs
        assert "slot" in log_probs
        assert "blueprint" in log_probs
        assert "style" in log_probs
        assert "tempo" in log_probs
        assert "alpha_target" in log_probs
        assert "alpha_speed" in log_probs
        assert "alpha_curve" in log_probs
        assert "op" in log_probs

        # All should be [batch, seq]
        assert log_probs["slot"].shape == (2, 5)
        assert log_probs["blueprint"].shape == (2, 5)

    def test_get_action_returns_per_head_log_probs(self):
        """get_action must return per-head log probs for buffer storage."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 50)  # [batch, state_dim]

        result = net.get_action(state)

        # Should have per-head actions and log_probs
        assert "slot" in result.actions
        assert "blueprint" in result.actions
        assert "style" in result.actions
        assert "tempo" in result.actions
        assert "alpha_target" in result.actions
        assert "alpha_speed" in result.actions
        assert "alpha_curve" in result.actions
        assert "op" in result.actions

        assert "slot" in result.log_probs
        assert "blueprint" in result.log_probs
        assert "style" in result.log_probs
        assert "tempo" in result.log_probs
        assert "alpha_target" in result.log_probs
        assert "alpha_speed" in result.log_probs
        assert "alpha_curve" in result.log_probs
        assert "op" in result.log_probs

        # Actions should be batch-sized
        assert result.actions["slot"].shape == (2,)
        assert result.log_probs["slot"].shape == (2,)

    def test_entropy_normalized_per_head(self):
        """Entropy should be normalized by max entropy for each head."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(2, 5, 50)
        actions = {
            "slot": torch.zeros(2, 5, dtype=torch.long),
            "blueprint": torch.zeros(2, 5, dtype=torch.long),
            "style": torch.zeros(2, 5, dtype=torch.long),
            "tempo": torch.zeros(2, 5, dtype=torch.long),
            "alpha_target": torch.zeros(2, 5, dtype=torch.long),
            "alpha_speed": torch.zeros(2, 5, dtype=torch.long),
            "alpha_curve": torch.zeros(2, 5, dtype=torch.long),
            "op": torch.zeros(2, 5, dtype=torch.long),
        }

        _, _, entropy, _ = net.evaluate_actions(state, actions)

        # Normalized entropy should be between 0 and 1
        for key in [
            "slot",
            "blueprint",
            "style",
            "tempo",
            "alpha_target",
            "alpha_speed",
            "alpha_curve",
            "op",
        ]:
            assert (entropy[key] >= 0).all(), f"{key} entropy has negative values"
            assert (entropy[key] <= 1.01).all(), f"{key} entropy exceeds 1 (not normalized)"

    def test_deterministic_action(self):
        """Deterministic mode should always return argmax."""
        net = FactoredRecurrentActorCritic(state_dim=50)
        state = torch.randn(1, 50)

        # Run multiple times - deterministic should be consistent
        result1 = net.get_action(state, deterministic=True)
        result2 = net.get_action(state, deterministic=True)

        for key in [
            "slot",
            "blueprint",
            "style",
            "tempo",
            "alpha_target",
            "alpha_speed",
            "alpha_curve",
            "op",
        ]:
            assert result1.actions[key] == result2.actions[key], f"{key} action not deterministic"


def test_style_mask_forces_default_when_not_germinate():
    """Non-GERMINATE ops should force style to a single default choice."""
    net = FactoredRecurrentActorCritic(state_dim=20)
    state = torch.randn(3, 20)

    op_mask = torch.zeros(3, NUM_OPS, dtype=torch.bool)
    op_mask[:, LifecycleOp.WAIT] = True
    style_mask = torch.ones(3, NUM_STYLES, dtype=torch.bool)

    result = net.get_action(
        state,
        op_mask=op_mask,
        style_mask=style_mask,
    )

    assert (result.actions["op"] == LifecycleOp.WAIT).all()
    assert (result.actions["style"] == int(GerminationStyle.SIGMOID_ADD)).all()


def test_style_not_forced_for_set_alpha_target():
    """SET_ALPHA_TARGET should not force style to the default choice."""
    net = FactoredRecurrentActorCritic(state_dim=20)
    state = torch.randn(2, 20)

    # Force op = SET_ALPHA_TARGET
    op_mask = torch.zeros(2, NUM_OPS, dtype=torch.bool)
    op_mask[:, LifecycleOp.SET_ALPHA_TARGET] = True

    # Force style logits to prefer GATED_GATE (a non-default style).
    style_mask = torch.ones(2, NUM_STYLES, dtype=torch.bool)
    with torch.no_grad():
        # style_head is Linear -> ReLU -> Linear; zeroing the first layer makes the
        # second layer bias dominate, giving deterministic argmax control.
        net.style_head[0].weight.zero_()
        net.style_head[0].bias.zero_()
        net.style_head[2].weight.zero_()
        net.style_head[2].bias.zero_()
        net.style_head[2].bias[int(GerminationStyle.GATED_GATE)] = 10.0

    result = net.get_action(
        state,
        deterministic=True,
        op_mask=op_mask,
        style_mask=style_mask,
    )

    assert (result.actions["op"] == LifecycleOp.SET_ALPHA_TARGET).all()
    assert (result.actions["style"] == int(GerminationStyle.GATED_GATE)).all()


def test_masking_produces_valid_softmax():
    """Verify masking produces valid probabilities after softmax.

    Critical test from PyTorch expert review: must verify softmax doesn't
    produce NaN/Inf, not just that mask value is finite.
    """
    net = FactoredRecurrentActorCritic(state_dim=35)
    state = torch.randn(2, 3, 35)

    # Mask that disables some actions
    slot_mask = torch.ones(2, 3, 3, dtype=torch.bool)
    slot_mask[:, :, 1] = False  # Mask out middle action

    output = net.forward(state, slot_mask=slot_mask)

    # Test that _MASK_VALUE produces valid softmax across dtypes
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        logits = output["slot_logits"].to(dtype)
        probs = torch.softmax(logits, dim=-1)

        # Masked positions should have ~0 probability
        assert (probs[:, :, 1] < 1e-3).all(), \
            f"Masked action should have near-zero probability with {dtype}"
        # Valid positions should have valid probabilities
        assert not torch.isnan(probs).any(), \
            f"Softmax should not produce NaN with {dtype}"
        assert not torch.isinf(probs).any(), \
            f"Softmax should not produce Inf with {dtype}"
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 3, dtype=dtype), atol=1e-2), \
            f"Probabilities should sum to 1 with {dtype}"


def test_logits_no_inf_after_masking():
    """Verify masked logits use large negative value, not -inf.

    This is critical for torch.compile and mixed precision compatibility.
    """
    net = FactoredRecurrentActorCritic(state_dim=35)
    state = torch.randn(2, 3, 35)

    # Mask that disables some actions
    slot_mask = torch.ones(2, 3, 3, dtype=torch.bool)
    slot_mask[:, :, 1] = False  # Mask out middle action

    output = net.forward(state, slot_mask=slot_mask)

    # Should not contain inf - masked values should use -1e4, not -inf
    for key in [
        "slot_logits",
        "blueprint_logits",
        "style_logits",
        "tempo_logits",
        "alpha_target_logits",
        "alpha_speed_logits",
        "alpha_curve_logits",
        "op_logits",
        "value",
    ]:
        tensor = output[key]
        assert not torch.isinf(tensor).any(), f"{key} should not contain -inf"
        assert not torch.isnan(tensor).any(), f"{key} should not contain NaN"


def test_entropy_normalization_with_single_action():
    """Verify entropy normalization handles single-action case gracefully."""
    # Edge case: num_slots=1 means log(1)=0, could cause division issues
    net = FactoredRecurrentActorCritic(
        state_dim=35,
        num_slots=1,  # log(1) = 0!
        num_blueprints=5,
        num_styles=NUM_STYLES,
        num_tempo=NUM_TEMPO,
        num_alpha_targets=NUM_ALPHA_TARGETS,
        num_alpha_speeds=NUM_ALPHA_SPEEDS,
        num_alpha_curves=NUM_ALPHA_CURVES,
        num_ops=NUM_OPS,
    )

    state = torch.randn(2, 3, 35)
    actions = {
        "slot": torch.zeros(2, 3, dtype=torch.long),  # Only one option
        "blueprint": torch.randint(0, 5, (2, 3)),
        "style": torch.randint(0, NUM_STYLES, (2, 3)),
        "tempo": torch.randint(0, NUM_TEMPO, (2, 3)),
        "alpha_target": torch.randint(0, NUM_ALPHA_TARGETS, (2, 3)),
        "alpha_speed": torch.randint(0, NUM_ALPHA_SPEEDS, (2, 3)),
        "alpha_curve": torch.randint(0, NUM_ALPHA_CURVES, (2, 3)),
        "op": torch.randint(0, NUM_OPS, (2, 3)),
    }

    log_probs, values, entropy, hidden = net.evaluate_actions(state, actions)

    # Entropy for single-action head should be 0 (no uncertainty), not inf/nan
    assert not torch.isnan(entropy["slot"]).any(), "Entropy should not be NaN"
    assert not torch.isinf(entropy["slot"]).any(), "Entropy should not be Inf"
    # With single action, normalized entropy should be 0 or 1 (not 1e8)
    assert entropy["slot"].abs().max() <= 1.0, f"Entropy out of range: {entropy['slot'].max()}"


def test_entropy_normalization_in_loss():
    """Verify entropy normalization doesn't blow up loss values."""
    net = FactoredRecurrentActorCritic(state_dim=35, num_slots=1)

    state = torch.randn(2, 3, 35)
    actions = {
        "slot": torch.zeros(2, 3, dtype=torch.long),
        "blueprint": torch.randint(0, 5, (2, 3)),
        "style": torch.randint(0, NUM_STYLES, (2, 3)),
        "tempo": torch.randint(0, NUM_TEMPO, (2, 3)),
        "alpha_target": torch.randint(0, NUM_ALPHA_TARGETS, (2, 3)),
        "alpha_speed": torch.randint(0, NUM_ALPHA_SPEEDS, (2, 3)),
        "alpha_curve": torch.randint(0, NUM_ALPHA_CURVES, (2, 3)),
        "op": torch.randint(0, NUM_OPS, (2, 3)),
    }

    log_probs, values, entropy, _ = net.evaluate_actions(state, actions)

    # Entropy loss should be bounded
    entropy_loss = sum(-ent.mean() for ent in entropy.values())
    assert entropy_loss.abs() < 100, f"Entropy loss too large: {entropy_loss}"


def test_get_action_raises_on_all_false_mask():
    """MaskedCategorical should raise when mask has no valid actions."""
    net = FactoredRecurrentActorCritic(state_dim=20)
    state = torch.randn(1, 20)
    invalid_mask = torch.zeros(1, 3, dtype=torch.bool)

    with pytest.raises(InvalidStateMachineError):
        net.get_action(state, slot_mask=invalid_mask)


def test_entropy_respects_valid_actions_only():
    """Entropy should be computed over valid actions and remain normalized."""
    net = FactoredRecurrentActorCritic(state_dim=20)
    state = torch.randn(1, 2, 20)
    slot_mask = torch.tensor([[[True, True, False]]])
    actions = {
        "slot": torch.zeros(1, 2, dtype=torch.long),
        "blueprint": torch.zeros(1, 2, dtype=torch.long),
        "style": torch.zeros(1, 2, dtype=torch.long),
        "tempo": torch.zeros(1, 2, dtype=torch.long),
        "alpha_target": torch.zeros(1, 2, dtype=torch.long),
        "alpha_speed": torch.zeros(1, 2, dtype=torch.long),
        "alpha_curve": torch.zeros(1, 2, dtype=torch.long),
        "op": torch.zeros(1, 2, dtype=torch.long),
    }

    _, _, entropy, _ = net.evaluate_actions(
        states=state,
        actions=actions,
        slot_mask=slot_mask,
    )

    slot_entropy = entropy["slot"]
    assert torch.isfinite(slot_entropy).all()
    assert (slot_entropy >= 0).all()
    assert (slot_entropy <= 1.01).all()
