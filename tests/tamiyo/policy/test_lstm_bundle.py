"""Tests for LSTMPolicyBundle."""

import pytest
import torch

from esper.tamiyo.policy import get_policy, list_policies
from esper.tamiyo.policy.lstm_bundle import LSTMPolicyBundle
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult
from esper.leyline.slot_config import SlotConfig
from esper.leyline.factored_actions import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)


@pytest.fixture
def slot_config():
    return SlotConfig.default()


@pytest.fixture
def lstm_bundle(slot_config):
    from esper.tamiyo.policy.features import get_feature_size

    return LSTMPolicyBundle(
        feature_dim=get_feature_size(slot_config),
        hidden_dim=64,
        num_lstm_layers=1,
        slot_config=slot_config,
    )


def test_lstm_bundle_registered():
    """LSTMPolicyBundle should be registered as 'lstm'."""
    assert "lstm" in list_policies()


def test_lstm_bundle_is_recurrent(lstm_bundle):
    """LSTMPolicyBundle should be recurrent."""
    assert lstm_bundle.is_recurrent is True


def test_lstm_bundle_does_not_support_off_policy(lstm_bundle):
    """LSTMPolicyBundle should not support off-policy (for now)."""
    assert lstm_bundle.supports_off_policy is False


def test_lstm_bundle_initial_hidden(lstm_bundle):
    """initial_hidden should return LSTM hidden state tuple."""
    hidden = lstm_bundle.initial_hidden(batch_size=4)
    assert hidden is not None
    h, c = hidden
    assert h.shape == (1, 4, 64)  # (num_layers, batch, hidden_dim)
    assert c.shape == (1, 4, 64)


def test_lstm_bundle_get_action(lstm_bundle, slot_config):
    """get_action should return ActionResult."""
    features = torch.randn(1, lstm_bundle.feature_dim)
    masks = {
        "slot": torch.ones(1, slot_config.num_slots, dtype=torch.bool),
        "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool),
        "style": torch.ones(1, NUM_STYLES, dtype=torch.bool),
        "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool),
        "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool),
        "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool),
        "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool),
        "op": torch.ones(1, NUM_OPS, dtype=torch.bool),
    }
    hidden = lstm_bundle.initial_hidden(batch_size=1)

    result = lstm_bundle.get_action(features, masks, hidden)

    assert isinstance(result, ActionResult)
    assert "op" in result.action
    assert "op" in result.log_prob
    assert result.hidden is not None


def test_lstm_bundle_evaluate_actions(lstm_bundle, slot_config):
    """evaluate_actions should return EvalResult with gradients."""
    features = torch.randn(1, 10, lstm_bundle.feature_dim)
    masks = {
        "slot": torch.ones(1, 10, slot_config.num_slots, dtype=torch.bool),
        "blueprint": torch.ones(1, 10, NUM_BLUEPRINTS, dtype=torch.bool),
        "style": torch.ones(1, 10, NUM_STYLES, dtype=torch.bool),
        "tempo": torch.ones(1, 10, NUM_TEMPO, dtype=torch.bool),
        "alpha_target": torch.ones(1, 10, NUM_ALPHA_TARGETS, dtype=torch.bool),
        "alpha_speed": torch.ones(1, 10, NUM_ALPHA_SPEEDS, dtype=torch.bool),
        "alpha_curve": torch.ones(1, 10, NUM_ALPHA_CURVES, dtype=torch.bool),
        "op": torch.ones(1, 10, NUM_OPS, dtype=torch.bool),
    }
    actions = {
        "slot": torch.zeros(1, 10, dtype=torch.long),
        "blueprint": torch.zeros(1, 10, dtype=torch.long),
        "style": torch.zeros(1, 10, dtype=torch.long),
        "tempo": torch.zeros(1, 10, dtype=torch.long),
        "alpha_target": torch.zeros(1, 10, dtype=torch.long),
        "alpha_speed": torch.zeros(1, 10, dtype=torch.long),
        "alpha_curve": torch.zeros(1, 10, dtype=torch.long),
        "op": torch.zeros(1, 10, dtype=torch.long),
    }
    # Don't use initial_hidden() here as it creates inference-mode tensors
    # evaluate_actions needs to support gradients, so pass None (network will create proper hidden state)
    hidden = None

    result = lstm_bundle.evaluate_actions(features, actions, masks, hidden)

    assert isinstance(result, EvalResult)
    assert result.log_prob["op"].requires_grad
    assert result.value.requires_grad


def test_lstm_bundle_state_dict(lstm_bundle):
    """state_dict should return serializable dict."""
    state = lstm_bundle.state_dict()
    assert isinstance(state, dict)
    # Should have network weights
    assert any("weight" in k or "bias" in k for k in state.keys())


def test_lstm_bundle_load_state_dict(lstm_bundle):
    """load_state_dict should restore policy state."""
    state = lstm_bundle.state_dict()
    lstm_bundle.load_state_dict(state)
    # Should not raise


def test_lstm_bundle_device_management(lstm_bundle):
    """to() should move policy to device."""
    assert lstm_bundle.device == torch.device("cpu")
    # Note: Can't test CUDA without GPU, but method should exist
    lstm_bundle.to("cpu")
    assert lstm_bundle.device == torch.device("cpu")


def test_lstm_bundle_dtype(lstm_bundle):
    """dtype should return network parameter dtype."""
    assert lstm_bundle.dtype == torch.float32


def test_get_policy_lstm(slot_config):
    """get_policy('lstm', ...) should return LSTMPolicyBundle."""
    from esper.tamiyo.policy.features import get_feature_size

    policy = get_policy("lstm", {
        "feature_dim": get_feature_size(slot_config),
        "hidden_dim": 64,
        "slot_config": slot_config,
    })
    assert isinstance(policy, LSTMPolicyBundle)


def test_lstm_bundle_forward(lstm_bundle, slot_config):
    """forward() should return ForwardResult with logits."""
    features = torch.randn(1, 1, lstm_bundle.feature_dim)
    masks = {
        "slot": torch.ones(1, slot_config.num_slots, dtype=torch.bool),
        "blueprint": torch.ones(1, NUM_BLUEPRINTS, dtype=torch.bool),
        "style": torch.ones(1, NUM_STYLES, dtype=torch.bool),
        "tempo": torch.ones(1, NUM_TEMPO, dtype=torch.bool),
        "alpha_target": torch.ones(1, NUM_ALPHA_TARGETS, dtype=torch.bool),
        "alpha_speed": torch.ones(1, NUM_ALPHA_SPEEDS, dtype=torch.bool),
        "alpha_curve": torch.ones(1, NUM_ALPHA_CURVES, dtype=torch.bool),
        "op": torch.ones(1, NUM_OPS, dtype=torch.bool),
    }
    # Pass None for hidden - network creates its own initial state
    # (initial_hidden() creates inference-mode tensors that can't be used with autograd)
    hidden = None

    result = lstm_bundle.forward(features, masks, hidden)

    assert isinstance(result, ForwardResult)
    assert "op" in result.logits
    assert "slot" in result.logits
    assert result.value is not None
    assert result.hidden is not None


def test_lstm_bundle_get_value(lstm_bundle):
    """get_value() should return state value estimate."""
    features = torch.randn(1, lstm_bundle.feature_dim)
    # Pass None for hidden - network creates its own initial state
    hidden = None

    value = lstm_bundle.get_value(features, hidden)

    assert isinstance(value, torch.Tensor)
    # Value should be scalar or batch dimension
    assert value.numel() == 1 or (value.dim() == 1 and value.shape[0] == 1)


def test_get_value_does_not_create_grad_graph(lstm_bundle):
    """get_value() should not create gradient computation graph."""
    # Ensure we're in a context where gradients would normally be tracked
    features = torch.randn(1, lstm_bundle.feature_dim).requires_grad_(True)

    value = lstm_bundle.get_value(features)

    # Value should not require grad (computed in inference_mode)
    assert not value.requires_grad, (
        "get_value() created gradient graph. Add @torch.inference_mode() decorator."
    )
