"""Tests for PolicyBundle protocol."""

import pytest
import torch

from esper.leyline import PolicyBundle, ActionResult, EvalResult


def test_policy_bundle_is_runtime_checkable():
    """PolicyBundle should be runtime_checkable for registration validation.

    We test the semantic behavior (isinstance works) rather than implementation
    details like __protocol_attrs__ which may vary across Python versions.
    """
    # Check the protocol flag (stable API)
    assert getattr(PolicyBundle, '_is_runtime_protocol', False), (
        "PolicyBundle must be decorated with @runtime_checkable"
    )

    # Verify isinstance actually works on a non-implementing class
    class NotAPolicy:
        pass

    assert not isinstance(NotAPolicy(), PolicyBundle)


def test_policy_bundle_protocol_methods():
    """PolicyBundle should define all required methods.

    Note: process_signals is NOT in this list - feature extraction is
    handled by Simic's signals_to_features() which requires training context.
    """
    assert callable(PolicyBundle.get_action)
    assert callable(PolicyBundle.forward)
    assert callable(PolicyBundle.evaluate_actions)
    assert callable(PolicyBundle.get_q_values)
    assert callable(PolicyBundle.sync_from)
    assert callable(PolicyBundle.get_value)
    assert callable(PolicyBundle.initial_hidden)
    assert callable(PolicyBundle.state_dict)
    assert callable(PolicyBundle.load_state_dict)
    assert callable(PolicyBundle.to)
    assert callable(PolicyBundle.enable_gradient_checkpointing)
    assert callable(PolicyBundle.compile)


def test_policy_bundle_protocol_properties():
    """PolicyBundle should define all required properties."""
    assert isinstance(PolicyBundle.is_recurrent, property)
    assert isinstance(PolicyBundle.supports_off_policy, property)
    assert isinstance(PolicyBundle.device, property)
    assert isinstance(PolicyBundle.dtype, property)
    assert isinstance(PolicyBundle.slot_config, property)
    assert isinstance(PolicyBundle.feature_dim, property)
    assert isinstance(PolicyBundle.hidden_dim, property)
    assert isinstance(PolicyBundle.network, property)
    assert isinstance(PolicyBundle.is_compiled, property)


def test_action_result_dataclass():
    """ActionResult should hold action selection results."""
    result = ActionResult(
        action={'slot': torch.tensor(0), 'blueprint': torch.tensor(1),
                'blend': torch.tensor(0), 'op': torch.tensor(2)},
        log_prob={'slot': torch.tensor(-0.5), 'blueprint': torch.tensor(-1.0),
                  'blend': torch.tensor(-0.3), 'op': torch.tensor(-0.8)},
        value=torch.tensor(0.5),
        hidden=(torch.zeros(1, 1, 256), torch.zeros(1, 1, 256)),
    )
    assert result.action['slot'].item() == 0
    assert result.value.item() == pytest.approx(0.5)


def test_eval_result_dataclass():
    """EvalResult should hold action evaluation results."""
    result = EvalResult(
        log_prob={'op': torch.tensor(-0.5)},
        value=torch.tensor(0.3),
        entropy={'op': torch.tensor(0.7)},
        hidden=None,
    )
    assert abs(result.entropy['op'].item() - 0.7) < 1e-6
