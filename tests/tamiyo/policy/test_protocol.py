"""Tests for PolicyBundle protocol."""

import torch
from typing import runtime_checkable

from esper.tamiyo.policy.protocol import PolicyBundle
from esper.tamiyo.policy.types import ActionResult, EvalResult, ForwardResult


def test_policy_bundle_is_runtime_checkable():
    """PolicyBundle should be runtime_checkable for registration validation."""
    assert hasattr(PolicyBundle, '__protocol_attrs__')


def test_policy_bundle_protocol_methods():
    """PolicyBundle should define all required methods."""
    required_methods = [
        'process_signals',
        'get_action',
        'forward',
        'evaluate_actions',
        'get_q_values',
        'sync_from',
        'get_value',
        'initial_hidden',
        'state_dict',
        'load_state_dict',
        'to',
        'enable_gradient_checkpointing',
    ]
    for method in required_methods:
        assert hasattr(PolicyBundle, method), f"Missing method: {method}"


def test_policy_bundle_protocol_properties():
    """PolicyBundle should define all required properties."""
    required_properties = [
        'is_recurrent',
        'supports_off_policy',
        'device',
        'dtype',
    ]
    for prop in required_properties:
        assert hasattr(PolicyBundle, prop), f"Missing property: {prop}"


def test_action_result_dataclass():
    """ActionResult should hold action selection results."""
    result = ActionResult(
        action={'slot': 0, 'blueprint': 1, 'blend': 0, 'op': 2},
        log_prob={'slot': torch.tensor(-0.5), 'blueprint': torch.tensor(-1.0),
                  'blend': torch.tensor(-0.3), 'op': torch.tensor(-0.8)},
        value=torch.tensor(0.5),
        hidden=(torch.zeros(1, 1, 256), torch.zeros(1, 1, 256)),
    )
    assert result.action['slot'] == 0
    assert result.value.item() == 0.5


def test_eval_result_dataclass():
    """EvalResult should hold action evaluation results."""
    result = EvalResult(
        log_prob={'op': torch.tensor(-0.5)},
        value=torch.tensor(0.3),
        entropy={'op': torch.tensor(0.7)},
        hidden=None,
    )
    assert abs(result.entropy['op'].item() - 0.7) < 1e-6
