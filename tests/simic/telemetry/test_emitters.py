"""Tests for simic telemetry emitters."""

import math
from unittest.mock import MagicMock

import torch
from torch import nn

from esper.simic.telemetry.emitters import (
    compute_grad_norm_surrogate,
    emit_ppo_update_event,
)


def test_emit_ppo_update_event_propagates_group_id():
    """emit_ppo_update_event should propagate group_id to TelemetryEvent."""
    hub = MagicMock()

    emit_ppo_update_event(
        hub=hub,
        metrics={"policy_loss": 0.1, "value_loss": 0.2, "entropy": 1.5},
        episodes_completed=10,
        batch_idx=5,
        epoch=100,
        optimizer=None,
        grad_norm=1.0,
        update_time_ms=50.0,
        group_id="B",  # New parameter
    )

    # Verify hub.emit was called
    hub.emit.assert_called_once()
    event = hub.emit.call_args[0][0]

    # Verify group_id was propagated
    assert event.group_id == "B"


class TestComputeGradNormSurrogate:
    """Tests for compute_grad_norm_surrogate numerical stability."""

    def test_returns_none_for_no_gradients(self):
        """Should return None when module has no gradients."""
        model = nn.Linear(10, 5)
        # No backward pass, so no gradients
        result = compute_grad_norm_surrogate(model)
        assert result is None

    def test_computes_correct_norm_for_simple_case(self):
        """Should compute correct L2 norm for known gradients."""
        model = nn.Linear(2, 1, bias=False)
        # Manually set gradient to known values: [[3, 4]] -> norm = 5
        model.weight.grad = torch.tensor([[3.0, 4.0]])

        result = compute_grad_norm_surrogate(model)
        assert result is not None
        assert math.isclose(result, 5.0, rel_tol=1e-5)

    def test_no_overflow_with_large_gradients(self):
        """B7-PT-01: Should not overflow to inf with very large gradients.

        The old implementation used g*g which overflows for |g| > ~1.84e19.
        The new implementation using torch._foreach_norm handles this correctly.
        """
        model = nn.Linear(10, 5)

        # Set gradients to 1e20 - would cause overflow with naive g*g
        for param in model.parameters():
            param.grad = torch.full_like(param, 1e20)

        result = compute_grad_norm_surrogate(model)

        assert result is not None
        assert not math.isinf(result), "Gradient norm overflowed to inf"
        assert not math.isnan(result), "Gradient norm became NaN"
        assert result > 0, "Gradient norm should be positive"

    def test_handles_mixed_gradient_scales(self):
        """Should handle parameters with vastly different gradient scales."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2),
        )

        # First layer: tiny gradients
        model[0].weight.grad = torch.full_like(model[0].weight, 1e-10)
        model[0].bias.grad = torch.full_like(model[0].bias, 1e-10)

        # Second layer: huge gradients
        model[1].weight.grad = torch.full_like(model[1].weight, 1e15)
        model[1].bias.grad = torch.full_like(model[1].bias, 1e15)

        result = compute_grad_norm_surrogate(model)

        assert result is not None
        assert not math.isinf(result)
        assert not math.isnan(result)
        # Result should be dominated by the large gradients
        assert result > 1e14
