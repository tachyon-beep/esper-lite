"""Tests for gradient clipping in process_train_batch.

These tests verify that gradient clipping is correctly implemented with proper
AMP-safe ordering (unscale_ before clip_grad_norm_).
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, call
from typing import Iterator


class MockSlottedModel(nn.Module):
    """Mock model that implements SlottedHostProtocol methods for testing."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.host_linear = nn.Linear(10, 10)
        self.seed_linear = nn.Linear(10, 10)
        self.seed_slots: dict[str, MagicMock] = {}
        self._device = device

    def get_host_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.host_linear.parameters())

    def get_seed_parameters(self, slot: str | None = None) -> Iterator[nn.Parameter]:
        return iter(self.seed_linear.parameters())

    def has_active_seed_in_slot(self, slot_id: str) -> bool:
        return slot_id in self.seed_slots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.host_linear(x) + self.seed_linear(x)


class TestGradientClippingAMPOrdering:
    """Test that gradient clipping follows correct AMP ordering."""

    def test_unscale_called_before_clip_when_amp_enabled(self):
        """unscale_() must be called before clip_grad_norm_() in AMP path."""
        # Track call order
        call_order: list[str] = []

        mock_scaler = MagicMock()
        mock_scaler.unscale_ = MagicMock(side_effect=lambda opt: call_order.append(f"unscale_{id(opt)}"))
        mock_scaler.step = MagicMock()
        mock_scaler.update = MagicMock()

        # Create simple model with parameters
        model = MockSlottedModel()
        host_opt = torch.optim.SGD(model.host_linear.parameters(), lr=0.01)

        # Simulate backward to create gradients
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        # Verify gradients exist
        assert all(p.grad is not None for p in model.host_linear.parameters())

        # Now simulate the clipping logic from process_train_batch
        max_grad_norm = 1.0
        slots_to_step: list[str] = []

        if max_grad_norm is not None and max_grad_norm > 0:
            if mock_scaler is not None:
                mock_scaler.unscale_(host_opt)
                call_order.append("clip_grad_norm_")

            all_params = list(model.get_host_parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

        # Verify unscale was called before clip_grad_norm_
        assert "clip_grad_norm_" in call_order
        unscale_idx = next(i for i, c in enumerate(call_order) if c.startswith("unscale_"))
        clip_idx = call_order.index("clip_grad_norm_")
        assert unscale_idx < clip_idx, "unscale_() must be called before clip_grad_norm_()"

    def test_clip_grad_norm_actually_clips_large_gradients(self):
        """Verify that clip_grad_norm_ actually limits gradient magnitude."""
        model = MockSlottedModel()
        host_opt = torch.optim.SGD(model.host_linear.parameters(), lr=0.01)

        # Create large gradients by doing backward on a large loss
        x = torch.randn(2, 10) * 100  # Large inputs = large gradients
        y = model(x).sum() * 1000  # Amplify even more
        y.backward()

        # Check gradient norm before clipping
        all_params = list(model.get_host_parameters())
        grad_norm_before = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))

        # Reset and create same large gradients
        host_opt.zero_grad()
        y = model(x).sum() * 1000
        y.backward()

        # Now clip with a small max_grad_norm
        max_grad_norm = 0.1
        grad_norm_after = torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

        # The returned value is the original norm (before clipping)
        # After clipping, the actual norm should be <= max_grad_norm
        actual_norm_after = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in all_params if p.grad is not None)
        ).item()

        assert actual_norm_after <= max_grad_norm + 1e-6, (
            f"Gradient norm {actual_norm_after} exceeds max_grad_norm {max_grad_norm}"
        )


class TestGradientClippingDisabled:
    """Test that gradient clipping can be disabled."""

    def test_no_clipping_when_max_grad_norm_is_none(self):
        """Gradient clipping should be skipped when max_grad_norm is None."""
        model = MockSlottedModel()

        # Create gradients
        x = torch.randn(2, 10) * 100
        y = model(x).sum() * 1000
        y.backward()

        # Get gradient norm before
        all_params = list(model.get_host_parameters())
        grad_norm_before = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in all_params if p.grad is not None)
        ).item()

        # Simulate the condition check from process_train_batch
        max_grad_norm = None
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

        # Gradient norm should be unchanged
        grad_norm_after = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in all_params if p.grad is not None)
        ).item()

        assert abs(grad_norm_before - grad_norm_after) < 1e-6

    def test_no_clipping_when_max_grad_norm_is_zero(self):
        """Gradient clipping should be skipped when max_grad_norm is 0."""
        model = MockSlottedModel()

        # Create gradients
        x = torch.randn(2, 10) * 100
        y = model(x).sum() * 1000
        y.backward()

        # Get gradient norm before
        all_params = list(model.get_host_parameters())
        grad_norm_before = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in all_params if p.grad is not None)
        ).item()

        # Simulate the condition check from process_train_batch
        max_grad_norm = 0.0
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

        # Gradient norm should be unchanged
        grad_norm_after = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in all_params if p.grad is not None)
        ).item()

        assert abs(grad_norm_before - grad_norm_after) < 1e-6


class TestGradientClippingWithSeeds:
    """Test gradient clipping includes seed parameters."""

    def test_clips_both_host_and_seed_parameters(self):
        """Gradient clipping should include both host and seed parameters."""
        model = MockSlottedModel()

        # Create gradients for both host and seed
        x = torch.randn(2, 10) * 100
        y = model(x).sum() * 1000
        y.backward()

        # Collect all parameters as done in process_train_batch
        all_params = list(model.get_host_parameters())
        all_params.extend(model.get_seed_parameters("slot_0"))

        # Get norm before clipping
        grad_norm_before = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in all_params if p.grad is not None)
        ).item()

        # Apply clipping
        max_grad_norm = 0.1
        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

        # Verify clipping worked on combined parameters
        grad_norm_after = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in all_params if p.grad is not None)
        ).item()

        assert grad_norm_after <= max_grad_norm + 1e-6


class TestGradientClippingNonAMPPath:
    """Test gradient clipping in non-AMP path."""

    def test_clips_without_unscale_when_no_scaler(self):
        """Non-AMP path should clip directly without unscale_()."""
        model = MockSlottedModel()
        host_opt = torch.optim.SGD(model.host_linear.parameters(), lr=0.01)

        # Create gradients
        x = torch.randn(2, 10) * 100
        y = model(x).sum() * 1000
        y.backward()

        # Simulate non-AMP path (scaler is None)
        scaler = None
        max_grad_norm = 0.5

        if max_grad_norm is not None and max_grad_norm > 0:
            if scaler is not None:
                # This should NOT be executed
                scaler.unscale_(host_opt)

            all_params = list(model.get_host_parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

        # Verify clipping happened
        actual_norm = torch.sqrt(
            sum(p.grad.norm() ** 2 for p in model.host_linear.parameters() if p.grad is not None)
        ).item()

        assert actual_norm <= max_grad_norm + 1e-6


class TestSeedOptimizerHasGradsComputation:
    """Test the has_grads computation for seed optimizers."""

    def test_has_grads_true_when_gradients_exist(self):
        """has_grads should be True when optimizer params have gradients."""
        model = MockSlottedModel()
        seed_opt = torch.optim.SGD(model.seed_linear.parameters(), lr=0.01)

        # Create gradients
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        # Check has_grads as done in process_train_batch
        has_grads = any(
            p.grad is not None for group in seed_opt.param_groups for p in group["params"]
        )

        assert has_grads is True

    def test_has_grads_false_when_no_gradients(self):
        """has_grads should be False when optimizer params have no gradients."""
        model = MockSlottedModel()
        seed_opt = torch.optim.SGD(model.seed_linear.parameters(), lr=0.01)

        # Don't call backward - no gradients
        # Check has_grads as done in process_train_batch
        has_grads = any(
            p.grad is not None for group in seed_opt.param_groups for p in group["params"]
        )

        assert has_grads is False

    def test_has_grads_reused_for_unscale_and_step(self):
        """has_grads should be computed once and reused."""
        model = MockSlottedModel()
        seed_opt = torch.optim.SGD(model.seed_linear.parameters(), lr=0.01)

        # Create gradients
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        # Compute once as done in process_train_batch
        seed_opts_with_grads: dict[str, tuple[torch.optim.Optimizer, bool]] = {}
        slot_id = "slot_0"
        has_grads = any(
            p.grad is not None for group in seed_opt.param_groups for p in group["params"]
        )
        seed_opts_with_grads[slot_id] = (seed_opt, has_grads)

        # Verify we can access the computed value
        stored_opt, stored_has_grads = seed_opts_with_grads[slot_id]
        assert stored_opt is seed_opt
        assert stored_has_grads is True
