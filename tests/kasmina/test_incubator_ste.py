"""Tests for Incubator (TRAINING) Straight-Through Estimator behavior in SeedSlot.

These tests validate the "magic residual" pattern used in SeedSlot.forward:

1. Forward isolation: output == host_features in TRAINING with alpha == 0.0
2. Backward learning: seed parameters receive non-zero gradients
3. Backward isolation: with input detachment enabled, host gradients are
   unaffected by the seed path
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from esper.kasmina.slot import SeedSlot, SeedState, SeedStage


class SimpleSeed(nn.Module):
    """Minimal seed module for STE tests."""

    def __init__(self, channels: int):
        super().__init__()
        self.layer = nn.Linear(channels, channels, bias=False)
        # Initialize as identity for predictable gradients
        nn.init.eye_(self.layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


@pytest.fixture
def slot() -> SeedSlot:
    """Create a SeedSlot configured for Incubator (TRAINING) mode."""
    slot = SeedSlot(slot_id="test_slot", channels=4, device="cpu")
    slot.seed = SimpleSeed(channels=4)
    slot.state = SeedState(
        seed_id="test_seed",
        blueprint_id="test_bp",
        stage=SeedStage.TRAINING,
    )
    slot.set_alpha(0.0)  # Incubator: alpha == 0.0
    slot.isolate_gradients = True  # Detach host input into seed
    return slot


tensor_strategy = arrays(
    dtype=np.float32,
    shape=(2, 4),
    elements=st.floats(min_value=-10.0, max_value=10.0, width=32),
).map(torch.from_numpy)


@given(host_input=tensor_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_incubator_ste_behavior(slot: SeedSlot, host_input: torch.Tensor) -> None:
    """Incubator mode should isolate forward and host gradients for all inputs."""
    # Filter out the degenerate all-zero case where loss == 0 and all gradients are zero.
    assume(host_input.abs().sum().item() > 0.0)

    # Host input needs grad to inspect host gradient path
    host_input = host_input.clone().requires_grad_(True)

    # Forward through SeedSlot in TRAINING + alpha == 0.0
    output = slot(host_input)

    # 1) Forward isolation: output numerically equals host input
    assert torch.allclose(output, host_input), "Incubator broke forward isolation"

    # Backward from a simple scalar loss
    loss = output.sum()
    loss.backward()

    # 2) Backward isolation: host gradient is exactly d/dx sum(x) = 1
    expected_grad = torch.ones_like(host_input)
    assert torch.allclose(
        host_input.grad, expected_grad
    ), f"Incubator broke backward isolation; host grad polluted: {host_input.grad}"


def test_incubator_ste_seed_receives_gradient(slot: SeedSlot) -> None:
    """Seed should receive gradients in Incubator mode for non-degenerate inputs."""
    host_input = torch.tensor(
        [[1.0, 2.0, -1.0, 0.5], [0.0, -0.5, 3.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    output = slot(host_input)
    loss = output.sum()
    loss.backward()

    seed_grad_norm = slot.seed.layer.weight.grad.norm().item()
    assert seed_grad_norm > 0.0, "Seed did not receive gradients in Incubator mode"


def test_incubator_ste_inactive_when_blending(slot: SeedSlot) -> None:
    """STE should turn off once we enter BLENDING (alpha > 0)."""
    host_input = torch.randn(2, 4, requires_grad=True)

    # Modify seed weights so BLENDING changes activations
    with torch.no_grad():
        slot.seed.layer.weight.copy_(2.0 * torch.eye(4))

    # Transition to BLENDING and enable blending
    slot.state.transition(SeedStage.BLENDING)
    slot.set_alpha(0.5)

    output = slot(host_input)

    # In BLENDING, output should differ from host_input for this seed
    assert not torch.allclose(
        output, host_input
    ), "STE remained active during blending; output still identical to host"
