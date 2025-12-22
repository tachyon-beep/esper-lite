"""Regression contracts for isolate_gradients + channels_last during BLENDING.

Phase 3 risk reduction: when blend operators are extended, we must not introduce:
- accidental detach/no_grad patterns that break ghost gradients, or
- unexpected memory-format coercions that degrade channels_last performance.

This file locks the current contract using the simplest possible seed ("noop")
so the assertions reflect SeedSlot.forward mechanics, not seed behavior.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from esper.kasmina.blueprints import BlueprintRegistry
from esper.kasmina.slot import SeedSlot
from esper.leyline import SeedStage
from esper.leyline.alpha import AlphaAlgorithm


def _make_blending_noop_slot(*, isolate_gradients: bool) -> SeedSlot:
    slot = SeedSlot(slot_id="r0c0", channels=16, fast_mode=False)
    slot.germinate("noop", seed_id="seed-contract")
    slot.state.transition(SeedStage.TRAINING)
    slot.state.transition(SeedStage.BLENDING)
    slot.set_alpha(0.5)
    slot.isolate_gradients = isolate_gradients
    return slot


class TestIsolationChannelsLastContracts:
    def test_blending_preserves_channels_last_when_not_isolating(self) -> None:
        slot = _make_blending_noop_slot(isolate_gradients=False)

        x = torch.randn(2, 16, 8, 8).to(memory_format=torch.channels_last).detach().requires_grad_(True)
        assert x.is_contiguous(memory_format=torch.channels_last)

        out = slot.forward(x)

        # Regression: no isolate_gradients means no forced contiguous() conversion.
        assert out.is_contiguous(memory_format=torch.channels_last)
        assert out.dtype == x.dtype
        assert out.device == x.device

        out.sum().backward()
        assert x.grad is not None

        # noop seed => seed_features == host_features, so lerp gradient collapses to 1.0
        assert torch.allclose(x.grad, torch.ones_like(x.grad))

    def test_blending_preserves_channels_last_when_isolating(self) -> None:
        slot = _make_blending_noop_slot(isolate_gradients=True)

        x = torch.randn(2, 16, 8, 8).to(memory_format=torch.channels_last).detach().requires_grad_(True)
        assert x.is_contiguous(memory_format=torch.channels_last)

        out = slot.forward(x)

        # BUG-005 fix: preserve channels_last output under isolation by making
        # only the DETACHED seed input contiguous_format (keep host_features channels_last).
        assert not out.is_contiguous()
        assert out.is_contiguous(memory_format=torch.channels_last)
        assert out.dtype == x.dtype
        assert out.device == x.device

        out.sum().backward()
        assert x.grad is not None

        # With isolate_gradients=True, the seed input is detached, so gradients flow
        # only through the direct host path: d_out/d_host = (1 - alpha) == 0.5.
        assert torch.allclose(x.grad, torch.full_like(x.grad, 0.5))

    def test_isolate_gradients_multiply_uses_detached_seed_input_reference(self) -> None:
        """MULTIPLY must remain compatible with isolate_gradients=True (CNN contract)."""

        @BlueprintRegistry.register(
            name="__test_shift__",
            topology="cnn",
            param_estimate=0,
            description="test-only: returns x + constant shift",
        )
        def _shift_blueprint(dim: int) -> nn.Module:
            class ShiftSeed(nn.Module):
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x + 10.0

            return ShiftSeed()

        try:
            slot = SeedSlot(slot_id="r0c0", channels=16, fast_mode=False)
            slot.germinate("__test_shift__", seed_id="seed-mul", alpha_algorithm=AlphaAlgorithm.MULTIPLY)
            slot.state.stage = SeedStage.BLENDING
            slot.set_alpha(0.7)
            slot.isolate_gradients = True

            x = torch.full((2, 16, 8, 8), 100.0).detach().requires_grad_(True)
            out = slot.forward(x)
            out.sum().backward()

            expected_multiplier = 1.0 + 0.7 * torch.tanh(torch.tensor(10.0)).item()
            assert torch.allclose(x.grad, torch.full_like(x.grad, expected_multiplier), atol=1e-6)
        finally:
            BlueprintRegistry.unregister("cnn", "__test_shift__")

    def test_isolate_gradients_gate_uses_detached_seed_input_reference(self) -> None:
        """GATE must use the detached seed_input reference under isolation."""

        class MeanGate(nn.Module):
            def get_alpha_for_blend(self, x: torch.Tensor) -> torch.Tensor:
                # x: (B, C, H, W) -> gate: (B, 1, 1, 1)
                return torch.sigmoid(x.mean(dim=(1, 2, 3), keepdim=True))

        slot = _make_blending_noop_slot(isolate_gradients=True)
        slot.state.alpha_algorithm = AlphaAlgorithm.GATE
        slot.alpha_schedule = MeanGate()
        slot.set_alpha(0.5)

        x = torch.randn(2, 16, 8, 8).detach().requires_grad_(True)
        out = slot.forward(x)
        out.sum().backward()

        seed_input = x.detach()
        gate = torch.sigmoid(seed_input.mean(dim=(1, 2, 3), keepdim=True))
        a_eff = (0.5 * gate).clamp(0.0, 1.0)
        expected_grad = (1.0 - a_eff).expand_as(x)

        assert x.grad is not None
        assert torch.allclose(x.grad, expected_grad, atol=1e-6)
