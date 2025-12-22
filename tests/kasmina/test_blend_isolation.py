"""Tests for blend_with_isolation after isinstance elimination.

Task 5 from docs/plans/2025-12-16-tolaria-kasmina-remediation.md:
Verify blend_with_isolation now requires tensor alpha and SeedSlot correctly
converts scalar to tensor in fallback path.
"""

import pytest
import torch

from esper.kasmina.isolation import blend_with_isolation
from esper.kasmina.slot import SeedSlot
from esper.leyline import SeedStage
from esper.leyline.alpha import AlphaAlgorithm


class TestBlendWithIsolation:
    """Test blend_with_isolation with tensor alpha requirement."""

    def test_blend_with_tensor_alpha(self):
        """Verify blend_with_isolation works with tensor alpha."""
        host = torch.randn(2, 16, 8, 8)
        seed = torch.randn(2, 16, 8, 8)
        alpha = torch.tensor(0.5)

        result = blend_with_isolation(host, seed, alpha)

        # Expected: lerp(host, seed, 0.5) = host + 0.5 * (seed - host)
        expected = torch.lerp(host, seed, 0.5)
        assert torch.allclose(result, expected)

    def test_blend_clamping_lower(self):
        """Verify alpha is clamped to [0, 1] - lower bound."""
        host = torch.randn(2, 16, 8, 8)
        seed = torch.randn(2, 16, 8, 8)
        alpha = torch.tensor(-0.5)

        result = blend_with_isolation(host, seed, alpha)

        # Alpha should be clamped to 0.0
        expected = torch.lerp(host, seed, 0.0)
        assert torch.allclose(result, expected)

    def test_blend_clamping_upper(self):
        """Verify alpha is clamped to [0, 1] - upper bound."""
        host = torch.randn(2, 16, 8, 8)
        seed = torch.randn(2, 16, 8, 8)
        alpha = torch.tensor(1.5)

        result = blend_with_isolation(host, seed, alpha)

        # Alpha should be clamped to 1.0
        expected = torch.lerp(host, seed, 1.0)
        assert torch.allclose(result, expected)

    def test_blend_device_dtype_matching(self):
        """Verify blend works when alpha matches input device/dtype."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        host = torch.randn(2, 16, 8, 8, device=device, dtype=dtype)
        seed = torch.randn(2, 16, 8, 8, device=device, dtype=dtype)
        alpha = torch.tensor(0.7, device=device, dtype=dtype)

        result = blend_with_isolation(host, seed, alpha)

        assert result.device.type == device.type
        assert result.dtype == dtype
        expected = torch.lerp(host, seed, 0.7)
        # fp16 needs higher tolerance due to reduced precision and clamp artifacts
        if dtype == torch.float16:
            assert torch.allclose(result, expected, rtol=1e-2, atol=1e-3)
        else:
            assert torch.allclose(result, expected, rtol=1e-5)

    def test_blend_gradient_flow(self):
        """Verify gradients flow to both host and seed."""
        host = torch.randn(2, 16, 8, 8, requires_grad=True)
        seed = torch.randn(2, 16, 8, 8, requires_grad=True)
        alpha = torch.tensor(0.5)

        result = blend_with_isolation(host, seed, alpha)
        loss = result.sum()
        loss.backward()

        # Both host and seed should receive gradients
        assert host.grad is not None
        assert seed.grad is not None
        assert host.grad.abs().sum() > 0
        assert seed.grad.abs().sum() > 0


class TestSeedSlotFallbackPath:
    """Test SeedSlot.forward() correctly converts scalar to tensor."""

    def test_seedslot_fallback_scalar_to_tensor(self):
        """Verify SeedSlot converts scalar alpha to tensor in fallback path."""
        slot = SeedSlot(slot_id="test", channels=16)

        # Germinate a seed using 'noop' blueprint (identity-like)
        slot.state = slot.germinate("noop", seed_id="test-seed")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        # Set scalar alpha (no alpha_schedule)
        slot.state.alpha = 0.5
        slot.alpha_schedule = None

        # Forward pass
        host = torch.randn(2, 16, 8, 8)
        result = slot.forward(host)

        # Should not raise, and should blend correctly
        assert result.shape == host.shape

    def test_seedslot_device_dtype_consistency(self):
        """Verify scalar-to-tensor conversion matches host features device/dtype."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        slot = SeedSlot(slot_id="test", channels=16, device=device)

        # Germinate and transition to BLENDING using 'noop' blueprint
        slot.state = slot.germinate("noop", seed_id="test-seed")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        # Set scalar alpha (no alpha_schedule)
        slot.state.alpha = 0.5
        slot.alpha_schedule = None

        # Forward with different dtype
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        host = torch.randn(2, 16, 8, 8, device=device, dtype=dtype)
        result = slot.forward(host)

        # Result should match host dtype/device
        assert result.device.type == device.type
        assert result.dtype == dtype

    def test_seedslot_with_alpha_schedule(self):
        """Verify SeedSlot with alpha_schedule (GATE) still runs.

        Phase 3 contract: alpha_schedule is reserved for per-sample gating only.
        """
        from esper.kasmina.blending import GatedBlend

        slot = SeedSlot(slot_id="test", channels=16)

        # Germinate with gated blending so alpha_schedule is valid.
        slot.state = slot.germinate(
            "noop",
            seed_id="test-seed",
            blend_algorithm_id="gated",
            alpha_algorithm=AlphaAlgorithm.GATE,
        )
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        slot.start_blending(total_steps=10)
        assert isinstance(slot.alpha_schedule, GatedBlend)
        slot.set_alpha(0.5)

        # Forward pass
        host = torch.randn(2, 16, 8, 8)
        result = slot.forward(host)

        # Should not raise
        assert result.shape == host.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
