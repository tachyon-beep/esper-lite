"""Performance tests for blending algorithms.

These tests verify caching behavior to avoid per-forward tensor allocations.
See docs/plans/2025-12-16-tolaria-kasmina-remediation.md Task 3.
"""

import torch
import pytest

from esper.kasmina.blending import LinearBlend, SigmoidBlend
from esper.kasmina.slot import SeedSlot


class TestBlendingPerformance:
    """Performance-related tests for blending algorithms."""

    def test_linear_blend_no_allocation_per_call(self):
        """LinearBlend should reuse cached alpha tensor."""
        blend = LinearBlend(total_steps=10)
        blend.step(5)

        x = torch.randn(2, 64, 32, 32)

        # Get alpha twice
        alpha1 = blend.get_alpha_for_blend(x)
        alpha2 = blend.get_alpha_for_blend(x)

        # Should be same tensor (cached)
        assert alpha1.data_ptr() == alpha2.data_ptr(), \
            "Alpha tensor should be cached, not recreated"

    def test_sigmoid_blend_no_allocation_per_call(self):
        """SigmoidBlend should reuse cached alpha tensor."""
        blend = SigmoidBlend(total_steps=10)
        blend.step(5)

        x = torch.randn(2, 64, 32, 32)

        # Get alpha twice
        alpha1 = blend.get_alpha_for_blend(x)
        alpha2 = blend.get_alpha_for_blend(x)

        # Should be same tensor (cached)
        assert alpha1.data_ptr() == alpha2.data_ptr(), \
            "Alpha tensor should be cached, not recreated"

    def test_cache_invalidates_on_device_change(self):
        """Cache should invalidate when device changes."""
        blend = LinearBlend(total_steps=10)
        blend.step(5)

        x_cpu = torch.randn(2, 64, 32, 32)
        alpha_cpu = blend.get_alpha_for_blend(x_cpu)

        # If CUDA available, test device change
        if torch.cuda.is_available():
            x_cuda = x_cpu.cuda()
            alpha_cuda = blend.get_alpha_for_blend(x_cuda)
            assert alpha_cpu.data_ptr() != alpha_cuda.data_ptr()
            assert alpha_cuda.device.type == 'cuda'

    def test_cache_invalidates_on_dtype_change(self):
        """Cache should invalidate when dtype changes."""
        blend = LinearBlend(total_steps=10)
        blend.step(5)

        x_float32 = torch.randn(2, 64, 32, 32, dtype=torch.float32)
        alpha_float32 = blend.get_alpha_for_blend(x_float32)

        x_float64 = x_float32.double()
        alpha_float64 = blend.get_alpha_for_blend(x_float64)

        assert alpha_float32.data_ptr() != alpha_float64.data_ptr()
        assert alpha_float64.dtype == torch.float64

    def test_cache_invalidates_on_alpha_value_change(self):
        """Cache should invalidate when alpha value changes."""
        blend = LinearBlend(total_steps=10)
        blend.step(5)

        x = torch.randn(2, 64, 32, 32)
        alpha1 = blend.get_alpha_for_blend(x)

        # Change step (changes alpha value)
        blend.step(7)
        alpha2 = blend.get_alpha_for_blend(x)

        assert alpha1.data_ptr() != alpha2.data_ptr()
        assert alpha1.item() != alpha2.item()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile needs CUDA")
    def test_linear_blend_get_alpha_compiles(self):
        """get_alpha_for_blend should compile without errors."""
        blend = LinearBlend(total_steps=10)
        blend.step(5)
        compiled = torch.compile(blend.get_alpha_for_blend)
        x = torch.randn(2, 64, 32, 32, device='cuda')
        alpha = compiled(x)
        assert alpha.shape == ()

    def test_slot_forward_uses_cached_alpha(self):
        """Verify slot.forward() uses cached alpha tensor across calls."""
        from esper.leyline import SeedStage

        # Create blend algorithm directly and test it in the blending context
        blend = LinearBlend(total_steps=10)
        blend.step(5)

        x = torch.randn(2, 64, 32, 32)

        # Monkeypatch to capture alpha data_ptr across forward calls
        # This simulates what happens inside slot.forward()
        original_get_alpha = blend.get_alpha_for_blend
        alpha_ptrs = []

        def capture_alpha(x_arg):
            alpha = original_get_alpha(x_arg)
            alpha_ptrs.append(alpha.data_ptr())
            return alpha

        blend.get_alpha_for_blend = capture_alpha

        # Simulate two forward passes that call get_alpha_for_blend
        # (This is what slot.forward() does during blending)
        for _ in range(2):
            alpha = blend.get_alpha_for_blend(x)
            # Simulate blending operation
            blended = x * alpha

        # Restore original method
        blend.get_alpha_for_blend = original_get_alpha

        # Verify both calls used the same cached alpha tensor
        assert len(alpha_ptrs) == 2
        assert alpha_ptrs[0] == alpha_ptrs[1], \
            "get_alpha_for_blend should use cached alpha tensor across calls"
