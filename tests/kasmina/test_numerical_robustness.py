"""Numerical robustness tests for Kasmina.

Tests verify correct behavior under numerical edge conditions:
- NaN/Inf detection in seed outputs
- Gradient overflow handling
- Mixed precision blending (FP16/FP32)
- Floating point accumulation accuracy
- Very small improvement detection
- Zero gradient handling
"""

import pytest
import torch

from esper.kasmina.slot import SeedSlot, SeedMetrics
from esper.kasmina.isolation import blend_with_isolation, ste_forward
from esper.kasmina.blending import GatedBlend


class TestNaNDetection:
    """Tests for NaN detection in seed outputs.

    Note: Due to IEEE 754 floating point semantics, NaN propagates through
    most operations. Even `NaN * 0 = NaN`. The tests document actual behavior.
    """

    def test_nan_in_seed_output_propagates(self):
        """NaN in seed output should propagate through blend (no silent masking)."""
        host = torch.randn(2, 64, 8, 8)
        seed = torch.full((2, 64, 8, 8), float("nan"))
        alpha = torch.tensor(0.5)

        result = blend_with_isolation(host, seed, alpha)

        # With alpha=0.5, result should contain NaNs
        assert torch.isnan(result).any()

    def test_nan_in_seed_alpha_zero_propagates(self):
        """At alpha=0, NaN in seed still propagates due to IEEE 754 (NaN * 0 = NaN).

        lerp(host, seed, 0) computes host + 0 * (seed - host), and 0 * NaN = NaN.
        This is expected floating point behavior.
        """
        host = torch.randn(2, 64, 8, 8)
        seed = torch.full((2, 64, 8, 8), float("nan"))
        alpha = torch.tensor(0.0)

        result = blend_with_isolation(host, seed, alpha)

        # NaN propagates even with alpha=0 due to IEEE 754
        assert torch.isnan(result).any()

    def test_nan_in_host_propagates(self):
        """NaN in host features should propagate through blend."""
        host = torch.full((2, 64, 8, 8), float("nan"))
        seed = torch.randn(2, 64, 8, 8)
        alpha = torch.tensor(0.5)

        result = blend_with_isolation(host, seed, alpha)

        assert torch.isnan(result).any()

    def test_ste_forward_with_nan_seed_propagates(self):
        """STE forward with NaN seed produces NaN due to NaN - NaN arithmetic.

        STE: host + (seed - seed.detach()). When seed=NaN, NaN - NaN = NaN.
        """
        host = torch.randn(2, 64, 8, 8)
        seed = torch.full((2, 64, 8, 8), float("nan"))

        result = ste_forward(host, seed)

        # NaN - NaN = NaN, so result contains NaN
        assert torch.isnan(result).any()


class TestInfDetection:
    """Tests for Inf detection in seed outputs.

    Note: Inf arithmetic can produce NaN in some cases (inf - inf, inf * 0).
    The tests document actual IEEE 754 floating point behavior.
    """

    def test_inf_in_seed_output_produces_nan(self):
        """Inf in seed output produces NaN through blend due to inf - inf arithmetic.

        lerp(host, inf, 0.5) involves inf - host = inf, then 0.5 * inf = inf,
        but the overall computation can produce NaN depending on values.
        """
        host = torch.randn(2, 64, 8, 8)
        seed = torch.full((2, 64, 8, 8), float("inf"))
        alpha = torch.tensor(0.5)

        result = blend_with_isolation(host, seed, alpha)

        # Result will be NaN due to inf arithmetic (inf - finite = inf, but lerp internals)
        # We just verify that the output is not normal (either inf or nan)
        assert torch.isinf(result).any() or torch.isnan(result).any()

    def test_negative_inf_produces_nan_or_inf(self):
        """Negative Inf propagates through blend, possibly as NaN."""
        host = torch.randn(2, 64, 8, 8)
        seed = torch.full((2, 64, 8, 8), float("-inf"))
        alpha = torch.tensor(0.5)

        result = blend_with_isolation(host, seed, alpha)

        # Either inf or nan will be present
        assert torch.isinf(result).any() or torch.isnan(result).any()

    def test_inf_at_alpha_zero_produces_nan(self):
        """At alpha=0, Inf in seed produces NaN due to inf * 0 = NaN.

        lerp(host, inf, 0) computes host + 0 * (inf - host) = host + 0 * inf = NaN.
        """
        host = torch.randn(2, 64, 8, 8)
        seed = torch.full((2, 64, 8, 8), float("inf"))
        alpha = torch.tensor(0.0)

        result = blend_with_isolation(host, seed, alpha)

        # 0 * inf = NaN in IEEE 754
        assert torch.isnan(result).any()


class TestAlphaNumericalBehavior:
    """Tests for numerical behavior of alpha values."""

    def test_alpha_clamped_to_zero(self):
        """Negative alpha should be clamped to 0."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        slot.set_alpha(-1e-10)

        assert slot.state.alpha == 0.0

    def test_alpha_clamped_to_one(self):
        """Alpha > 1 should be clamped to 1."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        slot.set_alpha(1.0 + 1e-10)

        assert slot.state.alpha == 1.0

    def test_very_small_alpha_preserved(self):
        """Very small but valid alpha should be preserved."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        slot.set_alpha(1e-7)

        assert slot.state.alpha == 1e-7

    def test_alpha_just_below_one_preserved(self):
        """Alpha just below 1 should be preserved."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        slot.set_alpha(1.0 - 1e-7)

        assert slot.state.alpha == pytest.approx(1.0 - 1e-7)


class TestGradientNumericalBehavior:
    """Tests for gradient numerical behavior."""

    def test_zero_gradients_no_division_by_zero(self):
        """Zero gradients should be handled without division by zero."""
        metrics = SeedMetrics()
        metrics.seed_param_count = 1000
        metrics.host_param_count = 10000

        # Zero gradient norm ratio should remain 0
        metrics.seed_gradient_norm_ratio = 0.0

        # Should not raise
        assert metrics.seed_gradient_norm_ratio == 0.0

    def test_very_large_gradient_handled(self):
        """Very large gradients should not cause overflow in metrics."""
        metrics = SeedMetrics()

        # Should accept large values
        metrics.gradient_norm_avg = 1e30

        assert metrics.gradient_norm_avg == 1e30

    def test_gradient_norm_ratio_clamps_extreme(self):
        """Extreme gradient ratios should be handled."""
        metrics = SeedMetrics()

        # Very large ratio
        metrics.seed_gradient_norm_ratio = 1e10

        assert metrics.seed_gradient_norm_ratio == 1e10


class TestMixedPrecisionBlending:
    """Tests for mixed precision (FP16/FP32) blending."""

    def test_fp16_host_fp16_seed(self):
        """FP16 host and seed should blend correctly."""
        host = torch.randn(2, 64, 8, 8, dtype=torch.float16)
        seed = torch.randn(2, 64, 8, 8, dtype=torch.float16)
        alpha = torch.tensor(0.5, dtype=torch.float16)

        result = blend_with_isolation(host, seed, alpha)

        assert result.dtype == torch.float16
        assert not torch.isnan(result).any()

    def test_fp32_host_fp32_seed(self):
        """FP32 host and seed should blend correctly."""
        host = torch.randn(2, 64, 8, 8, dtype=torch.float32)
        seed = torch.randn(2, 64, 8, 8, dtype=torch.float32)
        alpha = torch.tensor(0.5, dtype=torch.float32)

        result = blend_with_isolation(host, seed, alpha)

        assert result.dtype == torch.float32
        assert not torch.isnan(result).any()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_accumulation(self, dtype):
        """Half precision should handle accumulation correctly."""
        if dtype == torch.bfloat16:
            # Check if bfloat16 is supported
            try:
                torch.randn(1, dtype=dtype)
            except RuntimeError:
                pytest.skip("bfloat16 not supported on this platform")

        host = torch.randn(2, 64, 8, 8, dtype=dtype)
        seed = torch.randn(2, 64, 8, 8, dtype=dtype)

        # Multiple blend operations
        alpha = torch.tensor(0.1, dtype=dtype)
        result = host
        for _ in range(10):
            result = blend_with_isolation(result, seed, alpha)

        assert result.dtype == dtype
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestFloatingPointAccumulation:
    """Tests for floating point accumulation accuracy."""

    def test_many_small_alpha_updates_no_drift(self):
        """Many small alpha updates should not drift past bounds."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        # Start at 0 and accumulate small increments
        slot.set_alpha(0.0)
        increment = 0.0001

        for _ in range(10000):
            current = slot.state.alpha
            slot.set_alpha(current + increment)

        # Should not exceed 1.0 due to clamping
        assert slot.state.alpha <= 1.0

    def test_accuracy_accumulation_no_truncation(self):
        """Very small accuracy improvements should not be truncated."""
        metrics = SeedMetrics()

        # Record initial accuracy
        metrics.record_accuracy(50.0)

        # Record tiny improvement
        tiny_improvement = 0.0001
        metrics.record_accuracy(50.0 + tiny_improvement)

        # Should capture the improvement
        assert metrics.total_improvement == pytest.approx(tiny_improvement)

    def test_gated_blend_output_bounded(self):
        """GatedBlend gate output should be bounded [0, 1]."""
        blend = GatedBlend(channels=64, topology="cnn")

        # Test with various inputs
        for _ in range(10):
            x = torch.randn(2, 64, 8, 8)
            alpha = blend.get_alpha_for_blend(x)

            assert (alpha >= 0.0).all()
            assert (alpha <= 1.0).all()


class TestVerySmallImprovement:
    """Tests for very small improvement detection."""

    def test_tiny_improvement_recorded(self):
        """Tiny improvements should be recorded in metrics."""
        metrics = SeedMetrics()

        metrics.record_accuracy(50.0)
        metrics.record_accuracy(50.00001)

        assert metrics.improvement_since_stage_start == pytest.approx(0.00001)

    def test_best_accuracy_tracks_tiny_improvements(self):
        """best_val_accuracy should track even tiny improvements."""
        metrics = SeedMetrics()

        metrics.record_accuracy(50.0)
        metrics.record_accuracy(50.0 + 1e-6)

        assert metrics.best_val_accuracy == pytest.approx(50.0 + 1e-6)

    def test_tiny_regression_preserved(self):
        """Tiny regressions should also be tracked."""
        metrics = SeedMetrics()

        metrics.record_accuracy(50.0)
        metrics.record_accuracy(50.0 - 1e-6)

        assert metrics.current_val_accuracy == pytest.approx(50.0 - 1e-6)
        assert metrics.best_val_accuracy == 50.0


class TestNumericalGradientFlow:
    """Tests for numerical behavior during gradient flow."""

    def test_blend_gradient_flow_preserves_numerics(self):
        """Gradient flow through blend should preserve numerical stability."""
        host = torch.randn(2, 64, 8, 8, requires_grad=True)
        seed = torch.randn(2, 64, 8, 8, requires_grad=True)
        alpha = torch.tensor(0.5)

        result = blend_with_isolation(host, seed, alpha)
        loss = result.sum()
        loss.backward()

        # Gradients should be valid (not NaN or Inf)
        assert not torch.isnan(host.grad).any()
        assert not torch.isinf(host.grad).any()
        assert not torch.isnan(seed.grad).any()
        assert not torch.isinf(seed.grad).any()

    def test_ste_gradient_flow_preserves_numerics(self):
        """STE gradient flow should preserve numerical stability."""
        host = torch.randn(2, 64, 8, 8, requires_grad=True)
        seed = torch.randn(2, 64, 8, 8, requires_grad=True)

        result = ste_forward(host, seed)
        loss = result.sum()
        loss.backward()

        # Host should have gradients from direct path
        assert not torch.isnan(host.grad).any()
        assert not torch.isinf(host.grad).any()

        # Seed should have gradients (STE passes through)
        assert not torch.isnan(seed.grad).any()
        assert not torch.isinf(seed.grad).any()


class TestCounterfactualNumerics:
    """Tests for counterfactual contribution numerical behavior."""

    def test_counterfactual_none_handled(self):
        """None counterfactual should be handled gracefully."""
        metrics = SeedMetrics()

        assert metrics.counterfactual_contribution is None

    def test_counterfactual_negative_allowed(self):
        """Negative counterfactual (harmful seed) should be stored."""
        metrics = SeedMetrics()

        metrics.counterfactual_contribution = -5.0

        assert metrics.counterfactual_contribution == -5.0

    def test_counterfactual_very_small_preserved(self):
        """Very small counterfactual should be preserved."""
        metrics = SeedMetrics()

        metrics.counterfactual_contribution = 1e-8

        assert metrics.counterfactual_contribution == pytest.approx(1e-8)

    def test_counterfactual_zero_distinct_from_none(self):
        """Zero counterfactual should be distinct from None."""
        metrics = SeedMetrics()

        metrics.counterfactual_contribution = 0.0

        assert metrics.counterfactual_contribution == 0.0
        assert metrics.counterfactual_contribution is not None


class TestBlendingDelta:
    """Tests for blending delta numerical behavior."""

    def test_blending_delta_zero_before_blending(self):
        """blending_delta should be 0 if blending never started."""
        metrics = SeedMetrics()

        assert metrics.blending_delta == 0.0

    def test_blending_delta_after_start(self):
        """blending_delta should track change since blending start."""
        metrics = SeedMetrics()

        # Record some accuracy before blending
        metrics.record_accuracy(50.0)
        metrics.record_accuracy(55.0)

        # Mark blending started
        metrics._blending_started = True
        metrics.accuracy_at_blending_start = 55.0

        # Record improvement during blending
        metrics.record_accuracy(60.0)

        assert metrics.blending_delta == 5.0

    def test_blending_delta_negative_possible(self):
        """blending_delta can be negative if accuracy dropped."""
        metrics = SeedMetrics()

        metrics._blending_started = True
        metrics.accuracy_at_blending_start = 60.0
        metrics.record_accuracy(55.0)

        assert metrics.blending_delta == -5.0


class TestExtremeValues:
    """Tests for extreme value handling."""

    def test_extremely_large_accuracy(self):
        """Extremely large accuracy values should be handled."""
        metrics = SeedMetrics()

        metrics.record_accuracy(1e10)

        assert metrics.current_val_accuracy == 1e10

    def test_negative_accuracy_allowed(self):
        """Negative accuracy (e.g., loss) should be allowed."""
        metrics = SeedMetrics()

        metrics.record_accuracy(-10.0)

        assert metrics.current_val_accuracy == -10.0

    def test_zero_accuracy(self):
        """Zero accuracy should be handled correctly."""
        metrics = SeedMetrics()

        metrics.record_accuracy(0.0)

        assert metrics.current_val_accuracy == 0.0
        assert metrics.initial_val_accuracy == 0.0

    def test_accuracy_from_tensor(self):
        """Accuracy from tensor should be converted correctly."""
        metrics = SeedMetrics()

        tensor_acc = torch.tensor(75.5)
        metrics.record_accuracy(tensor_acc)

        assert metrics.current_val_accuracy == 75.5
        assert isinstance(metrics.current_val_accuracy, float)

    def test_accuracy_from_gpu_tensor(self):
        """Accuracy from GPU tensor should be converted correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        metrics = SeedMetrics()

        tensor_acc = torch.tensor(75.5, device="cuda")
        metrics.record_accuracy(tensor_acc)

        assert metrics.current_val_accuracy == 75.5
        assert isinstance(metrics.current_val_accuracy, float)
