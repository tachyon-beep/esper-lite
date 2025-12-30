"""Tests for PolicyBundle device/compile behavior.

These tests verify that compiled policies reject post-compile device/dtype mutations
that would invalidate the torch.compile graph.
"""

import pytest
import torch

from esper.tamiyo.policy.lstm_bundle import LSTMPolicyBundle


class TestCompiledPolicyImmutability:
    """Post-compile mutations must raise RuntimeError."""

    @pytest.fixture
    def compiled_policy(self) -> LSTMPolicyBundle:
        """Create a compiled policy for testing."""
        policy = LSTMPolicyBundle(feature_dim=8)
        policy.to("cpu")
        policy.compile(mode="default", dynamic=True)
        return policy

    def test_to_raises_after_compile(self, compiled_policy: LSTMPolicyBundle):
        """Compiled policies must reject .to() calls."""
        with pytest.raises(RuntimeError, match=r"\.to\(\).*compiled"):
            compiled_policy.to("cpu")

    def test_cpu_raises_after_compile(self, compiled_policy: LSTMPolicyBundle):
        """Compiled policies must reject .cpu() calls."""
        with pytest.raises(RuntimeError, match=r"\.cpu\(\).*compiled"):
            compiled_policy.cpu()

    def test_half_raises_after_compile(self, compiled_policy: LSTMPolicyBundle):
        """Compiled policies must reject .half() dtype conversion."""
        with pytest.raises(RuntimeError, match=r"\.half\(\).*compiled"):
            compiled_policy.half()

    def test_float_raises_after_compile(self, compiled_policy: LSTMPolicyBundle):
        """Compiled policies must reject .float() dtype conversion."""
        with pytest.raises(RuntimeError, match=r"\.float\(\).*compiled"):
            compiled_policy.float()

    def test_double_raises_after_compile(self, compiled_policy: LSTMPolicyBundle):
        """Compiled policies must reject .double() dtype conversion."""
        with pytest.raises(RuntimeError, match=r"\.double\(\).*compiled"):
            compiled_policy.double()

    def test_bfloat16_raises_after_compile(self, compiled_policy: LSTMPolicyBundle):
        """Compiled policies must reject .bfloat16() dtype conversion."""
        with pytest.raises(RuntimeError, match=r"\.bfloat16\(\).*compiled"):
            compiled_policy.bfloat16()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_raises_after_compile(self, compiled_policy: LSTMPolicyBundle):
        """Compiled policies must reject .cuda() calls."""
        with pytest.raises(RuntimeError, match=r"\.cuda\(\).*compiled"):
            compiled_policy.cuda()


class TestUncompiledPolicyMutability:
    """Pre-compile mutations must work normally."""

    def test_to_works_before_compile(self):
        """Uncompiled policies allow .to() calls."""
        policy = LSTMPolicyBundle(feature_dim=8)
        result = policy.to("cpu")
        assert result is policy  # Returns self for chaining
        assert policy.device == torch.device("cpu")

    def test_cpu_works_before_compile(self):
        """Uncompiled policies allow .cpu() calls."""
        policy = LSTMPolicyBundle(feature_dim=8)
        result = policy.cpu()
        assert result is policy
        assert policy.device == torch.device("cpu")

    def test_half_works_before_compile(self):
        """Uncompiled policies allow .half() dtype conversion."""
        policy = LSTMPolicyBundle(feature_dim=8)
        result = policy.half()
        assert result is policy
        assert policy.dtype == torch.float16

    def test_float_works_before_compile(self):
        """Uncompiled policies allow .float() dtype conversion."""
        policy = LSTMPolicyBundle(feature_dim=8)
        policy.half()  # First convert to half
        result = policy.float()  # Then back to float
        assert result is policy
        assert policy.dtype == torch.float32

    def test_double_works_before_compile(self):
        """Uncompiled policies allow .double() dtype conversion."""
        policy = LSTMPolicyBundle(feature_dim=8)
        result = policy.double()
        assert result is policy
        assert policy.dtype == torch.float64

    def test_bfloat16_works_before_compile(self):
        """Uncompiled policies allow .bfloat16() dtype conversion."""
        policy = LSTMPolicyBundle(feature_dim=8)
        result = policy.bfloat16()
        assert result is policy
        assert policy.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_works_before_compile(self):
        """Uncompiled policies allow .cuda() calls."""
        policy = LSTMPolicyBundle(feature_dim=8)
        result = policy.cuda()
        assert result is policy
        assert policy.device.type == "cuda"
