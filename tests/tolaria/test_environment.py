"""Tests for Tolaria environment module."""

import pytest
import torch

from esper.tolaria import create_model


class TestEnvironment:
    """Tests for tolaria.environment module."""

    def test_create_model_cpu(self):
        """Test model creation on CPU."""
        model = create_model(device="cpu", slots=["r0c1"])
        assert not next(model.parameters()).is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_model_cuda(self):
        """Test model creation on CUDA."""
        model = create_model(device="cuda", slots=["r0c1"])
        assert next(model.parameters()).is_cuda

    def test_create_model_invalid_cuda_raises(self):
        """Test error handling for invalid CUDA device."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test error case")

        with pytest.raises(RuntimeError, match="CUDA.*not available"):
            create_model(device="cuda", slots=["r0c1"])
