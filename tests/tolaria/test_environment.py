"""Tests for Tolaria environment module."""

import pytest
import torch

from esper.tolaria import create_model, parse_device, validate_device


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

    def test_create_model_returns_morphogenetic_model(self):
        """Verify create_model returns MorphogeneticModel, not just nn.Module."""
        from esper.kasmina.host import MorphogeneticModel

        model = create_model(device="cpu", slots=["r0c1"])
        assert isinstance(model, MorphogeneticModel)
        # Verify the interface that downstream code depends on
        assert hasattr(model, "seed_slots")
        assert hasattr(model, "get_host_parameters")
        assert hasattr(model, "host")


class TestParseDevice:
    """Tests for parse_device function."""

    def test_parse_cpu(self):
        """Parse 'cpu' device string."""
        dev = parse_device("cpu")
        assert dev.type == "cpu"
        assert dev.index is None

    def test_parse_cuda_bare(self):
        """Parse bare 'cuda' device string."""
        dev = parse_device("cuda")
        assert dev.type == "cuda"
        assert dev.index is None

    def test_parse_cuda_indexed(self):
        """Parse indexed 'cuda:N' device string."""
        dev = parse_device("cuda:0")
        assert dev.type == "cuda"
        assert dev.index == 0

        dev = parse_device("cuda:1")
        assert dev.type == "cuda"
        assert dev.index == 1

    def test_parse_malformed_cuda_raises(self):
        """Malformed device strings like 'cuda0' should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid device 'cuda0'"):
            parse_device("cuda0")

    def test_parse_invalid_string_raises(self):
        """Completely invalid device strings should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            parse_device("notadevice")


class TestValidateDevice:
    """Tests for validate_device function."""

    def test_validate_cpu_always_succeeds(self):
        """CPU device validation should always succeed."""
        dev = validate_device("cpu")
        assert dev.type == "cpu"

    def test_validate_cpu_with_require_explicit_index(self):
        """require_explicit_index should not affect CPU devices."""
        dev = validate_device("cpu", require_explicit_index=True)
        assert dev.type == "cpu"

    def test_validate_cuda_unavailable_raises(self):
        """CUDA validation should fail when CUDA is unavailable."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test error case")

        with pytest.raises(RuntimeError, match="CUDA.*not available"):
            validate_device("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_validate_cuda_bare_succeeds(self):
        """Bare 'cuda' should succeed when not requiring explicit index."""
        dev = validate_device("cuda", require_explicit_index=False)
        assert dev.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_validate_cuda_bare_fails_with_require_explicit_index(self):
        """Bare 'cuda' should fail when requiring explicit index."""
        with pytest.raises(ValueError, match="must include an explicit index"):
            validate_device("cuda", require_explicit_index=True)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_validate_cuda_explicit_index_succeeds(self):
        """'cuda:0' should succeed when requiring explicit index."""
        dev = validate_device("cuda:0", require_explicit_index=True)
        assert dev.type == "cuda"
        assert dev.index == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_validate_cuda_out_of_range_raises(self):
        """Out-of-range CUDA index should raise RuntimeError."""
        available = torch.cuda.device_count()
        out_of_range = f"cuda:{available + 100}"

        with pytest.raises(RuntimeError, match="only .* device.*available"):
            validate_device(out_of_range)

    def test_validate_malformed_device_raises(self):
        """Malformed device strings should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            validate_device("cuda0")  # Missing colon
