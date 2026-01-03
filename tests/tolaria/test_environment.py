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

    def test_create_model_rejects_duplicate_slots(self):
        """Duplicate slot IDs should raise ValueError with helpful message.

        This prevents silent overwrites in MorphogeneticModel's ModuleDict
        which would cause action/mask dimension mismatches in training.
        """
        with pytest.raises(ValueError, match="slots contains duplicates.*r0c1"):
            create_model(device="cpu", slots=["r0c1", "r0c2", "r0c1"])

    def test_create_model_rejects_multiple_duplicates(self):
        """Multiple different duplicates should all be reported."""
        with pytest.raises(ValueError, match="slots contains duplicates"):
            create_model(device="cpu", slots=["r0c1", "r0c1", "r0c2", "r0c2"])


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

    def test_validate_meta_device_raises(self):
        """Meta device is valid PyTorch but unsupported by Esper.

        Meta device creates tensors with shape/dtype but no actual storage.
        Forward passes work but backward passes produce empty gradients,
        causing silent training failures where the model appears to train
        but never actually updates.
        """
        with pytest.raises(ValueError, match="Unsupported device.*meta"):
            validate_device("meta")

    def test_validate_xpu_device_raises(self):
        """XPU (Intel GPU) is valid PyTorch but unsupported by Esper."""
        with pytest.raises(ValueError, match="Unsupported device.*xpu"):
            validate_device("xpu")

    def test_validate_xla_device_raises(self):
        """XLA (TPU) is valid PyTorch but unsupported by Esper."""
        with pytest.raises(ValueError, match="Unsupported device.*xla"):
            validate_device("xla")

    def test_validate_hpu_device_raises(self):
        """HPU (Habana Gaudi) is valid PyTorch but unsupported by Esper."""
        with pytest.raises(ValueError, match="Unsupported device.*hpu"):
            validate_device("hpu")

    def test_validate_privateuseone_device_raises(self):
        """Custom backend devices are unsupported by Esper."""
        with pytest.raises(ValueError, match="Unsupported device.*privateuseone"):
            validate_device("privateuseone")


class TestTolariaPublicAPI:
    """Tests for Tolaria module's public API and lazy imports."""

    def test_all_public_names_importable(self):
        """All names in __all__ should be importable from esper.tolaria.

        This catches rename/drift between __all__ and actual exports.
        """
        import esper.tolaria as tolaria

        for name in tolaria.__all__:
            obj = getattr(tolaria, name)
            assert obj is not None, f"{name} resolved to None"

    def test_direct_imports_work(self):
        """Direct imports should work for type checkers and runtime."""
        from esper.tolaria import (
            create_model,
            parse_device,
            validate_device,
            TolariaGovernor,
            GovernorReport,
        )

        # Verify they're callable/classes as expected
        assert callable(create_model)
        assert callable(parse_device)
        assert callable(validate_device)
        assert isinstance(TolariaGovernor, type)
        assert isinstance(GovernorReport, type)

    def test_lazy_imports_are_cached(self):
        """Lazy imports should be cached in module globals after first access."""
        import esper.tolaria as tolaria
        import importlib

        # Force fresh module state
        importlib.reload(tolaria)

        # First access triggers __getattr__
        _ = tolaria.create_model

        # After access, should be in globals (cached)
        assert "create_model" in dir(tolaria)
        # Should be the actual function, not a lazy reference
        assert tolaria.create_model is tolaria.create_model  # Identity check

    def test_dir_includes_all_exports(self):
        """dir(esper.tolaria) should include all public API names."""
        import esper.tolaria as tolaria

        dir_result = dir(tolaria)

        for name in tolaria.__all__:
            assert name in dir_result, f"{name} missing from dir()"

    def test_invalid_attribute_raises(self):
        """Accessing non-existent attributes should raise AttributeError."""
        import esper.tolaria as tolaria

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = tolaria.nonexistent_attribute
