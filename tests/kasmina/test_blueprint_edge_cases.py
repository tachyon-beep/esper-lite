"""Edge case tests for Kasmina Blueprints.

Tests verify correct behavior at boundary conditions:
- NOOP blueprints have zero parameters
- Blueprints work with tiny and large channel dimensions
- Mixed precision (bfloat16/float16) handling
- Shape preservation for all blueprints
"""

import pytest
import torch

from esper.kasmina.blueprints.registry import BlueprintRegistry


class TestNoopBlueprintZeroParams:
    """Tests for NOOP blueprints having zero parameters."""

    def test_cnn_noop_zero_params(self):
        """CNN NOOP blueprint should have zero parameters."""
        module = BlueprintRegistry.create("cnn", "noop", dim=64)

        param_count = sum(p.numel() for p in module.parameters())
        assert param_count == 0

    def test_transformer_noop_zero_params(self):
        """Transformer NOOP blueprint should have zero parameters."""
        module = BlueprintRegistry.create("transformer", "noop", dim=64)

        param_count = sum(p.numel() for p in module.parameters())
        assert param_count == 0

    def test_noop_is_identity(self):
        """NOOP should pass through input unchanged."""
        # CNN
        cnn_module = BlueprintRegistry.create("cnn", "noop", dim=64)
        x_cnn = torch.randn(2, 64, 8, 8)
        torch.testing.assert_close(cnn_module(x_cnn), x_cnn)

        # Transformer
        transformer_module = BlueprintRegistry.create("transformer", "noop", dim=64)
        x_transformer = torch.randn(2, 8, 64)
        torch.testing.assert_close(transformer_module(x_transformer), x_transformer)


class TestBlueprintTinyChannels:
    """Tests for blueprints with minimum channel dimension."""

    @pytest.mark.parametrize("blueprint_name", ["noop", "norm", "attention", "depthwise"])
    def test_cnn_blueprints_tiny_channels(self, blueprint_name: str):
        """CNN blueprints should work with channels=1."""
        module = BlueprintRegistry.create("cnn", blueprint_name, dim=1)

        x = torch.randn(2, 1, 8, 8)
        output = module(x)

        assert output.shape == x.shape

    @pytest.mark.parametrize("blueprint_name", ["noop", "norm", "lora"])
    def test_transformer_blueprints_tiny_channels(self, blueprint_name: str):
        """Transformer blueprints should work with dim=1 (where applicable)."""
        module = BlueprintRegistry.create("transformer", blueprint_name, dim=1)

        x = torch.randn(2, 8, 1)
        output = module(x)

        assert output.shape == x.shape

    def test_transformer_attention_needs_divisible_dim(self):
        """Transformer attention requires dim divisible by n_head."""
        # Default n_head=4, so dim=1 should fail
        with pytest.raises(ValueError, match="dim.*n_head"):
            BlueprintRegistry.create("transformer", "attention", dim=1)


class TestBlueprintLargeChannels:
    """Tests for blueprints with large channel dimensions."""

    def test_cnn_norm_large_channels(self):
        """CNN norm blueprint should work with large channels."""
        module = BlueprintRegistry.create("cnn", "norm", dim=4096)

        # Use small spatial to limit memory
        x = torch.randn(1, 4096, 4, 4)
        output = module(x)

        assert output.shape == x.shape

    def test_transformer_norm_large_dim(self):
        """Transformer norm blueprint should work with large dim."""
        module = BlueprintRegistry.create("transformer", "norm", dim=4096)

        x = torch.randn(1, 8, 4096)
        output = module(x)

        assert output.shape == x.shape

    def test_transformer_lora_large_dim(self):
        """Transformer LoRA blueprint should work with large dim."""
        module = BlueprintRegistry.create("transformer", "lora", dim=4096, rank=16)

        x = torch.randn(1, 8, 4096)
        output = module(x)

        assert output.shape == x.shape


class TestBlueprintMixedPrecision:
    """Tests for mixed precision (bfloat16/float16) handling."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_cnn_norm_mixed_precision(self, dtype: torch.dtype):
        """CNN norm blueprint should work with half precision."""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            # bfloat16 on CPU requires specific hardware
            cpu_backend = getattr(torch.backends, "cpu", None)
            get_capability = getattr(cpu_backend, "get_cpu_capability", None) if cpu_backend is not None else None
            capability = get_capability() if callable(get_capability) else "default"
            if capability == "default":
                pytest.skip("bfloat16 not well supported on this CPU")

        module = BlueprintRegistry.create("cnn", "norm", dim=64).to(dtype)

        x = torch.randn(2, 64, 8, 8, dtype=dtype)
        output = module(x)

        assert output.dtype == dtype
        assert output.shape == x.shape

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_transformer_norm_mixed_precision(self, dtype: torch.dtype):
        """Transformer norm blueprint should work with half precision."""
        module = BlueprintRegistry.create("transformer", "norm", dim=64).to(dtype)

        x = torch.randn(2, 8, 64, dtype=dtype)
        output = module(x)

        assert output.dtype == dtype
        assert output.shape == x.shape

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_transformer_lora_mixed_precision(self, dtype: torch.dtype):
        """Transformer LoRA blueprint should work with half precision."""
        module = BlueprintRegistry.create("transformer", "lora", dim=64).to(dtype)

        x = torch.randn(2, 8, 64, dtype=dtype)
        output = module(x)

        assert output.dtype == dtype
        assert output.shape == x.shape


class TestBlueprintShapePreservation:
    """Tests verifying all blueprints preserve input shape."""

    @pytest.mark.parametrize("blueprint_name", ["noop", "norm", "attention", "depthwise", "conv_light", "conv_heavy"])
    def test_cnn_blueprints_preserve_shape(self, blueprint_name: str):
        """All CNN blueprints should preserve input shape."""
        module = BlueprintRegistry.create("cnn", blueprint_name, dim=64)

        x = torch.randn(2, 64, 16, 16)
        output = module(x)

        assert output.shape == x.shape

    @pytest.mark.parametrize("blueprint_name", ["noop", "norm", "lora", "attention", "mlp"])
    def test_transformer_blueprints_preserve_shape(self, blueprint_name: str):
        """All transformer blueprints should preserve input shape."""
        # Use dim divisible by 4 for attention blueprints
        module = BlueprintRegistry.create("transformer", blueprint_name, dim=64)

        x = torch.randn(2, 16, 64)
        output = module(x)

        assert output.shape == x.shape


class TestBlueprintRegistryEdgeCases:
    """Tests for BlueprintRegistry edge cases."""

    def test_unknown_blueprint_raises(self):
        """Unknown blueprint should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown blueprint"):
            BlueprintRegistry.create("cnn", "nonexistent_blueprint", dim=64)

    def test_unknown_topology_raises(self):
        """Unknown topology should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown blueprint"):
            BlueprintRegistry.create("nonexistent_topology", "norm", dim=64)

    def test_list_for_topology_returns_sorted(self):
        """list_for_topology should return blueprints sorted by param_estimate."""
        blueprints = BlueprintRegistry.list_for_topology("cnn")

        param_estimates = [s.param_estimate for s in blueprints]
        assert param_estimates == sorted(param_estimates)

    def test_get_spec_provides_metadata(self):
        """BlueprintSpec should provide useful metadata."""
        spec = BlueprintRegistry.get("cnn", "norm")

        assert spec.name == "norm"
        assert spec.topology == "cnn"
        assert spec.param_estimate > 0
        assert len(spec.description) > 0


class TestBlueprintGradientFlow:
    """Tests for proper gradient flow through blueprints."""

    @pytest.mark.parametrize("blueprint_name", ["norm", "attention", "depthwise"])
    def test_cnn_blueprint_gradient_flow(self, blueprint_name: str):
        """CNN blueprints should allow gradient flow."""
        module = BlueprintRegistry.create("cnn", blueprint_name, dim=64)

        x = torch.randn(2, 64, 8, 8, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # Parameters should have gradients
        for param in module.parameters():
            assert param.grad is not None

    @pytest.mark.parametrize("blueprint_name", ["norm", "lora", "attention"])
    def test_transformer_blueprint_gradient_flow(self, blueprint_name: str):
        """Transformer blueprints should allow gradient flow."""
        module = BlueprintRegistry.create("transformer", blueprint_name, dim=64)

        x = torch.randn(2, 8, 64, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

        # Parameters should have gradients (except NOOP)
        for param in module.parameters():
            assert param.grad is not None


class TestBlueprintSpecialCases:
    """Tests for special cases in specific blueprints."""

    def test_attention_reduction_clamped(self):
        """CNN attention blueprint should handle small channels with reduction."""
        # With channels=4 and default reduction=4, reduced would be 1
        module = BlueprintRegistry.create("cnn", "attention", dim=4)

        x = torch.randn(2, 4, 8, 8)
        output = module(x)

        assert output.shape == x.shape

    def test_lora_rank_override(self):
        """LoRA blueprint should accept custom rank."""
        module = BlueprintRegistry.create("transformer", "lora", dim=64, rank=16)

        # Check parameter shapes reflect rank=16
        down_weight = module.down.weight
        up_weight = module.up.weight

        assert down_weight.shape == (16, 64)  # (rank, dim)
        assert up_weight.shape == (64, 16)  # (dim, rank)

    def test_transformer_attention_n_head_validation(self):
        """Transformer attention should validate n_head divisibility."""
        # dim=65 is not divisible by n_head=4
        with pytest.raises(ValueError, match="dim.*n_head"):
            BlueprintRegistry.create("transformer", "attention", dim=65)

    def test_mlp_checkpoint_option(self):
        """MLP blueprint should support checkpoint option."""
        # Create with checkpointing enabled
        module = BlueprintRegistry.create("transformer", "mlp", dim=64, checkpoint=True)
        assert module.use_checkpoint is True

        # Create without checkpointing
        module_no_ckpt = BlueprintRegistry.create("transformer", "mlp", dim=64, checkpoint=False)
        assert module_no_ckpt.use_checkpoint is False


class TestGroupNormAdaptation:
    """Tests for GroupNorm adaptation in CNN blueprints."""

    def test_get_num_groups_divisible(self):
        """get_num_groups should return divisor that meets target group size."""
        from esper.kasmina.blueprints.cnn import get_num_groups

        # 64 channels, target 16 per group -> 4 groups
        assert get_num_groups(64) == 4

        # 128 channels, target 16 per group -> 8 groups
        assert get_num_groups(128) == 8

    def test_get_num_groups_prime(self):
        """get_num_groups should fallback for prime channels."""
        from esper.kasmina.blueprints.cnn import get_num_groups

        # Prime numbers fall back to 1 group
        assert get_num_groups(31) == 1
        assert get_num_groups(37) == 1
        assert get_num_groups(41) == 1

    def test_get_num_groups_small(self):
        """get_num_groups should handle very small channels."""
        from esper.kasmina.blueprints.cnn import get_num_groups

        assert get_num_groups(1) == 1
        assert get_num_groups(2) in [1, 2]
        assert get_num_groups(4) in [1, 2, 4]
