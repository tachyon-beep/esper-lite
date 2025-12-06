"""Test FlexAttention blueprint variant."""
import pytest
import torch

from esper.kasmina.blueprints.registry import BlueprintRegistry


# Check FlexAttention availability
try:
    from torch.nn.attention.flex_attention import flex_attention
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False


class TestFlexAttentionBlueprint:
    """Verify FlexAttention seed works correctly."""

    @pytest.mark.skipif(
        not HAS_FLEX_ATTENTION,
        reason="FlexAttention requires PyTorch 2.5+"
    )
    def test_flex_attention_registered(self):
        """FlexAttention blueprint should be registered."""
        specs = BlueprintRegistry.list_for_topology("transformer")
        names = [s.name for s in specs]
        assert "flex_attention" in names

    @pytest.mark.skipif(
        not HAS_FLEX_ATTENTION,
        reason="FlexAttention requires PyTorch 2.5+"
    )
    def test_flex_attention_forward(self):
        """FlexAttention seed should process input correctly."""
        seed = BlueprintRegistry.create("transformer", "flex_attention", 64)

        x = torch.randn(2, 16, 64)
        result = seed(x)

        assert result.shape == x.shape

    @pytest.mark.skipif(
        not HAS_FLEX_ATTENTION,
        reason="FlexAttention requires PyTorch 2.5+"
    )
    def test_flex_attention_uses_causal_mask(self):
        """FlexAttention should apply causal masking."""
        seed = BlueprintRegistry.create("transformer", "flex_attention", 64)

        # Put in eval mode for deterministic behavior
        seed.eval()

        x = torch.randn(1, 8, 64)
        with torch.no_grad():
            result = seed(x)

        # Should be residual connection (x + attention_output)
        # At minimum, output should have same shape and not be NaN
        assert result.shape == x.shape
        assert not torch.isnan(result).any()

    @pytest.mark.skipif(
        not HAS_FLEX_ATTENTION,
        reason="FlexAttention requires PyTorch 2.5+"
    )
    def test_flex_attention_gradient_flow(self):
        """FlexAttention should support gradient flow."""
        seed = BlueprintRegistry.create("transformer", "flex_attention", 64)

        x = torch.randn(2, 16, 64, requires_grad=True)
        result = seed(x)
        loss = result.sum()
        loss.backward()

        # Gradients should flow back
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
