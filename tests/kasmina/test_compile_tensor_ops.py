"""Test that tensor operations are torch.compile compatible."""
import torch

from esper.kasmina.isolation import blend_with_isolation, ste_forward


class TestCompilableTensorOps:
    """Verify tensor ops compile without graph breaks."""

    def test_ste_forward_compiles_fullgraph(self):
        """STE forward should compile with fullgraph=True (no graph breaks)."""
        compiled_ste = torch.compile(ste_forward, fullgraph=True)

        host = torch.randn(2, 32, 8, 8, requires_grad=True)
        seed = torch.randn(2, 32, 8, 8, requires_grad=True)

        result = compiled_ste(host, seed)

        assert result.shape == host.shape
        # Verify STE property: forward equals host
        assert torch.allclose(result, host, atol=1e-6)

    def test_ste_forward_gradient_flow(self):
        """STE backward should flow gradients to seed only."""
        host = torch.randn(2, 32, 8, 8, requires_grad=True)
        seed = torch.randn(2, 32, 8, 8, requires_grad=True)

        result = ste_forward(host, seed)
        loss = result.sum()
        loss.backward()

        # Host should have gradients (it's in the computation)
        assert host.grad is not None
        # Seed should have gradients via STE
        assert seed.grad is not None

    def test_blend_with_isolation_compiles_fullgraph(self):
        """Blend should compile with fullgraph=True."""
        compiled_blend = torch.compile(blend_with_isolation, fullgraph=True)

        host = torch.randn(2, 32, 8, 8)
        seed = torch.randn(2, 32, 8, 8)

        result = compiled_blend(host, seed, 0.5)

        assert result.shape == host.shape
