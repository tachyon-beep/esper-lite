"""Test that tensor operations are torch.compile compatible."""
import torch

from esper.kasmina.blend_ops import blend_add, blend_gate, blend_multiply
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
        """Blend should compile with fullgraph=True.

        Task 5: blend_with_isolation now requires tensor alpha.
        """
        compiled_blend = torch.compile(blend_with_isolation, fullgraph=True)

        host = torch.randn(2, 32, 8, 8)
        seed = torch.randn(2, 32, 8, 8)
        alpha = torch.tensor(0.5)

        result = compiled_blend(host, seed, alpha)

        assert result.shape == host.shape

    def test_blend_ops_compile_fullgraph(self):
        """Phase 3 smoke: blend operators are fullgraph-compilable."""
        compiled_add = torch.compile(blend_add, fullgraph=True)
        compiled_mul = torch.compile(blend_multiply, fullgraph=True)
        compiled_gate = torch.compile(blend_gate, fullgraph=True)

        host = torch.randn(2, 32, 8, 8)
        seed = torch.randn(2, 32, 8, 8)
        alpha = torch.tensor(0.7)
        gate = torch.rand(2, 1, 1, 1)

        out_add = compiled_add(host, seed, alpha)
        out_mul = compiled_mul(host, seed, alpha)
        out_gate = compiled_gate(host, seed, alpha, gate)

        assert out_add.shape == host.shape
        assert out_mul.shape == host.shape
        assert out_gate.shape == host.shape
