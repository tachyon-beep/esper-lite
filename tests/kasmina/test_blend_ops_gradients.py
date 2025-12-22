"""Gradient + ghost-gradient contracts for Phase 3 blend operators."""

from __future__ import annotations

import torch
import torch.nn as nn

from esper.kasmina.blend_ops import blend_add, blend_gate, blend_multiply
from esper.kasmina.blending import GatedBlend


def _freeze_module_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


class TestBlendOpsGradients:
    def test_add_grads_flow_to_both_inputs(self) -> None:
        host = torch.randn(2, 16, 8, 8, requires_grad=True)
        seed = torch.randn(2, 16, 8, 8, requires_grad=True)
        out = blend_add(host, seed, torch.tensor(0.5))
        out.sum().backward()

        assert host.grad is not None
        assert seed.grad is not None
        assert host.grad.shape == host.shape
        assert seed.grad.shape == seed.shape
        assert torch.isfinite(host.grad).all()
        assert torch.isfinite(seed.grad).all()

    def test_multiply_grads_flow_to_both_inputs(self) -> None:
        host = torch.randn(2, 16, 8, 8, requires_grad=True)
        seed = torch.randn(2, 16, 8, 8, requires_grad=True)
        out = blend_multiply(host, seed, torch.tensor(0.5))
        out.sum().backward()

        assert host.grad is not None
        assert seed.grad is not None
        assert host.grad.shape == host.shape
        assert seed.grad.shape == seed.shape
        assert torch.isfinite(host.grad).all()
        assert torch.isfinite(seed.grad).all()

    def test_gate_grads_flow_to_host_seed_and_gate(self) -> None:
        host = torch.randn(2, 16, 8, 8, requires_grad=True)
        seed = torch.randn(2, 16, 8, 8, requires_grad=True)
        gate = torch.rand(2, 1, 1, 1, requires_grad=True)
        out = blend_gate(host, seed, torch.tensor(0.7), gate)
        out.sum().backward()

        assert host.grad is not None
        assert seed.grad is not None
        assert gate.grad is not None
        assert host.grad.shape == host.shape
        assert seed.grad.shape == seed.shape
        assert gate.grad.shape == gate.shape
        assert torch.isfinite(host.grad).all()
        assert torch.isfinite(seed.grad).all()
        assert torch.isfinite(gate.grad).all()


class TestGhostGradients:
    def test_freeze_seed_params_preserves_host_gradients_add(self) -> None:
        x = torch.randn(2, 16, 8, 8)
        host_encoder = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        seed = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        _freeze_module_params(seed)

        u = host_encoder(x)
        host_features = u.detach()  # remove direct host path (forces ghost-gradient dependency)
        seed_features = seed(u)

        assert seed_features.requires_grad, "requires_grad=False on params must not detach the graph"

        out = blend_add(host_features, seed_features, torch.tensor(0.5))
        out.sum().backward()

        assert host_encoder.weight.grad is not None
        assert host_encoder.weight.grad.abs().sum() > 0
        assert seed.weight.grad is None

    def test_freeze_seed_params_preserves_host_gradients_multiply(self) -> None:
        x = torch.randn(2, 16, 8, 8)
        host_encoder = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        seed = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        _freeze_module_params(seed)

        u = host_encoder(x)
        host_features = u.detach()
        seed_features = seed(u)

        assert seed_features.requires_grad

        out = blend_multiply(host_features, seed_features, torch.tensor(0.7))
        out.sum().backward()

        assert host_encoder.weight.grad is not None
        assert host_encoder.weight.grad.abs().sum() > 0
        assert seed.weight.grad is None

    def test_isolate_gradients_multiply_uses_seed_input_reference(self) -> None:
        """Isolation contract: with a detached seed_input reference, host gradients are stable.

        This locks the Phase 3 rule that MULTIPLY modulation is defined around the
        seed input reference (which is detached when isolate_gradients=True).
        """
        host = torch.randn(2, 16, 8, 8).detach().requires_grad_(True)
        seed_input = host.detach()

        # Non-zero modulation, but computed from detached inputs.
        seed_features = seed_input + 10.0  # tanh(delta) ~= 1 (constant)
        alpha_value = 0.7
        alpha = torch.tensor(alpha_value)

        out = blend_multiply(host, seed_features, alpha, seed_input=seed_input)
        out.sum().backward()

        assert host.grad is not None

        # dy/dh == multiplier (constant, positive, no sign flips)
        expected_multiplier = 1.0 + alpha_value * torch.tanh(torch.tensor(10.0)).item()
        assert torch.allclose(host.grad, torch.full_like(host.grad, expected_multiplier), atol=1e-6)

    def test_freeze_seed_and_gate_params_preserves_host_gradients_gate(self) -> None:
        x = torch.randn(2, 16, 8, 8)
        host_encoder = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        seed = nn.Conv2d(16, 16, kernel_size=1, bias=False)
        gate = GatedBlend(channels=16, topology="cnn", total_steps=10)

        _freeze_module_params(seed)
        _freeze_module_params(gate)

        u = host_encoder(x)
        host_features = u.detach()
        seed_features = seed(u)
        per_sample_gate = gate.get_alpha_for_blend(u)

        assert seed_features.requires_grad
        assert per_sample_gate.requires_grad, "Frozen gate params must still allow grads to flow to host features"

        out = blend_gate(host_features, seed_features, torch.tensor(0.8), per_sample_gate)
        out.sum().backward()

        assert host_encoder.weight.grad is not None
        assert host_encoder.weight.grad.abs().sum() > 0

        # "Freeze" is param-only: no grads to seed or gate params, but graph remains intact.
        assert seed.weight.grad is None
        assert all(p.grad is None for p in gate.parameters())
