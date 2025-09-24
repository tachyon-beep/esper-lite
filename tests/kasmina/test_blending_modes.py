from __future__ import annotations

import torch
import pytest

from esper.kasmina.blending import (
    BlendMode,
    BlenderConfig,
    blend_residual,
    blend_channelwise,
    compute_confidence_gate,
    blend_confidence,
)
from esper.kasmina import KasminaSeedManager, SeedContext


def test_residual_blend_detaches_host_branch() -> None:
    host = torch.tensor([[1.0, 2.0]], requires_grad=True)
    seed = torch.tensor([[3.0, 5.0]], requires_grad=True)
    alpha = 0.3
    out = blend_residual(host, seed, alpha)
    loss = out.sum()
    loss.backward()
    # Host branch must be detached (no/zero gradients)
    assert host.grad is None or torch.allclose(host.grad, torch.zeros_like(host.grad))
    # d(out)/d(seed) = alpha
    assert pytest.approx(seed.grad.mean().item(), rel=1e-5) == alpha


def test_channelwise_blend_broadcasts_over_channels() -> None:
    # Shape [N, C]
    host = torch.ones((2, 3), requires_grad=True)
    seed = torch.ones((2, 3), requires_grad=True)
    alpha_base = 0.5
    alpha_vec = torch.tensor([0.2, 0.6, 1.0])  # per-channel weights
    out = blend_channelwise(host, seed, alpha_base, alpha_vec)
    # Effective alpha per channel
    a_eff = alpha_base * alpha_vec
    expected = a_eff * seed + (1 - a_eff) * host.detach()
    assert torch.allclose(out, expected)
    # Backward: host detached; seed grad equals a_eff (broadcast)
    out.sum().backward()
    assert host.grad is None or torch.allclose(host.grad, torch.zeros_like(host.grad))
    for c in range(3):
        assert pytest.approx(seed.grad[:, c].mean().item(), rel=1e-5) == pytest.approx(a_eff[c].item(), rel=1e-5)


def test_confidence_gate_blend_clamps_alpha() -> None:
    # Simulate logits for 2 classes; first sample confident, second uncertain
    logits = torch.tensor([[3.0, 0.5], [1.0, 0.9]], requires_grad=True)
    host = torch.zeros_like(logits, requires_grad=True)
    alpha_base = 0.6
    gate = compute_confidence_gate(logits, k=4.0, tau=0.1)  # [N]
    out = blend_confidence(host, logits, alpha_base, gate, alpha_lo=0.1, alpha_hi=0.9)
    # α_eff within [0.1, 0.9]
    # Expand gate for comparison
    a_eff = torch.clamp(alpha_base * gate, 0.1, 0.9).unsqueeze(-1).expand_as(logits)
    expected = a_eff * logits + (1 - a_eff) * host.detach()
    assert torch.allclose(out, expected)
    out.sum().backward()
    # Host detached; seed grad equals α_eff per element
    assert host.grad is None or torch.allclose(host.grad, torch.zeros_like(host.grad))
    # Grad check: each element of seed gets weighted by corresponding α_eff
    assert torch.allclose(logits.grad, a_eff)


def test_manager_blend_uses_config_when_present() -> None:
    class _Runtime:
        def fetch_kernel(self, *_args, **_kwargs):
            import torch.nn as nn

            return nn.Identity(), 0.0

    mgr = KasminaSeedManager(_Runtime(), fallback_blueprint_id=None)
    # Pre-create seed context for test
    seed_id = "seed-mode"
    mgr._seeds[seed_id] = SeedContext(seed_id)
    # Set config via helper
    cfg = BlenderConfig(mode=BlendMode.RESIDUAL)
    mgr._set_blend_config_for_test(seed_id, cfg)
    # Also set alpha to a non-trivial value
    mgr._seeds[seed_id].alpha = 0.4  # type: ignore[index]
    host = torch.ones((1, 2))
    seed = 2.0 * torch.ones((1, 2))
    out = mgr.blend(host, seed, seed_id=seed_id)
    # Residual: host.detach() + α * seed = 1 + 0.4*2 = 1.8
    assert torch.allclose(out, torch.full_like(out, 1.8))
