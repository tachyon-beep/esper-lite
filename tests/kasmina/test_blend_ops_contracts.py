"""Blend-operator math contracts (Phase 3 risk reduction).

These tests are intentionally *pure* (no SeedSlot wiring) to lock the operator
formulas and invariants before Phase 3 integrates them into the lifecycle.
"""

from __future__ import annotations

import torch

from esper.kasmina.blend_ops import (
    blend_add,
    blend_gate,
    blend_multiply,
    gate_effective_alpha,
    multiply_valve_multiplier,
)


class TestBlendOpsMathContracts:
    def test_add_is_identity_at_alpha_zero(self) -> None:
        host = torch.randn(2, 16, 8, 8)
        seed = torch.randn(2, 16, 8, 8)
        out = blend_add(host, seed, torch.tensor(0.0))
        assert torch.allclose(out, host, atol=0.0, rtol=0.0)

    def test_multiply_is_identity_at_alpha_zero(self) -> None:
        host = torch.randn(2, 16, 8, 8)
        seed = torch.randn(2, 16, 8, 8)
        out = blend_multiply(host, seed, torch.tensor(0.0))
        assert torch.allclose(out, host, atol=0.0, rtol=0.0)

    def test_multiply_is_identity_when_seed_matches_host(self) -> None:
        host = torch.randn(2, 16, 8, 8)
        seed = host.clone()
        for alpha_value in (0.0, 0.3, 0.7, 1.0):
            out = blend_multiply(host, seed, torch.tensor(alpha_value))
            assert torch.allclose(out, host, atol=0.0, rtol=0.0)

    def test_multiply_is_affine_in_alpha_for_fixed_features(self) -> None:
        host = torch.randn(2, 16, 8, 8)
        seed = torch.randn(2, 16, 8, 8)

        a0 = 0.2
        a1 = 0.8
        amid = 0.5 * (a0 + a1)

        y0 = blend_multiply(host, seed, torch.tensor(a0))
        y1 = blend_multiply(host, seed, torch.tensor(a1))
        ymid = blend_multiply(host, seed, torch.tensor(amid))

        assert torch.allclose(ymid, 0.5 * (y0 + y1), atol=1e-6)

    def test_multiply_is_monotone_under_alpha_ramp_for_positive_delta(self) -> None:
        host = torch.rand(2, 16, 8, 8) + 0.1  # strictly positive
        seed_input = host.detach()
        seed = seed_input + 1.0  # positive modulation => tanh(delta) > 0

        prev = blend_multiply(host, seed, torch.tensor(0.0), seed_input=seed_input)
        for alpha_value in (0.2, 0.5, 0.8, 1.0):
            cur = blend_multiply(host, seed, torch.tensor(alpha_value), seed_input=seed_input)
            assert torch.all(cur >= prev)
            prev = cur

    def test_multiply_multiplier_is_bounded(self) -> None:
        seed_modulation = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
        for alpha_value in (0.0, 0.2, 0.5, 1.0):
            alpha = torch.tensor(alpha_value)
            multiplier = multiply_valve_multiplier(alpha, seed_modulation)
            assert torch.all(multiplier >= (1.0 - alpha_value))
            assert torch.all(multiplier <= (1.0 + alpha_value))

    def test_gate_effective_alpha_is_clamped_and_scaled(self) -> None:
        alpha = torch.tensor(0.7)
        gate = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0]).view(-1, 1, 1, 1)
        eff = gate_effective_alpha(alpha, gate)
        assert torch.all((0.0 <= eff) & (eff <= 1.0))
        assert torch.allclose(eff[0], torch.tensor(0.0))  # gate < 0 clamps to 0
        assert torch.allclose(eff[1], torch.tensor(0.0))  # gate == 0
        assert torch.allclose(eff[2], torch.tensor(0.35))  # 0.7 * 0.5
        assert torch.allclose(eff[3], torch.tensor(0.7))  # 0.7 * 1.0
        assert torch.allclose(eff[4], torch.tensor(0.7))  # gate > 1 clamps to 1

    def test_gate_is_convex_mix_of_host_and_seed(self) -> None:
        host = torch.randn(2, 16, 8, 8)
        seed = torch.randn(2, 16, 8, 8)

        alpha = torch.tensor(0.6)
        gate = torch.full((2, 1, 1, 1), 0.5)  # effective alpha = 0.3
        eff = gate_effective_alpha(alpha, gate)
        out = blend_gate(host, seed, alpha, gate)
        expected = torch.lerp(host, seed, eff)

        assert torch.allclose(out, expected, atol=1e-6)
        assert torch.isfinite(out).all()

    def test_ops_do_not_emit_nans_for_finite_inputs(self) -> None:
        host = torch.randn(2, 16, 8, 8) * 1e3
        seed = torch.randn(2, 16, 8, 8) * 1e3
        alpha = torch.tensor(0.8)
        gate = torch.rand(2, 1, 1, 1)

        out_add = blend_add(host, seed, alpha)
        out_mul = blend_multiply(host, seed, alpha)
        out_gate = blend_gate(host, seed, alpha, gate)

        assert torch.isfinite(out_add).all()
        assert torch.isfinite(out_mul).all()
        assert torch.isfinite(out_gate).all()
