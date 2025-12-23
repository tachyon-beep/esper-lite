"""Kasmina blend operators (Phase 3 risk-reduction contracts).

These are the *composition* operators applied in SeedSlot.forward() once alpha
amplitude is chosen (by AlphaController) and, optionally, per-sample gating is
applied (via GatedBlend).

This module is intentionally self-contained and pure-tensor so it can be:
- unit tested in isolation (math + gradients),
- smoke-tested under torch.compile (fullgraph=True), and
- wired into SeedSlot with minimal ambiguity.

Contracts (locked here; tests enforce):
- Identity at alpha == 0 for all operators.
- MULTIPLY is bounded via tanh, and is identity when seed_modulation == 0.
- GATE clamps gate to [0, 1] and preserves convex mixing guarantees.
"""

from __future__ import annotations

import torch


def _clamp_unit_interval(x: torch.Tensor) -> torch.Tensor:
    """Clamp x to [0, 1] in a torch.compile friendly way."""
    return x.clamp(0.0, 1.0)


def blend_add(
    host_features: torch.Tensor,
    seed_features: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """ADD operator (default): convex mix / cross-fade.

    Formula:
        y = lerp(h, s, a) = h + a * (s - h)
    """
    alpha = _clamp_unit_interval(alpha)
    # Ensure all tensors match host_features dtype (required for BF16 autocast compatibility)
    target_dtype = host_features.dtype
    seed_features = seed_features.to(target_dtype)
    alpha = alpha.to(target_dtype)
    return torch.lerp(host_features, seed_features, alpha)


def multiply_valve_multiplier(alpha: torch.Tensor, seed_modulation: torch.Tensor) -> torch.Tensor:
    """MULTIPLY valve multiplier (bounded, identity at alpha=0 or seed_modulation=0).

    Locked formula:
        m = 1 + a * tanh(seed_modulation)

    Bounds (for a in [0, 1]):
        1 - a <= m <= 1 + a
    """
    alpha = _clamp_unit_interval(alpha)
    return 1.0 + alpha * torch.tanh(seed_modulation)


def blend_multiply(
    host_features: torch.Tensor,
    seed_features: torch.Tensor,
    alpha: torch.Tensor,
    *,
    seed_input: torch.Tensor | None = None,
) -> torch.Tensor:
    """MULTIPLY operator (valve): modulate host by a bounded, identity-at-zero multiplier.

    In Kasmina, most seeds are *residual* modules that emit features close to the
    host stream (often: `seed(x) = x + delta(x)`). For valve semantics we want a
    modulation signal that is near zero at birth when the seed is identity-like.

    Therefore we define the valve modulation signal as the residual around the
    seed's *input* reference (not necessarily `host_features`):
        m = seed_features - seed_input

    This matters for the CNN isolation contract:
    - When `isolate_gradients=True`, SeedSlot feeds `seed_input = host_features.detach()`
      into the seed. Passing that same `seed_input` here makes m value-identical
      to (seed_features - host_features) but removes the gradient term that would
      otherwise flow through `-host_features` inside tanh(), avoiding sign flips
      and host-gradient coupling to seed outputs.

    Locked formula:
        y = h * (1 + a * tanh(m))
    """
    # Ensure all tensors match host_features dtype (required for BF16 autocast compatibility)
    target_dtype = host_features.dtype
    seed_features = seed_features.to(target_dtype)
    alpha = alpha.to(target_dtype)

    seed_input = host_features if seed_input is None else seed_input.to(target_dtype)
    seed_modulation = seed_features - seed_input
    multiplier = multiply_valve_multiplier(alpha, seed_modulation)
    return host_features * multiplier


def gate_effective_alpha(alpha: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Combine controller amplitude alpha with a per-sample gate in [0, 1]."""
    alpha = _clamp_unit_interval(alpha)
    gate = _clamp_unit_interval(gate)
    return _clamp_unit_interval(alpha * gate)


def blend_gate(
    host_features: torch.Tensor,
    seed_features: torch.Tensor,
    alpha: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """GATE operator: per-sample gated ADD (convex mix).

    This keeps the ADD composition but makes alpha input-dependent:
        a_eff(x) = alpha_amplitude(t) * gate(x)
        y = lerp(h, s, a_eff)

    The per-sample gate is expected to come from `esper.kasmina.blending.GatedBlend`
    (or an equivalent learned gate), and must be in [0, 1].
    """
    effective_alpha = gate_effective_alpha(alpha, gate)
    # Ensure all tensors match host_features dtype (required for BF16 autocast compatibility)
    target_dtype = host_features.dtype
    seed_features = seed_features.to(target_dtype)
    effective_alpha = effective_alpha.to(target_dtype)
    return torch.lerp(host_features, seed_features, effective_alpha)


__all__ = [
    "blend_add",
    "blend_gate",
    "blend_multiply",
    "gate_effective_alpha",
    "multiply_valve_multiplier",
]
