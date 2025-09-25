"""Blending utilities for seed grafting.

Implements the default convex blend and extended executor-only modes per
prototype-delta guidance (Residual, Channel/Group-wise, Confidence-gated).

All paths preserve isolation by detaching the host branch.

Design: docs/prototype-delta/kasmina/blending-upgrade.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import math
import torch


@dataclass(slots=True)
class AlphaSchedule:
    """Maintains alpha progression for blending."""

    total_steps: int
    temperature: float = 1.0

    def value(self, step: int) -> float:
        if self.total_steps <= 0:
            return 1.0
        midpoint = self.total_steps / 2
        scaled = (step - midpoint) / max(self.temperature, 1e-6)
        return 0.5 * (1.0 + math.tanh(scaled * 0.5))


class AlphaBlender:
    """Produces detached host/seed blends."""

    def blend(self, host: torch.Tensor, seed: torch.Tensor, alpha: float) -> torch.Tensor:
        alpha_clamped = float(max(0.0, min(1.0, alpha)))
        return alpha_clamped * seed + (1.0 - alpha_clamped) * host.detach()


@dataclass(slots=True)
class BlenderConfig:
    """Configuration for per-seed blending.

    - mode: which blend kernel to use.
    - alpha_vec: optional channel/group-wise alpha weights (vector-like).
    - gate_k, gate_tau: confidence gate hyperparameters.
    - alpha_lo, alpha_hi: clamp bounds for effective alpha in confidence mode.
    """

    mode: str = "CONVEX"
    alpha_vec: Optional[Iterable[float]] = None
    gate_k: float = 4.0
    gate_tau: float = 0.2
    alpha_lo: float = 0.0
    alpha_hi: float = 1.0


def _clamp01(value: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(value, torch.Tensor):
        return value.clamp_(0.0, 1.0)
    return float(max(0.0, min(1.0, float(value))))


def _broadcast_alpha_vec(alpha_vec: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a channel vector [C] across reference tensor with channels at dim=1.

    Resulting shape: [1, C, 1, 1, ...] with as many trailing singleton dims as needed
    to match `ref.dim()`.
    """

    a = alpha_vec
    while a.dim() > 1:
        a = a.squeeze()
    # Ensure at least [1, C]
    if a.dim() == 0:
        a = a.unsqueeze(0)
    a = a.unsqueeze(0) if a.dim() == 1 else a  # [1, C]
    # Add spatial singleton dims to match ref rank
    while a.dim() < ref.dim():
        a = a.unsqueeze(-1)
    return a


def blend_convex(host: torch.Tensor, seed: torch.Tensor, alpha: float) -> torch.Tensor:
    a = float(_clamp01(alpha))
    return a * seed + (1.0 - a) * host.detach()


def blend_residual(host: torch.Tensor, seed: torch.Tensor, alpha: float) -> torch.Tensor:
    a = float(_clamp01(alpha))
    return host.detach() + a * seed


def blend_channelwise(
    host: torch.Tensor,
    seed: torch.Tensor,
    alpha_base: float,
    alpha_vec: torch.Tensor,
) -> torch.Tensor:
    a_base = float(_clamp01(alpha_base))
    # Broadcast and clamp
    a_vec = _broadcast_alpha_vec(alpha_vec.to(device=seed.device, dtype=seed.dtype), seed)
    a_eff = _clamp01(a_base) * a_vec
    if isinstance(a_eff, torch.Tensor):
        a_eff = a_eff.clamp_(0.0, 1.0)
    return a_eff * seed + (1 - a_eff) * host.detach()


def compute_confidence_gate(seed_logits: torch.Tensor, k: float, tau: float) -> torch.Tensor:
    """Compute per-sample confidence gate from logits.

    gate_n = sigmoid(k * (margin - tau)), margin = top1 - top2 across classes.
    Returned shape: [N] (batch).
    """

    if seed_logits.dim() < 2:
        # Degenerate case: treat as single-class outputs; gate=1
        return torch.ones((seed_logits.shape[0] if seed_logits.dim() > 0 else 1,), device=seed_logits.device, dtype=seed_logits.dtype)
    top2 = torch.topk(seed_logits, k=2, dim=1).values  # [N, 2]
    margin = top2[:, 0] - top2[:, 1]  # [N]
    return torch.sigmoid(torch.as_tensor(k, device=margin.device, dtype=margin.dtype) * (margin - torch.as_tensor(tau, device=margin.device, dtype=margin.dtype)))


def blend_confidence(
    host: torch.Tensor,
    seed: torch.Tensor,
    alpha_base: float,
    gate: torch.Tensor,
    *,
    alpha_lo: float = 0.0,
    alpha_hi: float = 1.0,
) -> torch.Tensor:
    a_base = torch.tensor(float(_clamp01(alpha_base)), device=seed.device, dtype=seed.dtype)
    # Gate is treated as a non-trainable modulation signal; detach to avoid backprop through gating path.
    eff = (a_base * gate.detach().to(device=seed.device, dtype=seed.dtype)).clamp_(
        float(alpha_lo), float(alpha_hi)
    )
    # Broadcast eff [N] to match seed dims: unsqueeze along trailing dims
    while eff.dim() < seed.dim():
        eff = eff.unsqueeze(-1)
    return eff * seed + (1 - eff) * host.detach()


def blend_with_config(
    host: torch.Tensor,
    seed: torch.Tensor,
    alpha_base: float,
    config: BlenderConfig,
) -> torch.Tensor:
    mode = (config.mode or "").upper()
    if mode == "CONVEX":
        return blend_convex(host, seed, alpha_base)
    if mode == "RESIDUAL":
        return blend_residual(host, seed, alpha_base)
    if mode == "CHANNEL":
        if config.alpha_vec is None:
            # Fallback to convex when vector missing
            return blend_convex(host, seed, alpha_base)
        alpha_vec = torch.as_tensor(list(config.alpha_vec), device=seed.device, dtype=seed.dtype)
        return blend_channelwise(host, seed, alpha_base, alpha_vec)
    if mode == "CONFIDENCE":
        # Derive gate from seed logits; if shape unexpected, gate=1
        try:
            gate = compute_confidence_gate(seed, config.gate_k, config.gate_tau)
        except Exception:
            gate = torch.ones((seed.shape[0] if seed.dim() > 0 else 1,), device=seed.device, dtype=seed.dtype)
        return blend_confidence(
            host,
            seed,
            alpha_base,
            gate,
            alpha_lo=config.alpha_lo,
            alpha_hi=config.alpha_hi,
        )
    # Default fallback
    return blend_convex(host, seed, alpha_base)


def blend_mode_name(mode: str | int) -> str:
    if isinstance(mode, str):
        m = mode.strip().upper()
        return m if m in {"CONVEX", "RESIDUAL", "CHANNEL", "CONFIDENCE"} else "CONVEX"
    try:
        # Backward compatibility if integer sneaks in
        mapping = {0: "CONVEX", 1: "RESIDUAL", 2: "CHANNEL", 3: "CONFIDENCE"}
        return mapping.get(int(mode), "CONVEX")
    except Exception:
        return "CONVEX"
