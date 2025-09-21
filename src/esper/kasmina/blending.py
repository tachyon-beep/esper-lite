"""Alpha blending utilities for seed grafting."""

from __future__ import annotations

from dataclasses import dataclass

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
        return torch.sigmoid(torch.tensor(scaled, dtype=torch.float32)).item()


class AlphaBlender:
    """Produces detached host/seed blends."""

    def blend(self, host: torch.Tensor, seed: torch.Tensor, alpha: float) -> torch.Tensor:
        alpha_clamped = float(max(0.0, min(1.0, alpha)))
        return alpha_clamped * seed + (1.0 - alpha_clamped) * host.detach()

