"""Kasmina alpha controller (pure scheduling logic).

This module is intentionally isolated from SeedSlot wiring so we can unit-test
the alpha scheduling invariants (monotonicity, snap-to-target, HOLD-only
retargeting, checkpoint round-trips) without touching the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

from esper.leyline.alpha import AlphaCurve, AlphaMode


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function.

    For large positive x, exp(-x) is tiny → safe.
    For large negative x, exp(x) is tiny → safe.
    This avoids OverflowError from math.exp() with extreme steepness values.
    """
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _curve_progress(t: float, curve: AlphaCurve, steepness: float = 12.0) -> float:
    """Apply easing curve to linear progress t.

    Args:
        t: Linear progress in [0, 1].
        curve: Which easing curve to apply.
        steepness: Sigmoid steepness (only affects SIGMOID curve).
            Higher values = sharper transition. Default 12.0.

    Returns:
        Eased progress in [0, 1].
    """
    t = max(0.0, min(1.0, t))
    match curve:
        case AlphaCurve.LINEAR:
            return t
        case AlphaCurve.COSINE:
            # Smooth start/end: 0 -> 1 with zero slope at endpoints.
            return 0.5 * (1.0 - math.cos(math.pi * t))
        case AlphaCurve.SIGMOID:
            # Logistic curve normalized to [0, 1] at t in [0, 1].
            # Uses _sigmoid() to avoid OverflowError with extreme steepness.
            x = steepness * (t - 0.5)
            raw = _sigmoid(x)
            raw0 = _sigmoid(-0.5 * steepness)
            raw1 = _sigmoid(0.5 * steepness)
            if raw1 == raw0:
                # Guard against division by zero if steepness -> 0
                return t
            return (raw - raw0) / (raw1 - raw0)
        case _:
            raise ValueError(f"Unknown AlphaCurve: {curve!r}")


@dataclass(slots=True)
class AlphaController:
    """Schedule alpha from start -> target over N controller ticks."""

    alpha: float = 0.0
    alpha_start: float = 0.0
    alpha_target: float = 0.0
    alpha_mode: AlphaMode = AlphaMode.HOLD
    alpha_curve: AlphaCurve = AlphaCurve.LINEAR
    alpha_steepness: float = 12.0  # Sigmoid steepness (default matches original)
    alpha_steps_total: int = 0
    alpha_steps_done: int = 0

    def __post_init__(self) -> None:
        self.alpha = _clamp01(self.alpha)
        self.alpha_start = _clamp01(self.alpha_start)
        self.alpha_target = _clamp01(self.alpha_target)
        self.alpha_steepness = max(0.1, float(self.alpha_steepness))  # Prevent div-by-zero
        self.alpha_steps_total = max(0, int(self.alpha_steps_total))
        self.alpha_steps_done = max(0, int(self.alpha_steps_done))
        self.alpha_steps_done = min(self.alpha_steps_done, self.alpha_steps_total)

    def retarget(
        self,
        *,
        alpha_target: float,
        alpha_steps_total: int,
        alpha_curve: AlphaCurve | None = None,
        alpha_steepness: float | None = None,
    ) -> None:
        """Set a new target and schedule from the current alpha.

        Args:
            alpha_target: Target alpha value in [0, 1].
            alpha_steps_total: Number of controller ticks to reach target.
            alpha_curve: Easing curve (None to keep current).
            alpha_steepness: Sigmoid steepness (None to keep current).

        Contract: retargeting is only allowed from HOLD to prevent alpha dithering
        during a transition.
        """
        if self.alpha_mode != AlphaMode.HOLD:
            raise ValueError("AlphaController.retarget() is only allowed from HOLD")

        target = _clamp01(alpha_target)
        steps_total = max(0, int(alpha_steps_total))

        self.alpha_start = self.alpha
        self.alpha_target = target
        if alpha_curve is not None:
            self.alpha_curve = alpha_curve
        if alpha_steepness is not None:
            self.alpha_steepness = max(0.1, float(alpha_steepness))

        self.alpha_steps_total = steps_total
        self.alpha_steps_done = 0

        if target > self.alpha:
            self.alpha_mode = AlphaMode.UP
        elif target < self.alpha:
            self.alpha_mode = AlphaMode.DOWN
        else:
            self.alpha_mode = AlphaMode.HOLD
            self.alpha = target

        if steps_total == 0:
            self.alpha = target
            self.alpha_mode = AlphaMode.HOLD

    def step(self) -> bool:
        """Advance one controller tick.

        Returns:
            True if the target was reached (snap-to-target applied), else False.
        """
        if self.alpha_mode == AlphaMode.HOLD or self.alpha_steps_total == 0:
            return False

        self.alpha_steps_done += 1

        if self.alpha_steps_done >= self.alpha_steps_total:
            self.alpha_steps_done = self.alpha_steps_total
            self.alpha = self.alpha_target
            self.alpha_mode = AlphaMode.HOLD
            return True

        t = self.alpha_steps_done / max(self.alpha_steps_total, 1)
        progress = _curve_progress(t, self.alpha_curve, self.alpha_steepness)
        value = self.alpha_start + (self.alpha_target - self.alpha_start) * progress
        self.alpha = _clamp01(value)

        if self.alpha_mode == AlphaMode.UP:
            self.alpha = max(self.alpha, self.alpha_start)
            self.alpha = min(self.alpha, self.alpha_target)
        elif self.alpha_mode == AlphaMode.DOWN:
            self.alpha = min(self.alpha, self.alpha_start)
            self.alpha = max(self.alpha, self.alpha_target)

        return False

    def to_dict(self) -> dict[str, int | float]:
        """Primitive serialization for checkpointing."""
        return {
            "alpha": self.alpha,
            "alpha_start": self.alpha_start,
            "alpha_target": self.alpha_target,
            "alpha_mode": int(self.alpha_mode),
            "alpha_curve": int(self.alpha_curve),
            "alpha_steepness": self.alpha_steepness,
            "alpha_steps_total": self.alpha_steps_total,
            "alpha_steps_done": self.alpha_steps_done,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlphaController":
        """Deserialize from checkpoint dict.

        Raises:
            KeyError: If required fields are missing (corrupt checkpoint).
        """
        return cls(
            alpha=float(data["alpha"]),
            alpha_start=float(data["alpha_start"]),
            alpha_target=float(data["alpha_target"]),
            alpha_mode=AlphaMode(int(data["alpha_mode"])),
            alpha_curve=AlphaCurve(int(data["alpha_curve"])),
            alpha_steepness=float(data["alpha_steepness"]),
            alpha_steps_total=int(data["alpha_steps_total"]),
            alpha_steps_done=int(data["alpha_steps_done"]),
        )


__all__ = [
    "AlphaController",
]
