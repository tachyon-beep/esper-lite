from __future__ import annotations

import math

import pytest


def _schedule_values(
    curve: str,
    steps: int,
    *,
    start: float = 1.0,
    target: float = 0.0,
    steepness: float = 1.0,
) -> list[float]:
    """Deterministic alpha schedule with exact endpoints (snap-to-target).

    This is a Phase-0 calibration helper: it encodes the *math* we intend to use
    for convex shock calibration without depending on runtime Kasmina code.
    """
    if steps <= 0:
        return [start, target]

    values = [start]
    for i in range(1, steps):
        t = i / steps
        if curve == "linear":
            u = t
        elif curve == "cosine":
            u = 0.5 * (1.0 - math.cos(math.pi * t))
        elif curve == "sigmoid":
            x = (t - 0.5) * 12.0 * steepness
            u = 1.0 / (1.0 + math.exp(-x))
        else:
            raise ValueError(f"Unknown curve: {curve}")

        values.append(start + (target - start) * u)

    values.append(target)
    return values


def _shock_sum(values: list[float]) -> float:
    """Convex shock: Σ (Δalpha^2)."""
    return sum((b - a) ** 2 for a, b in zip(values, values[1:]))


@pytest.mark.property
@pytest.mark.parametrize("curve", ["linear", "cosine", "sigmoid"])
def test_convex_shock_prefers_slower_schedules(curve: str) -> None:
    s1 = _shock_sum(_schedule_values(curve, 1))
    s3 = _shock_sum(_schedule_values(curve, 3))
    s5 = _shock_sum(_schedule_values(curve, 5))
    s8 = _shock_sum(_schedule_values(curve, 8))
    assert s1 > s3 > s5 > s8 > 0.0

