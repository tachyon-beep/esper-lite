from __future__ import annotations

import math

import pytest


def _multiply_valve(host: float, seed_output: float, alpha: float) -> float:
    """Locked valve formula from the kasmina blend/prune retooling plan."""
    return host * (1.0 + alpha * math.tanh(seed_output))


@pytest.mark.property
def test_multiply_valve_is_identity_at_alpha_zero() -> None:
    for host in (-3.0, -0.2, 0.0, 0.2, 3.0):
        for seed_output in (-100.0, -1.0, 0.0, 1.0, 100.0):
            assert _multiply_valve(host, seed_output, 0.0) == host


@pytest.mark.property
def test_multiply_valve_is_identity_when_seed_output_is_zero() -> None:
    for host in (-3.0, -0.2, 0.0, 0.2, 3.0):
        for alpha in (0.0, 0.5, 1.0):
            assert _multiply_valve(host, 0.0, alpha) == host


@pytest.mark.property
def test_multiply_valve_multiplier_is_bounded() -> None:
    for alpha in (0.0, 0.2, 0.5, 1.0):
        for seed_output in (-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0):
            multiplier = 1.0 + alpha * math.tanh(seed_output)
            assert (1.0 - alpha) <= multiplier <= (1.0 + alpha)

