"""Leyline alpha-controller contracts.

These enums define how Kasmina schedules alpha changes. They are shared
contracts because Tamiyo/Simic will observe and act on them.
"""

from __future__ import annotations

from enum import IntEnum


class AlphaMode(IntEnum):
    """Direction/state of the alpha controller."""

    HOLD = 0
    UP = 1
    DOWN = 2


class AlphaCurve(IntEnum):
    """Easing curve for alpha transitions."""

    LINEAR = 1
    COSINE = 2
    SIGMOID = 3


class AlphaAlgorithm(IntEnum):
    """Blend composition / per-sample gating mode.

    This is a shared contract because Tamiyo/Simic will observe and act on it.

    Notes:
    - This is intentionally separate from AlphaCurve (schedule shape).
    - "GATE" means per-sample gating is enabled; the composition remains ADD.
    """

    ADD = 1
    MULTIPLY = 2
    GATE = 3


__all__ = [
    "AlphaMode",
    "AlphaCurve",
    "AlphaAlgorithm",
]
