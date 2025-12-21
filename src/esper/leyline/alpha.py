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


__all__ = [
    "AlphaMode",
    "AlphaCurve",
]
