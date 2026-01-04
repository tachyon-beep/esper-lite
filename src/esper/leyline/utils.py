"""Leyline Utilities - Pure functions used across domains.

These utilities are domain-agnostic and provide common numerical operations
needed by multiple subsystems.
"""

from __future__ import annotations

import math


def safe(v: float | int | None, default: float = 0.0, max_val: float = 100.0) -> float:
    """Safely convert value to float, handling None/inf/nan.

    Handles Python floats, numpy scalars, and 0-dim torch tensors.
    Raises TypeError for non-numeric types to avoid masking contract violations.

    Args:
        v: Value to convert (can be None, float, int, numpy scalar, 0-dim tensor)
        default: Default value for None/inf/nan
        max_val: Maximum absolute value (clips to [-max_val, max_val])

    Returns:
        Safe float value

    Raises:
        TypeError: If v is not a numeric type that can be converted to float
    """
    if v is None:
        return default
    try:
        v_float = float(v)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"safe() expected numeric, got {type(v)!r}") from exc
    if not math.isfinite(v_float):
        return default
    return max(-max_val, min(v_float, max_val))


__all__ = ["safe"]
