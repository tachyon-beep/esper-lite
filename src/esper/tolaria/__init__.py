"""Tolaria - Model Training Infrastructure

This package provides:
- environment: Model factory (create_model)
- governor: Fail-safe watchdog for catastrophic failure detection

Training loops are implemented inline in simic/training/vectorized.py
for performance (CUDA streams, AMP, multi-env parallelism).

IMPORTANT: This module uses PEP 562 lazy imports to avoid loading torch
and the telemetry hub at import time. Imports are deferred until first access.
"""

__all__ = [
    "create_model",
    "TolariaGovernor",
    "GovernorReport",
]


from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy import using PEP 562.

    Defers heavy imports (torch, telemetry hub) until actual use.
    """
    if name == "create_model":
        from esper.tolaria.environment import create_model
        return create_model
    elif name == "TolariaGovernor":
        from esper.tolaria.governor import TolariaGovernor
        return TolariaGovernor
    elif name == "GovernorReport":
        from esper.tolaria.governor import GovernorReport
        return GovernorReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
