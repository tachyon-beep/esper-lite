"""Tolaria - Model Training Infrastructure

This package provides:
- environment: Model factory (create_model)
- governor: Fail-safe watchdog for catastrophic failure detection

Training loops are implemented inline in simic/training/vectorized.py
for performance (CUDA streams, AMP, multi-env parallelism).

IMPORTANT: This module uses PEP 562 lazy imports to avoid loading torch
and the telemetry hub at import time. Imports are deferred until first access.
"""

from typing import TYPE_CHECKING, Any

__all__ = [
    "create_model",
    "parse_device",
    "validate_device",
    "TolariaGovernor",
    "GovernorReport",
]

# TYPE_CHECKING imports for static analysis (mypy, IDE autocomplete)
# These are never executed at runtime, preserving lazy import behavior.
if TYPE_CHECKING:
    from esper.tolaria.environment import create_model as create_model
    from esper.tolaria.environment import parse_device as parse_device
    from esper.tolaria.environment import validate_device as validate_device
    from esper.tolaria.governor import GovernorReport as GovernorReport
    from esper.tolaria.governor import TolariaGovernor as TolariaGovernor


# Mapping from public name to (module, attr) for lazy imports
# Single source of truth for __getattr__ and __dir__
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "create_model": ("esper.tolaria.environment", "create_model"),
    "parse_device": ("esper.tolaria.environment", "parse_device"),
    "validate_device": ("esper.tolaria.environment", "validate_device"),
    "TolariaGovernor": ("esper.tolaria.governor", "TolariaGovernor"),
    "GovernorReport": ("esper.tolaria.governor", "GovernorReport"),
}


def __getattr__(name: str) -> Any:
    """Lazy import using PEP 562.

    Defers heavy imports (torch, telemetry hub) until actual use.
    Caches the result in module globals to avoid repeated imports.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache in module globals so subsequent accesses bypass __getattr__
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public API for introspection (dir(), IDE, debuggers)."""
    # Combine __all__ exports with actual module globals
    return sorted(set(__all__) | set(globals().keys()))
