"""Lazy blueprint module loader.

This module provides on-demand loading of blueprint modules to avoid
eager imports of heavy dependencies (e.g., FlexAttention in transformer
blueprints) when they're not needed.

The loader is topology-scoped: only the blueprint modules for requested
topologies are imported. This means CNN-only training never imports
transformer blueprints or their FlexAttention/Triton dependencies.
"""

from __future__ import annotations

import importlib

# Map topology names to their blueprint modules
_TOPOLOGY_MODULES: dict[str, str] = {
    "cnn": "esper.kasmina.blueprints.cnn",
    "transformer": "esper.kasmina.blueprints.transformer",
}

_LOADED: set[str] = set()


def ensure_loaded(topology: str) -> None:
    """Load blueprint module for a topology if not already loaded.

    This is idempotent - calling multiple times has no effect after first load.
    Thread-safe due to Python's import lock and GIL.

    Args:
        topology: The topology name (e.g., "cnn", "transformer")

    Raises:
        ImportError: If the blueprint module fails to import (e.g., missing
            FlexAttention for transformer on unsupported hardware)
    """
    if topology in _LOADED:
        return

    module_path = _TOPOLOGY_MODULES.get(topology)
    if module_path is None:
        # Unknown topology - no built-in blueprints, but user-registered
        # blueprints may exist. Not an error.
        return

    try:
        importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to load {topology} blueprints from {module_path}: {e}"
        ) from e

    _LOADED.add(topology)


def ensure_all_loaded() -> None:
    """Load all blueprint modules.

    Useful for introspection, listing all available blueprints, or tests
    that need the full registry populated.
    """
    for topology in _TOPOLOGY_MODULES:
        ensure_loaded(topology)


def get_known_topologies() -> frozenset[str]:
    """Return the set of known topology names with built-in blueprints."""
    return frozenset(_TOPOLOGY_MODULES.keys())


__all__ = ["ensure_loaded", "ensure_all_loaded", "get_known_topologies"]
