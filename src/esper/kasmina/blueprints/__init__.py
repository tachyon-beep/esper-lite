"""Kasmina Blueprints - Plugin registry for seed architectures.

Blueprint modules are loaded lazily on first access to avoid importing
heavy dependencies (like FlexAttention) when they're not needed. The
registry self-populates when you call list_for_topology() or get().

Example:
    # This doesn't import transformer blueprints
    from esper.kasmina.blueprints import BlueprintRegistry

    # This loads only CNN blueprints
    cnn_blueprints = BlueprintRegistry.list_for_topology("cnn")

    # ConvBlock is available via lazy attribute access
    from esper.kasmina.blueprints import ConvBlock
"""

from __future__ import annotations

from .registry import BlueprintFactory, BlueprintSpec, BlueprintRegistry

__all__ = [
    "BlueprintFactory",
    "BlueprintSpec",
    "BlueprintRegistry",
    "ConvBlock",
]


def __getattr__(name: str) -> object:
    """Lazy attribute access for heavy imports (PEP 562)."""
    if name == "ConvBlock":
        from .cnn import ConvBlock

        return ConvBlock
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
