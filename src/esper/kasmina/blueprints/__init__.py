"""Kasmina Blueprints - Plugin registry for seed architectures."""

from __future__ import annotations

from .registry import BlueprintFactory, BlueprintSpec, BlueprintRegistry

# Import blueprints to trigger registration
from . import cnn  # noqa: F401
from . import transformer  # noqa: F401

# Re-export CNN building block for convenience
from .cnn import ConvBlock

__all__ = [
    "BlueprintFactory",
    "BlueprintSpec",
    "BlueprintRegistry",
    "ConvBlock",
]
