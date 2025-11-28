"""Kasmina Blueprints - Seed module implementations.

Blueprints define the architecture of injectable seed modules.
Each blueprint creates a specific type of enhancement:
- ConvEnhance: Additional convolutional capacity
- Attention: Self-attention mechanism (SE-style)
- Norm: Normalization layers
- Depthwise: Depthwise separable convolutions
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBlock(nn.Module):
    """Standard conv-bn-relu block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


# =============================================================================
# Blueprint Implementations
# =============================================================================

class ConvEnhanceSeed(nn.Module):
    """Convolutional enhancement seed."""

    blueprint_id = "conv_enhance"

    def __init__(self, channels: int):
        super().__init__()
        self.enhance = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.enhance(x)


class AttentionSeed(nn.Module):
    """Channel attention seed (SE-style)."""

    blueprint_id = "attention"

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class NormSeed(nn.Module):
    """Normalization enhancement seed."""

    blueprint_id = "norm"

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * (self.norm(x) - x)


class DepthwiseSeed(nn.Module):
    """Depthwise separable convolution seed."""

    blueprint_id = "depthwise"

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return residual + F.relu(x)


# =============================================================================
# Blueprint Catalog
# =============================================================================

class BlueprintCatalog:
    """Registry of available seed blueprints."""

    _blueprints: dict[str, type[nn.Module]] = {
        "conv_enhance": ConvEnhanceSeed,
        "attention": AttentionSeed,
        "norm": NormSeed,
        "depthwise": DepthwiseSeed,
    }

    @classmethod
    def list_blueprints(cls) -> list[str]:
        """List available blueprint IDs."""
        return list(cls._blueprints.keys())

    @classmethod
    def create_seed(cls, blueprint_id: str, channels: int, **kwargs) -> nn.Module:
        """Create a seed module from a blueprint."""
        if blueprint_id not in cls._blueprints:
            raise ValueError(f"Unknown blueprint: {blueprint_id}. "
                           f"Available: {cls.list_blueprints()}")
        return cls._blueprints[blueprint_id](channels, **kwargs)

    @classmethod
    def register_blueprint(cls, blueprint_id: str, blueprint_class: type[nn.Module]) -> None:
        """Register a new blueprint type."""
        cls._blueprints[blueprint_id] = blueprint_class


__all__ = [
    "ConvBlock",
    "ConvEnhanceSeed",
    "AttentionSeed",
    "NormSeed",
    "DepthwiseSeed",
    "BlueprintCatalog",
]
