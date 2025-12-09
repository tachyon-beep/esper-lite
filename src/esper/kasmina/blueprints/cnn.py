"""CNN Blueprints - Seed modules for convolutional hosts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import BlueprintRegistry


def get_num_groups(channels: int, target_group_size: int = 16) -> int:
    """Select optimal num_groups for GroupNorm.

    Prefers 32 groups, falls back to smaller counts if channels isn't divisible.
    Targets at least `target_group_size` channels per group for statistical stability.
    """
    for num_groups in [32, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0 and channels // num_groups >= target_group_size:
            return num_groups
    # Fallback: just find a divisor
    for num_groups in [32, 16, 8, 4, 2, 1]:
        if channels % num_groups == 0:
            return num_groups
    return 1


class ConvBlock(nn.Module):
    """Standard conv-bn-relu block.

    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        kernel_size: Convolution kernel size (default 3)
        track_running_stats: Whether BatchNorm tracks running mean/var (default True).
            Set False for DTensor/FSDP2 distributed training compatibility.
            When False, uses batch statistics in both train and eval modes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, track_running_stats=track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class SeedConvBlock(nn.Module):
    """Conv-groupnorm-relu block for seeds (no running stats drift)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.gn = nn.GroupNorm(get_num_groups(out_channels), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gn(self.conv(x)))


@BlueprintRegistry.register("norm", "cnn", param_estimate=100, description="GroupNorm enhancement")
def create_norm_seed(channels: int) -> nn.Module:
    """Normalization enhancement seed."""

    class NormSeed(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Bound scale to [-1, 1] via tanh to prevent gradient explosion
            return x + torch.tanh(self.scale) * (self.norm(x) - x)

    return NormSeed(channels)


@BlueprintRegistry.register(
    "attention", "cnn", param_estimate=2000, description="SE-style channel attention"
)
def create_attention_seed(channels: int, reduction: int = 4) -> nn.Module:
    """Channel attention seed using Squeeze-and-Excitation (SE) pattern.

    This implements the canonical SE block from Hu et al. 2018 ("Squeeze-and-Excitation
    Networks"). Unlike additive residual seeds (norm, depthwise, conv_*), SE uses
    multiplicative channel-wise gating: output = x * sigmoid(fc(pool(x))).

    Design notes:
    - Multiplicative gating is intentional, not a bug. SE learns to recalibrate
      channel importance rather than adding learned features.
    - Sigmoid outputs are in [0, 1], theoretically allowing channels to be zeroed.
      In practice, standard initialization keeps outputs near 0.5 (neutral scaling).
    - No residual connection needed: the multiplication preserves spatial structure
      and gradient flow through x. Unlike additive seeds, SE modulates existing
      features rather than injecting new ones.

    Reference: https://arxiv.org/abs/1709.01507
    """

    class AttentionSeed(nn.Module):
        def __init__(self, channels: int, reduction: int):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            # Ensure reduced dimension is at least 1
            reduced = max(1, channels // reduction)
            self.fc = nn.Sequential(
                nn.Linear(channels, reduced, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(reduced, channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Squeeze: global average pooling to channel descriptor
            b, c, _, _ = x.size()
            y = self.avg_pool(x).reshape(b, c)
            # Excitation: channel-wise attention weights via FC bottleneck
            y = self.fc(y).reshape(b, c, 1, 1)
            # Scale: multiplicative recalibration (canonical SE formulation)
            return x * y.expand_as(x)

    return AttentionSeed(channels, reduction)


@BlueprintRegistry.register("depthwise", "cnn", param_estimate=4800, description="Depthwise-separable conv")
def create_depthwise_seed(channels: int) -> nn.Module:
    """Depthwise separable convolution seed."""

    class DepthwiseSeed(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.depthwise = nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
            )
            self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.gn = nn.GroupNorm(get_num_groups(channels), channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.depthwise(x)
            x = self.pointwise(x)
            x = self.gn(x)
            return residual + F.relu(x)

    return DepthwiseSeed(channels)


@BlueprintRegistry.register("conv_light", "cnn", param_estimate=37000, description="Light conv block")
def create_conv_light_seed(channels: int) -> nn.Module:
    """Single convolution enhancement seed - lighter alternative to conv_heavy."""

    class ConvLightSeed(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.enhance = SeedConvBlock(channels, channels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.enhance(x)

    return ConvLightSeed(channels)


@BlueprintRegistry.register("conv_heavy", "cnn", param_estimate=74000, description="Heavy conv block")
def create_conv_heavy_seed(channels: int) -> nn.Module:
    """Double convolution enhancement seed - heavier but potentially more powerful."""

    class ConvHeavySeed(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.enhance = nn.Sequential(
                SeedConvBlock(channels, channels),
                SeedConvBlock(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.enhance(x)

    return ConvHeavySeed(channels)


__all__ = ["ConvBlock", "SeedConvBlock", "get_num_groups"]
