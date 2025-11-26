"""Architecture factory for diverse model creation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_ARCHITECTURES = [
    "HostCNN",
    "HostCNN-Wide",
    "HostCNN-Deep",
    "ResNet-18",
    "ResNet-34",
]


def create_model(architecture: str, num_classes: int = 10) -> nn.Module:
    """Create a model by architecture name."""
    if architecture == "HostCNN":
        return HostCNN(num_classes=num_classes)
    elif architecture == "HostCNN-Wide":
        return HostCNNWide(num_classes=num_classes)
    elif architecture == "HostCNN-Deep":
        return HostCNNDeep(num_classes=num_classes)
    elif architecture == "ResNet-18":
        return ResNetCIFAR(num_classes=num_classes, depth=18)
    elif architecture == "ResNet-34":
        return ResNetCIFAR(num_classes=num_classes, depth=34)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Supported: {SUPPORTED_ARCHITECTURES}")


# =============================================================================
# CNN Variants
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


class HostCNN(nn.Module):
    """Standard 3-block CNN for CIFAR."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class HostCNNWide(nn.Module):
    """Wide CNN with 2x channels."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.MaxPool2d(2),
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class HostCNNDeep(nn.Module):
    """Deep CNN with 5 blocks."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# =============================================================================
# ResNet for CIFAR (smaller input size)
# =============================================================================

class BasicBlock(nn.Module):
    """Basic ResNet block."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNetCIFAR(nn.Module):
    """ResNet adapted for CIFAR (32x32 input)."""

    def __init__(self, num_classes: int = 10, depth: int = 18):
        super().__init__()

        if depth == 18:
            num_blocks = [2, 2, 2, 2]
        elif depth == 34:
            num_blocks = [3, 4, 6, 3]
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        self.in_planes = 64

        # CIFAR: smaller first conv, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
