"""Kasmina Host - The graftable host network.

The MorphogeneticModel is the host network that accepts seed grafts.
It manages the injection points where seeds can be attached.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from esper.leyline import SeedStage
from esper.kasmina.slot import SeedSlot
from esper.kasmina.isolation import GradientIsolationMonitor
from esper.kasmina.blueprints import ConvBlock  # Reuse shared building block


class HostCNN(nn.Module):
    """Host CNN with injection point."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Linear(128, num_classes)

        # Injection point info
        self.injection_channels = 64  # After block2

    def forward_to_injection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        return x

    def forward_from_injection(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.block3(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_to_injection(x)
        return self.forward_from_injection(x)


# =============================================================================
# Morphogenetic Model
# =============================================================================

class MorphogeneticModel(nn.Module):
    """Model with Kasmina seed slot."""

    def __init__(self, host: HostCNN, device: str = "cpu"):
        super().__init__()
        self.host = host
        self._device = device

        # Single seed slot at injection point
        self.seed_slot = SeedSlot(
            slot_id="injection_point",
            channels=host.injection_channels,
            device=device,
        )

        # Isolation monitor
        self.isolation_monitor = GradientIsolationMonitor()

    def to(self, *args, **kwargs):
        """Override to() to propagate device change to SeedSlot.

        SeedSlot is not an nn.Module, so PyTorch's recursive to() doesn't
        reach it. We manually propagate the device change to ensure the
        slot creates new seeds on the correct device.
        """
        result = super().to(*args, **kwargs)

        # Determine the new device from model parameters
        try:
            new_device = next(self.parameters()).device
        except StopIteration:
            return result  # No parameters, nothing to update

        # Propagate to seed slot
        self.seed_slot.device = new_device
        self._device = str(new_device)

        # Move existing seed if present
        if self.seed_slot.seed is not None:
            self.seed_slot.seed = self.seed_slot.seed.to(new_device)

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.host.forward_to_injection(x)
        features = self.seed_slot.forward(features)
        return self.host.forward_from_injection(features)

    def germinate_seed(self, blueprint_id: str, seed_id: str) -> None:
        """Germinate a new seed."""
        state = self.seed_slot.germinate(
            blueprint_id=blueprint_id,
            seed_id=seed_id,
            host_module=self.host,
        )
        print(f"    [Kasmina] Germinated seed '{seed_id}' with blueprint '{blueprint_id}'")

    def cull_seed(self) -> None:
        """Cull the current seed."""
        if self.seed_slot.state:
            print(f"    [Kasmina] Culling seed '{self.seed_slot.state.seed_id}'")
        self.seed_slot.cull()

    def get_seed_parameters(self):
        return self.seed_slot.get_parameters()

    def get_host_parameters(self):
        return self.host.parameters()

    @property
    def has_active_seed(self) -> bool:
        return self.seed_slot.is_active

    @property
    def seed_state(self):
        return self.seed_slot.state


__all__ = [
    "ConvBlock",
    "HostCNN",
    "MorphogeneticModel",
]
