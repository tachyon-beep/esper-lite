"""Tolaria Environment - Model factory for Model Alpha.

This module provides the model factory for creating Model Alpha instances.
Dataset loading is handled separately by esper.utils.data.
"""

from __future__ import annotations

import torch

from esper.kasmina import HostCNN, MorphogeneticModel


def create_model(device: str = "cuda") -> MorphogeneticModel:
    """Create a MorphogeneticModel with HostCNN.

    Args:
        device: Target device (cuda/cpu).

    Returns:
        Initialized MorphogeneticModel on the specified device.

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device '{device}' requested but CUDA is not available. "
            f"Use device='cpu' or check your CUDA installation."
        )

    host = HostCNN(num_classes=10)
    model = MorphogeneticModel(host, device=device)
    return model.to(device)
