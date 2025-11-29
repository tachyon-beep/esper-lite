"""Data loading utilities.

Dataset loaders for training Model Alpha. Currently supports CIFAR-10,
with room to grow for ImageNet, synthetic datasets, etc.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def load_cifar10(
    batch_size: int = 128,
    generator: torch.Generator | None = None,
    data_root: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataset.

    Args:
        batch_size: Batch size for DataLoaders.
        generator: Optional torch.Generator for reproducible shuffling.
            Use different generators per environment to avoid GIL contention
            when multiple CUDA streams iterate shared DataLoaders.
        data_root: Root directory for dataset storage.

    Returns:
        Tuple of (trainloader, testloader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading workers
        pin_memory=True,  # Required for non_blocking=True to work
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Queue 4 batches per worker ahead of time
        generator=generator,
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Parallel data loading workers
        pin_memory=True,  # Required for non_blocking=True to work
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Queue 4 batches per worker ahead of time
    )

    return trainloader, testloader
