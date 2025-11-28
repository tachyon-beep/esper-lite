"""Esper Utils - Shared utilities (the bit bucket).

This package contains shared utilities that don't belong to any specific
domain subsystem:

- data: Dataset loading utilities (CIFAR-10, future datasets)
"""

from esper.utils.data import load_cifar10

__all__ = [
    "load_cifar10",
]
