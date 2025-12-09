"""Esper Utils - Shared utilities (the bit bucket).

This package contains shared utilities that don't belong to any specific
domain subsystem:

- data: Dataset loading utilities (CIFAR-10, future datasets)
- loss: Consolidated loss computation utilities
"""

from esper.utils.data import load_cifar10
from esper.utils.loss import compute_task_loss, compute_task_loss_with_metrics

__all__ = [
    "load_cifar10",
    "compute_task_loss",
    "compute_task_loss_with_metrics",
]
