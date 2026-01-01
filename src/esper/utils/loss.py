"""Loss computation utilities.

Consolidates loss computation patterns used across simic and tolaria.
Provides two functions with clear separation of concerns:

- compute_task_loss: Returns loss tensor only (for validation/inference)
- compute_task_loss_with_metrics: Returns loss + accuracy metrics (for training)

This module replaces duplicate loss/metric patterns that previously lived
inline in Simic/Tolaria training loops.

Usage:
    from esper.utils.loss import compute_task_loss, compute_task_loss_with_metrics

    # Validation (loss only)
    loss = compute_task_loss(outputs, targets, criterion, "classification")

    # Training (loss + metrics for logging)
    loss, correct, total = compute_task_loss_with_metrics(
        outputs, targets, criterion, "classification"
    )
    accuracy = 100.0 * correct / total
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_task_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> torch.Tensor:
    """Compute loss for classification or language modeling task.

    Use when you only need the loss value (e.g., validation, inference).

    Args:
        outputs: Model outputs
            - Classification: (batch, num_classes)
            - Language modeling: (batch, seq_len, vocab_size)
        targets: Ground truth labels
            - Classification: (batch,) with class indices
            - Language modeling: (batch, seq_len) with token indices
        criterion: Loss function (e.g., nn.CrossEntropyLoss)
        task_type: "classification" or "lm"

    Returns:
        Loss tensor (scalar, differentiable)

    Example:
        >>> outputs = model(inputs)
        >>> loss = compute_task_loss(outputs, targets, nn.CrossEntropyLoss(), "classification")
        >>> loss.backward()
    """
    if task_type == "lm":
        vocab = outputs.size(-1)
        # Use reshape instead of view - handles non-contiguous tensors safely
        return criterion(outputs.reshape(-1, vocab), targets.reshape(-1))  # type: ignore[no-any-return]
    return criterion(outputs, targets)  # type: ignore[no-any-return]


def compute_task_loss_with_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute loss and accuracy metrics for classification or LM task.

    Use when you need both loss and accuracy (e.g., training loops with logging).

    Args:
        outputs: Model outputs
            - Classification: (batch, num_classes)
            - Language modeling: (batch, seq_len, vocab_size)
        targets: Ground truth labels
            - Classification: (batch,) with class indices
            - Language modeling: (batch, seq_len) with token indices
        criterion: Loss function (e.g., nn.CrossEntropyLoss)
        task_type: "classification" or "lm"

    Returns:
        Tuple of (loss, correct_count, total_count):
        - loss: CrossEntropyLoss tensor (differentiable, for backprop)
        - correct_count: Tensor on same device (call .item() after stream sync)
        - total_count: Total samples/tokens in batch (int)

    Example:
        Hot path - tensor accumulation, no sync:
        >>> running_loss = torch.zeros(1, device=device)
        >>> running_correct = torch.zeros(1, device=device, dtype=torch.long)
        >>> running_total = 0
        >>> for inputs, targets in dataloader:
        ...     loss, correct, total = compute_task_loss_with_metrics(
        ...         model(inputs), targets, criterion, "classification"
        ...     )
        ...     loss.backward()
        ...     running_loss.add_(loss.detach())
        ...     running_correct.add_(correct)
        ...     running_total += total
        >>> # Single sync at end of epoch
        >>> accuracy = 100.0 * running_correct.item() / running_total
    """
    loss = compute_task_loss(outputs, targets, criterion, task_type)

    with torch.no_grad():
        if task_type == "lm":
            # Fused eq().sum() - stays on device, no .item() sync
            correct = outputs.argmax(dim=-1).eq(targets).sum()
            total = targets.numel()
        else:
            # Fused eq().sum() - stays on device, no .item() sync
            correct = outputs.argmax(dim=1).eq(targets).sum()
            total = targets.size(0)

    return loss, correct, total


__all__ = [
    "compute_task_loss",
    "compute_task_loss_with_metrics",
]
