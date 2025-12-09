"""Loss computation utilities.

Consolidates loss computation patterns used across simic and tolaria.
Provides two functions with clear separation of concerns:

- compute_task_loss: Returns loss tensor only (for validation/inference)
- compute_task_loss_with_metrics: Returns loss + accuracy metrics (for training)

This module replaces duplicate implementations in:
- esper.simic.training._loss_and_correct
- esper.tolaria.trainer._compute_loss
- esper.scripts.evaluate._loss_and_correct

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
        return criterion(outputs.view(-1, vocab), targets.view(-1))
    return criterion(outputs, targets)


def compute_task_loss_with_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> tuple[torch.Tensor, float, int]:
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
        - correct_count: Number of correct predictions (float)
        - total_count: Total samples/tokens in batch (int)

    Example:
        >>> running_loss, running_correct, running_total = 0.0, 0.0, 0
        >>> for inputs, targets in dataloader:
        ...     loss, correct, total = compute_task_loss_with_metrics(
        ...         model(inputs), targets, criterion, "classification"
        ...     )
        ...     loss.backward()
        ...     running_loss += loss.item()
        ...     running_correct += correct
        ...     running_total += total
        >>> accuracy = 100.0 * running_correct / running_total
    """
    loss = compute_task_loss(outputs, targets, criterion, task_type)

    if task_type == "lm":
        predicted = outputs.argmax(dim=-1)
        correct = float((predicted == targets).sum().item())
        total = targets.numel()
    else:
        predicted = outputs.argmax(dim=1)
        correct = float((predicted == targets).sum().item())
        total = targets.size(0)

    return loss, correct, total


__all__ = [
    "compute_task_loss",
    "compute_task_loss_with_metrics",
]
