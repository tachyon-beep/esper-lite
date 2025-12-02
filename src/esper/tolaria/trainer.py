"""Tolaria Trainer - Epoch training functions for Model Alpha.

This module provides the core training loop functions for different seed states:
- train_epoch_normal: Standard training without seed
- train_epoch_womb_mode: STE training (seed output isolated, both train)
- train_epoch_blended: Joint host+seed training
- validate_and_get_metrics: Validation and metric computation

These functions are generic and work with any DataLoader, not tied to CIFAR-10.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _compute_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    task_type: str,
) -> torch.Tensor:
    """Compute loss for classification or language modeling."""
    if task_type == "lm":
        vocab = outputs.size(-1)
        return criterion(outputs.view(-1, vocab), labels.view(-1))
    return criterion(outputs, labels)


def train_epoch_normal(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    task_type: str = "classification",
) -> None:
    """Train one epoch without seed.

    Standard training loop - all model parameters updated.

    Args:
        model: The model to train.
        trainloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer for model parameters.
        device: Device to train on.
    """
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = _compute_loss(outputs, labels, criterion, task_type)
        loss.backward()
        optimizer.step()


def train_epoch_womb_mode(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer,
    device: str,
    task_type: str = "classification",
) -> None:
    """Train one epoch with seed in isolation (seed output doesn't affect forward pass).

    During TRAINING stage (womb mode), the seed uses a Straight-Through Estimator:
    - Forward: output = host_features (seed contribution is zero)
    - Backward: seed receives gradients as if fully blended

    CRITICAL: Both host AND seed train. The "isolation" refers to the seed's
    output not affecting the loss computation (alpha=0), NOT to freezing the host.
    Without this, on large models with frequent seed cycling, the host would
    never train.

    Args:
        model: The model to train.
        trainloader: Training data loader.
        criterion: Loss function.
        host_optimizer: Optimizer for host parameters.
        seed_optimizer: Optimizer for seed parameters.
        device: Device to train on.
        task_type: Task type (classification or lm).
    """
    model.train()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        host_optimizer.zero_grad(set_to_none=True)
        seed_optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = _compute_loss(outputs, labels, criterion, task_type)
        loss.backward()
        host_optimizer.step()
        seed_optimizer.step()


def train_epoch_blended(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer | None,
    device: str,
    task_type: str = "classification",
) -> None:
    """Train one epoch with blended host+seed.

    Used during BLENDING and FOSSILIZED stages when both host and seed
    parameters are updated together.

    Args:
        model: The model to train.
        trainloader: Training data loader.
        criterion: Loss function.
        host_optimizer: Optimizer for host parameters.
        seed_optimizer: Optimizer for seed parameters (optional).
        device: Device to train on.
    """
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        host_optimizer.zero_grad(set_to_none=True)
        if seed_optimizer:
            seed_optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = _compute_loss(outputs, labels, criterion, task_type)
        loss.backward()
        host_optimizer.step()
        if seed_optimizer:
            seed_optimizer.step()


def validate_and_get_metrics(
    model: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    compute_per_class: bool = False,
    num_classes: int = 10,
    task_type: str = "classification",
) -> tuple[float, float, float, float, dict[int, float] | None, float | None]:
    """Get validation and training metrics.

    Args:
        model: The model to evaluate.
        trainloader: Training data loader (for quick train metrics).
        testloader: Validation data loader.
        criterion: Loss function.
        device: Device to evaluate on.
        compute_per_class: If True, compute per-class accuracy (for telemetry).
        num_classes: Number of classes in the dataset.
        task_type: Task type ("classification" or "lm").

    Returns:
        Tuple of (val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc, perplexity)
        per_class_acc is None if compute_per_class is False.
        perplexity is only computed for language modeling tasks.
    """
    model.eval()

    # Validation - accumulate on GPU to avoid CPU-GPU sync per batch
    val_loss_tensor = torch.tensor(0.0, device=device)
    val_correct_tensor = torch.tensor(0, dtype=torch.long, device=device)
    val_total = 0

    # Per-class tracking (vectorized)
    if compute_per_class:
        class_correct_tensor = torch.zeros(num_classes, dtype=torch.long, device=device)
        class_total_tensor = torch.zeros(num_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = _compute_loss(outputs, labels, criterion, task_type)
            val_loss_tensor += loss
            if task_type == "lm":
                predicted = outputs.argmax(dim=-1)
                val_total += labels.numel()
                val_correct_tensor += (predicted == labels).sum()
            else:
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct_tensor += predicted.eq(labels).sum()

                if compute_per_class:
                    # Vectorized per-class counting using bincount
                    # Note: bincount handles empty inputs gracefully (returns zeros)
                    class_total_tensor += torch.bincount(labels, minlength=num_classes)
                    correct_mask = predicted == labels
                    class_correct_tensor += torch.bincount(
                        labels[correct_mask], minlength=num_classes
                    )

    # Single CPU sync point at end of validation
    val_loss = val_loss_tensor.item() / max(len(testloader), 1)
    val_correct = val_correct_tensor.item()
    val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0

    per_class_acc = None
    if compute_per_class:
        # Single sync for per-class stats
        class_correct = class_correct_tensor.cpu()
        class_total = class_total_tensor.cpu()
        per_class_acc = {
            i: 100.0 * class_correct[i].item() / class_total[i].item()
            if class_total[i].item() > 0 else 0.0
            for i in range(num_classes)
        }

    # Training metrics (quick sample) - use islice to avoid graph-breaking if-break
    train_loss_tensor = torch.tensor(0.0, device=device)
    train_correct_tensor = torch.tensor(0, dtype=torch.long, device=device)
    train_total = 0
    train_batches = 0

    with torch.no_grad():
        for inputs, labels in itertools.islice(trainloader, 10):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = _compute_loss(outputs, labels, criterion, task_type)
            train_loss_tensor += loss
            train_batches += 1
            if task_type == "lm":
                predicted = outputs.argmax(dim=-1)
                train_total += labels.numel()
                train_correct_tensor += (predicted == labels).sum()
            else:
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct_tensor += predicted.eq(labels).sum()

    # Single CPU sync point at end of training metrics
    train_loss = train_loss_tensor.item() / max(train_batches, 1)
    train_correct = train_correct_tensor.item()
    train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0

    perplexity = None
    if task_type == "lm":
        perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")

    return val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc, perplexity
