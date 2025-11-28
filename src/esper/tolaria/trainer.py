"""Tolaria Trainer - Epoch training functions for Model Alpha.

This module provides the core training loop functions for different seed states:
- train_epoch_normal: Standard training without seed
- train_epoch_seed_isolated: Seed-only training (host frozen)
- train_epoch_blended: Joint host+seed training
- validate_and_get_metrics: Validation and metric computation

These functions are generic and work with any DataLoader, not tied to CIFAR-10.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch_normal(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
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
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def train_epoch_seed_isolated(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    seed_optimizer: torch.optim.Optimizer,
    device: str,
) -> None:
    """Train one epoch with seed in isolation (only seed weights updated).

    RL Context: During TRAINING stage, only seed parameters update.
    The RL agent observes seed_improvement and decides when to ADVANCE.

    Note: We don't freeze host params because that breaks gradient flow.
    Instead, we just don't step the host optimizer - gradients flow through
    but host weights stay fixed.

    Args:
        model: The model to train.
        trainloader: Training data loader.
        criterion: Loss function.
        seed_optimizer: Optimizer for seed parameters only.
        device: Device to train on.
    """
    model.train()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero ALL grads to prevent accumulation
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # Only step seed optimizer - host grads computed but not applied
        seed_optimizer.step()


def train_epoch_blended(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer | None,
    device: str,
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
        inputs, labels = inputs.to(device), labels.to(device)
        host_optimizer.zero_grad()
        if seed_optimizer:
            seed_optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
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
) -> tuple[float, float, float, float, dict[int, float] | None]:
    """Get validation and training metrics.

    Args:
        model: The model to evaluate.
        trainloader: Training data loader (for quick train metrics).
        testloader: Validation data loader.
        criterion: Loss function.
        device: Device to evaluate on.
        compute_per_class: If True, compute per-class accuracy (for telemetry).
        num_classes: Number of classes in the dataset.

    Returns:
        Tuple of (val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc)
        per_class_acc is None if compute_per_class is False.
    """
    model.eval()

    # Validation
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Per-class tracking
    if compute_per_class:
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            if compute_per_class:
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

    val_loss /= len(testloader) if len(testloader) > 0 else 1
    val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0

    per_class_acc = None
    if compute_per_class:
        per_class_acc = {
            i: 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            for i in range(num_classes)
        }

    # Training metrics (quick sample)
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(trainloader):
            if i >= 10:  # Sample first 10 batches
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

    train_loss /= min(10, len(trainloader)) if len(trainloader) > 0 else 1
    train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0

    return val_loss, val_accuracy, train_loss, train_accuracy, per_class_acc
