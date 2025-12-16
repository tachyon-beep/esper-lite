"""Tolaria Trainer - Epoch training functions for Model Alpha.

This module provides the core training loop functions for different seed states:
- train_epoch_normal: Standard training without seed
- train_epoch_incubator_mode: STE training (seed output isolated, both train)
- train_epoch_blended: Joint host+seed training
- validate_and_get_metrics: Validation and metric computation
- validate_with_attribution: Counterfactual validation for true causal attribution

These functions are generic and work with any DataLoader, not tied to CIFAR-10.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from esper.utils.loss import compute_task_loss


@dataclass
class AttributionResult:
    """Result of counterfactual validation for seed attribution.

    Compares model performance with seed (real alpha) vs without seed (alpha=0)
    to determine the true causal contribution of the seed.
    """

    real_accuracy: float  # Accuracy with current alpha
    baseline_accuracy: float  # Accuracy with alpha=0 (host-only)
    seed_contribution: float  # real - baseline (positive = seed helps)
    real_loss: float
    baseline_loss: float


def _run_validation_pass(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    task_type: str,
) -> tuple[float, float]:
    """Run a single validation pass.

    Extracted from validate_with_attribution to enable reuse and testing.

    Args:
        model: The model to evaluate (in eval mode)
        testloader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on
        task_type: "classification" or "lm"

    Returns:
        Tuple of (average_loss, accuracy)
    """
    loss_tensor = torch.tensor(0.0, device=device)
    correct_tensor = torch.tensor(0, dtype=torch.long, device=device)
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = compute_task_loss(outputs, labels, criterion, task_type)
            loss_tensor += loss

            if task_type == "lm":
                predicted = outputs.argmax(dim=-1)
                total += labels.numel()
                correct_tensor += (predicted == labels).sum()
            else:
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct_tensor += predicted.eq(labels).sum()

    avg_loss = loss_tensor.item() / max(len(testloader), 1)
    accuracy = 100.0 * correct_tensor.item() / total if total > 0 else 0.0
    return avg_loss, accuracy


def train_epoch_normal(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    task_type: str = "classification",
    max_grad_norm: float | None = None,
) -> None:
    """Train one epoch without seed.

    Standard training loop - all model parameters updated.

    Args:
        model: The model to train.
        trainloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer for model parameters.
        device: Device to train on.
        task_type: Task type ("classification" or "lm").
        max_grad_norm: Maximum gradient norm for clipping. If None, no clipping is applied.
    """
    model.train()

    # Validate max_grad_norm parameter
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = compute_task_loss(outputs, labels, criterion, task_type)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()


def train_epoch_incubator_mode(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: nn.Module,
    host_optimizer: torch.optim.Optimizer,
    seed_optimizer: torch.optim.Optimizer,
    device: str,
    slot: str,
    task_type: str = "classification",
    gradient_telemetry_stride: int = 10,
    max_grad_norm: float | None = None,
) -> None:
    """Train one epoch with seed in isolation (seed output doesn't affect forward pass).

    During TRAINING stage (incubator mode), the seed uses a Straight-Through Estimator:
    - Forward: output = host_features (seed contribution is zero)
    - Backward: seed receives gradients as if fully blended

    CRITICAL: Both host AND seed train. The "isolation" refers to the seed's
    output not affecting the loss computation (alpha=0), NOT to freezing the host.
    Without this, on large models with frequent seed cycling, the host would
    never train.

    Args:
        model: The model to train (must have seed_slots attribute).
        trainloader: Training data loader.
        criterion: Loss function.
        host_optimizer: Optimizer for host parameters.
        seed_optimizer: Optimizer for seed parameters.
        device: Device to train on.
        slot: Which slot to train (required parameter).
        task_type: Task type (classification or lm).
        gradient_telemetry_stride: Capture gradient telemetry every N steps.
            Set to 0 to disable. Default 10 balances accuracy with GPU pipeline
            efficiency (each capture triggers a device-to-host sync).
        max_grad_norm: Maximum gradient norm for clipping. If None, no clipping is applied.
            Clips both host and seed parameters when set.
    """
    model.train()
    seed_slot = model.seed_slots[slot]

    # Validate max_grad_norm parameter
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")

    for step, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        host_optimizer.zero_grad(set_to_none=True)
        seed_optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = compute_task_loss(outputs, labels, criterion, task_type)
        loss.backward()

        # Capture gradient telemetry for G2 gate (strided to avoid GPU stalls)
        if gradient_telemetry_stride > 0 and step % gradient_telemetry_stride == 0:
            seed_slot.capture_gradient_telemetry()

        # Clip gradients for both host and seed parameters
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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
    max_grad_norm: float | None = None,
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
        task_type: Task type ("classification" or "lm").
        max_grad_norm: Maximum gradient norm for clipping. If None, no clipping is applied.
    """
    model.train()

    # Validate max_grad_norm parameter
    if max_grad_norm is not None:
        if max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {max_grad_norm}")

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        host_optimizer.zero_grad(set_to_none=True)
        if seed_optimizer:
            seed_optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = compute_task_loss(outputs, labels, criterion, task_type)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
            loss = compute_task_loss(outputs, labels, criterion, task_type)
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
            loss = compute_task_loss(outputs, labels, criterion, task_type)
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


def validate_with_attribution(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: str,
    slot: str,
    task_type: str = "classification",
) -> AttributionResult:
    """Counterfactual validation for true seed contribution measurement.

    Runs two forward passes through the validation set:
    1. Real pass: model with current alpha (seed contributing)
    2. Baseline pass: model with alpha=0 (host-only, seed output zeroed)

    The difference in accuracy between these passes gives the true causal
    attribution of the seed - how much accuracy the seed actually contributes
    vs natural host training gains.

    This addresses the "Scapegoat Problem" where seeds were blamed/credited
    for host accuracy changes during TRAINING stage when they had zero impact.

    Args:
        model: The model to evaluate (must have seed_slots attribute with force_alpha()).
        testloader: Validation data loader.
        criterion: Loss function.
        device: Device to evaluate on.
        slot: Which slot to validate (required parameter).
        task_type: Task type ("classification" or "lm").

    Returns:
        AttributionResult with real and baseline accuracies plus seed_contribution.
    """
    # Save original training mode to restore after validation
    was_training = model.training
    model.eval()

    try:
        # Pass 1: Real validation with current alpha
        real_loss, real_accuracy = _run_validation_pass(
            model, testloader, criterion, device, task_type
        )

        # Pass 2: Baseline validation with alpha=0 (host-only)
        # Use force_alpha context manager to temporarily override alpha
        seed_slot = model.seed_slots[slot]
        with seed_slot.force_alpha(0.0):
            baseline_loss, baseline_accuracy = _run_validation_pass(
                model, testloader, criterion, device, task_type
            )

        return AttributionResult(
            real_accuracy=real_accuracy,
            baseline_accuracy=baseline_accuracy,
            seed_contribution=real_accuracy - baseline_accuracy,
            real_loss=real_loss,
            baseline_loss=baseline_loss,
        )
    finally:
        # Restore original training mode
        model.train(was_training)
