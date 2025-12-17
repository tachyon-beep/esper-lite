"""Tests for LM validation metrics."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def test_validate_lm_returns_perplexity():
    """validate_and_get_metrics computes perplexity for LM."""
    from esper.tolaria.trainer import validate_and_get_metrics

    model = nn.Linear(10, 100)

    x = torch.randn(32, 10)
    y = torch.randint(0, 100, (32,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8)

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc, train_loss, train_acc, per_class, perplexity = validate_and_get_metrics(
        model=model,
        trainloader=loader,
        testloader=loader,
        criterion=criterion,
        device="cpu",
        task_type="lm",
    )

    assert perplexity is not None
    assert perplexity > 0


def test_validate_classification_no_perplexity():
    """Classification tasks return None for perplexity."""
    from esper.tolaria.trainer import validate_and_get_metrics

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))

    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=8)

    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc, train_loss, train_acc, per_class, perplexity = validate_and_get_metrics(
        model=model,
        trainloader=loader,
        testloader=loader,
        criterion=criterion,
        device="cpu",
        task_type="classification",
    )

    assert perplexity is None
