"""Tests for gradient collector performance and correctness."""
import pytest
import torch
import torch.nn as nn

from esper.simic.gradient_collector import (
    SeedGradientCollector,
    materialize_grad_stats,
)


def test_gradient_collector_vectorized():
    """Verify gradient collection uses vectorized operations."""
    # Create simple model with gradients
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )

    # Forward/backward to create gradients
    x = torch.randn(4, 10)
    y = model(x).sum()
    y.backward()

    collector = SeedGradientCollector()
    async_stats = collector.collect_async(model.parameters())
    stats = materialize_grad_stats(async_stats)

    # Basic correctness checks
    assert 'gradient_norm' in stats
    assert 'gradient_health' in stats
    assert stats['gradient_norm'] > 0
    assert 0 <= stats['gradient_health'] <= 1


def test_gradient_collector_empty():
    """Verify handling of parameters without gradients."""
    model = nn.Linear(10, 5)  # No backward called

    collector = SeedGradientCollector()
    stats = collector.collect(model.parameters())

    assert stats['gradient_norm'] == 0.0
    assert stats['gradient_health'] == 1.0
    assert stats['has_vanishing'] is False
    assert stats['has_exploding'] is False
