"""Tests for enhanced gradient collector."""

import torch
import torch.nn as nn
import pytest

from esper.simic.gradient_collector import (
    collect_seed_gradients,
    GradientHealthMetrics,
)


class TestGradientHealthMetrics:
    """Tests for GradientHealthMetrics dataclass."""

    def test_dataclass_fields(self):
        """GradientHealthMetrics has all required fields."""
        metrics = GradientHealthMetrics(
            gradient_norm=1.0,
            gradient_health=0.9,
            has_vanishing=False,
            has_exploding=False,
            min_layer_norm=0.1,
            max_layer_norm=2.0,
            norm_ratio=20.0,
            zero_grad_fraction=0.01,
            nan_count=0,
            inf_count=0,
        )
        assert metrics.norm_ratio == 20.0

    def test_is_healthy(self):
        """is_healthy detects gradient pathologies."""
        healthy = GradientHealthMetrics(
            gradient_norm=1.0,
            gradient_health=0.9,
            has_vanishing=False,
            has_exploding=False,
            min_layer_norm=0.5,
            max_layer_norm=2.0,
            norm_ratio=4.0,
            zero_grad_fraction=0.01,
            nan_count=0,
            inf_count=0,
        )
        assert healthy.is_healthy() is True

        # NaN detected
        nan_grads = GradientHealthMetrics(
            gradient_norm=1.0,
            gradient_health=0.5,
            has_vanishing=False,
            has_exploding=False,
            min_layer_norm=0.5,
            max_layer_norm=2.0,
            norm_ratio=4.0,
            zero_grad_fraction=0.0,
            nan_count=5,
            inf_count=0,
        )
        assert nan_grads.is_healthy() is False


class TestEnhancedCollector:
    """Tests for enhanced gradient collection."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        return model

    def test_collect_returns_enhanced_metrics(self, simple_model):
        """collect_seed_gradients returns GradientHealthMetrics."""
        # Forward/backward to generate gradients
        x = torch.randn(4, 10)
        loss = simple_model(x).sum()
        loss.backward()

        result = collect_seed_gradients(
            simple_model.parameters(),
            return_enhanced=True,
        )

        assert isinstance(result, GradientHealthMetrics)
        assert result.gradient_norm > 0
        assert result.min_layer_norm > 0
        assert result.max_layer_norm >= result.min_layer_norm
