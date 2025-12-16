"""Tests for gradient collector performance and correctness."""

import torch
import torch.nn as nn
import pytest

from esper.simic.gradient_collector import (
    SeedGradientCollector,
    materialize_grad_stats,
    collect_seed_gradients,
    collect_dual_gradients_async,
    materialize_dual_grad_stats,
    GradientHealthMetrics,
    DualGradientStats,
)


# =============================================================================
# Basic Gradient Collection Tests
# =============================================================================


class TestSeedGradientCollector:
    """Tests for basic SeedGradientCollector functionality."""

    def test_gradient_collector_vectorized(self):
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

    def test_gradient_collector_empty(self):
        """Verify handling of parameters without gradients."""
        model = nn.Linear(10, 5)  # No backward called

        collector = SeedGradientCollector()
        stats = collector.collect(model.parameters())

        assert stats['gradient_norm'] == 0.0
        assert stats['gradient_health'] == 1.0
        assert stats['has_vanishing'] is False
        assert stats['has_exploding'] is False


# =============================================================================
# Enhanced Gradient Metrics Tests
# =============================================================================


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


# =============================================================================
# Dual Gradient Collection Tests (Host + Seed)
# =============================================================================


class TestDualGradientCollection:
    """Tests for dual gradient collection (host + seed)."""

    @pytest.fixture
    def host_model(self):
        """Create host model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

    @pytest.fixture
    def seed_model(self):
        """Create seed model for testing."""
        return nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
        )

    def test_collect_returns_correct_keys(self, host_model, seed_model):
        """collect_dual_gradients_async returns dict with expected keys."""
        # Generate gradients
        x = torch.randn(4, 10)
        loss = host_model(x).sum()
        loss.backward()

        y = torch.randn(4, 5)
        seed_loss = seed_model(y).sum()
        seed_loss.backward()

        result = collect_dual_gradients_async(
            host_model.parameters(),
            seed_model.parameters(),
        )

        assert "_host_squared_sum" in result
        assert "_host_param_count" in result
        assert "_seed_squared_sum" in result
        assert "_seed_param_count" in result

    def test_returns_tensors_for_squared_sums(self, host_model, seed_model):
        """Squared sums are tensors (for async safety)."""
        x = torch.randn(4, 10)
        loss = host_model(x).sum()
        loss.backward()

        y = torch.randn(4, 5)
        seed_loss = seed_model(y).sum()
        seed_loss.backward()

        result = collect_dual_gradients_async(
            host_model.parameters(),
            seed_model.parameters(),
        )

        # Squared sums should be tensors (not floats) when grads exist
        assert isinstance(result["_host_squared_sum"], torch.Tensor)
        assert isinstance(result["_seed_squared_sum"], torch.Tensor)
        # Param counts are always ints
        assert isinstance(result["_host_param_count"], int)
        assert isinstance(result["_seed_param_count"], int)

    def test_handles_empty_parameters(self):
        """Handles empty parameter lists gracefully."""
        result = collect_dual_gradients_async(iter([]), iter([]))

        assert result["_host_squared_sum"] == 0.0
        assert result["_host_param_count"] == 0
        assert result["_seed_squared_sum"] == 0.0
        assert result["_seed_param_count"] == 0

    def test_materialize_produces_dual_stats(self, host_model, seed_model):
        """materialize_dual_grad_stats returns DualGradientStats."""
        x = torch.randn(4, 10)
        loss = host_model(x).sum()
        loss.backward()

        y = torch.randn(4, 5)
        seed_loss = seed_model(y).sum()
        seed_loss.backward()

        async_stats = collect_dual_gradients_async(
            host_model.parameters(),
            seed_model.parameters(),
        )
        dual_stats = materialize_dual_grad_stats(async_stats)

        assert isinstance(dual_stats, DualGradientStats)
        assert dual_stats.host_grad_norm > 0
        assert dual_stats.seed_grad_norm > 0
        assert dual_stats.host_param_count > 0
        assert dual_stats.seed_param_count > 0

    def test_normalized_ratio_computation(self, host_model, seed_model):
        """normalized_ratio computes parameter-normalized ratio."""
        x = torch.randn(4, 10)
        loss = host_model(x).sum()
        loss.backward()

        y = torch.randn(4, 5)
        seed_loss = seed_model(y).sum()
        seed_loss.backward()

        async_stats = collect_dual_gradients_async(
            host_model.parameters(),
            seed_model.parameters(),
        )
        dual_stats = materialize_dual_grad_stats(async_stats)

        # Ratio should be non-negative
        assert dual_stats.normalized_ratio >= 0.0
