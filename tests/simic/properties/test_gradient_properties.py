"""Property-based tests for gradient statistics collection.

Tests invariants and mathematical properties of SeedGradientCollector
using Hypothesis to generate hundreds of test cases automatically.

These tests verify that gradient statistics behave correctly across
a wide range of parameter configurations and gradient values.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hypothesis import given, strategies as st

from esper.simic.telemetry import SeedGradientCollector
from tests.strategies import bounded_floats


# =============================================================================
# Test Gradient Norm Properties
# =============================================================================


class TestGradientNormProperties:
    """Test mathematical properties of gradient norm computation.

    The gradient norm should satisfy:
    1. Non-negativity: norm >= 0 always
    2. Zero input produces zero norm
    3. Norm scales with gradient magnitude
    """

    @given(
        num_params=st.integers(min_value=1, max_value=10),
        param_size=st.integers(min_value=1, max_value=100),
    )
    def test_gradient_norm_non_negative(self, num_params: int, param_size: int):
        """Property: Gradient norm is always non-negative.

        For any set of parameters with gradients, the computed norm
        must be >= 0.
        """
        collector = SeedGradientCollector()

        # Create parameters with random gradients
        params = []
        for _ in range(num_params):
            param = nn.Parameter(torch.randn(param_size))
            param.grad = torch.randn(param_size)
            params.append(param)

        stats = collector.collect(iter(params))

        assert stats['gradient_norm'] >= 0.0, \
            f"Gradient norm must be non-negative, got {stats['gradient_norm']}"

    @given(
        num_params=st.integers(min_value=1, max_value=10),
        param_size=st.integers(min_value=1, max_value=100),
    )
    def test_zero_gradients_give_zero_norm(self, num_params: int, param_size: int):
        """Property: Zero gradients produce zero norm.

        If all gradients are exactly zero, the norm must be exactly zero.
        """
        collector = SeedGradientCollector()

        # Create parameters with zero gradients
        params = []
        for _ in range(num_params):
            param = nn.Parameter(torch.randn(param_size))
            param.grad = torch.zeros(param_size)
            params.append(param)

        stats = collector.collect(iter(params))

        assert stats['gradient_norm'] == 0.0, \
            f"Zero gradients must produce zero norm, got {stats['gradient_norm']}"

    def test_empty_parameters_give_zero_norm(self):
        """Property: Empty parameter list produces zero norm.

        When there are no parameters, norm should be 0.
        """
        collector = SeedGradientCollector()

        stats = collector.collect(iter([]))

        assert stats['gradient_norm'] == 0.0
        assert stats['gradient_health'] == 1.0
        assert stats['has_vanishing'] is False
        assert stats['has_exploding'] is False

    @given(
        num_params=st.integers(min_value=1, max_value=10),
        param_size=st.integers(min_value=1, max_value=100),
    )
    def test_none_gradients_ignored(self, num_params: int, param_size: int):
        """Property: Parameters without gradients are ignored.

        If some parameters have None gradients, they should not affect
        the computation.
        """
        collector = SeedGradientCollector()

        # Create mix of params with and without gradients
        params = []
        for i in range(num_params):
            param = nn.Parameter(torch.randn(param_size))
            # Only set gradients for half the parameters
            if i % 2 == 0:
                param.grad = torch.randn(param_size)
            # Leave param.grad as None for the rest
            params.append(param)

        # Should not crash and should compute from non-None grads only
        stats = collector.collect(iter(params))

        # If all grads were None, norm is 0
        has_any_grad = any(p.grad is not None for p in params)
        if not has_any_grad:
            assert stats['gradient_norm'] == 0.0
        else:
            # Otherwise, norm should be computed from available grads
            assert stats['gradient_norm'] >= 0.0


# =============================================================================
# Test Gradient Health Properties
# =============================================================================


class TestGradientHealthProperties:
    """Test gradient health score computation.

    The health score should satisfy:
    1. Bounded in [0, 1]
    2. Perfect health (1.0) when no vanishing/exploding
    3. Degraded health when vanishing/exploding detected
    """

    @given(
        num_params=st.integers(min_value=1, max_value=20),
        param_size=st.integers(min_value=1, max_value=100),
        vanishing_threshold=bounded_floats(1e-10, 1e-5),
        exploding_threshold=bounded_floats(10.0, 1000.0),
    )
    def test_gradient_health_bounded(
        self,
        num_params: int,
        param_size: int,
        vanishing_threshold: float,
        exploding_threshold: float,
    ):
        """Property: Health score is always in [0, 1].

        Regardless of gradient values or thresholds, health must be
        bounded between 0 and 1.
        """
        collector = SeedGradientCollector(
            vanishing_threshold=vanishing_threshold,
            exploding_threshold=exploding_threshold,
        )

        # Create parameters with random gradients
        params = []
        for _ in range(num_params):
            param = nn.Parameter(torch.randn(param_size))
            param.grad = torch.randn(param_size)
            params.append(param)

        stats = collector.collect(iter(params))

        assert 0.0 <= stats['gradient_health'] <= 1.0, \
            f"Health must be in [0, 1], got {stats['gradient_health']}"

    @given(
        num_params=st.integers(min_value=1, max_value=10),
        param_size=st.integers(min_value=1, max_value=100),
        grad_value=bounded_floats(1e-5, 10.0),  # Safe range
    )
    def test_perfect_health_when_no_issues(
        self,
        num_params: int,
        param_size: int,
        grad_value: float,
    ):
        """Property: Health is 1.0 when gradients are in healthy range.

        If all gradient norms are between vanishing and exploding thresholds,
        health should be perfect (1.0).
        """
        collector = SeedGradientCollector(
            vanishing_threshold=1e-7,
            exploding_threshold=100.0,
        )

        # Create parameters with gradients in healthy range
        params = []
        for _ in range(num_params):
            param = nn.Parameter(torch.randn(param_size))
            # Set gradient with controlled magnitude
            param.grad = torch.full((param_size,), grad_value)
            params.append(param)

        stats = collector.collect(iter(params))

        assert stats['gradient_health'] == 1.0, \
            f"Health should be 1.0 for healthy gradients, got {stats['gradient_health']}"
        assert stats['has_vanishing'] is False
        assert stats['has_exploding'] is False

    @given(
        num_params=st.integers(min_value=1, max_value=10),
        param_size=st.integers(min_value=1, max_value=100),
    )
    def test_degraded_health_with_vanishing(
        self,
        num_params: int,
        param_size: int,
    ):
        """Property: Health < 1.0 when vanishing gradients detected.

        If some parameters have vanishing gradients (norm < threshold),
        health score should be degraded.
        """
        vanishing_threshold = 1e-7
        collector = SeedGradientCollector(vanishing_threshold=vanishing_threshold)

        # Create parameters with vanishing gradients
        params = []
        for _ in range(num_params):
            param = nn.Parameter(torch.randn(param_size))
            # Set gradient below vanishing threshold
            param.grad = torch.full((param_size,), vanishing_threshold / 10.0)
            params.append(param)

        stats = collector.collect(iter(params))

        assert stats['has_vanishing'] is True, \
            "Should detect vanishing gradients"
        assert stats['gradient_health'] < 1.0, \
            f"Health should be degraded with vanishing gradients, got {stats['gradient_health']}"

    @given(
        num_params=st.integers(min_value=1, max_value=10),
        param_size=st.integers(min_value=1, max_value=100),
    )
    def test_degraded_health_with_exploding(
        self,
        num_params: int,
        param_size: int,
    ):
        """Property: Health < 1.0 when exploding gradients detected.

        If some parameters have exploding gradients (norm > threshold),
        health score should be degraded.
        """
        exploding_threshold = 100.0
        collector = SeedGradientCollector(exploding_threshold=exploding_threshold)

        # Create parameters with exploding gradients
        params = []
        for _ in range(num_params):
            param = nn.Parameter(torch.randn(param_size))
            # Set gradient above exploding threshold
            param.grad = torch.full((param_size,), exploding_threshold * 10.0)
            params.append(param)

        stats = collector.collect(iter(params))

        assert stats['has_exploding'] is True, \
            "Should detect exploding gradients"
        assert stats['gradient_health'] < 1.0, \
            f"Health should be degraded with exploding gradients, got {stats['gradient_health']}"

    def test_empty_parameters_perfect_health(self):
        """Property: Empty parameter list gives perfect health.

        When there are no parameters to evaluate, health should be 1.0
        (no problems detected).
        """
        collector = SeedGradientCollector()

        stats = collector.collect(iter([]))

        assert stats['gradient_health'] == 1.0


# =============================================================================
# Test Vanishing/Exploding Detection
# =============================================================================


class TestGradientDetectionProperties:
    """Test vanishing and exploding gradient detection.

    Detection should satisfy:
    1. Consistent with configured thresholds
    2. Deterministic (same input produces same output)
    3. Per-parameter granularity
    """

    @given(param_size=st.integers(min_value=1, max_value=100))
    def test_vanishing_detection_threshold(self, param_size: int):
        """Property: Vanishing detection respects threshold.

        Gradients with norm < threshold should be detected as vanishing.
        """
        vanishing_threshold = 1e-7
        collector = SeedGradientCollector(vanishing_threshold=vanishing_threshold)

        # Create parameter with gradient below threshold
        param = nn.Parameter(torch.randn(param_size))
        param.grad = torch.full((param_size,), vanishing_threshold / 100.0)

        stats = collector.collect(iter([param]))

        assert stats['has_vanishing'] is True

    @given(param_size=st.integers(min_value=1, max_value=100))
    def test_exploding_detection_threshold(self, param_size: int):
        """Property: Exploding detection respects threshold.

        Gradients with norm > threshold should be detected as exploding.
        """
        exploding_threshold = 100.0
        collector = SeedGradientCollector(exploding_threshold=exploding_threshold)

        # Create parameter with gradient above threshold
        param = nn.Parameter(torch.randn(param_size))
        param.grad = torch.full((param_size,), exploding_threshold * 2.0)

        stats = collector.collect(iter([param]))

        assert stats['has_exploding'] is True

    @given(
        num_params=st.integers(min_value=2, max_value=10),
        param_size=st.integers(min_value=1, max_value=100),
    )
    def test_mixed_gradient_detection(self, num_params: int, param_size: int):
        """Property: Can detect mixed vanishing and exploding gradients.

        When some parameters have vanishing and others have exploding gradients,
        both flags should be set.
        """
        vanishing_threshold = 1e-7
        exploding_threshold = 100.0
        collector = SeedGradientCollector(
            vanishing_threshold=vanishing_threshold,
            exploding_threshold=exploding_threshold,
        )

        params = []

        # Add vanishing gradient
        param1 = nn.Parameter(torch.randn(param_size))
        param1.grad = torch.full((param_size,), vanishing_threshold / 10.0)
        params.append(param1)

        # Add exploding gradient
        param2 = nn.Parameter(torch.randn(param_size))
        param2.grad = torch.full((param_size,), exploding_threshold * 2.0)
        params.append(param2)

        # Add remaining params with healthy gradients
        for _ in range(num_params - 2):
            param = nn.Parameter(torch.randn(param_size))
            param.grad = torch.ones(param_size)
            params.append(param)

        stats = collector.collect(iter(params))

        assert stats['has_vanishing'] is True, \
            "Should detect vanishing gradients in mixed scenario"
        assert stats['has_exploding'] is True, \
            "Should detect exploding gradients in mixed scenario"


# =============================================================================
# Test Integration with Real Models
# =============================================================================


class TestGradientCollectorIntegration:
    """Test gradient collector with actual neural network models.

    These tests verify that the collector works correctly with real
    PyTorch models and gradient computation.
    """

    @given(
        input_dim=st.integers(min_value=1, max_value=50),
        output_dim=st.integers(min_value=1, max_value=50),
        batch_size=st.integers(min_value=1, max_value=10),
    )
    def test_linear_layer_gradients(
        self,
        input_dim: int,
        output_dim: int,
        batch_size: int,
    ):
        """Property: Collector works with nn.Linear gradients.

        After computing gradients on a simple linear layer,
        the collector should return valid statistics.
        """
        collector = SeedGradientCollector()

        # Create simple model
        model = nn.Linear(input_dim, output_dim)

        # Forward pass
        x = torch.randn(batch_size, input_dim)
        y = model(x)

        # Backward pass (creates gradients)
        loss = y.sum()
        loss.backward()

        # Collect stats
        stats = collector.collect(model.parameters())

        # Verify stats are valid
        assert stats['gradient_norm'] >= 0.0
        assert 0.0 <= stats['gradient_health'] <= 1.0
        assert isinstance(stats['has_vanishing'], bool)
        assert isinstance(stats['has_exploding'], bool)

    @given(
        hidden_dim=st.integers(min_value=4, max_value=32),
        batch_size=st.integers(min_value=1, max_value=10),
    )
    def test_multi_layer_network_gradients(
        self,
        hidden_dim: int,
        batch_size: int,
    ):
        """Property: Collector works with multi-layer networks.

        After computing gradients on a multi-layer network,
        the collector should aggregate statistics across all layers.
        """
        collector = SeedGradientCollector()

        # Create multi-layer network
        model = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

        # Forward pass
        x = torch.randn(batch_size, 10)
        y = model(x)

        # Backward pass
        loss = y.sum()
        loss.backward()

        # Collect stats
        stats = collector.collect(model.parameters())

        # Verify stats are valid
        assert stats['gradient_norm'] >= 0.0
        assert 0.0 <= stats['gradient_health'] <= 1.0

    def test_zero_loss_produces_zero_gradients(self):
        """Property: Zero loss produces zero gradients.

        When loss is constant (multiplied by zero), all gradients should be zero.
        These will be detected as vanishing gradients.
        """
        collector = SeedGradientCollector()

        # Create simple model
        model = nn.Linear(5, 3)

        # Forward pass with zero-scaled loss
        x = torch.randn(2, 5)
        y = model(x)
        loss = y.sum() * 0.0  # Zero-scaled loss (still has grad_fn)

        # Backward pass
        loss.backward()

        # Collect stats
        stats = collector.collect(model.parameters())

        # All gradients should be zero (which counts as vanishing)
        assert stats['gradient_norm'] == 0.0
        assert stats['has_vanishing'] is True
        # Health is degraded due to vanishing gradients
        assert stats['gradient_health'] < 1.0
