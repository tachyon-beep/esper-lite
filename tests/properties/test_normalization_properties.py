"""Property-based tests for feature normalization.

Tests RunningMeanStd and normalization invariants using batch tensor operations.

The actual RunningMeanStd API:
- __init__(shape: tuple[int, ...], epsilon: float = 1e-4)
- update(x: torch.Tensor) -> None  # batch update
- normalize(x: torch.Tensor, clip: float = 10.0) -> torch.Tensor
"""

import math
import pytest
import torch
from hypothesis import given, assume, settings
from hypothesis import strategies as st
from tests.strategies import bounded_floats

from esper.simic.normalization import RunningMeanStd


class TestNormalizationConvergence:
    """Test that normalization converges to mean=0, std=1."""

    @given(
        values=st.lists(bounded_floats(-100.0, 100.0), min_size=100, max_size=1000)
    )
    @settings(max_examples=50, deadline=None)
    def test_normalized_values_zero_mean(self, values):
        """Property: After normalization, mean should be ≈ 0.

        This tests that the running mean estimator converges correctly
        and that normalization centers the data.
        """
        # Create normalizer for scalar observations
        normalizer = RunningMeanStd(shape=())

        # Convert values to tensor batches and update
        # We'll update in batches for efficiency
        batch_size = 32
        for i in range(0, len(values), batch_size):
            batch = torch.tensor(values[i:i+batch_size], dtype=torch.float32)
            normalizer.update(batch)

        # Normalize all values
        all_values = torch.tensor(values, dtype=torch.float32)
        normalized = normalizer.normalize(all_values)

        # Compute mean
        mean = normalized.mean().item()

        # Mean should be close to zero
        assert abs(mean) < 0.2, f"Normalized mean {mean} not close to 0"

    @given(
        values=st.lists(bounded_floats(-100.0, 100.0), min_size=100, max_size=1000)
    )
    @settings(max_examples=50, deadline=None)
    def test_normalized_values_unit_variance(self, values):
        """Property: After normalization, variance should be ≈ 1.

        This tests that the running variance estimator converges correctly
        and that normalization scales the data appropriately.
        """
        # Skip if all values are the same (zero variance case)
        all_values_tensor = torch.tensor(values, dtype=torch.float32)
        if all_values_tensor.var().item() < 1e-6:
            assume(False)  # Skip this example

        normalizer = RunningMeanStd(shape=())

        # Update with batches
        batch_size = 32
        for i in range(0, len(values), batch_size):
            batch = torch.tensor(values[i:i+batch_size], dtype=torch.float32)
            normalizer.update(batch)

        # Normalize all values
        all_values = torch.tensor(values, dtype=torch.float32)
        normalized = normalizer.normalize(all_values)

        # Compute variance
        variance = normalized.var(unbiased=False).item()

        # Variance should be close to 1
        assert abs(variance - 1.0) < 0.3, f"Normalized variance {variance} not close to 1"


class TestNormalizationBounds:
    """Test that normalized values are bounded."""

    @given(
        values=st.lists(bounded_floats(-100.0, 100.0), min_size=10, max_size=100),
    )
    def test_normalized_values_roughly_bounded(self, values):
        """Property: Normalized values should be roughly in [-3, 3] (99.7% for normal dist).

        This tests that the normalization produces reasonable outputs.
        Most values should fall within 3 standard deviations of the mean.
        """
        normalizer = RunningMeanStd(shape=())

        # Update with all values
        all_values = torch.tensor(values, dtype=torch.float32)
        normalizer.update(all_values)

        # Normalize
        normalized = normalizer.normalize(all_values)

        # Most values should be in [-3, 3] range (allow some outliers)
        in_range = ((normalized > -3.5) & (normalized < 3.5)).sum().item()
        ratio = in_range / len(values)

        assert ratio > 0.8, f"Only {ratio*100:.1f}% of values in [-3.5, 3.5]"

    @given(
        values=st.lists(bounded_floats(-100.0, 100.0), min_size=10, max_size=100),
        clip=st.sampled_from([5.0, 10.0, 20.0]),
    )
    def test_clipping_enforced(self, values, clip):
        """Property: normalize() with clip parameter should enforce bounds.

        This tests that the clipping mechanism works correctly.
        """
        normalizer = RunningMeanStd(shape=())

        all_values = torch.tensor(values, dtype=torch.float32)
        normalizer.update(all_values)

        # Normalize with specific clip value
        normalized = normalizer.normalize(all_values, clip=clip)

        # All values should be within [-clip, clip]
        assert (normalized >= -clip).all(), f"Found value below -{clip}"
        assert (normalized <= clip).all(), f"Found value above {clip}"


class TestMultidimensionalNormalization:
    """Test normalization for multi-dimensional observations."""

    @given(
        batch_size=st.integers(min_value=10, max_value=100),
        obs_dim=st.sampled_from([3, 5, 10]),
    )
    @settings(max_examples=30, deadline=None)
    def test_multidim_shape_preserved(self, batch_size, obs_dim):
        """Property: Normalization should preserve tensor shape.

        This tests that normalization works correctly for vector observations.
        """
        normalizer = RunningMeanStd(shape=(obs_dim,))

        # Create batch of observations
        observations = torch.randn(batch_size, obs_dim)

        # Update
        normalizer.update(observations)

        # Normalize
        normalized = normalizer.normalize(observations)

        # Shape should be preserved
        assert normalized.shape == observations.shape

    @given(
        num_batches=st.integers(min_value=5, max_value=20),
        batch_size=st.integers(min_value=16, max_value=64),
        obs_dim=st.integers(min_value=3, max_value=10),
    )
    @settings(max_examples=20, deadline=None)
    def test_multidim_convergence(self, num_batches, batch_size, obs_dim):
        """Property: Multi-dim normalization should converge per-dimension.

        This tests that each dimension is normalized independently.
        """
        normalizer = RunningMeanStd(shape=(obs_dim,))

        # Generate and update multiple batches
        all_observations = []
        for _ in range(num_batches):
            batch = torch.randn(batch_size, obs_dim)
            normalizer.update(batch)
            all_observations.append(batch)

        # Concatenate all observations
        all_obs = torch.cat(all_observations, dim=0)

        # Normalize
        normalized = normalizer.normalize(all_obs)

        # Each dimension should have approximately zero mean
        per_dim_mean = normalized.mean(dim=0)
        assert (per_dim_mean.abs() < 0.3).all(), f"Per-dim means not close to 0: {per_dim_mean}"


class TestNormalizationEdgeCases:
    """Test edge cases and numerical stability."""

    def test_constant_values_dont_crash(self):
        """Property: Normalizing constant values should not crash (despite zero variance).

        This tests numerical stability with epsilon handling.
        """
        normalizer = RunningMeanStd(shape=())

        # All same values (zero variance)
        constant_batch = torch.full((100,), 5.0)
        normalizer.update(constant_batch)

        # Should not crash (epsilon prevents div by zero)
        normalized = normalizer.normalize(constant_batch)

        # Should return finite values
        assert torch.isfinite(normalized).all()

    def test_single_update_then_normalize(self):
        """Property: Normalization should work after a single update.

        This tests the bootstrap case.
        """
        normalizer = RunningMeanStd(shape=())

        # Single batch
        batch = torch.randn(50)
        normalizer.update(batch)

        # Normalize the same batch
        normalized = normalizer.normalize(batch)

        # Should produce finite values
        assert torch.isfinite(normalized).all()

        # Mean should be approximately zero
        assert abs(normalized.mean().item()) < 0.5

    @given(
        values1=st.lists(bounded_floats(-50.0, 50.0), min_size=50, max_size=100),
        values2=st.lists(bounded_floats(-50.0, 50.0), min_size=50, max_size=100),
    )
    def test_incremental_update_consistency(self, values1, values2):
        """Property: Incremental updates should be consistent with batch update.

        This tests that Welford's algorithm produces consistent results.
        """
        # Create two normalizers
        normalizer_incremental = RunningMeanStd(shape=())
        normalizer_batch = RunningMeanStd(shape=())

        # Update incrementally
        batch1 = torch.tensor(values1, dtype=torch.float32)
        batch2 = torch.tensor(values2, dtype=torch.float32)
        normalizer_incremental.update(batch1)
        normalizer_incremental.update(batch2)

        # Update as single batch
        all_values = torch.cat([batch1, batch2])
        normalizer_batch.update(all_values)

        # Test values should normalize similarly
        test_batch = torch.randn(20)
        norm_incremental = normalizer_incremental.normalize(test_batch)
        norm_batch = normalizer_batch.normalize(test_batch)

        # Results should be very close (allowing for numerical precision)
        diff = (norm_incremental - norm_batch).abs().max().item()
        assert diff < 0.1, f"Incremental vs batch difference: {diff}"


class TestNormalizationFiniteness:
    """Test that normalized values are always finite."""

    @given(
        values=st.lists(bounded_floats(-1000.0, 1000.0), min_size=20, max_size=200)
    )
    def test_normalized_values_finite(self, values):
        """Property: All normalized values must be finite (no NaN, no Inf).

        This is critical for RL training stability.
        """
        normalizer = RunningMeanStd(shape=())

        all_values = torch.tensor(values, dtype=torch.float32)
        normalizer.update(all_values)

        normalized = normalizer.normalize(all_values)

        # Check no NaN
        assert not torch.isnan(normalized).any(), "Found NaN in normalized values"

        # Check no Inf
        assert not torch.isinf(normalized).any(), "Found Inf in normalized values"

    @given(
        batch_size=st.integers(min_value=10, max_value=100),
        obs_dim=st.integers(min_value=2, max_value=20),
    )
    def test_multidim_normalized_finite(self, batch_size, obs_dim):
        """Property: Multi-dimensional normalization produces finite values.

        This tests vector observations for numerical stability.
        """
        normalizer = RunningMeanStd(shape=(obs_dim,))

        observations = torch.randn(batch_size, obs_dim) * 100  # Large scale
        normalizer.update(observations)

        normalized = normalizer.normalize(observations)

        assert torch.isfinite(normalized).all(), "Found non-finite values in multi-dim normalization"
