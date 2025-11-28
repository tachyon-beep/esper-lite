"""Tests for observation normalization."""

import pytest
import torch

from esper.simic.normalization import RunningMeanStd


class TestRunningMeanStd:
    """Tests for RunningMeanStd normalizer."""

    def test_initial_state(self):
        """Test initial mean is zero and var is one."""
        rms = RunningMeanStd(shape=(4,))
        assert torch.allclose(rms.mean, torch.zeros(4))
        assert torch.allclose(rms.var, torch.ones(4))

    def test_update_changes_stats(self):
        """Test that update modifies running statistics."""
        rms = RunningMeanStd(shape=(2,))

        # Update with batch of known values
        batch = torch.tensor([[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]])
        rms.update(batch)

        # Mean should move toward batch mean (4.0, 8.0)
        assert rms.mean[0] > 0
        assert rms.mean[1] > 0

    def test_normalize_output_range(self):
        """Test that normalize clips to expected range."""
        rms = RunningMeanStd(shape=(3,))

        # Update with some data
        rms.update(torch.randn(100, 3))

        # Normalize should clip to [-10, 10] by default
        extreme = torch.tensor([[1000.0, -1000.0, 0.0]])
        normalized = rms.normalize(extreme, clip=10.0)

        assert normalized.max() <= 10.0
        assert normalized.min() >= -10.0

    def test_to_device(self):
        """Test moving stats to device."""
        rms = RunningMeanStd(shape=(5,))
        rms = rms.to("cpu")  # Should work even if already on CPU

        assert rms.mean.device.type == "cpu"
        assert rms.var.device.type == "cpu"

    def test_welford_stability(self):
        """Test numerical stability with Welford's algorithm."""
        rms = RunningMeanStd(shape=(1,))

        # Multiple small updates shouldn't cause numerical issues
        for _ in range(100):
            rms.update(torch.randn(10, 1) * 0.01 + 100.0)

        # Mean should be close to 100
        assert 99.0 < rms.mean[0] < 101.0
        # Var should be small (0.01^2 scale)
        assert rms.var[0] < 1.0
