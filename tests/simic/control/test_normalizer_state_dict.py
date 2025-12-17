"""Tests for normalizer state_dict methods."""

import pytest
import torch
from esper.simic.control.normalization import RunningMeanStd, RewardNormalizer


class TestRunningMeanStdStateDict:
    """Test state_dict/load_state_dict for RunningMeanStd."""

    def test_state_dict_roundtrip(self):
        """State can be saved and restored."""
        rms = RunningMeanStd(shape=(10,), device="cpu")
        # Update with some data
        rms.update(torch.randn(32, 10))
        rms.update(torch.randn(32, 10))

        # Save state
        state = rms.state_dict()

        # Create new instance and load
        rms2 = RunningMeanStd(shape=(10,), device="cpu")
        rms2.load_state_dict(state)

        # Verify identical
        assert torch.allclose(rms.mean, rms2.mean)
        assert torch.allclose(rms.var, rms2.var)
        assert torch.allclose(rms.count, rms2.count)

    def test_state_dict_keys(self):
        """state_dict has expected keys."""
        rms = RunningMeanStd(shape=(5,), device="cpu")
        state = rms.state_dict()
        assert set(state.keys()) == {"mean", "var", "count"}


class TestRewardNormalizerStateDict:
    """Test state_dict/load_state_dict for RewardNormalizer."""

    def test_state_dict_roundtrip(self):
        """State can be saved and restored."""
        rn = RewardNormalizer(clip=10.0)
        # Update with some data
        for _ in range(100):
            rn.update_and_normalize(torch.randn(1).item())

        # Save state
        state = rn.state_dict()

        # Create new instance and load
        rn2 = RewardNormalizer(clip=10.0)
        rn2.load_state_dict(state)

        # Verify identical
        assert rn.mean == rn2.mean
        assert rn.m2 == rn2.m2
        assert rn.count == rn2.count

    def test_state_dict_keys(self):
        """state_dict has expected keys."""
        rn = RewardNormalizer()
        state = rn.state_dict()
        assert set(state.keys()) == {"mean", "m2", "count"}
