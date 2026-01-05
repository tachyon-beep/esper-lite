"""Tests for normalizer state_dict methods."""

import torch
from esper.simic.control import RunningMeanStd, RewardNormalizer, ValueNormalizer


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


class TestValueNormalizerStateDict:
    """Test state_dict/load_state_dict for ValueNormalizer (P1 GAE fix)."""

    def test_state_dict_roundtrip(self):
        """State can be saved and restored."""
        vn = ValueNormalizer(device="cpu")
        # Update with enough data to pass warmup threshold
        for _ in range(10):
            vn.update(torch.randn(10))

        # Save state
        state = vn.state_dict()

        # Create new instance and load
        vn2 = ValueNormalizer(device="cpu")
        vn2.load_state_dict(state)

        # Verify identical
        assert torch.allclose(vn.mean, vn2.mean)
        assert torch.allclose(vn.var, vn2.var)
        assert torch.allclose(vn.count, vn2.count)

    def test_state_dict_keys(self):
        """state_dict has expected keys."""
        vn = ValueNormalizer()
        state = vn.state_dict()
        assert set(state.keys()) == {"mean", "var", "count"}

    def test_normalize_denormalize_roundtrip(self):
        """Normalize then denormalize returns original values (approx)."""
        vn = ValueNormalizer(device="cpu")
        # Warmup with enough samples
        for _ in range(10):
            vn.update(torch.randn(10) * 5.0 + 2.0)

        original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = vn.normalize(original)
        recovered = vn.denormalize(normalized)

        # Should recover original values
        assert torch.allclose(original, recovered, atol=1e-5)

    def test_scale_consistency_for_gae(self):
        """Critical test: normalize and denormalize use same scale.

        This is the P1 bug fix verification. The bug was:
        - Critic trained on normalized_returns = returns / batch_std
        - GAE used raw values (not denormalized)
        - Scale mismatch corrupted advantages

        With ValueNormalizer:
        - normalize() and denormalize() use same running std
        - GAE denormalizes values, computes delta, returns on raw scale
        - Critic receives normalized returns using same scale
        """
        vn = ValueNormalizer(device="cpu")

        # Simulate multiple batches with varying scales
        batch1 = torch.randn(50) * 2.0 + 1.0  # std ~2, mean ~1
        batch2 = torch.randn(50) * 2.0 + 1.0
        batch3 = torch.randn(50) * 2.0 + 1.0

        vn.update(batch1)
        vn.update(batch2)
        vn.update(batch3)

        # Simulate GAE computation
        raw_values = torch.tensor([1.0, 2.0, 3.0])  # Critic outputs (normalized scale)

        # GAE would denormalize for delta computation
        denorm_values = vn.denormalize(raw_values)

        # Returns computed in raw scale
        raw_returns = denorm_values + torch.randn(3)  # Returns = A + V

        # Normalize returns for critic training
        norm_returns = vn.normalize(raw_returns)
        assert norm_returns.shape == raw_returns.shape

        # Key invariant: std should be consistent
        scale = vn.get_scale()
        assert scale > 0.1, "Scale should be positive and reasonable"

        # Denorm -> norm should give back something proportional to original
        round_trip = vn.normalize(vn.denormalize(raw_values))
        assert torch.allclose(raw_values, round_trip, atol=1e-5)

    def test_warmup_returns_unchanged(self):
        """During warmup, normalize/denormalize return input unchanged."""
        vn = ValueNormalizer(device="cpu")

        # Before warmup (need 32 samples)
        vn.update(torch.randn(10))  # Only 10 samples

        original = torch.tensor([1.0, 2.0, 3.0])
        normalized = vn.normalize(original)
        denormalized = vn.denormalize(original)

        # Should return unchanged during warmup
        assert torch.allclose(original, normalized)
        assert torch.allclose(original, denormalized)
        assert vn.get_scale() == 1.0
