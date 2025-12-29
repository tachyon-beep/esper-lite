"""Tests for entropy-clip correlation tracking for policy collapse detection."""

import pytest
from collections import deque
from datetime import datetime, timezone

from esper.karn.sanctum.schema import compute_correlation, TamiyoState, SanctumSnapshot
from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline import PPOUpdatePayload, TelemetryEvent, TelemetryEventType


class TestCorrelation:
    """Test metric correlation calculation."""

    def test_perfect_negative_correlation(self):
        """When entropy drops and clip rises perfectly, correlation = -1."""
        entropy = deque([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        clip = deque([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        corr = compute_correlation(entropy, clip)
        assert corr < -0.95

    def test_perfect_positive_correlation(self):
        """When both rise together, correlation = +1."""
        entropy = deque([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        clip = deque([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        corr = compute_correlation(entropy, clip)
        assert corr > 0.95

    def test_no_correlation(self):
        """When metrics are uncorrelated, correlation near 0."""
        entropy = deque([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5])
        clip = deque([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        corr = compute_correlation(entropy, clip)
        assert abs(corr) < 0.3

    def test_short_history_returns_zero(self):
        """With <5 samples, return 0."""
        entropy = deque([1.0, 0.9])
        clip = deque([0.1, 0.2])
        corr = compute_correlation(entropy, clip)
        assert corr == 0.0

    def test_zero_variance_returns_zero(self):
        """Constant values should return 0 (not NaN/crash)."""
        entropy = deque([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        clip = deque([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        corr = compute_correlation(entropy, clip)
        assert corr == 0.0  # Not NaN

    def test_list_input_works(self):
        """Function should work with plain lists too."""
        entropy = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        clip = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        corr = compute_correlation(entropy, clip)
        assert corr < -0.95

    def test_single_variance_zero_returns_zero(self):
        """If only one series has zero variance, return 0."""
        entropy = deque([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        clip = deque([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        corr = compute_correlation(entropy, clip)
        assert corr == 0.0

    def test_uses_last_10_samples(self):
        """Should use only last 10 samples when history is longer."""
        # First 5 samples are noise, last 10 are perfectly negatively correlated
        entropy = deque([0.5, 0.5, 0.5, 0.5, 0.5] + [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        clip = deque([0.5, 0.5, 0.5, 0.5, 0.5] + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        corr = compute_correlation(entropy, clip)
        # Should still be close to -1 since we use last 10 samples
        assert corr < -0.95


class TestAggregatorCorrelation:
    """Test aggregator computes entropy-clip correlation."""

    def test_ppo_update_computes_correlation(self):
        """PPO updates should compute entropy-clip correlation."""
        aggregator = SanctumAggregator(num_envs=1)

        # Send 10 PPO updates with negatively correlated entropy/clip
        for i in range(10):
            entropy = 1.0 - i * 0.1  # 1.0, 0.9, 0.8, ...
            clip = 0.05 + i * 0.03  # 0.05, 0.08, 0.11, ...
            payload = PPOUpdatePayload(
                policy_loss=-0.1,
                value_loss=1.0,
                entropy=entropy,
                clip_fraction=clip,
                kl_divergence=0.01,
                grad_norm=1.0,
                nan_grad_count=0,
                inner_epoch=i,
                batch=i,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                timestamp=datetime.now(timezone.utc),
                data=payload,
            )
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        # Should have negative correlation (entropy falling, clip rising)
        assert snapshot.tamiyo.entropy_clip_correlation < -0.9

    def test_correlation_zero_before_sufficient_data(self):
        """Correlation should be 0 before having enough PPO updates."""
        aggregator = SanctumAggregator(num_envs=1)

        # Send just 3 PPO updates (less than required 5)
        for i in range(3):
            payload = PPOUpdatePayload(
                policy_loss=-0.1,
                value_loss=1.0,
                entropy=1.0 - i * 0.1,
                clip_fraction=0.1 + i * 0.05,
                kl_divergence=0.01,
                grad_norm=1.0,
                nan_grad_count=0,
                inner_epoch=i,
                batch=i,
            )
            event = TelemetryEvent(
                event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
                timestamp=datetime.now(timezone.utc),
                data=payload,
            )
            aggregator.process_event(event)

        snapshot = aggregator.get_snapshot()
        # Should be 0 (not enough data)
        assert snapshot.tamiyo.entropy_clip_correlation == 0.0


class TestTamiyoStateHasCorrelationField:
    """Test TamiyoState has entropy_clip_correlation field."""

    def test_field_defaults_to_zero(self):
        """entropy_clip_correlation should default to 0."""
        tamiyo = TamiyoState()
        assert tamiyo.entropy_clip_correlation == 0.0

    def test_field_present_in_snapshot(self):
        """SanctumSnapshot should include tamiyo.entropy_clip_correlation."""
        snapshot = SanctumSnapshot()
        assert hasattr(snapshot.tamiyo, "entropy_clip_correlation")
        assert snapshot.tamiyo.entropy_clip_correlation == 0.0
