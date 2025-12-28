"""Tests for GradientEMATracker (B7-DRL-01)."""

import pytest

from esper.simic.telemetry import GradientEMATracker


class TestGradientEMATracker:
    """Tests for gradient EMA tracking and drift detection."""

    def test_first_update_initializes_ema(self):
        """First update initializes EMA to current values with zero drift."""
        tracker = GradientEMATracker()
        metrics = tracker.update(grad_norm=5.0, grad_health=0.9)

        assert metrics["ema_grad_norm"] == 5.0
        assert metrics["ema_grad_health"] == 0.9
        assert metrics["norm_drift"] == 0.0
        assert metrics["health_drift"] == 0.0

    def test_ema_smooths_values(self):
        """EMA smooths values over time with momentum."""
        tracker = GradientEMATracker(momentum=0.9)

        # First update initializes
        tracker.update(grad_norm=10.0, grad_health=1.0)

        # Second update: EMA = 0.9 * 10.0 + 0.1 * 5.0 = 9.5
        metrics = tracker.update(grad_norm=5.0, grad_health=0.8)

        assert metrics["ema_grad_norm"] == pytest.approx(9.5, rel=1e-6)
        assert metrics["ema_grad_health"] == pytest.approx(0.98, rel=1e-6)

    def test_drift_detection_on_sudden_change(self):
        """Detects drift when values change suddenly from EMA."""
        tracker = GradientEMATracker()

        # Initialize with stable values
        tracker.update(grad_norm=10.0, grad_health=1.0)

        # Sudden change: norm goes from 10 to 20
        # drift = |20 - 10| / (10 + epsilon) = 1.0
        metrics = tracker.update(grad_norm=20.0, grad_health=1.0)

        assert metrics["norm_drift"] == pytest.approx(1.0, rel=1e-6)

    def test_check_drift_returns_flag_and_metrics(self):
        """check_drift() returns has_drift flag and metrics dict."""
        tracker = GradientEMATracker()

        # Initialize
        tracker.update(grad_norm=10.0, grad_health=1.0)

        # Check with moderate drift (should not flag)
        has_drift, metrics = tracker.check_drift(
            grad_norm=12.0, grad_health=0.9, drift_threshold=0.5
        )

        assert has_drift is False
        assert "norm_drift" in metrics
        assert "health_drift" in metrics

    def test_check_drift_flags_high_drift(self):
        """check_drift() flags when drift exceeds threshold."""
        tracker = GradientEMATracker()

        # Initialize
        tracker.update(grad_norm=10.0, grad_health=1.0)

        # Check with high drift (norm: 10 -> 25 = 150% drift)
        has_drift, metrics = tracker.check_drift(
            grad_norm=25.0, grad_health=1.0, drift_threshold=0.5
        )

        assert has_drift is True
        assert metrics["norm_drift"] > 0.5

    def test_state_dict_and_load(self):
        """State can be serialized and restored for checkpointing."""
        tracker1 = GradientEMATracker()
        tracker1.update(grad_norm=10.0, grad_health=0.9)
        tracker1.update(grad_norm=12.0, grad_health=0.85)

        state = tracker1.state_dict()

        tracker2 = GradientEMATracker()
        tracker2.load_state_dict(state)

        assert tracker2.ema_norm == tracker1.ema_norm
        assert tracker2.ema_health == tracker1.ema_health
        assert tracker2._initialized == tracker1._initialized
        assert tracker2._update_count == tracker1._update_count

    def test_momentum_affects_smoothing_rate(self):
        """Higher momentum = slower adaptation to changes."""
        slow_tracker = GradientEMATracker(momentum=0.99)
        fast_tracker = GradientEMATracker(momentum=0.5)

        # Both start at 10
        slow_tracker.update(grad_norm=10.0, grad_health=1.0)
        fast_tracker.update(grad_norm=10.0, grad_health=1.0)

        # Both see 20
        slow_metrics = slow_tracker.update(grad_norm=20.0, grad_health=1.0)
        fast_metrics = fast_tracker.update(grad_norm=20.0, grad_health=1.0)

        # Slow tracker barely moved (0.99 * 10 + 0.01 * 20 = 10.1)
        # Fast tracker moved more (0.5 * 10 + 0.5 * 20 = 15.0)
        assert slow_metrics["ema_grad_norm"] == pytest.approx(10.1, rel=1e-6)
        assert fast_metrics["ema_grad_norm"] == pytest.approx(15.0, rel=1e-6)
