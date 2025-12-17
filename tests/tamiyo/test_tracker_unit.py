"""Unit tests for SignalTracker."""

import pytest

from esper.tamiyo.tracker import SignalTracker
from esper.leyline import TrainingSignals


@pytest.mark.tamiyo
class TestSignalTrackerUpdate:
    """Tests for SignalTracker.update() method."""

    def test_update_returns_valid_signals(self, signal_tracker):
        """Should return valid TrainingSignals structure."""
        signals = signal_tracker.update(
            epoch=5,
            global_step=100,
            train_loss=0.5,
            train_accuracy=85.0,
            val_loss=0.6,
            val_accuracy=83.0,
            active_seeds=[],
        )

        assert isinstance(signals, TrainingSignals)
        assert signals.metrics.epoch == 5
        assert signals.metrics.global_step == 100
        assert signals.metrics.train_loss == 0.5
        assert signals.metrics.val_loss == 0.6
        assert signals.metrics.train_accuracy == 85.0
        assert signals.metrics.val_accuracy == 83.0

    def test_delta_computation_positive_improvement(self, signal_tracker):
        """Should correctly compute positive deltas on improvement."""
        # First update establishes baseline
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )

        # Second update shows improvement
        signals = signal_tracker.update(
            epoch=2,
            global_step=20,
            train_loss=0.8,
            train_accuracy=55.0,
            val_loss=1.0,
            val_accuracy=52.0,
            active_seeds=[],
        )

        # loss_delta = prev_loss - current_loss (positive = improvement)
        assert signals.metrics.loss_delta == pytest.approx(1.2 - 1.0)
        # accuracy_delta = current - prev (positive = improvement)
        assert signals.metrics.accuracy_delta == pytest.approx(52.0 - 48.0)

    def test_delta_computation_negative_improvement(self, signal_tracker):
        """Should correctly compute negative deltas on degradation."""
        # First update
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=0.5,
            train_accuracy=80.0,
            val_loss=0.6,
            val_accuracy=78.0,
            active_seeds=[],
        )

        # Second update shows degradation
        signals = signal_tracker.update(
            epoch=2,
            global_step=20,
            train_loss=0.7,
            train_accuracy=75.0,
            val_loss=0.8,
            val_accuracy=73.0,
            active_seeds=[],
        )

        # Loss increased (negative delta)
        assert signals.metrics.loss_delta == pytest.approx(0.6 - 0.8)
        # Accuracy decreased (negative delta)
        assert signals.metrics.accuracy_delta == pytest.approx(73.0 - 78.0)

    def test_history_window_respected(self):
        """History deque should not exceed maxlen."""
        tracker = SignalTracker(history_window=5)

        # Add 10 updates
        for i in range(10):
            tracker.update(
                epoch=i,
                global_step=i * 10,
                train_loss=1.0 - i * 0.05,
                train_accuracy=50.0 + i * 2,
                val_loss=1.2 - i * 0.05,
                val_accuracy=48.0 + i * 2,
                active_seeds=[],
            )

        # History should only contain last 5
        assert len(tracker._loss_history) == 5
        assert len(tracker._accuracy_history) == 5

        # Should contain values from epochs 5-9
        expected_val_losses = [1.2 - i * 0.05 for i in range(5, 10)]
        assert list(tracker._loss_history) == pytest.approx(expected_val_losses)


@pytest.mark.tamiyo
class TestBestAccuracyTracking:
    """Tests for best accuracy tracking."""

    def test_best_accuracy_tracked_on_improvement(self, signal_tracker):
        """Should update best_accuracy when validation accuracy improves."""
        # Initial update
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )
        assert signal_tracker._best_accuracy == 48.0

        # Improvement
        signal_tracker.update(
            epoch=2,
            global_step=20,
            train_loss=0.8,
            train_accuracy=55.0,
            val_loss=1.0,
            val_accuracy=52.0,
            active_seeds=[],
        )
        assert signal_tracker._best_accuracy == 52.0

    def test_best_accuracy_not_updated_on_degradation(self, signal_tracker):
        """Should keep best_accuracy when validation accuracy degrades."""
        # Set up with good accuracy
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=0.5,
            train_accuracy=80.0,
            val_loss=0.6,
            val_accuracy=78.0,
            active_seeds=[],
        )
        assert signal_tracker._best_accuracy == 78.0

        # Degradation
        signal_tracker.update(
            epoch=2,
            global_step=20,
            train_loss=0.7,
            train_accuracy=75.0,
            val_loss=0.8,
            val_accuracy=73.0,
            active_seeds=[],
        )
        # Best should remain at 78.0
        assert signal_tracker._best_accuracy == 78.0

    def test_best_accuracy_in_signals(self, signal_tracker):
        """TrainingSignals should include best_val_accuracy."""
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )

        signals = signal_tracker.update(
            epoch=2,
            global_step=20,
            train_loss=0.8,
            train_accuracy=55.0,
            val_loss=1.0,
            val_accuracy=52.0,
            active_seeds=[],
        )

        assert signals.metrics.best_val_accuracy == 52.0


@pytest.mark.tamiyo
class TestPlateauCounter:
    """Tests for plateau counter logic."""

    def test_plateau_counter_increments_on_low_improvement(self, signal_tracker):
        """Should increment plateau counter when accuracy delta is small."""
        # First update establishes baseline
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )

        # Small improvement (< 0.5 default plateau threshold)
        signals = signal_tracker.update(
            epoch=2,
            global_step=20,
            train_loss=0.95,
            train_accuracy=50.2,
            val_loss=1.15,
            val_accuracy=48.2,  # Only 0.2% improvement
            active_seeds=[],
        )

        assert signals.metrics.plateau_epochs == 1

    def test_plateau_counter_resets_on_significant_improvement(self, signal_tracker):
        """Should reset plateau counter when improvement is significant."""
        # Build up plateau
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )
        signal_tracker.update(
            epoch=2,
            global_step=20,
            train_loss=0.98,
            train_accuracy=50.1,
            val_loss=1.18,
            val_accuracy=48.1,  # Small improvement
            active_seeds=[],
        )
        assert signal_tracker._plateau_count == 1

        # Large improvement (>= 0.5 threshold)
        signals = signal_tracker.update(
            epoch=3,
            global_step=30,
            train_loss=0.8,
            train_accuracy=55.0,
            val_loss=1.0,
            val_accuracy=52.0,  # 3.9% improvement
            active_seeds=[],
        )

        assert signals.metrics.plateau_epochs == 0
        assert signal_tracker._plateau_count == 0

    def test_plateau_counter_custom_threshold(self):
        """Should respect custom plateau threshold."""
        tracker = SignalTracker(plateau_threshold_pct=1.0)  # Stricter threshold

        tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )

        # 0.8% improvement - normally not a plateau, but with threshold=1.0 it is
        signals = tracker.update(
            epoch=2,
            global_step=20,
            train_loss=0.95,
            train_accuracy=50.5,
            val_loss=1.15,
            val_accuracy=48.8,
            active_seeds=[],
        )

        assert signals.metrics.plateau_epochs == 1


@pytest.mark.tamiyo
class TestStabilization:
    """Tests for stabilization detection logic."""

    def test_stabilization_threshold_boundary(self, signal_tracker):
        """Should detect stability when improvement drops below threshold."""
        # Start with high loss
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=2.0,
            train_accuracy=30.0,
            val_loss=2.5,
            val_accuracy=28.0,
            active_seeds=[],
        )

        # Consecutive small improvements (below 3% threshold)
        # Relative improvement = (prev - current) / prev
        for i in range(2, 6):
            prev_loss = 2.5 - (i - 2) * 0.05
            # Make improvement < 3% of previous loss
            improvement = prev_loss * 0.02  # 2% improvement
            new_loss = prev_loss - improvement

            signals = signal_tracker.update(
                epoch=i,
                global_step=i * 10,
                train_loss=2.0 - i * 0.05,
                train_accuracy=30.0 + i * 2,
                val_loss=new_loss,
                val_accuracy=28.0 + i * 2,
                active_seeds=[],
            )

            # Should stabilize after 3 consecutive stable epochs
            if i >= 4:  # 3 stable epochs counted
                assert signals.metrics.host_stabilized == 1
                assert signal_tracker.is_stabilized
            else:
                assert signals.metrics.host_stabilized == 0

    def test_divergence_not_stable(self, signal_tracker):
        """Should not count as stable if loss spikes (divergence detection)."""
        # Establish baseline
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )

        # Small improvement but then spike
        signal_tracker.update(
            epoch=2,
            global_step=20,
            train_loss=0.95,
            train_accuracy=51.0,
            val_loss=1.18,
            val_accuracy=49.0,
            active_seeds=[],
        )

        # Divergence: loss spikes to > 1.5x previous
        signals = signal_tracker.update(
            epoch=3,
            global_step=30,
            train_loss=1.2,
            train_accuracy=48.0,
            val_loss=2.0,  # 2.0 > 1.18 * 1.5
            val_accuracy=46.0,
            active_seeds=[],
        )

        # Should reset stable count
        assert signal_tracker._stable_count == 0
        assert signals.metrics.host_stabilized == 0

    def test_custom_stabilization_parameters(self):
        """Should respect custom stabilization threshold and epochs."""
        tracker = SignalTracker(
            stabilization_threshold=0.01,  # Stricter: 1%
            stabilization_epochs=5,         # More epochs required
        )

        # Establish baseline
        tracker.update(
            epoch=1,
            global_step=10,
            train_loss=2.0,
            train_accuracy=30.0,
            val_loss=2.5,
            val_accuracy=28.0,
            active_seeds=[],
        )

        # Make small improvements below 1% threshold
        for i in range(2, 8):
            prev_loss = 2.5 - (i - 2) * 0.02
            improvement = prev_loss * 0.005  # 0.5% improvement
            new_loss = prev_loss - improvement

            signals = tracker.update(
                epoch=i,
                global_step=i * 10,
                train_loss=2.0 - i * 0.02,
                train_accuracy=30.0 + i,
                val_loss=new_loss,
                val_accuracy=28.0 + i,
                active_seeds=[],
            )

            # Should stabilize after 5 consecutive stable epochs (epoch 6)
            if i >= 6:
                assert signals.metrics.host_stabilized == 1
            else:
                assert signals.metrics.host_stabilized == 0

    def test_stabilization_latch_behavior(self, signal_tracker):
        """Once stabilized, should stay stabilized (latch behavior)."""
        # Establish baseline
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=2.0,
            train_accuracy=30.0,
            val_loss=2.5,
            val_accuracy=28.0,
            active_seeds=[],
        )

        # Stabilize
        for i in range(2, 6):
            prev_loss = 2.5 - (i - 2) * 0.05
            improvement = prev_loss * 0.02
            new_loss = prev_loss - improvement

            signal_tracker.update(
                epoch=i,
                global_step=i * 10,
                train_loss=2.0 - i * 0.05,
                train_accuracy=30.0 + i * 2,
                val_loss=new_loss,
                val_accuracy=28.0 + i * 2,
                active_seeds=[],
            )

        assert signal_tracker.is_stabilized

        # Now make a large improvement (would normally reset stable count)
        signals = signal_tracker.update(
            epoch=6,
            global_step=60,
            train_loss=1.0,
            train_accuracy=60.0,
            val_loss=1.5,  # Large improvement
            val_accuracy=58.0,
            active_seeds=[],
        )

        # Should remain stabilized (latch)
        assert signals.metrics.host_stabilized == 1
        assert signal_tracker.is_stabilized


@pytest.mark.tamiyo
class TestTrackerReset:
    """Tests for SignalTracker.reset() method."""

    def test_reset_clears_all_state(self, signal_tracker):
        """Should reset all tracker state to initial values."""
        # Populate tracker with state
        for i in range(5):
            signal_tracker.update(
                epoch=i,
                global_step=i * 10,
                train_loss=1.0 - i * 0.1,
                train_accuracy=50.0 + i * 5,
                val_loss=1.2 - i * 0.1,
                val_accuracy=48.0 + i * 5,
                active_seeds=[],
            )

        # Verify state is populated
        assert len(signal_tracker._loss_history) > 0
        assert signal_tracker._best_accuracy > 0
        assert signal_tracker._plateau_count >= 0

        # Reset
        signal_tracker.reset()

        # Verify reset
        assert len(signal_tracker._loss_history) == 0
        assert len(signal_tracker._accuracy_history) == 0
        assert signal_tracker._best_accuracy == 0.0
        assert signal_tracker._plateau_count == 0
        assert signal_tracker._prev_accuracy == 0.0
        assert signal_tracker._prev_loss == float('inf')
        assert signal_tracker._is_stabilized is False
        assert signal_tracker._stable_count == 0

    def test_reset_preserves_configuration(self, signal_tracker):
        """Should preserve configuration parameters after reset."""
        original_threshold = signal_tracker.plateau_threshold_pct  # P2-A: renamed
        original_window = signal_tracker.history_window

        # Modify state
        signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )

        signal_tracker.reset()

        # Configuration should remain
        assert signal_tracker.plateau_threshold_pct == original_threshold  # P2-A: renamed
        assert signal_tracker.history_window == original_window


@pytest.mark.tamiyo
class TestActiveSeedsIntegration:
    """Tests for active seeds integration in signals."""

    def test_signals_with_active_seeds(self, signal_tracker, mock_seed_factory):
        """Should populate seed fields when active seeds present."""
        from esper.leyline import SeedStage

        seed = mock_seed_factory(
            seed_id="test_seed_123",
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            alpha=0.0,
            improvement=2.5,
        )

        signals = signal_tracker.update(
            epoch=10,
            global_step=100,
            train_loss=0.5,
            train_accuracy=85.0,
            val_loss=0.6,
            val_accuracy=83.0,
            active_seeds=[seed],
        )

        assert signals.active_seeds == ["test_seed_123"]
        assert signals.seed_stage == int(SeedStage.TRAINING)
        assert signals.seed_epochs_in_stage == 5
        assert signals.seed_alpha == 0.0
        assert signals.seed_improvement == 2.5

    def test_signals_with_no_active_seeds(self, signal_tracker):
        """Should have default seed values when no active seeds."""
        signals = signal_tracker.update(
            epoch=10,
            global_step=100,
            train_loss=0.5,
            train_accuracy=85.0,
            val_loss=0.6,
            val_accuracy=83.0,
            active_seeds=[],
        )

        assert signals.active_seeds == []
        assert signals.seed_stage == 0
        assert signals.seed_epochs_in_stage == 0
        assert signals.seed_alpha == 0.0
        assert signals.seed_improvement == 0.0
