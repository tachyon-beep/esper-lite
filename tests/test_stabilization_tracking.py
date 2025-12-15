"""Tests for stabilization tracking in SignalTracker.

Stabilization gating ensures seeds only get credit for improvements AFTER
the explosive growth phase ends. The tracker monitors relative loss improvement
and flips a latch once improvement drops below 5% for 3 consecutive epochs.
"""


from esper.tamiyo.tracker import SignalTracker
from esper.leyline import DEFAULT_STABILIZATION_THRESHOLD, DEFAULT_STABILIZATION_EPOCHS

# Backwards compatibility aliases
STABILIZATION_THRESHOLD = DEFAULT_STABILIZATION_THRESHOLD
STABILIZATION_EPOCHS = DEFAULT_STABILIZATION_EPOCHS


class TestStabilizationTracking:
    """Tests for SignalTracker stabilization detection."""

    def test_not_stabilized_during_explosive_growth(self):
        """Tracker should not stabilize during explosive growth phase."""
        tracker = SignalTracker()

        # Explosive growth: 20% improvement per epoch (way above 5% threshold)
        losses = [2.0, 1.6, 1.28, 1.02, 0.82]  # Each ~20% better than prev

        for i, loss in enumerate(losses):
            signals = tracker.update(
                epoch=i,
                global_step=i * 100,
                train_loss=loss,
                train_accuracy=50.0 + i * 5,
                val_loss=loss,
                val_accuracy=50.0 + i * 5,
                active_seeds=[],
            )

        # Should NOT be stabilized after explosive growth
        assert not tracker.is_stabilized
        assert signals.metrics.host_stabilized == 0

    def test_stabilizes_after_consecutive_stable_epochs(self):
        """Tracker should stabilize after STABILIZATION_EPOCHS stable epochs."""
        tracker = SignalTracker()

        # First, a few epochs of moderate improvement (above threshold)
        losses = [2.0, 1.8, 1.6]
        for i, loss in enumerate(losses):
            tracker.update(
                epoch=i,
                global_step=i * 100,
                train_loss=loss,
                train_accuracy=50.0,
                val_loss=loss,
                val_accuracy=50.0,
                active_seeds=[],
            )

        assert not tracker.is_stabilized

        # Now consecutive stable epochs (< 3% improvement)
        # From 1.6, improvements of 2%, 1.5%, 1%
        stable_losses = [1.568, 1.544, 1.529]  # Each < 3% improvement
        for i, loss in enumerate(stable_losses):
            signals = tracker.update(
                epoch=len(losses) + i,
                global_step=(len(losses) + i) * 100,
                train_loss=loss,
                train_accuracy=55.0,
                val_loss=loss,
                val_accuracy=55.0,
                active_seeds=[],
            )

        # Should be stabilized after 3 stable epochs
        assert tracker.is_stabilized
        assert signals.metrics.host_stabilized == 1

    def test_latch_behavior_stays_true(self):
        """Once stabilized, tracker should stay stabilized even if improvement spikes."""
        tracker = SignalTracker()

        # Fast path to stabilization
        tracker._prev_loss = 1.0
        tracker._stable_count = STABILIZATION_EPOCHS - 1
        tracker._is_stabilized = False

        # One more stable epoch triggers stabilization
        tracker.update(
            epoch=10,
            global_step=1000,
            train_loss=0.98,  # 2% improvement (below 3%)
            train_accuracy=60.0,
            val_loss=0.98,
            val_accuracy=60.0,
            active_seeds=[],
        )
        assert tracker.is_stabilized

        # Now a big improvement spike - should NOT reset stabilization
        tracker.update(
            epoch=11,
            global_step=1100,
            train_loss=0.5,  # 48% improvement - huge spike
            train_accuracy=75.0,
            val_loss=0.5,
            val_accuracy=75.0,
            active_seeds=[],
        )

        # Latch: still stabilized
        assert tracker.is_stabilized

    def test_counter_resets_on_unstable_epoch(self):
        """Stable epoch counter should reset if an unstable epoch occurs.

        This tests the counter mechanics: stable epochs accumulate toward
        STABILIZATION_EPOCHS, but an unstable epoch resets the counter.
        """
        tracker = SignalTracker()

        # Set up: simulate 2 stable epochs (need 3 total for stabilization)
        tracker._prev_loss = 1.0
        tracker._stable_count = 2
        tracker._is_stabilized = False

        # Unstable epoch (10% improvement, above 3% threshold) - resets counter
        tracker.update(
            epoch=5, global_step=500,
            train_loss=0.90, train_accuracy=50.0,
            val_loss=0.90, val_accuracy=50.0,
            active_seeds=[],
        )
        # Counter should have reset to 0 because 10% > 3% threshold
        assert tracker._stable_count == 0
        assert not tracker.is_stabilized

        # Now accumulate 3 stable epochs to trigger stabilization
        stable_losses = [0.882, 0.865, 0.848]  # Each ~2% improvement (< 3%)
        for i, loss in enumerate(stable_losses):
            tracker.update(
                epoch=6 + i, global_step=600 + i * 100,
                train_loss=loss, train_accuracy=55.0,
                val_loss=loss, val_accuracy=55.0,
                active_seeds=[],
            )

        assert tracker._stable_count >= STABILIZATION_EPOCHS
        assert tracker.is_stabilized  # Latch triggered after 3 stable epochs

    def test_diverging_loss_not_counted_as_stable(self):
        """Loss spikes (divergence) should not count as stable epochs.

        The sanity check val_loss < prev_loss * 1.5 prevents counting
        diverging training as 'stable'.
        """
        tracker = SignalTracker()
        tracker._prev_loss = 1.0
        tracker._stable_count = 0
        tracker._is_stabilized = False

        # Loss spikes to 2.0 (100% increase) - not stable!
        tracker.update(
            epoch=5, global_step=500,
            train_loss=2.0, train_accuracy=30.0,
            val_loss=2.0, val_accuracy=30.0,
            active_seeds=[],
        )

        # Should not increment stable count
        assert tracker._stable_count == 0
        assert not tracker.is_stabilized

    def test_reset_clears_stabilization(self):
        """Reset should clear stabilization latch and counter."""
        tracker = SignalTracker()

        # Force stabilization
        tracker._is_stabilized = True
        tracker._stable_count = 5

        tracker.reset()

        assert not tracker.is_stabilized
        assert tracker._stable_count == 0

    def test_threshold_boundary(self):
        """Test behavior right at the 3% threshold boundary."""
        tracker = SignalTracker()
        tracker._prev_loss = 1.0
        tracker._stable_count = 0
        tracker._is_stabilized = False

        # Above 3% improvement (5%) - should NOT count as stable
        tracker.update(
            epoch=5, global_step=500,
            train_loss=0.95, train_accuracy=50.0,
            val_loss=0.95, val_accuracy=50.0,
            active_seeds=[],
        )
        assert tracker._stable_count == 0  # 5% is NOT < 3%

        # Below 3% improvement (2%) - should count as stable
        tracker._prev_loss = 1.0
        tracker.update(
            epoch=6, global_step=600,
            train_loss=0.98, train_accuracy=50.0,
            val_loss=0.98, val_accuracy=50.0,
            active_seeds=[],
        )
        assert tracker._stable_count == 1  # 2% is < 3%


class TestStabilizationConstants:
    """Tests to verify stabilization constants are sensible."""

    def test_threshold_is_three_percent(self):
        """Stabilization threshold should be 3% relative improvement."""
        assert STABILIZATION_THRESHOLD == 0.03

    def test_epochs_is_three(self):
        """Stabilization requires 3 consecutive stable epochs.

        This prevents germination during explosive growth phase where seeds
        could get credit for natural training improvements. Re-enabled per
        DRL expert review recommendation.
        """
        assert STABILIZATION_EPOCHS == 3
