"""Tests for stabilization tracking in SignalTracker.

Stabilization gating ensures seeds only get credit for improvements AFTER
the explosive growth phase ends. The tracker monitors relative loss improvement
and flips a latch once improvement drops below 5% for 3 consecutive epochs.
"""


from esper.tamiyo.tracker import SignalTracker
from esper.leyline import DEFAULT_STABILIZATION_THRESHOLD, DEFAULT_STABILIZATION_EPOCHS


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
        tracker._stable_count = DEFAULT_STABILIZATION_EPOCHS - 1
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

        assert tracker._stable_count >= DEFAULT_STABILIZATION_EPOCHS
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
        assert DEFAULT_STABILIZATION_THRESHOLD == 0.03

    def test_epochs_is_three(self):
        """Stabilization requires 3 consecutive stable epochs.

        This prevents germination during explosive growth phase where seeds
        could get credit for natural training improvements. Re-enabled per
        DRL expert review recommendation.
        """
        assert DEFAULT_STABILIZATION_EPOCHS == 3


class TestSymmetricStabilityWindow:
    """Tests for B9-DRL-02 symmetric stability window fix.

    The old logic used `loss_delta >= 0` which was too strict - any tiny
    regression (even normal PPO noise) would reset the stable counter.

    The new logic uses symmetric thresholds:
    - Explosive growth (>3% improvement): NOT stable
    - Divergence (>5% regression): NOT stable
    - Plateau/noise (-5% to +3%): stable
    """

    def test_small_regression_counts_as_stable(self):
        """Small regression (<5%) should count as stable (normal RL noise).

        This is the key fix from B9-DRL-02: PPO training has stochastic
        variance, so +2% loss fluctuation is normal and shouldn't block
        germination indefinitely.
        """
        tracker = SignalTracker()
        tracker._prev_loss = 1.0
        tracker._stable_count = 0
        tracker._is_stabilized = False

        # 2% regression (loss increased by 2%) - should count as stable
        tracker.update(
            epoch=5, global_step=500,
            train_loss=1.02, train_accuracy=50.0,
            val_loss=1.02, val_accuracy=50.0,  # +2% loss
            active_seeds=[],
        )
        assert tracker._stable_count == 1, (
            "Small regression (2%) should count as stable (PPO noise tolerance)"
        )

    def test_large_regression_resets_counter(self):
        """Large regression (>5%) should NOT count as stable (divergence).

        The regression_threshold (default 5%) catches diverging training
        without being too sensitive to normal noise.
        """
        tracker = SignalTracker()
        tracker._prev_loss = 1.0
        tracker._stable_count = 2  # Already had 2 stable epochs
        tracker._is_stabilized = False

        # 10% regression (loss increased by 10%) - should NOT count, reset
        tracker.update(
            epoch=5, global_step=500,
            train_loss=1.10, train_accuracy=45.0,
            val_loss=1.10, val_accuracy=45.0,  # +10% loss (divergence)
            active_seeds=[],
        )
        assert tracker._stable_count == 0, (
            "Large regression (10%) should reset stable counter (divergence)"
        )

    def test_boundary_at_five_percent_regression(self):
        """Test behavior at exactly 5% regression boundary."""
        tracker = SignalTracker()
        tracker._prev_loss = 1.0
        tracker._stable_count = 0
        tracker._is_stabilized = False

        # Exactly at 5% regression boundary - just outside stable window
        # relative_improvement = (1.0 - 1.05) / 1.0 = -0.05
        # Need > -0.05, so -0.05 is NOT stable (boundary exclusive)
        tracker.update(
            epoch=5, global_step=500,
            train_loss=1.05, train_accuracy=50.0,
            val_loss=1.05, val_accuracy=50.0,  # Exactly 5% regression
            active_seeds=[],
        )
        assert tracker._stable_count == 0, (
            "Exactly 5% regression should NOT count as stable (boundary)"
        )

        # Just under 5% should count
        tracker._prev_loss = 1.0
        tracker.update(
            epoch=6, global_step=600,
            train_loss=1.049, train_accuracy=50.0,
            val_loss=1.049, val_accuracy=50.0,  # 4.9% regression
            active_seeds=[],
        )
        assert tracker._stable_count == 1, (
            "4.9% regression should count as stable (just inside window)"
        )

    def test_unchanged_loss_counts_as_stable(self):
        """Unchanged loss (delta=0) should count as stable.

        This was ambiguous in the old code - now it's clearly in the
        stable window (-5% to +3%).
        """
        tracker = SignalTracker()
        tracker._prev_loss = 1.0
        tracker._stable_count = 0
        tracker._is_stabilized = False

        # Unchanged loss - relative_improvement = 0
        tracker.update(
            epoch=5, global_step=500,
            train_loss=1.0, train_accuracy=50.0,
            val_loss=1.0, val_accuracy=50.0,  # No change
            active_seeds=[],
        )
        assert tracker._stable_count == 1, (
            "Unchanged loss should count as stable (plateau)"
        )

    def test_stabilization_with_noise_tolerance(self):
        """Should stabilize even with normal PPO noise fluctuations.

        Real RL training has epochs where loss bounces slightly up and down.
        The symmetric window allows this while still detecting stabilization.
        """
        tracker = SignalTracker()

        # Initial epoch
        tracker.update(
            epoch=0, global_step=0,
            train_loss=1.0, train_accuracy=50.0,
            val_loss=1.0, val_accuracy=50.0,
            active_seeds=[],
        )

        # Noisy but stable epochs: -1%, +2%, -1.5% (all within -5% to +3%)
        noisy_losses = [0.99, 1.01, 0.995]  # Realistic PPO fluctuation
        for i, loss in enumerate(noisy_losses):
            tracker.update(
                epoch=i + 1, global_step=(i + 1) * 100,
                train_loss=loss, train_accuracy=52.0,
                val_loss=loss, val_accuracy=52.0,
                active_seeds=[],
            )

        # Should have stabilized after 3 "noisy but stable" epochs
        assert tracker.is_stabilized, (
            "Should stabilize with noisy but bounded loss fluctuations"
        )

    def test_custom_regression_threshold(self):
        """Should respect custom regression_threshold parameter."""
        tracker = SignalTracker(regression_threshold=0.02)  # Stricter: 2%

        tracker._prev_loss = 1.0
        tracker._stable_count = 0
        tracker._is_stabilized = False

        # 3% regression - would be stable with default 5%, but not with 2%
        tracker.update(
            epoch=5, global_step=500,
            train_loss=1.03, train_accuracy=50.0,
            val_loss=1.03, val_accuracy=50.0,
            active_seeds=[],
        )
        assert tracker._stable_count == 0, (
            "3% regression should NOT count with regression_threshold=2%"
        )
