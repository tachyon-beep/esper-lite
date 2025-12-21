"""Edge case tests for Tamiyo components.

These tests verify behavior at boundaries and extreme values to ensure
robustness and graceful handling of unusual conditions.
"""

import pytest
import math

from esper.tamiyo.tracker import SignalTracker
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig
from esper.leyline import SeedStage


@pytest.mark.tamiyo
class TestFirstEpochHandling:
    """Tests for epoch 0 and early training behavior."""

    def test_first_epoch_no_crash(self, signal_tracker):
        """Should handle epoch 0 without crashes."""
        # First update at epoch 0 should work
        signals = signal_tracker.update(
            epoch=0,
            global_step=0,
            train_loss=2.5,
            train_accuracy=10.0,
            val_loss=2.8,
            val_accuracy=8.0,
            active_seeds=[],
        )

        assert signals is not None
        assert signals.metrics.epoch == 0
        assert signals.metrics.val_loss == 2.8
        assert signals.metrics.val_accuracy == 8.0

    def test_first_epoch_delta_computation(self, signal_tracker):
        """First epoch should have sensible deltas (comparing to infinity)."""
        signals = signal_tracker.update(
            epoch=0,
            global_step=0,
            train_loss=2.0,
            train_accuracy=20.0,
            val_loss=2.5,
            val_accuracy=18.0,
            active_seeds=[],
        )

        # loss_delta = prev_loss - current_loss
        # prev_loss starts at float('inf'), so delta = inf - 2.5 = inf
        assert signals.metrics.loss_delta == float('inf')
        # accuracy_delta = current - prev (prev starts at 0)
        assert signals.metrics.accuracy_delta == 18.0

    def test_policy_first_epoch_no_germination(self, mock_signals_factory):
        """Policy should not germinate at epoch 0 (too early)."""
        policy = HeuristicTamiyo(topology="cnn")

        signals = mock_signals_factory(
            epoch=0,
            plateau_epochs=10,  # Even with plateau
            host_stabilized=1,   # And stabilized
        )

        decision = policy.decide(signals, [])

        # Should wait (below min_epochs_before_germinate=5)
        assert decision.action.name == "WAIT"
        assert "Too early" in decision.reason


@pytest.mark.tamiyo
class TestZeroLossHandling:
    """Tests for division by zero protection."""

    def test_zero_loss_no_division_error(self, signal_tracker):
        """Should handle zero loss gracefully in stabilization check."""
        # First update with normal loss
        signal_tracker.update(
            epoch=0,
            global_step=0,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )

        # Update with zero loss (perfect model!)
        signals = signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=0.0,
            train_accuracy=100.0,
            val_loss=0.0,
            val_accuracy=100.0,
            active_seeds=[],
        )

        # Should not crash, should have valid signals
        assert signals is not None
        assert signals.metrics.val_loss == 0.0
        # Division by zero is protected by EPS check (prev_loss > 1e-8)

    def test_very_small_loss_stabilization(self):
        """Should handle very small losses in stabilization calculation."""
        tracker = SignalTracker()

        # Start with small loss
        tracker.update(
            epoch=0,
            global_step=0,
            train_loss=1e-7,
            train_accuracy=99.9,
            val_loss=1e-7,
            val_accuracy=99.9,
            active_seeds=[],
        )

        # Even smaller improvement
        signals = tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1e-8,
            train_accuracy=99.99,
            val_loss=1e-8,
            val_accuracy=99.99,
            active_seeds=[],
        )

        # Should not crash from division issues
        assert signals is not None
        # EPS=1e-8 protects against division by values <= 1e-8


@pytest.mark.tamiyo
class TestInfiniteLossHandling:
    """Tests for infinite/NaN loss values."""

    def test_infinite_loss_graceful_handling(self, signal_tracker):
        """Should handle infinite loss without crashing."""
        # Normal start
        signal_tracker.update(
            epoch=0,
            global_step=0,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.2,
            val_accuracy=48.0,
            active_seeds=[],
        )

        # Infinite loss (divergence!)
        signals = signal_tracker.update(
            epoch=1,
            global_step=10,
            train_loss=float('inf'),
            train_accuracy=0.0,
            val_loss=float('inf'),
            val_accuracy=0.0,
            active_seeds=[],
        )

        assert signals is not None
        assert signals.metrics.val_loss == float('inf')
        # loss_delta = 1.2 - inf = -inf
        assert signals.metrics.loss_delta == -float('inf')

    def test_nan_loss_propagates(self, signal_tracker):
        """NaN losses should propagate (detector of training issues)."""
        signal_tracker.update(
            epoch=0,
            global_step=0,
            train_loss=float('nan'),
            train_accuracy=float('nan'),
            val_loss=float('nan'),
            val_accuracy=float('nan'),
            active_seeds=[],
        )

        # Should accept NaN (up to caller to check for training collapse)
        assert len(signal_tracker._loss_history) == 1
        assert math.isnan(signal_tracker._loss_history[0])


@pytest.mark.tamiyo
class TestEmptyActiveSeedsHandling:
    """Tests for empty active seeds list."""

    def test_empty_active_seeds_correct_behavior(self, mock_signals_factory):
        """Empty seeds list should trigger germination logic."""
        policy = HeuristicTamiyo(topology="cnn")

        signals = mock_signals_factory(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=1,
        )

        decision = policy.decide(signals, [])

        # Should try to germinate (no active seeds)
        assert decision.action.name.startswith("GERMINATE_")

    def test_empty_seeds_in_tracker(self, signal_tracker):
        """Tracker should handle empty seeds list correctly."""
        signals = signal_tracker.update(
            epoch=5,
            global_step=50,
            train_loss=0.8,
            train_accuracy=75.0,
            val_loss=0.9,
            val_accuracy=72.0,
            active_seeds=[],  # Empty
        )

        # Should have default seed values
        assert signals.active_seeds == []
        assert signals.seed_stage == 0
        assert signals.seed_epochs_in_stage == 0
        assert signals.seed_alpha == 0.0
        assert signals.seed_improvement == 0.0


@pytest.mark.tamiyo
class TestMultipleActiveSeedsHandling:
    """Tests for multiple seeds in active_seeds list."""

    def test_multiple_active_seeds_first_considered(
        self, mock_signals_factory, mock_seed_factory
    ):
        """Policy should only consider first seed in list."""

        policy = HeuristicTamiyo(topology="cnn")

        # Create two seeds: first is fine, second is failing
        seed1 = mock_seed_factory(
            seed_id="seed_1",
            stage=SeedStage.TRAINING,
            epochs_in_stage=2,
            improvement=1.5,  # Good
        )
        seed2 = mock_seed_factory(
            seed_id="seed_2",
            stage=SeedStage.TRAINING,
            epochs_in_stage=10,
            improvement=-5.0,  # Terrible
        )

        signals = mock_signals_factory(epoch=10)

        decision = policy.decide(signals, [seed1, seed2])

        # Should wait for seed1 (not cull seed2)
        assert decision.action.name == "WAIT"
        assert decision.target_seed_id == "seed_1"

    def test_tracker_summary_seed_selection_is_deterministic(self, signal_tracker, mock_seed_factory):
        """Tracker should select a deterministic summary seed from multiple slots."""

        seed1 = mock_seed_factory(
            seed_id="first",
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            improvement=2.0,
        )
        seed2 = mock_seed_factory(
            seed_id="second",
            stage=SeedStage.BLENDING,
            epochs_in_stage=10,
            improvement=5.0,
        )

        signals = signal_tracker.update(
            epoch=10,
            global_step=100,
            train_loss=0.5,
            train_accuracy=85.0,
            val_loss=0.6,
            val_accuracy=83.0,
            active_seeds=[seed1, seed2],
        )

        # Highest stage wins (BLENDING > TRAINING)
        assert signals.active_seeds == ["first", "second"]
        assert signals.seed_stage == int(SeedStage.BLENDING)
        assert signals.seed_epochs_in_stage == 10
        assert signals.seed_improvement == 5.0

    def test_tracker_summary_seed_tiebreaks_on_alpha(self, signal_tracker, mock_seed_factory) -> None:
        """When stages tie, higher alpha wins for summary seed selection."""

        seed_low_alpha = mock_seed_factory(
            seed_id="low_alpha",
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            alpha=0.2,
            improvement=1.0,
        )
        seed_high_alpha = mock_seed_factory(
            seed_id="high_alpha",
            stage=SeedStage.TRAINING,
            epochs_in_stage=3,
            alpha=0.7,
            improvement=2.0,
        )

        signals = signal_tracker.update(
            epoch=10,
            global_step=100,
            train_loss=0.5,
            train_accuracy=85.0,
            val_loss=0.6,
            val_accuracy=83.0,
            active_seeds=[seed_low_alpha, seed_high_alpha],
        )

        assert signals.seed_stage == int(SeedStage.TRAINING)
        assert signals.seed_alpha == 0.7
        assert signals.seed_epochs_in_stage == 3
        assert signals.seed_improvement == 2.0

    def test_tracker_summary_seed_tiebreaks_on_counterfactual(self, signal_tracker, mock_seed_factory) -> None:
        """When stage+alpha tie, most negative counterfactual wins (safety)."""

        seed_hurting_more = mock_seed_factory(
            seed_id="hurt_more",
            stage=SeedStage.HOLDING,
            epochs_in_stage=2,
            alpha=0.5,
            improvement=0.1,
            counterfactual=-1.0,
        )
        seed_hurting_less = mock_seed_factory(
            seed_id="hurt_less",
            stage=SeedStage.HOLDING,
            epochs_in_stage=5,
            alpha=0.5,
            improvement=0.2,
            counterfactual=-0.1,
        )

        signals = signal_tracker.update(
            epoch=10,
            global_step=100,
            train_loss=0.5,
            train_accuracy=85.0,
            val_loss=0.6,
            val_accuracy=83.0,
            active_seeds=[seed_hurting_less, seed_hurting_more],
        )

        assert signals.seed_stage == int(SeedStage.HOLDING)
        assert signals.seed_alpha == 0.5
        assert signals.seed_epochs_in_stage == 2
        assert signals.seed_improvement == 0.1


@pytest.mark.tamiyo
class TestTerminalStageFiltering:
    """Tests for filtering terminal stage seeds."""

    def test_fossilized_seeds_filtered(self, mock_signals_factory, mock_seed_factory):
        """FOSSILIZED seeds should be filtered out as terminal."""

        policy = HeuristicTamiyo(topology="cnn")

        fossilized = mock_seed_factory(
            stage=SeedStage.FOSSILIZED,
            improvement=10.0,
        )

        signals = mock_signals_factory(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=1,
        )

        decision = policy.decide(signals, [fossilized])

        # Should treat as no live seeds and try to germinate
        assert decision.action.name.startswith("GERMINATE_")

    def test_pruned_seeds_filtered(self, mock_signals_factory, mock_seed_factory):
        """PRUNED seeds should be filtered out as failure stage."""

        policy = HeuristicTamiyo(topology="cnn")

        pruned = mock_seed_factory(
            stage=SeedStage.PRUNED,
            improvement=-10.0,
        )

        signals = mock_signals_factory(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=1,
        )

        decision = policy.decide(signals, [pruned])

        # Should treat as no live seeds
        assert decision.action.name.startswith("GERMINATE_")

    def test_mixed_terminal_and_active_seeds(
        self, mock_signals_factory, mock_seed_factory
    ):
        """Should filter out terminal but consider active seeds."""

        policy = HeuristicTamiyo(topology="cnn")

        # Mix of terminal and active
        fossilized = mock_seed_factory(stage=SeedStage.FOSSILIZED)
        pruned = mock_seed_factory(stage=SeedStage.PRUNED)
        active = mock_seed_factory(
            seed_id="active_one",
            stage=SeedStage.TRAINING,
            epochs_in_stage=2,
            improvement=1.0,
        )

        signals = mock_signals_factory(epoch=10)

        # Order: terminal, active, terminal
        decision = policy.decide(signals, [fossilized, active, pruned])

        # Should filter terminals and only see active_one
        assert decision.action.name == "WAIT"
        assert decision.target_seed_id == "active_one"


@pytest.mark.tamiyo
class TestVeryLongTraining:
    """Tests for very large epoch numbers."""

    def test_epoch_10000_no_overflow(self, signal_tracker):
        """Should handle very large epoch numbers without issues."""
        signals = signal_tracker.update(
            epoch=10000,
            global_step=1000000,
            train_loss=0.01,
            train_accuracy=99.0,
            val_loss=0.02,
            val_accuracy=98.5,
            active_seeds=[],
        )

        assert signals.metrics.epoch == 10000
        assert signals.metrics.global_step == 1000000

    def test_policy_at_large_epoch(self, mock_signals_factory):
        """Policy should work normally at large epochs."""
        policy = HeuristicTamiyo(topology="cnn")

        signals = mock_signals_factory(
            epoch=50000,
            plateau_epochs=10,
            host_stabilized=1,
        )

        decision = policy.decide(signals, [])

        # Should still make valid decisions
        assert decision.action.name.startswith("GERMINATE_")

    def test_embargo_with_large_epochs(self, mock_signals_factory):
        """Embargo calculation should work with large epochs."""
        config = HeuristicPolicyConfig(embargo_epochs_after_prune=5)
        policy = HeuristicTamiyo(config=config, topology="cnn")

        policy._last_prune_epoch = 100000

        signals = mock_signals_factory(
            epoch=100002,  # 2 epochs after cull
            plateau_epochs=10,
            host_stabilized=1,
        )

        decision = policy.decide(signals, [])

        # Should still be in embargo
        assert decision.action.name == "WAIT"
        assert "Embargo" in decision.reason


@pytest.mark.tamiyo
class TestConfigEdgeValues:
    """Tests for edge values in configuration."""

    def test_embargo_zero_no_blocking(self, mock_signals_factory, mock_seed_factory):
        """embargo=0 should not block germination."""

        config = HeuristicPolicyConfig(
            embargo_epochs_after_prune=0,  # No embargo
            prune_after_epochs_without_improvement=1,
            prune_if_accuracy_drops_by=1.0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Cull a seed
        seed = mock_seed_factory(
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            improvement=-3.0,
        )
        signals1 = mock_signals_factory(epoch=10)
        decision1 = policy.decide(signals1, [seed])
        assert decision1.action.name == "PRUNE"

        # Immediately try to germinate (same epoch)
        signals2 = mock_signals_factory(
            epoch=10,
            plateau_epochs=5,
            host_stabilized=1,
        )
        decision2 = policy.decide(signals2, [])

        # Should germinate (no embargo)
        assert decision2.action.name.startswith("GERMINATE_")

    def test_plateau_one_epoch_germination(self, mock_signals_factory):
        """plateau=1 should germinate after 1 plateau epoch."""
        config = HeuristicPolicyConfig(
            plateau_epochs_to_germinate=1,
            min_epochs_before_germinate=0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        signals = mock_signals_factory(
            epoch=5,
            plateau_epochs=1,
            host_stabilized=1,
        )

        decision = policy.decide(signals, [])

        # Should germinate with just 1 plateau epoch
        assert decision.action.name.startswith("GERMINATE_")

    def test_min_epochs_zero_allows_early_germination(self, mock_signals_factory):
        """min_epochs=0 should allow germination from epoch 0."""
        config = HeuristicPolicyConfig(
            min_epochs_before_germinate=0,
            plateau_epochs_to_germinate=1,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        signals = mock_signals_factory(
            epoch=0,
            plateau_epochs=1,
            host_stabilized=1,
        )

        decision = policy.decide(signals, [])

        # Should allow germination at epoch 0
        assert decision.action.name.startswith("GERMINATE_")

    def test_cull_threshold_zero_culls_any_drop(
        self, mock_signals_factory, mock_seed_factory
    ):
        """prune_if_accuracy_drops_by=0 should cull on any negative improvement."""

        config = HeuristicPolicyConfig(
            prune_if_accuracy_drops_by=0.0,  # Cull on any drop
            prune_after_epochs_without_improvement=1,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Tiny negative improvement
        seed = mock_seed_factory(
            stage=SeedStage.TRAINING,
            epochs_in_stage=5,
            improvement=-0.01,  # Barely negative
        )

        signals = mock_signals_factory(epoch=10)
        decision = policy.decide(signals, [seed])

        # Should cull (improvement < -0.0)
        assert decision.action.name == "PRUNE"

    def test_stabilization_epochs_zero_disables_gate(self):
        """stabilization_epochs=0 should disable stabilization gating."""
        tracker = SignalTracker(
            stabilization_threshold=0.03,
            stabilization_epochs=0,  # Disabled
        )

        # First update
        signals = tracker.update(
            epoch=0,
            global_step=0,
            train_loss=2.0,
            train_accuracy=30.0,
            val_loss=2.5,
            val_accuracy=28.0,
            active_seeds=[],
        )

        # Should be stabilized immediately (disabled gate)
        # Note: The tracker still checks if _stable_count >= stabilization_epochs
        # With stabilization_epochs=0, this is True on first epoch with valid prev_loss
        # But first epoch has prev_loss=inf, so let's do second epoch with small improvement
        signals = tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.95,
            train_accuracy=31.0,
            val_loss=2.45,  # Small improvement (2% < 3% threshold)
            val_accuracy=29.0,
            active_seeds=[],
        )

        # With 0 epochs required, should stabilize immediately
        # (stable_count=1 >= 0) after first "stable" epoch
        # Actually, with 0 required, even stable_count=0 satisfies the condition
        # But we need at least one stable epoch to trigger the latch
        # Check: After first stable epoch with epochs=0, should be stabilized
        # Actually on review: stable_count starts at 0, and 0 >= 0 is True
        # But the logic only checks after incrementing stable_count in a stable epoch
        # So we need to trigger is_stable_epoch condition first
        # With the small improvement above, this should be a stable epoch
        # stable_count increments to 1, then 1 >= 0 is True -> stabilizes
        assert signals.metrics.host_stabilized == 1
