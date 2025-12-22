"""Regression tests for Tamiyo components.

These tests document and guard against known bugs and design flaws that have
been identified during development or production use.

Each test includes:
1. Description of the bug/issue
2. Scenario that triggers it
3. Expected behavior vs actual behavior
4. Verification of the fix
"""

import pytest

from esper.leyline import DEFAULT_BLUEPRINT_PENALTY_DECAY
from esper.tamiyo.tracker import SignalTracker
from esper.tamiyo.heuristic import HeuristicTamiyo, HeuristicPolicyConfig


@pytest.mark.tamiyo
class TestStabilizationFirstEpochSkip:
    """Regression: First epoch cannot count toward stabilization.

    BUG CONTEXT:
    The stabilization counter requires prev_loss to be set (not inf) before
    calculating relative improvement. This means the first epoch (epoch 0)
    cannot contribute to stable_count, as prev_loss is still infinity.

    IMPACT:
    - Stabilization requires N+1 epochs instead of N
    - First epoch is effectively wasted for stabilization tracking
    - Documentation claimed "3 epochs" but actually needs 4 total epochs

    FIX STATUS: Working as designed (documented limitation).
    This test documents the behavior to prevent confusion.
    """

    def test_first_epoch_cannot_stabilize(self):
        """First epoch cannot count toward stabilization (prev_loss is inf)."""
        tracker = SignalTracker(
            stabilization_threshold=0.03,
            stabilization_epochs=3,
        )

        # Epoch 0: prev_loss is inf, cannot compute relative improvement
        signals0 = tracker.update(
            epoch=0,
            global_step=0,
            train_loss=2.0,
            train_accuracy=30.0,
            val_loss=2.5,
            val_accuracy=28.0,
            active_seeds=[],
        )
        # Cannot stabilize yet
        assert signals0.metrics.host_stabilized == 0
        assert tracker._stable_count == 0

        # Epoch 1: First epoch where relative improvement can be computed
        # Make small improvement (< 3%)
        signals1 = tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.95,
            train_accuracy=31.0,
            val_loss=2.45,  # 2% improvement from 2.5
            val_accuracy=29.0,
            active_seeds=[],
        )
        # First stable epoch counted
        assert signals1.metrics.host_stabilized == 0
        assert tracker._stable_count == 1

        # Epoch 2: Second stable epoch
        signals2 = tracker.update(
            epoch=2,
            global_step=20,
            train_loss=1.90,
            train_accuracy=32.0,
            val_loss=2.40,
            val_accuracy=30.0,
            active_seeds=[],
        )
        assert signals2.metrics.host_stabilized == 0
        assert tracker._stable_count == 2

        # Epoch 3: Third stable epoch - NOW stabilizes
        signals3 = tracker.update(
            epoch=3,
            global_step=30,
            train_loss=1.85,
            train_accuracy=33.0,
            val_loss=2.35,
            val_accuracy=31.0,
            active_seeds=[],
        )
        # Finally stabilized after 4 total epochs (epochs 0-3)
        assert signals3.metrics.host_stabilized == 1
        assert tracker._stable_count == 3

        # REGRESSION GUARD: Needs 4 total epochs, not 3
        # This documents the off-by-one from infinity initialization

    def test_stabilization_count_accurate_after_first_epoch(self):
        """After first epoch, stabilization counting is accurate."""
        tracker = SignalTracker(
            stabilization_threshold=0.03,
            stabilization_epochs=2,  # Smaller number for clearer test
        )

        # Burn epoch 0 (prev_loss setup)
        tracker.update(
            epoch=0,
            global_step=0,
            train_loss=2.0,
            train_accuracy=30.0,
            val_loss=2.5,
            val_accuracy=28.0,
            active_seeds=[],
        )

        # Epoch 1: First stable
        tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.95,
            train_accuracy=31.0,
            val_loss=2.45,
            val_accuracy=29.0,
            active_seeds=[],
        )
        assert tracker._stable_count == 1

        # Epoch 2: Second stable - stabilizes
        signals = tracker.update(
            epoch=2,
            global_step=20,
            train_loss=1.90,
            train_accuracy=32.0,
            val_loss=2.40,
            val_accuracy=30.0,
            active_seeds=[],
        )
        assert signals.metrics.host_stabilized == 1
        # Accurate: 2 stable epochs required, got 2 (epochs 1-2)

    def test_regression_does_not_increment_stable_count(self):
        """Loss increases should reset stabilization counting instead of advancing it.

        BUG: The old stabilization check only verified relative_improvement < threshold.
        When loss increased (regression), loss_delta was negative, making
        relative_improvement negative, which always passed the < threshold check.

        FIX: Added loss_delta >= 0.0 to require loss actually decreased.
        """
        tracker = SignalTracker(
            stabilization_threshold=0.03,
            stabilization_epochs=2,
        )

        # Initialize previous loss
        tracker.update(
            epoch=0,
            global_step=0,
            train_loss=1.0,
            train_accuracy=50.0,
            val_loss=1.0,
            val_accuracy=50.0,
            active_seeds=[],
        )

        # Regression: validation loss increases, should not count toward stability
        tracker.update(
            epoch=1,
            global_step=10,
            train_loss=1.1,
            train_accuracy=49.0,
            val_loss=1.1,
            val_accuracy=49.0,
            active_seeds=[],
        )
        assert tracker._stable_count == 0
        assert tracker.is_stabilized is False


@pytest.mark.tamiyo
class TestPenaltyDecayEpochNotDecision:
    """Regression: Penalty decay is per-epoch, not per-decision.

    BUG CONTEXT:
    Original implementation decayed penalties on every decide() call, making
    penalties vanish too quickly. With decay=0.5, a penalty of 10.0 would:
    - Old (per-decision): 10 -> 5 -> 2.5 -> 1.25 (gone in 4 decisions)
    - New (per-epoch): 10 -> 5 -> 2.5 -> 1.25 (lasts ~10 epochs with 1 decision/epoch)

    IMPACT:
    - Blueprint penalties decayed too fast
    - Failed blueprints could be retried too quickly
    - Thrashing risk increased

    FIX:
    Added _last_decay_epoch tracking to ensure decay happens once per epoch,
    regardless of how many decisions are made in that epoch.
    """

    def test_penalty_decays_once_per_epoch(self, mock_signals_factory):
        """Penalty should decay only once per epoch, not per decision."""
        config = HeuristicPolicyConfig(
            blueprint_penalty_decay=DEFAULT_BLUEPRINT_PENALTY_DECAY,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Set up penalty
        policy._blueprint_penalties["conv_light"] = 10.0
        policy._last_decay_epoch = -1  # Enable first decay

        # Decision 1 at epoch 5
        signals1 = mock_signals_factory(epoch=5, plateau_epochs=0, host_stabilized=1)
        policy.decide(signals1, [])

        # Should decay: 10.0 * 0.5 = 5.0
        assert policy._blueprint_penalties["conv_light"] == pytest.approx(5.0)

        # Decision 2 at same epoch 5
        signals2 = mock_signals_factory(epoch=5, plateau_epochs=0, host_stabilized=1)
        policy.decide(signals2, [])

        # Should NOT decay again (same epoch)
        assert policy._blueprint_penalties["conv_light"] == pytest.approx(5.0)

        # Decision 3 at epoch 6
        signals3 = mock_signals_factory(epoch=6, plateau_epochs=0, host_stabilized=1)
        policy.decide(signals3, [])

        # Should decay again: 5.0 * 0.5 = 2.5
        assert policy._blueprint_penalties["conv_light"] == pytest.approx(2.5)

    def test_penalty_persists_longer_than_old_implementation(self, mock_signals_factory):
        """Penalties should persist ~10 epochs with per-epoch decay (vs ~4 with per-decision)."""
        config = HeuristicPolicyConfig(
            blueprint_penalty_decay=DEFAULT_BLUEPRINT_PENALTY_DECAY,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        # Start with penalty of 10.0
        policy._blueprint_penalties["conv_light"] = 10.0
        policy._last_decay_epoch = -1

        # Simulate 1 decision per epoch for 10 epochs
        for epoch in range(10):
            signals = mock_signals_factory(epoch=epoch, plateau_epochs=0, host_stabilized=0)
            policy.decide(signals, [])

        # After 10 epochs: 10.0 * (0.5^10) ≈ 0.0098
        # Should be removed (< 0.1 threshold)
        assert "conv_light" not in policy._blueprint_penalties

    def test_multiple_decisions_same_epoch_single_decay(self, mock_signals_factory):
        """Multiple decisions in same epoch should only decay once."""
        config = HeuristicPolicyConfig(
            blueprint_penalty_decay=0.8,  # Higher decay for clarity
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        policy._blueprint_penalties["conv_light"] = 100.0
        policy._last_decay_epoch = -1

        # 5 decisions at epoch 10
        for _ in range(5):
            signals = mock_signals_factory(epoch=10, plateau_epochs=0, host_stabilized=0)
            policy.decide(signals, [])

        # Should decay only once: 100.0 * 0.8 = 80.0
        assert policy._blueprint_penalties["conv_light"] == pytest.approx(80.0)


@pytest.mark.tamiyo
class TestCounterfactualPreferredForFossilize:
    """Regression: HeuristicTamiyo prefers counterfactual contribution when available.

    BUG CONTEXT:
    Original implementation used total_improvement for fossilization decisions,
    which could include credit from natural training progress. This allowed
    "ransomware seeds" to claim credit for improvements they didn't cause.

    DESIGN:
    When a seed reaches HOLDING stage, counterfactual_contribution should
    be available from Kasmina's rollback test. HeuristicTamiyo should prefer
    this over total_improvement for more accurate credit assignment.

    BEHAVIOR:
    - If counterfactual_contribution is not None: Use it (true causal impact)
    - If counterfactual_contribution is None: Fall back to total_improvement

    This test verifies the preference logic works correctly.
    """

    def test_counterfactual_preferred_over_total(
        self, mock_signals_factory, mock_seed_factory
    ):
        """Should use counterfactual when available, not total_improvement."""
        from esper.leyline import SeedStage

        policy = HeuristicTamiyo(topology="cnn")

        # Seed with both counterfactual and total improvement
        # counterfactual=2.0 (true impact), total=5.0 (includes natural progress)
        seed = mock_seed_factory(
            seed_id="test_seed",
            stage=SeedStage.HOLDING,
            improvement=0.0,  # stage-specific improvement
            total=5.0,         # total_improvement (inflated)
            counterfactual=2.0,  # counterfactual_contribution (accurate)
        )

        signals = mock_signals_factory(epoch=10)
        decision = policy.decide(signals, [seed])

        # Should fossilize based on counterfactual (2.0 > 0.0 threshold)
        assert decision.action.name == "FOSSILIZE"
        # Reason should mention counterfactual value
        assert "2.0" in decision.reason or "2.00" in decision.reason

    def test_total_improvement_fallback_when_no_counterfactual(
        self, mock_signals_factory, mock_seed_factory
    ):
        """Should fall back to total_improvement when counterfactual is None."""
        from esper.leyline import SeedStage

        policy = HeuristicTamiyo(topology="cnn")

        # Seed with no counterfactual (early implementation, or test mode)
        seed = mock_seed_factory(
            seed_id="test_seed",
            stage=SeedStage.HOLDING,
            improvement=0.0,
            total=3.5,          # total_improvement
            counterfactual=None,  # Not available
        )

        signals = mock_signals_factory(epoch=10)
        decision = policy.decide(signals, [seed])

        # Should fossilize using total_improvement fallback
        assert decision.action.name == "FOSSILIZE"
        assert "3.5" in decision.reason or "3.50" in decision.reason

    def test_negative_counterfactual_triggers_cull(
        self, mock_signals_factory, mock_seed_factory
    ):
        """Negative counterfactual should trigger cull even if total is positive."""
        from esper.leyline import SeedStage

        policy = HeuristicTamiyo(topology="cnn")

        # Ransomware seed: total looks good, but counterfactual reveals harm
        seed = mock_seed_factory(
            seed_id="ransomware_seed",
            stage=SeedStage.HOLDING,
            improvement=0.0,
            total=4.0,           # Looks good (misleading)
            counterfactual=-1.5,  # Actually harmful
        )

        signals = mock_signals_factory(epoch=10)
        decision = policy.decide(signals, [seed])

        # Should cull based on negative counterfactual
        assert decision.action.name == "PRUNE"
        assert decision.target_seed_id == "ransomware_seed"
        # Reason should show the negative contribution
        assert "-1.5" in decision.reason or "-1.50" in decision.reason

    def test_counterfactual_none_with_positive_total_fossilizes(
        self, mock_signals_factory, mock_seed_factory
    ):
        """When counterfactual is None, positive total should fossilize."""
        from esper.leyline import SeedStage

        config = HeuristicPolicyConfig(
            min_improvement_to_fossilize=1.0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        seed = mock_seed_factory(
            seed_id="test_seed",
            stage=SeedStage.HOLDING,
            improvement=0.0,
            total=2.5,
            counterfactual=None,  # Not available
        )

        signals = mock_signals_factory(epoch=10)
        decision = policy.decide(signals, [seed])

        # Should fossilize (total=2.5 >= threshold=1.0)
        assert decision.action.name == "FOSSILIZE"

    def test_both_below_threshold_culls(
        self, mock_signals_factory, mock_seed_factory
    ):
        """When both metrics are below threshold, should cull."""
        from esper.leyline import SeedStage

        config = HeuristicPolicyConfig(
            min_improvement_to_fossilize=2.0,
        )
        policy = HeuristicTamiyo(config=config, topology="cnn")

        seed = mock_seed_factory(
            seed_id="test_seed",
            stage=SeedStage.HOLDING,
            improvement=0.0,
            total=1.5,           # Below threshold
            counterfactual=1.0,  # Below threshold
        )

        signals = mock_signals_factory(epoch=10)
        decision = policy.decide(signals, [seed])

        # Should cull (counterfactual=1.0 < threshold=2.0)
        assert decision.action.name == "PRUNE"
