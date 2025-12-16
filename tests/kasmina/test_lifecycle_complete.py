"""Complete lifecycle tests for Kasmina seed progression.

Tests verify correct behavior through complete lifecycles:
- Full happy path: DORMANT → FOSSILIZED
- Early cull scenarios
- Probation timeout behavior
- Dwell epoch enforcement
- Blending progress tracking
"""

import pytest
import torch

from esper.kasmina.slot import SeedSlot, SeedState, SeedMetrics, QualityGates
from esper.kasmina.host import CNNHost
from esper.leyline import (
    SeedStage,
    GateLevel,
    DEFAULT_MIN_TRAINING_IMPROVEMENT,
    DEFAULT_MIN_BLENDING_EPOCHS,
    DEFAULT_ALPHA_COMPLETE_THRESHOLD,
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    DEFAULT_GRADIENT_RATIO_THRESHOLD,
    DEFAULT_MAX_PROBATION_EPOCHS,
)


class TestFullLifecycleHappyPath:
    """Tests for complete seed lifecycle traversal."""

    def test_lifecycle_dormant_to_germinated(self):
        """Germination should transition DORMANT → GERMINATED."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        state = slot.germinate("noop", seed_id="test")

        assert state.stage == SeedStage.GERMINATED
        assert state.seed_id == "test"
        assert state.blueprint_id == "noop"

    def test_lifecycle_germinated_to_training(self):
        """step_epoch() should advance GERMINATED → TRAINING."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        slot.step_epoch()

        assert slot.state.stage == SeedStage.TRAINING

    def test_lifecycle_training_to_blending(self):
        """step_epoch() should advance TRAINING → BLENDING when gate passes."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.step_epoch()  # → TRAINING

        # Simulate conditions for G2 gate
        slot.state.metrics.record_accuracy(50.0)
        slot.state.metrics.record_accuracy(60.0)  # Improvement >= threshold

        # Set gradient ratio to pass G2
        slot.state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1

        # Need enough epochs in TRAINING
        for _ in range(DEFAULT_MIN_BLENDING_EPOCHS):
            slot.state.metrics.epochs_in_current_stage += 1

        slot.step_epoch()

        assert slot.state.stage == SeedStage.BLENDING

    def test_lifecycle_blending_to_probationary(self):
        """step_epoch() should advance BLENDING → PROBATIONARY when complete."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)

        # Set up for BLENDING transition
        slot.state.metrics.record_accuracy(50.0)
        slot.state.metrics.record_accuracy(60.0)
        slot.state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1
        for _ in range(DEFAULT_MIN_BLENDING_EPOCHS):
            slot.state.metrics.epochs_in_current_stage += 1

        # Manually transition to BLENDING (to avoid full setup)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=5)

        # Simulate blending completion (record accuracy to accumulate epochs_in_current_stage for G3)
        for _ in range(5):
            slot.state.metrics.record_accuracy(65.0)  # G3 needs epochs_in_current_stage
            slot.step_epoch()

        assert slot.state.stage == SeedStage.PROBATIONARY

    def test_lifecycle_probationary_to_fossilized(self):
        """advance_stage() should transition PROBATIONARY → FOSSILIZED when gate passes."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        # Progress through stages
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.PROBATIONARY)
        slot.state.alpha = 1.0

        # Set up conditions for G5 gate
        slot.state.metrics.counterfactual_contribution = DEFAULT_MIN_FOSSILIZE_CONTRIBUTION + 1.0
        slot.state.is_healthy = True

        result = slot.advance_stage(SeedStage.FOSSILIZED)

        assert result.passed
        assert slot.state.stage == SeedStage.FOSSILIZED


class TestEarlyCullScenarios:
    """Tests for early cull behavior."""

    def test_cull_from_training(self):
        """Culling from TRAINING should succeed."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)

        result = slot.cull(reason="test_cull")

        assert result is True
        assert slot.state is None
        assert slot.seed is None

    def test_cull_from_blending(self):
        """Culling from BLENDING should succeed."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        result = slot.cull(reason="test_cull")

        assert result is True
        assert slot.state is None

    def test_cull_from_probationary(self):
        """Culling from PROBATIONARY should succeed."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.PROBATIONARY)

        result = slot.cull(reason="test_cull")

        assert result is True
        assert slot.state is None

    def test_cull_fossilized_fails(self):
        """Culling FOSSILIZED seed should fail (permanent)."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        # Force to FOSSILIZED
        slot.state.stage = SeedStage.FOSSILIZED

        result = slot.cull(reason="test_cull")

        assert result is False
        assert slot.state is not None
        assert slot.state.stage == SeedStage.FOSSILIZED


class TestProbationTimeout:
    """Tests for probation timeout behavior."""

    def test_probation_timeout_culls_seed(self):
        """Exceeding max probation epochs should auto-cull."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.PROBATIONARY)

        # Simulate epochs in probation exceeding timeout
        # Set counterfactual to non-negative so safety cull doesn't trigger first
        slot.state.metrics.counterfactual_contribution = 0.1
        slot.state.metrics.epochs_in_current_stage = DEFAULT_MAX_PROBATION_EPOCHS

        slot.step_epoch()

        # Should be culled due to timeout
        assert slot.state is None

    def test_probation_no_timeout_before_max_epochs(self):
        """Seed should survive if under max probation epochs."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.PROBATIONARY)

        # Just under timeout
        slot.state.metrics.counterfactual_contribution = 0.1
        slot.state.metrics.epochs_in_current_stage = DEFAULT_MAX_PROBATION_EPOCHS - 1

        slot.step_epoch()

        # Should still be in probation
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PROBATIONARY


class TestNegativeCounterfactualAutoCull:
    """Tests for negative counterfactual auto-cull behavior."""

    def test_negative_counterfactual_culls(self):
        """Negative counterfactual in PROBATIONARY should auto-cull."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.PROBATIONARY)

        # Set negative counterfactual
        slot.state.metrics.counterfactual_contribution = -1.0

        slot.step_epoch()

        # Should be culled
        assert slot.state is None

    def test_zero_counterfactual_culls(self):
        """Zero counterfactual in PROBATIONARY should auto-cull."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.PROBATIONARY)

        slot.state.metrics.counterfactual_contribution = 0.0

        slot.step_epoch()

        # Should be culled (0 is not positive contribution)
        assert slot.state is None

    def test_positive_counterfactual_survives(self):
        """Positive counterfactual should not trigger auto-cull."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.PROBATIONARY)

        slot.state.metrics.counterfactual_contribution = 1.0

        slot.step_epoch()

        # Should still be in probation
        assert slot.state is not None


class TestG2GateBehavior:
    """Tests for G2 gate requirements."""

    def test_g2_fails_without_gradient_telemetry(self):
        """G2 gate should fail when seed_gradient_norm_ratio is 0."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        # No gradient telemetry captured - ratio stays at 0
        state.metrics.seed_gradient_norm_ratio = 0.0
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(60.0)  # Good improvement

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert not result.passed
        assert any("gradient" in check for check in result.checks_failed)

    def test_g2_passes_with_gradient_telemetry(self):
        """G2 gate should pass when gradient ratio meets threshold."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        # Set conditions for passing
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(60.0)  # >= DEFAULT_MIN_TRAINING_IMPROVEMENT
        state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1
        state.metrics.epochs_in_current_stage = DEFAULT_MIN_BLENDING_EPOCHS

        result = gates.check_gate(state, SeedStage.BLENDING)

        assert result.passed


class TestG5GateBehavior:
    """Tests for G5 gate requirements."""

    def test_g5_fails_without_counterfactual(self):
        """G5 gate should fail when counterfactual is None."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.PROBATIONARY,
        )

        # No counterfactual set
        state.metrics.counterfactual_contribution = None
        state.is_healthy = True

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert "counterfactual_not_available" in result.checks_failed

    def test_g5_fails_with_low_counterfactual(self):
        """G5 gate should fail when counterfactual below threshold."""
        gates = QualityGates()
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.PROBATIONARY,
        )

        # Below minimum threshold
        state.metrics.counterfactual_contribution = DEFAULT_MIN_FOSSILIZE_CONTRIBUTION - 0.1
        state.is_healthy = True

        result = gates.check_gate(state, SeedStage.FOSSILIZED)

        assert not result.passed
        assert any("insufficient_contribution" in check for check in result.checks_failed)


class TestDwellEpochEnforcement:
    """Tests for dwell epoch requirements."""

    def test_training_dwell_prevents_early_transition(self):
        """TRAINING → BLENDING should not happen before dwell epochs."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.step_epoch()  # → TRAINING

        # Set conditions that would pass G2 except for dwell
        slot.state.metrics.record_accuracy(50.0)
        slot.state.metrics.record_accuracy(60.0)
        slot.state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1

        # Only 1 epoch in TRAINING, but need more for dwell
        slot.state.metrics.epochs_in_current_stage = 0

        slot.step_epoch()

        # Should still be in TRAINING (dwell not satisfied)
        assert slot.state.stage == SeedStage.TRAINING


class TestBlendingProgressTracking:
    """Tests for blending progress tracking."""

    def test_blending_steps_done_increments(self):
        """blending_steps_done should increment each step_epoch in BLENDING."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        initial_steps = slot.state.blending_steps_done

        slot.step_epoch()

        assert slot.state.blending_steps_done == initial_steps + 1

    def test_blending_steps_total_set_at_start(self):
        """blending_steps_total should be set when blending starts."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        total_steps = 15
        slot.start_blending(total_steps=total_steps)

        assert slot.state.blending_steps_total == total_steps

    def test_alpha_updates_during_blending(self):
        """Alpha should update each step during BLENDING."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        initial_alpha = slot.state.alpha

        slot.step_epoch()

        # Alpha should have increased (linear or sigmoid)
        assert slot.state.alpha >= initial_alpha


class TestRecycledSlot:
    """Tests for slot recycling after cull."""

    def test_slot_can_germinate_after_cull(self):
        """Slot should be reusable after culling a seed."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # First lifecycle
        slot.germinate("noop", seed_id="seed1")
        slot.state.transition(SeedStage.TRAINING)
        slot.cull()

        assert slot.state is None
        assert slot.seed is None

        # Second lifecycle
        state = slot.germinate("noop", seed_id="seed2")

        assert state.seed_id == "seed2"
        assert state.stage == SeedStage.GERMINATED

    def test_recycled_slot_fresh_metrics(self):
        """Recycled slot should have fresh metrics."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # First lifecycle with accumulated metrics
        slot.germinate("noop", seed_id="seed1")
        slot.state.metrics.record_accuracy(50.0)
        slot.state.metrics.record_accuracy(60.0)
        slot.state.transition(SeedStage.TRAINING)
        slot.cull()

        # Second lifecycle
        slot.germinate("noop", seed_id="seed2")

        # Metrics should be fresh
        assert slot.state.metrics.epochs_total == 0
        assert slot.state.metrics.best_val_accuracy == 0.0
