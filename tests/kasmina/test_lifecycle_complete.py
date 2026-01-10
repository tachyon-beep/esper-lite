"""Complete lifecycle tests for Kasmina seed progression.

Tests verify correct behavior through complete lifecycles:
- Full happy path: DORMANT → FOSSILIZED
- Early cull scenarios
- Holding timeout behavior
- Dwell epoch enforcement
- Blending progress tracking
"""

import pytest

from esper.kasmina.slot import SeedSlot, SeedState, QualityGates
from esper.leyline import (
    SeedStage,
    DEFAULT_MIN_BLENDING_EPOCHS,
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    DEFAULT_GRADIENT_RATIO_THRESHOLD,
    DEFAULT_MAX_PROBATION_EPOCHS,
    DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE,
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
        """advance_stage() should transition GERMINATED → TRAINING."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        result = slot.advance_stage(SeedStage.TRAINING)

        assert result.passed
        assert slot.state.stage == SeedStage.TRAINING

    def test_lifecycle_training_to_blending(self):
        """advance_stage() should transition TRAINING → BLENDING when gate passes."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        result = slot.advance_stage(SeedStage.TRAINING)
        assert result.passed

        # Simulate conditions for G2 gate
        slot.state.metrics.record_accuracy(50.0)
        slot.state.metrics.record_accuracy(60.0)  # Improvement >= threshold

        # Set gradient ratio to pass G2
        slot.state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1

        # Need enough epochs in TRAINING
        for _ in range(DEFAULT_MIN_BLENDING_EPOCHS):
            slot.state.metrics.epochs_in_current_stage += 1

        result = slot.advance_stage(SeedStage.BLENDING)

        assert result.passed
        assert slot.state.stage == SeedStage.BLENDING

    def test_lifecycle_blending_to_holding(self):
        """advance_stage() should transition BLENDING → HOLDING when ready."""
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
        # Must accumulate DEFAULT_MIN_BLENDING_EPOCHS (10) epochs in BLENDING
        for _ in range(DEFAULT_MIN_BLENDING_EPOCHS):
            slot.state.metrics.record_accuracy(65.0)  # G3 needs epochs_in_current_stage
            slot.step_epoch()

        result = slot.advance_stage(SeedStage.HOLDING)
        assert result.passed
        assert slot.state.stage == SeedStage.HOLDING

    def test_lifecycle_holding_to_fossilized(self):
        """advance_stage() should transition HOLDING → FOSSILIZED when gate passes."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        # Progress through stages
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.HOLDING)
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

        result = slot.prune(reason="test_cull")

        assert result is True
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED
        assert slot.seed is None

    def test_cull_from_blending(self):
        """Culling from BLENDING should succeed."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        result = slot.prune(reason="test_cull")

        assert result is True
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED

    def test_cull_from_holding(self):
        """Culling from HOLDING should succeed."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.HOLDING)

        result = slot.prune(reason="test_cull")

        assert result is True
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED

    def test_cull_fossilized_fails(self):
        """Culling FOSSILIZED seed should fail (permanent)."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        # Force to FOSSILIZED
        slot.state.stage = SeedStage.FOSSILIZED

        result = slot.prune(reason="test_cull")

        assert result is False
        assert slot.state is not None
        assert slot.state.stage == SeedStage.FOSSILIZED


class TestHoldingTimeout:
    """Tests for holding timeout behavior."""

    def test_holding_timeout_does_not_prune(self):
        """Exceeding max holding epochs should not auto-prune."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.HOLDING)

        # Simulate epochs in holding exceeding timeout
        # Set counterfactual to non-negative so safety cull doesn't trigger first
        slot.state.metrics.counterfactual_contribution = 0.1
        slot.state.metrics.epochs_in_current_stage = DEFAULT_MAX_PROBATION_EPOCHS

        slot.step_epoch()

        # Should remain in holding without auto-prune
        assert slot.state is not None
        assert slot.state.stage == SeedStage.HOLDING

    def test_holding_no_timeout_before_max_epochs(self):
        """Seed should survive if under max holding epochs."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.HOLDING)

        # Just under timeout
        slot.state.metrics.counterfactual_contribution = 0.1
        slot.state.metrics.epochs_in_current_stage = DEFAULT_MAX_PROBATION_EPOCHS - 1

        slot.step_epoch()

        # Should still be in holding
        assert slot.state is not None
        assert slot.state.stage == SeedStage.HOLDING


class TestNegativeCounterfactualNoAutoPrune:
    """Tests for negative counterfactual behavior without auto-prune."""

    def test_negative_counterfactual_does_not_auto_prune(self):
        """Negative counterfactual in HOLDING should not auto-prune."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.HOLDING)

        # Set negative counterfactual
        slot.state.metrics.counterfactual_contribution = -1.0

        slot.step_epoch()

        assert slot.state is not None
        assert slot.state.stage == SeedStage.HOLDING

    def test_zero_counterfactual_does_not_auto_prune(self):
        """Zero counterfactual in HOLDING should not auto-prune."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.HOLDING)

        slot.state.metrics.counterfactual_contribution = 0.0

        slot.step_epoch()

        assert slot.state is not None
        assert slot.state.stage == SeedStage.HOLDING

    def test_positive_counterfactual_survives(self):
        """Positive counterfactual should not trigger auto-cull."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.HOLDING)

        slot.state.metrics.counterfactual_contribution = 1.0

        slot.step_epoch()

        # Should still be in holding
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
            stage=SeedStage.HOLDING,
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
            stage=SeedStage.HOLDING,
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
        gate_result = slot.advance_stage(target_stage=SeedStage.TRAINING)
        assert gate_result.passed
        assert slot.state.stage == SeedStage.TRAINING

        # Set conditions that would pass G2 except for dwell
        slot.state.metrics.record_accuracy(50.0)
        slot.state.metrics.record_accuracy(60.0)
        slot.state.metrics.seed_gradient_norm_ratio = DEFAULT_GRADIENT_RATIO_THRESHOLD + 0.1

        # Only 1 epoch in TRAINING, but need more for dwell
        slot.state.metrics.epochs_in_current_stage = 0

        gate_result = slot.advance_stage(target_stage=SeedStage.BLENDING)
        assert not gate_result.passed

        # Should still be in TRAINING (dwell not satisfied)
        assert slot.state.stage == SeedStage.TRAINING


class TestBlendingProgressTracking:
    """Tests for blending progress tracking."""

    def test_blending_steps_done_increments(self):
        """alpha_controller steps should increment each step_epoch in BLENDING."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)

        initial_steps = slot.state.alpha_controller.alpha_steps_done

        slot.step_epoch()

        assert slot.state.alpha_controller.alpha_steps_done == initial_steps + 1

    def test_blending_steps_total_set_at_start(self):
        """alpha_controller steps_total should be set when blending starts."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        total_steps = 15
        slot.start_blending(total_steps=total_steps)

        assert slot.state.alpha_controller.alpha_steps_total == total_steps

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
        slot.prune()

        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED
        assert slot.seed is None

        # Cooldown must complete before slot is available again.
        for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
            slot.step_epoch()
        assert slot.state is None

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
        slot.prune()

        # Cooldown must complete before slot is available again.
        for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
            slot.step_epoch()
        assert slot.state is None

        # Second lifecycle
        slot.germinate("noop", seed_id="seed2")

        # Metrics should be fresh
        assert slot.state.metrics.epochs_total == 0
        assert slot.state.metrics.best_val_accuracy == 0.0


class TestAdvanceStageGuards:
    """Tests for advance_stage() API footgun prevention."""

    def test_advance_stage_rejects_failure_stage_target(self):
        """advance_stage() must not allow targeting failure stages like PRUNED."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        assert slot.seed is not None
        assert slot.state is not None
        stage_before = slot.state.stage
        seed_before = slot.seed

        with pytest.raises(ValueError, match="cannot target failure stage"):
            slot.advance_stage(target_stage=SeedStage.PRUNED)

        assert slot.seed is seed_before
        assert slot.state is not None
        assert slot.state.stage == stage_before
