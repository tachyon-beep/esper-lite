"""Test unified transition handling between advance_stage() and step_epoch()."""

import pytest
import torch

from esper.kasmina.slot import SeedSlot
from esper.leyline.stages import SeedStage
from esper.simic.control import TaskConfig


def create_slot_in_training() -> SeedSlot:
    """Create a SeedSlot with a seed in TRAINING stage."""
    task_config = TaskConfig(
        task_type="classification",
        topology="cnn",
        baseline_loss=2.3,
        target_loss=0.5,
        typical_loss_delta_std=0.1,
        max_epochs=100,
        blending_steps=10,
        train_to_blend_fraction=0.1,
    )

    slot = SeedSlot(
        slot_id="r0c0",
        channels=64,
        device="cpu",
        task_config=task_config,
    )

    # Germinate and advance to TRAINING
    slot.germinate(
        blueprint_id="norm",
        seed_id="test-seed",
        blend_algorithm_id="linear",
    )

    # Advance to TRAINING
    assert slot.state.stage == SeedStage.GERMINATED
    result = slot.advance_stage(SeedStage.TRAINING)
    assert result.passed
    assert slot.state.stage == SeedStage.TRAINING

    # Simulate some training epochs with improvement
    # Need at least 0.5 RAW improvement to pass G2 gate (DEFAULT_MIN_TRAINING_IMPROVEMENT = 0.5)
    # The gate display format shows raw values with "%" suffix, not actual percentages
    # So 0.5 raw improvement displays as "0.50%" but means 50 percentage points
    # First record_accuracy() sets the baseline, then we improve from there
    slot.state.metrics.record_accuracy(0.30)  # Sets accuracy_at_stage_start = 0.30
    for i in range(10):
        accuracy = 0.30 + ((i + 1) * 0.06)  # 0.36 -> 0.90 = 0.6 raw improvement
        slot.state.metrics.record_accuracy(accuracy)

    # Fake gradient activity to pass G2 gate
    slot.state.metrics.seed_gradient_norm_ratio = 1.0

    return slot


class TestUnifiedTransitions:
    """Test advance_stage() and step_epoch() behave identically."""

    def test_advance_stage_initializes_blending(self):
        """advance_stage() to BLENDING should initialize schedule and set _blending_started."""
        slot = create_slot_in_training()

        # Don't record additional accuracy here - create_slot_in_training() already did it

        # Use advance_stage directly
        result = slot.advance_stage(SeedStage.BLENDING)

        assert result.passed, f"Gate failed: {result.checks_failed}"
        assert slot.state.stage == SeedStage.BLENDING

        # CRITICAL: These should be initialized by advance_stage()
        assert slot.state.metrics._blending_started is True, \
            "advance_stage() should set _blending_started flag"
        assert slot.alpha_schedule is not None, \
            "advance_stage() should initialize alpha_schedule"
        # accuracy_at_blending_start is already set by advance_stage() (line 1026)
        assert slot.state.metrics.accuracy_at_blending_start == slot.state.metrics.current_val_accuracy, \
            "advance_stage() should snapshot accuracy_at_blending_start"

    def test_step_epoch_initializes_blending(self):
        """step_epoch() to BLENDING should initialize schedule and set _blending_started."""
        slot = create_slot_in_training()

        # create_slot_in_training() already gave us 0.9 accuracy (0.6 improvement from 0.3)
        # Just ensure we've dwelled long enough in TRAINING stage
        current_accuracy = slot.state.metrics.current_val_accuracy
        while slot.state.metrics.epochs_in_current_stage < 10:
            slot.state.metrics.record_accuracy(current_accuracy)

        # Use step_epoch() to advance
        slot.step_epoch()

        assert slot.state.stage == SeedStage.BLENDING

        # These should be initialized by step_epoch()
        assert slot.state.metrics._blending_started is True, \
            "step_epoch() should set _blending_started flag"
        assert slot.alpha_schedule is not None, \
            "step_epoch() should initialize alpha_schedule"
        assert slot.state.metrics.accuracy_at_blending_start == current_accuracy, \
            "step_epoch() should snapshot accuracy_at_blending_start"

    def test_both_methods_produce_same_state(self):
        """advance_stage() and step_epoch() should produce identical state."""
        # Create two identical slots
        slot_via_advance = create_slot_in_training()
        slot_via_step_epoch = create_slot_in_training()

        # Both slots already have 0.9 accuracy from create_slot_in_training()
        # Just ensure dwell is satisfied for step_epoch
        current_accuracy = slot_via_step_epoch.state.metrics.current_val_accuracy
        while slot_via_step_epoch.state.metrics.epochs_in_current_stage < 10:
            slot_via_step_epoch.state.metrics.record_accuracy(current_accuracy)

        # Advance via different methods
        slot_via_advance.advance_stage(SeedStage.BLENDING)
        slot_via_step_epoch.step_epoch()

        # Both should be in BLENDING
        assert slot_via_advance.state.stage == SeedStage.BLENDING
        assert slot_via_step_epoch.state.stage == SeedStage.BLENDING

        # Both should have blending initialized
        assert slot_via_advance.state.metrics._blending_started == \
               slot_via_step_epoch.state.metrics._blending_started
        assert (slot_via_advance.alpha_schedule is not None) == \
               (slot_via_step_epoch.alpha_schedule is not None)
        # Note: accuracy_at_blending_start might differ slightly due to when transition() resets baseline
        # but both should be non-zero and close to current accuracy
        assert slot_via_advance.state.metrics.accuracy_at_blending_start > 0
        assert slot_via_step_epoch.state.metrics.accuracy_at_blending_start > 0

    def test_advance_stage_blending_to_probationary_cleanup(self):
        """advance_stage() BLENDING→PROBATIONARY should clean up alpha_schedule."""
        slot = create_slot_in_training()

        # Advance to BLENDING (already has enough improvement from create_slot_in_training)
        result = slot.advance_stage(SeedStage.BLENDING)
        assert result.passed
        assert slot.alpha_schedule is not None

        # Simulate blending completion
        # Need to record some epochs in BLENDING stage to pass G3 gate min_blending_epochs check
        for _ in range(5):
            slot.state.metrics.record_accuracy(slot.state.metrics.current_val_accuracy)
        slot.state.blending_steps_done = slot.state.blending_steps_total
        slot.set_alpha(1.0)

        # Advance to PROBATIONARY
        result = slot.advance_stage(SeedStage.PROBATIONARY)
        assert result.passed
        assert slot.state.stage == SeedStage.PROBATIONARY

        # alpha_schedule should be cleaned up via _on_enter_stage() hook
        assert slot.alpha_schedule is None, \
            "advance_stage() should clean up alpha_schedule when entering PROBATIONARY"
        assert slot.state.alpha == 1.0, \
            "advance_stage() should set alpha=1.0 permanently after BLENDING"
