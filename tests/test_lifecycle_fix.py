"""Tests for lifecycle state machine fix."""

import pytest
from esper.kasmina.slot import SeedState, SeedSlot
from esper.leyline import SeedStage


class TestBlendingProgressTracking:
    """Test that SeedState tracks blending progress."""

    def test_seedstate_has_blending_fields(self):
        """SeedState should have blending progress fields."""
        state = SeedState(seed_id="test", blueprint_id="conv_enhance")

        assert state.blending_steps_done == 0
        assert state.blending_steps_total == 0

    def test_blending_fields_increment(self):
        """Blending fields should be mutable."""
        state = SeedState(seed_id="test", blueprint_id="conv_enhance")
        state.blending_steps_total = 5
        state.blending_steps_done = 3

        assert state.blending_steps_total == 5
        assert state.blending_steps_done == 3


class TestStartBlendingProgress:
    """Test that start_blending initializes progress tracking."""

    def test_start_blending_sets_total_steps(self):
        """start_blending should set blending_steps_total."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        # Germinate a seed first
        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_enhance", seed_id="test_seed", host_module=host)

        # Transition to TRAINING then BLENDING
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        # Start blending with 5 steps
        slot.start_blending(total_steps=5, temperature=1.0)

        assert slot.state.blending_steps_total == 5
        assert slot.state.blending_steps_done == 0

    def test_start_blending_resets_done_counter(self):
        """start_blending should reset blending_steps_done to 0."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_enhance", seed_id="test_seed", host_module=host)
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        # Manually set done to simulate prior state
        slot.state.blending_steps_done = 3

        # Start blending should reset
        slot.start_blending(total_steps=5, temperature=1.0)

        assert slot.state.blending_steps_done == 0


class TestStepEpochAutoAdvance:
    """Test that step_epoch auto-advances through mechanical stages."""

    def _create_blending_slot(self) -> SeedSlot:
        """Helper to create a slot in BLENDING stage."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_enhance", seed_id="test_seed", host_module=host)
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=3, temperature=1.0)

        return slot

    def test_step_epoch_increments_blending_progress(self):
        """step_epoch should increment blending_steps_done."""
        slot = self._create_blending_slot()

        assert slot.state.blending_steps_done == 0

        slot.step_epoch()
        assert slot.state.blending_steps_done == 1

        slot.step_epoch()
        assert slot.state.blending_steps_done == 2

    def test_step_epoch_updates_alpha(self):
        """step_epoch should update alpha based on progress."""
        slot = self._create_blending_slot()

        assert slot.alpha == 0.0

        slot.step_epoch()  # 1/3
        slot.step_epoch()  # 2/3
        slot.step_epoch()  # 3/3 = 1.0

        assert slot.alpha >= 0.99  # Should be at or near 1.0

    def test_step_epoch_auto_advances_when_blending_complete(self):
        """step_epoch should auto-advance BLENDING→SHADOWING→PROBATIONARY when α=1.0."""
        slot = self._create_blending_slot()

        # Run through all blending steps
        for _ in range(3):
            slot.step_epoch()

        # Should have auto-advanced through SHADOWING to PROBATIONARY
        assert slot.state.stage == SeedStage.PROBATIONARY
        assert slot.alpha >= 0.99

    def test_step_epoch_noop_when_not_blending(self):
        """step_epoch should be no-op when not in BLENDING stage."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_enhance", seed_id="test_seed", host_module=host)
        slot.state.transition(SeedStage.TRAINING)

        # Should not raise or change state
        slot.step_epoch()
        assert slot.state.stage == SeedStage.TRAINING

    def test_step_epoch_noop_when_no_seed(self):
        """step_epoch should be no-op when no active seed."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        # Should not raise
        slot.step_epoch()


class TestStrategicAdvanceOnly:
    """Test that ADVANCE only works at strategic decision points."""

    def test_advance_from_training_starts_blending(self):
        """ADVANCE from TRAINING should transition to BLENDING."""
        from esper.kasmina.host import MorphogeneticModel, HostCNN

        model = MorphogeneticModel(HostCNN(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)

        # Simulate ADVANCE action
        assert model.seed_state.stage == SeedStage.TRAINING
        model.seed_state.transition(SeedStage.BLENDING)
        model.seed_slot.start_blending(total_steps=5, temperature=1.0)

        assert model.seed_state.stage == SeedStage.BLENDING

    def test_advance_from_probationary_fossilizes(self):
        """ADVANCE from PROBATIONARY should transition to FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, HostCNN

        model = MorphogeneticModel(HostCNN(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)
        model.seed_state.transition(SeedStage.BLENDING)
        model.seed_state.transition(SeedStage.SHADOWING)
        model.seed_state.transition(SeedStage.PROBATIONARY)

        # ADVANCE from PROBATIONARY should work
        ok = model.seed_state.transition(SeedStage.FOSSILIZED)

        assert ok is True
        assert model.seed_state.stage == SeedStage.FOSSILIZED

    def test_advance_from_blending_is_noop(self):
        """ADVANCE from BLENDING should NOT transition directly to FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, HostCNN

        model = MorphogeneticModel(HostCNN(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)
        model.seed_state.transition(SeedStage.BLENDING)

        # This SHOULD fail (the bug we're fixing)
        ok = model.seed_state.transition(SeedStage.FOSSILIZED)

        assert ok is False
        assert model.seed_state.stage == SeedStage.BLENDING  # Unchanged
