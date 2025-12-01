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
            slot.state.metrics.record_accuracy(0.0)
            slot.step_epoch()

        # Record metrics to simulate validation and allow dwell accounting in SHADOWING
        slot.state.metrics.record_accuracy(0.0)

        # One more epoch to complete SHADOWING dwell
        slot.step_epoch()
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


class TestStrategicFossilizeOnly:
    """Test that FOSSILIZE only works at strategic decision points."""

    def test_fossilize_from_training_disallowed(self):
        """FOSSILIZE from TRAINING should not bypass blending."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)

        ok = model.seed_state.transition(SeedStage.FOSSILIZED)
        assert ok is False
        assert model.seed_state.stage == SeedStage.TRAINING

    def test_fossilize_from_probationary(self):
        """FOSSILIZE from PROBATIONARY should transition to FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)
        model.seed_state.transition(SeedStage.BLENDING)
        model.seed_state.transition(SeedStage.SHADOWING)
        model.seed_state.transition(SeedStage.PROBATIONARY)

        # FOSSILIZE from PROBATIONARY should work
        ok = model.seed_state.transition(SeedStage.FOSSILIZED)

        assert ok is True
        assert model.seed_state.stage == SeedStage.FOSSILIZED

    def test_fossilize_from_blending_is_noop(self):
        """FOSSILIZE from BLENDING should NOT transition directly to FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)
        model.seed_state.transition(SeedStage.BLENDING)

        # This SHOULD fail (the bug we're fixing)
        ok = model.seed_state.transition(SeedStage.FOSSILIZED)

        assert ok is False
        assert model.seed_state.stage == SeedStage.BLENDING  # Unchanged


class TestLifecycleIntegration:
    """Integration test for full lifecycle flow."""

    def test_full_lifecycle_with_auto_advance(self):
        """Test TRAINING→BLENDING→(auto)→PROBATIONARY→FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)

        # Tamiyo: action triggers blending start (mechanical now)
        model.seed_state.transition(SeedStage.BLENDING)
        model.seed_slot.start_blending(total_steps=3, temperature=1.0)

        assert model.seed_state.stage == SeedStage.BLENDING

        # Kasmina: auto-advance via step_epoch
        for _ in range(3):
            model.seed_state.metrics.record_accuracy(0.0)
            model.seed_slot.step_epoch()  # advance blending progress

        # Shadowing dwell requires a recorded epoch
        model.seed_state.metrics.record_accuracy(0.0)
        model.seed_slot.step_epoch()  # dwell → PROBATIONARY

        assert model.seed_state.stage == SeedStage.PROBATIONARY

        # Tamiyo: FOSSILIZE to finalize
        ok = model.seed_state.transition(SeedStage.FOSSILIZED)

        assert ok is True
        assert model.seed_state.stage == SeedStage.FOSSILIZED

    def test_full_state_machine_reaches_probationary(self):
        """Germinate → TRAINING → BLENDING → SHADOWING → PROBATIONARY via Kasmina mechanics."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu")
        model.germinate_seed("conv_enhance", "test_seed")

        # Advance G1 gate: GERMINATED -> TRAINING (mirrors Simic training path).
        result = model.seed_slot.advance_stage(SeedStage.TRAINING)
        assert result.passed
        assert model.seed_state.stage == SeedStage.TRAINING

        # Drive metrics until TRAINING → BLENDING triggers via step_epoch.
        acc = 60.0
        for _ in range(10):
            model.seed_state.metrics.record_accuracy(acc)
            model.seed_slot.step_epoch()
            acc += 1.0
            if model.seed_state.stage == SeedStage.BLENDING:
                break

        assert model.seed_state.stage == SeedStage.BLENDING, \
            f"Seed failed to leave TRAINING; current stage: {model.seed_state.stage}"

        # Continue driving epochs so BLENDING → SHADOWING → PROBATIONARY auto-advance.
        for _ in range(20):
            model.seed_state.metrics.record_accuracy(acc)
            model.seed_slot.step_epoch()
            acc += 0.5
            if model.seed_state.stage == SeedStage.PROBATIONARY:
                break

        assert model.seed_state.stage == SeedStage.PROBATIONARY, \
            f"Seed failed to reach PROBATIONARY; current stage: {model.seed_state.stage}"

    def test_fossilization_emits_telemetry(self):
        """Test that fossilization emits SEED_FOSSILIZED telemetry."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost
        from esper.leyline import TelemetryEventType

        model = MorphogeneticModel(CNNHost(), device="cpu")

        # Capture telemetry events
        captured_events = []
        def capture(event):
            captured_events.append(event)

        model.seed_slot.on_telemetry = capture
        model.seed_slot.fast_mode = False

        # Run through lifecycle
        model.germinate_seed("conv_enhance", "test_seed")
        model.seed_state.transition(SeedStage.TRAINING)
        model.seed_state.transition(SeedStage.BLENDING)
        model.seed_slot.start_blending(total_steps=3, temperature=1.0)

        # Simulate training/validation metrics to drive dwell counters and gates
        for acc in (60.0, 61.0, 62.0):
            model.seed_state.metrics.record_accuracy(acc)
            model.seed_slot.step_epoch()  # advance blending progress

        model.seed_state.metrics.record_accuracy(63.0)  # shadowing dwell epoch
        model.seed_slot.step_epoch()

        # Use advance_stage to fossilize (this emits telemetry)
        result = model.seed_slot.advance_stage(target_stage=SeedStage.FOSSILIZED)
        assert result.passed, f"Gate should pass with mocked improvement: {result}"

        # Check we got SEED_FOSSILIZED event
        fossilized_events = [
            e for e in captured_events
            if e.event_type == TelemetryEventType.SEED_FOSSILIZED
        ]

        assert len(fossilized_events) >= 1, f"Expected SEED_FOSSILIZED, got: {[e.event_type for e in captured_events]}"
