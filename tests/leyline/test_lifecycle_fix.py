"""Tests for lifecycle state machine fix."""

from esper.kasmina.slot import SeedState, SeedSlot
from esper.leyline import SeedStage, DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE
from esper.leyline.alpha import AlphaMode


class TestBlendingProgressTracking:
    """Test that SeedState tracks blending progress via AlphaController."""

    def test_seedstate_has_alpha_controller_defaults(self):
        """SeedState should have alpha controller fields with safe defaults."""
        state = SeedState(seed_id="test", blueprint_id="conv_heavy")

        assert state.alpha_controller.alpha_steps_done == 0
        assert state.alpha_controller.alpha_steps_total == 0
        assert state.alpha_controller.alpha_mode == AlphaMode.HOLD

    def test_alpha_controller_fields_mutate(self):
        """Alpha controller should be mutable (via retarget)."""
        state = SeedState(seed_id="test", blueprint_id="conv_heavy")
        state.alpha_controller.retarget(alpha_target=1.0, alpha_steps_total=5)

        assert state.alpha_controller.alpha_steps_total == 5
        assert state.alpha_controller.alpha_steps_done == 0
        assert state.alpha_controller.alpha_mode == AlphaMode.UP


class TestStartBlendingProgress:
    """Test that start_blending initializes alpha controller state."""

    def test_start_blending_sets_total_steps(self):
        """start_blending should configure alpha controller steps."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        # Germinate a seed first
        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_heavy", seed_id="test_seed", host_module=host)

        # Transition to TRAINING then BLENDING
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        # Start blending with 5 steps
        slot.start_blending(total_steps=5)

        assert slot.state.alpha_controller.alpha_steps_total == 5
        assert slot.state.alpha_controller.alpha_steps_done == 0

    def test_start_blending_resets_done_counter(self):
        """start_blending should reset alpha controller progress to 0."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_heavy", seed_id="test_seed", host_module=host)
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        # Manually set progress to simulate prior state
        slot.state.alpha_controller.alpha_steps_total = 5
        slot.state.alpha_controller.alpha_steps_done = 3

        # Start blending should reset
        slot.start_blending(total_steps=5)

        assert slot.state.alpha_controller.alpha_steps_done == 0


class TestStepEpochAutoAdvance:
    """Test that step_epoch advances alpha schedules without stage transitions."""

    def _create_blending_slot(self) -> SeedSlot:
        """Helper to create a slot in BLENDING stage."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_heavy", seed_id="test_seed", host_module=host)
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=3)

        return slot

    def test_step_epoch_increments_blending_progress(self):
        """step_epoch should increment alpha controller progress."""
        slot = self._create_blending_slot()

        assert slot.state.alpha_controller.alpha_steps_done == 0

        slot.step_epoch()
        assert slot.state.alpha_controller.alpha_steps_done == 1

        slot.step_epoch()
        assert slot.state.alpha_controller.alpha_steps_done == 2

    def test_step_epoch_updates_alpha(self):
        """step_epoch should update alpha based on progress."""
        slot = self._create_blending_slot()

        assert slot.alpha == 0.0

        slot.step_epoch()  # 1/3
        slot.step_epoch()  # 2/3
        slot.step_epoch()  # 3/3 = 1.0

        assert slot.alpha >= 0.99  # Should be at or near 1.0

    def test_step_epoch_does_not_advance_when_blending_complete(self):
        """step_epoch should not auto-advance BLENDING→HOLDING when α=1.0."""
        slot = self._create_blending_slot()

        # Drive epochs until blending completes.
        for _ in range(10):
            slot.state.metrics.record_accuracy(0.0)
            slot.step_epoch()

        assert slot.state.stage == SeedStage.BLENDING
        assert slot.alpha >= 0.99

    def test_step_epoch_noop_when_not_blending(self):
        """step_epoch should be no-op when not in BLENDING stage."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")

        from unittest.mock import MagicMock
        host = MagicMock()
        host.injection_channels = 64
        slot.germinate(blueprint_id="conv_heavy", seed_id="test_seed", host_module=host)
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

        model = MorphogeneticModel(CNNHost(), device="cpu", slots=["r0c1"])
        model.germinate_seed("conv_heavy", "test_seed", slot="r0c1")
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)

        ok = model.seed_slots["r0c1"].state.transition(SeedStage.FOSSILIZED)
        assert ok is False
        assert model.seed_slots["r0c1"].state.stage == SeedStage.TRAINING

    def test_fossilize_from_holding(self):
        """FOSSILIZE from HOLDING should transition to FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu", slots=["r0c1"])
        model.germinate_seed("conv_heavy", "test_seed", slot="r0c1")
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c1"].state.transition(SeedStage.BLENDING)
        model.seed_slots["r0c1"].state.transition(SeedStage.HOLDING)

        # FOSSILIZE from HOLDING should work
        ok = model.seed_slots["r0c1"].state.transition(SeedStage.FOSSILIZED)

        assert ok is True
        assert model.seed_slots["r0c1"].state.stage == SeedStage.FOSSILIZED

    def test_fossilize_from_blending_is_noop(self):
        """FOSSILIZE from BLENDING should NOT transition directly to FOSSILIZED."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu", slots=["r0c1"])
        model.germinate_seed("conv_heavy", "test_seed", slot="r0c1")
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c1"].state.transition(SeedStage.BLENDING)

        # This SHOULD fail (the bug we're fixing)
        ok = model.seed_slots["r0c1"].state.transition(SeedStage.FOSSILIZED)

        assert ok is False
        assert model.seed_slots["r0c1"].state.stage == SeedStage.BLENDING  # Unchanged


class TestLifecycleIntegration:
    """Integration test for full lifecycle flow."""

    def test_full_lifecycle_with_explicit_advance(self):
        """Test TRAINING→BLENDING→HOLDING→FOSSILIZED with explicit ADVANCE."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu", slots=["r0c1"])
        model.germinate_seed("conv_heavy", "test_seed", slot="r0c1")
        result = model.seed_slots["r0c1"].advance_stage(SeedStage.TRAINING)
        assert result.passed

        slot = model.seed_slots["r0c1"]
        # Prepare G2 gate inputs before ADVANCE to BLENDING.
        # G2 requires epochs_in_current_stage >= min_blending_epochs (default 10).
        # record_accuracy() increments epochs_in_current_stage.
        for i in range(10):
            slot.state.metrics.record_accuracy(50.0 + i * 0.2)
        slot.state.metrics.seed_gradient_norm_ratio = 0.2
        result = slot.advance_stage(SeedStage.BLENDING)
        assert result.passed

        assert slot.state.stage == SeedStage.BLENDING

        # Tick blending progress.
        # G3 requires epochs_in_current_stage >= min_blending_epochs (default 10).
        for _ in range(10):
            slot.state.metrics.record_accuracy(60.0)
            slot.step_epoch()

        result = slot.advance_stage(SeedStage.HOLDING)
        assert result.passed

        # Tamiyo: FOSSILIZE to finalize
        slot.state.metrics.counterfactual_contribution = 3.0
        slot.state.is_healthy = True
        ok = slot.advance_stage(SeedStage.FOSSILIZED)

        assert ok.passed is True
        assert slot.state.stage == SeedStage.FOSSILIZED

    def test_full_state_machine_reaches_holding(self):
        """Germinate → TRAINING → BLENDING → HOLDING with explicit ADVANCE."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu", slots=["r0c1"])
        model.germinate_seed("conv_heavy", "test_seed", slot="r0c1")

        # Advance G1 gate: GERMINATED -> TRAINING (mirrors Simic training path).
        result = model.seed_slots["r0c1"].advance_stage(SeedStage.TRAINING)
        assert result.passed
        assert model.seed_slots["r0c1"].state.stage == SeedStage.TRAINING

        # Drive metrics until TRAINING → BLENDING gate passes, then ADVANCE.
        # G2 requires epochs_in_current_stage >= min_blending_epochs (default 10).
        acc = 60.0
        slot = model.seed_slots["r0c1"]
        for _ in range(10):
            slot.state.metrics.record_accuracy(acc)
            slot.state.metrics.seed_gradient_norm_ratio = 0.2
            acc += 0.3

        result = slot.advance_stage(SeedStage.BLENDING)
        assert result.passed, f"Seed failed to leave TRAINING; current stage: {slot.state.stage}"
        assert slot.state.stage == SeedStage.BLENDING

        # Continue driving epochs to reach full amplitude, then ADVANCE.
        # G3 requires epochs_in_current_stage >= min_blending_epochs (default 10).
        for _ in range(10):
            slot.state.metrics.record_accuracy(acc)
            slot.step_epoch()
            acc += 0.5

        result = slot.advance_stage(SeedStage.HOLDING)
        assert result.passed, f"Seed failed to reach HOLDING; current stage: {slot.state.stage}"
        assert slot.state.stage == SeedStage.HOLDING

    def test_fossilization_emits_telemetry(self):
        """Test that fossilization emits SEED_FOSSILIZED telemetry."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost
        from esper.leyline import TelemetryEventType

        model = MorphogeneticModel(CNNHost(), device="cpu", slots=["r0c1"])

        # Capture telemetry events
        captured_events = []
        def capture(event):
            captured_events.append(event)

        model.seed_slots["r0c1"].on_telemetry = capture
        model.seed_slots["r0c1"].fast_mode = False

        # Run through lifecycle
        model.germinate_seed("conv_heavy", "test_seed", slot="r0c1")
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)
        slot = model.seed_slots["r0c1"]
        # G2 requires epochs_in_current_stage >= min_blending_epochs (default 10).
        for i in range(10):
            slot.state.metrics.record_accuracy(60.0 + i * 0.1)
        slot.state.metrics.seed_gradient_norm_ratio = 0.2
        result = slot.advance_stage(SeedStage.BLENDING)
        assert result.passed

        # Simulate training/validation metrics to drive dwell counters and alpha
        # G3 requires epochs_in_current_stage >= min_blending_epochs (default 10).
        for _ in range(10):
            slot.state.metrics.record_accuracy(60.0)
            slot.step_epoch()
        result = slot.advance_stage(SeedStage.HOLDING)
        assert result.passed
        assert slot.state.stage == SeedStage.HOLDING

        # Set counterfactual and health required for G5 gate
        slot.state.metrics.counterfactual_contribution = 3.0
        slot.state.is_healthy = True

        # Use advance_stage to fossilize (this emits telemetry)
        result = slot.advance_stage(target_stage=SeedStage.FOSSILIZED)
        assert result.passed, f"Gate should pass with mocked improvement: {result}"

        # Check we got SEED_FOSSILIZED event
        fossilized_events = [
            e for e in captured_events
            if e.event_type == TelemetryEventType.SEED_FOSSILIZED
        ]

        assert len(fossilized_events) >= 1, f"Expected SEED_FOSSILIZED, got: {[e.event_type for e in captured_events]}"


class TestFossilizedCullProtection:
    """Test that FOSSILIZED seeds cannot be culled."""

    def test_cull_fossilized_seed_returns_false(self):
        """Attempting to cull a FOSSILIZED seed should return False."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu", slots=["r0c1"])
        model.germinate_seed("conv_heavy", "test_seed", slot="r0c1")

        # Drive through lifecycle to FOSSILIZED (Phase 5+: stage advancement is explicit).
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)
        model.seed_slots["r0c1"].state.transition(SeedStage.BLENDING)
        model.seed_slots["r0c1"].start_blending(total_steps=3)
        model.seed_slots["r0c1"].state.transition(SeedStage.HOLDING)
        model.seed_slots["r0c1"].set_alpha(1.0)

        # Fossilize
        ok = model.seed_slots["r0c1"].state.transition(SeedStage.FOSSILIZED)
        assert ok is True
        assert model.seed_slots["r0c1"].state.stage == SeedStage.FOSSILIZED

        # Attempt to cull - should return False
        cull_result = model.seed_slots["r0c1"].prune("test_cull_attempt")
        assert cull_result is False, "FOSSILIZED seeds should not be cullable"

        # Seed should still be FOSSILIZED
        assert model.seed_slots["r0c1"].state is not None
        assert model.seed_slots["r0c1"].state.stage == SeedStage.FOSSILIZED

    def test_cull_non_fossilized_seed_works(self):
        """Culling non-FOSSILIZED seeds should still work."""
        from esper.kasmina.host import MorphogeneticModel, CNNHost

        model = MorphogeneticModel(CNNHost(), device="cpu", slots=["r0c1"])
        model.germinate_seed("conv_heavy", "test_seed", slot="r0c1")
        model.seed_slots["r0c1"].state.transition(SeedStage.TRAINING)

        assert model.seed_slots["r0c1"].state.stage == SeedStage.TRAINING

        # Cull from TRAINING - should work
        cull_result = model.seed_slots["r0c1"].prune("performance_issue")
        assert cull_result is True, "Non-FOSSILIZED seeds should be cullable"

        # Phase 4: seed is physically removed, but state persists for cooldown.
        assert model.seed_slots["r0c1"].seed is None
        assert model.seed_slots["r0c1"].state is not None
        assert model.seed_slots["r0c1"].state.stage == SeedStage.PRUNED

        for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
            model.seed_slots["r0c1"].step_epoch()
        assert model.seed_slots["r0c1"].state is None
