"""Edge case tests for Kasmina SeedSlot.

Tests verify correct behavior at boundary conditions:
- Alpha boundary values (0.0, 1.0, negative, > 1)
- Double germination error
- Invalid blueprint error
- Cull fossilized returns False
- step_epoch with no seed
- record_accuracy behavior
"""

import pytest
import torch

from esper.kasmina.slot import SeedSlot, SeedMetrics, SeedState, QualityGates
from esper.kasmina.isolation import blend_with_isolation
from esper.leyline import SeedStage, DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE


class TestAlphaBoundaryValues:
    """Tests for alpha at exact boundary values."""

    def test_alpha_exactly_zero_pure_host(self):
        """At alpha=0.0, blend should return pure host output."""
        host = torch.randn(2, 64, 8, 8)
        seed = torch.randn(2, 64, 8, 8)
        alpha = torch.tensor(0.0)

        result = blend_with_isolation(host, seed, alpha)

        torch.testing.assert_close(result, host)

    def test_alpha_exactly_one_pure_seed(self):
        """At alpha=1.0, blend should return pure seed output."""
        host = torch.randn(2, 64, 8, 8)
        seed = torch.randn(2, 64, 8, 8)
        alpha = torch.tensor(1.0)

        result = blend_with_isolation(host, seed, alpha)

        torch.testing.assert_close(result, seed)

    def test_alpha_negative_clamped_to_zero(self):
        """set_alpha(-0.1) should clamp to 0.0."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        slot.set_alpha(-0.1)

        assert slot.state.alpha == 0.0
        assert slot.state.metrics.current_alpha == 0.0

    def test_alpha_over_one_clamped_to_one(self):
        """set_alpha(1.5) should clamp to 1.0."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        slot.set_alpha(1.5)

        assert slot.state.alpha == 1.0
        assert slot.state.metrics.current_alpha == 1.0

    def test_alpha_exactly_half_blends_equally(self):
        """At alpha=0.5, blend should average host and seed."""
        host = torch.ones(2, 64, 8, 8) * 10.0
        seed = torch.ones(2, 64, 8, 8) * 20.0
        alpha = torch.tensor(0.5)

        result = blend_with_isolation(host, seed, alpha)

        expected = torch.ones(2, 64, 8, 8) * 15.0
        torch.testing.assert_close(result, expected)


class TestGerminationErrors:
    """Tests for germination error conditions."""

    def test_germinate_twice_raises_runtime_error(self):
        """Germinating when seed is already active should raise RuntimeError."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # First germination succeeds
        slot.germinate("noop", seed_id="seed1")

        # Second germination should fail
        with pytest.raises(RuntimeError, match="already has active seed"):
            slot.germinate("noop", seed_id="seed2")

    def test_germinate_after_cull_succeeds(self):
        """Germinating after cull should succeed (slot recycled)."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # First lifecycle
        slot.germinate("noop", seed_id="seed1")
        slot.state.transition(SeedStage.TRAINING)
        result = slot.prune()

        assert result is True
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED
        assert slot.seed is None

        # Cooldown must complete before slot is available again.
        for _ in range(DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE + 2):
            slot.step_epoch()
        assert slot.state is None

        # Second germination should work (slot is now empty)
        state = slot.germinate("noop", seed_id="seed2")

        assert state.seed_id == "seed2"
        assert state.stage == SeedStage.GERMINATED

    def test_germinate_invalid_blueprint_raises_value_error(self):
        """Germinating with unknown blueprint should raise ValueError."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        with pytest.raises(ValueError, match="not available"):
            slot.germinate("nonexistent_blueprint", seed_id="test")

    def test_germinate_with_valid_blueprints(self):
        """Germinating with valid blueprints should work."""
        valid_blueprints = ["noop", "conv_heavy", "norm", "depthwise"]

        for bp in valid_blueprints:
            slot = SeedSlot(slot_id="r0c0", channels=64)
            state = slot.germinate(bp, seed_id=f"test_{bp}")
            assert state.blueprint_id == bp
            assert state.stage == SeedStage.GERMINATED


class TestCullBehavior:
    """Tests for cull behavior at different stages."""

    def test_cull_from_training_succeeds(self):
        """Culling from TRAINING stage should succeed and remove seed."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)

        result = slot.prune()

        assert result is True
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED
        assert slot.seed is None

    def test_cull_from_blending_succeeds(self):
        """Culling from BLENDING stage should succeed and remove seed."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)

        result = slot.prune()

        assert result is True
        assert slot.state is not None
        assert slot.state.stage == SeedStage.PRUNED
        assert slot.seed is None

    def test_cull_fossilized_returns_false(self):
        """Culling a FOSSILIZED seed should return False (can't transition)."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        # Force to FOSSILIZED (bypassing gates for testing)
        slot.state.stage = SeedStage.FOSSILIZED

        # FOSSILIZED has no valid transitions
        result = slot.prune()

        assert result is False

    def test_cull_no_active_seed_returns_false(self):
        """Culling when no seed is active should return False."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        result = slot.prune()

        assert result is False


class TestStepEpochBehavior:
    """Tests for step_epoch behavior."""

    def test_step_epoch_no_seed_no_crash(self):
        """step_epoch() with no active seed should not crash."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # Should be a no-op, not raise
        slot.step_epoch()

    def test_step_epoch_germinated_advances_to_training(self):
        """step_epoch() in GERMINATED stage advances to TRAINING."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        # After germination, stage is GERMINATED
        assert slot.state.stage == SeedStage.GERMINATED

        # step_epoch should advance to TRAINING
        slot.step_epoch()

        assert slot.state.stage == SeedStage.TRAINING

    def test_step_epoch_training_stays_in_training(self):
        """step_epoch() in TRAINING stage stays in TRAINING (no auto-advance)."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)

        # step_epoch doesn't auto-advance from TRAINING without passing G2
        slot.step_epoch()

        # Should still be in TRAINING (G2 gate not passed)
        assert slot.state.stage == SeedStage.TRAINING


class TestRecordAccuracyBehavior:
    """Tests for SeedMetrics.record_accuracy behavior."""

    def test_record_accuracy_first_call_sets_initial(self):
        """First record_accuracy() call should set initial_val_accuracy."""
        metrics = SeedMetrics()

        metrics.record_accuracy(75.5)

        assert metrics.initial_val_accuracy == 75.5
        assert metrics.current_val_accuracy == 75.5
        assert metrics.best_val_accuracy == 75.5

    def test_record_accuracy_updates_best_when_higher(self):
        """Higher accuracy should update best_val_accuracy."""
        metrics = SeedMetrics()

        metrics.record_accuracy(50.0)
        metrics.record_accuracy(60.0)
        metrics.record_accuracy(70.0)

        assert metrics.best_val_accuracy == 70.0
        assert metrics.current_val_accuracy == 70.0

    def test_record_accuracy_keeps_best_when_lower(self):
        """Lower accuracy should not update best_val_accuracy."""
        metrics = SeedMetrics()

        metrics.record_accuracy(80.0)
        metrics.record_accuracy(60.0)
        metrics.record_accuracy(40.0)

        assert metrics.best_val_accuracy == 80.0
        assert metrics.current_val_accuracy == 40.0

    def test_record_accuracy_increments_epochs(self):
        """record_accuracy() should increment epochs_total."""
        metrics = SeedMetrics()

        metrics.record_accuracy(50.0)
        metrics.record_accuracy(55.0)
        metrics.record_accuracy(60.0)

        assert metrics.epochs_total == 3

    def test_record_accuracy_accepts_tensor(self):
        """record_accuracy() should accept torch.Tensor."""
        metrics = SeedMetrics()

        tensor_acc = torch.tensor(85.5)
        metrics.record_accuracy(tensor_acc)

        assert metrics.current_val_accuracy == 85.5
        assert isinstance(metrics.current_val_accuracy, float)


class TestStateTransitionEdgeCases:
    """Tests for state transition edge cases."""

    def test_transition_to_same_stage_fails(self):
        """Transitioning to the same stage should fail."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        result = state.transition(SeedStage.TRAINING)

        assert result is False

    def test_transition_skipping_stage_fails(self):
        """Skipping stages should fail (GERMINATED -> BLENDING)."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.GERMINATED,
        )

        result = state.transition(SeedStage.BLENDING)

        assert result is False

    def test_transition_backward_fails(self):
        """Backward transitions should fail (TRAINING -> GERMINATED)."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.TRAINING,
        )

        result = state.transition(SeedStage.GERMINATED)

        assert result is False

    def test_valid_transition_updates_previous_stage(self):
        """Valid transition should update previous_stage."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.GERMINATED,
        )

        state.transition(SeedStage.TRAINING)

        assert state.stage == SeedStage.TRAINING
        assert state.previous_stage == SeedStage.GERMINATED

    def test_transition_resets_stage_baseline(self):
        """Transition should reset epochs_in_current_stage."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
            stage=SeedStage.GERMINATED,
        )
        state.metrics.epochs_in_current_stage = 10

        state.transition(SeedStage.TRAINING)

        assert state.metrics.epochs_in_current_stage == 0


class TestForceAlphaContext:
    """Tests for force_alpha context manager."""

    def test_force_alpha_temporarily_overrides(self):
        """force_alpha() should temporarily override alpha."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.set_alpha(0.8)

        with slot.force_alpha(0.0):
            assert slot.state.alpha == 0.0

        # After context, alpha restored
        assert slot.state.alpha == 0.8

    def test_force_alpha_no_seed_no_crash(self):
        """force_alpha() with no active seed should not crash."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # Should be a no-op, not raise
        with slot.force_alpha(0.0):
            pass

    def test_force_alpha_disables_schedule(self):
        """force_alpha() should disable alpha_schedule temporarily."""
        from esper.kasmina.blending import GatedBlend

        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.alpha_schedule = GatedBlend(channels=64, topology="cnn", total_steps=10)

        with slot.force_alpha(0.0):
            assert slot.alpha_schedule is None

        # Schedule restored after context
        assert slot.alpha_schedule is not None
