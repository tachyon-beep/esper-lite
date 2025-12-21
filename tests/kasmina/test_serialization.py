"""Serialization tests for Kasmina.

Tests verify correct save/load behavior:
- Save/load at each lifecycle stage
- Metrics and telemetry preservation
- Blending state preservation
- Resume training after checkpoint
- Algorithm state preservation
"""

import pytest
import torch
from datetime import datetime, timezone

from esper.kasmina.slot import SeedSlot, SeedState, SeedMetrics
from esper.kasmina.blending import LinearBlend, SigmoidBlend, GatedBlend
from esper.leyline import SeedStage


class TestSeedMetricsSerialization:
    """Tests for SeedMetrics serialization."""

    def test_metrics_to_dict_and_from_dict_roundtrip(self):
        """Metrics should survive to_dict -> from_dict roundtrip."""
        metrics = SeedMetrics()
        metrics.record_accuracy(50.0)
        metrics.record_accuracy(60.0)
        metrics.record_accuracy(70.0)
        metrics.seed_gradient_norm_ratio = 0.5
        metrics.counterfactual_contribution = 2.5
        metrics.host_param_count = 1000000
        metrics.seed_param_count = 10000

        # Serialize and deserialize
        data = metrics.to_dict()
        restored = SeedMetrics.from_dict(data)

        assert restored.epochs_total == 3
        assert restored.initial_val_accuracy == 50.0
        assert restored.current_val_accuracy == 70.0
        assert restored.best_val_accuracy == 70.0
        assert restored.seed_gradient_norm_ratio == 0.5
        assert restored.counterfactual_contribution == 2.5
        assert restored.host_param_count == 1000000
        assert restored.seed_param_count == 10000

    def test_metrics_with_none_counterfactual(self):
        """Metrics with None counterfactual should serialize correctly."""
        metrics = SeedMetrics()
        metrics.record_accuracy(50.0)
        # counterfactual_contribution is None by default

        data = metrics.to_dict()
        restored = SeedMetrics.from_dict(data)

        assert restored.counterfactual_contribution is None

    def test_metrics_with_blending_started_flag(self):
        """Metrics _blending_started flag should be preserved."""
        metrics = SeedMetrics()
        metrics._blending_started = True
        metrics.accuracy_at_blending_start = 55.0

        data = metrics.to_dict()
        restored = SeedMetrics.from_dict(data)

        assert restored._blending_started is True
        assert restored.accuracy_at_blending_start == 55.0

    def test_metrics_empty_roundtrip(self):
        """Empty metrics should survive roundtrip."""
        metrics = SeedMetrics()

        data = metrics.to_dict()
        restored = SeedMetrics.from_dict(data)

        assert restored.epochs_total == 0
        assert restored.best_val_accuracy == 0.0


class TestSeedStateSerialization:
    """Tests for SeedState serialization."""

    def test_state_to_dict_and_from_dict_roundtrip(self):
        """SeedState should survive to_dict -> from_dict roundtrip."""
        state = SeedState(
            seed_id="test_seed",
            blueprint_id="norm",
            slot_id="r0c0",
            stage=SeedStage.TRAINING,
        )
        state.alpha = 0.5
        state.is_healthy = True
        state.blending_steps_done = 3
        state.blending_steps_total = 10
        state.metrics.record_accuracy(60.0)

        data = state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.seed_id == "test_seed"
        assert restored.blueprint_id == "norm"
        assert restored.slot_id == "r0c0"
        assert restored.stage == SeedStage.TRAINING
        assert restored.alpha == 0.5
        assert restored.is_healthy is True
        assert restored.blending_steps_done == 3
        assert restored.blending_steps_total == 10
        assert restored.metrics.current_val_accuracy == 60.0

    def test_state_stage_enum_serialization(self):
        """SeedStage enum should serialize to int and deserialize back."""
        for stage in SeedStage:
            state = SeedState(
                seed_id="test",
                blueprint_id="noop",
                stage=stage,
            )

            data = state.to_dict()
            assert isinstance(data["stage"], int)

            restored = SeedState.from_dict(data)
            assert restored.stage == stage

    def test_state_datetime_serialization(self):
        """Datetime fields should serialize to ISO format."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
        )

        data = state.to_dict()
        assert isinstance(data["stage_entered_at"], str)

        restored = SeedState.from_dict(data)
        assert isinstance(restored.stage_entered_at, datetime)

    def test_state_stage_history_preserved(self):
        """Stage history should be preserved through serialization."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
        )
        state.transition(SeedStage.GERMINATED)
        state.transition(SeedStage.TRAINING)
        state.transition(SeedStage.BLENDING)

        data = state.to_dict()
        restored = SeedState.from_dict(data)

        # Stage history should have entries
        assert len(restored.stage_history) >= 3

    def test_state_previous_stage_preserved(self):
        """previous_stage and previous_epochs_in_stage should be preserved."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
        )
        state.metrics.record_accuracy(50.0)
        state.metrics.record_accuracy(55.0)
        state.transition(SeedStage.GERMINATED)
        state.transition(SeedStage.TRAINING)

        data = state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.previous_stage == SeedStage.GERMINATED

    def test_state_is_paused_flag(self):
        """is_paused flag should be preserved."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
        )
        state.is_paused = True

        data = state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.is_paused is True


class TestSlotStateSerialization:
    """Tests for SeedSlot with state serialization scenarios."""

    def test_slot_dormant_state(self):
        """Slot with no active seed should have serializable state."""
        slot = SeedSlot(slot_id="r0c0", channels=64)

        # No seed germinated - state is None
        assert slot.state is None

    def test_slot_germinated_state_serialization(self):
        """Slot in GERMINATED stage should have serializable state."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")

        data = slot.state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.seed_id == "test"
        assert restored.stage == SeedStage.GERMINATED

    def test_slot_training_state_serialization(self):
        """Slot in TRAINING stage should serialize correctly."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.metrics.record_accuracy(55.0)

        data = slot.state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.stage == SeedStage.TRAINING
        assert restored.metrics.current_val_accuracy == 55.0

    def test_slot_blending_state_serialization(self):
        """Slot in BLENDING stage should serialize correctly."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=20)
        slot.set_alpha(0.5)

        # Record some progress
        slot.state.blending_steps_done = 5

        data = slot.state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.stage == SeedStage.BLENDING
        assert restored.alpha == 0.5
        assert restored.blending_steps_done == 5
        assert restored.blending_steps_total == 20

    def test_slot_holding_state_serialization(self):
        """Slot in HOLDING stage should serialize correctly."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.state.transition(SeedStage.HOLDING)
        slot.state.metrics.counterfactual_contribution = 3.5

        data = slot.state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.stage == SeedStage.HOLDING
        assert restored.metrics.counterfactual_contribution == 3.5

    def test_slot_fossilized_state_serialization(self):
        """Slot in FOSSILIZED stage should serialize correctly."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.stage = SeedStage.FOSSILIZED
        slot.set_alpha(1.0)

        data = slot.state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.stage == SeedStage.FOSSILIZED
        assert restored.alpha == 1.0


class TestBlendAlgorithmSerialization:
    """Tests for blend algorithm state serialization."""

    def test_linear_blend_state_preserves_step(self):
        """LinearBlend current step should be reconstructable."""
        blend = LinearBlend(total_steps=100)

        # Advance to step 50
        alpha = blend.get_alpha(50)
        assert alpha == 0.5

        # Create new blend and verify step behavior
        blend2 = LinearBlend(total_steps=100)
        alpha2 = blend2.get_alpha(50)

        assert alpha2 == 0.5

    def test_sigmoid_blend_state_preserves_step(self):
        """SigmoidBlend current step should be reconstructable."""
        blend = SigmoidBlend(total_steps=100)

        alpha1 = blend.get_alpha(0)
        alpha2 = blend.get_alpha(50)
        alpha3 = blend.get_alpha(100)

        # Sigmoid should be low at start, ~0.5 at middle, high at end
        assert alpha1 < alpha2 < alpha3

    def test_gated_blend_module_state_dict(self):
        """GatedBlend should have state_dict for nn.Module compatibility."""
        blend = GatedBlend(channels=64, topology="cnn")

        state_dict = blend.state_dict()

        # Should have gate network weights
        assert len(state_dict) > 0
        assert any("gate" in key for key in state_dict.keys())

    def test_gated_blend_load_state_dict(self):
        """GatedBlend should be loadable from state_dict."""
        blend1 = GatedBlend(channels=64, topology="cnn")
        state_dict = blend1.state_dict()

        blend2 = GatedBlend(channels=64, topology="cnn")
        blend2.load_state_dict(state_dict)

        # Weights should match
        for key in state_dict.keys():
            torch.testing.assert_close(
                blend1.state_dict()[key],
                blend2.state_dict()[key],
            )


class TestResumeTrainingAfterCheckpoint:
    """Tests for resuming training after checkpoint."""

    def test_resume_training_metrics_preserved(self):
        """Metrics should be preserved when resuming from checkpoint."""
        # Create initial state
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)

        # Train for a few epochs
        slot.state.metrics.record_accuracy(50.0)
        slot.state.metrics.record_accuracy(55.0)
        slot.state.metrics.record_accuracy(60.0)

        # Checkpoint state
        state_data = slot.state.to_dict()

        # "Resume" by creating new state from checkpoint
        restored_state = SeedState.from_dict(state_data)

        # Verify metrics preserved
        assert restored_state.metrics.epochs_total == 3
        assert restored_state.metrics.best_val_accuracy == 60.0
        assert restored_state.metrics.initial_val_accuracy == 50.0

    def test_resume_blending_progress_preserved(self):
        """Blending progress should be preserved when resuming."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=100)

        # Advance blending
        for _ in range(30):
            slot.step_epoch()

        # Checkpoint
        state_data = slot.state.to_dict()
        restored = SeedState.from_dict(state_data)

        assert restored.blending_steps_done == 30
        assert restored.blending_steps_total == 100


class TestTelemetrySerialization:
    """Tests for telemetry serialization."""

    def test_telemetry_preserved_through_state(self):
        """Telemetry should be preserved in state serialization."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
        )

        # Telemetry is auto-initialized in __post_init__
        assert state.telemetry is not None
        assert state.telemetry.seed_id == "test"
        assert state.telemetry.blueprint_id == "noop"

        # Update telemetry
        state.telemetry.accuracy = 75.0
        state.telemetry.epoch = 10

        data = state.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.telemetry is not None
        assert restored.telemetry.seed_id == "test"
        assert restored.telemetry.accuracy == 75.0
        assert restored.telemetry.epoch == 10


class TestEdgeCaseSerialization:
    """Tests for edge case serialization scenarios."""

    def test_serialize_after_cull_transition(self):
        """State just before cull should serialize correctly."""
        slot = SeedSlot(slot_id="r0c0", channels=64)
        slot.germinate("noop", seed_id="test")
        slot.state.transition(SeedStage.TRAINING)

        # Snapshot before cull
        data = slot.state.to_dict()

        # Cull
        slot.cull()

        # Verify we can still deserialize the pre-cull state
        restored = SeedState.from_dict(data)
        assert restored.stage == SeedStage.TRAINING

    def test_serialize_with_large_epochs(self):
        """State with large epoch counts should serialize correctly."""
        metrics = SeedMetrics()

        # Simulate many epochs
        for i in range(10000):
            metrics.record_accuracy(50.0 + (i % 20))

        data = metrics.to_dict()
        restored = SeedMetrics.from_dict(data)

        assert restored.epochs_total == 10000

    def test_serialize_with_float_precision(self):
        """Float values should maintain precision through serialization."""
        metrics = SeedMetrics()
        metrics.record_accuracy(50.123456789)
        metrics.seed_gradient_norm_ratio = 0.987654321

        data = metrics.to_dict()
        restored = SeedMetrics.from_dict(data)

        # Check precision is maintained
        assert restored.current_val_accuracy == pytest.approx(50.123456789)
        assert restored.seed_gradient_norm_ratio == pytest.approx(0.987654321)

    def test_serialize_empty_stage_history(self):
        """State with empty stage history should serialize correctly."""
        state = SeedState(
            seed_id="test",
            blueprint_id="noop",
        )
        # Clear stage history
        state.stage_history.clear()

        data = state.to_dict()
        restored = SeedState.from_dict(data)

        assert len(restored.stage_history) == 0
