"""Test SeedSlot checkpoint compatibility with PyTorch 2.9."""

import pytest
import torch
import tempfile
from pathlib import Path

from esper.kasmina.slot import SeedSlot
from esper.tamiyo.policy.features import TaskConfig
from esper.leyline.alpha import AlphaAlgorithm
from esper.leyline.stages import SeedStage


class TestSeedSlotCheckpoint:
    """Test SeedSlot save/load with weights_only=True."""

    @pytest.fixture
    def slot(self):
        """Create a SeedSlot in BLENDING stage."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(
                task_type="classification",
                topology="cnn",
                baseline_loss=2.3,
                target_loss=0.5,
                typical_loss_delta_std=0.1,
                max_epochs=25,
                blending_steps=10,
            ),
        )
        slot.germinate(
            blueprint_id="norm",
            seed_id="test-seed",
            blend_algorithm_id="linear",
        )
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=10)
        return slot

    def test_extra_state_contains_only_primitives(self, slot):
        """get_extra_state() returns only primitive types."""
        extra = slot.get_extra_state()

        # Recursively check no custom types
        def check_primitives(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_primitives(v, f"{path}.{k}")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    check_primitives(v, f"{path}[{i}]")
            elif obj is None:
                pass
            elif isinstance(obj, (str, int, float, bool)):
                pass
            else:
                pytest.fail(f"Non-primitive at {path}: {type(obj).__name__}")

        check_primitives(extra)

    def test_checkpoint_roundtrip_weights_only(self, slot):
        """Save and load with weights_only=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            # Save
            torch.save(slot.state_dict(), path)

            # Load with weights_only=True (PyTorch 2.9 default)
            loaded = torch.load(path, weights_only=True)

            # Should not raise
            assert "seed_slots.r0c0._extra_state" in str(loaded) or loaded is not None

            # After loading, verify we can actually restore state
            new_slot = SeedSlot(
                slot_id="r0c0",
                channels=64,
                device="cpu",
                task_config=slot.task_config,
            )
            # Need to germinate and transition to BLENDING so slot has same structure
            new_slot.germinate(
                blueprint_id="norm",
                seed_id="test-seed",
                blend_algorithm_id="linear",
            )
            new_slot.state.transition(SeedStage.TRAINING)
            new_slot.state.transition(SeedStage.BLENDING)
            new_slot.start_blending(total_steps=10)

            # Now load the state
            new_slot.load_state_dict(loaded)

            # Verify state was restored
            assert new_slot.state is not None
            assert new_slot.state.seed_id == slot.state.seed_id
            assert new_slot.state.stage == slot.state.stage


class TestAlphaScheduleCleanup:
    """Test alpha_schedule retention rules after BLENDING."""

    def test_alpha_schedule_kept_for_gate_on_holding_transition(self):
        """alpha_schedule should remain for AlphaAlgorithm.GATE after BLENDING -> HOLDING."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(
                task_type="classification",
                topology="cnn",
                baseline_loss=2.3,
                target_loss=0.5,
                typical_loss_delta_std=0.1,
                max_epochs=25,
                blending_steps=3,
            ),
        )
        slot.germinate(
            blueprint_id="norm",
            seed_id="test-seed",
            blend_algorithm_id="gated",
            alpha_algorithm=AlphaAlgorithm.GATE,
        )
        slot.state.transition(SeedStage.TRAINING)
        slot.state.transition(SeedStage.BLENDING)
        slot.start_blending(total_steps=3)

        # Verify schedule exists during BLENDING
        assert slot.alpha_schedule is not None

        # Force transition to HOLDING
        slot.state.alpha = 1.0
        slot.state.transition(SeedStage.HOLDING)
        slot._on_blending_complete()  # Cleanup hook

        # Schedule must persist for GATE (forward requires it).
        assert slot.alpha_schedule is not None
        assert slot.state.alpha == 1.0


class TestExtraStateValidation:
    """Test set_extra_state() fail-fast behavior for corrupt checkpoints."""

    @pytest.fixture
    def slot_with_state(self):
        """Create a SeedSlot with state for testing."""
        slot = SeedSlot(
            slot_id="r0c0",
            channels=64,
            device="cpu",
            task_config=TaskConfig(
                task_type="classification",
                topology="cnn",
                baseline_loss=2.3,
                target_loss=0.5,
                typical_loss_delta_std=0.1,
                max_epochs=25,
                blending_steps=10,
            ),
        )
        slot.germinate(
            blueprint_id="norm",
            seed_id="test-seed",
            blend_algorithm_id="linear",
        )
        return slot

    def test_rejects_missing_version(self, slot_with_state):
        """Missing _extra_state_version must raise KeyError."""
        slot = slot_with_state
        extra = slot.get_extra_state()
        del extra["_extra_state_version"]

        new_slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu")
        with pytest.raises(KeyError):
            new_slot.set_extra_state(extra)

    def test_rejects_schema_version_mismatch(self, slot_with_state):
        """Old checkpoint version must fail explicitly."""
        slot = slot_with_state
        extra = slot.get_extra_state()
        extra["_extra_state_version"] = 0  # Simulate old checkpoint

        new_slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu")
        with pytest.raises(ValueError, match="schema mismatch"):
            new_slot.set_extra_state(extra)

    def test_rejects_missing_isolate_gradients(self, slot_with_state):
        """Missing isolate_gradients must raise KeyError."""
        slot = slot_with_state
        extra = slot.get_extra_state()
        del extra["isolate_gradients"]

        new_slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu")
        with pytest.raises(KeyError):
            new_slot.set_extra_state(extra)

    def test_rejects_missing_blend_algorithm_id(self, slot_with_state):
        """Missing blend_algorithm_id must raise KeyError."""
        slot = slot_with_state
        extra = slot.get_extra_state()
        del extra["blend_algorithm_id"]

        new_slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu")
        with pytest.raises(KeyError):
            new_slot.set_extra_state(extra)

    def test_isolate_gradients_roundtrip_true(self, slot_with_state):
        """isolate_gradients=True must survive checkpoint exactly."""
        slot = slot_with_state
        slot.isolate_gradients = True

        extra = slot.get_extra_state()

        new_slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu")
        new_slot.set_extra_state(extra)

        assert new_slot.isolate_gradients is True

    def test_isolate_gradients_roundtrip_false(self, slot_with_state):
        """isolate_gradients=False must survive checkpoint exactly."""
        slot = slot_with_state
        slot.isolate_gradients = False

        extra = slot.get_extra_state()

        new_slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu")
        new_slot.set_extra_state(extra)

        assert new_slot.isolate_gradients is False

    def test_all_required_fields_present_in_get_extra_state(self, slot_with_state):
        """Verify get_extra_state() includes all required fields."""
        slot = slot_with_state
        extra = slot.get_extra_state()

        required_fields = [
            "_extra_state_version",
            "isolate_gradients",
            "blend_algorithm_id",
            "blend_tempo_epochs",
            "blend_alpha_target",
            "resolved_topology",
            "seed_state",
            "alpha_schedule_config",
        ]
        for field in required_fields:
            assert field in extra, f"Missing required field: {field}"

    def test_dormant_slot_roundtrip(self):
        """DORMANT slot with seed_state=None must roundtrip correctly.

        This tests the seed_state asymmetry fix: get_extra_state() must always
        include the seed_state key (with None value for DORMANT slots) so that
        set_extra_state() can access it without KeyError.
        """
        # Create DORMANT slot (no germinate call)
        slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu")
        assert slot.state is None, "DORMANT slots should have state=None"

        # Save checkpoint
        extra = slot.get_extra_state()

        # seed_state key must exist even when state is None
        assert "seed_state" in extra, "seed_state key must always be present"
        assert extra["seed_state"] is None, "DORMANT slot should have seed_state=None"

        # Restore into new slot
        new_slot = SeedSlot(slot_id="r0c0", channels=64, device="cpu")
        new_slot.set_extra_state(extra)

        # Verify state remained None after restore
        assert new_slot.state is None, "Restored DORMANT slot should have state=None"
