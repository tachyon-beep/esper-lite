"""Tests for SeedSlot checkpoint serialization.

These tests verify the fix for GatedBlend checkpoint persistence.
See docs/plans/2025-12-16-tolaria-kasmina-remediation.md for full investigation.
"""

import tempfile
import torch
import torch.nn as nn

from esper.kasmina.slot import SeedSlot, SeedState
from esper.kasmina.blending import GatedBlend
from esper.leyline import SeedStage


class TestGatedBlendSerialization:
    """Test that GatedBlend learned weights survive checkpoint round-trip.

    The key test is that we do NOT set _blend_algorithm_id on the loading slot.
    It must be restored from the checkpoint config.
    """

    def test_gatedblend_algorithm_id_restored_from_checkpoint(self):
        """_blend_algorithm_id must be restored from checkpoint, not default to sigmoid."""
        # Setup: Create slot with GatedBlend
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot._blend_algorithm_id = "gated"
        slot.seed = nn.Linear(64, 64)
        # Note: blueprint_id is required by SeedState but was omitted in the plan examples
        slot.state = SeedState(seed_id="test_seed", blueprint_id="test_blueprint", slot_id="test")
        slot.state.stage = SeedStage.BLENDING
        slot.start_blending(total_steps=10)

        assert isinstance(slot.alpha_schedule, GatedBlend), "Setup failed"

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(slot.state_dict(), f.name)

            # Create fresh slot - DO NOT set _blend_algorithm_id
            # This tests that it's restored from checkpoint
            new_slot = SeedSlot(slot_id="test", channels=64, device="cpu")
            new_slot.seed = nn.Linear(64, 64)
            # Note: NOT setting new_slot._blend_algorithm_id

            state_dict = torch.load(f.name, weights_only=True)
            missing, unexpected = new_slot.load_state_dict(state_dict, strict=False)

        # Verify no unexpected keys (would indicate orphaned GatedBlend weights)
        assert len(unexpected) == 0, f"Unexpected keys in checkpoint: {unexpected}"

        # The bug: without the fix, this would be SigmoidBlend
        assert isinstance(new_slot.alpha_schedule, GatedBlend), \
            f"Expected GatedBlend but got {type(new_slot.alpha_schedule).__name__}. " \
            "This means _blend_algorithm_id was not restored from checkpoint."

    def test_gatedblend_strict_load_succeeds(self):
        """Verify strict=True loading succeeds (no orphan keys after fix)."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot._blend_algorithm_id = "gated"
        slot.seed = nn.Linear(64, 64)
        slot.state = SeedState(seed_id="test_seed", blueprint_id="test_blueprint", slot_id="test")
        slot.state.stage = SeedStage.BLENDING
        slot.start_blending(total_steps=10)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(slot.state_dict(), f.name)

            new_slot = SeedSlot(slot_id="test", channels=64, device="cpu")
            new_slot.seed = nn.Linear(64, 64)

            state_dict = torch.load(f.name, weights_only=True)
            # This should succeed with strict=True after the fix
            # Before the fix, this would fail with RuntimeError about unexpected keys
            new_slot.load_state_dict(state_dict, strict=True)

        assert isinstance(new_slot.alpha_schedule, GatedBlend)

    def test_gatedblend_weights_persist_through_checkpoint(self):
        """GatedBlend gate network weights must survive save/load cycle."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot._blend_algorithm_id = "gated"
        slot.seed = nn.Linear(64, 64)
        slot.state = SeedState(seed_id="test_seed", blueprint_id="test_blueprint", slot_id="test")
        slot.state.stage = SeedStage.BLENDING
        slot.start_blending(total_steps=10)

        # Modify weights to known values (simulates training)
        with torch.no_grad():
            for param in slot.alpha_schedule.gate.parameters():
                param.fill_(0.42)

        original_weights = {
            k: v.clone() for k, v in slot.alpha_schedule.state_dict().items()
        }

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(slot.state_dict(), f.name)

            new_slot = SeedSlot(slot_id="test", channels=64, device="cpu")
            new_slot.seed = nn.Linear(64, 64)

            state_dict = torch.load(f.name, weights_only=True)
            new_slot.load_state_dict(state_dict, strict=False)

        # Verify weights restored
        for key, original_value in original_weights.items():
            loaded_value = new_slot.alpha_schedule.state_dict()[key]
            torch.testing.assert_close(
                loaded_value, original_value,
                msg=f"GatedBlend weight {key} was not restored correctly"
            )

    def test_non_blending_slot_loads_without_alpha_schedule(self):
        """Slot not in BLENDING stage should not create alpha_schedule on load."""
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot.seed = nn.Linear(64, 64)
        slot.state = SeedState(seed_id="test_seed", blueprint_id="test_blueprint", slot_id="test")
        slot.state.stage = SeedStage.TRAINING  # Not BLENDING

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(slot.state_dict(), f.name)

            new_slot = SeedSlot(slot_id="test", channels=64, device="cpu")
            new_slot.seed = nn.Linear(64, 64)

            state_dict = torch.load(f.name, weights_only=True)
            new_slot.load_state_dict(state_dict, strict=False)

        assert new_slot.alpha_schedule is None

    def test_alpha_schedule_serialization_uses_protocol_attributes(self):
        """Verify serialization accesses protocol attributes without getattr.

        This test ensures that when alpha_schedule is set, it satisfies
        AlphaScheduleProtocol and serialization can access attributes directly.
        Lines 2485-2489 in slot.py now use direct attribute access instead of getattr.
        """
        slot = SeedSlot(slot_id="test", channels=64, device="cpu")
        slot._blend_algorithm_id = "gated"
        slot.seed = nn.Linear(64, 64)
        slot.state = SeedState(seed_id="test_seed", blueprint_id="test_blueprint", slot_id="test")
        slot.state.stage = SeedStage.BLENDING
        slot.start_blending(total_steps=10)

        assert slot.alpha_schedule is not None

        # get_extra_state() should access attributes directly (no getattr fallbacks)
        extra_state = slot.get_extra_state()

        # Verify the alpha_schedule_config was serialized correctly
        assert "alpha_schedule_config" in extra_state
        config = extra_state["alpha_schedule_config"]
        assert config is not None
        assert config["algorithm_id"] == "gated"
        assert config["total_steps"] == 10
        assert config["current_step"] == 0  # Initial step

        # Verify the attributes exist on the alpha_schedule object
        assert slot.alpha_schedule.algorithm_id == "gated"
        assert slot.alpha_schedule.total_steps == 10
        assert slot.alpha_schedule._current_step == 0
