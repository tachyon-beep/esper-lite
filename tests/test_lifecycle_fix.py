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
