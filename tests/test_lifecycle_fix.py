"""Tests for lifecycle state machine fix."""

import pytest
from esper.kasmina.slot import SeedState
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
