"""Test SeedState dataclass safety."""
import pytest

from esper.kasmina.slot import SeedState
from esper.leyline import SeedStage


class TestSeedStateKwOnly:
    """Verify SeedState requires keyword arguments."""

    def test_positional_args_rejected(self):
        """SeedState should reject positional arguments."""
        with pytest.raises(TypeError, match="positional"):
            # This should fail - positional args not allowed
            SeedState("seed-1", "norm", "slot-1")

    def test_keyword_args_accepted(self):
        """SeedState should accept keyword arguments."""
        state = SeedState(
            seed_id="seed-1",
            blueprint_id="norm",
            slot_id="slot-1",
            stage=SeedStage.DORMANT,
        )

        assert state.seed_id == "seed-1"
        assert state.blueprint_id == "norm"
        assert state.slot_id == "slot-1"
        assert state.stage == SeedStage.DORMANT

    def test_minimum_required_fields(self):
        """SeedState should work with only required fields."""
        state = SeedState(
            seed_id="test-seed",
            blueprint_id="test-blueprint",
        )

        assert state.seed_id == "test-seed"
        assert state.blueprint_id == "test-blueprint"
        # Optional fields should have defaults
        assert state.slot_id == ""
        assert state.stage == SeedStage.DORMANT

    def test_mixed_positional_keyword_rejected(self):
        """SeedState should reject mixed positional and keyword arguments."""
        with pytest.raises(TypeError):
            # Even mixing positional and keyword should fail
            SeedState("seed-1", blueprint_id="norm")
