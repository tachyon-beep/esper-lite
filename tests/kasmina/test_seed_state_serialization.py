"""Test SeedState serialization for PyTorch 2.9 weights_only=True."""

import pytest
from datetime import datetime, timezone
from collections import deque

from esper.kasmina.slot import SeedState, SeedMetrics
from esper.leyline.stages import SeedStage


class TestSeedStateToDict:
    """Test SeedState.to_dict() produces only primitives."""

    def test_to_dict_returns_primitives_only(self):
        """to_dict() output contains no custom types."""
        state = SeedState(
            seed_id="test-seed",
            blueprint_id="norm",
            slot_id="r0c0",
            stage=SeedStage.TRAINING,
        )

        result = state.to_dict()

        # Must be a plain dict
        assert isinstance(result, dict)
        # Stage must be int, not Enum
        assert isinstance(result["stage"], int)
        assert result["stage"] == SeedStage.TRAINING.value
        # Datetime must be string
        assert isinstance(result["stage_entered_at"], str)
        # Stage history must be list, not deque
        assert isinstance(result["stage_history"], list)

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict() -> from_dict() preserves all state."""
        original = SeedState(
            seed_id="test-seed",
            blueprint_id="norm",
            slot_id="r0c0",
            stage=SeedStage.BLENDING,
            previous_stage=SeedStage.TRAINING,
        )
        original.alpha = 0.75
        original.stage_history.append((SeedStage.GERMINATED, datetime.now(timezone.utc)))

        data = original.to_dict()
        restored = SeedState.from_dict(data)

        assert restored.seed_id == original.seed_id
        assert restored.blueprint_id == original.blueprint_id
        assert restored.slot_id == original.slot_id
        assert restored.stage == original.stage
        assert restored.previous_stage == original.previous_stage
        assert restored.alpha == original.alpha
        assert len(restored.stage_history) == len(original.stage_history)

    def test_to_dict_handles_none_values(self):
        """to_dict() handles None values gracefully."""
        state = SeedState(
            seed_id="test-seed",
            blueprint_id="norm",
            slot_id="r0c0",
            stage=SeedStage.GERMINATED,
            previous_stage=None,  # None value
        )

        data = state.to_dict()
        restored = SeedState.from_dict(data)

        # None previous_stage should be preserved
        assert restored.previous_stage is None
        assert data["previous_stage"] is None

    def test_to_dict_converts_deque_to_list(self):
        """to_dict() converts deque to list for stage_history."""
        state = SeedState(
            seed_id="test-seed",
            blueprint_id="norm",
            slot_id="r0c0",
            stage=SeedStage.TRAINING,
        )
        state.stage_history.append((SeedStage.GERMINATED, datetime.now(timezone.utc)))
        state.stage_history.append((SeedStage.TRAINING, datetime.now(timezone.utc)))

        data = state.to_dict()

        # Must be list, not deque
        assert isinstance(data["stage_history"], list)
        assert len(data["stage_history"]) == 2
        # Each entry should be tuple of (int, str)
        for stage_val, ts_str in data["stage_history"]:
            assert isinstance(stage_val, int)
            assert isinstance(ts_str, str)
