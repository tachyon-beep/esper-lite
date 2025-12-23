"""Tests for StageSchema - centralized stage encoding contract.

Validates:
- All SeedStage values have index mappings
- One-hot encoding produces correct dimensions
- Reserved/invalid values are rejected
- Bidirectional index mappings are consistent
"""

import pytest

from esper.leyline.stages import SeedStage
from esper.leyline.stage_schema import (
    STAGE_SCHEMA_VERSION,
    VALID_STAGES,
    NUM_STAGES,
    STAGE_TO_INDEX,
    INDEX_TO_STAGE,
    VALID_STAGE_VALUES,
    RESERVED_STAGE_VALUES,
    stage_to_one_hot,
    stage_to_index,
    validate_stage_value,
)


class TestStageSchemaConstants:
    """Test schema constants are correctly defined."""

    def test_schema_version_is_positive_int(self):
        """Schema version should be a positive integer."""
        assert isinstance(STAGE_SCHEMA_VERSION, int)
        assert STAGE_SCHEMA_VERSION >= 1

    def test_num_stages_matches_valid_stages(self):
        """NUM_STAGES should match length of VALID_STAGES tuple."""
        assert NUM_STAGES == len(VALID_STAGES)

    def test_valid_stages_are_seed_stage_enums(self):
        """All VALID_STAGES should be SeedStage enum members."""
        for stage in VALID_STAGES:
            assert isinstance(stage, SeedStage)

    def test_reserved_value_5_is_marked(self):
        """Reserved stage value 5 (was SHADOWING) should be in RESERVED_STAGE_VALUES."""
        assert 5 in RESERVED_STAGE_VALUES

    def test_reserved_values_not_in_valid_values(self):
        """Reserved values should not appear in valid values."""
        for reserved in RESERVED_STAGE_VALUES:
            assert reserved not in VALID_STAGE_VALUES

    def test_valid_stages_excludes_gap_at_5(self):
        """Valid stage values should not include the reserved gap at value 5."""
        assert 5 not in VALID_STAGE_VALUES

    def test_valid_stages_includes_all_seed_stages(self):
        """All defined SeedStage enum values should be in VALID_STAGES."""
        for stage in SeedStage:
            assert stage in VALID_STAGES, f"SeedStage.{stage.name} missing from VALID_STAGES"


class TestStageToIndex:
    """Test STAGE_TO_INDEX mapping."""

    def test_covers_all_valid_stages(self):
        """Every valid SeedStage should have an index mapping."""
        for stage in VALID_STAGES:
            assert stage.value in STAGE_TO_INDEX, f"SeedStage.{stage.name} missing from STAGE_TO_INDEX"

    def test_indices_are_contiguous(self):
        """Indices should be contiguous from 0 to NUM_STAGES-1."""
        indices = sorted(STAGE_TO_INDEX.values())
        assert indices == list(range(NUM_STAGES))

    def test_index_to_stage_is_inverse(self):
        """INDEX_TO_STAGE should be inverse of STAGE_TO_INDEX."""
        for stage_val, idx in STAGE_TO_INDEX.items():
            assert INDEX_TO_STAGE[idx] == stage_val

    def test_reserved_value_5_not_in_mapping(self):
        """Reserved value 5 should not have an index mapping."""
        assert 5 not in STAGE_TO_INDEX


class TestStageToOneHot:
    """Test one-hot encoding function."""

    def test_one_hot_dimensions(self):
        """One-hot output should have NUM_STAGES dimensions."""
        for stage in VALID_STAGES:
            one_hot = stage_to_one_hot(stage.value)
            assert len(one_hot) == NUM_STAGES

    def test_one_hot_has_single_one(self):
        """One-hot should have exactly one 1.0 and rest 0.0."""
        for stage in VALID_STAGES:
            one_hot = stage_to_one_hot(stage.value)
            assert sum(one_hot) == 1.0
            assert one_hot.count(1.0) == 1
            assert one_hot.count(0.0) == NUM_STAGES - 1

    def test_one_hot_correct_index(self):
        """One-hot should have 1.0 at the correct index."""
        for stage in VALID_STAGES:
            one_hot = stage_to_one_hot(stage.value)
            expected_idx = STAGE_TO_INDEX[stage.value]
            assert one_hot[expected_idx] == 1.0

    def test_reserved_stage_5_rejected(self):
        """Reserved stage value 5 should be rejected."""
        with pytest.raises(ValueError, match="Invalid stage value 5"):
            stage_to_one_hot(5)

    def test_out_of_range_positive_rejected(self):
        """Out of range positive values should be rejected."""
        with pytest.raises(ValueError, match="Invalid stage value 999"):
            stage_to_one_hot(999)

    def test_out_of_range_negative_rejected(self):
        """Negative values should be rejected."""
        with pytest.raises(ValueError, match="Invalid stage value -1"):
            stage_to_one_hot(-1)

    def test_all_seed_stage_values_produce_valid_one_hot(self):
        """All SeedStage enum values should produce valid one-hot encoding."""
        for stage in SeedStage:
            one_hot = stage_to_one_hot(stage.value)
            assert len(one_hot) == NUM_STAGES
            assert sum(one_hot) == 1.0


class TestStageToIndex:
    """Test stage_to_index function."""

    def test_returns_correct_index(self):
        """stage_to_index should return correct contiguous index."""
        for stage in VALID_STAGES:
            idx = stage_to_index(stage.value)
            assert idx == STAGE_TO_INDEX[stage.value]

    def test_reserved_stage_5_rejected(self):
        """Reserved stage value 5 should be rejected."""
        with pytest.raises(ValueError, match="Invalid stage value 5"):
            stage_to_index(5)

    def test_out_of_range_rejected(self):
        """Out of range values should be rejected."""
        with pytest.raises(ValueError):
            stage_to_index(999)


class TestValidateStageValue:
    """Test validate_stage_value function."""

    def test_valid_stages_pass(self):
        """All valid stage values should pass validation."""
        for stage in VALID_STAGES:
            # Should not raise
            validate_stage_value(stage.value)

    def test_reserved_stage_rejected_with_context(self):
        """Reserved stage should be rejected with context in error message."""
        with pytest.raises(ValueError, match="reserved/retired"):
            validate_stage_value(5, context="TestContext")

    def test_invalid_stage_rejected(self):
        """Invalid stage values should be rejected."""
        with pytest.raises(ValueError, match="Invalid stage value"):
            validate_stage_value(999)

    def test_context_appears_in_error(self):
        """Context string should appear in error message."""
        with pytest.raises(ValueError, match="SeedTelemetry.stage"):
            validate_stage_value(5, context="SeedTelemetry.stage")


class TestSchemaConsistency:
    """Test consistency between schema components."""

    def test_stage_index_roundtrip(self):
        """stage_value -> index -> stage_value should roundtrip."""
        for stage in VALID_STAGES:
            idx = STAGE_TO_INDEX[stage.value]
            roundtrip_val = INDEX_TO_STAGE[idx]
            assert roundtrip_val == stage.value

    def test_one_hot_decoding(self):
        """One-hot encoding should be decodable back to stage value."""
        for stage in VALID_STAGES:
            one_hot = stage_to_one_hot(stage.value)
            # Find the index with 1.0
            decoded_idx = one_hot.index(1.0)
            decoded_val = INDEX_TO_STAGE[decoded_idx]
            assert decoded_val == stage.value

    def test_valid_stage_values_match_stage_to_index_keys(self):
        """VALID_STAGE_VALUES should exactly match STAGE_TO_INDEX keys."""
        assert VALID_STAGE_VALUES == frozenset(STAGE_TO_INDEX.keys())
