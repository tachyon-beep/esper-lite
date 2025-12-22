"""Tests for factored action lookup tables."""

import pytest

from esper.leyline.factored_actions import (
    BlueprintAction,
    BlendAction,
    LifecycleOp,
    OP_NAMES,
    BLUEPRINT_IDS,
    BLEND_IDS,
    OP_GERMINATE,
    OP_FOSSILIZE,
    OP_PRUNE,
    OP_WAIT,
    OP_ADVANCE,
)


class TestLookupTableSync:
    """Verify lookup tables stay in sync with enum definitions."""

    def test_op_names_match_lifecycle_op_enum(self):
        """OP_NAMES must match LifecycleOp enum names in order."""
        for i, op in enumerate(LifecycleOp):
            assert OP_NAMES[i] == op.name, f"OP_NAMES[{i}] = {OP_NAMES[i]} != {op.name}"

    def test_op_names_length_matches_enum(self):
        """OP_NAMES length must equal LifecycleOp enum length."""
        assert len(OP_NAMES) == len(LifecycleOp)

    def test_blueprint_ids_match_enum_method(self):
        """BLUEPRINT_IDS must match BlueprintAction.to_blueprint_id() for all values."""
        for i, bp in enumerate(BlueprintAction):
            assert BLUEPRINT_IDS[i] == bp.to_blueprint_id(), (
                f"BLUEPRINT_IDS[{i}] = {BLUEPRINT_IDS[i]} != {bp.to_blueprint_id()}"
            )

    def test_blueprint_ids_length_matches_enum(self):
        """BLUEPRINT_IDS length must equal BlueprintAction enum length."""
        assert len(BLUEPRINT_IDS) == len(BlueprintAction)

    def test_blend_ids_match_enum_method(self):
        """BLEND_IDS must match BlendAction.to_algorithm_id() for all values."""
        for i, blend in enumerate(BlendAction):
            assert BLEND_IDS[i] == blend.to_algorithm_id(), (
                f"BLEND_IDS[{i}] = {BLEND_IDS[i]} != {blend.to_algorithm_id()}"
            )

    def test_blend_ids_length_matches_enum(self):
        """BLEND_IDS length must equal BlendAction enum length."""
        assert len(BLEND_IDS) == len(BlendAction)


class TestOpIndexConstants:
    """Verify OP index constants match enum values."""

    def test_op_wait_matches_enum(self):
        assert OP_WAIT == LifecycleOp.WAIT.value

    def test_op_germinate_matches_enum(self):
        assert OP_GERMINATE == LifecycleOp.GERMINATE.value

    def test_op_prune_matches_enum(self):
        assert OP_PRUNE == LifecycleOp.PRUNE.value

    def test_op_fossilize_matches_enum(self):
        assert OP_FOSSILIZE == LifecycleOp.FOSSILIZE.value

    def test_op_advance_matches_enum(self):
        assert OP_ADVANCE == LifecycleOp.ADVANCE.value
