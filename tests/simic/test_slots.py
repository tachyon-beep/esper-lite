"""Tests for slot ordering and validation."""

from esper.leyline.slot_id import validate_slot_ids, SlotIdError
import pytest


def test_validate_slot_ids_returns_canonical_order() -> None:
    """validate_slot_ids sorts slots in row-major order."""
    assert validate_slot_ids(["r0c2", "r0c0"]) == ["r0c0", "r0c2"]
    assert validate_slot_ids(["r0c1"]) == ["r0c1"]
    assert validate_slot_ids(["r0c0", "r0c1", "r0c2"]) == ["r0c0", "r0c1", "r0c2"]
    assert validate_slot_ids(["r1c0", "r0c2", "r0c0"]) == ["r0c0", "r0c2", "r1c0"]


def test_validate_slot_ids_rejects_invalid_format() -> None:
    """validate_slot_ids raises SlotIdError for invalid formats."""
    with pytest.raises(SlotIdError, match="Invalid slot ID format"):
        validate_slot_ids(["unknown", "r0c1"])
    with pytest.raises(SlotIdError, match="Invalid slot ID format"):
        validate_slot_ids(["unknown"])
    with pytest.raises(SlotIdError, match="no longer supported"):
        validate_slot_ids(["early"])

