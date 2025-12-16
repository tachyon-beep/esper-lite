"""Tests for slot ID formatting and parsing."""

import pytest

from esper.leyline.slot_id import (
    SlotIdError,
    format_slot_id,
    parse_slot_id,
    validate_slot_id,
    slot_sort_key,
    validate_slot_ids,
)


class TestFormatSlotId:
    """Tests for format_slot_id()."""

    def test_basic_formatting(self):
        assert format_slot_id(0, 0) == "r0c0"
        assert format_slot_id(0, 1) == "r0c1"
        assert format_slot_id(1, 0) == "r1c0"
        assert format_slot_id(1, 3) == "r1c3"
        assert format_slot_id(10, 20) == "r10c20"

    def test_negative_row_raises(self):
        with pytest.raises(SlotIdError, match="Row must be non-negative"):
            format_slot_id(-1, 0)

    def test_negative_col_raises(self):
        with pytest.raises(SlotIdError, match="Column must be non-negative"):
            format_slot_id(0, -1)


class TestParseSlotId:
    """Tests for parse_slot_id()."""

    def test_basic_parsing(self):
        assert parse_slot_id("r0c0") == (0, 0)
        assert parse_slot_id("r0c1") == (0, 1)
        assert parse_slot_id("r1c0") == (1, 0)
        assert parse_slot_id("r10c20") == (10, 20)

    def test_legacy_names_rejected(self):
        with pytest.raises(SlotIdError, match="no longer supported"):
            parse_slot_id("early")
        with pytest.raises(SlotIdError, match="no longer supported"):
            parse_slot_id("mid")
        with pytest.raises(SlotIdError, match="no longer supported"):
            parse_slot_id("late")
        # Case insensitive rejection
        with pytest.raises(SlotIdError, match="no longer supported"):
            parse_slot_id("EARLY")

    def test_invalid_format_rejected(self):
        with pytest.raises(SlotIdError, match="Invalid slot ID format"):
            parse_slot_id("invalid")
        with pytest.raises(SlotIdError, match="Invalid slot ID format"):
            parse_slot_id("r0")
        with pytest.raises(SlotIdError, match="Invalid slot ID format"):
            parse_slot_id("c0")
        with pytest.raises(SlotIdError, match="Invalid slot ID format"):
            parse_slot_id("0c0")
        with pytest.raises(SlotIdError, match="Invalid slot ID format"):
            parse_slot_id("r-1c0")  # Negative in string
        with pytest.raises(SlotIdError, match="Invalid slot ID format"):
            parse_slot_id("")
        with pytest.raises(SlotIdError, match="Invalid slot ID format"):
            parse_slot_id("r0c0extra")

    def test_helpful_error_messages(self):
        """Error messages should guide users to correct format."""
        try:
            parse_slot_id("early")
        except SlotIdError as e:
            assert "r0c0" in str(e)  # Suggests replacement

        try:
            parse_slot_id("invalid")
        except SlotIdError as e:
            assert "rXcY" in str(e)  # Shows format
            assert "r0c0" in str(e)  # Shows example


class TestValidateSlotId:
    """Tests for validate_slot_id()."""

    def test_valid_ids(self):
        assert validate_slot_id("r0c0") is True
        assert validate_slot_id("r1c3") is True
        assert validate_slot_id("r10c20") is True

    def test_invalid_ids(self):
        assert validate_slot_id("early") is False
        assert validate_slot_id("mid") is False
        assert validate_slot_id("late") is False
        assert validate_slot_id("invalid") is False
        assert validate_slot_id("") is False


class TestSlotSortKey:
    """Tests for slot_sort_key() and ordering."""

    def test_row_major_ordering(self):
        """Slots sort in row-major order."""
        slots = ["r1c0", "r0c2", "r0c0", "r0c1", "r1c1"]
        sorted_slots = sorted(slots, key=slot_sort_key)
        assert sorted_slots == ["r0c0", "r0c1", "r0c2", "r1c0", "r1c1"]

    def test_single_row_ordering(self):
        """Within a row, columns sort ascending."""
        slots = ["r0c2", "r0c0", "r0c1"]
        sorted_slots = sorted(slots, key=slot_sort_key)
        assert sorted_slots == ["r0c0", "r0c1", "r0c2"]


class TestValidateSlotIds:
    """Tests for validate_slot_ids()."""

    def test_validates_and_sorts(self):
        result = validate_slot_ids(["r1c0", "r0c0", "r0c1"])
        assert result == ["r0c0", "r0c1", "r1c0"]

    def test_invalid_raises(self):
        with pytest.raises(SlotIdError):
            validate_slot_ids(["r0c0", "early", "r0c1"])

    def test_empty_list(self):
        assert validate_slot_ids([]) == []

    def test_single_slot(self):
        assert validate_slot_ids(["r0c0"]) == ["r0c0"]
