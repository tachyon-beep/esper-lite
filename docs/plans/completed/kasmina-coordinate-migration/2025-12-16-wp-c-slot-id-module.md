# Work Package C: Slot ID Module (Leaf-First)

**Status:** Ready for implementation
**Priority:** High (foundation for M1)
**Effort:** ~2-3 hours
**Dependencies:** None (can start immediately)

---

## Goal

Implement and exhaustively test `slot_id.py` as a standalone leaf module before touching any consumers.

## Why This De-risks M1

- The coordinate system is the foundation — bugs here cascade to 32+ files
- Property-based testing finds edge cases humans miss
- Once solid, M1.2-M1.5 become mechanical consumers of a proven API
- Clean separation of concerns: parsing/formatting vs integration

## Design Decisions

From the main work package:
- **Format:** `"r{row}c{col}"` (e.g., `"r0c0"`, `"r1c3"`)
- **No legacy aliases:** `"early"` raises `ValueError`, not a redirect
- **Sorting:** Row-major order (r0c0 < r0c1 < r0c2 < r1c0 < r1c1 < ...)
- **Validation:** Non-negative integers only

---

## Tasks

### C.1 Implement slot_id module

**File:** `src/esper/leyline/slot_id.py`

```python
"""Canonical slot ID formatting and parsing.

Slot IDs use a 2D coordinate system: "r{row}c{col}"
Examples: "r0c0", "r0c1", "r1c3"

This module is the single source of truth for slot ID handling.
All other modules should use these functions rather than parsing
slot IDs directly.
"""

from __future__ import annotations

import re

# Regex for canonical slot ID format
_SLOT_ID_PATTERN = re.compile(r"^r(\d+)c(\d+)$")

# Legacy names that are explicitly rejected (clean break)
_LEGACY_NAMES = frozenset({"early", "mid", "late"})


class SlotIdError(ValueError):
    """Raised for invalid slot ID format or values."""
    pass


def format_slot_id(row: int, col: int) -> str:
    """Format a (row, col) coordinate as a canonical slot ID.

    Args:
        row: Row index (0-indexed, non-negative)
        col: Column index (0-indexed, non-negative)

    Returns:
        Canonical slot ID string, e.g., "r0c0"

    Raises:
        SlotIdError: If row or col is negative

    Examples:
        >>> format_slot_id(0, 0)
        'r0c0'
        >>> format_slot_id(1, 3)
        'r1c3'
    """
    if row < 0:
        raise SlotIdError(f"Row must be non-negative, got {row}")
    if col < 0:
        raise SlotIdError(f"Column must be non-negative, got {col}")
    return f"r{row}c{col}"


def parse_slot_id(slot_id: str) -> tuple[int, int]:
    """Parse a canonical slot ID into (row, col) coordinates.

    Args:
        slot_id: Canonical slot ID string, e.g., "r0c0"

    Returns:
        Tuple of (row, col) as integers

    Raises:
        SlotIdError: If format is invalid or uses legacy names

    Examples:
        >>> parse_slot_id("r0c0")
        (0, 0)
        >>> parse_slot_id("r1c3")
        (1, 3)
    """
    # Check for legacy names first (clean break - no aliases)
    if slot_id.lower() in _LEGACY_NAMES:
        raise SlotIdError(
            f"Legacy slot name '{slot_id}' is no longer supported. "
            f"Use canonical format: 'r0c0' (early), 'r0c1' (mid), 'r0c2' (late)"
        )

    match = _SLOT_ID_PATTERN.match(slot_id)
    if match is None:
        raise SlotIdError(
            f"Invalid slot ID format: '{slot_id}'. "
            f"Expected format: 'rXcY' where X and Y are non-negative integers. "
            f"Examples: 'r0c0', 'r1c3'"
        )

    return int(match.group(1)), int(match.group(2))


def validate_slot_id(slot_id: str) -> bool:
    """Check if a string is a valid canonical slot ID.

    Args:
        slot_id: String to validate

    Returns:
        True if valid canonical format, False otherwise

    Note:
        This does NOT accept legacy names. Use parse_slot_id() for
        detailed error messages.

    Examples:
        >>> validate_slot_id("r0c0")
        True
        >>> validate_slot_id("early")
        False
        >>> validate_slot_id("invalid")
        False
    """
    if slot_id.lower() in _LEGACY_NAMES:
        return False
    return _SLOT_ID_PATTERN.match(slot_id) is not None


def slot_sort_key(slot_id: str) -> tuple[int, int]:
    """Return sort key for slot IDs (row-major ordering).

    Ordering: r0c0 < r0c1 < r0c2 < r1c0 < r1c1 < ...

    Args:
        slot_id: Canonical slot ID string

    Returns:
        Tuple (row, col) suitable for sorting

    Raises:
        SlotIdError: If slot_id is invalid

    Examples:
        >>> sorted(["r1c0", "r0c2", "r0c0"], key=slot_sort_key)
        ['r0c0', 'r0c2', 'r1c0']
    """
    return parse_slot_id(slot_id)


def validate_slot_ids(slot_ids: list[str]) -> list[str]:
    """Validate and sort a list of slot IDs.

    Args:
        slot_ids: List of slot ID strings

    Returns:
        Sorted list of validated slot IDs (row-major order)

    Raises:
        SlotIdError: If any slot ID is invalid
    """
    # Validate all (will raise on first invalid)
    for slot_id in slot_ids:
        parse_slot_id(slot_id)

    # Return sorted
    return sorted(slot_ids, key=slot_sort_key)


__all__ = [
    "SlotIdError",
    "format_slot_id",
    "parse_slot_id",
    "validate_slot_id",
    "slot_sort_key",
    "validate_slot_ids",
]
```

### C.2 Add to leyline exports

**File:** `src/esper/leyline/__init__.py`

Add to exports:
```python
from esper.leyline.slot_id import (
    SlotIdError,
    format_slot_id,
    parse_slot_id,
    validate_slot_id,
    slot_sort_key,
    validate_slot_ids,
)
```

### C.3 Implement comprehensive tests

**File:** `tests/leyline/test_slot_id.py`

```python
"""Tests for slot ID formatting and parsing."""

import pytest
from hypothesis import given, assume, settings
from hypothesis import strategies as st

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

    @given(st.integers(min_value=0, max_value=1000),
           st.integers(min_value=0, max_value=1000))
    def test_format_parse_roundtrip(self, row: int, col: int):
        """Property: format then parse returns original coordinates."""
        slot_id = format_slot_id(row, col)
        parsed_row, parsed_col = parse_slot_id(slot_id)
        assert parsed_row == row
        assert parsed_col == col


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

    @given(st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10),
            st.integers(min_value=0, max_value=10)
        ),
        min_size=1,
        max_size=20
    ))
    def test_sort_key_matches_tuple_sort(self, coords: list[tuple[int, int]]):
        """Property: sorting by slot_sort_key matches sorting coordinates as tuples."""
        slot_ids = [format_slot_id(r, c) for r, c in coords]

        sorted_by_key = sorted(slot_ids, key=slot_sort_key)
        sorted_by_coord = [format_slot_id(r, c) for r, c in sorted(coords)]

        assert sorted_by_key == sorted_by_coord


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
```

### C.4 Verify test coverage

- [ ] Run `pytest tests/leyline/test_slot_id.py -v --cov=esper.leyline.slot_id`
- [ ] Ensure 100% branch coverage
- [ ] Run Hypothesis tests with increased examples: `--hypothesis-seed=0 -x`

---

## Acceptance Criteria

- [ ] `src/esper/leyline/slot_id.py` implemented
- [ ] Exported from `esper.leyline`
- [ ] All tests pass
- [ ] 100% branch coverage
- [ ] Property tests pass with 1000+ examples
- [ ] Error messages are helpful (include examples, suggest corrections)

## Outputs

1. `src/esper/leyline/slot_id.py` — canonical implementation
2. `tests/leyline/test_slot_id.py` — comprehensive test suite

## Integration Notes

After this work package, M1.2+ can import:
```python
from esper.leyline import (
    format_slot_id,
    parse_slot_id,
    validate_slot_id,
    slot_sort_key,
    validate_slot_ids,
    SlotIdError,
)
```

No other code needs to parse slot IDs directly — all parsing goes through this module.
