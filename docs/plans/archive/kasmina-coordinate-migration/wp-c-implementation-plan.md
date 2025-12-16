# WP-C: Slot ID Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and exhaustively test `slot_id.py` as a standalone leaf module — the foundation for the coordinate system migration.

**Architecture:** Create a pure-function module in `leyline` that handles all slot ID formatting, parsing, validation, and sorting. All other modules will import from here rather than parsing slot IDs directly. Property-based testing ensures edge case coverage.

**Tech Stack:** Python 3.13, pytest, Hypothesis (property-based testing), regex

---

## Task 1: Create slot_id module with SlotIdError

**Files:**
- Create: `src/esper/leyline/slot_id.py`

**Step 1: Create the module with exception and constants**

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
```

**Step 2: Verify syntax**

Run: `python -m py_compile src/esper/leyline/slot_id.py && echo "Syntax OK"`
Expected: `Syntax OK`

---

## Task 2: Implement format_slot_id with TDD

**Files:**
- Modify: `src/esper/leyline/slot_id.py`
- Create: `tests/leyline/test_slot_id.py`

**Step 1: Write the failing test**

```python
"""Tests for slot ID formatting and parsing."""

import pytest

from esper.leyline.slot_id import (
    SlotIdError,
    format_slot_id,
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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py::TestFormatSlotId -v`
Expected: FAIL with `ImportError: cannot import name 'format_slot_id'`

**Step 3: Implement format_slot_id**

Add to `src/esper/leyline/slot_id.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py::TestFormatSlotId -v`
Expected: 3 passed

---

## Task 3: Implement parse_slot_id with TDD

**Files:**
- Modify: `src/esper/leyline/slot_id.py`
- Modify: `tests/leyline/test_slot_id.py`

**Step 1: Write the failing tests**

Add to `tests/leyline/test_slot_id.py`:

```python
from esper.leyline.slot_id import (
    SlotIdError,
    format_slot_id,
    parse_slot_id,
)


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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py::TestParseSlotId -v`
Expected: FAIL with `ImportError: cannot import name 'parse_slot_id'`

**Step 3: Implement parse_slot_id**

Add to `src/esper/leyline/slot_id.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py::TestParseSlotId -v`
Expected: 4 passed

---

## Task 4: Implement validate_slot_id with TDD

**Files:**
- Modify: `src/esper/leyline/slot_id.py`
- Modify: `tests/leyline/test_slot_id.py`

**Step 1: Write the failing tests**

Add to test file imports and add class:

```python
from esper.leyline.slot_id import (
    SlotIdError,
    format_slot_id,
    parse_slot_id,
    validate_slot_id,
)


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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py::TestValidateSlotId -v`
Expected: FAIL with `ImportError: cannot import name 'validate_slot_id'`

**Step 3: Implement validate_slot_id**

Add to `src/esper/leyline/slot_id.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py::TestValidateSlotId -v`
Expected: 2 passed

---

## Task 5: Implement slot_sort_key and validate_slot_ids with TDD

**Files:**
- Modify: `src/esper/leyline/slot_id.py`
- Modify: `tests/leyline/test_slot_id.py`

**Step 1: Write the failing tests**

Add to test file imports and add classes:

```python
from esper.leyline.slot_id import (
    SlotIdError,
    format_slot_id,
    parse_slot_id,
    validate_slot_id,
    slot_sort_key,
    validate_slot_ids,
)


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
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py::TestSlotSortKey tests/leyline/test_slot_id.py::TestValidateSlotIds -v`
Expected: FAIL with `ImportError: cannot import name 'slot_sort_key'`

**Step 3: Implement slot_sort_key and validate_slot_ids**

Add to `src/esper/leyline/slot_id.py`:

```python
def slot_sort_key(slot_id: str) -> tuple[int, int]:
    """Return sort key for slot IDs (row-major ordering).

    Ordering: r0c0 < r0c1 < r0c2 < r1c0 < r1c1 < ...

    STABILITY NOTE (from DRL Specialist):
    This ordering determines action indices in the RL policy network.
    Once a model is trained, changing this function would invalidate
    the policy (action indices would map to different slots).
    This function must remain stable across versions.

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
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py::TestSlotSortKey tests/leyline/test_slot_id.py::TestValidateSlotIds -v`
Expected: 6 passed

---

## Task 6: Add __all__ exports and commit

**Files:**
- Modify: `src/esper/leyline/slot_id.py`

**Step 1: Add __all__ to slot_id.py**

Add at end of `src/esper/leyline/slot_id.py`:

```python
__all__ = [
    "SlotIdError",
    "format_slot_id",
    "parse_slot_id",
    "validate_slot_id",
    "slot_sort_key",
    "validate_slot_ids",
]
```

**Step 2: Verify all tests pass**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/esper/leyline/slot_id.py tests/leyline/test_slot_id.py
git commit -m "feat(leyline): add slot_id module for coordinate system

Implements canonical slot ID format 'rXcY' with:
- format_slot_id(row, col) -> str
- parse_slot_id(slot_id) -> (row, col)
- validate_slot_id(slot_id) -> bool
- slot_sort_key for row-major ordering
- validate_slot_ids for batch validation

Legacy names (early/mid/late) are rejected with helpful error messages."
```

---

## Task 7: Add property-based tests with Hypothesis

**Files:**
- Modify: `tests/leyline/test_slot_id.py`

**Step 1: Add Hypothesis imports and roundtrip test**

Add to `tests/leyline/test_slot_id.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st


class TestFormatSlotId:
    # ... existing tests ...

    @given(
        st.integers(min_value=0, max_value=1000),
        st.integers(min_value=0, max_value=1000),
    )
    def test_format_parse_roundtrip(self, row: int, col: int):
        """Property: format then parse returns original coordinates."""
        slot_id = format_slot_id(row, col)
        parsed_row, parsed_col = parse_slot_id(slot_id)
        assert parsed_row == row
        assert parsed_col == col
```

**Step 2: Add sort key property test**

Add to `TestSlotSortKey` class:

```python
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=10),
                st.integers(min_value=0, max_value=10),
            ),
            min_size=1,
            max_size=20,
        )
    )
    def test_sort_key_matches_tuple_sort(self, coords: list[tuple[int, int]]):
        """Property: sorting by slot_sort_key matches sorting coordinates as tuples."""
        slot_ids = [format_slot_id(r, c) for r, c in coords]

        sorted_by_key = sorted(slot_ids, key=slot_sort_key)
        sorted_by_coord = [format_slot_id(r, c) for r, c in sorted(coords)]

        assert sorted_by_key == sorted_by_coord
```

**Step 3: Run property tests**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py -v --hypothesis-seed=0`
Expected: All tests pass (Hypothesis runs 100+ examples per property)

**Step 4: Run with more examples**

Run: `HYPOTHESIS_PROFILE=ci PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py -v`
Expected: All tests pass

**Step 5: Commit property tests**

```bash
git add tests/leyline/test_slot_id.py
git commit -m "test(leyline): add property-based tests for slot_id

Uses Hypothesis to verify:
- format/parse roundtrip preserves coordinates
- slot_sort_key produces same ordering as tuple sort"
```

---

## Task 8: Export from leyline __init__.py

**Files:**
- Modify: `src/esper/leyline/__init__.py`

**Step 1: Add import**

Add after other imports in `src/esper/leyline/__init__.py`:

```python
# Slot ID formatting and parsing
from esper.leyline.slot_id import (
    SlotIdError,
    format_slot_id,
    parse_slot_id,
    validate_slot_id,
    slot_sort_key,
    validate_slot_ids,
)
```

**Step 2: Add to __all__**

Add to `__all__` list:

```python
    # Slot ID
    "SlotIdError",
    "format_slot_id",
    "parse_slot_id",
    "validate_slot_id",
    "slot_sort_key",
    "validate_slot_ids",
```

**Step 3: Verify imports work**

Run: `PYTHONPATH=src uv run python -c "from esper.leyline import format_slot_id, parse_slot_id; print(format_slot_id(0, 0))"`
Expected: `r0c0`

**Step 4: Commit**

```bash
git add src/esper/leyline/__init__.py
git commit -m "feat(leyline): export slot_id functions from package

Adds to esper.leyline public API:
- SlotIdError, format_slot_id, parse_slot_id
- validate_slot_id, slot_sort_key, validate_slot_ids"
```

---

## Task 9: Final verification

**Step 1: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/leyline/test_slot_id.py -v --cov=esper.leyline.slot_id --cov-report=term-missing`
Expected: 100% coverage (or near it)

**Step 2: Verify no regressions**

Run: `PYTHONPATH=src uv run pytest tests/leyline/ -v`
Expected: All leyline tests pass

**Step 3: Verify clean git status**

Run: `git status`
Expected: Clean working tree

---

## Acceptance Checklist

- [ ] `src/esper/leyline/slot_id.py` implemented
- [ ] Exported from `esper.leyline`
- [ ] All unit tests pass
- [ ] Property tests pass with Hypothesis
- [ ] High branch coverage
- [ ] Error messages are helpful
- [ ] All changes committed

---

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
