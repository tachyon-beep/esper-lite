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


__all__ = [
    "SlotIdError",
    "format_slot_id",
    "parse_slot_id",
    "validate_slot_id",
    "slot_sort_key",
    "validate_slot_ids",
]
