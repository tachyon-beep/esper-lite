"""Slot ordering utilities for multi-slot features and masks."""

from __future__ import annotations

from typing import Iterable

CANONICAL_SLOTS: tuple[str, ...] = ("early", "mid", "late")


def ordered_slots(enabled_slots: Iterable[str]) -> tuple[str, ...]:
    """Return canonical slot order filtered to enabled slots.

    Deterministic order prevents feature/mask drift due to dict ordering.
    """
    enabled_set = set(enabled_slots)
    return tuple(slot for slot in CANONICAL_SLOTS if slot in enabled_set)


__all__ = ["CANONICAL_SLOTS", "ordered_slots"]
