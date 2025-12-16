"""Tests for slot ordering utilities."""

from esper.simic.slots import CANONICAL_SLOTS, ordered_slots


def test_ordered_slots_filters_to_canonical_order() -> None:
    assert ordered_slots(["r0c2", "r0c0"]) == ("r0c0", "r0c2")
    assert ordered_slots(["r0c1"]) == ("r0c1",)
    assert ordered_slots(["r0c0", "r0c1", "r0c2"]) == CANONICAL_SLOTS


def test_ordered_slots_ignores_unknown_slots() -> None:
    assert ordered_slots(["unknown", "r0c1"]) == ("r0c1",)
    assert ordered_slots(["unknown"]) == ()

