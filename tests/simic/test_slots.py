"""Tests for slot ordering utilities."""

from esper.simic.slots import CANONICAL_SLOTS, ordered_slots


def test_ordered_slots_filters_to_canonical_order() -> None:
    assert ordered_slots(["late", "early"]) == ("early", "late")
    assert ordered_slots(["mid"]) == ("mid",)
    assert ordered_slots(["early", "mid", "late"]) == CANONICAL_SLOTS


def test_ordered_slots_ignores_unknown_slots() -> None:
    assert ordered_slots(["unknown", "mid"]) == ("mid",)
    assert ordered_slots(["unknown"]) == ()

