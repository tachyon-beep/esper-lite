"""Tests for Karn subsystem contracts."""

from __future__ import annotations

import pytest

from esper.karn.contracts import KarnSlotConfig


def test_default_slot_config_matches_canonical_three_slots() -> None:
    config = KarnSlotConfig.default()

    assert config.slot_ids == ("r0c0", "r0c1", "r0c2")
    assert config.num_slots == 3
    assert config.index_for_slot_id("r0c1") == 1


def test_custom_slot_config_indexes_slots() -> None:
    config = KarnSlotConfig(slot_ids=("r1c0", "r1c1"), num_slots=2)

    assert config.index_for_slot_id("r1c0") == 0
    assert config.index_for_slot_id("r1c1") == 1


def test_slot_config_rejects_count_mismatch() -> None:
    with pytest.raises(ValueError, match="num_slots"):
        KarnSlotConfig(slot_ids=("r0c0", "r0c1"), num_slots=3)


def test_slot_config_rejects_unknown_slot() -> None:
    config = KarnSlotConfig.default()

    with pytest.raises(ValueError, match="Unknown slot_id"):
        config.index_for_slot_id("r9c9")
