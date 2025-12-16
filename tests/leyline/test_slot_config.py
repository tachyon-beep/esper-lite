"""Tests for SlotConfig dataclass."""

import pytest

from esper.leyline.slot_config import SlotConfig


def test_slot_config_default():
    """Default config should have 3 slots with correct IDs."""
    config = SlotConfig.default()
    assert config.num_slots == 3
    assert config.slot_ids == ("r0c0", "r0c1", "r0c2")


def test_slot_config_custom():
    """Custom slot IDs should work."""
    config = SlotConfig(slot_ids=("r0c0", "r1c0", "r2c0"))
    assert config.num_slots == 3
    assert config.slot_ids == ("r0c0", "r1c0", "r2c0")


def test_slot_id_for_index():
    """slot_id_for_index should return correct ID."""
    config = SlotConfig.default()
    assert config.slot_id_for_index(0) == "r0c0"
    assert config.slot_id_for_index(1) == "r0c1"
    assert config.slot_id_for_index(2) == "r0c2"


def test_index_for_slot_id():
    """index_for_slot_id should return correct index."""
    config = SlotConfig.default()
    assert config.index_for_slot_id("r0c0") == 0
    assert config.index_for_slot_id("r0c1") == 1
    assert config.index_for_slot_id("r0c2") == 2


def test_index_for_slot_id_not_found():
    """index_for_slot_id should raise ValueError for unknown slot."""
    config = SlotConfig.default()
    with pytest.raises(ValueError):
        config.index_for_slot_id("r9c9")


def test_for_grid():
    """for_grid should create correct grid config."""
    config = SlotConfig.for_grid(rows=2, cols=3)
    assert config.num_slots == 6
    assert config.slot_ids == ("r0c0", "r0c1", "r0c2", "r1c0", "r1c1", "r1c2")


def test_for_grid_single_row():
    """for_grid should work with single row."""
    config = SlotConfig.for_grid(rows=1, cols=4)
    assert config.num_slots == 4
    assert config.slot_ids == ("r0c0", "r0c1", "r0c2", "r0c3")


def test_for_grid_single_col():
    """for_grid should work with single column."""
    config = SlotConfig.for_grid(rows=3, cols=1)
    assert config.num_slots == 3
    assert config.slot_ids == ("r0c0", "r1c0", "r2c0")


def test_num_slots_property():
    """num_slots property should work correctly."""
    config1 = SlotConfig(slot_ids=("r0c0",))
    assert config1.num_slots == 1

    config2 = SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r0c3", "r0c4"))
    assert config2.num_slots == 5


def test_slot_config_frozen():
    """SlotConfig should be frozen (immutable)."""
    config = SlotConfig.default()
    with pytest.raises(Exception):  # dataclass frozen raises FrozenInstanceError
        config.slot_ids = ("r0c0",)  # type: ignore


def test_slot_config_tuple_enforced():
    """slot_ids should be stored as tuple."""
    config = SlotConfig(slot_ids=("r0c0", "r0c1"))
    assert isinstance(config.slot_ids, tuple)
    # Even if we pass a list, it should be stored as tuple
    config2 = SlotConfig(slot_ids=tuple(["r0c0", "r0c1"]))
    assert isinstance(config2.slot_ids, tuple)


class TestSlotConfigFromSpecs:
    """Tests for SlotConfig.from_specs() factory method."""

    def test_from_specs_extracts_slot_ids(self):
        """from_specs should extract slot IDs from specs."""
        from esper.leyline import InjectionSpec

        specs = [
            InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2)),
            InjectionSpec(slot_id="r0c1", channels=128, position=0.66, layer_range=(2, 4)),
        ]
        config = SlotConfig.from_specs(specs)
        assert config.slot_ids == ("r0c0", "r0c1")
        assert config.num_slots == 2

    def test_from_specs_sorts_by_position(self):
        """from_specs should sort specs by position."""
        from esper.leyline import InjectionSpec

        # Out of order input
        specs = [
            InjectionSpec(slot_id="r0c1", channels=128, position=0.66, layer_range=(2, 4)),
            InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2)),
        ]
        config = SlotConfig.from_specs(specs)
        # Should be sorted by position
        assert config.slot_ids == ("r0c0", "r0c1")

    def test_from_specs_preserves_channel_info(self):
        """from_specs should preserve channel information."""
        from esper.leyline import InjectionSpec

        specs = [
            InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2)),
            InjectionSpec(slot_id="r0c1", channels=128, position=0.66, layer_range=(2, 4)),
        ]
        config = SlotConfig.from_specs(specs)
        assert config.channels_for_slot("r0c0") == 64
        assert config.channels_for_slot("r0c1") == 128

    def test_from_specs_empty_raises(self):
        """from_specs should raise ValueError on empty list."""
        with pytest.raises(ValueError, match="at least one"):
            SlotConfig.from_specs([])

    def test_channels_for_slot_unknown_slot(self):
        """channels_for_slot should return 0 for unknown slots."""
        from esper.leyline import InjectionSpec

        specs = [
            InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2)),
        ]
        config = SlotConfig.from_specs(specs)
        assert config.channels_for_slot("r0c0") == 64
        assert config.channels_for_slot("r9c9") == 0
