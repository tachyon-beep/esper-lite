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


class TestSlotConfigEdgeCases:
    """Edge case tests for SlotConfig."""

    # --- Large Grid Configurations ---

    def test_large_grid_3x3(self):
        """9-slot (3x3) grid should work correctly."""
        config = SlotConfig.for_grid(rows=3, cols=3)
        assert config.num_slots == 9
        expected = (
            "r0c0", "r0c1", "r0c2",
            "r1c0", "r1c1", "r1c2",
            "r2c0", "r2c1", "r2c2",
        )
        assert config.slot_ids == expected

    def test_large_grid_5x5(self):
        """25-slot (5x5) grid should work correctly."""
        config = SlotConfig.for_grid(rows=5, cols=5)
        assert config.num_slots == 25
        # Verify row-major ordering
        for i, slot_id in enumerate(config.slot_ids):
            row = i // 5
            col = i % 5
            assert slot_id == f"r{row}c{col}"

    def test_asymmetric_grid_2x5(self):
        """Asymmetric 2x5 grid should work correctly."""
        config = SlotConfig.for_grid(rows=2, cols=5)
        assert config.num_slots == 10
        assert config.slot_ids[0] == "r0c0"
        assert config.slot_ids[4] == "r0c4"  # End of first row
        assert config.slot_ids[5] == "r1c0"  # Start of second row
        assert config.slot_ids[9] == "r1c4"

    def test_asymmetric_grid_5x2(self):
        """Asymmetric 5x2 grid should work correctly."""
        config = SlotConfig.for_grid(rows=5, cols=2)
        assert config.num_slots == 10
        assert config.slot_ids[0] == "r0c0"
        assert config.slot_ids[1] == "r0c1"  # End of first row (only 2 cols)
        assert config.slot_ids[2] == "r1c0"  # Start of second row
        assert config.slot_ids[9] == "r4c1"

    # --- Single Slot Configurations ---

    def test_single_slot_config(self):
        """Single slot configuration should work correctly."""
        config = SlotConfig(slot_ids=("r0c0",))
        assert config.num_slots == 1
        assert config.slot_id_for_index(0) == "r0c0"
        assert config.index_for_slot_id("r0c0") == 0

    def test_single_slot_grid_1x1(self):
        """1x1 grid should produce single slot."""
        config = SlotConfig.for_grid(rows=1, cols=1)
        assert config.num_slots == 1
        assert config.slot_ids == ("r0c0",)

    # --- Validation Edge Cases ---

    def test_empty_slot_ids_rejected(self):
        """Empty slot_ids tuple should raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            SlotConfig(slot_ids=())

    def test_duplicate_slot_ids_rejected(self):
        """Duplicate slot IDs should raise ValueError."""
        with pytest.raises(ValueError, match="unique"):
            SlotConfig(slot_ids=("r0c0", "r0c0"))

    def test_duplicate_slot_ids_multiple(self):
        """Multiple duplicates should be detected."""
        with pytest.raises(ValueError, match="unique"):
            SlotConfig(slot_ids=("r0c0", "r0c1", "r0c0", "r0c1"))

    def test_duplicate_slot_ids_with_unique(self):
        """Duplicates mixed with unique IDs should be rejected."""
        with pytest.raises(ValueError, match="unique"):
            SlotConfig(slot_ids=("r0c0", "r0c1", "r0c2", "r0c0"))

    # --- Index Boundary Cases ---

    def test_index_out_of_bounds_positive(self):
        """Index >= num_slots should raise IndexError."""
        config = SlotConfig.default()  # 3 slots
        with pytest.raises(IndexError):
            config.slot_id_for_index(3)
        with pytest.raises(IndexError):
            config.slot_id_for_index(100)

    def test_index_out_of_bounds_negative(self):
        """Negative indices are invalid (action indices are 0-based)."""
        config = SlotConfig.default()  # 3 slots: r0c0, r0c1, r0c2
        with pytest.raises(IndexError):
            config.slot_id_for_index(-1)
        with pytest.raises(IndexError):
            config.slot_id_for_index(-2)
        with pytest.raises(IndexError):
            config.slot_id_for_index(-3)

    # --- Equality and Hashing ---

    def test_slot_config_equality(self):
        """SlotConfigs with same slot_ids should be equal."""
        config1 = SlotConfig(slot_ids=("r0c0", "r0c1"))
        config2 = SlotConfig(slot_ids=("r0c0", "r0c1"))
        assert config1 == config2

    def test_slot_config_inequality(self):
        """SlotConfigs with different slot_ids should not be equal."""
        config1 = SlotConfig(slot_ids=("r0c0", "r0c1"))
        config2 = SlotConfig(slot_ids=("r0c0", "r0c2"))
        assert config1 != config2

    def test_slot_config_hashable(self):
        """SlotConfig should be hashable (frozen dataclass)."""
        config = SlotConfig.default()
        # Should not raise
        hash(config)
        # Should be usable in sets/dicts
        config_set = {config}
        assert config in config_set

    # --- Grid with Zero Dimensions ---

    def test_for_grid_zero_rows(self):
        """Zero rows grid should produce empty config (which is invalid)."""
        with pytest.raises(ValueError, match="at least one"):
            SlotConfig.for_grid(rows=0, cols=3)

    def test_for_grid_zero_cols(self):
        """Zero cols grid should produce empty config (which is invalid)."""
        with pytest.raises(ValueError, match="at least one"):
            SlotConfig.for_grid(rows=3, cols=0)

    def test_for_grid_both_zero(self):
        """Both zero dimensions should produce empty config (which is invalid)."""
        with pytest.raises(ValueError, match="at least one"):
            SlotConfig.for_grid(rows=0, cols=0)
