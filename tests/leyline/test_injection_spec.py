"""Tests for InjectionSpec dataclass."""

import pytest
from esper.leyline.injection_spec import InjectionSpec


class TestInjectionSpec:
    def test_basic_creation(self):
        spec = InjectionSpec(
            slot_id="r0c0",
            channels=64,
            position=0.33,
            layer_range=(0, 2),
        )
        assert spec.slot_id == "r0c0"
        assert spec.channels == 64
        assert spec.position == 0.33
        assert spec.layer_range == (0, 2)

    def test_position_must_be_0_to_1(self):
        with pytest.raises(ValueError, match="position must be between 0 and 1"):
            InjectionSpec(slot_id="r0c0", channels=64, position=1.5, layer_range=(0, 2))

    def test_layer_range_must_be_valid(self):
        with pytest.raises(ValueError, match="layer_range"):
            InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(5, 2))

    def test_frozen_immutable(self):
        spec = InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2))
        with pytest.raises(AttributeError):
            spec.channels = 128
