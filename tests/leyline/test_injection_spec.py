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


class TestInjectionSpecEdgeCases:
    """Edge case tests for InjectionSpec validation."""

    # --- Channel Validation ---

    def test_zero_channels_rejected(self):
        """Zero channels should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            InjectionSpec(slot_id="r0c0", channels=0, position=0.5, layer_range=(0, 1))

    def test_negative_channels_rejected(self):
        """Negative channels should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            InjectionSpec(slot_id="r0c0", channels=-1, position=0.5, layer_range=(0, 1))

        with pytest.raises(ValueError, match="positive"):
            InjectionSpec(slot_id="r0c0", channels=-100, position=0.5, layer_range=(0, 1))

    def test_positive_channels_accepted(self):
        """Positive channels should be accepted."""
        spec = InjectionSpec(slot_id="r0c0", channels=1, position=0.5, layer_range=(0, 1))
        assert spec.channels == 1

        spec_large = InjectionSpec(slot_id="r0c0", channels=4096, position=0.5, layer_range=(0, 1))
        assert spec_large.channels == 4096

    # --- Position Boundary Cases ---

    def test_position_exactly_zero(self):
        """Position at 0.0 should be valid (network input)."""
        spec = InjectionSpec(slot_id="r0c0", channels=64, position=0.0, layer_range=(0, 1))
        assert spec.position == 0.0

    def test_position_exactly_one(self):
        """Position at 1.0 should be valid (network output)."""
        spec = InjectionSpec(slot_id="r0c0", channels=64, position=1.0, layer_range=(0, 1))
        assert spec.position == 1.0

    def test_position_just_below_zero_rejected(self):
        """Position slightly below 0.0 should be rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            InjectionSpec(slot_id="r0c0", channels=64, position=-0.001, layer_range=(0, 1))

    def test_position_just_above_one_rejected(self):
        """Position slightly above 1.0 should be rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            InjectionSpec(slot_id="r0c0", channels=64, position=1.001, layer_range=(0, 1))

    def test_position_very_negative_rejected(self):
        """Very negative position should be rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            InjectionSpec(slot_id="r0c0", channels=64, position=-10.0, layer_range=(0, 1))

    def test_position_very_positive_rejected(self):
        """Position much greater than 1 should be rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            InjectionSpec(slot_id="r0c0", channels=64, position=10.0, layer_range=(0, 1))

    # --- Layer Range Edge Cases ---

    def test_single_layer_range(self):
        """layer_range=(N, N) should be valid for single-layer injection."""
        spec = InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(5, 5))
        assert spec.layer_range == (5, 5)

    def test_layer_range_at_zero(self):
        """layer_range starting at 0 should be valid."""
        spec = InjectionSpec(slot_id="r0c0", channels=64, position=0.0, layer_range=(0, 0))
        assert spec.layer_range == (0, 0)

    def test_layer_range_large_span(self):
        """Large layer range should be valid."""
        spec = InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(0, 100))
        assert spec.layer_range == (0, 100)

    def test_layer_range_inverted_rejected(self):
        """layer_range with start > end should be rejected."""
        with pytest.raises(ValueError, match="layer_range"):
            InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(10, 5))

    def test_layer_range_negative_start_rejected(self):
        """Negative layer_range start should be rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(-1, 5))

    def test_layer_range_both_negative_rejected(self):
        """Both negative layer indices should be rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(-5, -1))

    # --- Equality and Hashing ---

    def test_injection_spec_equality(self):
        """InjectionSpecs with same values should be equal."""
        spec1 = InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2))
        spec2 = InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2))
        assert spec1 == spec2

    def test_injection_spec_inequality(self):
        """InjectionSpecs with different values should not be equal."""
        spec1 = InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2))
        spec2 = InjectionSpec(slot_id="r0c1", channels=64, position=0.33, layer_range=(0, 2))
        assert spec1 != spec2

        spec3 = InjectionSpec(slot_id="r0c0", channels=128, position=0.33, layer_range=(0, 2))
        assert spec1 != spec3

    def test_injection_spec_hashable(self):
        """InjectionSpec should be hashable (frozen dataclass)."""
        spec = InjectionSpec(slot_id="r0c0", channels=64, position=0.33, layer_range=(0, 2))
        # Should not raise
        hash(spec)
        # Should be usable in sets/dicts
        spec_set = {spec}
        assert spec in spec_set

    # --- Slot ID Formats ---

    def test_various_slot_id_formats(self):
        """Various slot ID formats should be accepted (no validation on format)."""
        # Standard canonical format
        spec1 = InjectionSpec(slot_id="r0c0", channels=64, position=0.5, layer_range=(0, 1))
        assert spec1.slot_id == "r0c0"

        # Larger coordinates
        spec2 = InjectionSpec(slot_id="r10c20", channels=64, position=0.5, layer_range=(0, 1))
        assert spec2.slot_id == "r10c20"

        # InjectionSpec doesn't validate slot_id format (that's SlotConfig's job)
        spec3 = InjectionSpec(slot_id="custom_slot", channels=64, position=0.5, layer_range=(0, 1))
        assert spec3.slot_id == "custom_slot"
