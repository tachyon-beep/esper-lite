"""Tests for alpha_curve field in Sanctum schema."""

from esper.karn.sanctum.schema import SeedState


class TestSeedStateAlphaCurve:
    """Test alpha_curve field in SeedState."""

    def test_default_alpha_curve_is_linear(self):
        """SeedState should default to LINEAR curve."""
        seed = SeedState(slot_id="slot_0")
        assert seed.alpha_curve == "LINEAR"

    def test_alpha_curve_can_be_set(self):
        """SeedState should accept alpha_curve parameter."""
        seed = SeedState(slot_id="slot_0", alpha_curve="SIGMOID")
        assert seed.alpha_curve == "SIGMOID"
