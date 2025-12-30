"""Tests for sigmoid steepness parameterization in AlphaController."""

import math

from esper.kasmina.alpha_controller import AlphaController, _curve_progress
from esper.leyline.alpha import AlphaCurve


class TestCurveProgressSteepness:
    """Test _curve_progress with different steepness values."""

    def test_sigmoid_default_steepness_is_12(self):
        """Default steepness=12 should produce the original behavior."""
        result = _curve_progress(0.5, AlphaCurve.SIGMOID, steepness=12.0)
        assert abs(result - 0.5) < 1e-6

    def test_sigmoid_gentle_is_less_steep(self):
        """Gentle steepness=6 should have less curvature at t=0.25."""
        gentle = _curve_progress(0.25, AlphaCurve.SIGMOID, steepness=6.0)
        standard = _curve_progress(0.25, AlphaCurve.SIGMOID, steepness=12.0)
        # Gentle curve should be closer to linear (0.25) at this point
        assert gentle > standard

    def test_sigmoid_sharp_is_more_steep(self):
        """Sharp steepness=24 should have more curvature at t=0.25."""
        sharp = _curve_progress(0.25, AlphaCurve.SIGMOID, steepness=24.0)
        standard = _curve_progress(0.25, AlphaCurve.SIGMOID, steepness=12.0)
        # Sharp curve should be closer to 0 at t=0.25
        assert sharp < standard

    def test_linear_ignores_steepness(self):
        """LINEAR curve should ignore steepness parameter."""
        result = _curve_progress(0.5, AlphaCurve.LINEAR, steepness=100.0)
        assert result == 0.5

    def test_cosine_ignores_steepness(self):
        """COSINE curve should ignore steepness parameter."""
        result = _curve_progress(0.5, AlphaCurve.COSINE, steepness=100.0)
        expected = 0.5 * (1.0 - math.cos(math.pi * 0.5))
        assert abs(result - expected) < 1e-6


class TestAlphaControllerSteepness:
    """Test AlphaController with steepness field."""

    def test_default_steepness_is_12(self):
        """AlphaController should default to steepness=12."""
        controller = AlphaController()
        assert controller.alpha_steepness == 12.0

    def test_retarget_accepts_steepness(self):
        """retarget() should accept and store steepness."""
        controller = AlphaController()
        controller.retarget(
            alpha_target=1.0,
            alpha_steps_total=5,
            alpha_curve=AlphaCurve.SIGMOID,
            alpha_steepness=6.0,
        )
        assert controller.alpha_steepness == 6.0

    def test_steepness_affects_transition(self):
        """Different steepness should produce different alpha progression."""
        # Gentle curve - test at t=0.25 (step 1 of 4)
        gentle = AlphaController()
        gentle.retarget(alpha_target=1.0, alpha_steps_total=4, alpha_curve=AlphaCurve.SIGMOID, alpha_steepness=6.0)
        gentle.step()  # t=0.25
        alpha_gentle = gentle.alpha

        # Sharp curve - test at t=0.25 (step 1 of 4)
        sharp = AlphaController()
        sharp.retarget(alpha_target=1.0, alpha_steps_total=4, alpha_curve=AlphaCurve.SIGMOID, alpha_steepness=24.0)
        sharp.step()  # t=0.25
        alpha_sharp = sharp.alpha

        # At t=0.25, gentle curve is closer to linear (0.25) while sharp is closer to 0
        assert alpha_gentle > alpha_sharp


class TestAlphaControllerSteepnessSerialization:
    """Test checkpoint round-trip with steepness."""

    def test_to_dict_includes_steepness(self):
        """to_dict() should include alpha_steepness."""
        controller = AlphaController(alpha_steepness=6.0)
        data = controller.to_dict()
        assert "alpha_steepness" in data
        assert data["alpha_steepness"] == 6.0

    def test_from_dict_restores_steepness(self):
        """from_dict() should restore alpha_steepness."""
        original = AlphaController(alpha_steepness=24.0)
        data = original.to_dict()
        restored = AlphaController.from_dict(data)
        assert restored.alpha_steepness == 24.0

    def test_from_dict_defaults_steepness_for_old_checkpoints(self):
        """from_dict() should default steepness=12 for old checkpoints."""
        old_data = {
            "alpha": 0.5,
            "alpha_start": 0.0,
            "alpha_target": 1.0,
            "alpha_mode": 1,
            "alpha_curve": 3,
            "alpha_steps_total": 5,
            "alpha_steps_done": 2,
            # No alpha_steepness - old checkpoint
        }
        restored = AlphaController.from_dict(old_data)
        assert restored.alpha_steepness == 12.0  # Default
