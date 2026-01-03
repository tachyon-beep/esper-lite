"""Tests for sigmoid steepness parameterization in AlphaController."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from esper.kasmina.alpha_controller import AlphaController, _curve_progress, _sigmoid
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

    def test_from_dict_rejects_old_checkpoints_without_steepness(self):
        """from_dict() should reject checkpoints missing alpha_steepness (no legacy support)."""
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
        # Per no-legacy-code policy: old checkpoints must fail-fast
        with pytest.raises(KeyError, match="alpha_steepness"):
            AlphaController.from_dict(old_data)


class TestSigmoidNumericalStability:
    """Regression tests for SIGMOID overflow bug.

    Bug: _curve_progress(0.0, AlphaCurve.SIGMOID, steepness=2000.0) raised
    OverflowError because math.exp(1000.0) overflows. Fixed by using a
    numerically stable sigmoid that avoids computing exp(large_positive).
    """

    def test_extreme_steepness_no_overflow(self):
        """Large steepness values must not raise OverflowError."""
        # This was the exact reproduction case for the bug
        result = _curve_progress(0.0, AlphaCurve.SIGMOID, steepness=2000.0)
        assert result == pytest.approx(0.0, abs=1e-9)

        result = _curve_progress(1.0, AlphaCurve.SIGMOID, steepness=2000.0)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_extreme_steepness_boundaries(self):
        """Extreme steepness should still map 0→0 and 1→1."""
        for steepness in [1000.0, 2000.0, 5000.0, 10000.0]:
            at_0 = _curve_progress(0.0, AlphaCurve.SIGMOID, steepness=steepness)
            at_1 = _curve_progress(1.0, AlphaCurve.SIGMOID, steepness=steepness)
            assert at_0 == pytest.approx(0.0, abs=1e-9), f"steepness={steepness}"
            assert at_1 == pytest.approx(1.0, abs=1e-9), f"steepness={steepness}"

    def test_extreme_steepness_midpoint(self):
        """Midpoint should be 0.5 regardless of steepness."""
        for steepness in [12.0, 100.0, 1000.0, 5000.0]:
            at_mid = _curve_progress(0.5, AlphaCurve.SIGMOID, steepness=steepness)
            assert at_mid == pytest.approx(0.5, abs=1e-9), f"steepness={steepness}"

    def test_sigmoid_helper_equivalence(self):
        """_sigmoid helper must be equivalent to 1/(1+exp(-x)) for normal values."""
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            expected = 1.0 / (1.0 + math.exp(-x))
            actual = _sigmoid(x)
            assert actual == pytest.approx(expected, rel=1e-12)

    def test_sigmoid_helper_extreme_values(self):
        """_sigmoid must handle extreme values without overflow."""
        # Large positive → 1.0
        assert _sigmoid(1000.0) == pytest.approx(1.0, abs=1e-300)
        # Large negative → 0.0
        assert _sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-300)

    @given(
        t=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        steepness=st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_curve_progress_always_bounded(self, t: float, steepness: float):
        """Property: _curve_progress with SIGMOID never raises and stays in [0, 1]."""
        result = _curve_progress(t, AlphaCurve.SIGMOID, steepness=steepness)
        assert 0.0 <= result <= 1.0, f"t={t}, steepness={steepness}, result={result}"

    @given(steepness=st.floats(min_value=0.1, max_value=10000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_curve_progress_monotonic(self, steepness: float):
        """Property: _curve_progress is monotonically increasing in t."""
        prev = -1.0
        for i in range(11):
            t = i / 10.0
            result = _curve_progress(t, AlphaCurve.SIGMOID, steepness=steepness)
            assert result >= prev, f"Not monotonic at t={t}, steepness={steepness}"
            prev = result
