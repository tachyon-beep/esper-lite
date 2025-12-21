"""Property-based tests for the alpha controller invariants."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from esper.kasmina.alpha_controller import AlphaController
from esper.leyline.alpha import AlphaCurve, AlphaMode


_curves = st.sampled_from([AlphaCurve.LINEAR, AlphaCurve.COSINE, AlphaCurve.SIGMOID])


class TestAlphaControllerMonotonic:
    @given(
        alpha_start=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        alpha_target=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        alpha_steps_total=st.integers(min_value=0, max_value=25),
        alpha_curve=_curves,
    )
    @settings(max_examples=200)
    def test_monotone_no_overshoot_and_snaps(
        self,
        alpha_start: float,
        alpha_target: float,
        alpha_steps_total: int,
        alpha_curve: AlphaCurve,
    ) -> None:
        controller = AlphaController(alpha=alpha_start, alpha_start=alpha_start, alpha_target=alpha_start)
        controller.retarget(
            alpha_target=alpha_target,
            alpha_steps_total=alpha_steps_total,
            alpha_curve=alpha_curve,
        )

        alphas: list[float] = [controller.alpha]
        for _ in range(max(alpha_steps_total, 1) + 2):
            controller.step()
            alphas.append(controller.alpha)

        lo = min(alpha_start, alpha_target)
        hi = max(alpha_start, alpha_target)
        assert all(lo <= a <= hi for a in alphas), f"Overshoot: {alphas}"

        if alpha_target > alpha_start:
            assert all(a0 <= a1 for a0, a1 in zip(alphas, alphas[1:], strict=False))
        elif alpha_target < alpha_start:
            assert all(a0 >= a1 for a0, a1 in zip(alphas, alphas[1:], strict=False))
        else:
            assert all(a == alpha_target for a in alphas)

        if alpha_steps_total == 0:
            assert controller.alpha == alpha_target
            assert controller.alpha_mode == AlphaMode.HOLD
        else:
            for _ in range(alpha_steps_total):
                controller.step()
            assert controller.alpha == alpha_target
            assert controller.alpha_mode == AlphaMode.HOLD


class TestAlphaControllerRetargetRules:
    def test_retarget_only_allowed_from_hold(self) -> None:
        controller = AlphaController(alpha=0.0, alpha_start=0.0, alpha_target=0.0)
        controller.retarget(alpha_target=1.0, alpha_steps_total=5, alpha_curve=AlphaCurve.LINEAR)
        assert controller.alpha_mode == AlphaMode.UP

        with pytest.raises(ValueError, match="only allowed from HOLD"):
            controller.retarget(alpha_target=0.5, alpha_steps_total=5, alpha_curve=AlphaCurve.LINEAR)


class TestAlphaControllerCheckpointRoundtrip:
    @given(
        alpha_start=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        alpha_target=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        alpha_steps_total=st.integers(min_value=1, max_value=25),
        alpha_steps_done=st.integers(min_value=0, max_value=24),
        alpha_curve=_curves,
    )
    @settings(max_examples=100)
    def test_roundtrip_preserves_trajectory(
        self,
        alpha_start: float,
        alpha_target: float,
        alpha_steps_total: int,
        alpha_steps_done: int,
        alpha_curve: AlphaCurve,
    ) -> None:
        alpha_steps_done = min(alpha_steps_done, alpha_steps_total - 1)

        c1 = AlphaController(alpha=alpha_start, alpha_start=alpha_start, alpha_target=alpha_start)
        c1.retarget(alpha_target=alpha_target, alpha_steps_total=alpha_steps_total, alpha_curve=alpha_curve)
        for _ in range(alpha_steps_done):
            c1.step()

        saved = c1.to_dict()
        c2 = AlphaController.from_dict(saved)

        # Advance both controllers to completion
        while c1.alpha_mode != AlphaMode.HOLD:
            c1.step()
        while c2.alpha_mode != AlphaMode.HOLD:
            c2.step()

        assert c1.alpha == c2.alpha == alpha_target
        assert c1.alpha_steps_total == c2.alpha_steps_total


def test_alpha_controller_is_dataclass_slots() -> None:
    controller = AlphaController()
    assert is_dataclass(controller)
    # Sanity: asdict works (not required at runtime, but useful in tests/debugging)
    assert isinstance(asdict(controller), dict)
