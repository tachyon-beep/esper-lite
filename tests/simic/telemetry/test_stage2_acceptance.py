"""Tests for the Stage-2 HRA MAJOR-1 acceptance scorer.

Encodes the frozen predicates from
docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md. These tests ARE the
pre-registered contract — do not weaken them to make a run pass.
"""

import numpy as np
import pytest

from esper.simic.telemetry.stage2_acceptance import (
    GateResult,
    Verdict,
    _wilcoxon_signed_rank_p_greater,
    calibrate_delta,
    composite_verdict,
    leg_a,
    leg_b,
    level,
    mech_guard,
    safety_g1,
    safety_g2,
    vol,
)


# ---- LEG A: ev_sum non-inferiority floor (§3) ----


def test_leg_a_n5_screen_passes_when_4_of_5_seeds_non_inferior():
    # non-inferior := Δ_A >= -δ. Four seeds clear -0.05, one does not.
    delta_a = [0.02, -0.01, 0.10, -0.04, -0.20]
    assert leg_a(delta_a, delta=0.05, n=5) is GateResult.PASS


def test_leg_a_n5_screen_fails_when_only_2_of_5_seeds_non_inferior():
    delta_a = [0.02, -0.01, -0.10, -0.06, -0.20]
    assert leg_a(delta_a, delta=0.05, n=5) is GateResult.FAIL


def test_leg_a_n10_passes_with_clear_non_inferiority():
    # All comfortably >= -δ; the one-sided Wilcoxon on {Δ+δ} rejects "ON worse than OFF-δ".
    delta_a = [0.10, 0.08, 0.12, 0.05, 0.09, 0.11, 0.07, 0.06, 0.13, 0.10]
    assert leg_a(delta_a, delta=0.05, n=10) is GateResult.PASS


def test_leg_a_n10_fails_when_median_below_negative_margin():
    delta_a = [-0.10] * 10
    assert leg_a(delta_a, delta=0.05, n=10) is GateResult.FAIL


# ---- LEG B: advantage-path volatility reduction (§4) ----


def test_leg_b_n5_screen_passes_when_4_of_5_seeds_reduce_volatility():
    delta_b = [-0.20, -0.15, -0.30, -0.10, 0.05]  # 4 negative
    assert leg_b(delta_b, eps=0.10, n=5) is GateResult.PASS


def test_leg_b_n5_screen_inconclusive_at_3_of_5():
    delta_b = [-0.20, -0.15, -0.30, 0.10, 0.05]  # 3 negative
    assert leg_b(delta_b, eps=0.10, n=5) is GateResult.INCONCLUSIVE


def test_leg_b_n5_screen_fails_below_3_of_5():
    delta_b = [-0.20, -0.10, 0.30, 0.10, 0.05]  # 2 negative
    assert leg_b(delta_b, eps=0.10, n=5) is GateResult.FAIL


def test_leg_b_n10_passes_with_clear_volatility_reduction():
    delta_b = [-0.20] * 10  # median <= -eps and Wilcoxon-significant
    assert leg_b(delta_b, eps=0.10, n=10) is GateResult.PASS


def test_leg_b_n10_fails_when_median_non_negative():
    delta_b = [0.10] * 10  # ON no steadier / worse
    assert leg_b(delta_b, eps=0.10, n=10) is GateResult.FAIL


def test_leg_b_n10_inconclusive_when_reduction_below_epsilon():
    delta_b = [-0.03] * 10  # negative but does not clear the -eps floor
    assert leg_b(delta_b, eps=0.10, n=10) is GateResult.INCONCLUSIVE


# ---- MECH guard (§5) and safety gates (§6) ----


def test_mech_guard_holds_when_ev_main_vol_lower_on_4_of_5():
    ev_main_vol_on = [0.05, 0.04, 0.03, 0.06, 0.20]
    expl_vol_off = [0.10, 0.10, 0.10, 0.10, 0.10]
    assert mech_guard(ev_main_vol_on, expl_vol_off) is True


def test_mech_guard_fails_when_only_3_of_5_lower():
    ev_main_vol_on = [0.05, 0.04, 0.03, 0.60, 0.20]
    expl_vol_off = [0.10, 0.10, 0.10, 0.10, 0.10]
    assert mech_guard(ev_main_vol_on, expl_vol_off) is False


def test_safety_g1_holds_when_val_acc_median_within_deadband():
    d_acc = [0.1, -0.2, 0.0, -0.1, 0.05]  # median 0.0 >= -0.3pp
    assert safety_g1(d_acc, tau_acc=0.3) is GateResult.PASS


def test_safety_g1_fails_when_val_acc_regresses_past_deadband():
    d_acc = [-0.5, -0.6, -0.4, -0.7, -0.5]  # median -0.5 < -0.3pp
    assert safety_g1(d_acc, tau_acc=0.3) is GateResult.FAIL


def test_safety_g2_holds_when_param_inflation_below_ceiling():
    d_params = [100.0, -50.0, 0.0, 20.0, 10.0]  # median 10 <= cap
    assert safety_g2(d_params, delta_param_max=1000.0) is True


def test_safety_g2_fails_when_param_inflation_exceeds_ceiling():
    d_params = [2000.0, 1500.0, 3000.0, 1800.0, 2500.0]  # median 2000 > 1000
    assert safety_g2(d_params, delta_param_max=1000.0) is False


# ---- level / vol / delta calibration (§2, §3, §11) ----


def test_level_is_median_over_series():
    assert level([0.1, 0.2, 0.3, 0.4, 0.5]) == pytest.approx(0.3)


def test_vol_is_interquartile_range():
    assert vol([0, 1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.0)  # P75 - P25 = 6 - 2


def test_calibrate_delta_fallback_uses_half_median_iqr_when_few_off_runs():
    # <5 OFF runs → fallback max(0.05, 0.5 * median(OFF IQRs))
    d = calibrate_delta([0.30, 0.32, 0.31], [0.20, 0.24, 0.22])
    assert d == pytest.approx(0.11)


def test_calibrate_delta_floors_at_0_05_when_off_corpus_quiet():
    # >=5 OFF runs, all identical → bootstrap SE = 0 → clamped to the 0.05 floor
    d = calibrate_delta([0.30] * 6, [0.0] * 6, rng=np.random.default_rng(0))
    assert d == pytest.approx(0.05)


# ---- composite verdict (§7) ----

_BASE = dict(  # noqa: C408 - explicit for readability
    valid=True,
    n=10,
    leg_a_result=GateResult.PASS,
    leg_b_result=GateResult.PASS,
    mech_hold=True,
    g1_result=GateResult.PASS,
    g2_hold=True,
    g3_hold=True,
    g4_hold=True,
)


def test_composite_accepts_when_all_pass_at_n10():
    assert composite_verdict(**_BASE) is Verdict.ACCEPT


def test_composite_screen_pass_never_accepts_at_n5():
    # §7: ACCEPT is never banked at the n=5 screen — a clean screen is SCREEN_PASS,
    # a distinct outcome from a genuine n=10 null band.
    assert composite_verdict(**{**_BASE, "n": 5}) is Verdict.SCREEN_PASS


def test_composite_inconclusive_when_g4_channel_soft_fails():
    assert composite_verdict(**{**_BASE, "g4_hold": False}) is Verdict.INCONCLUSIVE


def test_composite_invalid_short_circuits_before_interpretation():
    assert composite_verdict(**{**_BASE, "valid": False}) is Verdict.INVALID


def test_composite_rejects_on_leg_a_fail_regardless_of_ev_main():
    # MAJOR-1 hard override
    assert composite_verdict(**{**_BASE, "leg_a_result": GateResult.FAIL}) is Verdict.REJECT


def test_composite_rejects_on_mech_fail():
    assert composite_verdict(**{**_BASE, "mech_hold": False}) is Verdict.REJECT


def test_composite_rejects_on_g1_regression():
    assert composite_verdict(**{**_BASE, "g1_result": GateResult.FAIL}) is Verdict.REJECT


def test_composite_rejects_on_leg_b_fail():
    assert composite_verdict(**{**_BASE, "leg_b_result": GateResult.FAIL}) is Verdict.REJECT


def test_composite_inconclusive_on_leg_b_inconclusive():
    assert (
        composite_verdict(**{**_BASE, "leg_b_result": GateResult.INCONCLUSIVE})
        is Verdict.INCONCLUSIVE
    )


def test_composite_inconclusive_when_safety_soft_fails_despite_leg_b_pass():
    assert composite_verdict(**{**_BASE, "g2_hold": False}) is Verdict.INCONCLUSIVE


def test_composite_does_not_accept_when_leg_a_inconclusive():
    # spec §7: ACCEPT ⟺ LEG_A==PASS — a non-PASS leg_a must never reach ACCEPT.
    assert (
        composite_verdict(**{**_BASE, "leg_a_result": GateResult.INCONCLUSIVE})
        is Verdict.INCONCLUSIVE
    )


def test_composite_does_not_accept_when_g1_inconclusive():
    assert (
        composite_verdict(**{**_BASE, "g1_result": GateResult.INCONCLUSIVE})
        is Verdict.INCONCLUSIVE
    )


# ---- fail-loud input validation (spec §1, repo no-defensive-programming policy) ----


def test_leg_a_raises_on_length_mismatch_with_tier():
    with pytest.raises(ValueError):
        leg_a([0.1, 0.2, 0.3], delta=0.05, n=10)


def test_leg_a_raises_on_invalid_tier():
    with pytest.raises(ValueError):
        leg_a([0.1] * 7, delta=0.05, n=7)


def test_leg_b_raises_on_length_mismatch_with_tier():
    with pytest.raises(ValueError):
        leg_b([0.1, 0.2], eps=0.10, n=5)


def test_leg_a_raises_on_non_finite_delta():
    with pytest.raises(ValueError):
        leg_a([0.1, float("nan"), 0.2, 0.3, 0.4], delta=0.05, n=5)


def test_level_raises_on_empty_series():
    with pytest.raises(ValueError):
        level([])


def test_calibrate_delta_raises_on_empty_off_corpus():
    with pytest.raises(ValueError):
        calibrate_delta([], [])


def test_wilcoxon_guards_against_enumeration_blowup():
    with pytest.raises(ValueError):
        _wilcoxon_signed_rank_p_greater([0.1] * 21)


def test_leg_a_n10_all_exactly_at_margin_fails():
    # Δ_A = -δ for every seed → {Δ+δ} all zero → Wilcoxon m=0 → p=1.0 → FAIL,
    # even though median(Δ_A) == -δ. Surprising but correct (evidence, not a tie-break).
    assert leg_a([-0.05] * 10, delta=0.05, n=10) is GateResult.FAIL


def test_mech_guard_scales_threshold_at_n10():
    on = [0.05] * 7 + [0.5] * 3  # only 7/10 lower — below ⌈0.8·10⌉ = 8
    off = [0.10] * 10
    assert mech_guard(on, off) is False


def test_mech_guard_holds_at_n10_with_8_of_10():
    on = [0.05] * 8 + [0.5] * 2
    off = [0.10] * 10
    assert mech_guard(on, off) is True
