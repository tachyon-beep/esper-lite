"""Tests for the Stage-2 HRA MAJOR-1 acceptance TELEMETRY WRAPPER.

The wrapper is the telemetry-reading half of the gate in
docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md: §1 validity, §0 provenance,
burn-in W, §8B floored-exclusion, §8E covariates, G3/G4, and the tier-gated call into the pure
scorer (stage2_acceptance.py). These tests ARE the pre-registered contract — do not weaken them
to make a run pass.
"""

import pytest

from esper.simic.telemetry.stage2_acceptance import GateResult, Verdict, calibrate_delta, level, vol
from esper.simic.telemetry.stage2_acceptance_packet import (
    ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
    CalibrationReport,
    FrozenThresholds,
    Leg,
    LegSeries,
    RunMeta,
    SeedPair,
    Stage2Report,
    UpdateRow,
    Validity,
    burn_in_discard,
    calibrate_off,
    floored_exclusion,
    leg_series,
    score,
    sqrt_unexplained_series,
    validate_pair,
    var_returns_series,
)


def _update(**overrides) -> UpdateRow:
    """Build an UpdateRow with benign defaults; override only the fields under test.

    Defaults describe a healthy OFF-leg update (ev_* per-stream columns null; comfortably
    above the ev_return_variance floor).
    """
    fields: dict[str, object] = dict(
        inner_epoch=0,
        batch=0,
        explained_variance=0.5,
        ev_sum=None,
        ev_main=None,
        ev_cf=None,
        ev_return_variance=10.0,
        pre_norm_advantage_std=1.0,
        return_std=3.0,
    )
    fields.update(overrides)
    return UpdateRow(**fields)  # type: ignore[arg-type]


# ---- S1: burn-in discard (§8-W) ----


def test_burn_in_discard_drops_first_w_updates():
    rows = [_update(batch=i) for i in range(5)]
    kept = burn_in_discard(rows, w=2)
    assert [r.batch for r in kept] == [2, 3, 4]


def test_burn_in_discard_zero_window_keeps_all():
    rows = [_update(batch=i) for i in range(3)]
    assert burn_in_discard(rows, w=0) == rows


def test_burn_in_discard_negative_window_is_rejected():
    # A negative burn-in would tail-slice — silently wrong. Fail loud.
    with pytest.raises(ValueError):
        burn_in_discard([_update()], w=-1)


# ---- S1: §8B floored-update exclusion ----


def test_floored_exclusion_drops_updates_at_or_below_floor():
    # §8B: exclude updates where ev_return_variance <= floor (the frozen 1.0), report the fraction.
    rows = [
        _update(ev_return_variance=2.0),
        _update(ev_return_variance=0.5),  # below floor -> excluded
        _update(ev_return_variance=1.0),  # AT floor -> excluded (<=)
        _update(ev_return_variance=3.0),
    ]
    kept, floored_fraction = floored_exclusion(rows, floor=1.0)
    assert [r.ev_return_variance for r in kept] == [2.0, 3.0]
    assert floored_fraction == pytest.approx(0.5)


def test_floored_exclusion_all_floored_reports_fraction_one_and_empty_kept():
    rows = [_update(ev_return_variance=0.5), _update(ev_return_variance=1.0)]
    kept, floored_fraction = floored_exclusion(rows, floor=1.0)
    assert kept == []
    assert floored_fraction == pytest.approx(1.0)


def test_floored_exclusion_none_floored_reports_zero_fraction():
    rows = [_update(ev_return_variance=5.0), _update(ev_return_variance=2.0)]
    kept, floored_fraction = floored_exclusion(rows, floor=1.0)
    assert len(kept) == 2
    assert floored_fraction == pytest.approx(0.0)


def test_floored_exclusion_empty_input_fails_loud():
    # An empty leg is a §1 validity failure surfaced upstream; the primitive must not
    # silently return a 0/0 fraction.
    with pytest.raises(ValueError):
        floored_exclusion([], floor=1.0)


# ---- S1: scale-invariant residual series (§4) ----


def test_sqrt_unexplained_series_maps_ev_to_sqrt_one_minus_ev():
    # §4: the return-scale-invariant residual is sqrt(1 - ev). ev=0.75->0.5, 1.0->0.0, 0.0->1.0.
    assert sqrt_unexplained_series([0.75, 1.0, 0.0]) == pytest.approx([0.5, 0.0, 1.0])


def test_sqrt_unexplained_series_clamps_tiny_fp_negative_to_zero():
    # ev is floored at <=1 by construction, but FP can make 1-ev a tiny negative; sqrt must
    # not NaN. ev slightly above 1.0.
    out = sqrt_unexplained_series([1.0 + 1e-12])
    assert out[0] == pytest.approx(0.0)


def test_sqrt_unexplained_series_empty_input_fails_loud():
    with pytest.raises(ValueError):
        sqrt_unexplained_series([])


# ---- S1: Var(returns_total) covariate series (§8E) ----


def test_var_returns_series_squares_return_std():
    # §8E: Var(returns_total) = return_std**2, reported per leg as the confound covariate.
    assert var_returns_series([2.0, 3.0]) == pytest.approx([4.0, 9.0])


def test_var_returns_series_empty_input_fails_loud():
    with pytest.raises(ValueError):
        var_returns_series([])


# ---- S2: per-leg reduction to LegSeries ----


def _off_rows(expl: list[float], **common) -> list[UpdateRow]:
    """OFF-leg rows: per-stream ev_* columns stay null; total EV is explained_variance."""
    return [_update(batch=i, explained_variance=e, **common) for i, e in enumerate(expl)]


def _on_rows(ev_sum: list[float], ev_main: list[float], **common) -> list[UpdateRow]:
    """ON-leg rows: ev_sum is the total-EV comparand; ev_main feeds the §5 MECH guard."""
    return [
        _update(batch=i, explained_variance=s, ev_sum=s, ev_main=m, ev_cf=0.1, **common)
        for i, (s, m) in enumerate(zip(ev_sum, ev_main, strict=True))
    ]


def test_leg_series_applies_burn_in_then_flooring_before_stats():
    # 2 burn-in updates + 1 floored update must not enter the scored series.
    rows = [
        _update(batch=0, explained_variance=0.9, ev_return_variance=5.0),  # burn-in (w=2)
        _update(batch=1, explained_variance=0.9, ev_return_variance=5.0),  # burn-in
        _update(batch=2, explained_variance=0.30, ev_return_variance=0.5),  # floored out (<=1.0)
        _update(batch=3, explained_variance=0.6, ev_return_variance=5.0),
        _update(batch=4, explained_variance=0.7, ev_return_variance=5.0),
        _update(batch=5, explained_variance=0.8, ev_return_variance=5.0),
    ]
    ls = leg_series(rows, leg=Leg.OFF, w=2, floor=1.0)
    assert ls.n_updates_scored == 3  # 6 total - 2 burn-in - 1 floored
    assert ls.floored_fraction == pytest.approx(1 / 4)  # 1 of the 4 post-burn-in updates floored
    assert ls.ev_level == pytest.approx(level([0.6, 0.7, 0.8]))


def test_leg_series_off_reads_explained_variance_on_reads_ev_sum():
    off = leg_series(_off_rows([0.4, 0.5, 0.6]), leg=Leg.OFF, w=0)
    on = leg_series(_on_rows([0.7, 0.8, 0.9], [0.1, 0.2, 0.3]), leg=Leg.ON, w=0)
    assert off.ev_level == pytest.approx(level([0.4, 0.5, 0.6]))
    assert on.ev_level == pytest.approx(level([0.7, 0.8, 0.9]))


def test_leg_series_ev_main_vol_present_on_on_leg_absent_on_off_leg():
    off = leg_series(_off_rows([0.4, 0.5, 0.6, 0.7]), leg=Leg.OFF, w=0)
    on = leg_series(
        _on_rows([0.7, 0.8, 0.9, 0.6], ev_main=[0.10, 0.20, 0.30, 0.40]), leg=Leg.ON, w=0
    )
    assert off.ev_main_vol is None
    assert on.ev_main_vol == pytest.approx(vol([0.10, 0.20, 0.30, 0.40]))


def test_leg_series_adv_residual_vol_is_iqr_of_sqrt_unexplained_not_raw_ev():
    # §4 gated statistic: IQR(sqrt(1 - ev)), NOT IQR(ev) and NOT raw pre_norm_advantage_std.
    ev = [0.75, 0.96, 0.51, 0.84]
    ls = leg_series(_off_rows(ev), leg=Leg.OFF, w=0)
    assert ls.adv_residual_vol == pytest.approx(vol(sqrt_unexplained_series(ev)))
    assert ls.adv_residual_vol != pytest.approx(vol(ev))


def test_leg_series_var_returns_covariate_from_return_std_squared():
    # §8E: level + IQR of Var(returns_total) = return_std**2.
    stds = [2.0, 3.0, 4.0, 5.0]
    rows = [_update(batch=i, explained_variance=0.5, return_std=s) for i, s in enumerate(stds)]
    ls = leg_series(rows, leg=Leg.OFF, w=0)
    assert ls.var_returns_level == pytest.approx(level(var_returns_series(stds)))
    assert ls.var_returns_vol == pytest.approx(vol(var_returns_series(stds)))


def test_leg_series_reports_ev_return_variance_distribution_for_the_caveat():
    # §8B per-stream blindness: the ev_return_variance distribution must travel into the packet.
    rows = [
        _update(batch=i, explained_variance=0.5, ev_return_variance=v)
        for i, v in enumerate([2.0, 4.0, 6.0, 8.0])
    ]
    ls = leg_series(rows, leg=Leg.OFF, w=0)
    assert ls.ev_return_variance_min == pytest.approx(2.0)
    assert ls.ev_return_variance_median == pytest.approx(5.0)
    assert ls.ev_return_variance_max == pytest.approx(8.0)


def test_leg_series_type_is_legseries():
    ls = leg_series(_off_rows([0.4, 0.5, 0.6]), leg=Leg.OFF, w=0)
    assert isinstance(ls, LegSeries)
    assert ls.leg is Leg.OFF


# ---- S3: §1 validity gates (validate_pair) ----


def _meta(
    leg: Leg,
    *,
    seed: int = 7,
    reward_mode: str = "SHAPED",
    actor_advantage_source: str = ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
) -> RunMeta:
    return RunMeta(
        run_dir=f"/run/{leg.value}",
        seed=seed,
        reward_mode=reward_mode,
        actor_advantage_source=actor_advantage_source,
    )


def _clean_on_rows(n: int = 3) -> list[UpdateRow]:
    return _on_rows(ev_sum=[0.8] * n, ev_main=[0.2] * n, ev_return_variance=5.0)


def _clean_off_rows(n: int = 3) -> list[UpdateRow]:
    return _off_rows([0.8] * n, ev_return_variance=5.0)


def test_validate_pair_accepts_a_clean_fresh_init_pair():
    v = validate_pair(
        _meta(Leg.ON), _clean_on_rows(), _meta(Leg.OFF), _clean_off_rows(), w=0, budget=3
    )
    assert isinstance(v, Validity)
    assert v.valid is True
    assert v.reasons == ()


def test_validate_pair_rejects_seed_mismatch():
    v = validate_pair(
        _meta(Leg.ON, seed=7), _clean_on_rows(),
        _meta(Leg.OFF, seed=8), _clean_off_rows(), w=0, budget=3,
    )
    assert v.valid is False
    assert any("seed" in r.lower() for r in v.reasons)


def test_validate_pair_rejects_reward_mode_mismatch():
    v = validate_pair(
        _meta(Leg.ON, reward_mode="SHAPED"), _clean_on_rows(),
        _meta(Leg.OFF, reward_mode="SIMPLIFIED"), _clean_off_rows(), w=0, budget=3,
    )
    assert v.valid is False
    assert any("reward_mode" in r for r in v.reasons)


def test_validate_pair_rejects_non_total_reconstructed_actor_source():
    # §0/§1 scope-fence: running objective B under this gate is a bug, not a null.
    v = validate_pair(
        _meta(Leg.ON, actor_advantage_source="main_only"), _clean_on_rows(),
        _meta(Leg.OFF), _clean_off_rows(), w=0, budget=3,
    )
    assert v.valid is False
    assert any("actor_advantage_source" in r for r in v.reasons)


def test_validate_pair_rejects_off_arm_carrying_hra_metric():
    # §1: an OFF arm that reads ev_sum/ev_main/ev_cf is a bug, not a null.
    off = _on_rows(ev_sum=[0.8, 0.8, 0.8], ev_main=[0.2, 0.2, 0.2], ev_return_variance=5.0)
    v = validate_pair(_meta(Leg.ON), _clean_on_rows(), _meta(Leg.OFF), off, w=0, budget=3)
    assert v.valid is False
    assert any("OFF" in r and "hra" in r.lower() for r in v.reasons)


def test_validate_pair_rejects_on_arm_missing_hra_metric():
    # §1: telemetry missing ev_main/ev_cf/ev_sum on the ON arm.
    on = _off_rows([0.8, 0.8, 0.8], ev_return_variance=5.0)  # ev_* all None -> looks OFF
    v = validate_pair(_meta(Leg.ON), on, _meta(Leg.OFF), _clean_off_rows(), w=0, budget=3)
    assert v.valid is False
    assert any("ON" in r and "hra" in r.lower() for r in v.reasons)


def test_validate_pair_rejects_incomplete_run_below_budget():
    # §1: fewer scored updates than the pre-registered budget after burn-in.
    v = validate_pair(
        _meta(Leg.ON), _clean_on_rows(3), _meta(Leg.OFF), _clean_off_rows(3), w=0, budget=5
    )
    assert v.valid is False
    assert any("budget" in r.lower() or "incomplete" in r.lower() for r in v.reasons)


def test_validate_pair_rejects_non_finite_scored_metric():
    on = _clean_on_rows(3)
    bad = _update(
        batch=1, explained_variance=0.8, ev_sum=0.8, ev_main=0.2, ev_cf=0.1,
        ev_return_variance=5.0, pre_norm_advantage_std=float("nan"),
    )
    on = [on[0], bad, on[2]]
    v = validate_pair(_meta(Leg.ON), on, _meta(Leg.OFF), _clean_off_rows(3), w=0, budget=3)
    assert v.valid is False
    assert any("finite" in r.lower() for r in v.reasons)


def test_validate_pair_rejects_materially_asymmetric_floored_fraction():
    # §8B(ii): a materially asymmetric floored fraction between arms invalidates the comparison.
    on = _on_rows(ev_sum=[0.8] * 4, ev_main=[0.2] * 4, ev_return_variance=5.0)  # 0% floored
    off = [  # 25% floored
        _update(batch=0, explained_variance=0.8, ev_return_variance=0.5),  # floored
        _update(batch=1, explained_variance=0.8, ev_return_variance=5.0),
        _update(batch=2, explained_variance=0.8, ev_return_variance=5.0),
        _update(batch=3, explained_variance=0.8, ev_return_variance=5.0),
    ]
    v = validate_pair(
        _meta(Leg.ON), on, _meta(Leg.OFF), off, w=0, budget=2, floored_asymmetry_max=0.10
    )
    assert v.valid is False
    assert any("floored" in r.lower() and "asymm" in r.lower() for r in v.reasons)


def test_validate_pair_reports_all_violations_not_just_first():
    # Two independent breaches -> both surfaced (the packet reports every problem).
    v = validate_pair(
        _meta(Leg.ON, seed=7, reward_mode="SHAPED"), _clean_on_rows(),
        _meta(Leg.OFF, seed=8, reward_mode="SIMPLIFIED"), _clean_off_rows(), w=0, budget=3,
    )
    assert v.valid is False
    assert len(v.reasons) >= 2


# ---- S4: assembly + calibrate/score entrypoints + provenance block ----


def _legseries(
    leg: Leg,
    *,
    ev_level: float,
    adv_residual_vol: float,
    ev_vol: float = 0.10,
    ev_main_vol: float | None = None,
    floored_fraction: float = 0.0,
) -> LegSeries:
    return LegSeries(
        leg=leg,
        n_updates_scored=200,
        floored_fraction=floored_fraction,
        ev_level=ev_level,
        ev_vol=ev_vol,
        adv_residual_vol=adv_residual_vol,
        raw_adv_std_vol=0.10,
        var_returns_level=1.0,
        var_returns_vol=0.10,
        ev_main_vol=ev_main_vol,
        ev_return_variance_min=2.0,
        ev_return_variance_median=5.0,
        ev_return_variance_max=8.0,
    )


def _accept_pair(seed: int) -> SeedPair:
    # ev_sum non-inferior (+0.05), adv-residual vol down 20%, ev_main vol below OFF baseline,
    # no acc regression, no param inflation -> a clean ACCEPT-shaped seed.
    return SeedPair(
        seed=seed,
        on=_legseries(Leg.ON, ev_level=0.85, adv_residual_vol=0.80, ev_main_vol=0.03),
        off=_legseries(Leg.OFF, ev_level=0.80, ev_vol=0.10, adv_residual_vol=1.00),
        d_val_acc=0.0,
        d_added_params=0.0,
    )


_THRESHOLDS = FrozenThresholds(delta=0.05, eps_rel=0.10, tau_acc=0.3, delta_param_max=1000.0, w=0)


def test_calibrate_off_freezes_delta_from_off_arms_and_echoes_eps_rel():
    off_legs = [
        _legseries(Leg.OFF, ev_level=e, ev_vol=0.10, adv_residual_vol=1.0)
        for e in [0.80, 0.82, 0.79, 0.81, 0.80]
    ]
    cal = calibrate_off(off_legs)
    assert isinstance(cal, CalibrationReport)
    # δ is the scorer's calibrate_delta over the OFF ev levels + IQRs (same fixed rng seed).
    assert cal.delta == pytest.approx(
        calibrate_delta([0.80, 0.82, 0.79, 0.81, 0.80], [0.10] * 5)
    )
    assert cal.eps_rel == pytest.approx(0.10)


def test_calibrate_off_reads_only_off_legs():
    # Passing an ON leg into calibration is a freeze-order breach (§10 no peeking).
    with pytest.raises(ValueError):
        calibrate_off([_legseries(Leg.ON, ev_level=0.8, adv_residual_vol=1.0, ev_main_vol=0.1)])


def test_score_clean_n10_pairset_accepts():
    pairs = [_accept_pair(i) for i in range(10)]
    report = score(
        pairs, _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True
    )
    assert isinstance(report, Stage2Report)
    assert report.verdict is Verdict.ACCEPT
    assert report.leg_a is GateResult.PASS
    assert report.leg_b is GateResult.PASS
    assert report.mech_hold is True


def test_score_clean_n5_pairset_is_screen_pass_never_accept():
    pairs = [_accept_pair(i) for i in range(5)]
    report = score(
        pairs, _THRESHOLDS, n=5, validity=Validity(True, ()), g3_hold=True, g4_hold=True
    )
    assert report.verdict is Verdict.SCREEN_PASS


def test_score_invalid_pair_is_invalid_and_computes_no_legs():
    pairs = [_accept_pair(i) for i in range(10)]
    report = score(
        pairs, _THRESHOLDS, n=10,
        validity=Validity(False, ("seed mismatch",)), g3_hold=True, g4_hold=True,
    )
    assert report.verdict is Verdict.INVALID
    assert report.leg_a is None  # no interpretation on a broken run (§1)


def test_score_ev_sum_regression_is_hard_reject():
    # §3 MAJOR-1 hard override: ev_sum regresses below -δ -> REJECT regardless of anything else.
    def regress(seed: int) -> SeedPair:
        p = _accept_pair(seed)
        return SeedPair(
            seed=seed,
            on=_legseries(Leg.ON, ev_level=0.60, adv_residual_vol=0.80, ev_main_vol=0.03),
            off=p.off, d_val_acc=0.0, d_added_params=0.0,
        )

    pairs = [regress(i) for i in range(10)]
    report = score(
        pairs, _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True
    )
    assert report.leg_a is GateResult.FAIL
    assert report.verdict is Verdict.REJECT


def test_score_builds_delta_a_and_delta_b_from_leg_series():
    pairs = [_accept_pair(i) for i in range(10)]
    report = score(
        pairs, _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True
    )
    # Δ_A = ev_level_on - ev_level_off = 0.85 - 0.80; Δ_B = (0.80 - 1.00)/1.00.
    assert report.delta_a[0] == pytest.approx(0.05)
    assert report.delta_b[0] == pytest.approx(-0.20)


def test_score_report_carries_stage0_provenance_block_verbatim():
    # §0/MAJOR-3: the report must carry the provenance block so Stage-0 evidence can never be
    # laundered into HRA evidence.
    report = score(
        [_accept_pair(i) for i in range(10)],
        _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True,
    )
    assert "Stage-0 variance gate" in report.provenance
    assert "HRA evidence: NO" in report.provenance


def test_score_marks_diagnostic_scalars_unavailable_until_emission_lands():
    # §9 value_main_target_scale / cf_value_target_scale are not emitted yet (S7); the report
    # must mark them unavailable, NOT fail and NOT silently default.
    report = score(
        [_accept_pair(i) for i in range(10)],
        _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True,
    )
    assert report.diagnostic_scalars_available is False
