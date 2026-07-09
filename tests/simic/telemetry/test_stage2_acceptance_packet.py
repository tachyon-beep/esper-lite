"""Tests for the Stage-2 HRA MAJOR-1 acceptance TELEMETRY WRAPPER.

The wrapper is the telemetry-reading half of the gate in
docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md: §1 validity, §0 provenance,
burn-in W, §8B floored-exclusion, §8E covariates, G3/G4, and the tier-gated call into the pure
scorer (stage2_acceptance.py). These tests ARE the pre-registered contract — do not weaken them
to make a run pass.
"""

import dataclasses

import pytest

from esper.leyline.telemetry import ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED
from esper.simic.telemetry.stage2_acceptance import GateResult, Verdict, calibrate_delta, level, vol
from esper.simic.telemetry.stage2_acceptance_packet import (
    STAGE0_PROVENANCE_BLOCK,
    CalibrationReport,
    ChurnRates,
    FrozenThresholds,
    GuardChannelCounts,
    Leg,
    LegSeries,
    RunMeta,
    SafetyEvidence,
    SeedPair,
    Stage2Report,
    UpdateRow,
    Validity,
    _optional_level_for_on,
    burn_in_discard,
    calibrate_off,
    floored_exclusion,
    leg_series,
    plateau,
    require_hra_signature,
    render_packet,
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
        value_main_target_scale=None,
        cf_value_target_scale=None,
        cf_value_loss=None,
        ev_return_variance=10.0,
        pre_norm_advantage_std=1.0,
        return_std=3.0,
        gradient_cv=0.10,
        advantage_std_floored=False,
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


def test_sqrt_unexplained_series_rejects_material_ev_above_one():
    with pytest.raises(ValueError, match="ev must be <= 1.0"):
        sqrt_unexplained_series([1.5])


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
    """ON-leg rows: ev_sum is the total-EV comparand; ev_main feeds the §5 MECH guard.

    ``cf_value_loss`` defaults to a flat (plateaued-from-birth) series so tests exercising
    other gates satisfy the §11 W-rule plateau check; override per test to probe it.
    """
    return [
        _update(
            batch=i, explained_variance=s, ev_sum=s, ev_main=m, ev_cf=0.1,
            cf_value_loss=0.02, **common,
        )
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


def test_leg_series_reports_on_target_scale_levels_when_fully_emitted():
    on = leg_series(
        _on_rows(
            [0.7, 0.8, 0.9],
            [0.1, 0.2, 0.3],
            value_main_target_scale=1.7,
            cf_value_target_scale=6.3,
        ),
        leg=Leg.ON,
        w=0,
    )
    off = leg_series(_off_rows([0.4, 0.5, 0.6]), leg=Leg.OFF, w=0)
    assert on.value_main_target_scale_level == pytest.approx(1.7)
    assert on.cf_value_target_scale_level == pytest.approx(6.3)
    assert off.value_main_target_scale_level is None
    assert off.cf_value_target_scale_level is None


def test_leg_series_rejects_partially_emitted_on_target_scale():
    rows = _on_rows([0.7, 0.8, 0.9], [0.1, 0.2, 0.3])
    rows[1] = _update(
        batch=1,
        explained_variance=0.8,
        ev_sum=0.8,
        ev_main=0.2,
        ev_cf=0.1,
        value_main_target_scale=1.7,
        cf_value_target_scale=6.3,
    )
    with pytest.raises(ValueError):
        leg_series(rows, leg=Leg.ON, w=0)


def test_optional_level_for_on_rejects_unknown_field_name():
    # An unrecognized diagnostic name must fail loud (KeyError), never silently fall through to
    # cf_value_target_scale and corrupt §9 diagnostics.
    rows = _on_rows([0.7, 0.8, 0.9], [0.1, 0.2, 0.3])
    with pytest.raises(KeyError):
        _optional_level_for_on(rows, "value_main_taget_scale")  # typo'd field name


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


def test_leg_series_reports_gradient_cv_volatility():
    cvs = [0.10, 0.20, 0.15, 0.30]
    rows = [_update(batch=i, explained_variance=0.5, gradient_cv=cv) for i, cv in enumerate(cvs)]
    ls = leg_series(rows, leg=Leg.OFF, w=0)
    assert ls.gradient_cv_vol == pytest.approx(vol(cvs))


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


def test_leg_series_reports_advantage_std_floor_fraction_for_contamination_flag():
    rows = [
        _update(batch=0, explained_variance=0.5, advantage_std_floored=False),
        _update(batch=1, explained_variance=0.5, advantage_std_floored=True),
        _update(batch=2, explained_variance=0.5, advantage_std_floored=False),
        _update(batch=3, explained_variance=0.5, advantage_std_floored=True),
    ]
    ls = leg_series(rows, leg=Leg.OFF, w=0)
    assert ls.advantage_std_floored_fraction == pytest.approx(0.5)


def test_leg_series_type_is_legseries():
    ls = leg_series(_off_rows([0.4, 0.5, 0.6]), leg=Leg.OFF, w=0)
    assert isinstance(ls, LegSeries)
    assert ls.leg is Leg.OFF


# ---- S3: §1 validity gates (validate_pair) ----


_META_PLACEMENT = (("env_devices_json", '["cuda:0"]'), ("policy_device", "cuda:0"))


def _meta(
    leg: Leg,
    *,
    seed: int = 7,
    reward_mode: str = "SHAPED",
    actor_advantage_source: str = ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
    uses_per_head_norm: bool = False,
) -> RunMeta:
    return RunMeta(
        run_dir=f"/run/{leg.value}",
        seed=seed,
        reward_mode=reward_mode,
        actor_advantage_source=actor_advantage_source,
        uses_per_head_norm=uses_per_head_norm,
        placement=_META_PLACEMENT,
    )


def _clean_on_rows(n: int = 3) -> list[UpdateRow]:
    return _on_rows(ev_sum=[0.8] * n, ev_main=[0.2] * n, ev_return_variance=5.0)


def _clean_off_rows(n: int = 3) -> list[UpdateRow]:
    return _off_rows([0.8] * n, ev_return_variance=5.0)


def test_validate_pair_accepts_a_clean_fresh_init_pair():
    # w=8 with 12 rows: the §11 ON cf_value_loss plateau check needs a trailing-8 window
    # inside the burn-in region, so a fully-clean pair carries a real-sized series.
    v = validate_pair(
        _meta(Leg.ON), _clean_on_rows(12), _meta(Leg.OFF), _clean_off_rows(12), w=8, budget=3
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


def test_validate_pair_rejects_per_head_advantage_norm_on_either_leg():
    # §8F(i): per_head_advantage_norm must be False on BOTH legs (frozen identical, §10) — it
    # changes pre_norm_advantage_std semantics and would contaminate the LEG-B comparison.
    v = validate_pair(
        _meta(Leg.ON, uses_per_head_norm=True), _clean_on_rows(),
        _meta(Leg.OFF), _clean_off_rows(), w=0, budget=3,
    )
    assert v.valid is False
    assert any("per-head" in r.lower() or "per_head" in r.lower() for r in v.reasons)


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


def test_validate_pair_rejects_off_arm_carrying_target_scale_metric():
    off = [
        _update(
            explained_variance=0.8,
            value_main_target_scale=1.7,
            cf_value_target_scale=6.3,
            ev_return_variance=5.0,
        )
        for _ in range(3)
    ]
    v = validate_pair(_meta(Leg.ON), _clean_on_rows(), _meta(Leg.OFF), off, w=0, budget=3)
    assert v.valid is False
    assert any("OFF" in r and "HRA metric" in r for r in v.reasons)


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


def test_validate_pair_rejects_frozen_config_mismatch():
    on_meta = _meta(Leg.ON)
    off_meta = RunMeta(
        run_dir="/run/off",
        seed=7,
        reward_mode="SHAPED",
        actor_advantage_source=ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED,
        uses_per_head_norm=False,
        frozen_config=(("lr", 0.002),),
    )
    on_meta = RunMeta(
        run_dir=on_meta.run_dir,
        seed=on_meta.seed,
        reward_mode=on_meta.reward_mode,
        actor_advantage_source=on_meta.actor_advantage_source,
        uses_per_head_norm=on_meta.uses_per_head_norm,
        frozen_config=(("lr", 0.001),),
    )
    v = validate_pair(on_meta, _clean_on_rows(), off_meta, _clean_off_rows(), w=0, budget=3)
    assert v.valid is False
    assert any("frozen run config" in r for r in v.reasons)


# ---- S4: assembly + calibrate/score entrypoints + provenance block ----


def _legseries(
    leg: Leg,
    *,
    ev_level: float,
    adv_residual_vol: float,
    ev_vol: float = 0.10,
    ev_main_vol: float | None = None,
    floored_fraction: float = 0.0,
    var_returns_vol: float = 0.10,
    raw_adv_std_vol: float = 0.10,
    advantage_std_floored_fraction: float = 0.0,
    gradient_cv_vol: float = 0.10,
    value_main_target_scale_level: float | None = None,
    cf_value_target_scale_level: float | None = None,
) -> LegSeries:
    return LegSeries(
        leg=leg,
        n_updates_scored=200,
        floored_fraction=floored_fraction,
        ev_level=ev_level,
        ev_vol=ev_vol,
        adv_residual_vol=adv_residual_vol,
        raw_adv_std_vol=raw_adv_std_vol,
        advantage_std_floored_fraction=advantage_std_floored_fraction,
        var_returns_level=1.0,
        var_returns_vol=var_returns_vol,
        gradient_cv_vol=gradient_cv_vol,
        ev_main_vol=ev_main_vol,
        value_main_target_scale_level=value_main_target_scale_level,
        cf_value_target_scale_level=cf_value_target_scale_level,
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


def test_score_rejects_duplicate_seed_pairset():
    pairs = [_accept_pair(i) for i in range(10)]
    pairs[1] = _accept_pair(0)
    with pytest.raises(ValueError, match="duplicate seed"):
        score(pairs, _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True)


def test_score_zero_return_variance_confound_names_return_variance():
    pairs = [_accept_pair(i) for i in range(5)]
    pairs[0] = SeedPair(
        seed=0,
        on=_legseries(Leg.ON, ev_level=0.85, adv_residual_vol=0.80, ev_main_vol=0.03),
        off=_legseries(
            Leg.OFF,
            ev_level=0.80,
            ev_vol=0.10,
            adv_residual_vol=1.00,
            var_returns_vol=0.0,
        ),
        d_val_acc=0.0,
        d_added_params=0.0,
    )

    with pytest.raises(ValueError, match="OFF return-variance volatility"):
        score(pairs, _THRESHOLDS, n=5, validity=Validity(True, ()), g3_hold=True, g4_hold=True)


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


def test_score_marks_diagnostic_scalars_unavailable_when_missing():
    # §9 value_main_target_scale / cf_value_target_scale are descriptive: old/all-missing
    # telemetry is marked unavailable, NOT silently defaulted.
    report = score(
        [_accept_pair(i) for i in range(10)],
        _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True,
    )
    assert report.diagnostic_scalars_available is False


def test_score_threads_diagnostic_scalars_when_present():
    def with_scales(seed: int) -> SeedPair:
        pair = _accept_pair(seed)
        return SeedPair(
            seed=seed,
            on=_legseries(
                Leg.ON,
                ev_level=0.85,
                adv_residual_vol=0.80,
                ev_main_vol=0.03,
                value_main_target_scale_level=1.7 + seed,
                cf_value_target_scale_level=6.3 + seed,
            ),
            off=pair.off,
            d_val_acc=0.0,
            d_added_params=0.0,
        )

    report = score(
        [with_scales(i) for i in range(10)],
        _THRESHOLDS,
        n=10,
        validity=Validity(True, ()),
        g3_hold=True,
        g4_hold=True,
    )
    assert report.diagnostic_scalars_available is True
    assert report.value_main_target_scale_levels[0] == pytest.approx(1.7)
    assert report.cf_value_target_scale_levels[0] == pytest.approx(6.3)


# ---- §4 LEG-B covariates: report + confound downgrade (drl-expert Finding 3) ----


def test_score_threads_leg_b_covariates_into_the_report():
    # §4 mandates reporting IQR(Var(returns_total)) + raw IQR(pre_norm_advantage_std) per leg —
    # they must travel with the verdict, not be dropped at the report boundary.
    report = score(
        [_accept_pair(i) for i in range(10)],
        _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True,
    )
    assert len(report.var_returns_vol_on) == 10
    assert len(report.var_returns_vol_off) == 10
    assert len(report.raw_adv_std_vol_on) == 10
    assert len(report.raw_adv_std_vol_off) == 10
    assert report.var_returns_vol_on[0] == pytest.approx(0.10)


def _confounded_pair(seed: int) -> SeedPair:
    # LEG-B "passes" (adv-residual vol -20%) BUT IQR(Var(returns)) drops the same 20% -> the win is
    # a return-regime artifact, not actor-path shielding.
    return SeedPair(
        seed=seed,
        on=_legseries(
            Leg.ON, ev_level=0.85, adv_residual_vol=0.80, ev_main_vol=0.03, var_returns_vol=0.08
        ),
        off=_legseries(Leg.OFF, ev_level=0.80, ev_vol=0.10, adv_residual_vol=1.00, var_returns_vol=0.10),
        d_val_acc=0.0,
        d_added_params=0.0,
    )


def test_score_downgrades_leg_b_when_var_returns_iqr_drops_as_much_as_the_residual():
    report = score(
        [_confounded_pair(i) for i in range(10)],
        _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True,
    )
    assert report.leg_b_confound_downgraded is True
    assert report.leg_b is GateResult.INCONCLUSIVE
    assert report.verdict is Verdict.INCONCLUSIVE  # was clean->ACCEPT; the confound un-cleans it


def test_score_downgrades_leg_b_when_gradient_cv_volatility_rises():
    def gradient_confounded(seed: int) -> SeedPair:
        return SeedPair(
            seed=seed,
            on=_legseries(
                Leg.ON,
                ev_level=0.85,
                adv_residual_vol=0.80,
                ev_main_vol=0.03,
                gradient_cv_vol=0.30,
            ),
            off=_legseries(
                Leg.OFF,
                ev_level=0.80,
                ev_vol=0.10,
                adv_residual_vol=1.00,
                gradient_cv_vol=0.10,
            ),
            d_val_acc=0.0,
            d_added_params=0.0,
        )

    report = score(
        [gradient_confounded(i) for i in range(10)],
        _THRESHOLDS,
        n=10,
        validity=Validity(True, ()),
        g3_hold=True,
        g4_hold=True,
    )
    assert report.leg_b_confound_downgraded is True
    assert report.leg_b is GateResult.INCONCLUSIVE


def test_score_does_not_downgrade_leg_b_when_var_returns_iqr_is_stable():
    report = score(
        [_accept_pair(i) for i in range(10)],
        _THRESHOLDS, n=10, validity=Validity(True, ()), g3_hold=True, g4_hold=True,
    )
    assert report.leg_b_confound_downgraded is False
    assert report.leg_b is GateResult.PASS
    assert report.verdict is Verdict.ACCEPT


# ---- S6: markdown verdict packet (render_packet) ----
#
# render_packet is a PURE function of the already-scored artifacts (Stage2Report +
# CalibrationReport + FrozenThresholds + Validity). It invents no telemetry — it faithfully
# surfaces the frozen scorer's verdict + covariates + provenance as a §-structured markdown packet,
# mirroring scripts/proof_packet.py. No I/O, no torch.


def _report(pairs_fn, n: int, *, g3_hold: bool = True, g4_hold: bool = True) -> Stage2Report:
    return score(
        [pairs_fn(i) for i in range(n)],
        _THRESHOLDS, n=n, validity=Validity(True, ()), g3_hold=g3_hold, g4_hold=g4_hold,
    )


def test_render_packet_headlines_verdict_and_powered_claim_tier():
    # §7: the verdict + its tier are the headline. n=10 is the powered-claim tier.
    packet = render_packet(_report(_accept_pair, 10), thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "ACCEPT" in packet
    assert "n=10" in packet


def test_render_packet_carries_stage0_provenance_block_verbatim():
    # §0/MAJOR-3: the provenance block must appear verbatim so Stage-0 variance evidence can never
    # be laundered into HRA acceptance evidence (the module says "do not paraphrase").
    packet = render_packet(_report(_accept_pair, 10), thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert STAGE0_PROVENANCE_BLOCK in packet


def _calibration() -> CalibrationReport:
    off_legs = [
        _legseries(Leg.OFF, ev_level=e, ev_vol=0.10, adv_residual_vol=1.0)
        for e in [0.80, 0.82, 0.79, 0.81, 0.80]
    ]
    return calibrate_off(off_legs)


def test_render_packet_invalid_lists_every_validity_reason_and_withholds_interpretation():
    # §1: an INVALID run lists EVERY breach and performs NO leg/guard interpretation.
    reasons = ("seed mismatch: ON seed=7 != OFF seed=8", "reward_mode mismatch: 'SHAPED' != 'SIMPLIFIED'")
    invalid = Validity(False, reasons)
    report = score(
        [_accept_pair(i) for i in range(10)], _THRESHOLDS, n=10,
        validity=invalid, g3_hold=True, g4_hold=True,
    )
    packet = render_packet(report, thresholds=_THRESHOLDS, validity=invalid)
    assert "INVALID" in packet
    for reason in reasons:
        assert reason in packet
    assert "no leg" in packet.lower() and "interpretation" in packet.lower()


def test_render_packet_reports_leg_a_result_and_frozen_delta():
    # §3 LEG-A: gate result + the frozen non-inferiority margin δ (0.05).
    packet = render_packet(_report(_accept_pair, 10), thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "LEG-A" in packet
    assert "0.0500" in packet  # δ = 0.05


def test_render_packet_reports_leg_b_result_and_eps_rel():
    # §4 LEG-B: gate result + the fixed relative-reduction floor ε_rel (0.10).
    packet = render_packet(_report(_accept_pair, 10), thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "LEG-B" in packet
    assert "0.10" in packet  # ε_rel


def test_render_packet_reports_mech_and_all_four_safety_guards():
    # §5 MECH + §6 G1/G2/G3/G4 — every guard is surfaced (none defaulted).
    packet = render_packet(_report(_accept_pair, 10), thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "MECH" in packet
    for guard in ("G1", "G2", "G3", "G4"):
        assert guard in packet


def test_render_packet_reports_g3_g4_safety_evidence():
    evidence = SafetyEvidence(
        g3_churn_on=(ChurnRates(germinate=1.0, prune=0.5, fossilize=0.25),),
        g3_churn_off=(ChurnRates(germinate=0.8, prune=0.4, fossilize=0.2),),
        g3_ratio_max=1.5,
        g4_counts_on=(
            GuardChannelCounts(
                governor_rollback=1,
                value_collapse=0,
                ratio_explosion=0,
                ratio_collapse=0,
                gradient_anomaly=0,
                gradient_pathology=0,
                numerical_instability=0,
                reward_hacking=0,
            ),
        ),
        g4_counts_off=(
            GuardChannelCounts(
                governor_rollback=0,
                value_collapse=0,
                ratio_explosion=0,
                ratio_collapse=0,
                gradient_anomaly=0,
                gradient_pathology=0,
                numerical_instability=0,
                reward_hacking=0,
            ),
        ),
        g4_ratio_max=2.0,
        g4_abs_floor=5,
    )
    report = score(
        [_accept_pair(i) for i in range(10)],
        _THRESHOLDS,
        n=10,
        validity=Validity(True, ()),
        g3_hold=True,
        g4_hold=True,
        safety_evidence=evidence,
    )
    packet = render_packet(report, thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "G3 materiality threshold: ratio_max = 1.5" in packet
    assert "g=1.000/p=0.500/f=0.250" in packet
    assert "G4 materiality thresholds: ratio_max = 2, abs_floor = 5" in packet
    assert "rollback=1" in packet
    assert "reward_hacking=0" in packet


def test_render_packet_reports_floored_distribution_and_per_stream_blindness_caveat():
    # §8B: the floored fractions + the per-stream-blindness caveat travel with the verdict.
    packet = render_packet(_report(_accept_pair, 10), thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "§8B" in packet or "floored" in packet.lower()
    assert "ev_return_variance" in packet


def test_render_packet_flags_observed_advantage_std_flooring_without_changing_verdict():
    def floored_advantage_pair(seed: int) -> SeedPair:
        return SeedPair(
            seed=seed,
            on=_legseries(
                Leg.ON,
                ev_level=0.85,
                adv_residual_vol=0.80,
                ev_main_vol=0.03,
                advantage_std_floored_fraction=0.5,
            ),
            off=_legseries(
                Leg.OFF,
                ev_level=0.80,
                ev_vol=0.10,
                adv_residual_vol=1.00,
                advantage_std_floored_fraction=0.0,
            ),
            d_val_acc=0.0,
            d_added_params=0.0,
        )

    report = _report(floored_advantage_pair, 10)
    assert report.verdict is Verdict.ACCEPT
    packet = render_packet(report, thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "advantage_std_floored" in packet
    assert "OBSERVED" in packet
    assert "0.500" in packet


def test_render_packet_states_confound_downgrade_when_leg_b_demoted():
    # §4 confound: a downgraded LEG-B must be visible in the packet, not silent.
    downgraded = _report(_confounded_pair, 10)
    assert downgraded.leg_b_confound_downgraded is True
    packet = render_packet(downgraded, thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "confound" in packet.lower()
    # and a non-downgraded run says so too (the field is always reported)
    clean = render_packet(_report(_accept_pair, 10), thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "confound" in clean.lower()


def test_render_packet_marks_diagnostic_scalars_unavailable():
    # §9: absent value_main/cf target scales are rendered as UNAVAILABLE; no fake defaults.
    packet = render_packet(_report(_accept_pair, 10), thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "§9" in packet
    assert "unavailable" in packet.lower()


def test_render_packet_reports_available_diagnostic_scalars():
    def with_scales(seed: int) -> SeedPair:
        pair = _accept_pair(seed)
        return SeedPair(
            seed=seed,
            on=_legseries(
                Leg.ON,
                ev_level=0.85,
                adv_residual_vol=0.80,
                ev_main_vol=0.03,
                value_main_target_scale_level=1.7,
                cf_value_target_scale_level=6.3,
            ),
            off=pair.off,
            d_val_acc=0.0,
            d_added_params=0.0,
        )

    report = score(
        [with_scales(i) for i in range(10)],
        _THRESHOLDS,
        n=10,
        validity=Validity(True, ()),
        g3_hold=True,
        g4_hold=True,
    )
    packet = render_packet(report, thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "value_main_target_scale" in packet
    assert "cf_value_target_scale" in packet
    assert "1.7000" in packet
    assert "6.3000" in packet


def test_render_packet_n5_screen_pass_states_never_banked():
    # §7/§10 tier: a clean n=5 is SCREEN_PASS and the packet must say it is NOT a banked ACCEPT.
    report = _report(_accept_pair, 5)
    assert report.verdict is Verdict.SCREEN_PASS
    packet = render_packet(report, thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "SCREEN_PASS" in packet
    assert "never" in packet.lower() and "bank" in packet.lower()


def test_render_packet_includes_calibration_provenance_when_provided():
    # When the OFF calibration is supplied, the packet records that δ was frozen from OFF (§10).
    packet = render_packet(
        _report(_accept_pair, 10), thresholds=_THRESHOLDS, validity=Validity(True, ()),
        calibration=_calibration(),
    )
    assert "OFF" in packet
    assert "frozen" in packet.lower() or "calibrat" in packet.lower()


def test_validate_pair_rejects_placement_mismatch():
    on_meta = dataclasses.replace(
        _meta(Leg.ON),
        placement=(("env_devices_json", '["cuda:0"]'), ("policy_device", "cuda:0")),
    )
    off_meta = dataclasses.replace(
        _meta(Leg.OFF),
        placement=(("env_devices_json", '["cuda:1"]'), ("policy_device", "cuda:1")),
    )
    v = validate_pair(on_meta, _clean_on_rows(), off_meta, _clean_off_rows(), w=0, budget=3)
    assert v.valid is False
    assert any("placement" in reason for reason in v.reasons)


# ---- §11 W-rule: ON cf_value_loss plateau validity check ----


def test_plateau_finds_first_stable_trailing_window():
    assert plateau([1.0] * 8) == 8
    halving = [2.0 ** -i for i in range(10)]  # 50% relative change every step
    assert plateau(halving) is None
    # Appended flat values equal halving[-1], so the first stable trailing-8 window ends at
    # update 17 (indices 9..16: the last halving value plus seven flat ones).
    assert plateau(halving + [halving[-1]] * 8) == 17


def test_plateau_zero_handling():
    assert plateau([0.0] * 8) == 8  # 0 -> 0 is zero change
    assert plateau([0.0] * 7 + [1.0]) is None  # 0 -> nonzero inside the window is unstable


def _on_rows_with_cf(cf: list[float], **common) -> list[UpdateRow]:
    return [
        _update(
            batch=i, explained_variance=0.8, ev_sum=0.8, ev_main=0.2, ev_cf=0.1,
            cf_value_loss=c, **common,
        )
        for i, c in enumerate(cf)
    ]


def test_validate_pair_does_not_gate_on_cf_value_loss_plateau():
    # §11.2 (owner-ruled 2026-07-10): the cf-plateau check is DESCRIPTIVE, not a gate —
    # real cf_value_loss is oscillatory (non-stationary cf target scale) and can never
    # satisfy the frozen plateau() at any W; an un-plateaued series must NOT invalidate.
    n = 20
    on = _on_rows_with_cf([2.0 ** -i for i in range(n)], ev_return_variance=5.0)
    off = _off_rows([0.8] * n, ev_return_variance=5.0)
    v = validate_pair(_meta(Leg.ON), on, _meta(Leg.OFF), off, w=8, budget=3)
    assert v.valid is True
    assert not any("plateau" in reason for reason in v.reasons)


def test_hra_signature_rejects_off_arm_carrying_cf_value_loss():
    # cf_value_loss is hra-gated exactly like ev_*: present on an OFF arm = leaked HRA
    # machinery, a bug not a null (§1/§9) — must fail the OFF telemetry signature.
    n = 12
    off = [
        _update(batch=i, explained_variance=0.8, ev_return_variance=5.0, cf_value_loss=0.02)
        for i in range(n)
    ]
    with pytest.raises(ValueError, match="hra-gated"):
        require_hra_signature(off, Leg.OFF, run_dir="/off/x")


def test_validate_pair_rejects_empty_placement():
    # placement provenance must be populated — an empty tuple would let the §10 per-pair
    # device-equality check pass vacuously.
    on_meta = dataclasses.replace(_meta(Leg.ON), placement=())
    off_meta = dataclasses.replace(_meta(Leg.OFF), placement=())
    v = validate_pair(on_meta, _clean_on_rows(12), off_meta, _clean_off_rows(12), w=8, budget=3)
    assert v.valid is False
    assert any("placement" in reason for reason in v.reasons)


def test_leg_series_computes_cf_warmup_descriptive_on_on_leg_only():
    # §11.2 descriptive block (report, do not gate): per ON arm — present/finite counts over
    # the scored window, plateau over the full series, loss and scale distribution stats.
    n = 12
    cf = [50.0, 10.0, 5.0, 1.0] + [0.5] * (n - 4)
    on = [
        _update(
            batch=i, explained_variance=0.8, ev_sum=0.8, ev_main=0.2, ev_cf=0.1,
            cf_value_loss=cf[i], value_main_target_scale=1.0 + i, cf_value_target_scale=2.0 + i,
            ev_return_variance=5.0,
        )
        for i in range(n)
    ]
    ls = leg_series(on, leg=Leg.ON, w=8)
    d = ls.cf_warmup
    assert d is not None
    assert d.n_scored == 4  # 12 rows - 8 burn-in, none floored
    assert d.present_scored == 4
    assert d.finite_scored == 4
    assert d.plateau_at == 12  # flat from index 4; first stable trailing-8 window ends at 12
    assert d.loss_max == pytest.approx(50.0)
    assert d.scale_max == pytest.approx(2.0 + (n - 1))
    assert d.scale_q1_median is not None and d.scale_q4_median is not None
    assert d.scale_q4_median > d.scale_q1_median  # growing scale => positive drift

    off = _off_rows([0.8] * n, ev_return_variance=5.0)
    assert leg_series(off, leg=Leg.OFF, w=8).cf_warmup is None


def test_render_packet_includes_cf_warmup_descriptive_section():
    pairs = [_accept_pair(i) for i in range(5)]
    report = score(
        pairs, _THRESHOLDS, n=5, validity=Validity(True, ()), g3_hold=True, g4_hold=True
    )
    text = render_packet(report, thresholds=_THRESHOLDS, validity=Validity(True, ()))
    assert "§11.2" in text
    assert "do not gate" in text


def test_validate_pair_rejects_on_missing_cf_value_loss():
    n = 20
    on = [
        _update(
            batch=i, explained_variance=0.8, ev_sum=0.8, ev_main=0.2, ev_cf=0.1,
            cf_value_loss=None, ev_return_variance=5.0,
        )
        for i in range(n)
    ]
    off = _off_rows([0.8] * n, ev_return_variance=5.0)
    v = validate_pair(_meta(Leg.ON), on, _meta(Leg.OFF), off, w=8, budget=3)
    assert v.valid is False
    assert any("cf_value_loss" in reason for reason in v.reasons)
