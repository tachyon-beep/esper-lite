"""Stage-2 HRA MAJOR-1 acceptance TELEMETRY WRAPPER.

The telemetry-reading half of the frozen gate in
docs/analysis/2026-07-06-stage2-hra-major1-acceptance-gate.md. The pure statistical predicates
live in ``stage2_acceptance`` (this module's sibling); the WRAPPER owns everything that reads
telemetry: §1 validity, §0 provenance, burn-in ``W`` discard, §8B floored-update exclusion, §8E
``Var(returns_total)`` covariate, G3/G4, the §9 diagnostic scalars, and the tier-gated call into
``composite_verdict``.

Layering (so the pure bulk stays synthetic-fixture-testable and touches no I/O):

- ``UpdateRow`` is a typed, frozen view of one ``ppo_updates`` row. The dict -> ``UpdateRow``
  conversion happens ONCE at the duckdb boundary (S5) and fails loud on a missing column — the
  repo forbids ``.get()`` masking, so the pure layer never touches raw dicts.
- The series primitives below are pure functions over ``UpdateRow`` lists / float lists.

These functions encode the pre-registered contract; do not weaken them to make a run pass.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from esper.simic.telemetry.stage2_acceptance import (
    GateResult,
    Verdict,
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


class Leg(enum.Enum):
    """Which arm of a fresh-init pair a run is (§2 pairing unit = seed)."""

    ON = "on"  # hra_value_decomposition=True — ev_main/ev_cf/ev_sum populated
    OFF = "off"  # hra_value_decomposition=False — total EV is explained_variance


# §0/§1 scope fence: the only actor-objective this gate accepts is A (single GAE on
# V_total = V_main + V_cf; per-stream returns feed the VALUE targets only). Objective B (actor
# advantage on R_main) is reserved for a later PDR with stricter behavioural gates, so a run whose
# actor advantage is not the total reconstruction is out of scope and must fail validity.
# TODO: [FUTURE FUNCTIONALITY] promote to leyline (torch-free home) in S7 when the ppo_agent/emitter
# emission of `actor_advantage_source` lands — the emitter becomes the second consumer, making this a
# genuine cross-domain contract per the leyline mandate.
ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED = "total_reconstructed"


# §0 / MAJOR-3 provenance block — emitted verbatim in every verdict so Stage-0 variance evidence
# can never be laundered into HRA acceptance evidence. Do not paraphrase.
STAGE0_PROVENANCE_BLOCK = """Stage-0 variance gate:
    metric:  Cov(R_cf, R_total) / Var(R_total)  (value-free, raw-scale, per-return)
    reading: median 1.017 (PDR-0032/0033), Stage-2-OFF control
    licenses: running the Stage-2 HRA A/B at all
    HRA evidence: NO — it does not test the HRA implementation

Stage-2 HRA acceptance (THIS gate):
    source:  fresh paired HRA-OFF / HRA-ON A/B (§10)
    metrics: EV_main, EV_sum, advantage-path volatility, safety (§6)"""


@dataclass(frozen=True)
class UpdateRow:
    """One ordered ``ppo_updates`` row, typed for the wrapper's pure layer (§2 data model).

    Per-stream EV columns (``ev_sum``/``ev_main``/``ev_cf``) are ``None`` on the OFF leg — they
    are emitted only under ``hra_value_decomposition=True`` (``ppo_agent.py:751``), which is
    exactly the telemetry signature §1 verifies the ON/OFF pairing against.
    """

    inner_epoch: int
    batch: int
    explained_variance: float
    ev_sum: float | None
    ev_main: float | None
    ev_cf: float | None
    ev_return_variance: float
    pre_norm_advantage_std: float
    return_std: float


def burn_in_discard(rows: list[UpdateRow], w: int) -> list[UpdateRow]:
    """Drop the first ``w`` updates (§8-W). ``w`` is the SAME absolute window on both legs.

    A negative window would tail-slice and silently corrupt the series — reject it.
    """
    if w < 0:
        raise ValueError(f"burn-in window w must be >= 0, got {w}")
    return list(rows[w:])


def floored_exclusion(
    rows: list[UpdateRow], floor: float = 1.0
) -> tuple[list[UpdateRow], float]:
    """§8B — drop updates whose ``ev_return_variance <= floor`` and report the floored fraction.

    ``EV = 1 - residual/max(Var(returns), floor)`` (``value_metrics.py:164``); at or below the
    floor the EV is a floor artifact, not a fit measurement, so those updates must not enter any
    level/vol statistic. ``floor`` MUST equal the run's frozen ``ev_return_variance_floor`` (§10:
    1.0). The floored fraction feeds the §8B arm-asymmetry reject and travels into the packet.

    An empty input is a §1 completeness failure surfaced upstream; refuse to compute a 0/0
    fraction here.
    """
    if not rows:
        raise ValueError("floored_exclusion requires a non-empty update series")
    kept = [row for row in rows if row.ev_return_variance > floor]
    floored_fraction = (len(rows) - len(kept)) / len(rows)
    return kept, floored_fraction


def sqrt_unexplained_series(ev_values: list[float]) -> list[float]:
    """§4 — the return-scale-invariant residual ``sqrt(1 - ev)`` per update.

    ``ev`` is per-update ``ev_sum`` (ON) or ``explained_variance`` (OFF). ``ev <= 1`` by
    construction, but floating-point can push ``1 - ev`` to a tiny negative; clamp at 0 so the
    root never NaNs. LEG-B's gated statistic is the IQR of this series (convention-immune, §8C).
    """
    if not ev_values:
        raise ValueError("sqrt_unexplained_series requires a non-empty ev series")
    return [math.sqrt(max(0.0, 1.0 - ev)) for ev in ev_values]


def var_returns_series(return_std_values: list[float]) -> list[float]:
    """§8E — ``Var(returns_total) = return_std**2`` per update, the LEG-B confound covariate."""
    if not return_std_values:
        raise ValueError("var_returns_series requires a non-empty return_std series")
    return [std * std for std in return_std_values]


def _require_present(name: str, value: float | None) -> float:
    """Fail loud if an ON-leg column is null on a scored update (a §1 bug, not a masked default)."""
    if value is None:
        raise ValueError(f"{name} is None on a scored ON-leg update (hra-gated column absent)")
    return value


@dataclass(frozen=True)
class LegSeries:
    """Per-leg (one run = one seed × one leg) reduction of the scored ``ppo_updates`` series.

    Built after burn-in (§8-W) and §8B floored-exclusion. Total-EV statistics read ``ev_sum`` on
    the ON leg and ``explained_variance`` on the OFF leg (§2 comparand convention). The
    ``ev_return_variance`` distribution is summarised over the POST-BURN-IN, PRE-FLOOR set so the
    §8B "does the floor bite?" caveat travels with the verdict even after the floored updates are
    excluded from the fit statistics.
    """

    leg: Leg
    n_updates_scored: int
    floored_fraction: float
    ev_level: float  # §3 LEG-A level (median total EV)
    ev_vol: float  # §3 calibrate fallback / §5 MECH OFF baseline (IQR total EV)
    adv_residual_vol: float  # §4 LEG-B gated statistic: IQR(sqrt(1 - ev))
    raw_adv_std_vol: float  # §4 descriptive covariate: IQR(pre_norm_advantage_std)
    var_returns_level: float  # §8E context: median Var(returns_total)
    var_returns_vol: float  # §8E confound: IQR(Var(returns_total))
    ev_main_vol: float | None  # §5 MECH: IQR(ev_main); ON only, None on OFF
    ev_return_variance_min: float  # §8B caveat distribution
    ev_return_variance_median: float
    ev_return_variance_max: float


def leg_series(
    rows: list[UpdateRow], *, leg: Leg, w: int, floor: float = 1.0
) -> LegSeries:
    """Reduce one leg's ordered ``ppo_updates`` rows to a ``LegSeries`` (§2/§3/§4/§5/§8B/§8E).

    Pipeline: burn-in discard (§8-W) -> §8B floored-exclusion -> level/vol statistics on the
    scored set, using the same median/IQR convention as the pure scorer (imported ``level``/
    ``vol``, correction-immune within a series, §8C).
    """
    post_burnin = burn_in_discard(rows, w)
    if not post_burnin:
        raise ValueError(f"no updates survive the burn-in window w={w} (a §1 completeness failure)")
    scored, floored_fraction = floored_exclusion(post_burnin, floor)
    if not scored:
        raise ValueError(
            "every post-burn-in update is floored (ev_return_variance <= floor); the fit is a "
            "floor artifact (§8B)"
        )

    ev_totals = (
        [_require_present("ev_sum", row.ev_sum) for row in scored]
        if leg is Leg.ON
        else [row.explained_variance for row in scored]
    )
    var_returns = var_returns_series([row.return_std for row in scored])
    erv = [row.ev_return_variance for row in post_burnin]

    ev_main_vol = (
        vol([_require_present("ev_main", row.ev_main) for row in scored])
        if leg is Leg.ON
        else None
    )

    return LegSeries(
        leg=leg,
        n_updates_scored=len(scored),
        floored_fraction=floored_fraction,
        ev_level=level(ev_totals),
        ev_vol=vol(ev_totals),
        adv_residual_vol=vol(sqrt_unexplained_series(ev_totals)),
        raw_adv_std_vol=vol([row.pre_norm_advantage_std for row in scored]),
        var_returns_level=level(var_returns),
        var_returns_vol=vol(var_returns),
        ev_main_vol=ev_main_vol,
        ev_return_variance_min=min(erv),
        ev_return_variance_median=level(erv),
        ev_return_variance_max=max(erv),
    )


@dataclass(frozen=True)
class RunMeta:
    """Run-level provenance/config for one leg (§1/§10 frozen fields).

    ``actor_advantage_source`` and ``seed``/``reward_mode`` come from the Karn ``runs`` view;
    ``uses_per_head_norm`` is derived from ``ppo_updates`` (§8F(i), read_run_uses_per_head_norm).
    """

    run_dir: str
    seed: int
    reward_mode: str
    actor_advantage_source: str
    uses_per_head_norm: bool


@dataclass(frozen=True)
class Validity:
    """§1 outcome for a pair. ``reasons`` lists EVERY breach found (not just the first), so the
    packet can report all problems; empty iff valid."""

    valid: bool
    reasons: tuple[str, ...]


def _nonfinite_fields(row: UpdateRow, leg: Leg) -> list[str]:
    """Names of the scored numeric fields on ``row`` that are NaN/Inf (§1 finiteness).

    Explicit field-by-field access (not ``getattr``) — the schema is fixed and every field is
    required, so this is a finiteness check, not defensive attribute-masking.
    """
    bad: list[str] = []
    if not math.isfinite(row.explained_variance):
        bad.append("explained_variance")
    if not math.isfinite(row.pre_norm_advantage_std):
        bad.append("pre_norm_advantage_std")
    if not math.isfinite(row.return_std):
        bad.append("return_std")
    if not math.isfinite(row.ev_return_variance):
        bad.append("ev_return_variance")
    if leg is Leg.ON:
        if row.ev_sum is not None and not math.isfinite(row.ev_sum):
            bad.append("ev_sum")
        if row.ev_main is not None and not math.isfinite(row.ev_main):
            bad.append("ev_main")
        if row.ev_cf is not None and not math.isfinite(row.ev_cf):
            bad.append("ev_cf")
    return bad


def _hra_signature_violation(rows: list[UpdateRow], leg: Leg) -> str | None:
    """§1 telemetry signature: ON must carry ev_sum/ev_main/ev_cf on every update; OFF must carry
    none of them (they are emitted only under hra_value_decomposition=True). This is how the
    wrapper VERIFIES the caller-supplied ON/OFF pairing instead of trusting a self-reported flag."""
    if leg is Leg.ON:
        if any(r.ev_sum is None or r.ev_main is None or r.ev_cf is None for r in rows):
            return (
                "ON arm missing hra-gated ev_* metric (ev_sum/ev_main/ev_cf) — telemetry "
                "signature does not match a hra_value_decomposition=True run"
            )
        return None
    if any(r.ev_sum is not None or r.ev_main is not None or r.ev_cf is not None for r in rows):
        return "OFF arm carries a hra-gated ev_* metric — a bug, not a null (§1)"
    return None


def _validate_leg(
    meta: RunMeta,
    rows: list[UpdateRow],
    leg: Leg,
    *,
    w: int,
    floor: float,
    budget: int,
) -> tuple[list[str], float | None]:
    """Per-arm §1 checks. Returns (reasons, floored_fraction | None-if-no-data)."""
    reasons: list[str] = []
    if meta.actor_advantage_source != ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED:
        reasons.append(
            f"{leg.value.upper()} arm actor_advantage_source="
            f"{meta.actor_advantage_source!r} != {ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED!r} "
            "(§0 scope-fence: objective-B run under an objective-A gate)"
        )

    signature = _hra_signature_violation(rows, leg)
    if signature is not None:
        reasons.append(signature)

    post_burnin = burn_in_discard(rows, w)
    if not post_burnin:
        reasons.append(
            f"{leg.value.upper()} arm incomplete: no updates survive the burn-in window w={w}"
        )
        return reasons, None

    nonfinite = sorted({name for row in post_burnin for name in _nonfinite_fields(row, leg)})
    if nonfinite:
        reasons.append(
            f"{leg.value.upper()} arm has non-finite (NaN/Inf) scored metric(s): "
            f"{', '.join(nonfinite)}"
        )
        # A non-finite ev_return_variance corrupts the floored-fraction; do not compute it.
        if "ev_return_variance" in nonfinite:
            return reasons, None

    kept, floored_fraction = floored_exclusion(post_burnin, floor)
    if len(kept) < budget:
        reasons.append(
            f"{leg.value.upper()} arm incomplete: {len(kept)} scored updates after burn-in + "
            f"floored-exclusion < pre-registered budget {budget}"
        )
    return reasons, floored_fraction


def validate_pair(
    on_meta: RunMeta,
    on_rows: list[UpdateRow],
    off_meta: RunMeta,
    off_rows: list[UpdateRow],
    *,
    w: int,
    budget: int,
    floor: float = 1.0,
    floored_asymmetry_max: float = 0.10,
) -> Validity:
    """§1 validity gates for one fresh-init pair — run BEFORE any EV/downstream interpretation.

    Accumulates every breach: scope-fence (actor_advantage_source), the ON/OFF telemetry
    signature, completeness vs ``budget`` after burn-in + floored-exclusion, finiteness, seed
    match, reward_mode match, and the §8B floored-fraction arm asymmetry. ``floored_asymmetry_max``
    is a frozen-candidate threshold (10 pp) — freeze it with the other §11 thresholds before ON.
    """
    reasons: list[str] = []

    if on_meta.seed != off_meta.seed:
        reasons.append(
            f"seed mismatch: ON seed={on_meta.seed} != OFF seed={off_meta.seed} "
            "(arms are not the same fresh-init seed, §1)"
        )
    if on_meta.reward_mode != off_meta.reward_mode:
        reasons.append(
            f"reward_mode mismatch: ON={on_meta.reward_mode!r} != OFF={off_meta.reward_mode!r} "
            "(the toggle must be decomposition-only, §10)"
        )
    if on_meta.uses_per_head_norm or off_meta.uses_per_head_norm:
        reasons.append(
            "per-head advantage normalization active (§8F(i) requires per_head_advantage_norm=False "
            f"on both legs): ON={on_meta.uses_per_head_norm}, OFF={off_meta.uses_per_head_norm} "
            "— it changes pre_norm_advantage_std semantics and contaminates the LEG-B comparison"
        )

    on_reasons, on_floored = _validate_leg(
        on_meta, on_rows, Leg.ON, w=w, floor=floor, budget=budget
    )
    off_reasons, off_floored = _validate_leg(
        off_meta, off_rows, Leg.OFF, w=w, floor=floor, budget=budget
    )
    reasons.extend(on_reasons)
    reasons.extend(off_reasons)

    if on_floored is not None and off_floored is not None:
        if abs(on_floored - off_floored) > floored_asymmetry_max:
            reasons.append(
                f"materially asymmetric floored fraction (§8B): ON={on_floored:.3f} vs "
                f"OFF={off_floored:.3f}, |Δ|={abs(on_floored - off_floored):.3f} > "
                f"{floored_asymmetry_max:.3f}"
            )

    return Validity(valid=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True)
class FrozenThresholds:
    """The §11 thresholds, frozen (from OFF calibration + owner) BEFORE any ON scoring."""

    delta: float  # §3 ev_sum non-inferiority margin (from calibrate_off)
    eps_rel: float  # §4 adv-vol relative reduction floor (fixed 0.10)
    tau_acc: float  # §6 G1 val-acc regression deadband (0.3 pp)
    delta_param_max: float  # §6 G2 added-param ceiling (owner-set)
    w: int  # §8-W burn-in updates discarded (owner-set from the plateau rule)


@dataclass(frozen=True)
class CalibrationReport:
    """Output of the OFF-only calibration pass (§10 freeze order step 2)."""

    delta: float
    eps_rel: float
    eps_rel_off_spread_anchor: float  # OFF seed-to-seed relative spread of adv-residual vol
    off_ev_levels: tuple[float, ...]
    off_ev_iqrs: tuple[float, ...]


@dataclass(frozen=True)
class SeedPair:
    """One fresh-init seed's ON/OFF reduced series plus its paired episode-level safety deltas.

    ``d_val_acc`` / ``d_added_params`` are ON-minus-OFF (§6 G1/G2); they come from the
    ``episode_outcomes`` reader, not from ``ppo_updates``, so the caller supplies them.
    """

    seed: int
    on: LegSeries
    off: LegSeries
    d_val_acc: float
    d_added_params: float


@dataclass(frozen=True)
class Stage2Report:
    """The scored Stage-2 verdict (§7) plus the covariates + provenance the packet renders.

    The leg / guard fields are ``None`` exactly when the run is INVALID — no interpretation is
    performed on a broken run (§1).
    """

    verdict: Verdict
    n: int
    leg_a: GateResult | None
    leg_b: GateResult | None
    mech_hold: bool | None
    g1: GateResult | None
    g2_hold: bool | None
    g3_hold: bool
    g4_hold: bool
    delta_a: tuple[float, ...]
    delta_b: tuple[float, ...]
    on_floored_fractions: tuple[float, ...]
    off_floored_fractions: tuple[float, ...]
    # §4 reported covariates (per seed): the IQR(Var(returns_total)) confound + raw advantage-std IQR.
    var_returns_vol_on: tuple[float, ...]
    var_returns_vol_off: tuple[float, ...]
    raw_adv_std_vol_on: tuple[float, ...]
    raw_adv_std_vol_off: tuple[float, ...]
    # §4 confound downgrade: LEG-B PASS demoted to INCONCLUSIVE when the ON adv-residual-vol reduction
    # is matched by a proportional IQR(Var(returns_total)) drop (a return-regime artifact, not shielding).
    leg_b_confound_downgraded: bool
    provenance: str
    diagnostic_scalars_available: bool


def calibrate_off(
    off_legs: list[LegSeries], *, eps_rel: float = 0.10, floor_margin: float = 0.05
) -> CalibrationReport:
    """§10 step 2 — freeze δ (and echo ε_rel) from the OFF arms ONLY, before any ON scoring.

    Reading an ON leg here is a freeze-order breach ("no peeking") — reject it. δ is the scorer's
    ``calibrate_delta`` over the OFF ev levels + IQRs; ε_rel is the fixed 0.10 floor, reported
    alongside the OFF seed-to-seed relative spread of the adv-residual volatility as its
    sanity anchor (§4: ε_rel should exceed that spread).
    """
    if not off_legs:
        raise ValueError("calibrate_off requires at least one OFF leg")
    for series in off_legs:
        if series.leg is not Leg.OFF:
            raise ValueError(
                "calibrate_off reads OFF legs only (§10 freeze order — no peeking at ON)"
            )
    off_ev_levels = [series.ev_level for series in off_legs]
    off_ev_iqrs = [series.ev_vol for series in off_legs]
    delta = calibrate_delta(off_ev_levels, off_ev_iqrs, floor=floor_margin)
    adv = [series.adv_residual_vol for series in off_legs]
    med = level(adv)
    anchor = (max(adv) - min(adv)) / med if med > 0 else float("inf")
    return CalibrationReport(
        delta=delta,
        eps_rel=eps_rel,
        eps_rel_off_spread_anchor=anchor,
        off_ev_levels=tuple(off_ev_levels),
        off_ev_iqrs=tuple(off_ev_iqrs),
    )


def _relative_reduction(on_value: float, off_value: float) -> float:
    """§4 Δ_B: (ON − OFF) / OFF adv-residual volatility. A zero OFF volatility is a degenerate,
    constant-EV leg — fail loud rather than divide by zero."""
    if off_value == 0.0:
        raise ValueError("OFF adv-residual volatility is zero — degenerate leg (constant EV series)")
    return (on_value - off_value) / off_value


def score(
    pairs: list[SeedPair],
    thresholds: FrozenThresholds,
    *,
    n: int,
    validity: Validity,
    g3_hold: bool,
    g4_hold: bool,
) -> Stage2Report:
    """§10 step 4 — score the ON arms against the FROZEN thresholds and emit the verdict (§7).

    INVALID short-circuits with no leg interpretation (§1). Otherwise the LegSeries pairs are
    assembled into the scorer's per-seed arrays and run through the tier-aware ``composite_verdict``
    (ACCEPT banked only at n=10). The §0 provenance block always travels with the verdict; the §9
    diagnostic scalars are marked unavailable until the emission slice lands (typed, not defaulted).
    """
    if not validity.valid:
        return Stage2Report(
            verdict=Verdict.INVALID,
            n=n,
            leg_a=None,
            leg_b=None,
            mech_hold=None,
            g1=None,
            g2_hold=None,
            g3_hold=g3_hold,
            g4_hold=g4_hold,
            delta_a=(),
            delta_b=(),
            on_floored_fractions=(),
            off_floored_fractions=(),
            var_returns_vol_on=(),
            var_returns_vol_off=(),
            raw_adv_std_vol_on=(),
            raw_adv_std_vol_off=(),
            leg_b_confound_downgraded=False,
            provenance=STAGE0_PROVENANCE_BLOCK,
            diagnostic_scalars_available=False,
        )
    if len(pairs) != n:
        raise ValueError(f"n={n} but received {len(pairs)} pairs")

    delta_a = [pair.on.ev_level - pair.off.ev_level for pair in pairs]
    delta_b = [
        _relative_reduction(pair.on.adv_residual_vol, pair.off.adv_residual_vol) for pair in pairs
    ]
    ev_main_vol_on = [_require_present("ev_main_vol", pair.on.ev_main_vol) for pair in pairs]
    expl_vol_off = [pair.off.ev_vol for pair in pairs]

    leg_a_result = leg_a(delta_a, thresholds.delta, n)
    leg_b_result = leg_b(delta_b, thresholds.eps_rel, n)
    mech_hold = mech_guard(ev_main_vol_on, expl_vol_off)
    g1 = safety_g1([pair.d_val_acc for pair in pairs], thresholds.tau_acc)
    g2_hold = safety_g2([pair.d_added_params for pair in pairs], thresholds.delta_param_max)

    # §4 confound: if IQR(Var(returns_total)) drops at least as much (relatively) as the gated
    # residual, a LEG-B PASS is not distinguishable from a return-regime artifact -> INCONCLUSIVE.
    delta_var = [
        _relative_reduction(pair.on.var_returns_vol, pair.off.var_returns_vol) for pair in pairs
    ]
    leg_b_confound_downgraded = False
    if leg_b_result is GateResult.PASS and level(delta_var) <= level(delta_b):
        leg_b_result = GateResult.INCONCLUSIVE
        leg_b_confound_downgraded = True

    verdict = composite_verdict(
        valid=True,
        n=n,
        leg_a_result=leg_a_result,
        leg_b_result=leg_b_result,
        mech_hold=mech_hold,
        g1_result=g1,
        g2_hold=g2_hold,
        g3_hold=g3_hold,
        g4_hold=g4_hold,
    )
    return Stage2Report(
        verdict=verdict,
        n=n,
        leg_a=leg_a_result,
        leg_b=leg_b_result,
        mech_hold=mech_hold,
        g1=g1,
        g2_hold=g2_hold,
        g3_hold=g3_hold,
        g4_hold=g4_hold,
        delta_a=tuple(delta_a),
        delta_b=tuple(delta_b),
        on_floored_fractions=tuple(pair.on.floored_fraction for pair in pairs),
        off_floored_fractions=tuple(pair.off.floored_fraction for pair in pairs),
        var_returns_vol_on=tuple(pair.on.var_returns_vol for pair in pairs),
        var_returns_vol_off=tuple(pair.off.var_returns_vol for pair in pairs),
        raw_adv_std_vol_on=tuple(pair.on.raw_adv_std_vol for pair in pairs),
        raw_adv_std_vol_off=tuple(pair.off.raw_adv_std_vol for pair in pairs),
        leg_b_confound_downgraded=leg_b_confound_downgraded,
        provenance=STAGE0_PROVENANCE_BLOCK,
        diagnostic_scalars_available=False,
    )
