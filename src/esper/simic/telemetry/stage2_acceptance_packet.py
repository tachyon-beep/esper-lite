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
from collections.abc import Callable
from dataclasses import dataclass

from esper.leyline.telemetry import ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED
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


FrozenConfigValue = str | int | float | bool | None


class Leg(enum.Enum):
    """Which arm of a fresh-init pair a run is (§2 pairing unit = seed)."""

    ON = "on"  # hra_value_decomposition=True — ev_main/ev_cf/ev_sum populated
    OFF = "off"  # hra_value_decomposition=False — total EV is explained_variance


# §0/§1 scope fence: the only actor-objective this gate accepts is A (single GAE on
# V_total = V_main + V_cf; per-stream returns feed the VALUE targets only). Objective B (actor
# advantage on R_main) is reserved for a later PDR with stricter behavioural gates, so a run whose
# actor advantage is not the total reconstruction is out of scope and must fail validity.
# The constant lives in leyline (ACTOR_ADVANTAGE_SOURCE_TOTAL_RECONSTRUCTED, imported above):
# the TRAINING_STARTED emitter is its second consumer, making it a cross-domain contract (S7).


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
    value_main_target_scale: float | None
    cf_value_target_scale: float | None
    # ON-only like the ev_* columns. §1 signature requires PRESENCE on every ON update and
    # absence on OFF; finiteness is checked on scored updates. The §11 plateau over this
    # series is DESCRIPTIVE ONLY (§11.2 owner ruling, 2026-07-10) — it must not gate.
    cf_value_loss: float | None
    ev_return_variance: float
    pre_norm_advantage_std: float
    return_std: float
    gradient_cv: float
    advantage_std_floored: bool


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
    material_overshoot = [ev for ev in ev_values if ev > 1.0 + 1e-9]
    if material_overshoot:
        raise ValueError(
            "ev must be <= 1.0 except for tiny floating-point overshoot; "
            f"got {material_overshoot[0]}"
        )
    return [math.sqrt(max(0.0, 1.0 - ev)) for ev in ev_values]


def var_returns_series(return_std_values: list[float]) -> list[float]:
    """§8E — ``Var(returns_total) = return_std**2`` per update, the LEG-B confound covariate."""
    if not return_std_values:
        raise ValueError("var_returns_series requires a non-empty return_std series")
    return [std * std for std in return_std_values]


def plateau(series: list[float], *, window: int = 8, rel_tol: float = 0.05) -> int | None:
    """§11 plateau rule: first update ``u`` (1-based) whose trailing ``window`` updates have
    max consecutive relative change < ``rel_tol``; ``None`` if the series never stabilizes.

    A zero-to-nonzero step inside the window is unstable (relative change from zero is
    unbounded); zero-to-zero is zero change. A non-finite value never certifies a window.
    """
    if window < 2:
        raise ValueError("plateau window must span at least 2 updates")
    for end in range(window, len(series) + 1):
        chunk = series[end - window : end]
        stable = True
        for prev, cur in zip(chunk, chunk[1:], strict=False):
            if not (math.isfinite(prev) and math.isfinite(cur)):
                stable = False
                break
            if prev == 0.0:
                if cur != 0.0:
                    stable = False
                    break
                continue
            if abs(cur - prev) / abs(prev) >= rel_tol:
                stable = False
                break
        if stable:
            return end
    return None


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
    advantage_std_floored_fraction: float  # §8F contamination flag: share of scored updates floored
    var_returns_level: float  # §8E context: median Var(returns_total)
    var_returns_vol: float  # §8E confound: IQR(Var(returns_total))
    gradient_cv_vol: float  # §4 corroboration: IQR(gradient_cv)
    ev_main_vol: float | None  # §5 MECH: IQR(ev_main); ON only, None on OFF
    value_main_target_scale_level: float | None  # §9 diagnostic scale level; ON only when emitted
    cf_value_target_scale_level: float | None  # §9 diagnostic scale level; ON only when emitted
    ev_return_variance_min: float  # §8B caveat distribution
    ev_return_variance_median: float
    ev_return_variance_max: float
    # §11.2 cf-warmup descriptive block; ON only, None on OFF. Report, do not gate.
    cf_warmup: CfWarmupDescriptive | None = None


@dataclass(frozen=True)
class CfWarmupDescriptive:
    """§11.2 cf-warmup DESCRIPTIVE block for one ON arm (report, do not gate).

    Presence/finiteness counts run over the SCORED window (mirroring the §1 hard gates);
    the plateau index and distribution stats run over the FULL series — warmup lives before
    burn-in, which is exactly what this block exists to make visible. The plateau result
    MUST NOT feed any gate (§11.2 owner ruling, 2026-07-10).
    """

    n_scored: int
    present_scored: int
    finite_scored: int
    plateau_at: int | None  # frozen §11 plateau() over the full cf_value_loss series
    loss_p50: float
    loss_p90: float
    loss_max: float
    scale_p50: float | None  # cf_value_target_scale stats; None when the §9 scales are absent
    scale_p90: float | None
    scale_max: float | None
    scale_q1_median: float | None  # first/last quartile medians of the full series (drift read)
    scale_q4_median: float | None


def _pctl(sorted_values: list[float], frac: float) -> float:
    """Nearest-rank percentile over a pre-sorted, non-empty list (descriptive use only)."""
    return sorted_values[max(0, math.ceil(frac * len(sorted_values)) - 1)]


def _cf_warmup_descriptive(rows: list[UpdateRow], scored: list[UpdateRow]) -> CfWarmupDescriptive:
    """Reduce one ON arm's cf streams to the §11.2 descriptive block.

    Assumes the §1 signature gate runs alongside: a missing cf_value_loss or a partially
    emitted scale stream is a validity breach, so this reducer fails loud rather than
    describing an arm the gate must reject.
    """
    full = [row.cf_value_loss for row in rows]
    n_missing = sum(1 for value in full if value is None)
    if n_missing:
        raise ValueError(
            f"cf_value_loss missing on {n_missing} ON update(s) — the §1 signature gate "
            "rejects this arm; §11.2 descriptive reduction is undefined on it"
        )
    losses = [value for value in full if value is not None]
    finite_losses = sorted(value for value in losses if math.isfinite(value))
    if not finite_losses:
        raise ValueError(
            "cf_value_loss has no finite values — §1 finiteness rejects this arm before "
            "§11.2 descriptive reduction"
        )

    scales = [row.cf_value_target_scale for row in rows]
    n_scale_missing = sum(1 for value in scales if value is None)
    if n_scale_missing not in (0, len(scales)):
        raise ValueError(
            "cf_value_target_scale is partially emitted — §9 requires all-present or "
            "all-absent; the signature gate rejects this arm"
        )
    # §1 finiteness covers only the SCORED window; the full series may carry transient
    # non-finite values inside burn-in — exactly the warmup this block describes. Filter
    # to the finite (chronological) subset rather than crash on a pair validity accepted.
    finite_scales = [
        value for value in scales if value is not None and math.isfinite(value)
    ]
    if n_scale_missing == 0 and finite_scales:
        sorted_scales = sorted(finite_scales)
        quartile = max(1, len(finite_scales) // 4)
        scale_p50: float | None = _pctl(sorted_scales, 0.5)
        scale_p90: float | None = _pctl(sorted_scales, 0.9)
        scale_max: float | None = sorted_scales[-1]
        scale_q1_median: float | None = level(finite_scales[:quartile])
        scale_q4_median: float | None = level(finite_scales[-quartile:])
    else:
        scale_p50 = scale_p90 = scale_max = scale_q1_median = scale_q4_median = None

    return CfWarmupDescriptive(
        n_scored=len(scored),
        present_scored=sum(1 for row in scored if row.cf_value_loss is not None),
        finite_scored=sum(
            1
            for row in scored
            if row.cf_value_loss is not None and math.isfinite(row.cf_value_loss)
        ),
        plateau_at=plateau(losses),
        loss_p50=_pctl(finite_losses, 0.5),
        loss_p90=_pctl(finite_losses, 0.9),
        loss_max=finite_losses[-1],
        scale_p50=scale_p50,
        scale_p90=scale_p90,
        scale_max=scale_max,
        scale_q1_median=scale_q1_median,
        scale_q4_median=scale_q4_median,
    )


# §9 ON-only diagnostic scalars: the accessor is selected by an explicit mapping so an unknown
# field name is an immediate KeyError, never a silent read of the wrong column (fail-loud, §0).
_ON_DIAGNOSTIC_ACCESSORS: dict[str, Callable[[UpdateRow], float | None]] = {
    "value_main_target_scale": lambda row: row.value_main_target_scale,
    "cf_value_target_scale": lambda row: row.cf_value_target_scale,
}


def _optional_level_for_on(rows: list[UpdateRow], field_name: str) -> float | None:
    """Return the median when an ON-only diagnostic is fully emitted; fail on partial streams."""
    accessor = _ON_DIAGNOSTIC_ACCESSORS[field_name]
    values: list[float] = []
    missing = 0
    for row in rows:
        value = accessor(row)
        if value is None:
            missing += 1
        else:
            values.append(value)
    if missing == len(rows):
        return None
    if missing:
        raise ValueError(
            f"{field_name} is partially emitted on scored ON-leg updates "
            f"({len(values)}/{len(rows)} present); §9 diagnostics must be all-present or all-absent"
        )
    return level(values)


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
        advantage_std_floored_fraction=(
            sum(1 for row in scored if row.advantage_std_floored) / len(scored)
        ),
        var_returns_level=level(var_returns),
        var_returns_vol=vol(var_returns),
        ev_main_vol=ev_main_vol,
        gradient_cv_vol=vol([row.gradient_cv for row in scored]),
        value_main_target_scale_level=(
            _optional_level_for_on(scored, "value_main_target_scale") if leg is Leg.ON else None
        ),
        cf_value_target_scale_level=(
            _optional_level_for_on(scored, "cf_value_target_scale") if leg is Leg.ON else None
        ),
        ev_return_variance_min=min(erv),
        ev_return_variance_median=level(erv),
        ev_return_variance_max=max(erv),
        cf_warmup=_cf_warmup_descriptive(rows, scored) if leg is Leg.ON else None,
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
    frozen_config: tuple[tuple[str, FrozenConfigValue], ...] = ()
    # Device placement (policy_device / env_devices_json): §10 freezes the training
    # configuration, not run placement — seeds may legitimately spread across GPUs
    # (2026-07-09 owner ruling). Cross-seed homogeneity checks ignore this field;
    # validate_pair enforces ON == OFF placement per pair.
    placement: tuple[tuple[str, FrozenConfigValue], ...] = ()


@dataclass(frozen=True)
class Validity:
    """§1 outcome for a pair. ``reasons`` lists EVERY breach found (not just the first), so the
    packet can report all problems; empty iff valid."""

    valid: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ChurnRates:
    """Per-episode germinate/prune/fossilize means for one run (§6 G3 input).

    Defined here (pure, torch-free) and produced by the io reader — same layering as
    ``UpdateRow``."""

    germinate: float
    prune: float
    fossilize: float


@dataclass(frozen=True)
class GuardChannelCounts:
    """§6 G4 — per-channel anomaly counts for one run, from the Karn ``anomalies`` view.

    The channels are the gate doc's guard set (governor rollbacks, value-collapse,
    gradient-anomaly/pathology, ratio explosion/collapse, numerical instability).
    PLATEAU_DETECTED is deliberately NOT a channel — it is a benign progress signal.
    """

    governor_rollback: int
    value_collapse: int
    ratio_explosion: int
    ratio_collapse: int
    gradient_anomaly: int
    gradient_pathology: int
    numerical_instability: int
    reward_hacking: int = 0

    def total(self) -> int:
        return (
            self.governor_rollback
            + self.value_collapse
            + self.ratio_explosion
            + self.ratio_collapse
            + self.gradient_anomaly
            + self.gradient_pathology
            + self.numerical_instability
            + self.reward_hacking
        )


@dataclass(frozen=True)
class SafetyEvidence:
    """Auditable §6 G3/G4 inputs rendered with the verdict packet.

    The booleans decide the gate; the evidence lets a reader see the actual churn rates,
    guard-channel counts, and frozen materiality thresholds that produced those booleans.
    """

    g3_churn_on: tuple[ChurnRates, ...]
    g3_churn_off: tuple[ChurnRates, ...]
    g3_ratio_max: float
    g4_counts_on: tuple[GuardChannelCounts, ...]
    g4_counts_off: tuple[GuardChannelCounts, ...]
    g4_ratio_max: float
    g4_abs_floor: int


def _materially_elevated(on: int, off: int, *, ratio_max: float, abs_floor: int) -> bool:
    """§6 "materially elevated": BOTH a ratio breach AND an absolute breach.

    The conjunction keeps small-count noise from tripping the gate (1 vs 0 is a 'ratio
    breach' but noise) while a zero OFF baseline still trips on a real absolute elevation.
    DEFINITIONAL-PENDING: ratio_max / abs_floor have NO defaults — they must be frozen in
    the gate doc (§11) before ON scoring; the shape itself is flagged for drl review.
    """
    return (on - off) > abs_floor and on > ratio_max * off


def g4_hold_from_counts(
    on: GuardChannelCounts, off: GuardChannelCounts, *, ratio_max: float, abs_floor: int
) -> bool:
    """§6 G4 — guard channels not ON-elevated: holds iff NO channel is materially elevated.

    Per PDR-0026, governor-rollback asymmetry is an RCA trigger rather than a lone hard
    gate — but a MATERIAL elevation (both thresholds breached) still fails the hold here;
    the packet carries the per-channel counts so the RCA can follow.
    """
    pairs = (
        (on.governor_rollback, off.governor_rollback),
        (on.value_collapse, off.value_collapse),
        (on.ratio_explosion, off.ratio_explosion),
        (on.ratio_collapse, off.ratio_collapse),
        (on.gradient_anomaly, off.gradient_anomaly),
        (on.gradient_pathology, off.gradient_pathology),
        (on.numerical_instability, off.numerical_instability),
        (on.reward_hacking, off.reward_hacking),
    )
    if on.reward_hacking > 0:
        return False
    return not any(
        _materially_elevated(on_count, off_count, ratio_max=ratio_max, abs_floor=abs_floor)
        for on_count, off_count in pairs
    )


def g3_hold_from_churn(on: ChurnRates, off: ChurnRates, *, ratio_max: float) -> bool:
    """§6 G3 — churn not reward-farmed: ON germinate/prune per-episode rates must not be
    materially elevated over OFF (rate_on <= ratio_max * rate_off, per channel).
    Fossilize is included because the gate doc's churn guard names germinate/prune/fossilize,
    and fossilization-heavy reward gaming can otherwise pass G3 while changing lifecycle pressure.
    DEFINITIONAL-PENDING: ratio_max has no default — freeze it in the gate doc (§11) before ON
    scoring; shape flagged for drl review.
    """
    return (
        on.germinate <= ratio_max * off.germinate
        and on.prune <= ratio_max * off.prune
        and on.fossilize <= ratio_max * off.fossilize
    )


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
    if not math.isfinite(row.gradient_cv):
        bad.append("gradient_cv")
    if not math.isfinite(row.ev_return_variance):
        bad.append("ev_return_variance")
    if leg is Leg.ON:
        if row.ev_sum is not None and not math.isfinite(row.ev_sum):
            bad.append("ev_sum")
        if row.ev_main is not None and not math.isfinite(row.ev_main):
            bad.append("ev_main")
        if row.ev_cf is not None and not math.isfinite(row.ev_cf):
            bad.append("ev_cf")
        if row.value_main_target_scale is not None and not math.isfinite(row.value_main_target_scale):
            bad.append("value_main_target_scale")
        if row.cf_value_target_scale is not None and not math.isfinite(row.cf_value_target_scale):
            bad.append("cf_value_target_scale")
        if row.cf_value_loss is not None and not math.isfinite(row.cf_value_loss):
            bad.append("cf_value_loss")
    return bad


def _hra_signature_violation(rows: list[UpdateRow], leg: Leg) -> str | None:
    """§1 telemetry signature: ON must carry ev_sum/ev_main/ev_cf on every update; OFF must carry
    none of them (they are emitted only under hra_value_decomposition=True). This is how the
    wrapper VERIFIES the caller-supplied ON/OFF pairing instead of trusting a self-reported flag."""
    if leg is Leg.ON:
        if any(
            r.ev_sum is None or r.ev_main is None or r.ev_cf is None or r.cf_value_loss is None
            for r in rows
        ):
            return (
                "ON arm missing hra-gated metric (ev_sum/ev_main/ev_cf/cf_value_loss) — "
                "telemetry signature does not match a hra_value_decomposition=True run"
            )
        return None
    if any(
        r.ev_sum is not None
        or r.ev_main is not None
        or r.ev_cf is not None
        or r.value_main_target_scale is not None
        or r.cf_value_target_scale is not None
        or r.cf_value_loss is not None
        for r in rows
    ):
        return "OFF arm carries a hra-gated HRA metric — a bug, not a null (§1/§9)"
    return None


def require_hra_signature(rows: list[UpdateRow], leg: Leg, *, run_dir: str) -> None:
    """Fail closed when a caller-supplied leg label disagrees with the telemetry signature."""
    violation = _hra_signature_violation(rows, leg)
    if violation is not None:
        raise ValueError(f"{run_dir!r}: {violation}")
    if leg is Leg.ON:
        scale_pairs = [(row.value_main_target_scale, row.cf_value_target_scale) for row in rows]
        present = [a is not None and b is not None for a, b in scale_pairs]
        absent = [a is None and b is None for a, b in scale_pairs]
        if not (all(present) or all(absent)):
            raise ValueError(
                f"{run_dir!r}: §9 target-scale diagnostics are partially emitted; "
                "value_main_target_scale and cf_value_target_scale must be both present on every "
                "ON update, or both absent on every ON update"
            )
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

    try:
        require_hra_signature(rows, leg, run_dir=meta.run_dir)
    except ValueError as exc:
        reasons.append(str(exc))

    # cf_value_loss presence lives in _hra_signature_violation (ON must carry it, OFF must
    # not); the §11 cf-plateau VALIDITY GATE is demoted to descriptive reporting (§11.2,
    # owner-ruled 2026-07-10) — see CfWarmupDescriptive; it must not gate here.
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
    if on_meta.frozen_config != off_meta.frozen_config:
        reasons.append(
            "frozen run config mismatch: only hra_value_decomposition may differ (§10); "
            f"ON={on_meta.frozen_config!r} OFF={off_meta.frozen_config!r}"
        )
    if not on_meta.placement or not off_meta.placement:
        empty_arm = "ON" if not on_meta.placement else "OFF"
        reasons.append(
            f"placement provenance missing (empty) on the {empty_arm} RunMeta — the §10 "
            "per-pair device-equality check must not pass vacuously"
        )
    elif on_meta.placement != off_meta.placement:
        reasons.append(
            "placement mismatch: each ON arm must run on the same device as its OFF "
            f"partner (§10 pairing); ON={on_meta.placement!r} OFF={off_meta.placement!r}"
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
    safety_evidence: SafetyEvidence | None
    delta_a: tuple[float, ...]
    delta_b: tuple[float, ...]
    on_floored_fractions: tuple[float, ...]
    off_floored_fractions: tuple[float, ...]
    # §4 reported covariates (per seed): the IQR(Var(returns_total)) confound + raw advantage-std IQR.
    var_returns_vol_on: tuple[float, ...]
    var_returns_vol_off: tuple[float, ...]
    raw_adv_std_vol_on: tuple[float, ...]
    raw_adv_std_vol_off: tuple[float, ...]
    advantage_std_floored_fractions_on: tuple[float, ...]
    advantage_std_floored_fractions_off: tuple[float, ...]
    gradient_cv_vol_on: tuple[float, ...]
    gradient_cv_vol_off: tuple[float, ...]
    value_main_target_scale_levels: tuple[float, ...]
    cf_value_target_scale_levels: tuple[float, ...]
    # §4 confound downgrade: LEG-B PASS demoted to INCONCLUSIVE when the ON adv-residual-vol reduction
    # is matched by a proportional IQR(Var(returns_total)) drop (a return-regime artifact, not shielding).
    leg_b_confound_downgraded: bool
    provenance: str
    diagnostic_scalars_available: bool
    # §11.2 per-ON-seed cf-warmup descriptive blocks (seed, block); report, do not gate.
    cf_warmup_descriptives: tuple[tuple[int, CfWarmupDescriptive | None], ...] = ()


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


def _relative_reduction(
    on_value: float, off_value: float, *, denominator_label: str
) -> float:
    """Relative change ``(ON − OFF) / OFF``; fail loud on a degenerate OFF denominator."""
    if off_value == 0.0:
        raise ValueError(f"{denominator_label} is zero — degenerate leg")
    return (on_value - off_value) / off_value


def score(
    pairs: list[SeedPair],
    thresholds: FrozenThresholds,
    *,
    n: int,
    validity: Validity,
    g3_hold: bool,
    g4_hold: bool,
    safety_evidence: SafetyEvidence | None = None,
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
            safety_evidence=safety_evidence,
            delta_a=(),
            delta_b=(),
            on_floored_fractions=(),
            off_floored_fractions=(),
            var_returns_vol_on=(),
            var_returns_vol_off=(),
            raw_adv_std_vol_on=(),
            raw_adv_std_vol_off=(),
            advantage_std_floored_fractions_on=(),
            advantage_std_floored_fractions_off=(),
            gradient_cv_vol_on=(),
            gradient_cv_vol_off=(),
            value_main_target_scale_levels=(),
            cf_value_target_scale_levels=(),
            leg_b_confound_downgraded=False,
            provenance=STAGE0_PROVENANCE_BLOCK,
            diagnostic_scalars_available=False,
        )
    if len(pairs) != n:
        raise ValueError(f"n={n} but received {len(pairs)} pairs")
    seeds = [pair.seed for pair in pairs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seed(s) in Stage-2 pairset: {seeds!r}")

    delta_a = [pair.on.ev_level - pair.off.ev_level for pair in pairs]
    delta_b = [
        _relative_reduction(
            pair.on.adv_residual_vol,
            pair.off.adv_residual_vol,
            denominator_label="OFF adv-residual volatility",
        )
        for pair in pairs
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
        _relative_reduction(
            pair.on.var_returns_vol,
            pair.off.var_returns_vol,
            denominator_label="OFF return-variance volatility",
        )
        for pair in pairs
    ]
    gradient_cv_vol_on = [pair.on.gradient_cv_vol for pair in pairs]
    gradient_cv_vol_off = [pair.off.gradient_cv_vol for pair in pairs]
    delta_gradient_cv = [on - off for on, off in zip(gradient_cv_vol_on, gradient_cv_vol_off, strict=True)]
    leg_b_confound_downgraded = False
    if leg_b_result is GateResult.PASS and level(delta_var) <= level(delta_b):
        leg_b_result = GateResult.INCONCLUSIVE
        leg_b_confound_downgraded = True
    if leg_b_result is GateResult.PASS and level(delta_gradient_cv) > 0.0:
        leg_b_result = GateResult.INCONCLUSIVE
        leg_b_confound_downgraded = True

    value_main_scales = tuple(
        pair.on.value_main_target_scale_level
        for pair in pairs
        if pair.on.value_main_target_scale_level is not None
    )
    cf_value_scales = tuple(
        pair.on.cf_value_target_scale_level
        for pair in pairs
        if pair.on.cf_value_target_scale_level is not None
    )
    diagnostic_scalars_available = len(value_main_scales) == n and len(cf_value_scales) == n

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
        safety_evidence=safety_evidence,
        delta_a=tuple(delta_a),
        delta_b=tuple(delta_b),
        on_floored_fractions=tuple(pair.on.floored_fraction for pair in pairs),
        off_floored_fractions=tuple(pair.off.floored_fraction for pair in pairs),
        var_returns_vol_on=tuple(pair.on.var_returns_vol for pair in pairs),
        var_returns_vol_off=tuple(pair.off.var_returns_vol for pair in pairs),
        raw_adv_std_vol_on=tuple(pair.on.raw_adv_std_vol for pair in pairs),
        raw_adv_std_vol_off=tuple(pair.off.raw_adv_std_vol for pair in pairs),
        advantage_std_floored_fractions_on=tuple(
            pair.on.advantage_std_floored_fraction for pair in pairs
        ),
        advantage_std_floored_fractions_off=tuple(
            pair.off.advantage_std_floored_fraction for pair in pairs
        ),
        gradient_cv_vol_on=tuple(gradient_cv_vol_on),
        gradient_cv_vol_off=tuple(gradient_cv_vol_off),
        value_main_target_scale_levels=value_main_scales,
        cf_value_target_scale_levels=cf_value_scales,
        leg_b_confound_downgraded=leg_b_confound_downgraded,
        provenance=STAGE0_PROVENANCE_BLOCK,
        diagnostic_scalars_available=diagnostic_scalars_available,
        cf_warmup_descriptives=tuple((pair.seed, pair.on.cf_warmup) for pair in pairs),
    )


# --- S6: markdown verdict packet ---------------------------------------------------------------
#
# render_packet is PURE over the scored artifacts — it mirrors scripts/proof_packet.py's
# markdown-packet idiom but touches no telemetry (the duckdb assembly lives in
# stage2_acceptance_io.build_report). It surfaces the frozen scorer's verdict, the §3/§4 leg
# deltas, the §5 MECH / §6 safety guards, the §8B floored distribution, the §4 covariates +
# confound-downgrade, and the §0 provenance VERBATIM. It weakens no predicate.


def _tier_label(n: int) -> str:
    """§7/§10 tier: n=10 banks a claim; a clean n=5 is only a screen and is NEVER banked ACCEPT."""
    if n >= 10:
        return "powered-claim tier"
    return "direction-screen tier — a clean screen is SCREEN_PASS, NEVER banked as ACCEPT"


def _fmt_floats(values: tuple[float, ...], prec: int = 4) -> str:
    """Render a per-seed float tuple; an empty tuple (e.g. INVALID) renders as an em-dash."""
    return ", ".join(f"{v:.{prec}f}" for v in values) if values else "—"


def _fmt_churn(values: tuple[ChurnRates, ...]) -> str:
    if not values:
        return "—"
    return "; ".join(
        f"g={value.germinate:.3f}/p={value.prune:.3f}/f={value.fossilize:.3f}"
        for value in values
    )


def _fmt_guard_counts(values: tuple[GuardChannelCounts, ...]) -> str:
    if not values:
        return "—"
    return "; ".join(
        "rollback={rollback}, value_collapse={value_collapse}, ratio_explosion={ratio_explosion}, "
        "ratio_collapse={ratio_collapse}, gradient_anomaly={gradient_anomaly}, "
        "gradient_pathology={gradient_pathology}, numerical_instability={numerical_instability}, "
        "reward_hacking={reward_hacking}".format(
            rollback=value.governor_rollback,
            value_collapse=value.value_collapse,
            ratio_explosion=value.ratio_explosion,
            ratio_collapse=value.ratio_collapse,
            gradient_anomaly=value.gradient_anomaly,
            gradient_pathology=value.gradient_pathology,
            numerical_instability=value.numerical_instability,
            reward_hacking=value.reward_hacking,
        )
        for value in values
    )


def _gate(result: GateResult | None) -> str:
    """Render a 3-valued gate result; ``None`` (no interpretation on INVALID) is an em-dash."""
    return result.value if result is not None else "—"


def _provenance_section(report: Stage2Report) -> str:
    """§0 provenance — the block travels VERBATIM (do not paraphrase), fenced so it renders intact."""
    return "\n".join(
        [
            "## §0 Provenance (Stage-0 evidence is NOT HRA evidence)",
            "",
            "```",
            report.provenance,
            "```",
        ]
    )


def render_packet(
    report: Stage2Report,
    *,
    thresholds: FrozenThresholds,
    validity: Validity,
    calibration: CalibrationReport | None = None,
) -> str:
    """Render the Stage-2 HRA MAJOR-1 acceptance verdict as a §-structured markdown packet.

    Pure over the already-scored artifacts (no I/O, no torch). The §0 provenance block travels
    VERBATIM; the tier caveat (§7/§10) is stated so an n=5 screen can never read as a banked ACCEPT.
    An INVALID run lists every §1 breach and performs NO leg/guard interpretation.
    """
    lines: list[str] = [
        "# Stage-2 HRA MAJOR-1 acceptance verdict",
        "",
        f"**Verdict: {report.verdict.value}**  (n={report.n}, {_tier_label(report.n)})",
        "",
    ]

    if report.verdict is Verdict.INVALID:
        lines += [
            "## §1 Validity — INVALID",
            "",
            "The pair failed validity; per §1, NO leg/guard interpretation is performed.",
            "",
            "Breaches:",
        ]
        lines += [f"- {reason}" for reason in validity.reasons] or ["- (no reasons recorded)"]
        lines += ["", _provenance_section(report)]
        return "\n".join(lines)

    lines += ["## §1 Validity", "", "VALID — pairing, finiteness, and completeness gates pass.", ""]

    lines += [
        "## §3 LEG-A — ev_sum non-inferiority (MAJOR-1 floor)",
        "",
        f"Result: **{_gate(report.leg_a)}**   (δ = {thresholds.delta:.4f}, frozen from OFF calibration)",
        f"Per-seed Δ_A (ev_level ON − OFF): {_fmt_floats(report.delta_a)}",
        "",
    ]

    downgrade = (
        "YES — the reduction is matched by a return-regime or gradient-CV volatility confound "
        "(demoted to INCONCLUSIVE)"
        if report.leg_b_confound_downgraded
        else "NO"
    )
    lines += [
        "## §4 LEG-B — advantage-path volatility reduction (gated)",
        "",
        f"Result: **{_gate(report.leg_b)}**   (ε_rel = {thresholds.eps_rel:.2f})",
        f"Per-seed Δ_B (rel. adv-residual IQR ON vs OFF): {_fmt_floats(report.delta_b)}",
        f"Covariate IQR(Var(returns_total)) — ON:  {_fmt_floats(report.var_returns_vol_on)}",
        f"Covariate IQR(Var(returns_total)) — OFF: {_fmt_floats(report.var_returns_vol_off)}",
        f"Covariate raw IQR(pre_norm_advantage_std) — ON:  {_fmt_floats(report.raw_adv_std_vol_on)}",
        f"Covariate raw IQR(pre_norm_advantage_std) — OFF: {_fmt_floats(report.raw_adv_std_vol_off)}",
        f"Corroboration IQR(gradient_cv) — ON:  {_fmt_floats(report.gradient_cv_vol_on)}",
        f"Corroboration IQR(gradient_cv) — OFF: {_fmt_floats(report.gradient_cv_vol_off)}",
        f"§4 confound downgrade: {downgrade}",
        "",
    ]

    lines += [
        "## §5 MECH — EV_main mechanism guard",
        "",
        f"Hold: {report.mech_hold}   (ON ev_main volatility below the OFF explained_variance "
        "baseline on ≥⌈0.8n⌉ seeds)",
        "",
    ]

    lines += [
        "## §6 Safety guards",
        "",
        f"G1 val-acc non-regression: **{_gate(report.g1)}**   (τ_acc = {thresholds.tau_acc:.2f} pp)",
        f"G2 added-params non-inflation: {report.g2_hold}   (Δparam_max = {thresholds.delta_param_max:g})",
        f"G3 churn not reward-farmed: {report.g3_hold}",
        f"G4 guard channels within OFF baseline: {report.g4_hold}",
    ]
    if report.safety_evidence is not None:
        evidence = report.safety_evidence
        lines += [
            f"G3 materiality threshold: ratio_max = {evidence.g3_ratio_max:g}",
            f"G3 churn rates ON:  {_fmt_churn(evidence.g3_churn_on)}",
            f"G3 churn rates OFF: {_fmt_churn(evidence.g3_churn_off)}",
            f"G4 materiality thresholds: ratio_max = {evidence.g4_ratio_max:g}, "
            f"abs_floor = {evidence.g4_abs_floor}",
            f"G4 guard counts ON:  {_fmt_guard_counts(evidence.g4_counts_on)}",
            f"G4 guard counts OFF: {_fmt_guard_counts(evidence.g4_counts_off)}",
        ]
    lines += [""]

    lines += [
        "## §8B — floored-update exclusion",
        "",
        f"Floored fraction ON:  {_fmt_floats(report.on_floored_fractions, 3)}",
        f"Floored fraction OFF: {_fmt_floats(report.off_floored_fractions, 3)}",
        "Caveat: telemetry exposes only the total `ev_return_variance` (per-stream blindness); the "
        "floor bites on the total, so the M1/MECH per-stream artifact travels with this verdict.",
        "",
    ]

    adv_floor_observed = any(report.advantage_std_floored_fractions_on) or any(
        report.advantage_std_floored_fractions_off
    )
    lines += [
        "## §8F — advantage-normalizer floor contamination",
        "",
        f"advantage_std_floored flag: {'OBSERVED' if adv_floor_observed else 'clear'}",
        f"advantage_std_floored fraction ON:  {_fmt_floats(report.advantage_std_floored_fractions_on, 3)}",
        f"advantage_std_floored fraction OFF: {_fmt_floats(report.advantage_std_floored_fractions_off, 3)}",
        "Interpretation: nonzero fractions are contamination evidence for LEG-B; this packet reports "
        "the exact fractions because no frozen frequency cutoff is registered in §11.",
        "",
    ]

    if report.diagnostic_scalars_available:
        lines += [
            "## §9 — diagnostic scalars",
            "",
            "available",
            f"value_main_target_scale per seed: {_fmt_floats(report.value_main_target_scale_levels)}",
            f"cf_value_target_scale per seed: {_fmt_floats(report.cf_value_target_scale_levels)}",
            "",
        ]
    else:
        lines += [
            "## §9 — diagnostic scalars",
            "",
            "UNAVAILABLE — value_main_target_scale / cf_value_target_scale are absent from the ON leg",
            "",
        ]

    lines += ["## §11.2 — cf-warmup descriptive (report, do not gate)", ""]
    if not report.cf_warmup_descriptives:
        lines += ["unavailable (INVALID run — no scored ON arms)", ""]
    else:
        for seed, block in report.cf_warmup_descriptives:
            if block is None:
                lines.append(f"- seed {seed}: not computed")
                continue
            plateau_txt = (
                "never" if block.plateau_at is None else f"update {block.plateau_at}"
            )
            if block.scale_p50 is None:
                scale_txt = "absent"
            else:
                scale_txt = (
                    f"p50/p90/max {block.scale_p50:.3g}/{block.scale_p90:.3g}/"
                    f"{block.scale_max:.3g}, Q1→Q4 median "
                    f"{block.scale_q1_median:.3g}→{block.scale_q4_median:.3g}"
                )
            lines.append(
                f"- seed {seed}: cf_value_loss present {block.present_scored}/"
                f"{block.n_scored} scored, finite {block.finite_scored}/{block.n_scored}; "
                f"plateau {plateau_txt}; loss p50/p90/max {block.loss_p50:.3g}/"
                f"{block.loss_p90:.3g}/{block.loss_max:.3g}; cf scale {scale_txt}"
            )
        lines += [
            "Interpretation: presence+finiteness are the §1 hard gates; the plateau index and "
            "scale drift are descriptive warmup evidence only (§11.2) and MUST NOT affect the "
            "screen verdict.",
            "",
        ]

    if calibration is not None:
        lines += [
            "## §10 — OFF calibration (freeze provenance)",
            "",
            f"δ frozen from the OFF arms: {calibration.delta:.4f}",
            f"ε_rel: {calibration.eps_rel:.2f}   (OFF seed-to-seed adv-residual spread anchor: "
            f"{calibration.eps_rel_off_spread_anchor:.4f})",
            "",
        ]

    lines += [_provenance_section(report)]
    return "\n".join(lines)
