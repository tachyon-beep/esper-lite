"""Phase 0: the J integrand — alpha-weighted counterfactual on-output-path residency.

The objective J (methodology §0, form LOCKED, owner-ratified) is:

    J = Σ_seed  (1 / params(seed))  Σ_t  c_t(seed) · α_t(seed) · 1[stage_t ≥ BLENDING]

where ``c_t`` is the PER-SEED leave-one-out marginal counterfactual ``seed_contribution =
val_acc − baseline_accs[slot]`` (PIN A — NOT the env-joint ``val_acc − all_disabled``, which
is constant across seeds and would reinstate the rejected raw-residency form), and ``α_t`` is
the seed's blend weight gated to on-output-path stages (``stage ≥ BLENDING``, ``α > 0``).

``SeedResidencyAccumulator`` is the per-seed accumulator the live ``EnvState`` hook drives
each step (reset per episode). It also tracks the REJECTED raw-residency form (``Σ α``) as a
diagnostic — so the freeloader guard can be demonstrated — the committed/uncommitted split
(residency while FOSSILIZED vs BLENDING/HOLDING — the survival-time-accrual readout), and the
None-LOO step count (the J-reliability caveat).

This module is telemetry/measurement only: J is computed OFFLINE as a yardstick and does NOT
enter the reward.
"""

from __future__ import annotations

from dataclasses import dataclass

from esper.leyline.telemetry_contracts import SeedResidencyTelemetry
from esper.simic.rewards.types import STAGE_BLENDING, STAGE_FOSSILIZED


@dataclass(slots=True)
class SeedResidencyAccumulator:
    """Per-seed alpha-weighted counterfactual residency integral (J integrand).

    ``params`` is the seed's own active-parameter count (PIN B: ``effective_seed_params``),
    snapshotted as the per-seed J denominator.
    """

    params: int
    cf_weighted_integral: float = 0.0  # Σ c_t·α_t over on-path steps with measured LOO
    cf_weighted_integral_committed: float = 0.0  # subset accrued while stage == FOSSILIZED
    raw_alpha_integral: float = 0.0  # Σ α_t over on-path steps (the REJECTED raw form)
    n_on_path_steps: int = 0  # steps with stage ≥ BLENDING and α > 0
    n_none_steps: int = 0  # on-path steps with no measured LOO (seed_contribution is None)

    def add_step(
        self, *, stage: int, alpha: float, seed_contribution: float | None
    ) -> None:
        """Fold one step into the integral.

        On-output-path gate: ``stage ≥ BLENDING`` and ``alpha > 0``. Off-path steps
        contribute nothing. On a measured step the integrand is ``seed_contribution · alpha``
        (PIN A, the per-seed LOO marginal); a ``None`` LOO contributes 0 (no measured
        counterfactual ⇒ no credit) but is counted so the None-fraction is reportable.
        """
        if stage < STAGE_BLENDING or alpha <= 0.0:
            return
        self.n_on_path_steps += 1
        self.raw_alpha_integral += alpha
        if seed_contribution is None:
            self.n_none_steps += 1
            return
        contribution = seed_contribution * alpha
        self.cf_weighted_integral += contribution
        if stage == STAGE_FOSSILIZED:
            self.cf_weighted_integral_committed += contribution

    @property
    def cf_weighted_integral_uncommitted(self) -> float:
        """Residency accrued while BLENDING/HOLDING (pre-commit) — the survival-time bucket."""
        return self.cf_weighted_integral - self.cf_weighted_integral_committed

    def j(self) -> float:
        """This seed's contribution to J = cf-weighted residency integral / params.

        Returns 0.0 for a seed with no measured params (guards a genuine 0; production
        supplies ``effective_seed_params > 0``).
        """
        if self.params <= 0:
            return 0.0
        return self.cf_weighted_integral / self.params

    def to_telemetry(
        self, *, env_id: int, episode_idx: int, seed_id: str
    ) -> SeedResidencyTelemetry:
        """Build the leyline wire contract for this seed's residency (emitted at episode end)."""
        return SeedResidencyTelemetry(
            env_id=env_id,
            episode_idx=episode_idx,
            seed_id=seed_id,
            params=self.params,
            cf_weighted_integral=self.cf_weighted_integral,
            cf_weighted_integral_committed=self.cf_weighted_integral_committed,
            cf_weighted_integral_uncommitted=self.cf_weighted_integral_uncommitted,
            raw_alpha_integral=self.raw_alpha_integral,
            n_on_path_steps=self.n_on_path_steps,
            n_none_steps=self.n_none_steps,
            j_per_param=self.j(),
        )


def compute_residency_j(accumulators: list[SeedResidencyAccumulator]) -> float:
    """Run-level raw J = Σ_seed (cf-weighted residency integral / params)."""
    return sum(acc.j() for acc in accumulators)


@dataclass(slots=True)
class SlotResidencySample:
    """One slot's per-epoch residency inputs (built from the live model in the sweep caller).

    ``seed_id`` is ``None`` for an empty slot. ``params`` is the slot's active-seed param count
    (PIN B). The per-seed LOO marginal counterfactual is computed in the sweep from
    ``baseline_accs`` keyed by ``slot_id``.
    """

    slot_id: str
    seed_id: str | None
    stage: int
    alpha: float
    params: int


def accumulate_residency_for_env(
    accumulators: dict[str, SeedResidencyAccumulator],
    *,
    val_acc: float,
    baseline_accs_env: dict[str, float],
    slot_samples: list[SlotResidencySample],
) -> None:
    """Fold one epoch of per-slot state into the per-seed residency accumulator map.

    For each slot with a seed, computes the PER-SEED LOO marginal (PIN A)
    ``seed_contribution = val_acc − baseline_accs_env[slot_id]`` — or ``None`` when that slot
    was not ablated this epoch (LOO unmeasured) — and delegates the on-output-path gate
    (``stage ≥ BLENDING``, ``α > 0``) to ``SeedResidencyAccumulator.add_step``. Accumulators are
    keyed by ``seed_id`` so a slot reused by successive seed instances attributes residency
    correctly; ``params`` is refreshed to the latest sample (PIN B snapshot).

    Mutates ``accumulators`` in place. No-ops for empty slots (``seed_id is None``).
    """
    for sample in slot_samples:
        if sample.seed_id is None:
            continue
        acc = accumulators.get(sample.seed_id)
        if acc is None:
            acc = SeedResidencyAccumulator(params=sample.params)
            accumulators[sample.seed_id] = acc
        acc.params = sample.params
        # PIN A: per-seed LOO marginal. ``None`` is a real "LOO unmeasured this epoch" state
        # (the first-BLENDING-step gap), NOT a defensive default — the accumulator credits it 0
        # and counts it so the None-fraction is reportable.
        if sample.slot_id in baseline_accs_env:
            seed_contribution: float | None = val_acc - baseline_accs_env[sample.slot_id]
        else:
            seed_contribution = None
        acc.add_step(
            stage=sample.stage, alpha=sample.alpha, seed_contribution=seed_contribution
        )
