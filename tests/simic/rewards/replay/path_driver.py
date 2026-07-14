"""Replay path driver: synthetic seed trajectories through the PRODUCTION reward path.

The driver constructs the per-epoch ``ContributionRewardInputs`` a single-seed
trajectory presents to ``compute_reward`` and collects the per-component
ledger. It never reimplements reward math: every reward number in the ledger
comes from the shipping ``compute_reward`` dispatch (the PRUNE-instant cull is
derived from production output by subtracting the two production non-PBRS
PRUNE terms, with the analytic closed form kept only as a fail-loud desync
detector). The only analytic values computed here are the PBRS *potentials*
(from the shipping ``STAGE_POTENTIALS`` + config constants) needed to state
the telescoping identities the ledger is checked against.

Two timing conventions are supported, because the trainer's live ordering and
the ordering the reward-path audit assumed DISAGREE about what the first
reward call after a stage transition sees:

- ``convention="live"`` - the trainer's actual ordering: ``record_accuracy``
  ticks ``epochs_in_current_stage`` for every slot with state during the
  metrics phase (vectorized_trainer.py:1870-1875), the reward is computed on
  the pre-action state (action_execution.py:226-230, :973-977, :1089), and
  the mutation + ``step_epoch`` run after (action_execution.py dispatch phase,
  :1643). The first post-transition reward row therefore sees
  ``epochs_in_stage == 1`` and the PBRS ``eis == 0`` branch
  (contribution.py:1235) never fires: cross-stage transition deltas are
  SKIPPED. Pinned against the real kasmina state machine in
  test_kasmina_fidelity.py.
- ``convention="audit"`` - the ordering the audit (and the eis==0 branch's
  own design, per its warning message) assumed: the first post-transition
  reward row sees ``epochs_in_stage == 0`` with ``previous_*`` describing the
  exited stage, so every transition delta pays. Under this convention the
  stream satisfies the full Ng telescoping identity.

Modeling assumptions (per audit A3's per-decision semantics; documented, not
hidden):

- Single seed, single slot; the policy targets this slot every epoch. The stay
  branch is WAIT-on-slot each epoch, so ``holding_warning`` fires on every
  eligible epoch - the audit's stay-branch ledger, an upper bound on
  indecision-penalty traffic.
- Paths begin at BLENDING entry. Germination is out of Phase-1 scope (the
  GERMINATE PBRS timing discount is a known, intentional telescoping
  deviation and the S/A contrast lives from HOLDING onward).
- ``val_acc`` constant, ``acc_at_germination`` set so progress exceeds any
  test contribution: ``attributed == c`` exactly on measured epochs.
- ``total_params == host_params`` (zero overhead -> compute rent 0) and one
  seed never exceeds ``free_slots`` (occupancy rent 0). Both stay in the
  ledger and must read 0; multi-seed occupancy is Phase 2 (G-CROSS-SLOT).
- Post-PRUNE cooldown stages (PRUNED/EMBARGOED/RESETTING) are modeled as
  seed-gone: live they retain state and could be reward-visible if targeted,
  but only if the policy targets a dying slot - out of Phase-1 scope.
- A-path scenarios transition to FOSSILIZED on the commit row unconditionally.
  For every load-bearing scenario (lifetime cf >= 0 with c >= 1) the kasmina
  G5 gate would also pass; the sub-threshold case-matrix smoke rows model the
  reward consequences of a commit the strict-mode gate could block (permissive
  mode passes on cf alone) - no finding rests on them.
- ``slot_id``/``seed_id`` stay None so the telemetry hub is never touched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Literal

from esper.leyline import DEFAULT_GAMMA, LifecycleOp, SeedStage
from esper.leyline.telemetry_contracts import RewardComponentsTelemetry
from esper.simic.rewards import (
    ContributionRewardConfig,
    ContributionRewardInputs,
    RewardMode,
    SeedInfo,
    compute_reward,
)
from esper.simic.rewards.contribution import _contribution_prune_shaping
from esper.simic.rewards.shaping import STAGE_POTENTIALS

PathKind = Literal["S", "A", "PRUNE"]
Convention = Literal["live", "audit"]

# The settlement's fixed commitment prior, frozen as an EXPRESSION evaluated
# in-code (PDR-0084 #2): the matched-control consolidation of both shipping
# flat FOSSILIZE terms (0.5 base + 3*tanh(1/3) immediate).
PREMIUM_EXPRESSION_VALUE = 0.5 + 3.0 * math.tanh(1.0 / 3.0)


@dataclass(frozen=True)
class PathScenario:
    """One synthetic single-seed trajectory (A6 case-matrix point).

    ``action_dwell`` is the HOLDING ``epochs_in_stage`` value AT the action
    row (what legitimacy reads), identical across conventions; the calendar
    epoch of the action differs by one between them.
    """

    kind: PathKind
    spot_contribution: float | None
    lifetime_counterfactual: float | None
    start_epoch: int  # BLENDING-entry epoch (first reward row of the path)
    blending_dwell: int  # physical epochs spent in BLENDING before HOLDING
    prior_training_dwell: int  # epochs_in_stage at the end of TRAINING
    action_dwell: int | None  # eis at FOSSILIZE/PRUNE row; None for S
    max_epochs: int
    germination_epoch: int  # >= warmup(10) so the D3 timing discount is 1.0
    val_acc: float = 60.0
    acc_at_germination: float = 40.0
    # None => a valid measurement every epoch ("fresh"). Otherwise the set of
    # epochs with a valid measurement; unmeasured epochs pass contribution=None
    # (the "stale/never" statuses of the A6 matrix).
    measured_epochs: frozenset[int] | None = None
    # The seed's own stage-relative improvement (feeds the PROXY attribution
    # branch, contribution.py:602-642, which pays 0.3x on unmeasured epochs
    # for stage >= BLENDING). None models a flat-val_acc world where the
    # stage-relative signal never materializes; set positive to exercise the
    # live proxy path.
    stage_improvement: float | None = None

    @property
    def holding_entry_epoch(self) -> int:
        return self.start_epoch + self.blending_dwell

    @property
    def action_epoch(self) -> int | None:
        """Calendar epoch of the action row - convention-independent for
        "live" (eis is 1-based: entry+dwell-1) and one later for "audit"
        (eis 0-based). Exposed as the LIVE epoch; audit paths compute their
        own inside the sequencer."""
        if self.action_dwell is None:
            return None
        return self.holding_entry_epoch + self.action_dwell - 1


@dataclass(frozen=True)
class _SeedRowSpec:
    epoch: int
    stage: SeedStage
    eis: int
    previous_stage: SeedStage
    previous_dwell: int
    action: LifecycleOp


@dataclass(frozen=True)
class LedgerRow:
    epoch: int
    action: LifecycleOp
    stage: int | None  # SeedStage.value during this epoch; None once seed gone
    epochs_in_stage: int | None
    previous_stage: int | None
    previous_epochs_in_stage: int | None
    reward_total: float
    components: RewardComponentsTelemetry
    # PBRS paid inside action_shaping at the action instant (the PRUNE cull);
    # the per-step stream lives in components.pbrs_bonus. Derived from the
    # shipping constants - see prune_cull_pbrs().
    embedded_pbrs: float


@dataclass
class PathLedger:
    scenario: PathScenario
    convention: Convention
    rows: list[LedgerRow]
    config: ContributionRewardConfig

    def discount(self, epoch: int, base_epoch: int | None = None) -> float:
        base = self.scenario.start_epoch if base_epoch is None else base_epoch
        return self.config.gamma ** (epoch - base)

    def pv(self, component: str, from_epoch: int | None = None) -> float:
        """Discounted PV of one components-field (from from_epoch onward)."""
        base = self.scenario.start_epoch if from_epoch is None else from_epoch
        total = 0.0
        for row in self.rows:
            if row.epoch < base:
                continue
            value = getattr(row.components, component)
            if value is None:
                value = 0.0
            total += self.discount(row.epoch, base) * value
        return total

    def pv_total(self, from_epoch: int | None = None) -> float:
        base = self.scenario.start_epoch if from_epoch is None else from_epoch
        return sum(
            self.discount(row.epoch, base) * row.reward_total
            for row in self.rows
            if row.epoch >= base
        )

    def pv_pbrs(self, from_epoch: int | None = None) -> float:
        """Discounted PV of ALL PBRS terms: per-step stream + embedded cull."""
        base = self.scenario.start_epoch if from_epoch is None else from_epoch
        total = 0.0
        for row in self.rows:
            if row.epoch < base:
                continue
            per_step = row.components.pbrs_bonus
            if per_step is None:
                per_step = 0.0
            total += self.discount(row.epoch, base) * (per_step + row.embedded_pbrs)
        return total


def potential(
    stage: SeedStage, epochs_in_stage: int, config: ContributionRewardConfig
) -> float:
    """The shipping PBRS potential: stage base + capped dwell progress."""
    return STAGE_POTENTIALS[stage] + min(
        epochs_in_stage * config.epoch_progress_bonus, config.max_progress_bonus
    )


def prune_cull_pbrs(
    stage: SeedStage, epochs_in_stage: int, config: ContributionRewardConfig
) -> float:
    """The PRUNE-instant cull paid inside action_shaping (contribution.py PRUNE branch)."""
    phi_current = potential(stage, epochs_in_stage, config)
    return config.pbrs_weight * (config.gamma * 0.0 - phi_current)


def make_config() -> ContributionRewardConfig:
    """The experiment regime: SHAPED, production defaults."""
    return ContributionRewardConfig(reward_mode=RewardMode.SHAPED)


def _validate(scenario: PathScenario) -> None:
    if scenario.blending_dwell < 1:
        raise ValueError("blending_dwell must be >= 1")
    if scenario.kind in ("A", "PRUNE") and scenario.action_dwell is None:
        raise ValueError(f"{scenario.kind}-path requires action_dwell")
    if scenario.kind == "S" and scenario.action_dwell is not None:
        raise ValueError("S-path takes no action_dwell")
    if scenario.kind != "S" and (scenario.action_dwell or 0) < 1:
        raise ValueError("action_dwell must be >= 1 (eis at the action row)")
    if scenario.germination_epoch > scenario.start_epoch:
        raise ValueError("germination must precede BLENDING entry")
    if scenario.germination_epoch < 10:
        raise ValueError(
            "keep germination_epoch >= warmup(10) so the timing discount is 1.0"
        )


def _seed_rows(scenario: PathScenario, convention: Convention) -> Iterator[_SeedRowSpec]:
    """The SeedInfo sequence the reward path sees, per timing convention.

    live: eis is 1-based within each stage (the tick precedes the reward
    call); previous_dwell mirrors kasmina's transition() record (the ticked
    value at transition).
    audit: eis is 0-based (transition visible at eis==0); previous_dwell is
    the LAST-PAID eis of the exited stage, so the stream's anchors chain and
    the full Ng identity can hold exactly.
    """
    base = 1 if convention == "live" else 0
    h = scenario.holding_entry_epoch
    blending_rows = scenario.blending_dwell
    action_row_epoch = (
        None
        if scenario.action_dwell is None
        else h + scenario.action_dwell - base
    )
    if action_row_epoch is not None and action_row_epoch > scenario.max_epochs:
        raise ValueError("action epoch beyond horizon")
    prev_blending_dwell = (
        scenario.blending_dwell if convention == "live" else scenario.blending_dwell - 1
    )

    for epoch in range(scenario.start_epoch, scenario.max_epochs + 1):
        if epoch < h:
            yield _SeedRowSpec(
                epoch=epoch,
                stage=SeedStage.BLENDING,
                eis=(epoch - scenario.start_epoch) + base,
                previous_stage=SeedStage.TRAINING,
                previous_dwell=scenario.prior_training_dwell,
                action=LifecycleOp.WAIT,
            )
        elif action_row_epoch is None or epoch <= action_row_epoch:
            action = LifecycleOp.WAIT
            if action_row_epoch is not None and epoch == action_row_epoch:
                action = (
                    LifecycleOp.FOSSILIZE
                    if scenario.kind == "A"
                    else LifecycleOp.PRUNE
                )
            yield _SeedRowSpec(
                epoch=epoch,
                stage=SeedStage.HOLDING,
                eis=(epoch - h) + base,
                previous_stage=SeedStage.BLENDING,
                previous_dwell=prev_blending_dwell,
                action=action,
            )
        elif scenario.kind == "A":
            yield _SeedRowSpec(
                epoch=epoch,
                stage=SeedStage.FOSSILIZED,
                eis=(epoch - action_row_epoch - 1) + base,
                previous_stage=SeedStage.HOLDING,
                previous_dwell=scenario.action_dwell or 0,
                action=LifecycleOp.WAIT,
            )
        else:
            return  # PRUNE aftermath: slot is empty; no more seed rows


def replay_path(
    scenario: PathScenario,
    convention: Convention = "live",
    config: ContributionRewardConfig | None = None,
) -> PathLedger:
    """Drive the production reward path over the scenario, one call per epoch."""
    if config is None:
        config = make_config()
    _validate(scenario)

    specs = {spec.epoch: spec for spec in _seed_rows(scenario, convention)}

    rows: list[LedgerRow] = []
    for epoch in range(scenario.start_epoch, scenario.max_epochs + 1):
        spec = specs.get(epoch)

        if spec is None:  # PRUNE aftermath
            seed_info = None
            seed_contribution = None
            action = LifecycleOp.WAIT
            n_active = 0
            num_fossilized = 0
            embedded_pbrs = 0.0
        else:
            measured = (
                scenario.measured_epochs is None or epoch in scenario.measured_epochs
            )
            fossilized = spec.stage == SeedStage.FOSSILIZED
            # Fossils are excluded from the counterfactual ablation by design
            # (vectorized_trainer.py:959-978) -> contribution is None.
            seed_contribution = (
                scenario.spot_contribution if (measured and not fossilized) else None
            )
            counterfactual = (
                scenario.lifetime_counterfactual if seed_contribution is not None
                else (scenario.lifetime_counterfactual if fossilized else None)
            )
            seed_info = SeedInfo(
                stage=spec.stage.value,
                improvement_since_stage_start=scenario.stage_improvement,
                total_improvement=counterfactual,
                epochs_in_stage=spec.eis,
                seed_params=0,
                previous_stage=spec.previous_stage.value,
                previous_epochs_in_stage=spec.previous_dwell,
                seed_age_epochs=epoch - scenario.germination_epoch,
                counterfactual_total_improvement=counterfactual,
            )
            action = spec.action
            n_active = 0 if fossilized else 1
            num_fossilized = 1 if fossilized else 0

        result = compute_reward(
            ContributionRewardInputs(
                action=action,
                seed_contribution=seed_contribution,
                val_acc=scenario.val_acc,
                seed_info=seed_info,
                epoch=epoch,
                max_epochs=scenario.max_epochs,
                total_params=100_000,
                host_params=100_000,
                acc_at_germination=scenario.acc_at_germination,
                acc_delta=0.0,
                config=config,
                return_components=True,
                num_fossilized_seeds=num_fossilized,
                num_contributing_fossilized=num_fossilized,
                n_active_seeds=n_active,
            )
        )
        assert isinstance(result, tuple)
        reward_total, components = result[0], result[1]

        # The PRUNE-instant cull is DERIVED from the production output
        # (action_shaping minus the two non-PBRS PRUNE terms, both production
        # functions), so a production change cannot silently desync the
        # ledger; the analytic closed form is kept as a fail-loud detector.
        embedded_pbrs = 0.0
        if action == LifecycleOp.PRUNE:
            assert spec is not None and seed_info is not None
            embedded_pbrs = (
                components.action_shaping
                - config.prune_cost
                - _contribution_prune_shaping(seed_info, seed_contribution, config)
            )
            analytic = prune_cull_pbrs(spec.stage, spec.eis, config)
            if abs(embedded_pbrs - analytic) > 1e-12:
                raise AssertionError(
                    f"PRUNE cull desync: production-derived {embedded_pbrs} != "
                    f"analytic {analytic} at epoch {epoch} - the shipping PRUNE "
                    "shaping decomposition changed; update prune_cull_pbrs()"
                )

        rows.append(
            LedgerRow(
                epoch=epoch,
                action=action,
                stage=None if spec is None else spec.stage.value,
                epochs_in_stage=None if spec is None else spec.eis,
                previous_stage=None if spec is None else spec.previous_stage.value,
                previous_epochs_in_stage=None if spec is None else spec.previous_dwell,
                reward_total=reward_total,
                components=components,
                embedded_pbrs=embedded_pbrs,
            )
        )

    return PathLedger(scenario=scenario, convention=convention, rows=rows, config=config)


def expected_pbrs_pv(ledger: PathLedger, exclude_embedded: bool = False) -> float:
    """Closed-form value of the per-step PBRS stream, by stage-group telescoping.

    Each maximal run of consecutive seed rows in one stage contributes
    ``w * (gamma^(b+1) * Phi(stage, eis_last) - gamma^a * anchor)`` where a/b
    are the first/last row offsets from path start and ``anchor`` replicates
    the production phi_prev branch for the group's first row (eis==0 -> the
    exited stage's potential at previous_dwell; else the same stage one epoch
    earlier). Under "audit" sequencing the group anchors chain, so the total
    collapses to the full Ng identity; under "live" they do not - the gap IS
    the unpaid transition deltas.
    """
    config = ledger.config
    w = config.pbrs_weight
    gamma = config.gamma
    start = ledger.scenario.start_epoch

    total = 0.0
    group: list[LedgerRow] = []

    def flush(group_rows: list[LedgerRow]) -> float:
        first, last = group_rows[0], group_rows[-1]
        a = first.epoch - start
        b = last.epoch - start
        phi_last = potential(SeedStage(last.stage), last.epochs_in_stage, config)
        if first.epochs_in_stage == 0:
            anchor = potential(
                SeedStage(first.previous_stage),
                first.previous_epochs_in_stage,
                config,
            )
        else:
            anchor = potential(
                SeedStage(first.stage), first.epochs_in_stage - 1, config
            )
        return w * (gamma ** (b + 1) * phi_last - gamma**a * anchor)

    for row in ledger.rows:
        if row.stage is None:
            if group:
                total += flush(group)
                group = []
            continue
        if group and row.stage != group[-1].stage:
            total += flush(group)
            group = []
        group.append(row)
    if group:
        total += flush(group)

    if not exclude_embedded:
        total += sum(
            ledger.discount(row.epoch) * row.embedded_pbrs for row in ledger.rows
        )
    return total


def render_ledger_markdown(ledger: PathLedger) -> str:
    """Per-component discounted-PV table (the A6 'printing' requirement)."""
    components = [
        "bounded_attribution",
        "blending_warning",
        "holding_warning",
        "pbrs_bonus",
        "synergy_bonus",
        "compute_rent",
        "occupancy_rent",
        "fossilized_rent",
        "action_shaping",
        "terminal_bonus",
    ]
    s = ledger.scenario
    lines = [
        f"### {s.kind}-path ({ledger.convention}) | c={s.spot_contribution} "
        f"cf={s.lifetime_counterfactual} | HOLDING entry {s.holding_entry_epoch} "
        f"| action_dwell {s.action_dwell} | horizon {s.max_epochs}",
        "",
        "| component | discounted PV (from path start) |",
        "|---|---|",
    ]
    for name in components:
        lines.append(f"| {name} | {ledger.pv(name):+.6f} |")
    embedded = sum(ledger.discount(r.epoch) * r.embedded_pbrs for r in ledger.rows)
    lines.append(f"| embedded PBRS (prune cull) | {embedded:+.6f} |")
    lines.append(f"| **total** | **{ledger.pv_total():+.6f}** |")
    lines.append("")
    residual = ledger.pv_pbrs() - expected_pbrs_pv(ledger)
    lines.append(f"PBRS group-telescoping residual: {residual:+.9f}")
    return "\n".join(lines)


assert math.isclose(DEFAULT_GAMMA, 0.995), "regime drift: DEFAULT_GAMMA changed"
