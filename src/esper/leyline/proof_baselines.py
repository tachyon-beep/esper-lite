"""Shared contracts for blueprint-health proof baseline cohorts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from esper.leyline.factored_actions import (
    AlphaCurveAction,
    AlphaSpeedAction,
    AlphaTargetAction,
    BlueprintAction,
    FactoredAction,
    GerminationStyle,
    LifecycleOp,
    TEMPO_TO_EPOCHS,
    TempoAction,
)
from esper.leyline.telemetry import TopologyManifestPayload


class ProofBaselineMode(str, Enum):
    """Required control modes for blueprint-health proof packets."""

    OFF_SWITCH = "off_switch"
    STATIC_INITIAL = "static_initial"
    STATIC_FINAL = "static_final"
    FIXED_SCHEDULE = "fixed_schedule"
    LOCKSTEP_REWARD_AB = "lockstep_reward_ab"
    # Causal-contribution harness (R1 pilot). NOT a blueprint-health baseline:
    # deliberately absent from REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES.
    SUPPRESS_SLOT = "suppress_slot"


STATIC_FINAL_SOURCE_COHORT_ID = "static_final_source"
STATIC_FINAL_SOURCE_MODE = "static_final_source"
STATIC_FINAL_SOURCE_LIFECYCLE_POLICY = "evolve_source_final_topology"
STATIC_FINAL_SOURCE_TOPOLOGY_V1 = "static-final-source-topology-v1"
STATIC_FINAL_SOURCE_TOPOLOGY_VERSION = 1
STATIC_FINAL_SOURCE_TRAINING_DWELL_EPOCHS = 10
STATIC_FINAL_SOURCE_BLEND_RAMP_EPOCHS = TEMPO_TO_EPOCHS[TempoAction.STANDARD]
STATIC_FINAL_SOURCE_GERMINATE_EPOCH = 1
STATIC_FINAL_SOURCE_TO_TRAINING_EPOCH = 2
STATIC_FINAL_SOURCE_TO_BLENDING_EPOCH = (
    STATIC_FINAL_SOURCE_TO_TRAINING_EPOCH
    + STATIC_FINAL_SOURCE_TRAINING_DWELL_EPOCHS
)
STATIC_FINAL_SOURCE_TO_HOLDING_EPOCH = (
    STATIC_FINAL_SOURCE_TO_BLENDING_EPOCH
    + STATIC_FINAL_SOURCE_BLEND_RAMP_EPOCHS
)
STATIC_FINAL_SOURCE_FOSSILIZE_EPOCH = STATIC_FINAL_SOURCE_TO_HOLDING_EPOCH + 1

REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES: tuple[str, ...] = (
    ProofBaselineMode.OFF_SWITCH.value,
    ProofBaselineMode.STATIC_INITIAL.value,
    ProofBaselineMode.STATIC_FINAL.value,
    ProofBaselineMode.FIXED_SCHEDULE.value,
    ProofBaselineMode.LOCKSTEP_REWARD_AB.value,
)


FIXED_SCHEDULE_GERMINATE_R0C0_V1 = "fixed-schedule-germinate-r0c0-v1"
FIXED_SCHEDULE_GERMINATE_R0C0_VERSION = 1
FIXED_SCHEDULE_GERMINATE_R0C0_EXPECTED_HASH = (
    "e816388d3566fb7fd2a213cd4e44610aecb791e3e3afe498048920f49a03e797"
)


@dataclass(frozen=True, slots=True)
class ProofBaselineScheduleStep:
    """One declared fixed-schedule lifecycle decision."""

    epoch: int
    action: FactoredAction


FIXED_SCHEDULE_GERMINATE_R0C0_STEPS: tuple[ProofBaselineScheduleStep, ...] = (
    ProofBaselineScheduleStep(
        epoch=1,
        action=FactoredAction(
            slot_idx=0,
            blueprint=BlueprintAction.NORM,
            style=GerminationStyle.SIGMOID_ADD,
            tempo=TempoAction.STANDARD,
            alpha_target=AlphaTargetAction.FULL,
            alpha_speed=AlphaSpeedAction.INSTANT,
            alpha_curve=AlphaCurveAction.LINEAR,
            op=LifecycleOp.GERMINATE,
        ),
    ),
)


WAIT_FIXED_SCHEDULE_ACTION = FactoredAction(
    slot_idx=0,
    blueprint=BlueprintAction.NOOP,
    style=GerminationStyle.SIGMOID_ADD,
    tempo=TempoAction.STANDARD,
    alpha_target=AlphaTargetAction.FULL,
    alpha_speed=AlphaSpeedAction.INSTANT,
    alpha_curve=AlphaCurveAction.LINEAR,
    op=LifecycleOp.WAIT,
)


_SOURCE_ADVANCE_R0C0_ACTION = FactoredAction(
    slot_idx=0,
    blueprint=BlueprintAction.NOOP,
    style=GerminationStyle.SIGMOID_ADD,
    tempo=TempoAction.STANDARD,
    alpha_target=AlphaTargetAction.FULL,
    alpha_speed=AlphaSpeedAction.INSTANT,
    alpha_curve=AlphaCurveAction.LINEAR,
    op=LifecycleOp.ADVANCE,
)


_SOURCE_FOSSILIZE_R0C0_ACTION = FactoredAction(
    slot_idx=0,
    blueprint=BlueprintAction.NOOP,
    style=GerminationStyle.SIGMOID_ADD,
    tempo=TempoAction.STANDARD,
    alpha_target=AlphaTargetAction.FULL,
    alpha_speed=AlphaSpeedAction.INSTANT,
    alpha_curve=AlphaCurveAction.LINEAR,
    op=LifecycleOp.FOSSILIZE,
)


STATIC_FINAL_SOURCE_TOPOLOGY_STEPS: tuple[ProofBaselineScheduleStep, ...] = (
    ProofBaselineScheduleStep(
        epoch=STATIC_FINAL_SOURCE_GERMINATE_EPOCH,
        action=FIXED_SCHEDULE_GERMINATE_R0C0_STEPS[0].action,
    ),
    ProofBaselineScheduleStep(
        epoch=STATIC_FINAL_SOURCE_TO_TRAINING_EPOCH,
        action=_SOURCE_ADVANCE_R0C0_ACTION,
    ),
    ProofBaselineScheduleStep(
        epoch=STATIC_FINAL_SOURCE_TO_BLENDING_EPOCH,
        action=_SOURCE_ADVANCE_R0C0_ACTION,
    ),
    ProofBaselineScheduleStep(
        epoch=STATIC_FINAL_SOURCE_TO_HOLDING_EPOCH,
        action=_SOURCE_ADVANCE_R0C0_ACTION,
    ),
    ProofBaselineScheduleStep(
        epoch=STATIC_FINAL_SOURCE_FOSSILIZE_EPOCH,
        action=_SOURCE_FOSSILIZE_R0C0_ACTION,
    ),
)
STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS = STATIC_FINAL_SOURCE_TOPOLOGY_STEPS[-1].epoch


# --- PIN-E 3-slot placebo hold schedule (esper-lite-94869250f1) ---
# Three staggered near-inert placebo lifecycles, all held at HOLDING alpha=1
# for the remainder of the run. Deliberately NO fossilize step: fossilized
# slots are excluded from the ablation family at shapley_synergy_scale=0, so a
# fossilized placebo would be invisible to the noise-floor measurement. The
# 10-epoch TRAINING dwell per slot satisfies the permissive G2 gate.
FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1 = "fixed-schedule-hold-placebo-3slot-v1"
FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_VERSION = 1


def _placebo_germinate_action(slot_idx: int) -> FactoredAction:
    return FactoredAction(
        slot_idx=slot_idx,
        blueprint=BlueprintAction.PLACEBO,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.INSTANT,
        alpha_curve=AlphaCurveAction.LINEAR,
        op=LifecycleOp.GERMINATE,
    )


def _advance_action(slot_idx: int) -> FactoredAction:
    return FactoredAction(
        slot_idx=slot_idx,
        blueprint=BlueprintAction.NOOP,
        style=GerminationStyle.SIGMOID_ADD,
        tempo=TempoAction.STANDARD,
        alpha_target=AlphaTargetAction.FULL,
        alpha_speed=AlphaSpeedAction.INSTANT,
        alpha_curve=AlphaCurveAction.LINEAR,
        op=LifecycleOp.ADVANCE,
    )


FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_STEPS: tuple[ProofBaselineScheduleStep, ...] = tuple(
    ProofBaselineScheduleStep(epoch=epoch, action=action)
    for epoch, action in (
        (1, _placebo_germinate_action(0)),
        (2, _placebo_germinate_action(1)),
        (3, _placebo_germinate_action(2)),
        (4, _advance_action(0)),   # -> TRAINING
        (5, _advance_action(1)),
        (6, _advance_action(2)),
        (14, _advance_action(0)),  # -> BLENDING (10-epoch G2 dwell: 4..13)
        (15, _advance_action(1)),
        (16, _advance_action(2)),
        (17, _advance_action(0)),  # -> HOLDING (alpha=1, INSTANT)
        (18, _advance_action(1)),
        (19, _advance_action(2)),
    )
)


def _schedule_hash(steps: tuple[ProofBaselineScheduleStep, ...]) -> str:
    payload = [
        {
            "epoch": step.epoch,
            "action": step.action.to_indices(),
        }
        for step in steps
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


FIXED_SCHEDULE_GERMINATE_R0C0_HASH = _schedule_hash(
    FIXED_SCHEDULE_GERMINATE_R0C0_STEPS
)
FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT = len(
    FIXED_SCHEDULE_GERMINATE_R0C0_STEPS
)
STATIC_FINAL_SOURCE_TOPOLOGY_HASH = _schedule_hash(STATIC_FINAL_SOURCE_TOPOLOGY_STEPS)
STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT = len(STATIC_FINAL_SOURCE_TOPOLOGY_STEPS)
STATIC_FINAL_SOURCE_TOPOLOGY_EXPECTED_HASH = (
    "ec5bbce4686177b486454a3a87b43a45c0724e790028ea66eda9c2078f838f3c"
)
FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH = _schedule_hash(
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_STEPS
)
FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_ACTION_COUNT = len(
    FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_STEPS
)
FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_EXPECTED_HASH = (
    "8484773896a8ba74137b7c29106def29b83c316f6c6bb3383d0dad854cf4b7c3"
)


@dataclass(frozen=True, slots=True)
class DeclaredSchedule:
    """A hash-guarded declared proof-baseline lifecycle schedule."""

    schedule_id: str
    version: int
    steps: tuple[ProofBaselineScheduleStep, ...]
    expected_hash: str

    @property
    def hash(self) -> str:
        return _schedule_hash(self.steps)

    @property
    def action_count(self) -> int:
        return len(self.steps)

    def action_for_epoch(self, epoch: int) -> FactoredAction:
        """Return the declared action for an epoch (WAIT when unscheduled)."""
        if epoch < 1:
            raise ValueError(f"declared-schedule epoch must be >= 1, got {epoch}")
        for step in self.steps:
            if step.epoch == epoch:
                return step.action
        return WAIT_FIXED_SCHEDULE_ACTION


DECLARED_SCHEDULES: dict[str, DeclaredSchedule] = {
    schedule.schedule_id: schedule
    for schedule in (
        DeclaredSchedule(
            schedule_id=FIXED_SCHEDULE_GERMINATE_R0C0_V1,
            version=FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
            steps=FIXED_SCHEDULE_GERMINATE_R0C0_STEPS,
            expected_hash=FIXED_SCHEDULE_GERMINATE_R0C0_EXPECTED_HASH,
        ),
        DeclaredSchedule(
            schedule_id=STATIC_FINAL_SOURCE_TOPOLOGY_V1,
            version=STATIC_FINAL_SOURCE_TOPOLOGY_VERSION,
            steps=STATIC_FINAL_SOURCE_TOPOLOGY_STEPS,
            expected_hash=STATIC_FINAL_SOURCE_TOPOLOGY_EXPECTED_HASH,
        ),
        DeclaredSchedule(
            schedule_id=FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1,
            version=FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_VERSION,
            steps=FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_STEPS,
            expected_hash=FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_EXPECTED_HASH,
        ),
    )
}

for _schedule in DECLARED_SCHEDULES.values():
    if _schedule.hash != _schedule.expected_hash:
        raise RuntimeError(
            f"declared-schedule hash drift for {_schedule.schedule_id!r}: "
            "update the schedule version and proof fixtures"
        )
del _schedule


def declared_schedule_action_for_epoch(schedule_id: str, epoch: int) -> FactoredAction:
    """Dispatch to a registered declared schedule by id."""
    if schedule_id not in DECLARED_SCHEDULES:
        raise ValueError(
            f"Unknown declared schedule {schedule_id!r}; "
            f"known: {sorted(DECLARED_SCHEDULES)}"
        )
    return DECLARED_SCHEDULES[schedule_id].action_for_epoch(epoch)


def declared_schedule_germinate_blueprints(
    schedule_id: str,
) -> frozenset[BlueprintAction]:
    """Blueprints a declared schedule germinates (for mask availability union)."""
    if schedule_id not in DECLARED_SCHEDULES:
        raise ValueError(
            f"Unknown declared schedule {schedule_id!r}; "
            f"known: {sorted(DECLARED_SCHEDULES)}"
        )
    return frozenset(
        step.action.blueprint
        for step in DECLARED_SCHEDULES[schedule_id].steps
        if step.action.op == LifecycleOp.GERMINATE
    )


@dataclass(frozen=True, slots=True)
class ProofBaselineCohort:
    """One named cohort in a blueprint-health proof plan."""

    cohort_id: str
    mode: ProofBaselineMode
    reward_mode: str
    training_seed: int
    lifecycle_policy: str
    proof_baseline_pair_id: str
    proof_baseline_schedule_id: str | None
    proof_baseline_schedule_hash: str | None
    proof_baseline_schedule_version: int | None
    proof_baseline_schedule_action_count: int | None
    current_runner_supported: bool


@dataclass(frozen=True, slots=True)
class ProofBaselinePlan:
    """A complete proof-baseline cohort plan for blueprint-health claims."""

    plan_id: str
    cohorts: tuple[ProofBaselineCohort, ...]


@dataclass(frozen=True, slots=True)
class StaticFinalSourceManifestRef:
    """Reference to one emitted source-final topology manifest."""

    payload: TopologyManifestPayload
    run_dir: str
    group_id: str
    episode_idx: int
    event_id: str


__all__ = [
    "ProofBaselineCohort",
    "ProofBaselineMode",
    "ProofBaselinePlan",
    "ProofBaselineScheduleStep",
    "FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT",
    "FIXED_SCHEDULE_GERMINATE_R0C0_EXPECTED_HASH",
    "FIXED_SCHEDULE_GERMINATE_R0C0_HASH",
    "FIXED_SCHEDULE_GERMINATE_R0C0_STEPS",
    "FIXED_SCHEDULE_GERMINATE_R0C0_V1",
    "FIXED_SCHEDULE_GERMINATE_R0C0_VERSION",
    "REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES",
    "STATIC_FINAL_SOURCE_COHORT_ID",
    "STATIC_FINAL_SOURCE_BLEND_RAMP_EPOCHS",
    "STATIC_FINAL_SOURCE_FOSSILIZE_EPOCH",
    "STATIC_FINAL_SOURCE_GERMINATE_EPOCH",
    "STATIC_FINAL_SOURCE_LIFECYCLE_POLICY",
    "STATIC_FINAL_SOURCE_MODE",
    "STATIC_FINAL_SOURCE_TO_BLENDING_EPOCH",
    "STATIC_FINAL_SOURCE_TO_HOLDING_EPOCH",
    "STATIC_FINAL_SOURCE_TO_TRAINING_EPOCH",
    "STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT",
    "STATIC_FINAL_SOURCE_TOPOLOGY_HASH",
    "STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS",
    "STATIC_FINAL_SOURCE_TOPOLOGY_STEPS",
    "STATIC_FINAL_SOURCE_TOPOLOGY_V1",
    "STATIC_FINAL_SOURCE_TOPOLOGY_VERSION",
    "STATIC_FINAL_SOURCE_TRAINING_DWELL_EPOCHS",
    "STATIC_FINAL_SOURCE_TOPOLOGY_EXPECTED_HASH",
    "DECLARED_SCHEDULES",
    "DeclaredSchedule",
    "FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_ACTION_COUNT",
    "FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_EXPECTED_HASH",
    "FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_HASH",
    "FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_STEPS",
    "FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_V1",
    "FIXED_SCHEDULE_HOLD_PLACEBO_3SLOT_VERSION",
    "StaticFinalSourceManifestRef",
    "WAIT_FIXED_SCHEDULE_ACTION",
    "declared_schedule_action_for_epoch",
    "declared_schedule_germinate_blueprints",
]
