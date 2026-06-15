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
if (
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH
    != FIXED_SCHEDULE_GERMINATE_R0C0_EXPECTED_HASH
):
    raise RuntimeError(
        "fixed-schedule hash drift: update the schedule version and proof fixtures"
    )
FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT = len(
    FIXED_SCHEDULE_GERMINATE_R0C0_STEPS
)
STATIC_FINAL_SOURCE_TOPOLOGY_HASH = _schedule_hash(STATIC_FINAL_SOURCE_TOPOLOGY_STEPS)
STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT = len(STATIC_FINAL_SOURCE_TOPOLOGY_STEPS)


def fixed_schedule_action_for_epoch(epoch: int) -> FactoredAction:
    """Return the declared fixed-schedule action for an epoch.

    Epochs without an explicit lifecycle operation are scheduled WAIT steps.
    """
    if epoch < 1:
        raise ValueError(f"fixed-schedule epoch must be >= 1, got {epoch}")

    for step in FIXED_SCHEDULE_GERMINATE_R0C0_STEPS:
        if step.epoch == epoch:
            return step.action
    return WAIT_FIXED_SCHEDULE_ACTION


def static_final_source_action_for_epoch(epoch: int) -> FactoredAction:
    """Return the deterministic source-final topology action for an epoch."""
    if epoch < 1:
        raise ValueError(f"static-final source epoch must be >= 1, got {epoch}")

    for step in STATIC_FINAL_SOURCE_TOPOLOGY_STEPS:
        if step.epoch == epoch:
            return step.action
    return WAIT_FIXED_SCHEDULE_ACTION


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
    "StaticFinalSourceManifestRef",
    "WAIT_FIXED_SCHEDULE_ACTION",
    "fixed_schedule_action_for_epoch",
    "static_final_source_action_for_epoch",
]
