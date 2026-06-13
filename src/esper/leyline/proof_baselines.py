"""Shared contracts for blueprint-health proof baseline cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProofBaselineMode(str, Enum):
    """Required control modes for blueprint-health proof packets."""

    OFF_SWITCH = "off_switch"
    STATIC_INITIAL = "static_initial"
    STATIC_FINAL = "static_final"
    FIXED_SCHEDULE = "fixed_schedule"
    LOCKSTEP_REWARD_AB = "lockstep_reward_ab"


REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES: tuple[str, ...] = (
    ProofBaselineMode.OFF_SWITCH.value,
    ProofBaselineMode.STATIC_INITIAL.value,
    ProofBaselineMode.STATIC_FINAL.value,
    ProofBaselineMode.FIXED_SCHEDULE.value,
    ProofBaselineMode.LOCKSTEP_REWARD_AB.value,
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
    current_runner_supported: bool


@dataclass(frozen=True, slots=True)
class ProofBaselinePlan:
    """A complete proof-baseline cohort plan for blueprint-health claims."""

    plan_id: str
    cohorts: tuple[ProofBaselineCohort, ...]


__all__ = [
    "ProofBaselineCohort",
    "ProofBaselineMode",
    "ProofBaselinePlan",
    "REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES",
]
