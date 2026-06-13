"""Blueprint-health proof baseline contracts.

These contracts make proof cohorts explicit before a run can support
blueprint-health claims. The current PPO runtime does not yet enforce every
control policy here, so the runner fails closed until those controls are wired.
"""

from __future__ import annotations

from typing import NoReturn

from esper.leyline.proof_baselines import (
    REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES,
    ProofBaselineCohort,
    ProofBaselineMode,
    ProofBaselinePlan,
)
from esper.simic.rewards import RewardMode


def missing_required_baseline_modes(observed_modes: tuple[str, ...]) -> tuple[str, ...]:
    """Return required blueprint-health baseline modes absent from observed runs."""
    observed = set(observed_modes)
    return tuple(
        mode
        for mode in REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES
        if mode not in observed
    )


def build_blueprint_health_proof_plan(
    *,
    primary_reward_mode: RewardMode = RewardMode.SHAPED,
    comparison_reward_mode: RewardMode = RewardMode.SIMPLIFIED,
    base_seed: int = 42,
    plan_id: str = "blueprint-health-proof",
) -> ProofBaselinePlan:
    """Build the required control cohorts for blueprint-health proof claims."""
    return ProofBaselinePlan(
        plan_id=plan_id,
        cohorts=(
            ProofBaselineCohort(
                cohort_id="off_switch",
                mode=ProofBaselineMode.OFF_SWITCH,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 10_000,
                lifecycle_policy="force_wait_only",
                proof_baseline_pair_id=plan_id,
                current_runner_supported=False,
            ),
            ProofBaselineCohort(
                cohort_id="static_initial",
                mode=ProofBaselineMode.STATIC_INITIAL,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 20_000,
                lifecycle_policy="freeze_initial_topology",
                proof_baseline_pair_id=plan_id,
                current_runner_supported=False,
            ),
            ProofBaselineCohort(
                cohort_id="static_final",
                mode=ProofBaselineMode.STATIC_FINAL,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 30_000,
                lifecycle_policy="freeze_replayed_final_topology",
                proof_baseline_pair_id=plan_id,
                current_runner_supported=False,
            ),
            ProofBaselineCohort(
                cohort_id="fixed_schedule",
                mode=ProofBaselineMode.FIXED_SCHEDULE,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 40_000,
                lifecycle_policy="apply_declared_lifecycle_schedule",
                proof_baseline_pair_id=plan_id,
                current_runner_supported=False,
            ),
            ProofBaselineCohort(
                cohort_id="lockstep_reward_ab_A",
                mode=ProofBaselineMode.LOCKSTEP_REWARD_AB,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 50_000,
                lifecycle_policy="paired_lockstep_reward_comparison",
                proof_baseline_pair_id=plan_id,
                current_runner_supported=False,
            ),
            ProofBaselineCohort(
                cohort_id="lockstep_reward_ab_B",
                mode=ProofBaselineMode.LOCKSTEP_REWARD_AB,
                reward_mode=comparison_reward_mode.value,
                training_seed=base_seed + 50_000,
                lifecycle_policy="paired_lockstep_reward_comparison",
                proof_baseline_pair_id=plan_id,
                current_runner_supported=False,
            ),
        ),
    )


def train_blueprint_health_proof_baselines(
    *,
    plan: ProofBaselinePlan,
) -> NoReturn:
    """Fail closed until proof-control policies are enforced by the PPO runtime."""
    unsupported = tuple(
        cohort.mode.value
        for cohort in plan.cohorts
        if not cohort.current_runner_supported
    )
    raise NotImplementedError(
        "Blueprint-health proof baselines are not wired into the PPO runtime: "
        + ", ".join(unsupported)
    )


__all__ = [
    "ProofBaselineCohort",
    "ProofBaselineMode",
    "ProofBaselinePlan",
    "REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES",
    "build_blueprint_health_proof_plan",
    "missing_required_baseline_modes",
    "train_blueprint_health_proof_baselines",
]
