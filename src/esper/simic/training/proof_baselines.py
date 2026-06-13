"""Blueprint-health proof baseline contracts.

These contracts make proof cohorts explicit before a run can support
blueprint-health claims. The runner passes each cohort's lifecycle policy into
the PPO runtime so controls are enforced before action sampling.
"""

from __future__ import annotations

from typing import Any, Callable

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
                current_runner_supported=True,
            ),
            ProofBaselineCohort(
                cohort_id="static_initial",
                mode=ProofBaselineMode.STATIC_INITIAL,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 20_000,
                lifecycle_policy="freeze_initial_topology",
                proof_baseline_pair_id=plan_id,
                current_runner_supported=True,
            ),
            # static_final requires replaying a previously-evolved FINAL
            # topology and holding it fixed. The runtime has no topology
            # persistence/replay machinery, so forcing WAIT-only from epoch 0
            # would leave the run at its INITIAL topology — a fake control
            # indistinguishable from static_initial. Mark unsupported rather
            # than ship a silently-WAIT-only impostor control.
            ProofBaselineCohort(
                cohort_id="static_final",
                mode=ProofBaselineMode.STATIC_FINAL,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 30_000,
                lifecycle_policy="freeze_replayed_final_topology",
                proof_baseline_pair_id=plan_id,
                current_runner_supported=False,
            ),
            # fixed_schedule requires applying morphogenesis ops on a
            # predetermined (non-policy) schedule. The runtime has no schedule
            # injection machinery — it can only mask the op head to WAIT — so a
            # "scheduled" cohort would in fact apply no ops at all. Mark
            # unsupported rather than ship a silently-WAIT-only impostor control.
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
                current_runner_supported=True,
            ),
            ProofBaselineCohort(
                cohort_id="lockstep_reward_ab_B",
                mode=ProofBaselineMode.LOCKSTEP_REWARD_AB,
                reward_mode=comparison_reward_mode.value,
                training_seed=base_seed + 50_000,
                lifecycle_policy="paired_lockstep_reward_comparison",
                proof_baseline_pair_id=plan_id,
                current_runner_supported=True,
            ),
        ),
    )


def train_blueprint_health_proof_baselines(
    *,
    plan: ProofBaselinePlan,
    train_kwargs: dict[str, Any] | None = None,
    train_cohort: Callable[..., tuple[Any, list[dict[str, Any]]]] | None = None,
) -> dict[str, tuple[Any, list[dict[str, Any]]]]:
    return _train_blueprint_health_proof_baselines(
        plan=plan,
        train_kwargs={} if train_kwargs is None else train_kwargs,
        train_cohort=train_cohort,
    )


def _train_blueprint_health_proof_baselines(
    *,
    plan: ProofBaselinePlan,
    train_kwargs: dict[str, Any],
    train_cohort: Callable[..., tuple[Any, list[dict[str, Any]]]] | None,
) -> dict[str, tuple[Any, list[dict[str, Any]]]]:
    """Train each blueprint-health proof cohort with its runtime controls."""
    unsupported = tuple(
        cohort.mode.value
        for cohort in plan.cohorts
        if not cohort.current_runner_supported
    )
    if unsupported:
        raise RuntimeError(
            "Blueprint-health proof baselines are not supported by this runner: "
            + ", ".join(unsupported)
        )

    if train_cohort is None:
        from esper.simic.training.vectorized import train_ppo_vectorized

        train_cohort = train_ppo_vectorized

    results: dict[str, tuple[Any, list[dict[str, Any]]]] = {}
    for cohort in plan.cohorts:
        cohort_kwargs = dict(train_kwargs)
        cohort_kwargs["group_id"] = cohort.cohort_id
        cohort_kwargs["seed"] = cohort.training_seed
        cohort_kwargs["reward_mode"] = cohort.reward_mode
        cohort_kwargs["proof_baseline_mode"] = cohort.mode.value
        cohort_kwargs["proof_baseline_pair_id"] = cohort.proof_baseline_pair_id
        cohort_kwargs["proof_baseline_lifecycle_policy"] = cohort.lifecycle_policy
        results[cohort.cohort_id] = train_cohort(**cohort_kwargs)

    return results


__all__ = [
    "ProofBaselineCohort",
    "ProofBaselineMode",
    "ProofBaselinePlan",
    "REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES",
    "build_blueprint_health_proof_plan",
    "missing_required_baseline_modes",
    "train_blueprint_health_proof_baselines",
]
