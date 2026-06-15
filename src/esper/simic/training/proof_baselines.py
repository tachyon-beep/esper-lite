"""Blueprint-health proof baseline contracts.

These contracts make proof cohorts explicit before a run can support
blueprint-health claims. The runner passes each cohort's lifecycle policy into
the PPO runtime so controls are enforced before action sampling.
"""

from __future__ import annotations

from typing import Any, Callable

from esper.leyline.proof_baselines import (
    FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT,
    FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
    FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
    FIXED_SCHEDULE_GERMINATE_R0C0_V1,
    REQUIRED_BLUEPRINT_HEALTH_BASELINE_MODE_VALUES,
    STATIC_FINAL_SOURCE_COHORT_ID,
    STATIC_FINAL_SOURCE_LIFECYCLE_POLICY,
    STATIC_FINAL_SOURCE_MODE,
    STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT,
    STATIC_FINAL_SOURCE_TOPOLOGY_HASH,
    STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS,
    STATIC_FINAL_SOURCE_TOPOLOGY_VERSION,
    STATIC_FINAL_SOURCE_TOPOLOGY_V1,
    ProofBaselineCohort,
    ProofBaselineMode,
    ProofBaselinePlan,
    StaticFinalSourceManifestRef,
)
from esper.leyline.telemetry import TopologyManifestPayload
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
                proof_baseline_schedule_id=None,
                proof_baseline_schedule_hash=None,
                proof_baseline_schedule_version=None,
                proof_baseline_schedule_action_count=None,
                current_runner_supported=True,
            ),
            ProofBaselineCohort(
                cohort_id="static_initial",
                mode=ProofBaselineMode.STATIC_INITIAL,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 20_000,
                lifecycle_policy="freeze_initial_topology",
                proof_baseline_pair_id=plan_id,
                proof_baseline_schedule_id=None,
                proof_baseline_schedule_hash=None,
                proof_baseline_schedule_version=None,
                proof_baseline_schedule_action_count=None,
                current_runner_supported=True,
            ),
            # static_final is supported only through the runner's source
            # preflight: evolve a final topology, capture its manifest, replay
            # it into this cohort, then freeze lifecycle changes after replay
            # validation.
            ProofBaselineCohort(
                cohort_id="static_final",
                mode=ProofBaselineMode.STATIC_FINAL,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 30_000,
                lifecycle_policy="freeze_replayed_final_topology",
                proof_baseline_pair_id=plan_id,
                proof_baseline_schedule_id=None,
                proof_baseline_schedule_hash=None,
                proof_baseline_schedule_version=None,
                proof_baseline_schedule_action_count=None,
                current_runner_supported=True,
            ),
            ProofBaselineCohort(
                cohort_id="fixed_schedule",
                mode=ProofBaselineMode.FIXED_SCHEDULE,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 40_000,
                lifecycle_policy="apply_declared_lifecycle_schedule",
                proof_baseline_pair_id=plan_id,
                proof_baseline_schedule_id=FIXED_SCHEDULE_GERMINATE_R0C0_V1,
                proof_baseline_schedule_hash=FIXED_SCHEDULE_GERMINATE_R0C0_HASH,
                proof_baseline_schedule_version=FIXED_SCHEDULE_GERMINATE_R0C0_VERSION,
                proof_baseline_schedule_action_count=(
                    FIXED_SCHEDULE_GERMINATE_R0C0_ACTION_COUNT
                ),
                current_runner_supported=True,
            ),
            ProofBaselineCohort(
                cohort_id="lockstep_reward_ab_A",
                mode=ProofBaselineMode.LOCKSTEP_REWARD_AB,
                reward_mode=primary_reward_mode.value,
                training_seed=base_seed + 50_000,
                lifecycle_policy="paired_lockstep_reward_comparison",
                proof_baseline_pair_id=plan_id,
                proof_baseline_schedule_id=None,
                proof_baseline_schedule_hash=None,
                proof_baseline_schedule_version=None,
                proof_baseline_schedule_action_count=None,
                current_runner_supported=True,
            ),
            ProofBaselineCohort(
                cohort_id="lockstep_reward_ab_B",
                mode=ProofBaselineMode.LOCKSTEP_REWARD_AB,
                reward_mode=comparison_reward_mode.value,
                training_seed=base_seed + 50_000,
                lifecycle_policy="paired_lockstep_reward_comparison",
                proof_baseline_pair_id=plan_id,
                proof_baseline_schedule_id=None,
                proof_baseline_schedule_hash=None,
                proof_baseline_schedule_version=None,
                proof_baseline_schedule_action_count=None,
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
        static_final_source_ref = None
        if cohort.mode == ProofBaselineMode.STATIC_FINAL:
            source_result = _train_static_final_source_cohort(
                cohort=cohort,
                train_kwargs=train_kwargs,
                train_cohort=train_cohort,
            )
            results[STATIC_FINAL_SOURCE_COHORT_ID] = source_result
            static_final_source_ref = _extract_static_final_source_manifest(
                source_result
            )

        cohort_kwargs = dict(train_kwargs)
        cohort_kwargs["group_id"] = cohort.cohort_id
        cohort_kwargs["seed"] = cohort.training_seed
        cohort_kwargs["reward_mode"] = cohort.reward_mode
        cohort_kwargs["proof_baseline_mode"] = cohort.mode.value
        cohort_kwargs["proof_baseline_pair_id"] = cohort.proof_baseline_pair_id
        cohort_kwargs["proof_baseline_lifecycle_policy"] = cohort.lifecycle_policy
        cohort_kwargs["proof_baseline_schedule_id"] = cohort.proof_baseline_schedule_id
        cohort_kwargs["proof_baseline_schedule_hash"] = (
            cohort.proof_baseline_schedule_hash
        )
        cohort_kwargs["proof_baseline_schedule_version"] = (
            cohort.proof_baseline_schedule_version
        )
        cohort_kwargs["proof_baseline_schedule_action_count"] = (
            cohort.proof_baseline_schedule_action_count
        )
        if static_final_source_ref is not None:
            cohort_kwargs["static_final_source_manifest"] = (
                static_final_source_ref.payload
            )
            cohort_kwargs["static_final_source_run_dir"] = static_final_source_ref.run_dir
            cohort_kwargs["static_final_source_group_id"] = (
                static_final_source_ref.group_id
            )
            cohort_kwargs["static_final_source_episode_idx"] = (
                static_final_source_ref.episode_idx
            )
            cohort_kwargs["static_final_source_event_id"] = (
                static_final_source_ref.event_id
            )
        results[cohort.cohort_id] = train_cohort(**cohort_kwargs)

    return results


def _train_static_final_source_cohort(
    *,
    cohort: ProofBaselineCohort,
    train_kwargs: dict[str, Any],
    train_cohort: Callable[..., tuple[Any, list[dict[str, Any]]]],
) -> tuple[Any, list[dict[str, Any]]]:
    source_kwargs = dict(train_kwargs)
    source_kwargs["group_id"] = STATIC_FINAL_SOURCE_COHORT_ID
    source_kwargs["seed"] = cohort.training_seed
    source_kwargs["reward_mode"] = cohort.reward_mode
    source_kwargs["proof_baseline_mode"] = STATIC_FINAL_SOURCE_MODE
    source_kwargs["proof_baseline_pair_id"] = cohort.proof_baseline_pair_id
    source_kwargs["proof_baseline_lifecycle_policy"] = (
        STATIC_FINAL_SOURCE_LIFECYCLE_POLICY
    )
    source_kwargs["proof_baseline_schedule_id"] = STATIC_FINAL_SOURCE_TOPOLOGY_V1
    source_kwargs["proof_baseline_schedule_hash"] = STATIC_FINAL_SOURCE_TOPOLOGY_HASH
    source_kwargs["proof_baseline_schedule_version"] = (
        STATIC_FINAL_SOURCE_TOPOLOGY_VERSION
    )
    source_kwargs["proof_baseline_schedule_action_count"] = (
        STATIC_FINAL_SOURCE_TOPOLOGY_ACTION_COUNT
    )
    source_kwargs["n_envs"] = 1
    source_kwargs["n_episodes"] = 1
    if "max_epochs" in source_kwargs:
        source_kwargs["max_epochs"] = max(
            source_kwargs["max_epochs"],
            STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS,
        )
    else:
        source_kwargs["max_epochs"] = STATIC_FINAL_SOURCE_TOPOLOGY_MIN_EPOCHS
    return train_cohort(**source_kwargs)


def _extract_static_final_source_manifest(
    source_result: tuple[Any, list[dict[str, Any]]],
) -> StaticFinalSourceManifestRef:
    _agent, history = source_result
    source_records: list[dict[str, Any]] = []
    for batch in history:
        try:
            topology_manifests = batch["topology_manifests"]
        except KeyError as exc:
            raise RuntimeError(
                "Static-final source run did not return topology manifest history"
            ) from exc
        for record in topology_manifests:
            payload = TopologyManifestPayload.from_dict(record["payload"])
            if payload.manifest_role == "source_final":
                source_records.append(record)

    if len(source_records) != 1:
        raise RuntimeError(
            "Static-final source run must return exactly one source-final topology "
            f"manifest, got {len(source_records)}"
        )

    record = source_records[0]
    payload = TopologyManifestPayload.from_dict(record["payload"])
    run_dir = record["run_dir"]
    if run_dir is None or run_dir == "":
        raise RuntimeError(
            "Static-final source manifest is missing run_dir; run the source cohort "
            "with telemetry_dir so Karn can join the source/replay evidence."
        )
    return StaticFinalSourceManifestRef(
        payload=payload,
        run_dir=run_dir,
        group_id=record["group_id"],
        episode_idx=record["episode_idx"],
        event_id=record["event_id"],
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
