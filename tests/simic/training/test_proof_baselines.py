"""Tests for blueprint-health proof baseline planning."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from esper.simic.rewards import RewardMode
from esper.simic.training.proof_baselines import (
    ProofBaselineMode,
    build_blueprint_health_proof_plan,
    missing_required_baseline_modes,
    train_blueprint_health_proof_baselines,
)
from esper.simic.training.vectorized_trainer import apply_proof_baseline_action_controls


def test_blueprint_health_plan_declares_required_control_modes() -> None:
    plan = build_blueprint_health_proof_plan(
        primary_reward_mode=RewardMode.SHAPED,
        comparison_reward_mode=RewardMode.SIMPLIFIED,
        base_seed=100,
    )

    modes = tuple(cohort.mode for cohort in plan.cohorts)
    assert modes == (
        ProofBaselineMode.OFF_SWITCH,
        ProofBaselineMode.STATIC_INITIAL,
        ProofBaselineMode.STATIC_FINAL,
        ProofBaselineMode.FIXED_SCHEDULE,
        ProofBaselineMode.LOCKSTEP_REWARD_AB,
        ProofBaselineMode.LOCKSTEP_REWARD_AB,
    )
    assert tuple(cohort.cohort_id for cohort in plan.cohorts) == (
        "off_switch",
        "static_initial",
        "static_final",
        "fixed_schedule",
        "lockstep_reward_ab_A",
        "lockstep_reward_ab_B",
    )
    assert plan.cohorts[-2].training_seed == plan.cohorts[-1].training_seed
    assert plan.cohorts[-2].reward_mode == RewardMode.SHAPED.value
    assert plan.cohorts[-1].reward_mode == RewardMode.SIMPLIFIED.value


def test_missing_required_baseline_modes_reports_contract_gaps() -> None:
    missing = missing_required_baseline_modes(
        (
            "off_switch",
            "static_initial",
            "fixed_schedule",
            "lockstep_reward_ab",
        )
    )

    assert missing == ("static_final",)


def test_static_final_and_fixed_schedule_declared_unsupported() -> None:
    """static_final and fixed_schedule are fake (WAIT-only) controls here.

    The runner has no topology-replay machinery (static_final) and no schedule
    injector (fixed_schedule), so both are declared unsupported rather than
    silently degrading to a WAIT-only impostor that invalidates the proof.
    """
    plan = build_blueprint_health_proof_plan()
    support = {cohort.cohort_id: cohort.current_runner_supported for cohort in plan.cohorts}

    assert support["static_final"] is False
    assert support["fixed_schedule"] is False
    # The genuine controls remain supported.
    assert support["off_switch"] is True
    assert support["static_initial"] is True
    assert support["lockstep_reward_ab_A"] is True
    assert support["lockstep_reward_ab_B"] is True


def test_proof_baseline_runner_refuses_unsupported_cohorts() -> None:
    """The plan runner fails loudly when an unsupported cohort is present."""
    plan = build_blueprint_health_proof_plan()

    def fake_train_cohort(**kwargs: object) -> tuple[object, list[dict[str, object]]]:
        raise AssertionError("unsupported cohorts must not reach training")

    with pytest.raises(RuntimeError, match="not supported by this runner"):
        train_blueprint_health_proof_baselines(
            plan=plan,
            train_kwargs={"n_episodes": 1, "device": "cpu"},
            train_cohort=fake_train_cohort,
        )


def test_proof_baseline_runner_invokes_supported_runtime_control_cohorts() -> None:
    plan = build_blueprint_health_proof_plan()
    supported = tuple(c for c in plan.cohorts if c.current_runner_supported)
    supported_plan = replace(plan, cohorts=supported)
    calls: list[dict[str, object]] = []

    def fake_train_cohort(**kwargs: object) -> tuple[object, list[dict[str, object]]]:
        calls.append(kwargs)
        return object(), [{"avg_accuracy": 0.0}]

    results = train_blueprint_health_proof_baselines(
        plan=supported_plan,
        train_kwargs={"n_episodes": 1, "n_envs": 1, "max_epochs": 2, "device": "cpu"},
        train_cohort=fake_train_cohort,
    )

    assert tuple(results) == tuple(cohort.cohort_id for cohort in supported_plan.cohorts)
    assert tuple(call["proof_baseline_mode"] for call in calls) == tuple(
        cohort.mode.value for cohort in supported_plan.cohorts
    )
    assert tuple(call["proof_baseline_lifecycle_policy"] for call in calls) == tuple(
        cohort.lifecycle_policy for cohort in supported_plan.cohorts
    )
    assert calls[-2]["seed"] == calls[-1]["seed"]
    assert calls[-2]["proof_baseline_pair_id"] == calls[-1]["proof_baseline_pair_id"]


@pytest.mark.parametrize(
    "policy",
    (
        "force_wait_only",
        "freeze_initial_topology",
    ),
)
def test_proof_baseline_action_controls_force_wait_for_frozen_lifecycle_policies(
    policy: str,
) -> None:
    masks_batch = {
        "op": torch.ones(2, 6, dtype=torch.bool),
        "slot_by_op": torch.ones(2, 6, 3, dtype=torch.bool),
    }

    apply_proof_baseline_action_controls(
        masks_batch=masks_batch,
        lifecycle_policy=policy,
    )

    assert masks_batch["op"].tolist() == [
        [True, False, False, False, False, False],
        [True, False, False, False, False, False],
    ]


@pytest.mark.parametrize(
    "policy",
    (
        "freeze_replayed_final_topology",
        "apply_declared_lifecycle_schedule",
    ),
)
def test_proof_baseline_action_controls_refuse_fake_wait_only_controls(
    policy: str,
) -> None:
    """static_final / fixed_schedule must fail loudly, not degrade to WAIT-only.

    Masking the op head to WAIT for these policies would produce an impostor
    control (frozen at INITIAL topology, applying no ops), invalidating the
    morphogenesis proof. The action-control layer refuses rather than silently
    degrade.
    """
    masks_batch = {
        "op": torch.ones(2, 6, dtype=torch.bool),
        "slot_by_op": torch.ones(2, 6, 3, dtype=torch.bool),
    }

    with pytest.raises(RuntimeError, match="not supported by this runner"):
        apply_proof_baseline_action_controls(
            masks_batch=masks_batch,
            lifecycle_policy=policy,
        )

    # Masks must be left untouched when the policy is refused.
    assert masks_batch["op"].all()
