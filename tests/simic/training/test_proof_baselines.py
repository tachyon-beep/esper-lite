"""Tests for blueprint-health proof baseline planning."""

from __future__ import annotations

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


def test_proof_baseline_runner_invokes_all_runtime_control_cohorts() -> None:
    plan = build_blueprint_health_proof_plan()
    calls: list[dict[str, object]] = []

    def fake_train_cohort(**kwargs: object) -> tuple[object, list[dict[str, object]]]:
        calls.append(kwargs)
        return object(), [{"avg_accuracy": 0.0}]

    results = train_blueprint_health_proof_baselines(
        plan=plan,
        train_kwargs={"n_episodes": 1, "n_envs": 1, "max_epochs": 2, "device": "cpu"},
        train_cohort=fake_train_cohort,
    )

    assert tuple(results) == tuple(cohort.cohort_id for cohort in plan.cohorts)
    assert tuple(call["proof_baseline_mode"] for call in calls) == tuple(
        cohort.mode.value for cohort in plan.cohorts
    )
    assert tuple(call["proof_baseline_lifecycle_policy"] for call in calls) == tuple(
        cohort.lifecycle_policy for cohort in plan.cohorts
    )
    assert calls[-2]["seed"] == calls[-1]["seed"]
    assert calls[-2]["proof_baseline_pair_id"] == calls[-1]["proof_baseline_pair_id"]


@pytest.mark.parametrize(
    "policy",
    (
        "force_wait_only",
        "freeze_initial_topology",
        "freeze_replayed_final_topology",
        "apply_declared_lifecycle_schedule",
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
