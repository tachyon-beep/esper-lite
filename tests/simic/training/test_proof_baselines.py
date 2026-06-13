"""Tests for blueprint-health proof baseline planning."""

from __future__ import annotations

import pytest

from esper.simic.rewards import RewardMode
from esper.simic.training.proof_baselines import (
    ProofBaselineMode,
    build_blueprint_health_proof_plan,
    missing_required_baseline_modes,
    train_blueprint_health_proof_baselines,
)


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


def test_proof_baseline_runner_rejects_unenforced_runtime_controls() -> None:
    plan = build_blueprint_health_proof_plan()

    with pytest.raises(NotImplementedError, match="not wired into the PPO runtime"):
        train_blueprint_health_proof_baselines(plan=plan)
