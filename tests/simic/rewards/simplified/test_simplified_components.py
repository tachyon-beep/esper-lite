"""Executable spec: SIMPLIFIED reward components."""

from __future__ import annotations

import pytest

from esper.simic.rewards import compute_simplified_reward

from tests.simic.rewards.simplified.harness import (
    LifecycleOp,
    SeedStage,
    seed_info,
    simplified_config,
)


def test_nonterminal_wait_with_seed_is_pbrs_only() -> None:
    config = simplified_config(
        pbrs_weight=0.5,
        gamma=0.9,
        epoch_progress_bonus=0.2,
        disable_terminal_reward=True,
    )

    reward = compute_simplified_reward(
        action=LifecycleOp.WAIT,
        seed_info=seed_info(
            stage=SeedStage.TRAINING,
            epochs_in_stage=2,
            previous_stage=SeedStage.TRAINING,
            previous_epochs_in_stage=1,
        ),
        epoch=1,
        max_epochs=5,
        val_acc=0.0,
        num_contributing_fossilized=0,
        config=config,
    )

    # PBRS: pbrs_weight * (gamma * phi_current - phi_prev)
    # STAGE_POTENTIALS[TRAINING] = 2.0
    # phi_current = 2.0 + (2 * 0.2) = 2.4
    # phi_prev    = 2.0 + (1 * 0.2) = 2.2
    expected_pbrs = 0.5 * (0.9 * 2.4 - 2.2)
    assert reward == pytest.approx(expected_pbrs)


def test_pbrs_requires_seed_info_and_is_skipped_when_disabled() -> None:
    base = dict(
        action=LifecycleOp.WAIT,
        epoch=1,
        max_epochs=5,
        val_acc=0.0,
        num_contributing_fossilized=0,
    )

    reward_no_seed = compute_simplified_reward(
        seed_info=None,
        config=simplified_config(disable_terminal_reward=True),
        **base,
    )
    assert reward_no_seed == pytest.approx(0.0)

    reward_disabled = compute_simplified_reward(
        seed_info=seed_info(stage=SeedStage.TRAINING, epochs_in_stage=2),
        config=simplified_config(disable_pbrs=True, disable_terminal_reward=True),
        **base,
    )
    assert reward_disabled == pytest.approx(0.0)


def test_uniform_intervention_cost_applies_to_any_non_wait_action() -> None:
    config = simplified_config(disable_pbrs=True, disable_terminal_reward=True)

    reward_wait = compute_simplified_reward(
        action=LifecycleOp.WAIT,
        seed_info=None,
        epoch=1,
        max_epochs=5,
        val_acc=0.0,
        num_contributing_fossilized=0,
        config=config,
    )
    assert reward_wait == pytest.approx(0.0)

    reward_germinate = compute_simplified_reward(
        action=LifecycleOp.GERMINATE,
        seed_info=None,
        epoch=1,
        max_epochs=5,
        val_acc=0.0,
        num_contributing_fossilized=0,
        config=config,
    )
    assert reward_germinate == pytest.approx(-0.01)

    reward_prune = compute_simplified_reward(
        action=LifecycleOp.PRUNE,
        seed_info=None,
        epoch=1,
        max_epochs=5,
        val_acc=0.0,
        num_contributing_fossilized=0,
        config=config,
    )
    assert reward_prune == pytest.approx(-0.01)


def test_terminal_bonus_is_linear_in_accuracy_and_contributing_fossils() -> None:
    config = simplified_config(disable_pbrs=True, disable_terminal_reward=False)

    reward = compute_simplified_reward(
        action=LifecycleOp.WAIT,
        seed_info=None,
        epoch=5,
        max_epochs=5,
        val_acc=75.0,
        num_contributing_fossilized=2,
        config=config,
    )

    expected_terminal = (75.0 / 100.0) * 3.0 + 2 * 2.0
    assert reward == pytest.approx(expected_terminal)


def test_terminal_bonus_respects_disable_terminal_reward() -> None:
    config = simplified_config(disable_pbrs=True, disable_terminal_reward=True)

    reward = compute_simplified_reward(
        action=LifecycleOp.WAIT,
        seed_info=None,
        epoch=5,
        max_epochs=5,
        val_acc=99.0,
        num_contributing_fossilized=3,
        config=config,
    )
    assert reward == pytest.approx(0.0)


def test_terminal_bonus_stacks_with_intervention_cost() -> None:
    config = simplified_config(disable_pbrs=True, disable_terminal_reward=False)

    reward = compute_simplified_reward(
        action=LifecycleOp.PRUNE,
        seed_info=None,
        epoch=5,
        max_epochs=5,
        val_acc=100.0,
        num_contributing_fossilized=1,
        config=config,
    )

    expected = (100.0 / 100.0) * 3.0 + 1 * 2.0 - 0.01
    assert reward == pytest.approx(expected)


def test_pbrs_is_independent_of_improvement_and_params_fields() -> None:
    config = simplified_config(
        pbrs_weight=1.0,
        gamma=0.9,
        epoch_progress_bonus=0.2,
        disable_terminal_reward=True,
    )

    baseline = seed_info(
        stage=SeedStage.TRAINING,
        epochs_in_stage=2,
        previous_stage=SeedStage.TRAINING,
        previous_epochs_in_stage=1,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        seed_params=0,
    )
    perturbed = seed_info(
        stage=SeedStage.TRAINING,
        epochs_in_stage=2,
        previous_stage=SeedStage.TRAINING,
        previous_epochs_in_stage=1,
        improvement_since_stage_start=123.0,
        total_improvement=-999.0,
        seed_params=65_535,
    )

    reward_baseline = compute_simplified_reward(
        action=LifecycleOp.WAIT,
        seed_info=baseline,
        epoch=1,
        max_epochs=5,
        val_acc=0.0,
        num_contributing_fossilized=0,
        config=config,
    )
    reward_perturbed = compute_simplified_reward(
        action=LifecycleOp.WAIT,
        seed_info=perturbed,
        epoch=1,
        max_epochs=5,
        val_acc=0.0,
        num_contributing_fossilized=0,
        config=config,
    )

    assert reward_baseline == pytest.approx(reward_perturbed)

