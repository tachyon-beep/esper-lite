"""Executable spec: SIMPLIFIED dispatcher wiring."""

from __future__ import annotations

import pytest

from esper.simic.rewards import compute_reward

from tests.simic.rewards.simplified.harness import (
    LifecycleOp,
    SeedStage,
    seed_info,
    simplified_config,
)


def test_compute_reward_dispatches_simplified_and_ignores_seed_contribution() -> None:
    config = simplified_config(disable_pbrs=True, disable_terminal_reward=True)

    reward = compute_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=10.0,  # SHAPED would pay attribution here
        val_acc=60.0,
        seed_info=seed_info(stage=SeedStage.HOLDING, total_improvement=-10.0, epochs_in_stage=50),
        epoch=1,
        max_epochs=5,
        total_params=500_000,
        host_params=500_000,
        acc_at_germination=50.0,
        acc_delta=0.5,
        num_contributing_fossilized=0,
        config=config,
    )

    assert reward == pytest.approx(0.0)


def test_compute_reward_simplified_return_components_sets_base_fields() -> None:
    config = simplified_config(disable_pbrs=True, disable_terminal_reward=True)
    info = seed_info(stage=SeedStage.TRAINING, epochs_in_stage=2)

    reward, components = compute_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=42.0,
        seed_info=info,
        epoch=3,
        max_epochs=5,
        total_params=100,
        host_params=100,
        acc_at_germination=None,
        acc_delta=1.25,
        num_contributing_fossilized=0,
        config=config,
        return_components=True,
    )

    assert components.total_reward == pytest.approx(reward)
    assert components.action_name == "WAIT"
    assert components.epoch == 3
    assert components.seed_stage == info.stage
    assert components.val_acc == pytest.approx(42.0)
    assert components.base_acc_delta == pytest.approx(1.25)

