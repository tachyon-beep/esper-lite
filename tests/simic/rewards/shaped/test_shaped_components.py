"""Executable spec: SHAPED warnings, costs, and secondary components."""

from __future__ import annotations

import math

import pytest

from esper.simic.rewards import compute_contribution_reward

from tests.simic.rewards.shaped.harness import (
    LifecycleOp,
    SeedStage,
    seed_info,
    shaped_config,
)


def test_blending_warning_escalates_with_negative_trajectory() -> None:
    config = shaped_config()

    def get_warning(epochs_in_stage: int) -> float:
        _, components = compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=None,  # keep attribution at 0
            val_acc=0.0,
            seed_info=seed_info(
                stage=SeedStage.BLENDING,
                total_improvement=-1.0,
                epochs_in_stage=epochs_in_stage,
            ),
            epoch=1,
            max_epochs=10,
            config=config,
            return_components=True,
        )
        return components.blending_warning

    assert get_warning(1) == pytest.approx(-0.15)  # -0.1 - 0.05
    assert get_warning(3) == pytest.approx(-0.25)  # -0.1 - 0.15
    assert get_warning(6) == pytest.approx(-0.4)  # cap: -0.1 - 0.3


def test_holding_warning_applies_only_to_positive_attribution_wait() -> None:
    config = shaped_config()

    # Bound attribution to a constant (progress unknown => 0.5 * contribution).
    seed_contribution = 2.0  # attributed = 1.0
    base_attribution = 1.0

    def get_components(epochs_in_stage: int):
        return compute_contribution_reward(
            action=LifecycleOp.WAIT,
            seed_contribution=seed_contribution,
            val_acc=70.0,
            seed_info=seed_info(
                stage=SeedStage.HOLDING,
                total_improvement=1.0,
                epochs_in_stage=epochs_in_stage,
            ),
            epoch=10,
            max_epochs=25,
            acc_at_germination=None,
            config=config,
            return_components=True,
        )[1]

    # Grace period.
    assert get_components(1).holding_warning == 0.0

    # Linear schedule: epoch 2: -0.1, epoch 3: -0.15, epoch 4: -0.2.
    assert get_components(2).holding_warning == pytest.approx(-0.1)
    assert get_components(3).holding_warning == pytest.approx(-0.15)
    assert get_components(4).holding_warning == pytest.approx(-0.2)

    components = get_components(3)
    assert components.bounded_attribution == pytest.approx(base_attribution)
    assert components.total_reward == pytest.approx(base_attribution + components.holding_warning)


def test_synergy_bonus_is_bounded_and_gated() -> None:
    config = shaped_config()

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=2.0,  # attributed = 1.0 (progress unknown)
        val_acc=70.0,
        seed_info=seed_info(
            stage=SeedStage.BLENDING,
            total_improvement=1.0,
            interaction_sum=2.0,
        ),
        epoch=5,
        max_epochs=25,
        acc_at_germination=None,
        config=config,
        return_components=True,
    )

    expected_synergy = math.tanh(2.0 * 0.5) * 0.1
    assert components.synergy_bonus == pytest.approx(expected_synergy)
    assert 0.0 < components.synergy_bonus <= 0.1
    assert reward == pytest.approx(components.total_reward)


def test_compute_rent_penalty_scales_by_episode_horizon() -> None:
    config = shaped_config(rent_weight=1.0, rent_host_params_floor=1)

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=0.0,
        seed_info=None,
        epoch=1,
        max_epochs=10,
        total_params=200,
        host_params=100,
        config=config,
        return_components=True,
    )

    growth_ratio = (200 - 100) / 100
    expected_rent = math.log(1.0 + growth_ratio) / 10
    assert components.growth_ratio == pytest.approx(growth_ratio)
    assert components.compute_rent == pytest.approx(-expected_rent)
    assert reward == pytest.approx(-expected_rent)


def test_alpha_shock_applies_only_when_anti_gaming_enabled() -> None:
    config = shaped_config(alpha_shock_coef=2.0, disable_anti_gaming=False)

    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=0.0,
        seed_info=None,
        epoch=1,
        max_epochs=10,
        alpha_delta_sq_sum=0.25,
        config=config,
        return_components=True,
    )

    assert components.alpha_shock == pytest.approx(-0.5)
    assert reward == pytest.approx(-0.5)


def test_germinate_pbrs_deposit_is_suppressed_on_terminal_step() -> None:
    config = shaped_config(disable_pbrs=False, pbrs_weight=1.0)
    expected_deposit = config.gamma * 1.0  # gamma * phi(GERMINATED), phi(no-seed)=0.0

    nonterminal_reward, nonterminal = compute_contribution_reward(
        action=LifecycleOp.GERMINATE,
        seed_contribution=None,
        val_acc=0.0,
        seed_info=None,
        epoch=1,
        max_epochs=10,
        acc_at_germination=None,
        config=config,
        return_components=True,
    )

    assert nonterminal.action_shaping == pytest.approx(expected_deposit)
    assert nonterminal_reward == pytest.approx(expected_deposit)

    terminal_reward, terminal = compute_contribution_reward(
        action=LifecycleOp.GERMINATE,
        seed_contribution=None,
        val_acc=0.0,
        seed_info=None,
        epoch=10,
        max_epochs=10,
        acc_at_germination=None,
        config=config,
        return_components=True,
    )

    assert terminal.action_shaping == pytest.approx(0.0)
    assert terminal_reward == pytest.approx(0.0)
