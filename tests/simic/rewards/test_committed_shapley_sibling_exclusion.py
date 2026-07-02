"""F1/F8 sibling-channel mutual exclusion (WI-6).

When shapley_synergy_scale > 0 the two live-but-near-dormant sibling synergy
channels must not pay — the Committed-Shapley top-up REPLACES them (the
reviewer's double-pay guard; the new term's incentive could otherwise WAKE
them into feedback double-pay). At scale=0 (default) both behave exactly as
today.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from esper.leyline import DEFAULT_MIN_FOSSILIZE_CONTRIBUTION, LifecycleOp, SeedStage
from esper.simic.rewards import (
    ContributionRewardConfig,
    RewardComponentsTelemetry,
    SeedInfo,
    compute_contribution_reward,
)
from esper.simic.training.handlers import HandlerContext, execute_fossilize
from esper.simic.training.parallel_env_state import ParallelEnvState

_TOPUP_ON = dict(
    shapley_synergy_scale=0.5,
    shapley_synergy_cap=3.0,
    shapley_synergy_normalized_cap=5.0,
)


def _interaction_seed_info() -> SeedInfo:
    return SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=0.05,
        total_improvement=0.05,
        epochs_in_stage=3,
        seed_params=10000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=5,
        seed_age_epochs=8,
        interaction_sum=2.5,
        boost_received=1.2,
        counterfactual_total_improvement=0.05,
    )


def _reward(config: ContributionRewardConfig) -> tuple[float, RewardComponentsTelemetry]:
    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.05,
        val_acc=70.0,
        seed_info=_interaction_seed_info(),
        epoch=10,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
        return_components=True,
    )
    return reward, components


def test_interaction_bonus_paid_at_scale_zero():
    _, components = _reward(ContributionRewardConfig())
    assert components.interaction_bonus > 0.0


def test_interaction_bonus_not_paid_when_topup_on():
    reward_off, comp_off = _reward(ContributionRewardConfig())
    reward_on, comp_on = _reward(ContributionRewardConfig(**_TOPUP_ON))
    assert comp_on.interaction_bonus == 0.0
    assert reward_on == pytest.approx(reward_off - comp_off.interaction_bonus)


def _fossilize_ctx(shapley_synergy_scale: float) -> HandlerContext:
    env_state = MagicMock(spec=ParallelEnvState)
    env_state.seeds_fossilized = 0
    env_state.fossilize_count = 0
    env_state.contributing_fossilized = 0
    env_state.acc_at_germination = {"r0c0": 0.5}
    # Scaffold r0c1 boosted THIS seed (r0c0, the beneficiary) at epoch 5 ->
    # fossilizing r0c0 accrues hindsight credit toward r0c1's ledger entries.
    env_state.scaffold_boost_ledger = {"r0c1": [(1.5, "r0c0", 5)]}
    env_state.pending_hindsight_credit = 0.0
    env_state.seed_optimizers = {"r0c0": object()}
    env_state.needs_governor_snapshot = False

    seed_state = MagicMock()
    seed_state.stage = SeedStage.HOLDING

    return HandlerContext(
        env_idx=0,
        slot_id="r0c0",
        env_state=env_state,
        model=MagicMock(),
        slot=MagicMock(),
        seed_state=seed_state,
        epoch=12,
        max_epochs=150,
        episodes_completed=0,
        shapley_synergy_scale=shapley_synergy_scale,
    )


def _fossilize(ctx: HandlerContext) -> None:
    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.2,
        total_improvement=DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
        epochs_in_stage=3,
    )
    result = execute_fossilize(ctx, seed_info, lambda _model, _slot_id: True)
    assert result.success is True


def test_hindsight_credit_accrued_at_scale_zero():
    ctx = _fossilize_ctx(shapley_synergy_scale=0.0)
    _fossilize(ctx)
    assert ctx.env_state.pending_hindsight_credit > 0.0


def test_hindsight_credit_not_accrued_when_topup_on():
    ctx = _fossilize_ctx(shapley_synergy_scale=0.5)
    _fossilize(ctx)
    assert ctx.env_state.pending_hindsight_credit == 0.0
