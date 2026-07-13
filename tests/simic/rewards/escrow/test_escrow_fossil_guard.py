"""Fail-closed ESCROW-at-fossilization guard + the H4 balance-mutation sequence.

Design context: `docs/analysis/2026-07-14-advantage-loo-read-preregistration.md` (round-11)
and PDR-0074/0075. A FOSSILIZED seed's counterfactual contribution is structurally
unmeasured (fossils are excluded from the ablation in `vectorized_trainer.py`). ESCROW
claws back accrued credit on a *measured loss of value*. "Missing observability must not
be interpreted as a measured loss of value."

These tests pin two things:

1. THE GUARD (fail-closed): constructing a `RewardMode.ESCROW` config without an explicit
   `escrow_fossil_settlement` strategy raises at construction. The opt-in unblocks it.

2. THE H4 SEQUENCE (documents current behaviour, and the LATENT hazard the guard exists
   for): the synthetic [3,3,3,None] / [HOLDING x3, FOSSILIZED] path FREEZES accrued credit
   (no clawback today); a *measured host-damage negative* is what actually triggers the
   catastrophic clawback + negative payment. H4 did NOT confirm a clawback on the current
   None-path, so the guard is a construction-time raise, NOT (yet) a hard error.
"""

from __future__ import annotations

import pytest

from esper.leyline import LifecycleOp, SeedStage
from esper.simic.rewards import (
    ContributionRewardConfig,
    RewardMode,
    compute_contribution_reward,
)
from esper.simic.rewards.types import SeedInfo


# ---------------------------------------------------------------------------
# 1. The fail-closed guard
# ---------------------------------------------------------------------------
def test_escrow_without_settlement_strategy_raises() -> None:
    """Fail-closed: ESCROW + default (unconfigured) settlement raises at construction."""
    with pytest.raises(ValueError, match="missing observability"):
        ContributionRewardConfig(reward_mode=RewardMode.ESCROW)


def test_escrow_with_settlement_optin_constructs() -> None:
    """The explicit opt-in unblocks ESCROW construction (scoped escape hatch)."""
    cfg = ContributionRewardConfig(
        reward_mode=RewardMode.ESCROW, escrow_fossil_settlement="configured"
    )
    assert cfg.reward_mode == RewardMode.ESCROW
    assert cfg.escrow_fossil_settlement == "configured"


def test_non_escrow_modes_unaffected() -> None:
    """The guard is scoped to ESCROW; other modes construct with the default field."""
    for mode in (
        RewardMode.SHAPED,
        RewardMode.BASIC,
        RewardMode.SPARSE,
        RewardMode.SIMPLIFIED,
    ):
        cfg = ContributionRewardConfig(reward_mode=mode)
        assert cfg.escrow_fossil_settlement == "unconfigured"


# ---------------------------------------------------------------------------
# 2. The H4 balance-mutation sequence
# ---------------------------------------------------------------------------
def _escrow_cfg() -> ContributionRewardConfig:
    """Isolated escrow config: shaping zeroed so reward == escrow bounded_attribution."""
    return ContributionRewardConfig(
        reward_mode=RewardMode.ESCROW,
        escrow_fossil_settlement="configured",
        contribution_weight=1.0,
        disable_pbrs=True,
        disable_terminal_reward=True,
        disable_anti_gaming=True,
        rent_weight=0.0,
        alpha_shock_coef=0.0,
        germinate_cost=0.0,
        fossilize_cost=0.0,
        prune_cost=0.0,
        set_alpha_target_cost=0.0,
        seed_occupancy_cost=0.0,
        fossilized_maintenance_cost=0.0,
        free_slots=99,
        fossilize_base_bonus=0.0,
        fossilize_contribution_scale=0.0,
        fossilize_noncontributing_penalty=0.0,
        invalid_fossilize_penalty=0.0,
    )


def _seed(stage: SeedStage, improvement: float) -> SeedInfo:
    return SeedInfo(
        stage=stage.value,
        improvement_since_stage_start=0.0,
        total_improvement=improvement,
        epochs_in_stage=3,
        seed_params=0,
        previous_stage=SeedStage.HOLDING.value,
        previous_epochs_in_stage=2,
        seed_age_epochs=10,
        counterfactual_total_improvement=improvement,
    )


def _step(cfg, action, contribution, stage, credit_prev, improvement=3.0):
    _, comp = compute_contribution_reward(
        action=action,
        seed_contribution=contribution,
        val_acc=80.0,
        seed_info=_seed(stage, improvement),
        epoch=10,
        max_epochs=150,
        total_params=0,
        host_params=1,
        config=cfg,
        acc_at_germination=None,
        acc_delta=0.0,
        return_components=True,
        stable_val_acc=80.0,
        escrow_credit_prev=credit_prev,
    )
    return comp


def test_h4_fossilize_none_path_freezes_no_clawback() -> None:
    """H4: [3,3,3,None]/[HOLDING x3, FOSSILIZED] FREEZES accrued credit (no clawback)."""
    cfg = _escrow_cfg()
    credit = 0.0
    seq = [
        (LifecycleOp.SET_ALPHA_TARGET, 3.0, SeedStage.HOLDING),
        (LifecycleOp.SET_ALPHA_TARGET, 3.0, SeedStage.HOLDING),
        (LifecycleOp.FOSSILIZE, 3.0, SeedStage.HOLDING),      # staleness==0 at commit
        (LifecycleOp.WAIT, None, SeedStage.FOSSILIZED),        # first post-fossilize epoch
    ]
    deltas = []
    for action, contribution, stage in seq:
        comp = _step(cfg, action, contribution, stage, credit)
        deltas.append(comp.escrow_delta)
        credit = comp.escrow_credit_next

    # Credit accrued during HOLDING, then held flat across FOSSILIZE and the post step.
    assert credit == pytest.approx(1.5)
    # The FOSSILIZE step and the first post-fossilize (None) step are FREEZES: no clawback.
    assert deltas[2] == pytest.approx(0.0)  # FOSSILIZE decision
    assert deltas[3] == pytest.approx(0.0)  # None @ FOSSILIZED
    # No negative delta anywhere in the current-path sequence (no clawback).
    assert min(deltas) >= 0.0


def test_h4_measured_negative_triggers_latent_clawback() -> None:
    """H4 (the hazard the guard exists for): a MEASURED host-damage negative claws back
    the accrued credit AND pays contribution_weight * negative."""
    cfg = _escrow_cfg()
    credit = 1.5  # accrued during HOLDING
    comp = _step(
        cfg, LifecycleOp.WAIT, -8.0, SeedStage.HOLDING, credit, improvement=-8.0
    )
    # Credit target zeroed => full clawback of the accrued 1.5.
    assert comp.escrow_credit_target == pytest.approx(0.0)
    assert comp.escrow_delta == pytest.approx(-1.5)
    # bounded_attribution = weight * (-8) + clawback(-1.5) = -9.5 (the catastrophic misread).
    assert comp.bounded_attribution == pytest.approx(-9.5)
    assert comp.escrow_credit_next == pytest.approx(0.0)
