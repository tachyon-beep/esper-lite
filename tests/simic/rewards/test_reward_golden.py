"""Golden-value tests for reward outputs."""

from __future__ import annotations

import pytest

from esper.leyline import LifecycleOp, LossRewardConfig, SeedStage
from esper.simic.rewards import (
    ContributionRewardConfig,
    ContributionRewardInputs,
    LossRewardInputs,
    RewardMode,
    SeedInfo,
    compute_loss_reward,
    compute_reward,
)
from esper.simic.rewards.contribution import compute_contribution_reward


def test_contribution_reward_golden_simplified_pbrs() -> None:
    """Simplified reward: PBRS + action cost stays stable."""
    config = ContributionRewardConfig(reward_mode=RewardMode.SIMPLIFIED)
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=0.0,
        total_improvement=0.0,
        epochs_in_stage=2,
        seed_params=10_000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=1,
        seed_age_epochs=5,
        counterfactual_total_improvement=0.0,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.PRUNE,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=seed_info,
            epoch=5,
            max_epochs=10,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=65.0,
            acc_delta=0.1,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            num_contributing_fossilized=0,
            config=config,
        )
    )

    assert reward == pytest.approx(0.07610000000000011, abs=1e-6)


def test_contribution_reward_golden_basic_mode() -> None:
    """Basic reward: rent only for non-terminal steps, accuracy at terminal.

    Non-terminal WAIT with no seed: only rent penalty applies.
    rent = param_penalty_weight * (effective_seed_params / param_budget)
         = 0.1 * (25_000 / 500_000) = 0.005
    reward = -rent = -0.005
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        basic_acc_delta_weight=5.0,
        param_budget=500_000,
        param_penalty_weight=0.1,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.WAIT,
            seed_contribution=None,
            val_acc=70.0,
            seed_info=None,
            epoch=1,
            max_epochs=10,
            total_params=125_000,
            host_params=100_000,
            acc_at_germination=None,
            acc_delta=2.0,
            committed_val_acc=70.0,
            fossilized_seed_params=0,
            effective_seed_params=25_000,
            config=config,
        )
    )

    # DRL Expert design: accuracy bonus only at terminal epoch
    # Non-terminal = rent only = -0.005
    assert reward == pytest.approx(-0.005, abs=1e-6)


def test_loss_reward_golden_terminal() -> None:
    """Loss reward: normalized loss delta + terminal bonus stays stable."""
    config = LossRewardConfig.default()
    reward = compute_loss_reward(
        LossRewardInputs(
            action=LifecycleOp.WAIT,
            loss_delta=-0.05,
            val_loss=1.5,
            seed_info=None,
            epoch=10,
            max_epochs=10,
            total_params=120,
            host_params=100,
            config=config,
        )
    )

    assert reward == pytest.approx(2.890883922160302, abs=1e-6)


def test_basic_mode_prune_forfeits_pbrs() -> None:
    """PRUNE in BASIC mode forfeits accumulated PBRS potential.

    DRL Expert review 2026-01-12: PRUNE must be net-negative to prevent
    GERMINATE→train→PRUNE gaming. The forfeiture is:
        γ × Φ(no_seed) - Φ(current) = 0.995 × 0 - Φ(TRAINING + progress)
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        param_budget=500_000,
        param_penalty_weight=0.1,
        pbrs_weight=0.3,
        epoch_progress_bonus=0.3,
        gamma=0.995,
    )
    seed_info = SeedInfo(
        stage=SeedStage.TRAINING.value,
        improvement_since_stage_start=1.0,
        total_improvement=2.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.GERMINATED.value,
        previous_epochs_in_stage=1,
        seed_age_epochs=6,
        counterfactual_total_improvement=2.0,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.PRUNE,
            seed_contribution=1.5,
            val_acc=72.0,
            seed_info=seed_info,
            epoch=6,
            max_epochs=150,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=2.0,
            committed_val_acc=72.0,
            fossilized_seed_params=0,
            effective_seed_params=10_000,
            config=config,
        )
    )

    # PBRS forfeiture: -0.3 * (STAGE_POTENTIAL[TRAINING] + min(5*0.3, 2.0))
    # STAGE_POTENTIAL[TRAINING] = 2.0, epoch_bonus = min(1.5, 2.0) = 1.5
    # forfeiture = -0.3 * (2.0 + 1.5) = -1.05
    # rent = 0.1 * (10_000 / 500_000) = 0.002
    # total ≈ -1.05 - 0.002 = -1.052
    assert reward < 0, "PRUNE should be net-negative in BASIC mode"
    assert reward == pytest.approx(-1.052, abs=0.01)


def test_basic_mode_fossilize_good_seed() -> None:
    """FOSSILIZE in BASIC mode rewards f(improvement, contribution).

    DRL Expert review 2026-01-12: Use harmonic mean of improvement and
    contribution. Both must be positive for meaningful reward.
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        param_budget=500_000,
        param_penalty_weight=0.1,
        basic_fossilize_base_bonus=0.3,
        basic_contribution_scale=0.5,
        attribution_formula="harmonic",
    )
    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,  # Positive improvement
        epochs_in_stage=5,  # Full legitimacy
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
        counterfactual_total_improvement=5.0,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.FOSSILIZE,
            seed_contribution=5.0,  # Positive contribution
            val_acc=75.0,
            seed_info=seed_info,
            epoch=20,
            max_epochs=150,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=5.0,
            committed_val_acc=75.0,
            fossilized_seed_params=0,
            effective_seed_params=10_000,
            config=config,
        )
    )

    # harmonic(5.0, 5.0) = 5.0, legitimacy = 1.0
    # fossilize_bonus = (0.3 + 0.5 * 5.0) * 1.0 = 2.8
    # pbrs_bonus for HOLDING stage ≈ 0.12
    # rent = 0.002
    # total ≈ 2.8 + 0.12 - 0.002 ≈ 2.92
    assert reward > 2.0, "Good seed should get substantial reward"
    assert reward == pytest.approx(2.918, abs=0.05)


def test_basic_mode_fossilize_gates_on_counterfactual_not_host_drift() -> None:
    """FOSSILIZE legitimacy must follow the CLEAN counterfactual, not host drift.

    Regression for esper-lite-ea19d665a3: in a declining/plateaued host,
    total_improvement (full-model accuracy change since germination) goes negative
    purely from host drift the seed did NOT cause. The OLD code gated fossilize on
    total_improvement >= 0, so a genuinely valuable seed (positive counterfactual) was
    wrongly routed to the non-contributing/ransomware penalty. With the gate keyed on
    counterfactual_total_improvement, the good seed fossilizes normally even though the
    host declined. This is the exact confound ea19 fixes — the two values DIFFER here.
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        param_budget=500_000,
        param_penalty_weight=0.1,
        basic_fossilize_base_bonus=0.3,
        basic_contribution_scale=0.5,
        attribution_formula="harmonic",
    )
    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=-5.0,  # HOST DRIFT: full model declined since germination
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
        counterfactual_total_improvement=5.0,  # but the seed itself is genuinely valuable
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.FOSSILIZE,
            seed_contribution=5.0,
            val_acc=75.0,
            seed_info=seed_info,
            epoch=20,
            max_epochs=150,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=5.0,
            committed_val_acc=75.0,
            fossilized_seed_params=0,
            effective_seed_params=10_000,
            config=config,
        )
    )

    # Gate uses the counterfactual (+5.0): harmonic(5.0, 5.0) = 5.0, same substantial
    # bonus as the good-seed golden. Under the old total_improvement (-5.0) gate this
    # would have been the non-contributing penalty (improvement <= 0).
    assert reward > 2.0, "Good seed in a declining host must still fossilize on its counterfactual"
    assert reward == pytest.approx(2.918, abs=0.05)


def test_basic_mode_fossilize_ransomware_penalty() -> None:
    """FOSSILIZE in BASIC mode penalizes ransomware seeds.

    DRL Expert review 2026-01-12: Seeds with high contribution but
    negative improvement are gaming - they made themselves important
    without helping the host.
    """
    config = ContributionRewardConfig(
        reward_mode=RewardMode.BASIC,
        param_budget=500_000,
        param_penalty_weight=0.1,
        basic_fossilize_invalid_penalty=-0.5,
        attribution_formula="harmonic",
    )
    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=-1.0,
        total_improvement=-3.0,  # Negative improvement (ransomware!)
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
        counterfactual_total_improvement=-3.0,
    )

    reward = compute_reward(
        ContributionRewardInputs(
            action=LifecycleOp.FOSSILIZE,
            seed_contribution=15.0,  # High contribution (made itself important)
            val_acc=67.0,
            seed_info=seed_info,
            epoch=20,
            max_epochs=150,
            total_params=110_000,
            host_params=100_000,
            acc_at_germination=70.0,
            acc_delta=-3.0,
            committed_val_acc=67.0,
            fossilized_seed_params=0,
            effective_seed_params=10_000,
            config=config,
        )
    )

    # Ransomware: improvement=-3.0 < 0 AND contribution=15.0 > 0.1
    # fossilize_bonus = -0.5 (invalid penalty)
    # This seed should NOT be rewarded despite high contribution
    assert reward < 0, "Ransomware seed should get penalty"


def test_basic_mode_fossilize_drip_split() -> None:
    """FOSSILIZE in BASIC mode splits reward into immediate and drip pool.

    With drip_fraction=0.7, only 30% of the bonus is paid immediately.
    """
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
        FossilizedSeedDripState,
    )
    from esper.simic.rewards.types import SeedInfo
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig(
        drip_fraction=0.7,
        basic_fossilize_base_bonus=0.3,
        basic_contribution_scale=0.5,
    )

    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
        counterfactual_total_improvement=5.0,
    )

    result = compute_basic_reward(
        acc_delta=5.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=20,
        max_epochs=150,
        seed_info=seed_info,
        action=LifecycleOp.FOSSILIZE,
        seed_contribution=5.0,
        seed_id="test-seed",
        slot_id="r0c1",
    )

    # Result now includes drip state
    reward, rent, growth, pbrs, foss_bonus, new_drip, drip_epoch = result

    # Immediate should be 30% of full bonus
    # Full bonus calculation depends on implementation - just check it's reduced
    assert new_drip is not None
    assert isinstance(new_drip, FossilizedSeedDripState)
    assert new_drip.drip_total > 0
    assert new_drip.seed_id == "test-seed"
    assert new_drip.slot_id == "r0c1"
    assert new_drip.remaining_epochs == 130  # 150 - 20

    # No drip payout this epoch (just fossilized)
    assert drip_epoch == 0.0


def test_basic_mode_drip_disabled_when_fraction_zero() -> None:
    """When drip_fraction=0, full bonus is immediate and no drip state created."""
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
    )
    from esper.simic.rewards.types import SeedInfo
    from esper.leyline import LifecycleOp, SeedStage

    config = ContributionRewardConfig(
        drip_fraction=0.0,  # Disabled
        basic_fossilize_base_bonus=0.3,
        basic_contribution_scale=0.5,
    )

    seed_info = SeedInfo(
        stage=SeedStage.HOLDING.value,
        improvement_since_stage_start=0.5,
        total_improvement=5.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.BLENDING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=20,
        counterfactual_total_improvement=5.0,
    )

    result = compute_basic_reward(
        acc_delta=5.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=20,
        max_epochs=150,
        seed_info=seed_info,
        action=LifecycleOp.FOSSILIZE,
        seed_contribution=5.0,
        seed_id="test-seed",
        slot_id="r0c1",
    )

    reward, rent, growth, pbrs, foss_bonus, new_drip, drip_epoch = result

    # No drip state created when drip disabled
    assert new_drip is None
    # Full bonus paid immediately
    assert foss_bonus > 0


def test_basic_mode_drip_payout_positive() -> None:
    """Drip payout rewards continued positive contribution."""
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
        FossilizedSeedDripState,
    )
    from esper.leyline import LifecycleOp

    config = ContributionRewardConfig(
        drip_fraction=0.7,
        max_drip_per_epoch=0.1,
        negative_drip_ratio=0.5,
    )

    drip_state = FossilizedSeedDripState(
        seed_id="test-seed",
        slot_id="r0c1",
        fossilize_epoch=20,
        max_epochs=150,
        drip_total=1.96,
        drip_scale=1.96 / 130,  # ~0.015 per epoch
    )

    result = compute_basic_reward(
        acc_delta=0.1,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=25,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": 3.0},
    )

    reward, _, _, _, _, _, drip_epoch = result

    # Drip = drip_scale * contribution = 0.015 * 3.0 = 0.045
    assert drip_epoch == pytest.approx(0.045, abs=0.005)
    assert drip_epoch > 0


def test_basic_mode_drip_payout_negative() -> None:
    """Drip payout penalizes negative contribution (seed now hurting)."""
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
        FossilizedSeedDripState,
    )
    from esper.leyline import LifecycleOp

    config = ContributionRewardConfig(
        drip_fraction=0.7,
        max_drip_per_epoch=0.1,
        negative_drip_ratio=0.5,
    )

    drip_state = FossilizedSeedDripState(
        seed_id="test-seed",
        slot_id="r0c1",
        fossilize_epoch=20,
        max_epochs=150,
        drip_total=1.96,
        drip_scale=0.015,
    )

    result = compute_basic_reward(
        acc_delta=-1.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=25,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": -2.0},
    )

    reward, _, _, _, _, _, drip_epoch = result

    # Drip = 0.015 * (-2.0) = -0.03
    assert drip_epoch == pytest.approx(-0.03, abs=0.005)
    assert drip_epoch < 0, "Negative contribution should produce negative drip"


def test_basic_mode_drip_asymmetric_clipping() -> None:
    """Large negative drip is clipped more aggressively than positive."""
    from esper.simic.rewards.contribution import (
        compute_basic_reward,
        ContributionRewardConfig,
        FossilizedSeedDripState,
    )
    from esper.leyline import LifecycleOp

    config = ContributionRewardConfig(
        drip_fraction=0.7,
        max_drip_per_epoch=0.1,
        negative_drip_ratio=0.5,  # -0.05 cap
    )

    drip_state = FossilizedSeedDripState(
        seed_id="test-seed",
        slot_id="r0c1",
        fossilize_epoch=140,  # Late fossilization = high drip_scale
        max_epochs=150,
        drip_total=1.96,
        drip_scale=1.96 / 10,  # ~0.196 per epoch
    )

    # Test positive clipping
    result_pos = compute_basic_reward(
        acc_delta=0.1,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=145,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": 5.0},
    )
    _, _, _, _, _, _, drip_pos = result_pos
    assert drip_pos == pytest.approx(0.1, abs=0.001), "Positive clipped to +0.1"

    # Reset drip_paid for next test
    drip_state.drip_paid = 0.0

    # Test negative clipping (asymmetric - tighter)
    result_neg = compute_basic_reward(
        acc_delta=-1.0,
        effective_seed_params=10_000,
        total_params=110_000,
        host_params=100_000,
        config=config,
        epoch=145,
        max_epochs=150,
        seed_info=None,
        action=LifecycleOp.WAIT,
        fossilized_drip_states=[drip_state],
        fossilized_contributions={"test-seed": -5.0},
    )
    _, _, _, _, _, _, drip_neg = result_neg
    assert drip_neg == pytest.approx(-0.05, abs=0.001), "Negative clipped to -0.05 (asymmetric)"


# ===========================================================================
# Phase −1 cheap-lever scale-falsifier flags
# (shaped_attribution_clip / attribution_unit_normalize)
#
# See docs/plans/concepts/2026-06-24-reward-redesign-methodology.md §5 (Phase −1).
# Both flags default OFF => the SHAPED dense-attribution path is byte-identical
# to status quo. These tests drive the new flag behaviour and lock the OFF path.
#
# Design decisions encoded here (validated by the Phase −1 design-review panel):
#   D1 PROXY channel is covered by both levers (clamp/normalize the UNIFIED
#      bounded_attribution, not just the clean-counterfactual path).
#   D2 The clip is POSITIVE-ONLY (min(clip, x)) — it caps the farmable positive
#      tail and leaves the negative-contribution penalty intact (a symmetric
#      clamp would be a confound; cited-run negatives reach -15.3).
#   D3 The unit-normalize divides BOTH signs by exactly 100 (dimensional
#      consistency; no 100:1 penalty:credit asymmetry).
# ===========================================================================


def _shaped_clean_components(config: ContributionRewardConfig):
    """SHAPED clean-counterfactual positive path, WAIT step.

    progress = val_acc - acc_at_germination = 70 - 50 = 20; seed_contribution = 20;
    harmonic(20, 20) = 20; attribution_discount = 1.0 (no regression); timing = 1.0
    (germination epoch 30-10=20 >= warmup 10); ratio_penalty = 0 (ratio 1.0 < 5.0).
    => raw bounded_attribution = 20.0, well above the 2.0 clip so the lever can bite.
    """
    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=5.0,
        total_improvement=20.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=10,
        counterfactual_total_improvement=20.0,
    )
    _, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=20.0,
        val_acc=70.0,
        seed_info=seed_info,
        epoch=30,
        max_epochs=150,
        total_params=110_000,
        host_params=100_000,
        config=config,
        acc_at_germination=50.0,
        acc_delta=2.0,
        return_components=True,
    )
    return components


def _shaped_negative_components(config: ContributionRewardConfig):
    """SHAPED negative-contribution branch, WAIT step.

    seed_contribution = -10 => bounded_attribution = contribution_weight * -10 = -10.0
    (the negative branch applies no discounts; ratio_penalty = 0 since contribution < 1.0).
    """
    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=-1.0,
        total_improvement=-5.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=10,
        counterfactual_total_improvement=-5.0,
    )
    _, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=-10.0,
        val_acc=60.0,
        seed_info=seed_info,
        epoch=30,
        max_epochs=150,
        total_params=110_000,
        host_params=100_000,
        config=config,
        acc_at_germination=65.0,
        acc_delta=-1.0,
        return_components=True,
    )
    return components


def _shaped_proxy_components(config: ContributionRewardConfig):
    """SHAPED proxy path (seed_contribution is None, seed on the output path).

    proxy_contribution_weight = contribution_weight * 0.3 = 0.3;
    improvement_since_stage_start = 30 => bounded_attribution = 0.3 * 30 = 9.0.
    """
    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=30.0,
        total_improvement=30.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=10,
        counterfactual_total_improvement=None,
    )
    _, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=None,
        val_acc=80.0,
        seed_info=seed_info,
        epoch=30,
        max_epochs=150,
        total_params=110_000,
        host_params=100_000,
        config=config,
        acc_at_germination=50.0,
        acc_delta=2.0,
        return_components=True,
    )
    return components


def test_shaped_flags_off_byte_identical_golden() -> None:
    """OFF path (default flags) is the un-clipped status quo: locks the raw values."""
    base = ContributionRewardConfig(reward_mode=RewardMode.SHAPED)
    assert _shaped_clean_components(base).bounded_attribution == pytest.approx(20.0, abs=1e-6)
    assert _shaped_negative_components(base).bounded_attribution == pytest.approx(-10.0, abs=1e-6)
    assert _shaped_proxy_components(base).bounded_attribution == pytest.approx(9.0, abs=1e-6)


def test_shaped_attribution_clip_caps_positive_tail_at_bound() -> None:
    """Arm A: clip=2.0 caps the farmable clean-counterfactual tail at exactly 2.0."""
    off = _shaped_clean_components(ContributionRewardConfig(reward_mode=RewardMode.SHAPED))
    clipped = _shaped_clean_components(
        ContributionRewardConfig(reward_mode=RewardMode.SHAPED, shaped_attribution_clip=2.0)
    )
    assert off.bounded_attribution > 2.0  # precondition: the tail exists
    assert clipped.bounded_attribution == pytest.approx(2.0, abs=1e-9)


def test_shaped_attribution_clip_sweep_5() -> None:
    """Arm A sweep: clip=5.0 caps at exactly 5.0."""
    clip5 = _shaped_clean_components(
        ContributionRewardConfig(reward_mode=RewardMode.SHAPED, shaped_attribution_clip=5.0)
    )
    assert clip5.bounded_attribution == pytest.approx(5.0, abs=1e-9)


def test_shaped_attribution_clip_covers_proxy_channel() -> None:
    """D1: the clip also bounds the PROXY dense channel (else it stays farmable)."""
    off = _shaped_proxy_components(ContributionRewardConfig(reward_mode=RewardMode.SHAPED))
    clipped = _shaped_proxy_components(
        ContributionRewardConfig(reward_mode=RewardMode.SHAPED, shaped_attribution_clip=2.0)
    )
    assert off.bounded_attribution > 2.0  # proxy tail: 0.3 * 30 = 9.0
    assert clipped.bounded_attribution == pytest.approx(2.0, abs=1e-9)


def test_shaped_attribution_clip_is_positive_only() -> None:
    """D2: the clip is positive-only — a large negative penalty is left intact."""
    off = _shaped_negative_components(ContributionRewardConfig(reward_mode=RewardMode.SHAPED))
    clipped = _shaped_negative_components(
        ContributionRewardConfig(reward_mode=RewardMode.SHAPED, shaped_attribution_clip=2.0)
    )
    assert off.bounded_attribution < -2.0  # precondition: a large penalty exists
    assert clipped.bounded_attribution == pytest.approx(off.bounded_attribution, rel=1e-9)


def test_shaped_attribution_unit_normalize_divides_by_100_exactly() -> None:
    """Arm B: the clean-counterfactual term is divided by exactly 100."""
    off = _shaped_clean_components(ContributionRewardConfig(reward_mode=RewardMode.SHAPED))
    norm = _shaped_clean_components(
        ContributionRewardConfig(reward_mode=RewardMode.SHAPED, attribution_unit_normalize=True)
    )
    assert norm.bounded_attribution == pytest.approx(off.bounded_attribution / 100.0, rel=1e-9)
    assert norm.bounded_attribution == pytest.approx(0.2, abs=1e-9)


def test_shaped_attribution_unit_normalize_scales_proxy_channel() -> None:
    """D1/D3: normalize also rescales the proxy dense channel by 1/100."""
    off = _shaped_proxy_components(ContributionRewardConfig(reward_mode=RewardMode.SHAPED))
    norm = _shaped_proxy_components(
        ContributionRewardConfig(reward_mode=RewardMode.SHAPED, attribution_unit_normalize=True)
    )
    assert norm.bounded_attribution == pytest.approx(off.bounded_attribution / 100.0, rel=1e-9)


def test_shaped_attribution_unit_normalize_scales_negative_branch() -> None:
    """D3: normalize rescales BOTH signs (no 100:1 penalty:credit asymmetry)."""
    off = _shaped_negative_components(ContributionRewardConfig(reward_mode=RewardMode.SHAPED))
    norm = _shaped_negative_components(
        ContributionRewardConfig(reward_mode=RewardMode.SHAPED, attribution_unit_normalize=True)
    )
    assert norm.bounded_attribution == pytest.approx(off.bounded_attribution / 100.0, rel=1e-9)
    assert norm.bounded_attribution == pytest.approx(-0.1, abs=1e-9)


def test_phase_minus1_flags_do_not_touch_escrow() -> None:
    """ESCROW path has its own escrow_delta_clip; the SHAPED flags must not alter it."""
    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=5.0,
        total_improvement=20.0,
        epochs_in_stage=5,
        seed_params=10_000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=3,
        seed_age_epochs=10,
        counterfactual_total_improvement=20.0,
    )
    kwargs = dict(
        action=LifecycleOp.WAIT,
        seed_contribution=20.0,
        val_acc=70.0,
        seed_info=seed_info,
        epoch=30,
        max_epochs=150,
        total_params=110_000,
        host_params=100_000,
        acc_at_germination=50.0,
        acc_delta=2.0,
        stable_val_acc=70.0,
        return_components=True,
    )
    _, base = compute_contribution_reward(
        config=ContributionRewardConfig(reward_mode=RewardMode.ESCROW), **kwargs
    )
    _, withflags = compute_contribution_reward(
        config=ContributionRewardConfig(
            reward_mode=RewardMode.ESCROW,
            shaped_attribution_clip=2.0,
            attribution_unit_normalize=True,
        ),
        **kwargs,
    )
    assert withflags.bounded_attribution == pytest.approx(base.bounded_attribution, rel=1e-9)