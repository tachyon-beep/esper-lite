"""Tests for scaffolding reward shaping (Phase 3.1)."""

import pytest
from esper.simic.rewards import compute_contribution_reward, ContributionRewardConfig, SeedInfo
from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp


def test_synergy_bonus_added_for_positive_interaction():
    """Verify synergy bonus is added when interaction_sum > 0."""
    config = ContributionRewardConfig()

    seed_info = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=0.05,
        total_improvement=0.05,
        epochs_in_stage=3,
        seed_params=10000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=5,
        seed_age_epochs=8,
        interaction_sum=2.5,  # Strong positive interaction
        boost_received=1.2,
    )

    reward_with_synergy = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.05,
        val_acc=70.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
    )

    # Same but with no interaction
    seed_info_no_synergy = SeedInfo(
        stage=SeedStage.BLENDING.value,
        improvement_since_stage_start=0.05,
        total_improvement=0.05,
        epochs_in_stage=3,
        seed_params=10000,
        previous_stage=SeedStage.TRAINING.value,
        previous_epochs_in_stage=5,
        seed_age_epochs=8,
        interaction_sum=0.0,
        boost_received=0.0,
    )

    reward_no_synergy = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.05,
        val_acc=70.0,
        seed_info=seed_info_no_synergy,
        epoch=10,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
    )

    assert reward_with_synergy > reward_no_synergy, (
        f"Expected synergy bonus: {reward_with_synergy} > {reward_no_synergy}"
    )


def test_interaction_metrics_extracted_from_seed_state():
    """Integration test: verify SeedInfo.from_seed_state() extracts interaction metrics.

    This is a critical data pipeline test - if interaction_sum and boost_received
    are not extracted from SeedMetrics, the synergy bonus will always be 0.0 in production.
    """
    from esper.kasmina.slot import SeedState, SeedMetrics

    # Create SeedMetrics with interaction data
    metrics = SeedMetrics(
        epochs_total=8,
        epochs_in_current_stage=3,
        initial_val_accuracy=65.0,
        current_val_accuracy=70.0,
        accuracy_at_stage_start=68.0,
        interaction_sum=2.5,  # Strong positive interaction
        boost_received=1.2,   # Strongest single partner
    )

    # Create SeedState with these metrics
    seed_state = SeedState(
        seed_id="test-seed",
        blueprint_id="test-blueprint",
        stage=SeedStage.BLENDING,
        previous_stage=SeedStage.TRAINING,
        previous_epochs_in_stage=5,
        metrics=metrics,
    )

    # Convert to SeedInfo (this is the critical pipeline step)
    seed_info = SeedInfo.from_seed_state(seed_state, seed_params=10000)

    # Verify interaction metrics were extracted correctly
    assert seed_info is not None, "SeedInfo should not be None"
    assert seed_info.interaction_sum == 2.5, (
        f"interaction_sum not extracted: got {seed_info.interaction_sum}, expected 2.5"
    )
    assert seed_info.boost_received == 1.2, (
        f"boost_received not extracted: got {seed_info.boost_received}, expected 1.2"
    )

    # Verify synergy bonus is non-zero when using this SeedInfo
    config = ContributionRewardConfig()
    reward, components = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.05,
        val_acc=70.0,
        seed_info=seed_info,
        epoch=10,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
        return_components=True,
    )

    assert components.synergy_bonus > 0, (
        f"Synergy bonus should be positive, got {components.synergy_bonus}"
    )


def test_scaffold_credit_on_beneficiary_success():
    """Verify scaffold credit is added when beneficiary fossilizes."""
    from esper.simic.rewards import compute_scaffold_hindsight_credit

    # Seed A provided boost of 1.5 to seed B
    # Seed B just fossilized with 3% improvement
    credit = compute_scaffold_hindsight_credit(
        boost_given=1.5,
        beneficiary_improvement=3.0,
        credit_weight=0.2,
    )

    assert credit > 0, f"Expected positive credit, got {credit}"
    assert credit <= 0.2, f"Credit should be bounded by weight"
