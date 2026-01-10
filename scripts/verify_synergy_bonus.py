#!/usr/bin/env python3
"""Quick verification that synergy bonus computation works end-to-end."""

from esper.simic.rewards import (
    compute_contribution_reward,
    ContributionRewardConfig,
    SeedInfo,
)
from esper.kasmina.slot import SeedState, SeedMetrics
from esper.leyline import SeedStage
from esper.leyline.factored_actions import LifecycleOp


def verify_synergy_pipeline() -> None:
    """Verify that interaction metrics flow through the full pipeline."""
    print("=" * 70)
    print("Verifying Synergy Bonus Pipeline")
    print("=" * 70)

    # Test 1: Direct SeedInfo construction
    print("\n1. Testing direct SeedInfo construction...")
    seed_info_direct = SeedInfo(
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
    )

    config = ContributionRewardConfig()
    result = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.05,
        val_acc=70.0,
        seed_info=seed_info_direct,
        epoch=10,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
        return_components=True,
    )
    assert isinstance(result, tuple)
    reward, components = result

    print(f"   Interaction sum: {seed_info_direct.interaction_sum}")
    print(f"   Boost received: {seed_info_direct.boost_received}")
    print(f"   Synergy bonus: {components.synergy_bonus:.6f}")
    assert components.synergy_bonus > 0, "Synergy bonus should be positive!"
    print("   ✓ Direct construction works")

    # Test 2: SeedState -> SeedInfo pipeline
    print("\n2. Testing SeedState -> SeedInfo pipeline...")
    metrics = SeedMetrics(
        epochs_total=8,
        epochs_in_current_stage=3,
        initial_val_accuracy=65.0,
        current_val_accuracy=70.0,
        accuracy_at_stage_start=68.0,
        interaction_sum=2.5,
        boost_received=1.2,
    )

    seed_state = SeedState(
        seed_id="test-seed",
        blueprint_id="test-blueprint",
        stage=SeedStage.BLENDING,
        previous_stage=SeedStage.TRAINING,
        previous_epochs_in_stage=5,
        metrics=metrics,
    )

    seed_info_from_state = SeedInfo.from_seed_state(seed_state, seed_params=10000)
    assert seed_info_from_state is not None, "from_seed_state returned None!"

    print(f"   SeedMetrics interaction_sum: {metrics.interaction_sum}")
    print(f"   SeedInfo interaction_sum: {seed_info_from_state.interaction_sum}")
    assert seed_info_from_state.interaction_sum == 2.5, "interaction_sum not extracted!"
    assert seed_info_from_state.boost_received == 1.2, "boost_received not extracted!"
    print("   ✓ Metrics extraction works")

    result2 = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.05,
        val_acc=70.0,
        seed_info=seed_info_from_state,
        epoch=10,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
        return_components=True,
    )
    assert isinstance(result2, tuple)
    reward2, components2 = result2

    print(f"   Synergy bonus: {components2.synergy_bonus:.6f}")
    assert components2.synergy_bonus > 0, "Synergy bonus should be positive!"
    print("   ✓ End-to-end pipeline works")

    # Test 3: Verify bonus scales correctly
    print("\n3. Testing bonus scaling...")
    print(f"   Reward with synergy: {reward:.6f}")

    # Same scenario but no interaction
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

    result3 = compute_contribution_reward(
        action=LifecycleOp.WAIT,
        seed_contribution=0.05,
        val_acc=70.0,
        seed_info=seed_info_no_synergy,
        epoch=10,
        max_epochs=25,
        total_params=110000,
        host_params=100000,
        config=config,
        return_components=True,
    )
    assert isinstance(result3, tuple)
    reward_no_synergy, components_no_synergy = result3

    print(f"   Reward without synergy: {reward_no_synergy:.6f}")
    print(f"   Synergy bonus difference: {reward - reward_no_synergy:.6f}")
    assert reward > reward_no_synergy, "Synergy should increase reward!"
    print("   ✓ Bonus scaling works")

    print("\n" + "=" * 70)
    print("All verifications PASSED! Synergy bonus is working correctly.")
    print("=" * 70)


if __name__ == "__main__":
    verify_synergy_pipeline()
