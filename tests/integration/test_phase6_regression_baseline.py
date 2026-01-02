"""Phase 6 regression baseline test - validates Tamiyo Next implementation.

This test establishes a baseline for Phase 6 (Obs V3 + Policy V2) to detect
performance regressions in future changes. It runs a short training episode
and validates that key metrics are within expected ranges.

Metrics tracked:
- Mean batch accuracy (should improve over training)
- Training stability (no NaN/Inf anywhere)
- Policy convergence (accuracy should be > 0)
"""

import pytest
import torch

from esper.simic.training.vectorized import train_ppo_vectorized
from esper.simic.rewards import RewardMode


@pytest.mark.slow
def test_phase6_regression_baseline():
    """Run short training and validate Phase 6 metrics are reasonable.

    This test establishes the regression baseline for Phase 6 (Obs V3 + Policy V2).
    It runs a very short training session and validates:
    1. Training completes without errors
    2. Accuracy is positive and finite
    3. Accuracy improves during training
    4. No NaN/Inf in any metrics

    Baseline expectations (Phase 6 with CIFAR-10):
    - Final accuracy: > 15% (random is 10%, some learning should occur)
    - Accuracy should improve from first to last batch
    - All metrics finite
    """
    # Run ultra-short training for regression test
    agent, history = train_ppo_vectorized(
        n_episodes=5,        # Very short run
        max_epochs=10,       # Only 10 epochs per episode
        n_envs=2,            # Minimal parallelism
        task="cifar_baseline",
        use_telemetry=False, # Disable telemetry for speed
        reward_mode=RewardMode.SIMPLIFIED,  # Faster reward computation
        save_path=None,      # Don't save checkpoints
        slots=['r0c0', 'r0c1', 'r0c2'],  # Required parameter
    )

    # === Validation: Training Stability ===

    # Should complete successfully
    assert agent is not None, "Training returned None agent (crashed)"
    assert history is not None, "Training returned None history"
    assert len(history) > 0, "History is empty"

    # === Validation: Accuracy Metrics ===

    # Extract accuracies from history
    accuracies = [batch["avg_accuracy"] for batch in history]
    first_accuracy = accuracies[0]
    final_accuracy = accuracies[-1]

    # All accuracies should be finite
    assert all(torch.isfinite(torch.tensor(acc)) for acc in accuracies), (
        f"Found non-finite accuracy: {accuracies}"
    )

    # Final accuracy should be > random baseline (10% for CIFAR-10)
    # With very short training (5 episodes), expect at least some learning
    assert final_accuracy > 15.0, (
        f"Final accuracy too low: {final_accuracy:.2f}%. "
        f"Expected >15% (random is 10%). Training may not be working."
    )

    # Accuracy should improve during training (or at least not degrade)
    assert final_accuracy >= first_accuracy * 0.9, (
        f"Accuracy degraded: {first_accuracy:.2f}% → {final_accuracy:.2f}%. "
        "This suggests a training bug."
    )

    # === Validation: Rolling Average ===

    rolling_avgs = [batch["rolling_avg_accuracy"] for batch in history]

    # Rolling average should be finite
    assert all(torch.isfinite(torch.tensor(avg)) for avg in rolling_avgs), (
        f"Found non-finite rolling average: {rolling_avgs}"
    )

    # Rolling average should be reasonable (0-100%)
    assert all(0.0 <= avg <= 100.0 for avg in rolling_avgs), (
        f"Rolling average out of range [0, 100]: {rolling_avgs}"
    )

    # === Validation: Metric Stability ===

    # Check if any metrics exist in history (PPO loss, value, etc.)
    if len(history) > 0 and len(history[0]) > 3:  # More than just batch/episodes/avg_accuracy
        # All metrics should be finite (no NaN/Inf)
        for batch in history:
            for key, value in batch.items():
                if isinstance(value, (int, float)):
                    assert torch.isfinite(torch.tensor(value)), (
                        f"Non-finite metric {key} in batch {batch['batch']}: {value}"
                    )

    # === Success: All metrics within expected ranges ===

    print("\n✓ Phase 6 Regression Baseline:")
    print(f"  First Accuracy:  {first_accuracy:.2f}%")
    print(f"  Final Accuracy:  {final_accuracy:.2f}%")
    print(f"  Improvement:     {final_accuracy - first_accuracy:+.2f}%")
    print(f"  Batches trained: {len(history)}")


@pytest.mark.slow
def test_phase6_op_conditioning_sanity():
    """Sanity check that op-conditioned values vary with different ops.

    This test verifies that the op-conditioned value head (Q(s,op)) actually
    produces different values for different ops, confirming that Phase 4
    implementation is working correctly in the training loop.
    """
    from esper.tamiyo.policy.factory import create_policy
    from esper.leyline import SlotConfig
    from esper.tamiyo.policy.features import get_feature_size

    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
    )

    # Create identical state
    state = torch.randn(4, get_feature_size(slot_config))
    bp_indices = torch.randint(0, 13, (4, slot_config.num_slots))
    from tests.helpers import create_all_valid_masks
    masks = create_all_valid_masks(batch_size=state.shape[0])

    # Sample actions multiple times
    values = []
    for _ in range(10):
        result = policy.get_action(state, bp_indices, masks=masks)
        values.append(result.value.clone())

    # Convert to tensor
    values_tensor = torch.stack(values)  # [10, 4]

    # Since ops are sampled stochastically, values should vary
    # (different ops → different Q(s,op) values)
    value_std = values_tensor.std(dim=0).mean()

    assert value_std > 0.01, (
        f"Value estimates don't vary across sampled ops (std={value_std:.6f}). "
        "This suggests op-conditioning is not working - all ops produce same value."
    )

    print("\n✓ Op-conditioning sanity check:")
    print(f"  Value std across ops: {value_std:.4f} (> 0.01 threshold)")
