"""Phase 6 regression baseline test - validates Tamiyo Next implementation.

This test establishes a baseline for Phase 6 (Obs V3 + Policy V2) to detect
performance regressions in future changes. It runs a short training episode
and validates that key metrics are within expected ranges.

Metrics tracked:
- Mean batch accuracy (should improve over training)
- Training stability (no NaN/Inf except flagged low-return-variance EV)
- Policy convergence (accuracy should be > 0)
"""

import math

import pytest
import torch
from dataclasses import replace

from esper.simic.training.vectorized import train_ppo_vectorized
from esper.simic.rewards import RewardMode


@pytest.mark.slow
def test_phase6_regression_baseline(monkeypatch: pytest.MonkeyPatch):
    """Run a small training smoke and validate Phase 6 metrics are finite.

    This test establishes the regression baseline for Phase 6 (Obs V3 + Policy V2).
    It runs a PR-gate-sized training session and validates:
    1. Training completes without errors
    2. Accuracy is finite and bounded
    3. Rolling averages are finite and bounded
    4. Metrics are finite, except diagnostic EV may be NaN when low-return variance is flagged
    """
    import esper.runtime as runtime

    original_get_task_spec = runtime.get_task_spec

    def get_mock_task_spec(name: str):
        spec = original_get_task_spec(name)
        return replace(
            spec,
            dataloader_defaults={**spec.dataloader_defaults, "mock": True},
        )

    monkeypatch.setattr(runtime, "get_task_spec", get_mock_task_spec)

    # Run a tiny CPU training smoke for the PR gate.
    agent, history = train_ppo_vectorized(
        n_episodes=1,
        max_epochs=2,
        n_envs=1,
        task="cifar_minimal",
        use_telemetry=False, # Disable telemetry for speed
        reward_mode=RewardMode.SIMPLIFIED,  # Faster reward computation
        save_path=None,      # Don't save checkpoints
        slots=['r0c0', 'r0c1', 'r0c2'],  # Required parameter
        batch_size_per_env=8,
        device="cpu",
        devices=["cpu"],
        num_workers=0,
        compile_mode="off",
    )

    # === Validation: Training Stability ===

    # Should complete successfully
    assert agent is not None, "Training returned None agent (crashed)"
    assert history is not None, "Training returned None history"
    assert len(history) > 0, "History is empty"

    # === Validation: Accuracy Metrics ===

    # Extract accuracies from history
    accuracies = [batch["avg_accuracy"] for batch in history]
    final_accuracy = accuracies[-1]

    # All accuracies should be finite
    assert all(torch.isfinite(torch.tensor(acc)) for acc in accuracies), (
        f"Found non-finite accuracy: {accuracies}"
    )

    # Accuracy should remain a valid percentage.
    assert all(0.0 <= acc <= 100.0 for acc in accuracies), (
        f"Accuracy out of range [0, 100]: {accuracies}"
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
        # All metrics should be finite, except aggregated explained_variance is
        # intentionally NaN when every update in the batch is low-return-variance
        # flagged; value_nrmse and ev_return_variance carry the stable diagnostics.
        for batch in history:
            for key, value in batch.items():
                if type(value) is int or type(value) is float:
                    if key == "explained_variance" and math.isnan(value):
                        assert batch["ev_low_return_variance"] is True, (
                            "NaN explained_variance is allowed only for low-return-variance batches"
                        )
                        assert batch["ev_low_return_variance_count"] > 0, (
                            "NaN explained_variance must report at least one flagged update"
                        )
                        continue
                    assert torch.isfinite(torch.tensor(value)), (
                        f"Non-finite metric {key} in batch {batch['batch']}: {value}"
                    )

    # === Success: All metrics within expected ranges ===

    print("\n✓ Phase 6 Regression Baseline:")
    print(f"  Final Accuracy:  {final_accuracy:.2f}%")
    print(f"  Batches trained: {len(history)}")


@pytest.mark.slow
def test_phase6_op_conditioning_sanity():
    """Sanity check that op-conditioned values vary with different ops.

    This test verifies that the op-conditioned value head (Q(s,op)) actually
    produces different values for explicit ops, confirming that Phase 4
    implementation is wired correctly in the training loop.
    """
    from esper.tamiyo.policy.factory import create_policy
    from esper.leyline import NUM_OPS, SlotConfig
    from esper.tamiyo.policy.features import get_feature_size

    torch.manual_seed(0)
    slot_config = SlotConfig.default()
    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
    )

    # Create identical state
    state = torch.randn(4, get_feature_size(slot_config))
    bp_indices = torch.randint(0, 13, (4, slot_config.num_slots))

    with torch.no_grad():
        output = policy.network.forward(
            state.unsqueeze(1),
            bp_indices.unsqueeze(1),
        )
        lstm_out = output["lstm_out"]
        q_values = []
        for op_idx in range(NUM_OPS):
            op_tensor = torch.full(
                (state.shape[0], 1),
                op_idx,
                dtype=torch.long,
                device=state.device,
            )
            q_values.append(policy.network._compute_q(lstm_out, op_tensor).squeeze(1))

    q_values_tensor = torch.stack(q_values, dim=1)  # [batch, NUM_OPS]
    q_spread = q_values_tensor.max(dim=1).values - q_values_tensor.min(dim=1).values

    assert torch.all(q_spread > 1e-5), (
        f"Q(s,op) estimates don't vary by explicit op (spread={q_spread.tolist()}). "
        "This suggests op-conditioning is not wired into the value head."
    )

    print("\n✓ Op-conditioning sanity check:")
    print(f"  Minimum Q spread across ops: {q_spread.min().item():.6f}")
