"""Integration test: Q-values flow from Policy V2 → PPO → Telemetry → Sanctum UI."""

import math
import torch

from tests.helpers import create_all_valid_masks
from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline import NUM_OPS
from esper.leyline.slot_config import SlotConfig
from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType, PPOUpdatePayload
from esper.simic.agent import PPOAgent
from esper.tamiyo.policy.factory import create_policy
from esper.tamiyo.policy.features import get_feature_size


def test_q_values_end_to_end_flow():
    """Q-values flow from policy → PPO → telemetry → aggregator → UI."""
    # Setup aggregator
    aggregator = SanctumAggregator(num_envs=2)

    # Create policy and agent
    slot_config = SlotConfig.default()
    state_dim = get_feature_size(slot_config)

    policy = create_policy(
        policy_type="lstm",
        slot_config=slot_config,
        device="cpu",
        compile_mode="off",
    )

    agent = PPOAgent(
        policy=policy,
        slot_config=slot_config,
        num_envs=2,
        max_steps_per_env=10,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        recurrent_n_epochs=1,
        device="cpu",
    )

    # Fill buffer with synthetic rollout data
    device = torch.device("cpu")
    hidden = agent.policy.network.get_initial_hidden(1, device)

    for env_id in range(2):
        agent.buffer.start_episode(env_id=env_id)

        for step in range(10):
            # Create synthetic state
            state = torch.randn(1, state_dim, device=device)
            blueprint_indices = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=device)
            masks = create_all_valid_masks(batch_size=1)

            # Get action from policy
            pre_hidden = hidden
            result = agent.policy.network.get_action(
                state,
                blueprint_indices,
                hidden,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )
            hidden = result.hidden

            # Add to buffer
            agent.buffer.add(
                env_id=env_id,
                state=state.squeeze(0),
                blueprint_indices=blueprint_indices.squeeze(0),
                slot_action=result.actions["slot"],
                blueprint_action=result.actions["blueprint"],
                style_action=result.actions["style"],
                tempo_action=result.actions["tempo"],
                alpha_target_action=result.actions["alpha_target"],
                alpha_speed_action=result.actions["alpha_speed"],
                alpha_curve_action=result.actions["alpha_curve"],
                op_action=result.actions["op"],
                effective_op_action=result.actions["op"],
                slot_log_prob=result.log_probs["slot"],
                blueprint_log_prob=result.log_probs["blueprint"],
                style_log_prob=result.log_probs["style"],
                tempo_log_prob=result.log_probs["tempo"],
                alpha_target_log_prob=result.log_probs["alpha_target"],
                alpha_speed_log_prob=result.log_probs["alpha_speed"],
                alpha_curve_log_prob=result.log_probs["alpha_curve"],
                op_log_prob=result.log_probs["op"],
                value=result.values.item(),  # Extract scalar value
                reward=0.5,  # Synthetic reward
                done=False,
                slot_mask=masks["slot"].squeeze(0),
                blueprint_mask=masks["blueprint"].squeeze(0),
                style_mask=masks["style"].squeeze(0),
                tempo_mask=masks["tempo"].squeeze(0),
                alpha_target_mask=masks["alpha_target"].squeeze(0),
                alpha_speed_mask=masks["alpha_speed"].squeeze(0),
                alpha_curve_mask=masks["alpha_curve"].squeeze(0),
                op_mask=masks["op"].squeeze(0),
                hidden_h=pre_hidden[0],
                hidden_c=pre_hidden[1],
                bootstrap_value=0.0,
            )

        agent.buffer.end_episode(env_id=env_id)

    # Trigger PPO update (collects Q-values)
    metrics = agent.update(clear_buffer=True)

    # Verify metrics contain Q-values
    assert "op_q_values" in metrics, "Op-conditioned Q-value vector missing"
    assert "op_valid_mask" in metrics, "Op mask missing"
    assert "q_variance" in metrics, "Q-variance missing"
    assert "q_spread" in metrics, "Q-spread missing"

    assert len(metrics["op_q_values"]) == NUM_OPS, "Expected NUM_OPS entries"
    assert len(metrics["op_valid_mask"]) == NUM_OPS, "Expected NUM_OPS mask entries"
    assert metrics["q_variance"] >= 0.0, "Q-variance should be non-negative"
    assert metrics["q_spread"] >= 0.0, "Q-spread should be non-negative"

    # Create telemetry event manually (simulating emit_ppo_update_event)
    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        epoch=1,
        data=PPOUpdatePayload(
            policy_loss=metrics["policy_loss"],
            value_loss=metrics["value_loss"],
            entropy=metrics["entropy"],
            grad_norm=metrics.get("grad_norm", 0.0),  # Keep .get() - not asserted
            kl_divergence=metrics["approx_kl"],
            clip_fraction=metrics["clip_fraction"],
            nan_grad_count=metrics.get("nan_grad_count", 0),  # Keep .get() - not asserted
            # Q-values (already asserted to exist)
            op_q_values=metrics["op_q_values"],
            op_valid_mask=metrics["op_valid_mask"],
            q_variance=metrics["q_variance"],
            q_spread=metrics["q_spread"],
        ),
    )

    # Process event through aggregator
    aggregator.process_event(event)

    # Verify aggregator received and wired Q-values
    snapshot = aggregator.get_snapshot()

    # Q-values should be finite (not NaN/inf)
    assert all(math.isfinite(q) for q in snapshot.tamiyo.op_q_values), "q-values should be finite"
    assert snapshot.tamiyo.q_variance >= 0.0, "q_variance should be non-negative"
    assert snapshot.tamiyo.q_spread >= 0.0, "q_spread should be non-negative"

    # Print Q-values for verification
    print("\n=== Q-Value Telemetry Flow Test ===")
    print(f"Q-variance: {snapshot.tamiyo.q_variance:.4f}")
    print(f"Q-spread: {snapshot.tamiyo.q_spread:.4f}")
    print("Q-values:")
    for idx, value in enumerate(snapshot.tamiyo.op_q_values):
        print(f"  {idx}: {value:.2f}")

    # If variance > 0, Q-values are differentiated (op-conditioning works)
    # If variance ≈ 0, all Q-values are same (critic ignoring ops - BAD)
    if snapshot.tamiyo.q_variance > 0.01:
        print("✓ Q-values are differentiated (op-conditioning working)")
    else:
        print("⚠ Q-values have low variance (may indicate critic not conditioning on ops)")
