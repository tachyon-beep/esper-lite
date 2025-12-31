"""Integration test: Q-values flow from Policy V2 → PPO → Telemetry → Sanctum UI."""

import torch
from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.nissa import NissaHub
from esper.leyline.slot_config import SlotConfig
from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType, PPOUpdatePayload
from esper.leyline import (
    NUM_ALPHA_CURVES,
    NUM_ALPHA_SPEEDS,
    NUM_ALPHA_TARGETS,
    NUM_BLUEPRINTS,
    NUM_OPS,
    NUM_STYLES,
    NUM_TEMPO,
)
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

    # Create action masks (all valid)
    def _create_all_valid_masks(batch_size: int = 1) -> dict[str, torch.Tensor]:
        """Create all-valid per-head action masks for testing."""
        return {
            "slot": torch.ones(batch_size, 3, dtype=torch.bool),
            "blueprint": torch.ones(batch_size, NUM_BLUEPRINTS, dtype=torch.bool),
            "style": torch.ones(batch_size, NUM_STYLES, dtype=torch.bool),
            "tempo": torch.ones(batch_size, NUM_TEMPO, dtype=torch.bool),
            "alpha_target": torch.ones(batch_size, NUM_ALPHA_TARGETS, dtype=torch.bool),
            "alpha_speed": torch.ones(batch_size, NUM_ALPHA_SPEEDS, dtype=torch.bool),
            "alpha_curve": torch.ones(batch_size, NUM_ALPHA_CURVES, dtype=torch.bool),
            "op": torch.ones(batch_size, NUM_OPS, dtype=torch.bool),
        }

    # Fill buffer with synthetic rollout data
    device = torch.device("cpu")
    hidden = agent.policy.network.get_initial_hidden(1, device)

    for env_id in range(2):
        agent.buffer.start_episode(env_id=env_id)

        for step in range(10):
            # Create synthetic state
            state = torch.randn(1, state_dim, device=device)
            blueprint_indices = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=device)
            masks = _create_all_valid_masks(batch_size=1)

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
    assert "q_germinate" in metrics, "Q-value for GERMINATE missing"
    assert "q_advance" in metrics, "Q-value for ADVANCE missing"
    assert "q_fossilize" in metrics, "Q-value for FOSSILIZE missing"
    assert "q_prune" in metrics, "Q-value for PRUNE missing"
    assert "q_wait" in metrics, "Q-value for WAIT missing"
    assert "q_set_alpha" in metrics, "Q-value for SET_ALPHA missing"
    assert "q_variance" in metrics, "Q-variance missing"
    assert "q_spread" in metrics, "Q-spread missing"

    assert metrics["q_variance"] >= 0.0, "Q-variance should be non-negative"
    assert metrics["q_spread"] >= 0.0, "Q-spread should be non-negative"

    # Create telemetry event manually (simulating emit_ppo_update_event)
    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        epoch=1,
        data=PPOUpdatePayload(
            policy_loss=metrics.get("policy_loss", 0.0),
            value_loss=metrics.get("value_loss", 0.0),
            entropy=metrics.get("entropy", 0.0),
            grad_norm=metrics.get("grad_norm", 0.0),
            kl_divergence=metrics.get("approx_kl", 0.0),
            clip_fraction=metrics.get("clip_fraction", 0.0),
            nan_grad_count=metrics.get("nan_grad_count", 0),
            q_germinate=metrics["q_germinate"],
            q_advance=metrics["q_advance"],
            q_fossilize=metrics["q_fossilize"],
            q_prune=metrics["q_prune"],
            q_wait=metrics["q_wait"],
            q_set_alpha=metrics["q_set_alpha"],
            q_variance=metrics["q_variance"],
            q_spread=metrics["q_spread"],
        ),
    )

    # Process event through aggregator
    aggregator.process_event(event)

    # Verify aggregator received and wired Q-values
    snapshot = aggregator.get_snapshot()

    # Q-values should be populated in TamiyoState
    assert snapshot.tamiyo.q_germinate != 0.0, "q_germinate should have real value"
    assert snapshot.tamiyo.q_advance != 0.0, "q_advance should have real value"
    assert snapshot.tamiyo.q_fossilize != 0.0, "q_fossilize should have real value"
    assert snapshot.tamiyo.q_prune != 0.0, "q_prune should have real value"
    assert snapshot.tamiyo.q_wait != 0.0, "q_wait should have real value"
    assert snapshot.tamiyo.q_set_alpha != 0.0, "q_set_alpha should have real value"
    assert snapshot.tamiyo.q_variance >= 0.0, "q_variance should be non-negative"
    assert snapshot.tamiyo.q_spread >= 0.0, "q_spread should be non-negative"

    # Print Q-values for verification
    print(f"\n=== Q-Value Telemetry Flow Test ===")
    print(f"Q-variance: {snapshot.tamiyo.q_variance:.4f}")
    print(f"Q-spread: {snapshot.tamiyo.q_spread:.4f}")
    print(f"Q-values:")
    print(f"  GERMINATE:  {snapshot.tamiyo.q_germinate:.2f}")
    print(f"  ADVANCE:    {snapshot.tamiyo.q_advance:.2f}")
    print(f"  FOSSILIZE:  {snapshot.tamiyo.q_fossilize:.2f}")
    print(f"  PRUNE:      {snapshot.tamiyo.q_prune:.2f}")
    print(f"  WAIT:       {snapshot.tamiyo.q_wait:.2f}")
    print(f"  SET_ALPHA:  {snapshot.tamiyo.q_set_alpha:.2f}")

    # If variance > 0, Q-values are differentiated (op-conditioning works)
    # If variance ≈ 0, all Q-values are same (critic ignoring ops - BAD)
    if snapshot.tamiyo.q_variance > 0.01:
        print("✓ Q-values are differentiated (op-conditioning working)")
    else:
        print("⚠ Q-values have low variance (may indicate critic not conditioning on ops)")
