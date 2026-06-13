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


def _fill_buffer_with_fixed_hidden(
    agent: PPOAgent,
    *,
    state: torch.Tensor,
    blueprint_indices: torch.Tensor,
    masks: dict[str, torch.Tensor],
    hidden_h: torch.Tensor,
    hidden_c: torch.Tensor,
    num_envs: int,
    num_steps: int,
) -> None:
    """Fill an agent's rollout buffer with IDENTICAL observations on every row.

    Actions, log-probs and values come from a REAL ``get_action`` call so the
    recorded old log-probs match the policy and the PPO update runs a full epoch
    (degenerate placeholder log-probs would blow up the ratio and early-stop,
    leaving zero epochs and NaN Q telemetry).

    The first valid row (env 0, step 0) — which the Q-value telemetry path
    samples — stores the supplied ``hidden_h``/``hidden_c`` as its recurrent
    INPUT state, mirroring how rollout records ``pre_step_hiddens``. The stored
    hidden only influences the telemetry forward pass, not the already-recorded
    rollout log-probs, so we can pin it independently without destabilising PPO.

    Args:
        hidden_h/hidden_c: Stored hidden for env 0, step 0 — shape
            [lstm_layers, 1, hidden_dim] (LSTM convention with batch dim).
    """
    device = torch.device("cpu")
    init_hidden = agent.policy.network.get_initial_hidden(1, device)
    for env_id in range(num_envs):
        agent.buffer.start_episode(env_id=env_id)
        running_hidden = init_hidden
        for step in range(num_steps):
            result = agent.policy.network.get_action(
                state,
                blueprint_indices,
                running_hidden,
                slot_mask=masks["slot"],
                blueprint_mask=masks["blueprint"],
                style_mask=masks["style"],
                tempo_mask=masks["tempo"],
                alpha_target_mask=masks["alpha_target"],
                alpha_speed_mask=masks["alpha_speed"],
                alpha_curve_mask=masks["alpha_curve"],
                op_mask=masks["op"],
            )
            running_hidden = result.hidden

            # Pin the controlled hidden state on the row Q telemetry samples first
            # (env 0, step 0); other rows store their genuine pre-step hidden.
            if env_id == 0 and step == 0:
                row_h, row_c = hidden_h, hidden_c
            else:
                row_h = torch.zeros_like(hidden_h)
                row_c = torch.zeros_like(hidden_c)

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
                value=result.values.item(),
                reward=0.5,
                done=False,
                slot_mask=masks["slot"].squeeze(0),
                blueprint_mask=masks["blueprint"].squeeze(0),
                style_mask=masks["style"].squeeze(0),
                tempo_mask=masks["tempo"].squeeze(0),
                alpha_target_mask=masks["alpha_target"].squeeze(0),
                alpha_speed_mask=masks["alpha_speed"].squeeze(0),
                alpha_curve_mask=masks["alpha_curve"].squeeze(0),
                op_mask=masks["op"].squeeze(0),
                hidden_h=row_h,
                hidden_c=row_c,
                bootstrap_value=0.0,
            )
        agent.buffer.end_episode(env_id=env_id)


def _make_agent_and_inputs() -> tuple[PPOAgent, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Create a fresh LSTM agent plus a fixed observation/blueprint/mask triple."""
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
        device="cpu",
        recurrent_n_epochs=1,
    )
    device = torch.device("cpu")
    # Deterministic, identical observation for BOTH runs (seeded).
    torch.manual_seed(1234)
    state = torch.randn(1, state_dim, device=device)
    blueprint_indices = torch.zeros(1, slot_config.num_slots, dtype=torch.long, device=device)
    masks = create_all_valid_masks(batch_size=1)
    return agent, state, blueprint_indices, masks


def test_q_telemetry_conditions_on_rollout_hidden_state():
    """TPD-002: identical observations with DIFFERENT stored hidden states must
    produce DIFFERENT Q telemetry.

    This is the regression guard against the recurrent policy's Q telemetry
    being computed with ``hidden=None`` (the network's initial zero-state),
    which ignored the rollout row's actual recurrent context and produced a
    fake, context-free diagnostic. With the fix, the telemetry path consumes
    the buffer's stored per-row hidden state, so two runs whose ONLY difference
    is the stored hidden state must yield different Q values.
    """
    # Two agents sharing identical network WEIGHTS so the only varying input is
    # the stored hidden state. We build one agent, snapshot its weights, and load
    # them into the second agent.
    agent_a, state, blueprint_indices, masks = _make_agent_and_inputs()
    agent_b, _, _, _ = _make_agent_and_inputs()
    agent_b.policy.network.load_state_dict(agent_a.policy.network.state_dict())

    lstm_layers = agent_a.buffer.lstm_layers
    hidden_dim = agent_a.lstm_hidden_dim

    # Sanity: the network must actually be hidden-sensitive for this test to be
    # meaningful. A zero hidden vs a large random hidden should move lstm_out.
    zero_h = torch.zeros(lstm_layers, 1, hidden_dim)
    zero_c = torch.zeros(lstm_layers, 1, hidden_dim)
    torch.manual_seed(7)
    rand_h = torch.randn(lstm_layers, 1, hidden_dim)
    rand_c = torch.randn(lstm_layers, 1, hidden_dim)

    obs = state.unsqueeze(1)  # [1, 1, state_dim]
    bp = blueprint_indices.unsqueeze(1)  # [1, 1, num_slots]
    with torch.no_grad():
        out_zero = agent_a.policy.network.forward(
            state=obs, blueprint_indices=bp, hidden=(zero_h, zero_c)
        )["lstm_out"]
        out_rand = agent_a.policy.network.forward(
            state=obs, blueprint_indices=bp, hidden=(rand_h, rand_c)
        )["lstm_out"]
    assert not torch.allclose(out_zero, out_rand, atol=1e-5), (
        "Network is not hidden-sensitive; test cannot distinguish hidden states. "
        "This indicates the LSTM is not propagating its recurrent state."
    )

    # Run A: stored hidden = zero-state.
    _fill_buffer_with_fixed_hidden(
        agent_a,
        state=state,
        blueprint_indices=blueprint_indices,
        masks=masks,
        hidden_h=zero_h,
        hidden_c=zero_c,
        num_envs=2,
        num_steps=10,
    )
    metrics_a = agent_a.update(clear_buffer=True)
    # Returned metric is a tuple of NUM_OPS floats (see PPOUpdateMetricsBuilder).
    q_a = torch.tensor(metrics_a["op_q_values"])

    # Run B: identical observations, identical weights, but a DIFFERENT stored
    # hidden state on the sampled row.
    _fill_buffer_with_fixed_hidden(
        agent_b,
        state=state,
        blueprint_indices=blueprint_indices,
        masks=masks,
        hidden_h=rand_h,
        hidden_c=rand_c,
        num_envs=2,
        num_steps=10,
    )
    metrics_b = agent_b.update(clear_buffer=True)
    q_b = torch.tensor(metrics_b["op_q_values"])

    # Both Q vectors must be finite and well-shaped.
    assert q_a.shape == (NUM_OPS,)
    assert q_b.shape == (NUM_OPS,)
    assert torch.isfinite(q_a).all()
    assert torch.isfinite(q_b).all()

    # The CORE assertion: differing stored hidden states => differing Q telemetry.
    # If the telemetry path ignored the stored hidden (the TPD-002 bug), both runs
    # would forward with the same initial hidden and these would be identical.
    assert not torch.allclose(q_a, q_b, atol=1e-5), (
        "Q telemetry is identical for different stored hidden states — the "
        "recurrent rollout hidden state is being ignored (TPD-002 regression)."
    )
