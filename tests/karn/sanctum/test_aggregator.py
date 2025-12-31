"""Tests for SanctumAggregator telemetry event processing."""

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import PPOUpdatePayload, SeedGateEvaluatedPayload


def test_ppo_update_populates_history():
    """PPO_UPDATE_COMPLETED should append to history deques."""
    agg = SanctumAggregator(num_envs=4)

    # Simulate 3 PPO updates
    for i in range(3):
        event = TelemetryEvent(
            event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
            data=PPOUpdatePayload(
                policy_loss=0.1 * (i + 1),
                value_loss=0.2 * (i + 1),
                grad_norm=1.0 * (i + 1),
                entropy=1.5 - (0.1 * i),
                explained_variance=0.3 * (i + 1),
                kl_divergence=0.01 * (i + 1),
                clip_fraction=0.1 + (0.02 * i),
                nan_grad_count=0,
            ),
        )
        agg.process_event(event)

    snapshot = agg.get_snapshot()
    tamiyo = snapshot.tamiyo

    # Should have 3 values in each history
    assert len(tamiyo.policy_loss_history) == 3
    assert len(tamiyo.value_loss_history) == 3
    assert len(tamiyo.grad_norm_history) == 3
    assert len(tamiyo.entropy_history) == 3
    assert len(tamiyo.explained_variance_history) == 3
    assert len(tamiyo.kl_divergence_history) == 3
    assert len(tamiyo.clip_fraction_history) == 3

    # Values should be in order (use approximate comparison for floats)
    policy_losses = list(tamiyo.policy_loss_history)
    assert len(policy_losses) == 3
    assert abs(policy_losses[0] - 0.1) < 1e-9
    assert abs(policy_losses[1] - 0.2) < 1e-9
    assert abs(policy_losses[2] - 0.3) < 1e-9

    value_losses = list(tamiyo.value_loss_history)
    assert len(value_losses) == 3
    assert abs(value_losses[0] - 0.2) < 1e-9
    assert abs(value_losses[1] - 0.4) < 1e-9
    assert abs(value_losses[2] - 0.6) < 1e-9

    grad_norms = list(tamiyo.grad_norm_history)
    assert grad_norms == [1.0, 2.0, 3.0]

    entropies = list(tamiyo.entropy_history)
    assert len(entropies) == 3
    assert abs(entropies[0] - 1.5) < 1e-9
    assert abs(entropies[1] - 1.4) < 1e-9
    assert abs(entropies[2] - 1.3) < 1e-9

    explained_variances = list(tamiyo.explained_variance_history)
    assert len(explained_variances) == 3
    assert abs(explained_variances[0] - 0.3) < 1e-9
    assert abs(explained_variances[1] - 0.6) < 1e-9
    assert abs(explained_variances[2] - 0.9) < 1e-9

    kl_divergences = list(tamiyo.kl_divergence_history)
    assert kl_divergences == [0.01, 0.02, 0.03]

    clip_fractions = list(tamiyo.clip_fraction_history)
    assert len(clip_fractions) == 3
    assert abs(clip_fractions[0] - 0.1) < 1e-9
    assert abs(clip_fractions[1] - 0.12) < 1e-9
    assert abs(clip_fractions[2] - 0.14) < 1e-9


def test_get_snapshot_returns_isolated_copy() -> None:
    """get_snapshot() must never expose live, mutable aggregator state."""
    agg = SanctumAggregator(num_envs=2)

    snapshot1 = agg.get_snapshot()
    snapshot1.envs[0].host_accuracy = 123.0
    snapshot1.envs[0].action_counts["WAIT"] = 999
    snapshot1.tamiyo.policy_loss_history.append(9.99)

    snapshot2 = agg.get_snapshot()
    assert snapshot2.envs[0].host_accuracy == 0.0
    assert snapshot2.envs[0].action_counts["WAIT"] == 0
    assert list(snapshot2.tamiyo.policy_loss_history) == []


def test_ppo_update_populates_head_entropies():
    """PPO_UPDATE_COMPLETED should populate all 8 head entropies when available."""
    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.2,
            entropy=1.5,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
            # Per-head entropies (when neural network emits them)
            head_slot_entropy=1.0,
            head_blueprint_entropy=2.0,
            head_style_entropy=1.2,
            head_tempo_entropy=0.9,
            head_alpha_target_entropy=0.8,
            head_alpha_speed_entropy=1.1,
            head_alpha_curve_entropy=0.7,
            head_op_entropy=1.5,
        ),
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    tamiyo = snapshot.tamiyo

    # Should have all 8 head entropies
    assert tamiyo.head_slot_entropy == 1.0
    assert tamiyo.head_blueprint_entropy == 2.0
    assert tamiyo.head_style_entropy == 1.2
    assert tamiyo.head_tempo_entropy == 0.9
    assert tamiyo.head_alpha_target_entropy == 0.8
    assert tamiyo.head_alpha_speed_entropy == 1.1
    assert tamiyo.head_alpha_curve_entropy == 0.7
    assert tamiyo.head_op_entropy == 1.5


def test_ppo_update_extracts_group_id():
    """PPO_UPDATE_COMPLETED should extract group_id for A/B testing."""
    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
        ),
        group_id="B",
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    assert snapshot.tamiyo.group_id == "B"


def test_ppo_update_filters_default_group_id():
    """group_id='default' should NOT set tamiyo.group_id (single-policy mode)."""
    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
        ),
        group_id="default",  # This is the default for single-policy
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    # Should NOT be set to "default" - that would show [default] label
    assert snapshot.tamiyo.group_id is None


def test_seed_gate_evaluated_event_is_handled() -> None:
    """SEED_GATE_EVALUATED must not crash Sanctum aggregation."""
    agg = SanctumAggregator(num_envs=1)

    event = TelemetryEvent(
        event_type=TelemetryEventType.SEED_GATE_EVALUATED,
        data=SeedGateEvaluatedPayload(
            slot_id="r0c0",
            env_id=0,
            gate="G2",
            passed=False,
            target_stage="BLENDING",
            checks_passed=(),
            checks_failed=("insufficient_contribution",),
            message="not ready",
        ),
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    assert snapshot.envs[0].seeds["r0c0"].slot_id == "r0c0"

    assert snapshot.event_log, "Expected gate event to be present in event_log"
    entry = snapshot.event_log[-1]
    assert entry.event_type == "SEED_GATE_EVALUATED"
    assert entry.env_id == 0
    assert entry.metadata["gate"] == "G2"
    assert entry.metadata["result"] == "FAIL"


def test_ppo_update_with_none_group_id():
    """group_id=None should leave tamiyo.group_id as None."""
    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
        ),
        group_id=None,
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    assert snapshot.tamiyo.group_id is None


def test_ppo_update_group_id_transition():
    """Group ID change (A→B) should update tamiyo.group_id."""
    agg = SanctumAggregator(num_envs=4)

    # First event with group A
    event_a = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
        ),
        group_id="A",
    )
    agg.process_event(event_a)
    assert agg.get_snapshot().tamiyo.group_id == "A"

    # Second event with group B
    event_b = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
        ),
        group_id="B",
    )
    agg.process_event(event_b)
    assert agg.get_snapshot().tamiyo.group_id == "B"


def test_ppo_update_group_c():
    """Group C should be accepted (not just A and B)."""
    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
        ),
        group_id="C",
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    assert snapshot.tamiyo.group_id == "C"


def test_ppo_update_unknown_group_id():
    """Unknown group_id (e.g., 'experiment_42') should be accepted."""
    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
        ),
        group_id="experiment_42",  # Arbitrary identifier
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    assert snapshot.tamiyo.group_id == "experiment_42"


def test_snapshot_tracks_last_action_env_id():
    """Snapshot should track which env received the last action."""
    from esper.leyline.telemetry import AnalyticsSnapshotPayload

    agg = SanctumAggregator(num_envs=4)

    # Simulate an action on env 2 via ANALYTICS_SNAPSHOT(last_action)
    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        data=AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=2,
            action_name="GERMINATE",
            action_confidence=0.85,
            slot_id="slot_0",
            blueprint_id="conv_light",
        ),
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    assert snapshot.last_action_env_id == 2
    assert snapshot.last_action_timestamp is not None


def test_decision_snapshot_populates_head_choices():
    """DecisionSnapshot should include blueprint, tempo, style, curve from payload."""
    from esper.leyline.telemetry import AnalyticsSnapshotPayload

    agg = SanctumAggregator(num_envs=4)

    # Simulate a GERMINATE action with head choice details
    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        data=AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=0,
            action_name="GERMINATE",
            action_confidence=0.92,
            slot_id="slot_0",
            blueprint_id="conv_light",
            tempo_idx=1,  # STANDARD (index 1 in TEMPO_NAMES)
            style="LINEAR_ADD",
            alpha_curve="LINEAR",
        ),
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    decisions = snapshot.tamiyo.recent_decisions
    assert len(decisions) == 1

    decision = decisions[0]
    assert decision.chosen_action == "GERMINATE"
    assert decision.chosen_blueprint == "conv_light"
    assert decision.chosen_tempo == "STANDARD"
    assert decision.chosen_style == "LINEAR_ADD"
    assert decision.chosen_curve == "LINEAR"


def test_decision_snapshot_handles_missing_head_choices():
    """DecisionSnapshot should handle None head choice fields gracefully."""
    from esper.leyline.telemetry import AnalyticsSnapshotPayload

    agg = SanctumAggregator(num_envs=4)

    # Simulate a WAIT action (no head choices)
    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        data=AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=0,
            action_name="WAIT",
            action_confidence=0.85,
            # No blueprint_id, tempo_idx, style, or alpha_curve
        ),
    )
    agg.process_event(event)

    snapshot = agg.get_snapshot()
    decisions = snapshot.tamiyo.recent_decisions
    assert len(decisions) == 1

    decision = decisions[0]
    assert decision.chosen_action == "WAIT"
    assert decision.chosen_blueprint is None
    assert decision.chosen_tempo is None
    assert decision.chosen_style is None
    assert decision.chosen_curve is None


def test_aggregator_reads_reward_components_dataclass():
    """Aggregator should read from nested RewardComponentsTelemetry."""
    from datetime import datetime, timezone
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline.telemetry import AnalyticsSnapshotPayload, TelemetryEvent, TelemetryEventType
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

    agg = SanctumAggregator(num_envs=1)
    agg._connected = True
    agg._ensure_env(0)

    rc = RewardComponentsTelemetry(
        bounded_attribution=0.3,
        compute_rent=-0.05,
        stage_bonus=0.1,
        ratio_penalty=0.0,
        alpha_shock=0.0,
        base_acc_delta=0.02,
        hindsight_credit=0.05,
        total_reward=0.42,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        total_reward=0.42,
        action_name="WAIT",
        action_confidence=0.8,
        reward_components=rc,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=10,
    )

    agg.process_event(event)

    env = agg._envs[0]
    assert env.reward_components.bounded_attribution == 0.3
    assert env.reward_components.compute_rent == -0.05
    assert env.reward_components.stage_bonus == 0.1
    assert env.reward_components.hindsight_credit == 0.05


def test_aggregator_populates_nested_metrics():
    """Aggregator should populate infrastructure and gradient_quality nested fields."""
    agg = SanctumAggregator(num_envs=4)

    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        data=PPOUpdatePayload(
            policy_loss=0.1,
            value_loss=0.2,
            entropy=1.0,
            grad_norm=0.5,
            kl_divergence=0.01,
            clip_fraction=0.15,
            nan_grad_count=0,
            # Gradient quality fields
            clip_fraction_positive=0.10,
            clip_fraction_negative=0.05,
            gradient_cv=0.42,
            # Infrastructure fields
            cuda_memory_allocated_gb=4.2,
            cuda_memory_reserved_gb=8.0,
            cuda_memory_peak_gb=6.5,
            cuda_memory_fragmentation=0.475,
        ),
    )

    agg.process_event(event)
    snapshot = agg.get_snapshot()

    # Gradient quality (nested)
    assert snapshot.tamiyo.gradient_quality.clip_fraction_positive == 0.10
    assert snapshot.tamiyo.gradient_quality.clip_fraction_negative == 0.05
    assert snapshot.tamiyo.gradient_quality.gradient_cv == 0.42

    # Infrastructure (nested)
    assert snapshot.tamiyo.infrastructure.cuda_memory_allocated_gb == 4.2
    assert snapshot.tamiyo.infrastructure.cuda_memory_reserved_gb == 8.0
    assert snapshot.tamiyo.infrastructure.cuda_memory_peak_gb == 6.5
    assert snapshot.tamiyo.infrastructure.cuda_memory_fragmentation == 0.475


def test_decision_snapshot_populates_from_head_telemetry():
    """DecisionSnapshot should include confidence and entropy from HeadTelemetry."""
    from datetime import datetime, timezone
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline.telemetry import (
        AnalyticsSnapshotPayload,
        HeadTelemetry,
        TelemetryEvent,
        TelemetryEventType,
        TrainingStartedPayload,
    )

    agg = SanctumAggregator()

    # Initialize with training started (all required fields)
    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        epoch=0,
        data=TrainingStartedPayload(
            n_envs=1,
            max_epochs=25,
            max_batches=100,
            task="mnist",
            host_params=1000000,
            slot_ids=("r0c0", "r0c1"),
            seed=42,
            n_episodes=100,
            lr=3e-4,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=500000,
            policy_device="cuda:0",
            env_devices=("cuda:0",),
            reward_mode="shaped",
        ),
    ))

    # Send last_action with HeadTelemetry
    head_telem = HeadTelemetry(
        op_confidence=0.85,
        slot_confidence=0.72,
        blueprint_confidence=0.91,
        style_confidence=0.65,
        tempo_confidence=0.88,
        alpha_target_confidence=0.77,
        alpha_speed_confidence=0.69,
        curve_confidence=0.82,
        op_entropy=0.3,
        slot_entropy=0.8,
        blueprint_entropy=0.5,
        style_entropy=0.6,
        tempo_entropy=0.4,
        alpha_target_entropy=0.55,
        alpha_speed_entropy=0.45,
        curve_entropy=0.35,
    )

    agg.process_event(TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        epoch=1,
        timestamp=datetime.now(timezone.utc),
        data=AnalyticsSnapshotPayload(
            kind="last_action",
            env_id=0,
            action_name="GERMINATE",
            slot_id="r0c0",
            blueprint_id="conv_light",
            style="LINEAR_ADD",
            tempo_idx=1,
            alpha_target=0.7,
            alpha_speed="MEDIUM",
            alpha_curve="COSINE",
            action_confidence=0.85,
            head_telemetry=head_telem,
        ),
    ))

    snapshot = agg.get_snapshot()
    decisions = snapshot.tamiyo.recent_decisions
    assert len(decisions) == 1

    decision = decisions[0]
    # Confidence
    assert decision.op_confidence == 0.85
    assert decision.slot_confidence == 0.72
    assert decision.blueprint_confidence == 0.91
    assert decision.style_confidence == 0.65
    assert decision.tempo_confidence == 0.88
    assert decision.alpha_target_confidence == 0.77
    assert decision.alpha_speed_confidence == 0.69
    assert decision.curve_confidence == 0.82
    # Entropy
    assert decision.op_entropy == 0.3
    assert decision.slot_entropy == 0.8
    assert decision.blueprint_entropy == 0.5
    assert decision.style_entropy == 0.6
    assert decision.tempo_entropy == 0.4
    assert decision.alpha_target_entropy == 0.55
    assert decision.alpha_speed_entropy == 0.45
    assert decision.curve_entropy == 0.35


def test_aggregator_populates_compile_status():
    """Aggregator should populate compile status from TrainingStartedPayload."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline import TelemetryEvent, TelemetryEventType, TrainingStartedPayload

    aggregator = SanctumAggregator()

    # TrainingStartedPayload has many required fields - must include all
    event = TelemetryEvent(
        event_type=TelemetryEventType.TRAINING_STARTED,
        data=TrainingStartedPayload(
            # Required fields (from leyline/telemetry.py TrainingStartedPayload)
            n_envs=4,
            max_epochs=25,
            max_batches=100,
            task="mnist",
            host_params=1000000,
            slot_ids=("slot_0", "slot_1", "slot_2"),
            seed=42,
            n_episodes=100,
            lr=3e-4,
            clip_ratio=0.2,
            entropy_coef=0.01,
            param_budget=500000,
            policy_device="cuda:0",
            env_devices=("cuda:0", "cuda:0", "cuda:0", "cuda:0"),
            reward_mode="shaped",
            # The fields we're testing
            compile_enabled=True,
            compile_backend="inductor",
            compile_mode="reduce-overhead",
        ),
    )

    aggregator.process_event(event)
    snapshot = aggregator.get_snapshot()

    assert snapshot.tamiyo.infrastructure.compile_enabled is True
    assert snapshot.tamiyo.infrastructure.compile_backend == "inductor"
    assert snapshot.tamiyo.infrastructure.compile_mode == "reduce-overhead"


def test_aggregator_wires_q_values():
    """Aggregator wires Q-values from PPO_UPDATE_COMPLETED to TamiyoState."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline.telemetry import TelemetryEvent, TelemetryEventType, PPOUpdatePayload

    aggregator = SanctumAggregator(num_envs=4)

    # Emit PPO_UPDATE_COMPLETED with Q-values
    event = TelemetryEvent(
        event_type=TelemetryEventType.PPO_UPDATE_COMPLETED,
        epoch=1,
        data=PPOUpdatePayload(
            policy_loss=0.5,
            value_loss=0.3,
            entropy=1.2,
            grad_norm=2.0,
            kl_divergence=0.01,
            clip_fraction=0.15,
            nan_grad_count=0,
            q_germinate=5.2,
            q_advance=3.1,
            q_fossilize=2.8,
            q_prune=-1.5,
            q_wait=0.5,
            q_set_alpha=4.0,
            q_variance=2.3,
            q_spread=6.7,
        ),
    )

    aggregator.process_event(event)
    snapshot = aggregator.get_snapshot()

    # Verify Q-values are wired to TamiyoState
    assert snapshot.tamiyo.q_germinate == 5.2
    assert snapshot.tamiyo.q_advance == 3.1
    assert snapshot.tamiyo.q_fossilize == 2.8
    assert snapshot.tamiyo.q_prune == -1.5
    assert snapshot.tamiyo.q_wait == 0.5
    assert snapshot.tamiyo.q_set_alpha == 4.0
    assert snapshot.tamiyo.q_variance == 2.3
    assert snapshot.tamiyo.q_spread == 6.7
