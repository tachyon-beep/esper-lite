"""Tests for SanctumAggregator telemetry event processing."""

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import PPOUpdatePayload


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
