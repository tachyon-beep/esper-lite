"""Tests for telemetry payload dataclasses."""


def test_analytics_snapshot_payload_accepts_reward_components_dataclass():
    """AnalyticsSnapshotPayload should accept RewardComponentsTelemetry."""
    from esper.leyline.telemetry import AnalyticsSnapshotPayload
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

    rc = RewardComponentsTelemetry(
        bounded_attribution=0.5,
        compute_rent=-0.1,
        seed_stage=2,
        action_shaping=0.05,
        total_reward=0.45,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        reward_components=rc,
    )

    assert payload.reward_components is rc
    assert payload.reward_components.seed_stage == 2
    assert payload.reward_components.bounded_attribution == 0.5


def test_telemetry_event_serializes_nested_reward_components():
    """TelemetryEvent should serialize reward_components dataclass to JSON.

    The serialized JSON should include the shaped_reward_ratio property,
    which is only available via to_dict() (not asdict()).
    """
    import json
    from datetime import datetime, timezone

    from esper.karn.serialization import serialize_event
    from esper.leyline.telemetry import (
        AnalyticsSnapshotPayload,
        TelemetryEvent,
        TelemetryEventType,
    )
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

    rc = RewardComponentsTelemetry(
        bounded_attribution=0.5,
        seed_stage=2,
        total_reward=0.45,
    )

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        reward_components=rc,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=10,
    )

    json_str = serialize_event(event)
    parsed = json.loads(json_str)

    # Verify nested structure in JSON
    assert "reward_components" in parsed["data"]
    assert parsed["data"]["reward_components"]["seed_stage"] == 2
    assert parsed["data"]["reward_components"]["bounded_attribution"] == 0.5

    # Critical: shaped_reward_ratio should be included (from to_dict())
    assert "shaped_reward_ratio" in parsed["data"]["reward_components"]


# =============================================================================
# PPOUpdatePayload Extensions (per UX specialist enhancements plan)
# =============================================================================


def test_ppo_update_payload_has_gradient_quality_metrics():
    """PPOUpdatePayload should include directional clip and gradient CV fields."""
    from esper.leyline.telemetry import PPOUpdatePayload

    # Create with gradient quality fields
    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.5,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.15,
        nan_grad_count=0,
        # New gradient quality fields
        clip_fraction_positive=0.10,
        clip_fraction_negative=0.05,
        gradient_cv=0.35,
    )

    assert payload.clip_fraction_positive == 0.10
    assert payload.clip_fraction_negative == 0.05
    assert payload.gradient_cv == 0.35


def test_ppo_update_payload_gradient_quality_defaults():
    """New gradient quality fields should default to 0.0."""
    from esper.leyline.telemetry import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.5,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.15,
        nan_grad_count=0,
    )

    assert payload.clip_fraction_positive == 0.0
    assert payload.clip_fraction_negative == 0.0
    assert payload.gradient_cv == 0.0


def test_ppo_update_payload_has_infrastructure_metrics():
    """PPOUpdatePayload should include CUDA memory fields."""
    from esper.leyline.telemetry import PPOUpdatePayload

    payload = PPOUpdatePayload(
        policy_loss=0.1,
        value_loss=0.2,
        entropy=1.5,
        grad_norm=0.5,
        kl_divergence=0.01,
        clip_fraction=0.15,
        nan_grad_count=0,
        # New infrastructure fields
        cuda_memory_allocated_gb=4.2,
        cuda_memory_reserved_gb=8.0,
        cuda_memory_peak_gb=6.5,
        cuda_memory_fragmentation=0.25,
    )

    assert payload.cuda_memory_allocated_gb == 4.2
    assert payload.cuda_memory_reserved_gb == 8.0
    assert payload.cuda_memory_peak_gb == 6.5
    assert payload.cuda_memory_fragmentation == 0.25


def test_ppo_update_payload_from_dict_parses_new_fields():
    """PPOUpdatePayload.from_dict should parse the new fields."""
    from esper.leyline.telemetry import PPOUpdatePayload

    data = {
        "policy_loss": 0.1,
        "value_loss": 0.2,
        "entropy": 1.5,
        "grad_norm": 0.5,
        "kl_divergence": 0.01,
        "clip_fraction": 0.15,
        "nan_grad_count": 0,
        # New fields
        "clip_fraction_positive": 0.12,
        "clip_fraction_negative": 0.08,
        "gradient_cv": 0.45,
        "cuda_memory_allocated_gb": 5.0,
        "cuda_memory_reserved_gb": 10.0,
        "cuda_memory_peak_gb": 7.5,
        "cuda_memory_fragmentation": 0.30,
    }

    payload = PPOUpdatePayload.from_dict(data)

    assert payload.clip_fraction_positive == 0.12
    assert payload.clip_fraction_negative == 0.08
    assert payload.gradient_cv == 0.45
    assert payload.cuda_memory_allocated_gb == 5.0
    assert payload.cuda_memory_reserved_gb == 10.0
    assert payload.cuda_memory_peak_gb == 7.5
    assert payload.cuda_memory_fragmentation == 0.30


# =============================================================================
# HeadTelemetry Dataclass Tests
# =============================================================================


def test_head_telemetry_dataclass():
    """HeadTelemetry should hold confidence and entropy for all 8 heads."""
    from esper.leyline.telemetry import HeadTelemetry

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

    # Verify all confidence fields
    assert head_telem.op_confidence == 0.85
    assert head_telem.slot_confidence == 0.72
    assert head_telem.blueprint_confidence == 0.91
    assert head_telem.style_confidence == 0.65
    assert head_telem.tempo_confidence == 0.88
    assert head_telem.alpha_target_confidence == 0.77
    assert head_telem.alpha_speed_confidence == 0.69
    assert head_telem.curve_confidence == 0.82

    # Verify all entropy fields
    assert head_telem.op_entropy == 0.3
    assert head_telem.slot_entropy == 0.8
    assert head_telem.blueprint_entropy == 0.5
    assert head_telem.style_entropy == 0.6
    assert head_telem.tempo_entropy == 0.4
    assert head_telem.alpha_target_entropy == 0.55
    assert head_telem.alpha_speed_entropy == 0.45
    assert head_telem.curve_entropy == 0.35


def test_analytics_snapshot_payload_accepts_head_telemetry():
    """AnalyticsSnapshotPayload should accept HeadTelemetry."""
    from esper.leyline.telemetry import AnalyticsSnapshotPayload, HeadTelemetry

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

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        head_telemetry=head_telem,
    )

    assert payload.head_telemetry is head_telem
    assert payload.head_telemetry.op_confidence == 0.85
    assert payload.head_telemetry.op_entropy == 0.3


def test_telemetry_event_serializes_nested_head_telemetry():
    """TelemetryEvent should serialize head_telemetry dataclass to JSON.

    Verifies the full round-trip: HeadTelemetry -> JSON -> HeadTelemetry
    """
    import json
    from datetime import datetime, timezone

    from esper.karn.serialization import serialize_event
    from esper.leyline.telemetry import (
        AnalyticsSnapshotPayload,
        HeadTelemetry,
        TelemetryEvent,
        TelemetryEventType,
    )

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

    payload = AnalyticsSnapshotPayload(
        kind="last_action",
        env_id=0,
        head_telemetry=head_telem,
    )

    event = TelemetryEvent(
        event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
        timestamp=datetime.now(timezone.utc),
        data=payload,
        epoch=10,
    )

    # Serialize to JSON
    json_str = serialize_event(event)
    parsed = json.loads(json_str)

    # Verify nested structure in JSON
    assert "head_telemetry" in parsed["data"]
    assert parsed["data"]["head_telemetry"]["op_confidence"] == 0.85
    assert parsed["data"]["head_telemetry"]["slot_entropy"] == 0.8

    # Round-trip: deserialize back to HeadTelemetry via AnalyticsSnapshotPayload.from_dict
    restored_payload = AnalyticsSnapshotPayload.from_dict(parsed["data"])
    assert restored_payload.head_telemetry is not None
    assert restored_payload.head_telemetry.op_confidence == 0.85
    assert restored_payload.head_telemetry.slot_confidence == 0.72
    assert restored_payload.head_telemetry.blueprint_confidence == 0.91
    assert restored_payload.head_telemetry.style_confidence == 0.65
    assert restored_payload.head_telemetry.tempo_confidence == 0.88
    assert restored_payload.head_telemetry.alpha_target_confidence == 0.77
    assert restored_payload.head_telemetry.alpha_speed_confidence == 0.69
    assert restored_payload.head_telemetry.curve_confidence == 0.82
    assert restored_payload.head_telemetry.op_entropy == 0.3
    assert restored_payload.head_telemetry.slot_entropy == 0.8
    assert restored_payload.head_telemetry.blueprint_entropy == 0.5
    assert restored_payload.head_telemetry.style_entropy == 0.6
    assert restored_payload.head_telemetry.tempo_entropy == 0.4
    assert restored_payload.head_telemetry.alpha_target_entropy == 0.55
    assert restored_payload.head_telemetry.alpha_speed_entropy == 0.45
    assert restored_payload.head_telemetry.curve_entropy == 0.35