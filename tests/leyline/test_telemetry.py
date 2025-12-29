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