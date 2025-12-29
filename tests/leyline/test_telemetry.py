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
