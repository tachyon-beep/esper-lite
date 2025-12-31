"""Test that ANALYTICS_SNAPSHOT(last_action) populates PolicySnapshot.

This test verifies the fix for the removed REWARD_COMPUTED handler.
The new handler extracts action/reward data from ANALYTICS_SNAPSHOT events.
"""

import pytest

from esper.karn.collector import KarnCollector
from esper.leyline.telemetry import (
    TelemetryEvent,
    TelemetryEventType,
    TrainingStartedPayload,
    AnalyticsSnapshotPayload,
)


def make_training_started_payload(**kwargs) -> TrainingStartedPayload:
    """Create TrainingStartedPayload with sensible defaults."""
    defaults = {
        "seed": 42,
        "task": "classification",
        "reward_mode": "shaped",
        "max_epochs": 10,
        "max_batches": 100,
        "n_envs": 1,
        "host_params": 1000,
        "slot_ids": ("slot0",),
        "n_episodes": 1,
        "lr": 3e-4,
        "clip_ratio": 0.2,
        "entropy_coef": 0.01,
        "param_budget": 10000,
        "policy_device": "cpu",
        "env_devices": ("cpu",),
    }
    defaults.update(kwargs)
    return TrainingStartedPayload(**defaults)


class TestAnalyticsSnapshotPopulatesPolicy:
    """Verify ANALYTICS_SNAPSHOT(last_action) updates PolicySnapshot."""

    @pytest.fixture
    def collector(self) -> KarnCollector:
        """Fresh collector for each test."""
        return KarnCollector()

    def test_last_action_populates_reward_total(self, collector: KarnCollector):
        """ANALYTICS_SNAPSHOT(last_action) should set policy.reward_total."""
        # Start episode
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=make_training_started_payload(),
        ))

        # Emit ANALYTICS_SNAPSHOT with last_action data
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            epoch=1,
            data=AnalyticsSnapshotPayload(
                kind="last_action",
                env_id=0,
                total_reward=0.75,
                action_name="GERMINATE",
                action_confidence=0.85,
                value_estimate=0.5,
            ),
        ))

        # Verify policy snapshot was populated
        assert collector.store.current_epoch is not None
        assert collector.store.current_epoch.policy is not None
        policy = collector.store.current_epoch.policy

        assert policy.reward_total == 0.75
        assert policy.action_op == "GERMINATE"
        assert policy.value_estimate == 0.5

    def test_ignores_non_last_action_kinds(self, collector: KarnCollector):
        """Other ANALYTICS_SNAPSHOT kinds should not create PolicySnapshot."""
        # Start episode
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=make_training_started_payload(),
        ))

        # Emit different kind of ANALYTICS_SNAPSHOT
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            epoch=1,
            data=AnalyticsSnapshotPayload(
                kind="action_distribution",
                action_counts={"WAIT": 5, "GERMINATE": 3},
            ),
        ))

        # Policy should NOT be created
        assert collector.store.current_epoch is not None
        assert collector.store.current_epoch.policy is None

    def test_handles_partial_data(self, collector: KarnCollector):
        """last_action with only some fields should not fail."""
        # Start episode
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data=make_training_started_payload(),
        ))

        # Emit ANALYTICS_SNAPSHOT with only action_name
        collector.emit(TelemetryEvent(
            event_type=TelemetryEventType.ANALYTICS_SNAPSHOT,
            epoch=1,
            data=AnalyticsSnapshotPayload(
                kind="last_action",
                env_id=0,
                action_name="WAIT",
                # total_reward and value_estimate intentionally omitted
            ),
        ))

        # Should still work
        assert collector.store.current_epoch is not None
        assert collector.store.current_epoch.policy is not None
        policy = collector.store.current_epoch.policy

        assert policy.action_op == "WAIT"
        assert policy.reward_total == 0.0  # default
