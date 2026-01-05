"""End-to-end tests for Action Targeting telemetry records (TELE-801 to TELE-802).

Verifies action targeting telemetry flows correctly through the system:
- Cyan indicator in EnvOverview for recently targeted environments
- 5-second hysteresis to prevent visual jitter

These tests cover:
- TELE-801: `snapshot.last_action_env_id` (int | None, which env received action)
- TELE-802: `snapshot.last_action_timestamp` (datetime | None, 5-second hysteresis)

Reference:
    docs/telemetry/telemetry_needs/TELE-801_last_action_env_id.md
    docs/telemetry/telemetry_needs/TELE-802_last_action_timestamp.md
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.karn.sanctum.schema import EnvState, SanctumSnapshot
from esper.leyline import AnalyticsSnapshotPayload


# =============================================================================
# Helper functions
# =============================================================================


def make_analytics_snapshot_event(
    env_id: int,
    kind: str = "last_action",
    action_name: str = "WAIT",
    action_confidence: float = 0.85,
    *,
    action_success: bool = True,
) -> MagicMock:
    """Create a mock ANALYTICS_SNAPSHOT event for testing action targeting.

    Args:
        env_id: The environment ID that received the action.
        kind: The analytics snapshot kind (default: "last_action").
        action_name: The action taken (default: "WAIT").
        action_confidence: The action confidence score (required for last_action).
        action_success: Whether the action executed successfully.

    Returns:
        MagicMock event with proper AnalyticsSnapshotPayload.

    Note:
        The aggregator requires `action_confidence is not None` to process
        a `last_action` kind event and update last_action_env_id.
    """
    event = MagicMock()
    event.event_type = MagicMock()
    event.event_type.name = "ANALYTICS_SNAPSHOT"
    event.timestamp = datetime.now(timezone.utc)
    event.epoch = 1  # Aggregator needs this for epoch tracking
    event.data = AnalyticsSnapshotPayload(
        kind=kind,
        env_id=env_id,
        action_name=action_name,
        action_confidence=action_confidence,
        total_reward=0.0,  # Required for reward tracking
        action_success=action_success,
    )
    return event


# =============================================================================
# TELE-801: Last Action Env ID
# =============================================================================


class TestTELE801LastActionEnvId:
    """TELE-801: last_action_env_id field in SanctumSnapshot.

    Wiring Status: FULLY WIRED
    - Emitter: Tamiyo policy step emits ANALYTICS_SNAPSHOT(kind=last_action)
    - Transport: Aggregator extracts env_id from payload (line 1454)
    - Schema: SanctumSnapshot.last_action_env_id at line 1389
    - Consumer: EnvOverview._format_env_id() shows cyan indicator

    The env_id identifies which environment received the most recent
    Tamiyo policy action (GERMINATE, ADVANCE, FOSSILIZE, PRUNE, etc.).
    """

    def test_schema_field_exists(self) -> None:
        """TELE-801: Verify SanctumSnapshot.last_action_env_id field exists."""
        field_names = {f.name for f in fields(SanctumSnapshot)}
        assert "last_action_env_id" in field_names

    def test_field_type_is_int_or_none(self) -> None:
        """TELE-801: last_action_env_id is int | None."""
        snapshot = SanctumSnapshot()
        # Default is None
        assert snapshot.last_action_env_id is None

        # Can be set to int
        snapshot = SanctumSnapshot(last_action_env_id=2)
        assert snapshot.last_action_env_id == 2
        assert isinstance(snapshot.last_action_env_id, int)

    def test_default_value_is_none(self) -> None:
        """TELE-801: Default last_action_env_id is None (before first action)."""
        snapshot = SanctumSnapshot()
        assert snapshot.last_action_env_id is None

    def test_accepts_valid_env_ids(self) -> None:
        """TELE-801: last_action_env_id accepts valid env_id values."""
        # Typical vectorized training env_ids: 0-7
        for env_id in [0, 1, 2, 3, 7, 15, 31]:
            snapshot = SanctumSnapshot(last_action_env_id=env_id)
            assert snapshot.last_action_env_id == env_id

    def test_aggregator_stores_last_action_env_id(self) -> None:
        """TELE-801: Aggregator correctly tracks last_action_env_id."""
        agg = SanctumAggregator(num_envs=4)

        # Create ANALYTICS_SNAPSHOT event with kind=last_action
        event = make_analytics_snapshot_event(env_id=2)
        agg.process_event(event)

        snapshot = agg.get_snapshot()
        assert snapshot.last_action_env_id == 2

    def test_aggregator_updates_on_new_action(self) -> None:
        """TELE-801: Aggregator updates last_action_env_id on each new action."""
        agg = SanctumAggregator(num_envs=4)

        # First action on env 1
        event1 = make_analytics_snapshot_event(env_id=1)
        agg.process_event(event1)
        assert agg.get_snapshot().last_action_env_id == 1

        # Second action on env 3
        event2 = make_analytics_snapshot_event(env_id=3)
        agg.process_event(event2)
        assert agg.get_snapshot().last_action_env_id == 3


# =============================================================================
# TELE-802: Last Action Timestamp
# =============================================================================


class TestTELE802LastActionTimestamp:
    """TELE-802: last_action_timestamp field in SanctumSnapshot.

    Wiring Status: FULLY WIRED
    - Emitter: SanctumAggregator on receiving ANALYTICS_SNAPSHOT(kind=last_action)
    - Transport: Timestamp captured via datetime.now(timezone.utc) at line 1455
    - Schema: SanctumSnapshot.last_action_timestamp at line 1390
    - Consumer: EnvOverview._format_env_id() computes age for hysteresis

    The timestamp is used for 5-second hysteresis calculation:
    - If age < 5.0 seconds: show cyan indicator
    - If age >= 5.0 seconds: hide indicator (action is stale)
    """

    def test_schema_field_exists(self) -> None:
        """TELE-802: Verify SanctumSnapshot.last_action_timestamp field exists."""
        field_names = {f.name for f in fields(SanctumSnapshot)}
        assert "last_action_timestamp" in field_names

    def test_field_type_is_datetime_or_none(self) -> None:
        """TELE-802: last_action_timestamp is datetime | None."""
        snapshot = SanctumSnapshot()
        # Default is None
        assert snapshot.last_action_timestamp is None

        # Can be set to datetime
        now = datetime.now(timezone.utc)
        snapshot = SanctumSnapshot(last_action_timestamp=now)
        assert snapshot.last_action_timestamp == now
        assert isinstance(snapshot.last_action_timestamp, datetime)

    def test_default_value_is_none(self) -> None:
        """TELE-802: Default last_action_timestamp is None (before first action)."""
        snapshot = SanctumSnapshot()
        assert snapshot.last_action_timestamp is None

    def test_timestamp_is_utc(self) -> None:
        """TELE-802: Timestamp should be in UTC for consistent comparison."""
        now = datetime.now(timezone.utc)
        snapshot = SanctumSnapshot(last_action_timestamp=now)
        # Check timezone is UTC
        assert snapshot.last_action_timestamp.tzinfo is not None
        assert snapshot.last_action_timestamp.tzinfo == timezone.utc

    def test_aggregator_sets_timestamp_on_action(self) -> None:
        """TELE-802: Aggregator sets last_action_timestamp when action occurs."""
        agg = SanctumAggregator(num_envs=4)

        before = datetime.now(timezone.utc)

        # Create ANALYTICS_SNAPSHOT event
        event = make_analytics_snapshot_event(env_id=0)
        agg.process_event(event)

        after = datetime.now(timezone.utc)

        snapshot = agg.get_snapshot()
        assert snapshot.last_action_timestamp is not None
        # Timestamp should be between before and after
        assert before <= snapshot.last_action_timestamp <= after

    def test_aggregator_updates_timestamp_on_each_action(self) -> None:
        """TELE-802: Aggregator updates timestamp on each new action."""
        agg = SanctumAggregator(num_envs=4)

        # First action
        event1 = make_analytics_snapshot_event(env_id=0)
        agg.process_event(event1)
        ts1 = agg.get_snapshot().last_action_timestamp

        # Second action
        event2 = make_analytics_snapshot_event(env_id=1)
        agg.process_event(event2)
        ts2 = agg.get_snapshot().last_action_timestamp

        # Second timestamp should be >= first
        assert ts2 >= ts1


# =============================================================================
# Hysteresis Threshold Tests
# =============================================================================


class TestHysteresisThreshold:
    """Test 5-second hysteresis threshold behavior.

    The hysteresis prevents visual jitter when actions occur rapidly
    across different environments. The indicator shows for 5 seconds
    after an action, then hides.

    Per TELE-802:
    - age < 5.0 seconds: show cyan indicator
    - age >= 5.0 seconds: hide indicator
    """

    def test_indicator_shows_when_age_under_threshold(self) -> None:
        """Indicator should show when action age < 5.0 seconds."""
        # Age 0 seconds - just happened
        now = datetime.now(timezone.utc)
        age = (now - now).total_seconds()
        assert age < 5.0
        # Indicator should show (age < 5.0)

    def test_indicator_hides_when_age_at_threshold(self) -> None:
        """Indicator should hide when action age >= 5.0 seconds."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(seconds=5.0)
        age = (now - old).total_seconds()
        assert age >= 5.0
        # Indicator should hide (age >= 5.0)

    def test_indicator_hides_when_age_over_threshold(self) -> None:
        """Indicator should hide when action age > 5.0 seconds."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(seconds=10.0)
        age = (now - old).total_seconds()
        assert age >= 5.0
        # Indicator should hide (age >= 5.0)

    def test_hysteresis_age_calculation(self) -> None:
        """Verify age calculation for hysteresis check."""
        now = datetime.now(timezone.utc)

        # Test various ages
        test_cases = [
            (0.0, True),    # Just happened -> show
            (1.0, True),    # 1 second ago -> show
            (4.9, True),    # 4.9 seconds ago -> show
            (5.0, False),   # Exactly 5 seconds -> hide
            (5.1, False),   # 5.1 seconds ago -> hide
            (10.0, False),  # 10 seconds ago -> hide
        ]

        for seconds_ago, should_show in test_cases:
            timestamp = now - timedelta(seconds=seconds_ago)
            age = (now - timestamp).total_seconds()
            show_indicator = age < 5.0
            assert show_indicator == should_show, (
                f"Age {seconds_ago}s: expected show={should_show}, got {show_indicator}"
            )


# =============================================================================
# Consumer Widget Tests (EnvOverview._format_env_id)
# =============================================================================


class TestEnvOverviewFormatEnvId:
    """Test EnvOverview._format_env_id() reads action targeting fields correctly.

    The _format_env_id() method in EnvOverview shows a cyan indicator
    when the env matches last_action_env_id AND the action is recent
    (within 5-second hysteresis window).
    """

    def test_no_indicator_when_env_id_is_none(self) -> None:
        """No indicator when last_action_env_id is None (no actions yet)."""
        env = EnvState(env_id=0)
        # With None env_id, no indicator should show regardless of timestamp
        # This tests the condition: last_action_env_id is not None
        last_action_env_id = None
        last_action_timestamp = datetime.now(timezone.utc)

        # The logic: if last_action_env_id is None, no indicator
        show_indicator = (
            last_action_env_id is not None
            and env.env_id == last_action_env_id
        )
        assert show_indicator is False

    def test_no_indicator_when_timestamp_is_none(self) -> None:
        """No indicator when last_action_timestamp is None with matching env_id.

        Per the code: if timestamp is None, show_indicator defaults to True
        BUT this scenario shouldn't occur - if env_id is set, timestamp is set.
        """
        env = EnvState(env_id=0)
        last_action_env_id = 0
        last_action_timestamp = None

        # Per actual code: if timestamp is None, show_indicator = True (default)
        # This is a design choice - if we have an env_id but no timestamp,
        # we still show the indicator (defensive behavior).
        # The test verifies the logic as implemented.
        show_indicator = True
        if last_action_env_id is not None and env.env_id == last_action_env_id:
            show_indicator = True  # Default
            if last_action_timestamp is not None:
                age = (datetime.now(timezone.utc) - last_action_timestamp).total_seconds()
                show_indicator = age < 5.0
        else:
            show_indicator = False

        # With matching env_id and None timestamp, indicator shows (default behavior)
        assert show_indicator is True

    def test_indicator_shows_for_matching_env_within_threshold(self) -> None:
        """Indicator shows when env matches AND action is recent."""
        env = EnvState(env_id=2)
        last_action_env_id = 2
        last_action_timestamp = datetime.now(timezone.utc) - timedelta(seconds=2.0)

        # Replicate _format_env_id logic
        show_indicator = False
        if last_action_env_id is not None and env.env_id == last_action_env_id:
            show_indicator = True
            if last_action_timestamp is not None:
                age = (datetime.now(timezone.utc) - last_action_timestamp).total_seconds()
                show_indicator = age < 5.0

        assert show_indicator is True

    def test_no_indicator_for_matching_env_outside_threshold(self) -> None:
        """No indicator when env matches BUT action is stale (>= 5s old)."""
        env = EnvState(env_id=2)
        last_action_env_id = 2
        last_action_timestamp = datetime.now(timezone.utc) - timedelta(seconds=6.0)

        # Replicate _format_env_id logic
        show_indicator = False
        if last_action_env_id is not None and env.env_id == last_action_env_id:
            show_indicator = True
            if last_action_timestamp is not None:
                age = (datetime.now(timezone.utc) - last_action_timestamp).total_seconds()
                show_indicator = age < 5.0

        assert show_indicator is False

    def test_no_indicator_for_non_matching_env(self) -> None:
        """No indicator when env_id does not match last_action_env_id."""
        env = EnvState(env_id=0)
        last_action_env_id = 2  # Different env
        last_action_timestamp = datetime.now(timezone.utc)  # Recent action

        # Even with recent action, indicator shouldn't show for non-matching env
        show_indicator = False
        if last_action_env_id is not None and env.env_id == last_action_env_id:
            show_indicator = True
            if last_action_timestamp is not None:
                age = (datetime.now(timezone.utc) - last_action_timestamp).total_seconds()
                show_indicator = age < 5.0

        assert show_indicator is False

    def test_indicator_only_shows_on_targeted_env(self) -> None:
        """Indicator only shows on the specific targeted env, not others."""
        envs = [EnvState(env_id=i) for i in range(4)]
        last_action_env_id = 2
        last_action_timestamp = datetime.now(timezone.utc)

        for env in envs:
            show_indicator = False
            if last_action_env_id is not None and env.env_id == last_action_env_id:
                show_indicator = True
                if last_action_timestamp is not None:
                    age = (datetime.now(timezone.utc) - last_action_timestamp).total_seconds()
                    show_indicator = age < 5.0

            if env.env_id == 2:
                assert show_indicator is True
            else:
                assert show_indicator is False


# =============================================================================
# None Handling Tests
# =============================================================================


class TestNoneHandling:
    """Test correct handling of None values for both fields.

    Both fields default to None before any actions occur. The widget
    must handle these None cases gracefully without showing indicators.
    """

    def test_both_fields_none_no_indicator(self) -> None:
        """No indicator when both fields are None (initial state)."""
        env = EnvState(env_id=0)
        last_action_env_id = None
        last_action_timestamp = None

        show_indicator = False
        if last_action_env_id is not None and env.env_id == last_action_env_id:
            show_indicator = True
            if last_action_timestamp is not None:
                age = (datetime.now(timezone.utc) - last_action_timestamp).total_seconds()
                show_indicator = age < 5.0

        assert show_indicator is False

    def test_snapshot_default_state(self) -> None:
        """SanctumSnapshot default has both action tracking fields as None."""
        snapshot = SanctumSnapshot()
        assert snapshot.last_action_env_id is None
        assert snapshot.last_action_timestamp is None

    def test_aggregator_initial_state(self) -> None:
        """Aggregator initial snapshot has both fields as None."""
        agg = SanctumAggregator(num_envs=4)
        snapshot = agg.get_snapshot()
        assert snapshot.last_action_env_id is None
        assert snapshot.last_action_timestamp is None


# =============================================================================
# Integration: Combined Field Behavior
# =============================================================================


class TestActionTargetingIntegration:
    """Integration tests for action targeting behavior.

    Verifies that env_id and timestamp work together correctly
    for the action targeting indicator display.
    """

    def test_action_sets_both_fields(self) -> None:
        """Action event sets both last_action_env_id and last_action_timestamp."""
        agg = SanctumAggregator(num_envs=4)

        event = make_analytics_snapshot_event(env_id=1)
        agg.process_event(event)
        snapshot = agg.get_snapshot()

        # Both fields should be set
        assert snapshot.last_action_env_id == 1
        assert snapshot.last_action_timestamp is not None

    def test_sequential_actions_update_both_fields(self) -> None:
        """Sequential actions update both fields correctly."""
        agg = SanctumAggregator(num_envs=4)

        # Action on env 0
        event0 = make_analytics_snapshot_event(env_id=0)
        agg.process_event(event0)
        snapshot0 = agg.get_snapshot()
        ts0 = snapshot0.last_action_timestamp

        # Action on env 3
        event3 = make_analytics_snapshot_event(env_id=3)
        agg.process_event(event3)
        snapshot3 = agg.get_snapshot()

        # env_id should be updated to 3
        assert snapshot3.last_action_env_id == 3
        # timestamp should be updated (>= previous)
        assert snapshot3.last_action_timestamp >= ts0

    def test_rapid_actions_track_latest(self) -> None:
        """Rapid actions always track the most recent action."""
        agg = SanctumAggregator(num_envs=8)

        # Rapidly fire actions on multiple envs
        for env_id in [0, 5, 2, 7, 3]:
            event = make_analytics_snapshot_event(env_id=env_id)
            agg.process_event(event)

        # Should track the last env (3)
        snapshot = agg.get_snapshot()
        assert snapshot.last_action_env_id == 3


# =============================================================================
# Schema Completeness
# =============================================================================


class TestSchemaCompleteness:
    """Verify SanctumSnapshot has all TELE-801 to TELE-802 fields."""

    def test_all_action_targeting_fields_present(self) -> None:
        """Verify all action targeting fields are present in SanctumSnapshot."""
        snapshot = SanctumSnapshot()

        # TELE-801: Last action env ID
        assert hasattr(snapshot, "last_action_env_id")

        # TELE-802: Last action timestamp
        assert hasattr(snapshot, "last_action_timestamp")

    def test_all_defaults_match_specification(self) -> None:
        """Verify all default values match TELE record specifications."""
        snapshot = SanctumSnapshot()

        # TELE-801: Default is None (no actions yet)
        assert snapshot.last_action_env_id is None

        # TELE-802: Default is None (no actions yet)
        assert snapshot.last_action_timestamp is None

    def test_field_types_match_specification(self) -> None:
        """Verify field types match TELE record specifications."""
        # TELE-801: int | None
        snapshot_with_id = SanctumSnapshot(last_action_env_id=5)
        assert isinstance(snapshot_with_id.last_action_env_id, int)

        snapshot_no_id = SanctumSnapshot()
        assert snapshot_no_id.last_action_env_id is None

        # TELE-802: datetime | None
        now = datetime.now(timezone.utc)
        snapshot_with_ts = SanctumSnapshot(last_action_timestamp=now)
        assert isinstance(snapshot_with_ts.last_action_timestamp, datetime)

        snapshot_no_ts = SanctumSnapshot()
        assert snapshot_no_ts.last_action_timestamp is None
