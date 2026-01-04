"""End-to-end tests for EventLogEntry telemetry (TELE-810 to TELE-815).

Verifies EventLogEntry schema and EventLog widget behavior for event
log display in Sanctum. Tests cover:

- TELE-810: EventLogEntry schema overview
- TELE-811: timestamp field (HH:MM:SS format)
- TELE-812: event_type field with color mapping
- TELE-813: env_id field (per-env vs global events)
- TELE-814: message field (generic templates)
- TELE-815: metadata dict (structured event details)

Reference:
    docs/telemetry/telemetry_needs/TELE-810_event_log_entry_schema.md
    docs/telemetry/telemetry_needs/TELE-811_event_log_timestamp.md
    docs/telemetry/telemetry_needs/TELE-812_event_log_event_type.md
    docs/telemetry/telemetry_needs/TELE-813_event_log_env_id.md
    docs/telemetry/telemetry_needs/TELE-814_event_log_message.md
    docs/telemetry/telemetry_needs/TELE-815_event_log_metadata.md
"""

from __future__ import annotations

import pytest
from dataclasses import fields

from esper.karn.sanctum.schema import EventLogEntry


# Import widget constants for verification
from esper.karn.sanctum.widgets.event_log import (
    _EVENT_COLORS,
    _INDIVIDUAL_EVENTS,
    _STAGE_SHORT,
)


# =============================================================================
# TELE-810: EventLogEntry Schema Overview
# =============================================================================


class TestTELE810EventLogEntrySchema:
    """TELE-810: EventLogEntry schema field existence and types."""

    def test_schema_has_timestamp_field(self) -> None:
        """TELE-810: EventLogEntry has timestamp field of type str."""
        field_names = {f.name for f in fields(EventLogEntry)}
        assert "timestamp" in field_names

        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert isinstance(entry.timestamp, str)

    def test_schema_has_event_type_field(self) -> None:
        """TELE-810: EventLogEntry has event_type field of type str."""
        field_names = {f.name for f in fields(EventLogEntry)}
        assert "event_type" in field_names

        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert isinstance(entry.event_type, str)

    def test_schema_has_env_id_field(self) -> None:
        """TELE-810: EventLogEntry has env_id field of type int | None."""
        field_names = {f.name for f in fields(EventLogEntry)}
        assert "env_id" in field_names

        # Test with int
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.env_id == 0

        # Test with None (global event)
        entry_global = EventLogEntry(
            timestamp="14:32:05",
            event_type="PPO_UPDATE_COMPLETED",
            env_id=None,
            message="PPO update",
        )
        assert entry_global.env_id is None

    def test_schema_has_message_field(self) -> None:
        """TELE-810: EventLogEntry has message field of type str."""
        field_names = {f.name for f in fields(EventLogEntry)}
        assert "message" in field_names

        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert isinstance(entry.message, str)

    def test_schema_has_episode_field(self) -> None:
        """TELE-810: EventLogEntry has episode field with default 0."""
        field_names = {f.name for f in fields(EventLogEntry)}
        assert "episode" in field_names

        # Default should be 0
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.episode == 0

        # Should be settable
        entry_with_episode = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
            episode=42,
        )
        assert entry_with_episode.episode == 42

    def test_schema_has_relative_time_field(self) -> None:
        """TELE-810: EventLogEntry has relative_time field with default empty string."""
        field_names = {f.name for f in fields(EventLogEntry)}
        assert "relative_time" in field_names

        # Default should be empty
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.relative_time == ""

        # Should be settable
        entry_with_time = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
            relative_time="(2s ago)",
        )
        assert entry_with_time.relative_time == "(2s ago)"

    def test_schema_has_metadata_field(self) -> None:
        """TELE-810: EventLogEntry has metadata field with default empty dict."""
        field_names = {f.name for f in fields(EventLogEntry)}
        assert "metadata" in field_names

        # Default should be empty dict
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.metadata == {}
        assert isinstance(entry.metadata, dict)

    def test_schema_field_count(self) -> None:
        """TELE-810: EventLogEntry has exactly 7 fields as specified."""
        field_names = {f.name for f in fields(EventLogEntry)}
        # timestamp, event_type, env_id, message, episode, relative_time, metadata
        assert len(field_names) == 7


# =============================================================================
# TELE-811: Timestamp Field
# =============================================================================


class TestTELE811Timestamp:
    """TELE-811: timestamp field specifications."""

    def test_timestamp_format_hh_mm_ss(self) -> None:
        """TELE-811: timestamp is in HH:MM:SS 24-hour format."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        # Format validation: HH:MM:SS
        parts = entry.timestamp.split(":")
        assert len(parts) == 3
        assert all(len(p) == 2 for p in parts)  # Two digits each
        assert 0 <= int(parts[0]) <= 23  # Valid hour
        assert 0 <= int(parts[1]) <= 59  # Valid minute
        assert 0 <= int(parts[2]) <= 59  # Valid second

    def test_timestamp_various_times(self) -> None:
        """TELE-811: timestamp accepts various valid times."""
        test_times = ["00:00:00", "12:30:45", "23:59:59", "09:05:03"]
        for time_str in test_times:
            entry = EventLogEntry(
                timestamp=time_str,
                event_type="SEED_GERMINATED",
                env_id=0,
                message="Germinated",
            )
            assert entry.timestamp == time_str


# =============================================================================
# TELE-812: Event Type Field with Color Mapping
# =============================================================================


class TestTELE812EventType:
    """TELE-812: event_type field and color mapping specifications."""

    def test_event_type_accepts_string(self) -> None:
        """TELE-812: event_type field accepts SCREAMING_SNAKE_CASE strings."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.event_type == "SEED_GERMINATED"

    def test_event_colors_seed_germinated(self) -> None:
        """TELE-812: SEED_GERMINATED has bright_yellow color."""
        assert "SEED_GERMINATED" in _EVENT_COLORS
        assert _EVENT_COLORS["SEED_GERMINATED"] == "bright_yellow"

    def test_event_colors_seed_fossilized(self) -> None:
        """TELE-812: SEED_FOSSILIZED has bright_green color."""
        assert "SEED_FOSSILIZED" in _EVENT_COLORS
        assert _EVENT_COLORS["SEED_FOSSILIZED"] == "bright_green"

    def test_event_colors_seed_pruned(self) -> None:
        """TELE-812: SEED_PRUNED has bright_red color."""
        assert "SEED_PRUNED" in _EVENT_COLORS
        assert _EVENT_COLORS["SEED_PRUNED"] == "bright_red"

    def test_event_colors_seed_stage_changed(self) -> None:
        """TELE-812: SEED_STAGE_CHANGED has bright_white color."""
        assert "SEED_STAGE_CHANGED" in _EVENT_COLORS
        assert _EVENT_COLORS["SEED_STAGE_CHANGED"] == "bright_white"

    def test_event_colors_seed_gate_evaluated(self) -> None:
        """TELE-812: SEED_GATE_EVALUATED has bright_white color."""
        assert "SEED_GATE_EVALUATED" in _EVENT_COLORS
        assert _EVENT_COLORS["SEED_GATE_EVALUATED"] == "bright_white"

    def test_event_colors_ppo_update_completed(self) -> None:
        """TELE-812: PPO_UPDATE_COMPLETED has bright_magenta color."""
        assert "PPO_UPDATE_COMPLETED" in _EVENT_COLORS
        assert _EVENT_COLORS["PPO_UPDATE_COMPLETED"] == "bright_magenta"

    def test_event_colors_batch_epoch_completed(self) -> None:
        """TELE-812: BATCH_EPOCH_COMPLETED has bright_blue color."""
        assert "BATCH_EPOCH_COMPLETED" in _EVENT_COLORS
        assert _EVENT_COLORS["BATCH_EPOCH_COMPLETED"] == "bright_blue"

    def test_event_colors_training_started(self) -> None:
        """TELE-812: TRAINING_STARTED has bright_green color."""
        assert "TRAINING_STARTED" in _EVENT_COLORS
        assert _EVENT_COLORS["TRAINING_STARTED"] == "bright_green"

    def test_event_colors_reward_computed(self) -> None:
        """TELE-812: REWARD_COMPUTED has dim color (aggregated event)."""
        assert "REWARD_COMPUTED" in _EVENT_COLORS
        assert _EVENT_COLORS["REWARD_COMPUTED"] == "dim"

    def test_event_colors_epoch_completed(self) -> None:
        """TELE-812: EPOCH_COMPLETED has bright_blue color."""
        assert "EPOCH_COMPLETED" in _EVENT_COLORS
        assert _EVENT_COLORS["EPOCH_COMPLETED"] == "bright_blue"

    def test_individual_events_set(self) -> None:
        """TELE-812: Individual events set contains seed lifecycle and PPO events."""
        expected_individual = {
            "SEED_GERMINATED",
            "SEED_STAGE_CHANGED",
            "SEED_GATE_EVALUATED",
            "SEED_FOSSILIZED",
            "SEED_PRUNED",
            "PPO_UPDATE_COMPLETED",
            "BATCH_EPOCH_COMPLETED",
            "TRAINING_STARTED",
        }
        assert _INDIVIDUAL_EVENTS == expected_individual

    def test_aggregated_events_not_in_individual(self) -> None:
        """TELE-812: Aggregated events are NOT in individual events set."""
        aggregated_events = {"REWARD_COMPUTED", "EPOCH_COMPLETED"}
        for event in aggregated_events:
            assert event not in _INDIVIDUAL_EVENTS


# =============================================================================
# TELE-813: Env ID Field
# =============================================================================


class TestTELE813EnvId:
    """TELE-813: env_id field specifications."""

    def test_env_id_for_per_env_event(self) -> None:
        """TELE-813: Per-environment events have integer env_id."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=2,
            message="Germinated",
        )
        assert entry.env_id == 2

    def test_env_id_for_global_event(self) -> None:
        """TELE-813: Global events have None env_id."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="PPO_UPDATE_COMPLETED",
            env_id=None,
            message="PPO update",
        )
        assert entry.env_id is None

    def test_env_id_range(self) -> None:
        """TELE-813: env_id supports typical vectorized training ranges."""
        for env_id in [0, 1, 2, 3, 7, 15, 31]:
            entry = EventLogEntry(
                timestamp="14:32:05",
                event_type="SEED_GERMINATED",
                env_id=env_id,
                message="Germinated",
            )
            assert entry.env_id == env_id


# =============================================================================
# TELE-814: Message Field
# =============================================================================


class TestTELE814Message:
    """TELE-814: message field specifications."""

    def test_message_germinated(self) -> None:
        """TELE-814: SEED_GERMINATED message template is 'Germinated'."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.message == "Germinated"

    def test_message_stage_changed(self) -> None:
        """TELE-814: SEED_STAGE_CHANGED message template is 'Stage changed'."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_STAGE_CHANGED",
            env_id=0,
            message="Stage changed",
        )
        assert entry.message == "Stage changed"

    def test_message_fossilized(self) -> None:
        """TELE-814: SEED_FOSSILIZED message template is 'Fossilized'."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_FOSSILIZED",
            env_id=0,
            message="Fossilized",
        )
        assert entry.message == "Fossilized"

    def test_message_pruned(self) -> None:
        """TELE-814: SEED_PRUNED message template is 'Pruned'."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_PRUNED",
            env_id=0,
            message="Pruned",
        )
        assert entry.message == "Pruned"

    def test_message_ppo_update(self) -> None:
        """TELE-814: PPO_UPDATE_COMPLETED message template is 'PPO update'."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="PPO_UPDATE_COMPLETED",
            env_id=None,
            message="PPO update",
        )
        assert entry.message == "PPO update"

    def test_message_ppo_skipped(self) -> None:
        """TELE-814: PPO skipped case has message 'PPO skipped'."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="PPO_UPDATE_COMPLETED",
            env_id=None,
            message="PPO skipped",
        )
        assert entry.message == "PPO skipped"

    def test_message_batch_complete(self) -> None:
        """TELE-814: BATCH_EPOCH_COMPLETED message template is 'Batch complete'."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="BATCH_EPOCH_COMPLETED",
            env_id=None,
            message="Batch complete",
        )
        assert entry.message == "Batch complete"

    def test_message_is_generic(self) -> None:
        """TELE-814: Messages are generic without specific values."""
        # Message should NOT contain slot IDs, blueprint names, or numbers
        generic_messages = [
            "Germinated",
            "Stage changed",
            "Fossilized",
            "Pruned",
            "PPO update",
            "Batch complete",
        ]
        for msg in generic_messages:
            # Generic messages don't contain env IDs, slot patterns, or percentages
            assert not any(char.isdigit() for char in msg)
            assert "r0c0" not in msg
            assert "%" not in msg


# =============================================================================
# TELE-815: Metadata Field
# =============================================================================


class TestTELE815Metadata:
    """TELE-815: metadata field specifications."""

    def test_metadata_default_empty_dict(self) -> None:
        """TELE-815: metadata defaults to empty dict."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.metadata == {}

    def test_metadata_event_id(self) -> None:
        """TELE-815: metadata can contain event_id."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
            metadata={"event_id": "abc-123-def"},
        )
        assert entry.metadata["event_id"] == "abc-123-def"

    def test_metadata_seed_germinated_fields(self) -> None:
        """TELE-815: SEED_GERMINATED metadata includes slot_id and blueprint."""
        metadata = {
            "event_id": "abc-123",
            "slot_id": "r0c1",
            "blueprint": "conv_light",
        }
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
            metadata=metadata,
        )
        assert entry.metadata["slot_id"] == "r0c1"
        assert entry.metadata["blueprint"] == "conv_light"

    def test_metadata_seed_stage_changed_fields(self) -> None:
        """TELE-815: SEED_STAGE_CHANGED metadata includes from and to stages."""
        metadata = {
            "event_id": "abc-123",
            "slot_id": "r0c1",
            "from": "TRAINING",
            "to": "BLENDING",
        }
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_STAGE_CHANGED",
            env_id=0,
            message="Stage changed",
            metadata=metadata,
        )
        assert entry.metadata["from"] == "TRAINING"
        assert entry.metadata["to"] == "BLENDING"

    def test_metadata_seed_fossilized_fields(self) -> None:
        """TELE-815: SEED_FOSSILIZED metadata includes improvement."""
        metadata = {
            "event_id": "abc-123",
            "slot_id": "r0c1",
            "improvement": 2.3,
        }
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_FOSSILIZED",
            env_id=0,
            message="Fossilized",
            metadata=metadata,
        )
        assert entry.metadata["improvement"] == 2.3

    def test_metadata_seed_pruned_fields(self) -> None:
        """TELE-815: SEED_PRUNED metadata includes reason."""
        metadata = {
            "event_id": "abc-123",
            "slot_id": "r0c1",
            "reason": "stagnation",
        }
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_PRUNED",
            env_id=0,
            message="Pruned",
            metadata=metadata,
        )
        assert entry.metadata["reason"] == "stagnation"

    def test_metadata_ppo_update_fields(self) -> None:
        """TELE-815: PPO_UPDATE_COMPLETED metadata includes entropy and clip_fraction."""
        metadata = {
            "event_id": "abc-123",
            "entropy": 1.23,
            "clip_fraction": 0.15,
        }
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="PPO_UPDATE_COMPLETED",
            env_id=None,
            message="PPO update",
            metadata=metadata,
        )
        assert entry.metadata["entropy"] == 1.23
        assert entry.metadata["clip_fraction"] == 0.15

    def test_metadata_batch_epoch_fields(self) -> None:
        """TELE-815: BATCH_EPOCH_COMPLETED metadata includes batch and episodes."""
        metadata = {
            "event_id": "abc-123",
            "batch": 42,
            "episodes": 100,
        }
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="BATCH_EPOCH_COMPLETED",
            env_id=None,
            message="Batch complete",
            metadata=metadata,
        )
        assert entry.metadata["batch"] == 42
        assert entry.metadata["episodes"] == 100

    def test_metadata_gate_evaluated_fields(self) -> None:
        """TELE-815: SEED_GATE_EVALUATED metadata includes gate, target_stage, result."""
        metadata = {
            "event_id": "abc-123",
            "slot_id": "r0c1",
            "gate": "G2",
            "target_stage": "BLENDING",
            "result": "PASS",
        }
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GATE_EVALUATED",
            env_id=0,
            message="Gate evaluated",
            metadata=metadata,
        )
        assert entry.metadata["gate"] == "G2"
        assert entry.metadata["target_stage"] == "BLENDING"
        assert entry.metadata["result"] == "PASS"

    def test_metadata_accepts_mixed_types(self) -> None:
        """TELE-815: metadata dict accepts str, int, and float values."""
        metadata = {
            "string_field": "text",
            "int_field": 42,
            "float_field": 3.14,
        }
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
            metadata=metadata,
        )
        assert entry.metadata["string_field"] == "text"
        assert entry.metadata["int_field"] == 42
        assert entry.metadata["float_field"] == pytest.approx(3.14)


# =============================================================================
# Stage Abbreviations (_STAGE_SHORT mapping)
# =============================================================================


class TestStageAbbreviations:
    """Stage abbreviation mapping for display in EventLog widget."""

    def test_stage_short_dormant(self) -> None:
        """Stage abbreviation: DORMANT -> DORM."""
        assert _STAGE_SHORT["DORMANT"] == "DORM"

    def test_stage_short_germinated(self) -> None:
        """Stage abbreviation: GERMINATED -> GERM."""
        assert _STAGE_SHORT["GERMINATED"] == "GERM"

    def test_stage_short_training(self) -> None:
        """Stage abbreviation: TRAINING -> TRAIN."""
        assert _STAGE_SHORT["TRAINING"] == "TRAIN"

    def test_stage_short_holding(self) -> None:
        """Stage abbreviation: HOLDING -> HOLD."""
        assert _STAGE_SHORT["HOLDING"] == "HOLD"

    def test_stage_short_blending(self) -> None:
        """Stage abbreviation: BLENDING -> BLEND."""
        assert _STAGE_SHORT["BLENDING"] == "BLEND"

    def test_stage_short_fossilized(self) -> None:
        """Stage abbreviation: FOSSILIZED -> FOSS."""
        assert _STAGE_SHORT["FOSSILIZED"] == "FOSS"

    def test_stage_short_pruned(self) -> None:
        """Stage abbreviation: PRUNED -> PRUNE."""
        assert _STAGE_SHORT["PRUNED"] == "PRUNE"

    def test_stage_short_embargoed(self) -> None:
        """Stage abbreviation: EMBARGOED -> EMBG."""
        assert _STAGE_SHORT["EMBARGOED"] == "EMBG"

    def test_stage_short_resetting(self) -> None:
        """Stage abbreviation: RESETTING -> RESET."""
        assert _STAGE_SHORT["RESETTING"] == "RESET"

    def test_stage_short_all_stages_present(self) -> None:
        """All lifecycle stages have abbreviations."""
        expected_stages = {
            "DORMANT",
            "GERMINATED",
            "TRAINING",
            "HOLDING",
            "BLENDING",
            "FOSSILIZED",
            "PRUNED",
            "EMBARGOED",
            "RESETTING",
        }
        assert set(_STAGE_SHORT.keys()) == expected_stages


# =============================================================================
# EventLogEntry Default Values
# =============================================================================


class TestEventLogEntryDefaults:
    """Test EventLogEntry default values match TELE specifications."""

    def test_episode_default_zero(self) -> None:
        """Episode field defaults to 0."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.episode == 0

    def test_relative_time_default_empty(self) -> None:
        """Relative time field defaults to empty string."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.relative_time == ""

    def test_metadata_default_empty_dict(self) -> None:
        """Metadata field defaults to empty dict."""
        entry = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        assert entry.metadata == {}

    def test_metadata_default_independent_instances(self) -> None:
        """Each EventLogEntry gets independent metadata dict."""
        entry1 = EventLogEntry(
            timestamp="14:32:05",
            event_type="SEED_GERMINATED",
            env_id=0,
            message="Germinated",
        )
        entry2 = EventLogEntry(
            timestamp="14:32:06",
            event_type="SEED_PRUNED",
            env_id=1,
            message="Pruned",
        )
        # Modifying one should not affect the other
        entry1.metadata["slot_id"] = "r0c0"
        assert "slot_id" not in entry2.metadata
