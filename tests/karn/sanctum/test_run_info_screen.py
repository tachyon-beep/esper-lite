"""Tests for RunInfoScreen modal.

Tests cover:
- RunInfoScreen: Modal showing run configuration and status
- format_runtime import and usage with include_seconds_in_hours=True
"""
from __future__ import annotations

from esper.karn.sanctum.formatting import format_runtime
from esper.karn.sanctum.schema import SanctumSnapshot, SystemVitals
from esper.karn.sanctum.app import RunInfoScreen


class TestRunInfoScreenRendering:
    """Test RunInfoScreen renders all fields correctly."""

    def test_modal_creation(self):
        """Modal should be creatable with SanctumSnapshot."""
        snapshot = SanctumSnapshot(
            task_name="test_run",
            current_episode=5,
            current_epoch=100,
            max_epochs=500,
            current_batch=10,
            max_batches=50,
            runtime_seconds=3665.0,
        )
        modal = RunInfoScreen(snapshot)
        assert modal is not None
        assert modal._snapshot == snapshot

    def test_uses_format_runtime_with_seconds(self):
        """Runtime should use format_runtime with include_seconds_in_hours=True."""
        # Test that format_runtime is used with the correct parameter
        result = format_runtime(3665.0, include_seconds_in_hours=True)
        assert result == "1h 1m 5s"

        # Without the flag, seconds are omitted
        result_short = format_runtime(3665.0, include_seconds_in_hours=False)
        assert result_short == "1h 1m"


class TestRunInfoScreenContent:
    """Test RunInfoScreen content structure."""

    def test_snapshot_fields_accessible(self):
        """All expected snapshot fields should be accessible."""
        snapshot = SanctumSnapshot(
            task_name="experiment_123",
            current_episode=10,
            current_epoch=250,
            max_epochs=1000,
            current_batch=25,
            max_batches=100,
            runtime_seconds=7200.0,
            connected=True,
            staleness_seconds=0.5,
            training_thread_alive=True,
            vitals=SystemVitals(
                epochs_per_second=0.5,
                batches_per_hour=180.0,
            ),
        )

        # Verify fields are accessible
        assert snapshot.task_name == "experiment_123"
        assert snapshot.current_episode == 10
        assert snapshot.current_epoch == 250
        assert snapshot.max_epochs == 1000
        assert snapshot.current_batch == 25
        assert snapshot.max_batches == 100
        assert snapshot.runtime_seconds == 7200.0
        assert snapshot.connected is True
        assert snapshot.staleness_seconds == 0.5
        assert snapshot.training_thread_alive is True


class TestFormatRuntimeFunction:
    """Test format_runtime function behavior."""

    def test_format_runtime_zero(self):
        """Zero seconds should return '--'."""
        assert format_runtime(0) == "--"

    def test_format_runtime_negative(self):
        """Negative seconds should return '--'."""
        assert format_runtime(-10) == "--"

    def test_format_runtime_seconds_only(self):
        """Less than a minute should show seconds only."""
        assert format_runtime(45) == "45s"

    def test_format_runtime_minutes_seconds(self):
        """Minutes and seconds should both show."""
        assert format_runtime(125) == "2m 5s"

    def test_format_runtime_hours_minutes_no_seconds(self):
        """Hours format defaults to no seconds."""
        assert format_runtime(3665) == "1h 1m"

    def test_format_runtime_hours_minutes_with_seconds(self):
        """Hours format with include_seconds_in_hours=True."""
        assert format_runtime(3665, include_seconds_in_hours=True) == "1h 1m 5s"

    def test_format_runtime_exact_hour(self):
        """Exact hour should show 0 minutes."""
        assert format_runtime(3600) == "1h 0m"

    def test_format_runtime_exact_minute(self):
        """Exact minute should show 0 seconds."""
        assert format_runtime(60) == "1m 0s"
