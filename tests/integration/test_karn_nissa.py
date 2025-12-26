"""Integration tests for Karn-Nissa interaction.

Tests the core integration where:
- Karn TelemetryStore receives data from training
- Store tracks epoch history
- Export/import functionality works
"""

import pytest
import tempfile
from pathlib import Path

from esper.karn.store import (
    TelemetryStore,
    EpisodeContext,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def store():
    """Fresh TelemetryStore."""
    return TelemetryStore()


@pytest.fixture
def episode_context():
    """Episode context for store initialization."""
    return EpisodeContext(
        episode_id="test_episode_001",
        host_architecture="cnn_3block",
        host_params=1000000,
        max_epochs=25,
        task_type="classification",
        reward_mode="shaped",
    )


# =============================================================================
# TelemetryStore Lifecycle Tests
# =============================================================================


class TestTelemetryStoreLifecycle:
    """Tests for Karn TelemetryStore lifecycle."""

    def test_store_starts_episode(self, store, episode_context):
        """Store should accept episode context."""
        store.start_episode(episode_context)
        assert store.context == episode_context

    def test_store_tracks_epochs(self, store, episode_context):
        """Store should track epoch snapshots."""
        store.start_episode(episode_context)

        # Start and commit multiple epochs
        for epoch in range(3):
            store.start_epoch(epoch=epoch)
            store.commit_epoch()

        # Query recent epochs
        recent = store.get_recent_epochs(n=10)
        assert len(recent) == 3

    def test_store_latest_epoch(self, store, episode_context):
        """Store should provide latest epoch snapshot."""
        store.start_episode(episode_context)

        store.start_epoch(epoch=0)
        store.commit_epoch()

        store.start_epoch(epoch=1)
        store.commit_epoch()

        latest = store.latest_epoch
        assert latest is not None
        assert latest.epoch == 1


# =============================================================================
# Telemetry Export Tests
# =============================================================================


class TestTelemetryExport:
    """Tests for Karn telemetry export/import."""

    def test_export_to_jsonl(self, store, episode_context):
        """Store should export to JSONL format."""
        store.start_episode(episode_context)

        for epoch in range(3):
            store.start_epoch(epoch=epoch)
            store.commit_epoch()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "telemetry.jsonl"
            count = store.export_jsonl(path=str(path))

            assert count > 0
            assert path.exists()

    def test_export_creates_valid_file(self, store, episode_context):
        """Exported JSONL file should be valid and non-empty."""
        store.start_episode(episode_context)
        for epoch in range(3):
            store.start_epoch(epoch=epoch)
            store.commit_epoch()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "telemetry.jsonl"
            count = store.export_jsonl(path=str(path))

            # File should exist and have content
            assert path.exists()
            assert count > 0

            # File should contain valid JSON lines
            with open(path, "r") as f:
                lines = f.readlines()
            assert len(lines) > 0
