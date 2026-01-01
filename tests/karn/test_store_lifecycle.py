"""Tests for Karn TelemetryStore lifecycle.

These tests focus on TelemetryStore's in-memory lifecycle (episode/epoch tracking).
"""

import pytest

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
