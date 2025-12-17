"""Tests for Karn health module."""

import pytest
from esper.karn.health import VitalSignsMonitor, VitalSigns
from esper.karn.store import TelemetryStore, EpochSnapshot, SlotSnapshot, HostSnapshot, EpisodeContext
from esper.leyline import SeedStage


class TestVitalSignsEnumComparisons:
    """Test that vital signs correctly identify seed stages."""

    def test_counts_germinated_as_total_but_not_active(self) -> None:
        """GERMINATED seeds should count toward total_seeds but not active_seeds."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        # Create epoch with GERMINATED seed
        store.start_episode(EpisodeContext(episode_id="test_001"))
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.GERMINATED,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        # GERMINATED should count as started (total_seeds)
        # but active_seeds counts only TRAINING/BLENDING/PROBATIONARY/FOSSILIZED
        assert vitals.active_seeds == 0, "GERMINATED should not count as active"

    def test_counts_training_as_active(self) -> None:
        """TRAINING seeds should count as active."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode(EpisodeContext(episode_id="test"))
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.TRAINING,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        assert vitals.active_seeds == 1, "TRAINING should count as active"

    def test_counts_blending_as_active(self) -> None:
        """BLENDING seeds should count as active."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode(EpisodeContext(episode_id="test"))
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.BLENDING,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        assert vitals.active_seeds == 1, "BLENDING should count as active"

    def test_counts_probationary_as_active(self) -> None:
        """PROBATIONARY seeds should count as active."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode(EpisodeContext(episode_id="test"))
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.PROBATIONARY,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        assert vitals.active_seeds == 1, "PROBATIONARY should count as active"

    def test_counts_fossilized_as_active(self) -> None:
        """FOSSILIZED seeds should count as active (terminal success)."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode(EpisodeContext(episode_id="test"))
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.FOSSILIZED,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        assert vitals.active_seeds == 1, "FOSSILIZED should count as active"

    def test_counts_culled_correctly(self) -> None:
        """CULLED seeds should be counted in failure rate."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode(EpisodeContext(episode_id="test"))
        snapshot = store.start_epoch(1)
        # One germinated (counts as total), one culled
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.GERMINATED,
        )
        snapshot.slots["r0c1"] = SlotSnapshot(
            slot_id="r0c1",
            stage=SeedStage.CULLED,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        # Failure rate should be 1 culled / 2 total = 0.5
        assert vitals.seed_failure_rate == 0.5, f"Expected 0.5, got {vitals.seed_failure_rate}"

    def test_does_not_count_dormant(self) -> None:
        """DORMANT seeds should not count toward total or active."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode(EpisodeContext(episode_id="test"))
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.DORMANT,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        assert vitals.active_seeds == 0, "DORMANT should not count as active"

    def test_does_not_count_unknown(self) -> None:
        """UNKNOWN seeds should not count toward total or active."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode(EpisodeContext(episode_id="test"))
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.UNKNOWN,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        assert vitals.active_seeds == 0, "UNKNOWN should not count as active"
