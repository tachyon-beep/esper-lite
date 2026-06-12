"""Tests for Karn health module."""

from datetime import datetime, timedelta, timezone

import pytest

from esper.karn.health import GradientHealth, HealthMonitor, MemoryStats, SystemHealth, VitalSignsMonitor
from esper.karn.store import TelemetryStore, SlotSnapshot, EpisodeContext, EpochSnapshot
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
        # but active_seeds counts only TRAINING/BLENDING/HOLDING/FOSSILIZED
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

    def test_counts_holding_as_active(self) -> None:
        """HOLDING seeds should count as active."""
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store)

        store.start_episode(EpisodeContext(episode_id="test"))
        snapshot = store.start_epoch(1)
        snapshot.slots["r0c0"] = SlotSnapshot(
            slot_id="r0c0",
            stage=SeedStage.HOLDING,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        assert vitals.active_seeds == 1, "HOLDING should count as active"

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

    def test_counts_pruned_correctly(self) -> None:
        """PRUNED seeds should be counted in failure rate."""
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
            stage=SeedStage.PRUNED,
        )
        snapshot.host.val_accuracy = 0.5
        store.commit_epoch()

        vitals = monitor.check_vitals()
        # Failure rate should be 1 pruned / 2 total = 0.5
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


class TestHealthMonitorCallback:
    """Test HealthMonitor with emit callback injection."""

    def test_emits_memory_warning_via_callback(self) -> None:
        """Memory warning should be emitted via injected callback."""
        from esper.karn.health import HealthMonitor

        emitted_events: list = []

        def capture_emit(event):
            emitted_events.append(event)

        monitor = HealthMonitor(
            emit_callback=capture_emit,
            memory_warning_threshold=0.5,  # Low threshold to trigger easily
        )

        # Trigger memory warning check with high utilization
        warned = monitor._check_memory_and_warn(
            gpu_utilization=0.95,
            gpu_allocated_gb=10.0,
            gpu_total_gb=12.0,
        )

        assert warned is True
        assert len(emitted_events) == 1
        assert emitted_events[0].event_type.name == "MEMORY_WARNING"

    def test_no_emit_without_callback(self) -> None:
        """Without callback, no emission should occur (no crash)."""
        from esper.karn.health import HealthMonitor

        monitor = HealthMonitor(
            emit_callback=None,
            memory_warning_threshold=0.5,
        )

        # Should not crash even without callback
        warned = monitor._check_memory_and_warn(
            gpu_utilization=0.95,
            gpu_allocated_gb=10.0,
            gpu_total_gb=12.0,
        )

        # Still returns True (warning condition met) but no emission
        assert warned is True


class TestHealthMonitorStats:
    """HealthMonitor deterministic state calculations."""

    def test_memory_stats_formats_gpu_utilization(self) -> None:
        stats = MemoryStats(
            gpu_allocated=2 * 1024**3,
            gpu_total=8 * 1024**3,
        )

        assert stats.gpu_utilization == 0.25
        assert stats.gpu_allocated_gb == 2.0
        assert stats.gpu_total_gb == 8.0
        assert str(stats) == "GPU: 2.00/8.00GB (25.0%)"

    def test_gradient_health_property_tracks_bad_flags_and_error_norm(self) -> None:
        assert GradientHealth(mean_norm=10.0).is_healthy is True
        assert GradientHealth(has_nan=True).is_healthy is False
        assert GradientHealth(has_inf=True).is_healthy is False
        assert GradientHealth(mean_norm=10_001.0).is_healthy is False

    def test_system_health_records_warnings_and_errors(self) -> None:
        health = SystemHealth()

        health.add_warning("gpu hot")
        health.add_error("gradient exploded")

        assert health.warnings == ["gpu hot"]
        assert health.errors == ["gradient exploded"]
        assert health.is_healthy is False

    def test_memory_warning_respects_threshold_and_cooldown(self) -> None:
        events: list = []
        monitor = HealthMonitor(
            emit_callback=events.append,
            memory_warning_threshold=0.5,
            memory_warning_cooldown=60.0,
        )

        first = monitor._check_memory_and_warn(0.90, 9.0, 10.0)
        second = monitor._check_memory_and_warn(0.95, 9.5, 10.0)
        below_threshold = monitor._check_memory_and_warn(0.10, 1.0, 10.0)

        assert first is True
        assert second is False
        assert below_threshold is False
        assert len(events) == 1

    def test_epoch_timing_stats_and_reset(self) -> None:
        monitor = HealthMonitor()
        monitor.record_epoch_time(1.0)
        monitor.record_epoch_time(3.0)

        stats = monitor.get_epoch_stats()
        assert stats["mean"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["std"] == pytest.approx(1.41421356237)

        monitor.reset()

        assert monitor.get_epoch_stats() == {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

    def test_check_health_reports_latest_epoch_gradients_and_throughput(self) -> None:
        store = TelemetryStore()
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for epoch, grad_norm in ((1, 2.0), (2, 4.0), (3, 6.0)):
            snapshot = EpochSnapshot(epoch=epoch, timestamp=base + timedelta(seconds=epoch * 10))
            snapshot.host.host_grad_norm = grad_norm
            store.epoch_snapshots.append(snapshot)
        monitor = HealthMonitor(
            store=store,
            grad_norm_warning=5.0,
            grad_norm_error=10.0,
        )

        health = monitor.check_health()

        assert health.last_epoch == 3
        assert health.gradients.mean_norm == 4.0
        assert health.gradients.max_norm == 6.0
        assert health.warnings == ["High gradient norm: 6.00"]
        assert health.throughput.epochs_per_minute == 6.0
        assert health.check_time_ms >= 0.0

    def test_gradient_explosion_marks_health_unhealthy(self) -> None:
        store = TelemetryStore()
        snapshot = EpochSnapshot(epoch=1)
        snapshot.host.host_grad_norm = 20_000_000_000.0
        store.epoch_snapshots.append(snapshot)
        monitor = HealthMonitor(store=store)

        health = SystemHealth()
        gradients = monitor._check_gradients(health)

        assert gradients.has_inf is True
        assert health.is_healthy is False
        assert health.errors == ["Gradient explosion detected (possible Inf)"]


class TestVitalSignsTrends:
    """Vital signs trend and critical-state coverage."""

    def test_loss_trend_branches(self) -> None:
        monitor = VitalSignsMonitor()

        assert monitor._analyze_loss_trend() == (True, "stable")

        monitor._loss_history = [10.0, 10.0, 10.0, 10.0, 10.0, 8.0, 8.0, 8.0, 8.0, 8.0]
        assert monitor._analyze_loss_trend() == (True, "improving")

        monitor._loss_history = [8.0, 8.0, 8.0, 8.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        assert monitor._analyze_loss_trend() == (False, "degrading")

    def test_stagnation_becomes_critical_and_reset_clears_tracking(self) -> None:
        store = TelemetryStore()
        monitor = VitalSignsMonitor(store=store, stagnation_epochs=1)
        for epoch in range(1, 4):
            snapshot = EpochSnapshot(epoch=epoch)
            snapshot.host.val_accuracy = 0.5
            snapshot.host.val_loss = 1.0
            store.epoch_snapshots.append(snapshot)
            vitals = monitor.check_vitals()

        assert vitals.critical is True
        assert vitals.reason == "No improvement for 2 epochs"

        monitor.reset()

        assert monitor._best_accuracy == 0.0
        assert monitor._epochs_since_improvement == 0
        assert monitor._loss_history == []

    def test_high_seed_failure_rate_becomes_critical(self) -> None:
        store = TelemetryStore()
        snapshot = EpochSnapshot(epoch=1)
        snapshot.host.val_accuracy = 1.0
        snapshot.host.val_loss = 1.0
        snapshot.slots["r0c0"] = SlotSnapshot(slot_id="r0c0", stage=SeedStage.PRUNED)
        snapshot.slots["r0c1"] = SlotSnapshot(slot_id="r0c1", stage=SeedStage.PRUNED)
        snapshot.slots["r0c2"] = SlotSnapshot(slot_id="r0c2", stage=SeedStage.PRUNED)
        snapshot.slots["r0c3"] = SlotSnapshot(slot_id="r0c3", stage=SeedStage.GERMINATED)
        store.epoch_snapshots.append(snapshot)

        vitals = VitalSignsMonitor(store=store).check_vitals()

        assert vitals.critical is False
        assert vitals.seed_failure_rate == 0.75

        snapshot.slots["r0c4"] = SlotSnapshot(slot_id="r0c4", stage=SeedStage.PRUNED)
        snapshot.slots["r0c5"] = SlotSnapshot(slot_id="r0c5", stage=SeedStage.PRUNED)
        vitals = VitalSignsMonitor(store=store).check_vitals()

        assert vitals.critical is True
        assert vitals.reason == "High seed failure rate (83%)"
