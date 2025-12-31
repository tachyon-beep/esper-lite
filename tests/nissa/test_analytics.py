"""Tests for Nissa Blueprint Analytics."""

import logging

from esper.nissa.analytics import BlueprintStats, SeedScoreboard
from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.leyline.telemetry import (
    SeedGerminatedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    EpochCompletedPayload,
    PerformanceDegradationPayload,
)
from esper.nissa.analytics import BlueprintAnalytics


class TestBlueprintStats:
    """Tests for BlueprintStats dataclass."""

    def test_initial_values(self):
        """Stats start at zero."""
        stats = BlueprintStats()
        assert stats.germinated == 0
        assert stats.fossilized == 0
        assert stats.pruned == 0
        assert stats.acc_deltas == []
        assert stats.churns == []

    def test_mean_acc_delta_empty(self):
        """Empty acc_deltas returns 0."""
        stats = BlueprintStats()
        assert stats.mean_acc_delta == 0.0

    def test_mean_acc_delta_with_values(self):
        """Mean accuracy delta calculated correctly."""
        stats = BlueprintStats(acc_deltas=[1.0, 2.0, 3.0])
        assert stats.mean_acc_delta == 2.0

    def test_fossilization_rate_no_terminal(self):
        """Rate is 0% when no seeds reached terminal state."""
        stats = BlueprintStats(germinated=5)
        assert stats.fossilization_rate == 0.0

    def test_fossilization_rate_all_fossilized(self):
        """Rate is 100% when all seeds fossilized."""
        stats = BlueprintStats(germinated=5, fossilized=5, pruned=0)
        assert stats.fossilization_rate == 100.0

    def test_fossilization_rate_mixed(self):
        """Rate calculated correctly for mixed outcomes."""
        stats = BlueprintStats(germinated=10, fossilized=3, pruned=7)
        assert stats.fossilization_rate == 30.0


class TestSeedScoreboard:
    """Tests for SeedScoreboard dataclass."""

    def test_initial_values(self):
        """Scoreboard starts empty."""
        sb = SeedScoreboard()
        assert sb.total_germinated == 0
        assert sb.total_fossilized == 0
        assert sb.params_added == 0
        assert sb.live_blueprint is None

    def test_compute_cost_empty(self):
        """Empty scoreboard has 1.0x cost."""
        sb = SeedScoreboard()
        assert sb.compute_cost == 1.0

    def test_compute_cost_with_fossilized(self):
        """Compute cost accumulates from fossilized seeds."""
        sb = SeedScoreboard()
        sb.fossilized_by_blueprint["depthwise"] = 2  # 2 * 0.08 = 0.16 extra
        sb.fossilized_by_blueprint["attention"] = 1  # 1 * 0.35 = 0.35 extra
        # Total: 1.0 + 0.16 + 0.35 = 1.51
        assert abs(sb.compute_cost - 1.51) < 0.01

    def test_params_percentage(self):
        """Params percentage calculated correctly."""
        sb = SeedScoreboard(params_added=10000, host_params=100000)
        assert sb.params_percentage == 10.0


class TestBlueprintAnalytics:
    """Tests for BlueprintAnalytics OutputBackend."""

    def test_tracks_germination(self):
        """Germination events increment counters."""
        analytics = BlueprintAnalytics()
        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_GERMINATED,
            seed_id="seed_001",
            data=SeedGerminatedPayload(
                slot_id="slot_0",
                env_id=0,
                blueprint_id="depthwise",
                params=5000,
                alpha=0.0,
            ),
        )
        analytics.emit(event)

        assert analytics.stats["depthwise"].germinated == 1
        assert analytics.scoreboards[0].total_germinated == 1
        assert analytics.scoreboards[0].live_blueprint == "depthwise"

    def test_tracks_fossilization(self):
        """Fossilization events update stats and scoreboard."""
        analytics = BlueprintAnalytics()
        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_FOSSILIZED,
            seed_id="seed_001",
            data=SeedFossilizedPayload(
                slot_id="slot_0",
                env_id=0,
                blueprint_id="depthwise",
                improvement=2.5,
                params_added=10000,
                epochs_total=0,
                counterfactual=0.0,
            ),
        )
        analytics.emit(event)

        assert analytics.stats["depthwise"].fossilized == 1
        assert analytics.stats["depthwise"].acc_deltas == [2.5]
        assert analytics.scoreboards[0].total_fossilized == 1
        assert analytics.scoreboards[0].params_added == 10000
        assert analytics.scoreboards[0].fossilized_by_blueprint["depthwise"] == 1

    def test_tracks_prune(self):
        """Prune events update stats with churn."""
        analytics = BlueprintAnalytics()
        event = TelemetryEvent(
            event_type=TelemetryEventType.SEED_PRUNED,
            seed_id="seed_001",
            data=SeedPrunedPayload(
                slot_id="slot_0",
                env_id=1,
                blueprint_id="attention",
                reason="no_improvement",
                improvement=-0.3,
                epochs_total=0,
                counterfactual=0.0,
            ),
        )
        analytics.emit(event)

        assert analytics.stats["attention"].pruned == 1
        assert analytics.stats["attention"].churns == [-0.3]
        assert analytics.scoreboards[1].total_pruned == 1

    def test_ignores_irrelevant_events(self):
        """Non-lifecycle events are ignored."""
        analytics = BlueprintAnalytics()
        event = TelemetryEvent(
            event_type=TelemetryEventType.EPOCH_COMPLETED,
            seed_id="seed_001",
            data=EpochCompletedPayload(
                env_id=0,
                inner_epoch=0,
                val_loss=1.0,
                val_accuracy=75.0,
                seeds=None,
            ),
        )
        analytics.emit(event)

        assert len(analytics.stats) == 0
        assert len(analytics.scoreboards) == 0

    def test_accepts_typed_performance_degradation(self, caplog):
        """PERFORMANCE_DEGRADATION uses typed payload and is handled without warnings."""
        analytics = BlueprintAnalytics(quiet=True)
        event = TelemetryEvent(
            event_type=TelemetryEventType.PERFORMANCE_DEGRADATION,
            severity="warning",
            data=PerformanceDegradationPayload(
                env_id=0,
                current_acc=0.6,
                rolling_avg_acc=0.8,
                drop_percent=25.0,
                threshold_percent=10.0,
                training_progress=0.5,
            ),
        )

        with caplog.at_level(logging.WARNING, logger="esper.nissa.analytics"):
            analytics.emit(event)

        assert len(analytics.stats) == 0
        assert len(analytics.scoreboards) == 0
        assert not any("not yet migrated" in r.getMessage() for r in caplog.records)

    def test_summary_table_format(self):
        """Summary table is formatted correctly."""
        analytics = BlueprintAnalytics()
        # Add some test data
        analytics.stats["depthwise"].germinated = 10
        analytics.stats["depthwise"].fossilized = 6
        analytics.stats["depthwise"].pruned = 4
        analytics.stats["depthwise"].acc_deltas = [2.0, 2.5, 3.0, 2.0, 2.5, 3.0]

        table = analytics.summary_table()

        assert "Blueprint Stats:" in table
        assert "depthwise" in table
        assert "60.0%" in table  # fossilization rate

    def test_snapshot_serializable(self):
        """Snapshot returns serializable dict."""
        analytics = BlueprintAnalytics()
        analytics.stats["depthwise"].germinated = 5
        analytics._get_scoreboard(0).total_germinated = 5

        snapshot = analytics.snapshot()

        assert "stats" in snapshot
        assert "scoreboards" in snapshot
        assert snapshot["stats"]["depthwise"]["germinated"] == 5
