"""Integration test for blueprint analytics in training loop."""

import pytest
import torch

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.nissa import NissaHub, BlueprintAnalytics
from esper.tolaria import create_model


class TestBlueprintAnalyticsIntegration:
    """Integration tests for analytics with real models."""

    def test_full_lifecycle_tracking(self):
        """Track a complete germinate -> fossilize cycle."""
        analytics = BlueprintAnalytics()
        hub = NissaHub()
        hub.add_backend(analytics)

        # Create model with telemetry callback (create_model returns model with slots=["mid"])
        model = create_model(device="cpu")

        def callback(event: TelemetryEvent):
            event.data["env_id"] = 0
            hub.emit(event)

        model.seed_slots["mid"].on_telemetry = callback
        model.seed_slots["mid"].fast_mode = False

        # Set host params baseline
        analytics._get_scoreboard(0).host_params = sum(
            p.numel() for p in model.host.parameters() if p.requires_grad
        )

        # Germinate
        model.germinate_seed("depthwise", "test_seed", slot="mid")

        assert analytics.stats["depthwise"].germinated == 1
        assert analytics.scoreboards[0].live_blueprint == "depthwise"

        # Simulate improvement and fossilize
        model.seed_slots["mid"].state.metrics.initial_val_accuracy = 70.0
        model.seed_slots["mid"].state.metrics.current_val_accuracy = 75.0
        model.seed_slots["mid"].state.metrics.counterfactual_contribution = 5.0  # Required for G5

        # Force to PROBATIONARY stage for fossilization (per instructions)
        from esper.leyline import SeedStage
        model.seed_slots["mid"].state.stage = SeedStage.PROBATIONARY
        model.seed_slots["mid"].state.is_healthy = True  # G5 also requires health
        model.seed_slots["mid"].advance_stage(SeedStage.FOSSILIZED)

        assert analytics.stats["depthwise"].fossilized == 1
        assert analytics.stats["depthwise"].acc_deltas == [5.0]
        assert analytics.scoreboards[0].total_fossilized == 1
        assert analytics.scoreboards[0].params_added > 0

    def test_summary_tables_format(self):
        """Summary tables are readable and complete."""
        analytics = BlueprintAnalytics()

        # Add test data
        analytics.stats["depthwise"].germinated = 20
        analytics.stats["depthwise"].fossilized = 12
        analytics.stats["depthwise"].culled = 8
        analytics.stats["depthwise"].acc_deltas = [2.0] * 12

        analytics.stats["attention"].germinated = 10
        analytics.stats["attention"].fossilized = 1
        analytics.stats["attention"].culled = 9
        analytics.stats["attention"].churns = [-0.5] * 9

        analytics.scoreboards[0] = SeedScoreboard(
            total_fossilized=13,
            params_added=150000,
            host_params=1000000,
        )
        analytics.scoreboards[0].fossilized_by_blueprint["depthwise"] = 12
        analytics.scoreboards[0].fossilized_by_blueprint["attention"] = 1

        # Check summary table
        summary = analytics.summary_table()
        assert "depthwise" in summary
        assert "attention" in summary
        assert "60.0%" in summary  # depthwise rate

        # Check scoreboard
        scoreboard = analytics.scoreboard_table(0)
        assert "13" in scoreboard  # total fossilized
        assert "150.0K" in scoreboard  # params
        assert "15.0%" in scoreboard  # of host


from esper.nissa.analytics import SeedScoreboard
