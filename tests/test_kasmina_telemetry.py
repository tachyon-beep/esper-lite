"""Tests for enriched Kasmina telemetry events."""

import pytest
import torch

from esper.leyline import TelemetryEvent, TelemetryEventType
from esper.kasmina.slot import SeedSlot


class TestEnrichedTelemetry:
    """Tests for enriched telemetry event data."""

    def setup_method(self):
        """Set up test fixtures."""
        self.events: list[TelemetryEvent] = []

        def capture_event(event: TelemetryEvent):
            self.events.append(event)

        self.slot = SeedSlot(
            slot_id="test_slot",
            channels=64,
            device="cpu",
            on_telemetry=capture_event,
            fast_mode=False,
        )

    def test_germinate_emits_blueprint_id(self):
        """Germination event includes blueprint_id."""
        self.slot.germinate("depthwise", "seed_001")

        assert len(self.events) == 1
        event = self.events[0]
        assert event.event_type == TelemetryEventType.SEED_GERMINATED
        assert event.data["blueprint_id"] == "depthwise"
        assert event.data["seed_id"] == "seed_001"
        assert "params" in event.data
        assert event.data["params"] > 0

    def test_fossilize_emits_improvement(self):
        """Fossilization event includes improvement and params."""
        self.slot.germinate("depthwise", "seed_001")
        self.events.clear()

        # Simulate training improvement
        self.slot.state.metrics.initial_val_accuracy = 70.0
        self.slot.state.metrics.current_val_accuracy = 75.0
        self.slot.state.metrics.counterfactual_contribution = 5.0  # Required for G5

        # Advance through stages to FOSSILIZED (must go through PROBATIONARY)
        self.slot.state.stage = SeedStage.PROBATIONARY
        self.slot.state.is_healthy = True  # G5 also requires health
        self.slot.advance_stage(SeedStage.FOSSILIZED)

        # Find fossilization event
        foss_events = [e for e in self.events
                       if e.event_type == TelemetryEventType.SEED_FOSSILIZED]
        assert len(foss_events) == 1

        event = foss_events[0]
        assert event.data["blueprint_id"] == "depthwise"
        assert event.data["improvement"] == 5.0  # 75 - 70
        assert "params_added" in event.data

    def test_cull_emits_improvement(self):
        """Cull event includes improvement (churn metric)."""
        self.slot.germinate("attention", "seed_002")
        self.events.clear()

        # Simulate negative improvement
        self.slot.state.metrics.initial_val_accuracy = 70.0
        self.slot.state.metrics.current_val_accuracy = 69.5

        self.slot.cull("no_improvement")

        cull_events = [e for e in self.events
                       if e.event_type == TelemetryEventType.SEED_CULLED]
        assert len(cull_events) == 1

        event = cull_events[0]
        assert event.data["blueprint_id"] == "attention"
        assert event.data["improvement"] == -0.5
        assert event.data["reason"] == "no_improvement"


# Need this import for the test
from esper.leyline import SeedStage
