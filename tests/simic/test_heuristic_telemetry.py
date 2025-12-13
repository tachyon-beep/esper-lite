"""Tests for heuristic training telemetry emission."""

import pytest

from esper.leyline import TelemetryEvent, TelemetryEventType


class TestHeuristicTelemetry:
    """Tests for telemetry emission in heuristic training."""

    def test_training_started_event_contract(self):
        """Document expected TRAINING_STARTED event structure."""
        # This test documents the contract that heuristic training should emit
        # TRAINING_STARTED to activate Karn.

        event = TelemetryEvent(
            event_type=TelemetryEventType.TRAINING_STARTED,
            data={
                "episode_id": "heur_42",
                "seed": 42,
                "max_epochs": 75,
                "task": "cifar10",
            }
        )

        assert event.event_type == TelemetryEventType.TRAINING_STARTED
        assert "episode_id" in event.data
        assert "max_epochs" in event.data
