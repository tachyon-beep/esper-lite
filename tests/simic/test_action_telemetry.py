"""Tests for action distribution telemetry."""

import pytest

from esper.simic.action_telemetry import ActionTelemetry


class TestActionTelemetry:
    """Tests for ActionTelemetry tracking."""

    def test_record_action(self):
        """Can record actions and compute stats."""
        telemetry = ActionTelemetry()
        telemetry.record_action("WAIT", success=True)
        telemetry.record_action("WAIT", success=True)
        telemetry.record_action("GERMINATE_CONV", success=True)
        telemetry.record_action("CULL", success=False)

        stats = telemetry.get_stats()
        assert stats["action_counts"]["WAIT"] == 2
        assert stats["action_counts"]["GERMINATE_CONV"] == 1
        assert stats["successful_action_counts"]["WAIT"] == 2
        assert stats["action_success_rate"]["CULL"] == 0.0

    def test_reset(self):
        """Can reset telemetry for new batch."""
        telemetry = ActionTelemetry()
        telemetry.record_action("WAIT", success=True)
        telemetry.reset()

        stats = telemetry.get_stats()
        assert stats["action_counts"] == {}
