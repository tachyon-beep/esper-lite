"""Tests for telemetry event types."""

from esper.leyline import TelemetryEventType


class TestTelemetryEventTypes:
    """Tests for new telemetry event types."""

    def test_ppo_update_event_exists(self):
        """PPO_UPDATE_COMPLETED event type exists."""
        assert TelemetryEventType.PPO_UPDATE_COMPLETED

    def test_debug_event_types_exist(self):
        """Debug-level event types exist."""
        assert TelemetryEventType.RATIO_EXPLOSION_DETECTED
        assert TelemetryEventType.VALUE_COLLAPSE_DETECTED
        assert TelemetryEventType.GRADIENT_PATHOLOGY_DETECTED
        assert TelemetryEventType.NUMERICAL_INSTABILITY_DETECTED

    def test_ops_normal_event_types_exist(self):
        """Ops normal event types exist."""
        assert TelemetryEventType.MEMORY_WARNING
        assert TelemetryEventType.REWARD_HACKING_SUSPECTED
