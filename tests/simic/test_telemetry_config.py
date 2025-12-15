"""Tests for telemetry configuration."""


from esper.simic.telemetry_config import TelemetryLevel, TelemetryConfig


class TestTelemetryLevel:
    """Tests for TelemetryLevel enum."""

    def test_levels_are_ordered(self):
        """Telemetry levels have correct ordering."""
        assert TelemetryLevel.OFF < TelemetryLevel.MINIMAL
        assert TelemetryLevel.MINIMAL < TelemetryLevel.NORMAL
        assert TelemetryLevel.NORMAL < TelemetryLevel.DEBUG

    def test_level_comparison(self):
        """Can compare levels for conditional logging."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        assert config.level >= TelemetryLevel.NORMAL
        assert config.level < TelemetryLevel.DEBUG


class TestTelemetryConfig:
    """Tests for TelemetryConfig."""

    def test_default_config(self):
        """Default config has sensible values."""
        config = TelemetryConfig()
        assert config.level == TelemetryLevel.NORMAL
        assert config.auto_escalate_on_anomaly is True

    def test_should_collect_ops_normal(self):
        """should_collect returns True for ops normal at NORMAL level."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        assert config.should_collect("ops_normal") is True
        assert config.should_collect("debug") is False

    def test_should_collect_debug(self):
        """should_collect returns True for debug at DEBUG level."""
        config = TelemetryConfig(level=TelemetryLevel.DEBUG)
        assert config.should_collect("ops_normal") is True
        assert config.should_collect("debug") is True

    def test_escalate_temporarily(self):
        """escalate_temporarily increases level for N epochs."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        config.escalate_temporarily(epochs=5)
        assert config.effective_level == TelemetryLevel.DEBUG
        assert config.escalation_epochs_remaining == 5

    def test_tick_escalation(self):
        """tick_escalation decrements and returns to normal."""
        config = TelemetryConfig(level=TelemetryLevel.NORMAL)
        config.escalate_temporarily(epochs=2)
        config.tick_escalation()
        assert config.escalation_epochs_remaining == 1
        config.tick_escalation()
        assert config.escalation_epochs_remaining == 0
        assert config.effective_level == TelemetryLevel.NORMAL

    def test_should_collect_when_off(self):
        """should_collect returns False for all categories when OFF."""
        config = TelemetryConfig(level=TelemetryLevel.OFF)
        assert config.should_collect("ops_normal") is False
        assert config.should_collect("debug") is False

    def test_should_collect_when_minimal(self):
        """MINIMAL level collects neither ops_normal nor debug."""
        config = TelemetryConfig(level=TelemetryLevel.MINIMAL)
        assert config.should_collect("ops_normal") is False
        assert config.should_collect("debug") is False
