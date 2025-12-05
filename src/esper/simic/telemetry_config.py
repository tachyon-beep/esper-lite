"""Telemetry Configuration for Simic Training.

Controls what telemetry is collected at different verbosity levels,
with automatic escalation to DEBUG mode on anomaly detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class TelemetryLevel(IntEnum):
    """Telemetry verbosity levels."""

    OFF = 0      # No telemetry collection
    MINIMAL = 1  # Episode summaries only
    NORMAL = 2   # Per-batch PPO metrics (Ops Normal)
    DEBUG = 3    # Full diagnostics (Oh Shit mode)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection.

    Supports automatic escalation to DEBUG mode when anomalies
    are detected, with configurable escalation duration.
    """

    level: TelemetryLevel = TelemetryLevel.NORMAL

    # Ops Normal collection intervals
    gradient_check_interval: int = 1   # Every N epochs
    memory_check_interval: int = 5     # Every N epochs

    # Debug settings
    per_layer_gradients: bool = False
    activation_monitoring: bool = False
    weight_tracking: bool = False
    weight_track_interval: int = 10

    # Auto-escalation
    auto_escalate_on_anomaly: bool = True
    anomaly_escalation_epochs: int = 5

    # Internal state
    _escalation_epochs_remaining: int = field(default=0, repr=False)

    @property
    def effective_level(self) -> TelemetryLevel:
        """Return effective level (accounting for temporary escalation)."""
        if self._escalation_epochs_remaining > 0:
            return TelemetryLevel.DEBUG
        return self.level

    @property
    def escalation_epochs_remaining(self) -> int:
        """Return remaining escalation epochs."""
        return self._escalation_epochs_remaining

    def should_collect(self, category: str) -> bool:
        """Check if telemetry category should be collected.

        Args:
            category: One of 'ops_normal' or 'debug'

        Returns:
            True if category should be collected at current level
        """
        level = self.effective_level
        if category == "ops_normal":
            return level >= TelemetryLevel.NORMAL
        elif category == "debug":
            return level >= TelemetryLevel.DEBUG
        return False

    def escalate_temporarily(self, epochs: int | None = None) -> None:
        """Temporarily escalate to DEBUG level.

        Args:
            epochs: Number of epochs to stay escalated (default: anomaly_escalation_epochs)
        """
        if epochs is None:
            epochs = self.anomaly_escalation_epochs
        self._escalation_epochs_remaining = epochs

    def tick_escalation(self) -> None:
        """Decrement escalation counter (call once per epoch)."""
        if self._escalation_epochs_remaining > 0:
            self._escalation_epochs_remaining -= 1


__all__ = ["TelemetryLevel", "TelemetryConfig"]
