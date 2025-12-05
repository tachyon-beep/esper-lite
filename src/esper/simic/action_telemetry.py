"""Action Distribution Telemetry.

Tracks action selection patterns for detecting policy issues
like over-conservative behavior or action spam.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ActionTelemetry:
    """Tracks action distribution and success rates.

    Collects per-batch statistics about which actions are taken
    and whether they succeed, to detect policy pathologies.
    """

    _action_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _success_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_action(self, action_name: str, success: bool = True) -> None:
        """Record an action taken.

        Args:
            action_name: Name of action (e.g., "WAIT", "GERMINATE_CONV")
            success: Whether the action succeeded
        """
        self._action_counts[action_name] += 1
        if success:
            self._success_counts[action_name] += 1

    def get_stats(self) -> dict:
        """Get action distribution statistics.

        Returns:
            Dict with action_counts, successful_action_counts, action_success_rate
        """
        action_counts = dict(self._action_counts)
        success_counts = dict(self._success_counts)

        success_rates = {}
        for action, count in action_counts.items():
            if count > 0:
                success_rates[action] = success_counts.get(action, 0) / count
            else:
                success_rates[action] = 0.0

        return {
            "action_counts": action_counts,
            "successful_action_counts": success_counts,
            "action_success_rate": success_rates,
        }

    def reset(self) -> None:
        """Reset counters for new batch/episode."""
        self._action_counts.clear()
        self._success_counts.clear()

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


__all__ = ["ActionTelemetry"]
