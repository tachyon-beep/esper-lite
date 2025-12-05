"""Reward Telemetry Dataclasses.

Captures per-component breakdown of reward computation
for diagnosing reward hacking and tuning reward weights.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(slots=True)
class RewardComponentsTelemetry:
    """Breakdown of reward components for debugging.

    Each field represents one component of the total reward.
    All components should sum to total_reward.
    """

    # Base signal
    base_acc_delta: float = 0.0

    # Penalties
    compute_rent: float = 0.0

    # Bonuses
    stage_bonus: float = 0.0
    pbrs_bonus: float = 0.0
    action_shaping: float = 0.0
    terminal_bonus: float = 0.0

    # Contribution-primary specific (if applicable)
    seed_contribution: float | None = None
    bounded_attribution: float | None = None

    # Total
    total_reward: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for TelemetryEvent data field."""
        return asdict(self)


__all__ = ["RewardComponentsTelemetry"]
