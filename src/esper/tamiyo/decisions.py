"""Tamiyo Decisions - Strategic decision structures.

Defines the decisions made by strategic controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from esper.leyline import (
    Action,
    CommandType,
    RiskLevel,
    AdaptationCommand,
    SeedStage,
)


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


# Mapping from Action to leyline CommandType and target stage
_ACTION_TO_COMMAND: dict[Action, tuple[CommandType, SeedStage | None]] = {
    Action.WAIT: (CommandType.REQUEST_STATE, None),
    Action.GERMINATE_CONV: (CommandType.GERMINATE, SeedStage.GERMINATED),
    Action.GERMINATE_ATTENTION: (CommandType.GERMINATE, SeedStage.GERMINATED),
    Action.GERMINATE_NORM: (CommandType.GERMINATE, SeedStage.GERMINATED),
    Action.GERMINATE_DEPTHWISE: (CommandType.GERMINATE, SeedStage.GERMINATED),
    Action.ADVANCE: (CommandType.ADVANCE_STAGE, None),  # Target determined by current stage
    Action.CULL: (CommandType.CULL, SeedStage.CULLED),
}


@dataclass
class TamiyoDecision:
    """A decision made by Tamiyo.

    Uses the shared Action enum from leyline for compatibility with
    all controllers (heuristic, RL, etc.).
    """
    action: Action
    target_seed_id: str | None = None
    reason: str = ""
    confidence: float = 1.0

    def __str__(self) -> str:
        parts = [f"Action: {self.action.name}"]
        if self.target_seed_id:
            parts.append(f"Target: {self.target_seed_id}")
        if self.reason:
            parts.append(f"Reason: {self.reason}")
        return " | ".join(parts)

    @property
    def blueprint_id(self) -> str | None:
        """Get blueprint ID if this is a germinate action."""
        return Action.get_blueprint_id(self.action)

    def to_command(self) -> AdaptationCommand:
        """Convert to Leyline's canonical AdaptationCommand format."""
        command_type, target_stage = _ACTION_TO_COMMAND.get(
            self.action,
            (CommandType.REQUEST_STATE, None)
        )

        # Determine risk level based on action
        if self.action == Action.WAIT:
            risk = RiskLevel.GREEN
        elif Action.is_germinate(self.action):
            risk = RiskLevel.YELLOW
        elif self.action == Action.ADVANCE:
            risk = RiskLevel.YELLOW
        elif self.action == Action.CULL:
            risk = RiskLevel.ORANGE
        else:
            risk = RiskLevel.GREEN

        return AdaptationCommand(
            command_type=command_type,
            target_seed_id=self.target_seed_id,
            blueprint_id=self.blueprint_id,
            target_stage=target_stage,
            reason=self.reason,
            confidence=self.confidence,
            risk_level=risk,
        )


__all__ = [
    "TamiyoDecision",
]
