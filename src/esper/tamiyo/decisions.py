"""Tamiyo Decisions - Strategic decision structures.

Defines the decisions made by strategic controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from esper.leyline import CommandType, RiskLevel, AdaptationCommand, SeedStage
from esper.leyline.actions import is_germinate_action, get_blueprint_from_action


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


@dataclass
class TamiyoDecision:
    """A decision made by Tamiyo.

    Uses the topology-specific action enum from leyline.actions.
    """
    action: object
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
        return get_blueprint_from_action(self.action)

    def to_command(self) -> AdaptationCommand:
        """Convert to Leyline's canonical AdaptationCommand format."""
        action_name = self.action.name
        if action_name == "WAIT":
            command_type, target_stage = CommandType.REQUEST_STATE, None
        elif is_germinate_action(self.action):
            command_type, target_stage = CommandType.GERMINATE, SeedStage.GERMINATED
        elif action_name == "FOSSILIZE":
            command_type, target_stage = CommandType.ADVANCE_STAGE, SeedStage.FOSSILIZED
        elif action_name == "CULL":
            command_type, target_stage = CommandType.CULL, SeedStage.CULLED
        else:
            command_type, target_stage = CommandType.REQUEST_STATE, None

        # Determine risk level based on action
        if action_name == "WAIT":
            risk = RiskLevel.GREEN
        elif is_germinate_action(self.action):
            risk = RiskLevel.YELLOW
        elif action_name == "FOSSILIZE":
            risk = RiskLevel.YELLOW
        elif action_name == "CULL":
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
