"""Tamiyo Decisions - Strategic decision structures.

Defines the decisions made by strategic controllers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

from esper.leyline import CommandType, RiskLevel, AdaptationCommand, SeedStage

logger = logging.getLogger(__name__)


def _is_germinate_action(action: IntEnum) -> bool:
    """Check if action is any germinate variant (by name convention)."""
    return action.name.startswith("GERMINATE_")


def _get_blueprint_from_action(action: IntEnum) -> str | None:
    """Get blueprint name from a germinate action."""
    name = action.name
    if name.startswith("GERMINATE_"):
        return name[len("GERMINATE_"):].lower()
    return None


@dataclass
class TamiyoDecision:
    """A decision made by Tamiyo.

    Uses the topology-specific action enum from leyline.actions.build_action_enum()
    for heuristic baseline comparisons.
    """
    action: IntEnum
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
        return _get_blueprint_from_action(self.action)

    def to_command(self) -> AdaptationCommand:
        """Convert to Leyline's canonical AdaptationCommand format."""
        action_name = self.action.name
        is_germinate = _is_germinate_action(self.action)

        if action_name == "WAIT":
            command_type, target_stage = CommandType.REQUEST_STATE, None
        elif is_germinate:
            command_type, target_stage = CommandType.GERMINATE, SeedStage.GERMINATED
        elif action_name == "FOSSILIZE":
            command_type, target_stage = CommandType.ADVANCE_STAGE, SeedStage.FOSSILIZED
        elif action_name == "CULL":
            command_type, target_stage = CommandType.CULL, SeedStage.CULLED
        else:
            # Unknown action - log warning but fall back to safe behavior
            logger.warning(
                f"Unknown action '{action_name}' in to_command() - "
                f"falling back to REQUEST_STATE. This may indicate a missing mapping."
            )
            command_type, target_stage = CommandType.REQUEST_STATE, None

        # Determine risk level based on action
        if action_name == "WAIT":
            risk = RiskLevel.GREEN
        elif is_germinate:
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
