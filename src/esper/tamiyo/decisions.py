"""Tamiyo Decisions - Strategic decision structures.

Defines the decisions made by strategic controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any

from esper.leyline import (
    CommandType,
    RiskLevel,
    AdaptationCommand,
    SeedStage,
)


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class TamiyoAction(Enum):
    """Actions that Tamiyo can take."""
    WAIT = auto()              # Do nothing this epoch
    GERMINATE = auto()         # Start a new seed
    ADVANCE_TRAINING = auto()  # Move seed from GERMINATED to TRAINING
    ADVANCE_BLENDING = auto()  # Move seed from TRAINING to BLENDING
    ADVANCE_FOSSILIZE = auto() # Move seed from BLENDING to FOSSILIZED
    CULL = auto()              # Kill an underperforming seed
    CHANGE_BLUEPRINT = auto()  # Cull and try a different blueprint


# Mapping from TamiyoAction to leyline CommandType and target stage
_ACTION_TO_COMMAND: dict[TamiyoAction, tuple[CommandType, SeedStage | None]] = {
    TamiyoAction.WAIT: (CommandType.REQUEST_STATE, None),  # No-op, just query
    TamiyoAction.GERMINATE: (CommandType.GERMINATE, SeedStage.GERMINATED),
    TamiyoAction.ADVANCE_TRAINING: (CommandType.ADVANCE_STAGE, SeedStage.TRAINING),
    TamiyoAction.ADVANCE_BLENDING: (CommandType.ADVANCE_STAGE, SeedStage.BLENDING),
    TamiyoAction.ADVANCE_FOSSILIZE: (CommandType.ADVANCE_STAGE, SeedStage.FOSSILIZED),
    TamiyoAction.CULL: (CommandType.CULL, SeedStage.CULLED),
    TamiyoAction.CHANGE_BLUEPRINT: (CommandType.CULL, SeedStage.CULLED),  # Cull then germinate
}


@dataclass
class TamiyoDecision:
    """A decision made by Tamiyo.

    This is Tamiyo's internal decision format. For sending to Kasmina,
    use to_command() to convert to the canonical AdaptationCommand format.
    """
    action: TamiyoAction
    target_seed_id: str | None = None
    blueprint_id: str | None = None
    reason: str = ""
    confidence: float = 1.0

    def __str__(self) -> str:
        parts = [f"Action: {self.action.name}"]
        if self.target_seed_id:
            parts.append(f"Target: {self.target_seed_id}")
        if self.blueprint_id:
            parts.append(f"Blueprint: {self.blueprint_id}")
        if self.reason:
            parts.append(f"Reason: {self.reason}")
        return " | ".join(parts)

    def to_command(self) -> AdaptationCommand:
        """Convert to Leyline's canonical AdaptationCommand format."""
        command_type, target_stage = _ACTION_TO_COMMAND.get(
            self.action,
            (CommandType.REQUEST_STATE, None)
        )

        # Determine risk level based on action
        if self.action == TamiyoAction.WAIT:
            risk = RiskLevel.GREEN
        elif self.action == TamiyoAction.GERMINATE:
            risk = RiskLevel.YELLOW
        elif self.action in (TamiyoAction.ADVANCE_TRAINING, TamiyoAction.ADVANCE_BLENDING):
            risk = RiskLevel.YELLOW
        elif self.action == TamiyoAction.ADVANCE_FOSSILIZE:
            risk = RiskLevel.ORANGE  # Permanent change
        elif self.action in (TamiyoAction.CULL, TamiyoAction.CHANGE_BLUEPRINT):
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
    "TamiyoAction",
    "TamiyoDecision",
]
