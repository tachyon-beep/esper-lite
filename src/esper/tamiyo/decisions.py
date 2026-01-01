"""Tamiyo Decisions - Strategic decision structures.

Defines the decisions made by strategic controllers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from esper.leyline.actions import get_blueprint_from_action_name


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
        return get_blueprint_from_action_name(self.action.name)


__all__ = [
    "TamiyoDecision",
]
