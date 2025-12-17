"""Overwatch TUI Snapshot Schema.

Defines the data structures that flow from telemetry aggregator to UI renderer.
All schemas are JSON-serializable for replay support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlotChipState:
    """State of a single slot for UI rendering.

    Represents one slot chip in the Flight Board, showing seed lifecycle
    stage, blending progress, and gate status.
    """

    # Identity
    slot_id: str  # Canonical format: "r0c1" (row 0, column 1)
    stage: str  # SeedStage name: DORMANT, GERMINATED, TRAINING, etc.
    blueprint_id: str  # Blueprint used for this seed (empty if dormant)
    alpha: float  # Blend weight 0.0-1.0

    # Progress
    epochs_in_stage: int = 0
    epochs_total: int = 0  # Total epochs since germination

    # Gate status
    gate_last: str | None = None  # Last gate evaluated: G0, G1, G2, G3
    gate_passed: bool | None = None  # Did the gate pass?

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "slot_id": self.slot_id,
            "stage": self.stage,
            "blueprint_id": self.blueprint_id,
            "alpha": self.alpha,
            "epochs_in_stage": self.epochs_in_stage,
            "epochs_total": self.epochs_total,
            "gate_last": self.gate_last,
            "gate_passed": self.gate_passed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlotChipState:
        """Reconstruct from dict."""
        return cls(
            slot_id=data["slot_id"],
            stage=data["stage"],
            blueprint_id=data["blueprint_id"],
            alpha=data["alpha"],
            epochs_in_stage=data.get("epochs_in_stage", 0),
            epochs_total=data.get("epochs_total", 0),
            gate_last=data.get("gate_last"),
            gate_passed=data.get("gate_passed"),
        )
