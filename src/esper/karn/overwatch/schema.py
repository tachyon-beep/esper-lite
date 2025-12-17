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


@dataclass
class EnvSummary:
    """Summary of a single training environment for Flight Board.

    Contains all information needed to render one row in the Flight Board:
    environment identity, status, throughput, slots, and anomaly info.
    """

    # Identity
    env_id: int
    device_id: int  # GPU device index
    status: str  # OK, INFO, WARN, CRIT

    # Throughput
    throughput_fps: float = 0.0
    step_time_ms: float = 0.0

    # Metrics
    reward_last: float = 0.0
    task_metric: float = 0.0  # Task-specific metric (e.g., accuracy)
    task_metric_delta: float = 0.0

    # Slots (keyed by slot_id like "r0c1")
    slots: dict[str, SlotChipState] = field(default_factory=dict)

    # Anomaly detection
    anomaly_score: float = 0.0  # 0.0-1.0, higher = more anomalous
    anomaly_reasons: list[str] = field(default_factory=list)

    # Staleness
    last_update_ts: float = 0.0  # Unix timestamp of last telemetry

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "env_id": self.env_id,
            "device_id": self.device_id,
            "status": self.status,
            "throughput_fps": self.throughput_fps,
            "step_time_ms": self.step_time_ms,
            "reward_last": self.reward_last,
            "task_metric": self.task_metric,
            "task_metric_delta": self.task_metric_delta,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "anomaly_score": self.anomaly_score,
            "anomaly_reasons": self.anomaly_reasons,
            "last_update_ts": self.last_update_ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvSummary:
        """Reconstruct from dict."""
        slots = {
            k: SlotChipState.from_dict(v)
            for k, v in data.get("slots", {}).items()
        }
        return cls(
            env_id=data["env_id"],
            device_id=data["device_id"],
            status=data["status"],
            throughput_fps=data.get("throughput_fps", 0.0),
            step_time_ms=data.get("step_time_ms", 0.0),
            reward_last=data.get("reward_last", 0.0),
            task_metric=data.get("task_metric", 0.0),
            task_metric_delta=data.get("task_metric_delta", 0.0),
            slots=slots,
            anomaly_score=data.get("anomaly_score", 0.0),
            anomaly_reasons=data.get("anomaly_reasons", []),
            last_update_ts=data.get("last_update_ts", 0.0),
        )
