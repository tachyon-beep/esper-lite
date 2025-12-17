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


@dataclass
class TamiyoState:
    """Tamiyo agent state for the Tamiyo Strip and Detail Panel.

    Contains PPO vitals, action distribution, and decision confidence
    for rendering agent brain telemetry.
    """

    # Action distribution (counts over recent window)
    action_counts: dict[str, int] = field(default_factory=dict)
    recent_actions: list[str] = field(default_factory=list)  # Last N actions as codes

    # Confidence (from policy logprobs)
    confidence_mean: float = 0.0
    confidence_min: float = 0.0
    confidence_max: float = 0.0
    confidence_history: list[float] = field(default_factory=list)  # For sparkline

    # Exploration
    exploration_pct: float = 0.0  # Entropy as % of max entropy

    # PPO vitals
    kl_divergence: float = 0.0
    entropy: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0

    # Trends (for arrows: positive = rising, negative = falling)
    kl_trend: float = 0.0
    entropy_trend: float = 0.0
    ev_trend: float = 0.0

    # Status flags
    entropy_collapsed: bool = False
    ev_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "action_counts": self.action_counts,
            "recent_actions": self.recent_actions,
            "confidence_mean": self.confidence_mean,
            "confidence_min": self.confidence_min,
            "confidence_max": self.confidence_max,
            "confidence_history": self.confidence_history,
            "exploration_pct": self.exploration_pct,
            "kl_divergence": self.kl_divergence,
            "entropy": self.entropy,
            "clip_fraction": self.clip_fraction,
            "explained_variance": self.explained_variance,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "kl_trend": self.kl_trend,
            "entropy_trend": self.entropy_trend,
            "ev_trend": self.ev_trend,
            "entropy_collapsed": self.entropy_collapsed,
            "ev_warning": self.ev_warning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TamiyoState:
        """Reconstruct from dict."""
        return cls(
            action_counts=data.get("action_counts", {}),
            recent_actions=data.get("recent_actions", []),
            confidence_mean=data.get("confidence_mean", 0.0),
            confidence_min=data.get("confidence_min", 0.0),
            confidence_max=data.get("confidence_max", 0.0),
            confidence_history=data.get("confidence_history", []),
            exploration_pct=data.get("exploration_pct", 0.0),
            kl_divergence=data.get("kl_divergence", 0.0),
            entropy=data.get("entropy", 0.0),
            clip_fraction=data.get("clip_fraction", 0.0),
            explained_variance=data.get("explained_variance", 0.0),
            grad_norm=data.get("grad_norm", 0.0),
            learning_rate=data.get("learning_rate", 0.0),
            kl_trend=data.get("kl_trend", 0.0),
            entropy_trend=data.get("entropy_trend", 0.0),
            ev_trend=data.get("ev_trend", 0.0),
            entropy_collapsed=data.get("entropy_collapsed", False),
            ev_warning=data.get("ev_warning", False),
        )


@dataclass
class ConnectionStatus:
    """Connection status for header display.

    Tracks whether telemetry is flowing and how stale it is.
    """

    connected: bool
    last_event_ts: float  # Unix timestamp
    staleness_s: float  # Seconds since last event

    @property
    def display_text(self) -> str:
        """Human-readable status text."""
        if not self.connected:
            return "Disconnected"
        if self.staleness_s < 2.0:
            return "Live"
        if self.staleness_s < 5.0:
            return f"Live ({self.staleness_s:.0f}s)"
        return f"Stale ({self.staleness_s:.0f}s)"

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "connected": self.connected,
            "last_event_ts": self.last_event_ts,
            "staleness_s": self.staleness_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConnectionStatus:
        """Reconstruct from dict."""
        return cls(
            connected=data["connected"],
            last_event_ts=data["last_event_ts"],
            staleness_s=data["staleness_s"],
        )


@dataclass
class DeviceVitals:
    """GPU device vitals for header resource display."""

    device_id: int
    name: str
    utilization_pct: float
    memory_used_gb: float
    memory_total_gb: float
    temperature_c: int = 0

    @property
    def memory_pct(self) -> float:
        """Memory usage as percentage."""
        if self.memory_total_gb == 0:
            return 0.0
        return (self.memory_used_gb / self.memory_total_gb) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "utilization_pct": self.utilization_pct,
            "memory_used_gb": self.memory_used_gb,
            "memory_total_gb": self.memory_total_gb,
            "temperature_c": self.temperature_c,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceVitals:
        """Reconstruct from dict."""
        return cls(
            device_id=data["device_id"],
            name=data["name"],
            utilization_pct=data["utilization_pct"],
            memory_used_gb=data["memory_used_gb"],
            memory_total_gb=data["memory_total_gb"],
            temperature_c=data.get("temperature_c", 0),
        )
