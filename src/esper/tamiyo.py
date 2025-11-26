"""Tamiyo: Strategic Controller for Seed Lifecycle

This module handles:
- Training signal observation and analysis
- Policy decisions for seed lifecycle management
- Heuristic and (future) learned policies

Uses Leyline contracts for stage definitions and inter-component communication.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, TYPE_CHECKING

# Import contracts from leyline (source of truth for data shapes)
from esper.leyline import (
    SeedStage,
    is_terminal_stage,
    is_failure_stage,
    CommandType,
    RiskLevel,
    AdaptationCommand,
    TrainingMetrics as LeylineTrainingMetrics,
    TrainingSignals as LeylineTrainingSignals,
    FieldReport as LeylineFieldReport,
)

# Import runtime types from kasmina (avoid circular imports with TYPE_CHECKING)
if TYPE_CHECKING:
    from esper.kasmina import SeedState


# =============================================================================
# Training Signals
# =============================================================================

@dataclass
class TrainingSignals:
    """Observations from the training loop that Tamiyo uses to make decisions.

    This is Tamiyo's view of training signals, which includes full SeedState objects
    for direct access to seed information. For cross-component communication,
    use to_leyline() to convert to the canonical LeylineTrainingSignals format.
    """

    # Current state
    epoch: int
    global_step: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float

    # Derived signals
    loss_delta: float = 0.0          # Change from previous epoch
    accuracy_delta: float = 0.0      # Change from previous epoch
    plateau_epochs: int = 0          # Epochs without improvement
    best_val_accuracy: float = 0.0   # Best seen so far

    # Seed-related (full SeedState objects for Tamiyo's decision making)
    active_seeds: list["SeedState"] = field(default_factory=list)
    available_slots: int = 0

    # History (for trend analysis)
    loss_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)

    def to_leyline(self) -> LeylineTrainingSignals:
        """Convert to Leyline's canonical TrainingSignals format."""
        metrics = LeylineTrainingMetrics(
            epoch=self.epoch,
            global_step=self.global_step,
            train_loss=self.train_loss,
            val_loss=self.val_loss,
            loss_delta=self.loss_delta,
            train_accuracy=self.train_accuracy,
            val_accuracy=self.val_accuracy,
            accuracy_delta=self.accuracy_delta,
            plateau_epochs=self.plateau_epochs,
            best_val_accuracy=self.best_val_accuracy,
        )
        return LeylineTrainingSignals(
            metrics=metrics,
            active_seeds=[s.seed_id for s in self.active_seeds],
            available_slots=self.available_slots,
            loss_history=self.loss_history.copy(),
            accuracy_history=self.accuracy_history.copy(),
        )


class SignalTracker:
    """Tracks training signals over time and computes derived metrics."""

    def __init__(
        self,
        plateau_threshold: float = 0.5,  # Min improvement to not count as plateau
        history_window: int = 10,
    ):
        self.plateau_threshold = plateau_threshold
        self.history_window = history_window

        self._loss_history: deque[float] = deque(maxlen=history_window)
        self._accuracy_history: deque[float] = deque(maxlen=history_window)
        self._best_accuracy: float = 0.0
        self._plateau_count: int = 0
        self._prev_accuracy: float = 0.0
        self._prev_loss: float = float('inf')

    def update(
        self,
        epoch: int,
        global_step: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        active_seeds: list["SeedState"],
        available_slots: int = 1,
    ) -> TrainingSignals:
        """Update tracker and return current signals."""

        # Compute deltas
        loss_delta = self._prev_loss - val_loss  # Positive = improvement
        accuracy_delta = val_accuracy - self._prev_accuracy

        # Update plateau counter
        if accuracy_delta < self.plateau_threshold:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        # Update best
        if val_accuracy > self._best_accuracy:
            self._best_accuracy = val_accuracy

        # Update history
        self._loss_history.append(val_loss)
        self._accuracy_history.append(val_accuracy)

        # Build signals
        signals = TrainingSignals(
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            loss_delta=loss_delta,
            accuracy_delta=accuracy_delta,
            plateau_epochs=self._plateau_count,
            best_val_accuracy=self._best_accuracy,
            active_seeds=active_seeds,
            available_slots=available_slots,
            loss_history=list(self._loss_history),
            accuracy_history=list(self._accuracy_history),
        )

        # Update previous values for next iteration
        self._prev_loss = val_loss
        self._prev_accuracy = val_accuracy

        return signals

    def reset(self) -> None:
        """Reset tracker state."""
        self._loss_history.clear()
        self._accuracy_history.clear()
        self._best_accuracy = 0.0
        self._plateau_count = 0
        self._prev_accuracy = 0.0
        self._prev_loss = float('inf')


# =============================================================================
# Tamiyo Actions
# =============================================================================

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


# =============================================================================
# Policy Protocol
# =============================================================================

class TamiyoPolicy(Protocol):
    """Protocol for Tamiyo policy implementations."""

    def decide(self, signals: TrainingSignals) -> TamiyoDecision:
        """Make a decision based on current training signals."""
        ...

    def reset(self) -> None:
        """Reset policy state."""
        ...


# =============================================================================
# Heuristic Policy (Tiny Tamiyo)
# =============================================================================

@dataclass
class HeuristicPolicyConfig:
    """Configuration for the heuristic policy."""

    # When to germinate
    plateau_epochs_to_germinate: int = 3
    min_epochs_before_germinate: int = 5

    # Training phase
    min_training_epochs: int = 3
    training_improvement_threshold: float = 1.0  # Min % improvement to advance

    # Blending phase
    blending_epochs: int = 5

    # Culling
    cull_after_epochs_without_improvement: int = 5
    cull_if_accuracy_drops_by: float = 2.0  # % drop triggers cull

    # Blueprint selection
    blueprint_rotation: list[str] = field(
        default_factory=lambda: ["conv_enhance", "attention", "norm", "depthwise"]
    )


class HeuristicTamiyo:
    """Heuristic-based Tamiyo policy (Tiny Tamiyo).

    Makes decisions based on simple rules:
    - Germinate when training plateaus
    - Advance when seed shows improvement
    - Cull when seed hurts performance
    """

    def __init__(self, config: HeuristicPolicyConfig | None = None):
        self.config = config or HeuristicPolicyConfig()
        self._blueprint_index = 0
        self._germination_count = 0
        self._decisions_made: list[TamiyoDecision] = []

    def decide(self, signals: TrainingSignals) -> TamiyoDecision:
        """Make a decision based on training signals."""

        # Check if we have any active seeds
        active_seeds = signals.active_seeds

        if not active_seeds:
            # No active seeds - should we germinate?
            decision = self._decide_germination(signals)
        else:
            # Have active seeds - manage them
            decision = self._decide_seed_management(signals, active_seeds)

        self._decisions_made.append(decision)
        return decision

    def _decide_germination(self, signals: TrainingSignals) -> TamiyoDecision:
        """Decide whether to germinate a new seed."""

        # Don't germinate too early
        if signals.epoch < self.config.min_epochs_before_germinate:
            return TamiyoDecision(
                action=TamiyoAction.WAIT,
                reason=f"Too early (epoch {signals.epoch} < {self.config.min_epochs_before_germinate})"
            )

        # Check for plateau
        if signals.plateau_epochs >= self.config.plateau_epochs_to_germinate:
            blueprint_id = self._get_next_blueprint()
            self._germination_count += 1
            return TamiyoDecision(
                action=TamiyoAction.GERMINATE,
                blueprint_id=blueprint_id,
                reason=f"Plateau detected ({signals.plateau_epochs} epochs without improvement)",
                confidence=min(1.0, signals.plateau_epochs / 5.0),
            )

        # No action needed
        return TamiyoDecision(
            action=TamiyoAction.WAIT,
            reason="Training progressing normally"
        )

    def _decide_seed_management(
        self,
        signals: TrainingSignals,
        active_seeds: list["SeedState"],
    ) -> TamiyoDecision:
        """Decide how to manage active seeds."""

        for seed in active_seeds:
            # Skip terminal and failure states using leyline helpers
            if is_terminal_stage(seed.stage) or is_failure_stage(seed.stage):
                continue

            # Handle based on current stage
            if seed.stage == SeedStage.GERMINATED:
                # Ready to start training
                return TamiyoDecision(
                    action=TamiyoAction.ADVANCE_TRAINING,
                    target_seed_id=seed.seed_id,
                    reason="Seed germinated, starting isolated training",
                )

            elif seed.stage == SeedStage.TRAINING:
                decision = self._evaluate_training_seed(signals, seed)
                if decision.action != TamiyoAction.WAIT:
                    return decision

            elif seed.stage == SeedStage.BLENDING:
                decision = self._evaluate_blending_seed(signals, seed)
                if decision.action != TamiyoAction.WAIT:
                    return decision

        return TamiyoDecision(
            action=TamiyoAction.WAIT,
            reason="Seeds progressing normally"
        )

    def _evaluate_training_seed(
        self,
        signals: TrainingSignals,
        seed: "SeedState",
    ) -> TamiyoDecision:
        """Evaluate a seed in TRAINING stage."""

        # Need minimum training time
        if seed.epochs_in_stage < self.config.min_training_epochs:
            return TamiyoDecision(
                action=TamiyoAction.WAIT,
                target_seed_id=seed.seed_id,
                reason=f"Training epoch {seed.epochs_in_stage}/{self.config.min_training_epochs}"
            )

        # Check improvement
        improvement = seed.metrics.improvement_since_stage_start

        if improvement >= self.config.training_improvement_threshold:
            # Good improvement - advance to blending
            return TamiyoDecision(
                action=TamiyoAction.ADVANCE_BLENDING,
                target_seed_id=seed.seed_id,
                reason=f"Good improvement ({improvement:.2f}%), advancing to blending",
                confidence=min(1.0, improvement / 5.0),
            )

        # Check if we should cull
        if seed.epochs_in_stage >= self.config.cull_after_epochs_without_improvement:
            if improvement < -self.config.cull_if_accuracy_drops_by:
                return TamiyoDecision(
                    action=TamiyoAction.CULL,
                    target_seed_id=seed.seed_id,
                    reason=f"Seed hurting performance ({improvement:.2f}%)",
                )

        return TamiyoDecision(
            action=TamiyoAction.WAIT,
            target_seed_id=seed.seed_id,
            reason=f"Training in progress, improvement: {improvement:.2f}%"
        )

    def _evaluate_blending_seed(
        self,
        signals: TrainingSignals,
        seed: "SeedState",
    ) -> TamiyoDecision:
        """Evaluate a seed in BLENDING stage."""

        # Check if blending is complete
        if seed.epochs_in_stage >= self.config.blending_epochs:
            # Check final improvement
            improvement = seed.metrics.total_improvement

            if improvement > 0:
                return TamiyoDecision(
                    action=TamiyoAction.ADVANCE_FOSSILIZE,
                    target_seed_id=seed.seed_id,
                    reason=f"Blending complete, total improvement: {improvement:.2f}%",
                )
            else:
                return TamiyoDecision(
                    action=TamiyoAction.CULL,
                    target_seed_id=seed.seed_id,
                    reason=f"Blending complete but no improvement ({improvement:.2f}%)",
                )

        return TamiyoDecision(
            action=TamiyoAction.WAIT,
            target_seed_id=seed.seed_id,
            reason=f"Blending epoch {seed.epochs_in_stage}/{self.config.blending_epochs}"
        )

    def _get_next_blueprint(self) -> str:
        """Get the next blueprint to try (round-robin)."""
        blueprints = self.config.blueprint_rotation
        blueprint_id = blueprints[self._blueprint_index % len(blueprints)]
        self._blueprint_index += 1
        return blueprint_id

    def reset(self) -> None:
        """Reset policy state."""
        self._blueprint_index = 0
        self._germination_count = 0
        self._decisions_made.clear()

    @property
    def decisions(self) -> list[TamiyoDecision]:
        """Get history of decisions made."""
        return self._decisions_made.copy()


# =============================================================================
# Field Reports (for future Simic training)
# =============================================================================

@dataclass
class FieldReport:
    """Report of a seed's performance for training Tamiyo's policy.

    These are collected and used by Simic to train better policies.
    For cross-component communication, use to_leyline() to convert to
    the canonical LeylineFieldReport format.
    """
    seed_id: str
    blueprint_id: str
    signals_at_germination: TrainingSignals
    signals_at_completion: TrainingSignals
    final_stage: SeedStage
    decisions_made: list[TamiyoDecision]
    total_epochs: int
    improvement: float
    success: bool  # True if fossilized, False if culled

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "seed_id": self.seed_id,
            "blueprint_id": self.blueprint_id,
            "final_stage": self.final_stage.name,
            "total_epochs": self.total_epochs,
            "improvement": self.improvement,
            "success": self.success,
            "germination_epoch": self.signals_at_germination.epoch,
            "completion_epoch": self.signals_at_completion.epoch,
            "germination_accuracy": self.signals_at_germination.val_accuracy,
            "completion_accuracy": self.signals_at_completion.val_accuracy,
        }

    def to_leyline(self) -> LeylineFieldReport:
        """Convert to Leyline's canonical FieldReport format for Simic."""
        return LeylineFieldReport(
            seed_id=self.seed_id,
            blueprint_id=self.blueprint_id,
            final_stage=self.final_stage,
            success=self.success,
            total_epochs=self.total_epochs,
            accuracy_at_germination=self.signals_at_germination.val_accuracy,
            accuracy_at_completion=self.signals_at_completion.val_accuracy,
            total_improvement=self.improvement,
            best_accuracy_achieved=max(
                self.signals_at_germination.best_val_accuracy,
                self.signals_at_completion.best_val_accuracy,
            ),
            signals_at_germination=self.signals_at_germination.to_leyline(),
            commands_received=[d.to_command().command_id for d in self.decisions_made],
        )


class FieldReportCollector:
    """Collects field reports for Simic training."""

    def __init__(self):
        self._reports: list[FieldReport] = []
        self._pending: dict[str, dict] = {}  # seed_id -> partial report data

    def start_tracking(
        self,
        seed_id: str,
        blueprint_id: str,
        signals: TrainingSignals,
    ) -> None:
        """Start tracking a seed's lifecycle."""
        self._pending[seed_id] = {
            "seed_id": seed_id,
            "blueprint_id": blueprint_id,
            "signals_at_germination": signals,
            "decisions": [],
        }

    def record_decision(self, seed_id: str, decision: TamiyoDecision) -> None:
        """Record a decision made for a seed."""
        if seed_id in self._pending:
            self._pending[seed_id]["decisions"].append(decision)

    def complete_tracking(
        self,
        seed_id: str,
        signals: TrainingSignals,
        final_stage: SeedStage,
    ) -> FieldReport | None:
        """Complete tracking and generate field report."""
        if seed_id not in self._pending:
            return None

        data = self._pending.pop(seed_id)
        germ_signals = data["signals_at_germination"]

        report = FieldReport(
            seed_id=seed_id,
            blueprint_id=data["blueprint_id"],
            signals_at_germination=germ_signals,
            signals_at_completion=signals,
            final_stage=final_stage,
            decisions_made=data["decisions"],
            total_epochs=signals.epoch - germ_signals.epoch,
            improvement=signals.val_accuracy - germ_signals.val_accuracy,
            success=final_stage == SeedStage.FOSSILIZED,
        )

        self._reports.append(report)
        return report

    @property
    def reports(self) -> list[FieldReport]:
        """Get all collected reports."""
        return self._reports.copy()

    def get_leyline_reports(self) -> list[LeylineFieldReport]:
        """Get all reports converted to Leyline format for Simic."""
        return [r.to_leyline() for r in self._reports]

    def clear(self) -> None:
        """Clear all reports."""
        self._reports.clear()
        self._pending.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Signals
    "TrainingSignals",
    "SignalTracker",
    # Actions
    "TamiyoAction",
    "TamiyoDecision",
    # Policies
    "TamiyoPolicy",
    "HeuristicTamiyo",
    "HeuristicPolicyConfig",
    # Field Reports
    "FieldReport",
    "FieldReportCollector",
]
