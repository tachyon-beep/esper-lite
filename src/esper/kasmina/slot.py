"""Kasmina Slot - Seed lifecycle management.

The SeedSlot manages a single seed module through its lifecycle:
germination -> training -> blending -> fossilization/culling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import torch
import torch.nn as nn

from esper.leyline import (
    # Lifecycle
    SeedStage,
    VALID_TRANSITIONS,
    is_valid_transition,
    is_terminal_stage,
    is_active_stage,
    is_failure_stage,
    # Reports
    SeedMetrics as LeylineSeedMetrics,
    SeedStateReport,
    # Gates
    GateLevel,
    GateResult,
    # Telemetry
    TelemetryEvent,
    TelemetryEventType,
)


# =============================================================================
# Seed Metrics (extends Leyline contract with behavior)
# =============================================================================

@dataclass(slots=True)
class SeedMetrics:
    """Metrics tracked for a seed throughout its lifecycle.

    This extends the Leyline SeedMetrics contract with behavior.
    Uses __slots__ for reduced memory footprint and faster attribute access.
    """

    epochs_total: int = 0
    epochs_in_current_stage: int = 0

    initial_val_accuracy: float = 0.0
    current_val_accuracy: float = 0.0
    best_val_accuracy: float = 0.0
    accuracy_at_stage_start: float = 0.0

    isolation_violations: int = 0
    gradient_norm_avg: float = 0.0

    current_alpha: float = 0.0
    alpha_ramp_step: int = 0

    def record_accuracy(self, accuracy: float) -> None:
        """Record a new accuracy measurement."""
        if self.epochs_total == 0:
            self.initial_val_accuracy = accuracy
            self.accuracy_at_stage_start = accuracy
        self.current_val_accuracy = accuracy
        self.best_val_accuracy = max(self.best_val_accuracy, accuracy)
        self.epochs_total += 1
        self.epochs_in_current_stage += 1

    @property
    def total_improvement(self) -> float:
        return self.current_val_accuracy - self.initial_val_accuracy

    @property
    def improvement_since_stage_start(self) -> float:
        return self.current_val_accuracy - self.accuracy_at_stage_start

    def reset_stage_baseline(self) -> None:
        """Reset the stage start baseline (call on stage transitions)."""
        self.accuracy_at_stage_start = self.current_val_accuracy
        self.epochs_in_current_stage = 0

    def to_leyline(self) -> LeylineSeedMetrics:
        """Convert to Leyline contract type."""
        return LeylineSeedMetrics(
            epochs_total=self.epochs_total,
            epochs_in_current_stage=self.epochs_in_current_stage,
            initial_val_accuracy=self.initial_val_accuracy,
            current_val_accuracy=self.current_val_accuracy,
            best_val_accuracy=self.best_val_accuracy,
            accuracy_at_stage_start=self.accuracy_at_stage_start,
            total_improvement=self.total_improvement,
            improvement_since_stage_start=self.improvement_since_stage_start,
            isolation_violations=self.isolation_violations,
            gradient_norm_avg=self.gradient_norm_avg,
            current_alpha=self.current_alpha,
            alpha_ramp_step=self.alpha_ramp_step,
        )


# =============================================================================
# Seed State
# =============================================================================

@dataclass
class SeedState:
    """Complete state of a seed through its lifecycle."""

    seed_id: str
    blueprint_id: str
    slot_id: str = ""

    stage: SeedStage = SeedStage.DORMANT
    previous_stage: SeedStage = SeedStage.UNKNOWN
    stage_entered_at: datetime = field(default_factory=datetime.utcnow)

    alpha: float = 0.0
    metrics: SeedMetrics = field(default_factory=SeedMetrics)

    # Flags
    is_healthy: bool = True
    is_paused: bool = False

    # History
    stage_history: list[tuple[SeedStage, datetime]] = field(default_factory=list)

    def can_transition_to(self, new_stage: SeedStage) -> bool:
        """Check if transition to new_stage is valid per Leyline contract."""
        return is_valid_transition(self.stage, new_stage)

    def transition(self, new_stage: SeedStage) -> bool:
        """Transition to a new stage. Returns True if successful."""
        if not self.can_transition_to(new_stage):
            return False

        self.previous_stage = self.stage
        self.stage = new_stage
        self.stage_entered_at = datetime.utcnow()
        self.stage_history.append((new_stage, self.stage_entered_at))
        self.metrics.reset_stage_baseline()
        return True

    @property
    def epochs_in_stage(self) -> int:
        """Convenience property for epochs in current stage."""
        return self.metrics.epochs_in_current_stage

    def increment_epoch(self) -> None:
        """Increment epoch counters (use when accuracy is recorded separately)."""
        self.metrics.epochs_total += 1
        self.metrics.epochs_in_current_stage += 1

    def record_epoch(self, accuracy: float) -> None:
        """Record epoch with accuracy (preferred method)."""
        self.metrics.record_accuracy(accuracy)

    def to_report(self) -> SeedStateReport:
        """Convert to Leyline SeedStateReport."""
        return SeedStateReport(
            seed_id=self.seed_id,
            slot_id=self.slot_id,
            blueprint_id=self.blueprint_id,
            stage=self.stage,
            previous_stage=self.previous_stage,
            stage_entered_at=self.stage_entered_at,
            metrics=self.metrics.to_leyline(),
            is_healthy=self.is_healthy,
            is_improving=self.metrics.improvement_since_stage_start > 0,
            needs_attention=not self.is_healthy or self.metrics.isolation_violations > 0,
        )


# =============================================================================
# Quality Gates
# =============================================================================

class QualityGates:
    """Quality gate checks for stage transitions.

    Each gate validates that a seed is ready for the next stage.
    """

    def __init__(
        self,
        min_training_improvement: float = 0.5,
        min_blending_epochs: int = 3,
        max_isolation_violations: int = 10,
        min_shadowing_correlation: float = 0.9,
        min_probation_stability: float = 0.95,
    ):
        self.min_training_improvement = min_training_improvement
        self.min_blending_epochs = min_blending_epochs
        self.max_isolation_violations = max_isolation_violations
        self.min_shadowing_correlation = min_shadowing_correlation
        self.min_probation_stability = min_probation_stability

    def check_gate(self, state: SeedState, target_stage: SeedStage) -> GateResult:
        """Check if seed passes the gate for target stage."""

        gate = self._get_gate_level(target_stage)

        if gate == GateLevel.G0:
            return self._check_g0(state)
        elif gate == GateLevel.G1:
            return self._check_g1(state)
        elif gate == GateLevel.G2:
            return self._check_g2(state)
        elif gate == GateLevel.G3:
            return self._check_g3(state)
        elif gate == GateLevel.G4:
            return self._check_g4(state)
        elif gate == GateLevel.G5:
            return self._check_g5(state)

        # Default: pass
        return GateResult(gate=gate, passed=True, score=1.0)

    def _get_gate_level(self, target_stage: SeedStage) -> GateLevel:
        """Map target stage to gate level."""
        mapping = {
            SeedStage.GERMINATED: GateLevel.G0,
            SeedStage.TRAINING: GateLevel.G1,
            SeedStage.BLENDING: GateLevel.G2,
            SeedStage.SHADOWING: GateLevel.G3,
            SeedStage.PROBATIONARY: GateLevel.G4,
            SeedStage.FOSSILIZED: GateLevel.G5,
        }
        return mapping.get(target_stage, GateLevel.G0)

    def _check_g0(self, state: SeedState) -> GateResult:
        """G0: Basic sanity for germination."""
        checks_passed = []
        checks_failed = []

        # Just check that we have required fields
        if state.seed_id:
            checks_passed.append("seed_id_present")
        else:
            checks_failed.append("seed_id_missing")

        if state.blueprint_id:
            checks_passed.append("blueprint_id_present")
        else:
            checks_failed.append("blueprint_id_missing")

        passed = len(checks_failed) == 0
        return GateResult(
            gate=GateLevel.G0,
            passed=passed,
            score=1.0 if passed else 0.0,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            message="Germination gate" if passed else "Missing required fields",
        )

    def _check_g1(self, state: SeedState) -> GateResult:
        """G1: Training readiness."""
        # Currently just checks germination completed
        passed = state.stage == SeedStage.GERMINATED
        return GateResult(
            gate=GateLevel.G1,
            passed=passed,
            score=1.0 if passed else 0.0,
            checks_passed=["germinated"] if passed else [],
            checks_failed=[] if passed else ["not_germinated"],
            message="Ready for training" if passed else "Not germinated",
        )

    def _check_g2(self, state: SeedState) -> GateResult:
        """G2: Blending readiness - requires positive improvement."""
        checks_passed = []
        checks_failed = []

        improvement = state.metrics.improvement_since_stage_start

        if improvement >= self.min_training_improvement:
            checks_passed.append(f"improvement_{improvement:.2f}%")
        else:
            checks_failed.append(f"insufficient_improvement_{improvement:.2f}%")

        if state.metrics.isolation_violations <= self.max_isolation_violations:
            checks_passed.append("isolation_ok")
        else:
            checks_failed.append(f"isolation_violations_{state.metrics.isolation_violations}")

        passed = len(checks_failed) == 0
        score = min(1.0, improvement / 5.0) if improvement > 0 else 0.0

        return GateResult(
            gate=GateLevel.G2,
            passed=passed,
            score=score,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            message=f"Improvement: {improvement:.2f}%",
        )

    def _check_g3(self, state: SeedState) -> GateResult:
        """G3: Shadowing readiness - blending completed."""
        checks_passed = []
        checks_failed = []

        if state.metrics.epochs_in_current_stage >= self.min_blending_epochs:
            checks_passed.append("blending_complete")
        else:
            checks_failed.append(f"blending_incomplete_{state.metrics.epochs_in_current_stage}")

        if state.alpha >= 0.95:
            checks_passed.append("alpha_high")
        else:
            checks_failed.append(f"alpha_low_{state.alpha:.2f}")

        passed = len(checks_failed) == 0
        return GateResult(
            gate=GateLevel.G3,
            passed=passed,
            score=state.alpha,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def _check_g4(self, state: SeedState) -> GateResult:
        """G4: Probation readiness - shadowing validated."""
        # For now, just check shadowing was done
        passed = state.stage == SeedStage.SHADOWING
        return GateResult(
            gate=GateLevel.G4,
            passed=passed,
            score=1.0 if passed else 0.0,
            checks_passed=["shadowing_complete"] if passed else [],
            checks_failed=[] if passed else ["shadowing_incomplete"],
        )

    def _check_g5(self, state: SeedState) -> GateResult:
        """G5: Fossilization readiness - probation passed."""
        checks_passed = []
        checks_failed = []

        # Check total improvement
        if state.metrics.total_improvement > 0:
            checks_passed.append(f"positive_improvement_{state.metrics.total_improvement:.2f}%")
        else:
            checks_failed.append("no_improvement")

        # Check health
        if state.is_healthy:
            checks_passed.append("healthy")
        else:
            checks_failed.append("unhealthy")

        passed = len(checks_failed) == 0
        return GateResult(
            gate=GateLevel.G5,
            passed=passed,
            score=min(1.0, state.metrics.total_improvement / 10.0),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )


# =============================================================================
# Seed Slot
# =============================================================================

class SeedSlot:
    """A slot in the model where a seed can be attached.

    Manages the full lifecycle of a seed with quality gates.

    Args:
        slot_id: Unique identifier for this slot.
        channels: Number of channels for seed modules.
        device: Device to place seed on.
        gates: Quality gates for stage transitions.
        on_telemetry: Callback for telemetry events.
        fast_mode: If True, disable telemetry and isolation monitoring
            for high-throughput PPO rollouts. Default: False.
    """

    def __init__(
        self,
        slot_id: str,
        channels: int,
        device: torch.device | str = "cpu",
        gates: QualityGates | None = None,
        on_telemetry: Callable[[TelemetryEvent], None] | None = None,
        fast_mode: bool = False,
    ):
        self.slot_id = slot_id
        self.channels = channels
        self.device = device
        self.gates = gates or QualityGates()
        self.on_telemetry = on_telemetry
        self.fast_mode = fast_mode  # Disable telemetry/isolation for PPO

        self.seed: nn.Module | None = None
        self.state: SeedState | None = None
        self.alpha_schedule = None
        self.isolate_gradients: bool = False

        # Only create isolation monitor if not in fast mode
        self.isolation_monitor = None

    @property
    def is_active(self) -> bool:
        """Check if slot has an active seed."""
        return self.seed is not None and self.state is not None

    @property
    def alpha(self) -> float:
        """Current blending alpha."""
        return self.state.alpha if self.state else 0.0

    def germinate(
        self,
        blueprint_id: str,
        seed_id: str | None = None,
        host_module: nn.Module | None = None,
    ) -> SeedState:
        """Germinate a new seed in this slot."""
        # Import here to avoid circular dependency
        from esper.kasmina.blueprints import BlueprintCatalog

        if self.is_active and not is_failure_stage(self.state.stage):
            raise RuntimeError(f"Slot {self.slot_id} already has active seed")

        # Create seed
        self.seed = BlueprintCatalog.create_seed(blueprint_id, self.channels)
        self.seed = self.seed.to(self.device)

        # Initialize state
        seed_id = seed_id or f"{self.slot_id}-{blueprint_id}"
        self.state = SeedState(
            seed_id=seed_id,
            blueprint_id=blueprint_id,
            slot_id=self.slot_id,
            stage=SeedStage.DORMANT,
        )

        # Check G0 gate and transition to GERMINATED
        gate_result = self.gates.check_gate(self.state, SeedStage.GERMINATED)
        if gate_result.passed:
            self.state.transition(SeedStage.GERMINATED)
        else:
            raise RuntimeError(f"G0 gate failed: {gate_result.checks_failed}")

        # Register for isolation monitoring (skipped in fast_mode)
        if host_module is not None and self.isolation_monitor is not None:
            from esper.kasmina.isolation import GradientIsolationMonitor
            if self.isolation_monitor is None:
                self.isolation_monitor = GradientIsolationMonitor()
            self.isolation_monitor.register(host_module, self.seed)

        self._emit_telemetry(TelemetryEventType.SEED_GERMINATED)
        return self.state

    def advance_stage(self, target_stage: SeedStage | None = None) -> GateResult:
        """Advance seed to next stage (or specific target stage).

        Returns gate result indicating success/failure.
        """
        if not self.is_active:
            return GateResult(
                gate=GateLevel.G0,
                passed=False,
                checks_failed=["no_active_seed"],
            )

        # Determine target stage
        if target_stage is None:
            valid_next = VALID_TRANSITIONS.get(self.state.stage, ())
            # Pick first non-failure transition
            for candidate in valid_next:
                if not is_failure_stage(candidate):
                    target_stage = candidate
                    break

        if target_stage is None:
            return GateResult(
                gate=GateLevel.G0,
                passed=False,
                checks_failed=["no_valid_transition"],
            )

        # Check gate
        gate_result = self.gates.check_gate(self.state, target_stage)

        if gate_result.passed:
            old_stage = self.state.stage
            if self.state.transition(target_stage):
                self._emit_telemetry(
                    TelemetryEventType.SEED_STAGE_CHANGED,
                    data={"from": old_stage.name, "to": target_stage.name}
                )

                # Handle special stage entry logic
                if target_stage == SeedStage.FOSSILIZED:
                    self._emit_telemetry(TelemetryEventType.SEED_FOSSILIZED)
            else:
                gate_result = GateResult(
                    gate=gate_result.gate,
                    passed=False,
                    checks_failed=["transition_failed"],
                )

        return gate_result

    def cull(self, reason: str = "") -> None:
        """Cull the current seed."""
        if self.state:
            self.state.transition(SeedStage.CULLED)
            self._emit_telemetry(
                TelemetryEventType.SEED_CULLED,
                data={"reason": reason}
            )
        self.seed = None
        self.state = None
        self.alpha_schedule = None
        self.isolate_gradients = False
        if self.isolation_monitor is not None:
            self.isolation_monitor.reset()

    def forward(self, host_features: torch.Tensor) -> torch.Tensor:
        """Process features through this slot."""
        from esper.kasmina.isolation import blend_with_isolation

        if not self.is_active or self.alpha == 0.0:
            return host_features

        if not is_active_stage(self.state.stage):
            return host_features

        # Apply gradient isolation if enabled
        seed_input = host_features.detach() if self.isolate_gradients else host_features
        seed_features = self.seed(seed_input)

        # Blend
        return blend_with_isolation(host_features, seed_features, self.alpha)

    def get_parameters(self):
        """Get trainable parameters of the seed."""
        if self.seed is None:
            return iter([])
        return self.seed.parameters()

    def set_alpha(self, alpha: float) -> None:
        """Set the blending alpha."""
        if self.state:
            self.state.alpha = max(0.0, min(1.0, alpha))
            self.state.metrics.current_alpha = self.state.alpha

    def start_blending(self, total_steps: int, temperature: float = 1.0) -> None:
        """Initialize alpha schedule for blending phase."""
        from esper.kasmina.isolation import AlphaSchedule

        self.alpha_schedule = AlphaSchedule(
            total_steps=total_steps,
            start=0.0,
            end=1.0,
            temperature=temperature,
        )

    def update_alpha_for_step(self, step: int) -> float:
        """Update alpha based on schedule."""
        if self.alpha_schedule is not None:
            alpha = self.alpha_schedule(step)
            self.set_alpha(alpha)
            if self.state:
                self.state.metrics.alpha_ramp_step = step
            return alpha
        return self.alpha

    def get_state_report(self) -> SeedStateReport | None:
        """Get current state as Leyline report."""
        if not self.state:
            return None
        return self.state.to_report()

    def _emit_telemetry(
        self,
        event_type: TelemetryEventType,
        data: dict | None = None,
    ) -> None:
        """Emit a telemetry event.

        Skipped entirely in fast_mode for zero overhead in PPO rollouts.
        """
        if self.fast_mode or self.on_telemetry is None:
            return

        event = TelemetryEvent(
            event_type=event_type,
            seed_id=self.state.seed_id if self.state else None,
            slot_id=self.slot_id,
            data=data or {},
        )
        self.on_telemetry(event)


__all__ = [
    "SeedMetrics",
    "SeedState",
    "QualityGates",
    "SeedSlot",
]
