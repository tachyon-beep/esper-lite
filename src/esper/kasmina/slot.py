"""Kasmina Slot - Seed lifecycle management.

The SeedSlot manages a single seed module through its lifecycle:
germination -> training -> blending -> fossilization/culling.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, TYPE_CHECKING
import pickle

import torch
import torch.nn as nn

from esper.kasmina.isolation import GradientIsolationMonitor, blend_with_isolation

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
    SeedTelemetry,
)

if TYPE_CHECKING:
    from esper.simic.features import TaskConfig

# Canonical feature-shape probes for seed shape validation.
# These are smoke tests: seeds must be shape-preserving for arbitrary H/W or seq_len.
CNN_SHAPE_PROBE_SPATIAL = 8
TRANSFORMER_SHAPE_PROBE_SEQ_LEN = 4


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

    def record_accuracy(self, accuracy: float | torch.Tensor) -> None:
        """Record a new accuracy measurement.

        Accepts either a Python float or a tensor; tensors are
        detached and converted to float to keep metrics and
        checkpoints device-agnostic.
        """
        if isinstance(accuracy, torch.Tensor):
            accuracy = accuracy.detach().item()
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
    stage_entered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    alpha: float = 0.0
    metrics: SeedMetrics = field(default_factory=SeedMetrics)

    # Blending progress tracking
    blending_steps_done: int = 0
    blending_steps_total: int = 0

    # Flags
    is_healthy: bool = True
    is_paused: bool = False

    # History (bounded to prevent unbounded memory growth in long-running training)
    stage_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Telemetry (initialized in __post_init__)
    telemetry: SeedTelemetry = field(default=None)

    def __post_init__(self):
        """Initialize telemetry with seed identity."""
        if self.telemetry is None:
            self.telemetry = SeedTelemetry(
                seed_id=self.seed_id,
                blueprint_id=self.blueprint_id,
            )

    def sync_telemetry(
        self,
        gradient_norm: float,
        gradient_health: float,
        has_vanishing: bool,
        has_exploding: bool,
        epoch: int = 0,
        max_epochs: int = 25,
    ) -> None:
        """Sync telemetry from metrics + gradient signals.

        Call this once per epoch after validation to update telemetry.
        SeedMetrics remains the source of truth for accuracy/epoch data.
        """
        from datetime import timezone

        self.telemetry.accuracy = self.metrics.current_val_accuracy
        self.telemetry.accuracy_delta = self.metrics.improvement_since_stage_start
        self.telemetry.epochs_in_stage = self.metrics.epochs_in_current_stage
        self.telemetry.stage = self.stage.value
        self.telemetry.alpha = self.alpha

        self.telemetry.gradient_norm = gradient_norm
        self.telemetry.gradient_health = gradient_health
        self.telemetry.has_vanishing = has_vanishing
        self.telemetry.has_exploding = has_exploding

        self.telemetry.epoch = epoch
        self.telemetry.max_epochs = max_epochs
        self.telemetry.captured_at = datetime.now(timezone.utc)

    def can_transition_to(self, new_stage: SeedStage) -> bool:
        """Check if transition to new_stage is valid per Leyline contract."""
        return is_valid_transition(self.stage, new_stage)

    def transition(self, new_stage: SeedStage) -> bool:
        """Transition to a new stage. Returns True if successful."""
        if not self.can_transition_to(new_stage):
            return False

        self.previous_stage = self.stage
        self.stage = new_stage
        self.stage_entered_at = datetime.now(timezone.utc)
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

    def _seed_ready_for_blending(self, state: SeedState) -> bool:
        """Seed-specific readiness for BLENDING.

        Returns True when the seed has spent a minimum number of epochs in
        the current stage (TRAINING) so it has seen enough updates before
        we start blending. This complements the global improvement check,
        which looks at host + seed stability, by ensuring this particular
        seed is not effectively untrained.
        """
        # Use epochs_in_current_stage as "training age" in TRAINING.
        # We reuse min_blending_epochs as a conservative lower bound.
        return state.metrics.epochs_in_current_stage >= self.min_blending_epochs

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
        """G2: Blending readiness – global improvement + seed readiness."""
        checks_passed = []
        checks_failed = []

        improvement = state.metrics.improvement_since_stage_start

        # Global performance: host + training loop improving
        if improvement >= self.min_training_improvement:
            checks_passed.append(f"global_improvement_{improvement:.2f}%")
            perf_ok = True
        else:
            checks_failed.append(f"global_improvement_insufficient_{improvement:.2f}%")
            perf_ok = False

        # Global isolation guard
        if state.metrics.isolation_violations <= self.max_isolation_violations:
            checks_passed.append("isolation_ok")
            isolation_ok = True
        else:
            checks_failed.append(f"isolation_violations_{state.metrics.isolation_violations}")
            isolation_ok = False

        # Seed-specific readiness: enough TRAINING epochs to be worth blending
        if self._seed_ready_for_blending(state):
            checks_passed.append("seed_ready")
            seed_ok = True
        else:
            checks_failed.append("seed_not_ready")
            seed_ok = False

        passed = perf_ok and isolation_ok and seed_ok
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

class SeedSlot(nn.Module):
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
        task_config: "TaskConfig | None" = None,
    ):
        super().__init__()
        self.slot_id = slot_id
        self.channels = channels
        self.device = device
        self.gates = gates or QualityGates()
        self.on_telemetry = on_telemetry
        self.fast_mode = fast_mode  # Disable telemetry/isolation for PPO
        self.task_config = task_config

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

    @property
    def active_seed_params(self) -> int:
        """Return trainable params of active seed, or 0 if no seed."""
        if self.seed is None:
            return 0
        return sum(p.numel() for p in self.seed.parameters() if p.requires_grad)

    def germinate(
        self,
        blueprint_id: str,
        seed_id: str | None = None,
        host_module: nn.Module | None = None,
    ) -> SeedState:
        """Germinate a new seed in this slot."""
        from esper.kasmina.blueprints import BlueprintRegistry

        if self.is_active and not is_failure_stage(self.state.stage):
            raise RuntimeError(f"Slot {self.slot_id} already has active seed")

        # Default to "cnn" when no TaskConfig is provided to match legacy CNN tests.
        topology = self.task_config.topology if self.task_config is not None else "cnn"
        if topology not in ("cnn", "transformer"):
            raise AssertionError(f"Unknown topology '{topology}' for SeedSlot.germinate")
        try:
            self.seed = BlueprintRegistry.create(topology, blueprint_id, self.channels)
        except ValueError as exc:
            available = BlueprintRegistry.list_for_topology(topology)
            names = [s.name for s in available]
            raise AssertionError(
                f"Blueprint '{blueprint_id}' not available for topology '{topology}'. Available: {names}"
            ) from exc
        self.seed = self.seed.to(self.device)

        # Validate shape: ensure seed preserves feature shape in a host-agnostic way
        # without mutating host BatchNorm statistics. Smoke test only.
        if self.seed is not None:
            if topology == "cnn":
                # Canonical CNN feature shape: NCHW with known channel count.
                shape_probe = torch.randn(
                    1,
                    self.channels,
                    CNN_SHAPE_PROBE_SPATIAL,
                    CNN_SHAPE_PROBE_SPATIAL,
                    device=self.device,
                )
            else:
                # Canonical transformer feature shape: (batch, seq_len, dim).
                shape_probe = torch.randn(
                    2,
                    TRANSFORMER_SHAPE_PROBE_SEQ_LEN,
                    self.channels,
                    device=self.device,
                )

            expected_shape = shape_probe.shape
            seed_was_training = self.seed.training
            try:
                self.seed.eval()
                with torch.no_grad():
                    seed_out = self.seed(shape_probe)
                if isinstance(seed_out, torch.Tensor) and seed_out.shape != expected_shape:
                    raise AssertionError(
                        f"Seed '{blueprint_id}' changed shape: "
                        f"{seed_out.shape} vs {expected_shape}"
                    )
            finally:
                # Restore original training mode even if assertion fails.
                self.seed.train(seed_was_training)

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
        if host_module is not None and not self.fast_mode:
            if self.isolation_monitor is None:
                self.isolation_monitor = GradientIsolationMonitor()
            self.isolation_monitor.register(host_module, self.seed)

        self._emit_telemetry(
            TelemetryEventType.SEED_GERMINATED,
            data={
                "blueprint_id": blueprint_id,
                "seed_id": seed_id,
                "params": sum(p.numel() for p in self.seed.parameters() if p.requires_grad),
            }
        )
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

            # Capture metrics before transition resets stage counters
            metrics = self.state.metrics
            improvement = metrics.total_improvement
            epochs_total = metrics.epochs_total
            epochs_in_stage = metrics.epochs_in_current_stage
            blueprint_id = self.state.blueprint_id
            seed_id = self.state.seed_id

            if self.state.transition(target_stage):
                # Stage-specific gradient isolation hooks:
                # - GERMINATED → TRAINING: enable Womb isolation so the seed
                #   sees detached host features during its training phase.
                # - TRAINING → BLENDING: disable isolation so blended seeds can
                #   co-adapt with the host trunk.
                if old_stage == SeedStage.GERMINATED and target_stage == SeedStage.TRAINING:
                    self.isolate_gradients = True
                elif old_stage == SeedStage.TRAINING and target_stage == SeedStage.BLENDING:
                    self.isolate_gradients = False

                self._emit_telemetry(
                    TelemetryEventType.SEED_STAGE_CHANGED,
                    data={"from": old_stage.name, "to": target_stage.name}
                )

                # Handle special stage entry logic
                if target_stage == SeedStage.FOSSILIZED:
                    self._emit_telemetry(
                        TelemetryEventType.SEED_FOSSILIZED,
                        data={
                            "blueprint_id": blueprint_id,
                            "seed_id": seed_id,
                            "improvement": improvement,
                            "params_added": sum(
                                p.numel() for p in self.seed.parameters() if p.requires_grad
                            ),
                            "epochs_total": epochs_total,
                            "epochs_in_stage": epochs_in_stage,
                        }
                    )
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
            # Capture metrics before transition clears state
            improvement = self.state.metrics.total_improvement
            epochs_total = self.state.metrics.epochs_total
            epochs_in_stage = self.state.metrics.epochs_in_current_stage
            blueprint_id = self.state.blueprint_id
            seed_id = self.state.seed_id

            old_stage = self.state.stage
            self.state.transition(SeedStage.CULLED)
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data={"from": old_stage.name, "to": SeedStage.CULLED.name},
            )
            self._emit_telemetry(
                TelemetryEventType.SEED_CULLED,
                data={
                    "reason": reason,
                    "blueprint_id": blueprint_id,
                    "seed_id": seed_id,
                    "improvement": improvement,
                    "epochs_total": epochs_total,
                    "epochs_in_stage": epochs_in_stage,
                }
            )
        self.seed = None
        self.state = None
        self.alpha_schedule = None
        self.isolate_gradients = False
        if self.isolation_monitor is not None:
            self.isolation_monitor.reset()

    def forward(self, host_features: torch.Tensor) -> torch.Tensor:
        """Process features through this slot."""
        # blend_with_isolation imported at module level for torch.compile compatibility

        # 1. Early exit if there is no active seed or the lifecycle
        #    stage is inactive (CULLED/EMBARGOED/RESETTING).
        if not self.is_active or not is_active_stage(self.state.stage):
            return host_features

        # 2. Compute seed features. For Womb/Training we must detach the
        #    host input so seed gradients do not flow back into the host.
        seed_input = host_features.detach() if self.isolate_gradients else host_features
        seed_features = self.seed(seed_input)

        # 3. WOMB MODE (TRAINING stage, alpha == 0.0)
        #
        # Straight-Through Estimator:
        #   forward:  host + (seed - seed.detach()) == host
        #   backward: d loss / d seed_params == d loss / d seed_features
        #
        # This lets the seed see the error signal without changing the
        # host activations. With isolate_gradients=True, host gradients
        # are also identical to the no-seed case.
        if self.state.stage == SeedStage.TRAINING and self.alpha == 0.0:
            return host_features + (seed_features - seed_features.detach())

        # 4. BLENDING and later stages: topology-aware host isolation.
        detach_host = True
        topology = self.task_config.topology if self.task_config is not None else None
        if topology == "transformer" and self.state is not None:
            if self.state.stage in (
                SeedStage.BLENDING,
                SeedStage.SHADOWING,
                SeedStage.PROBATIONARY,
                SeedStage.FOSSILIZED,
            ):
                detach_host = False

        return blend_with_isolation(
            host_features,
            seed_features,
            self.alpha,
            detach_host=detach_host,
        )

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

        # Initialize blending progress tracking
        if self.state:
            self.state.blending_steps_total = total_steps
            self.state.blending_steps_done = 0

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

    def step_epoch(self) -> None:
        """Advance lifecycle mechanically once per epoch (Kasmina timekeeper)."""
        if not self.state:
            return

        stage = self.state.stage

        # GERMINATED → TRAINING: immediate advance (no dwell required)
        if stage == SeedStage.GERMINATED:
            gate_result = self.gates.check_gate(self.state, SeedStage.TRAINING)
            if not gate_result.passed:
                return

            old_stage = self.state.stage
            ok = self.state.transition(SeedStage.TRAINING)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition {self.state.stage} → TRAINING"
                )
            # Enable Womb isolation so seed sees detached host features
            self.isolate_gradients = True
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data={"from": old_stage.name, "to": self.state.stage.name},
            )
            return

        # TRAINING → BLENDING when dwell satisfied and gate passes
        if stage == SeedStage.TRAINING:
            dwell_epochs = 1
            if self.task_config:
                dwell_epochs = max(
                    1, int(self.task_config.max_epochs * self.task_config.train_to_blend_fraction)
                )
            if self.state.metrics.epochs_in_current_stage < dwell_epochs:
                return

            gate_result = self.gates.check_gate(self.state, SeedStage.BLENDING)
            if not gate_result.passed:
                return

            old_stage = self.state.stage
            ok = self.state.transition(SeedStage.BLENDING)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition {self.state.stage} → BLENDING"
                )
            # Leaving TRAINING: disable Womb isolation so BLENDING/SHADOWING/
            # FOSSILIZED can drive host trunk updates via the seed branch.
            self.isolate_gradients = False
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data={"from": old_stage.name, "to": self.state.stage.name},
            )
            # Use explicit task_config.blending_steps if provided, otherwise default to 5.
            total_steps = 5
            if self.task_config is not None and hasattr(self.task_config, "blending_steps"):
                configured_steps = self.task_config.blending_steps
                if isinstance(configured_steps, int) and configured_steps > 0:
                    total_steps = configured_steps
            self.start_blending(total_steps=total_steps, temperature=1.0)
            return

        # BLENDING → SHADOWING when alpha ramp completes and gate passes
        if stage == SeedStage.BLENDING:
            self.state.blending_steps_done += 1

            if self.alpha_schedule is not None:
                self.update_alpha_for_step(self.state.blending_steps_done)

            if self.state.blending_steps_done >= self.state.blending_steps_total:
                self.set_alpha(1.0)  # Ensure fully blended
                gate_result = self.gates.check_gate(self.state, SeedStage.SHADOWING)
                if not gate_result.passed:
                    return
                old_stage = self.state.stage
                ok = self.state.transition(SeedStage.SHADOWING)
                if not ok:
                    raise RuntimeError(
                        f"Illegal lifecycle transition {self.state.stage} → SHADOWING"
                    )
                self._emit_telemetry(
                    TelemetryEventType.SEED_STAGE_CHANGED,
                    data={"from": old_stage.name, "to": self.state.stage.name},
                )
            return

        # SHADOWING → PROBATIONARY after dwell and gate
        if stage == SeedStage.SHADOWING:
            dwell_epochs = 1
            if self.task_config:
                dwell_epochs = max(
                    1, int(self.task_config.max_epochs * self.task_config.shadowing_fraction)
                )
            if self.state.metrics.epochs_in_current_stage < dwell_epochs:
                return

            gate_result = self.gates.check_gate(self.state, SeedStage.PROBATIONARY)
            if not gate_result.passed:
                return
            old_stage = self.state.stage
            ok = self.state.transition(SeedStage.PROBATIONARY)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition {self.state.stage} → PROBATIONARY"
                )
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data={"from": old_stage.name, "to": self.state.stage.name},
            )
            # Do not immediately collapse; dwell handled above on next epochs

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

    def get_extra_state(self):
        """Persist SeedState and alpha schedule in checkpoints."""
        return {
            "seed_state": self.state,
            "alpha_schedule": self.alpha_schedule,
        }

    def set_extra_state(self, state: dict) -> None:
        """Restore SeedState and alpha schedule from checkpoints."""
        self.state = state.get("seed_state")
        self.alpha_schedule = state.get("alpha_schedule")


__all__ = [
    "SeedMetrics",
    "SeedState",
    "QualityGates",
    "SeedSlot",
]
