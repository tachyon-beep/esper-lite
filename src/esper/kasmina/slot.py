"""Kasmina Slot - Seed lifecycle management.

The SeedSlot manages a single seed module through its lifecycle:
germination -> training -> blending -> fossilization/culling.

torch.compile Strategy
----------------------
SeedSlot.forward() is compile-compatible. Dynamo creates ~6-8 specialized
graphs (one per stage/config combination). This is acceptable because:

1. Stage transitions happen once per epoch, not per forward pass
2. After warmup, no recompilation occurs within an epoch
3. Allowing compilation enables fusion with surrounding host network ops
4. The FOSSILIZED steady-state (dominant runtime) benefits most from compilation

The underlying tensor operations (ste_forward, blend_with_isolation) in
isolation.py are pure tensor ops that compile efficiently.

Use TORCH_LOGS=guards to monitor graph specialization during training.
"""

from __future__ import annotations

import os
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, TYPE_CHECKING

# Debug flag for STE gradient assertions (set ESPER_DEBUG_STE=1 to enable)
_DEBUG_STE = os.environ.get("ESPER_DEBUG_STE", "").lower() in ("1", "true", "yes")

# Gradient telemetry constants
# Minimum parameter-normalized gradient ratio for G2 gate (seed must be "awake")
DEFAULT_GRADIENT_RATIO_THRESHOLD: float = 0.05
# Epsilon for numerical stability in gradient ratio computation
GRADIENT_EPSILON: float = 1e-8
# EMA decay factor for gradient norm averaging (higher = slower adaptation)
GRADIENT_EMA_DECAY: float = 0.9
# Maximum gradient ratio to prevent outliers from skewing G2 gate decisions.
# Value of 10.0 corresponds to "seed has 100x higher per-parameter gradient intensity
# than host" after sqrt normalization - an extreme value indicating either a very
# small seed or numerical anomaly.
MAX_GRADIENT_RATIO: float = 10.0

import torch
import torch.nn as nn

from esper.kasmina.isolation import GradientIsolationMonitor, blend_with_isolation, ste_forward

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
# 32x32 chosen to match post-stride-32 dimensions (ResNet layer4) and CIFAR input size,
# while avoiding excessive memory for high-channel probes during torch.compile tracing.
CNN_SHAPE_PROBE_SPATIAL = 32
TRANSFORMER_SHAPE_PROBE_SEQ_LEN = 8  # Increased for windowed attention edge cases


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
    accuracy_at_blending_start: float = 0.0  # Snapshot at TRAINING→BLENDING

    isolation_violations: int = 0
    gradient_norm_avg: float = 0.0

    current_alpha: float = 0.0
    alpha_ramp_step: int = 0

    # Counterfactual contribution (set by vectorized training when available)
    # This is the TRUE causal attribution: real_acc - baseline_acc(alpha=0)
    counterfactual_contribution: float | None = None

    # Gradient-based seed activity metric (parameter-normalized)
    # Formula: (seed_norm / host_norm) * sqrt(host_params / seed_params)
    # This measures per-parameter gradient intensity, scale-invariant across architectures
    seed_gradient_norm_ratio: float = 0.0

    # Parameter counts for normalization (set once at germination)
    host_param_count: int = 0
    seed_param_count: int = 0

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

    @property
    def blending_delta(self) -> float:
        """Accuracy change since blending started (includes host drift).

        This is NOT causal attribution - it measures the total accuracy change
        during BLENDING stages, which conflates host training gains with seed
        impact. For true causal attribution, use counterfactual validation
        (real_acc - baseline_acc with alpha=0).

        Returns 0 if seed never reached BLENDING.
        """
        if self.accuracy_at_blending_start == 0.0:
            return 0.0
        return self.current_val_accuracy - self.accuracy_at_blending_start

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

@dataclass(kw_only=True)
class SeedState:
    """Complete state of a seed through its lifecycle."""

    seed_id: str
    blueprint_id: str
    slot_id: str = ""

    stage: SeedStage = SeedStage.DORMANT
    previous_stage: SeedStage = SeedStage.UNKNOWN
    previous_epochs_in_stage: int = 0  # Epochs in previous stage at transition (for PBRS)
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
    telemetry: SeedTelemetry | None = field(default=None)

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
        self.previous_epochs_in_stage = self.metrics.epochs_in_current_stage  # For PBRS telescoping
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
            previous_epochs_in_stage=self.previous_epochs_in_stage,
            stage_entered_at=self.stage_entered_at,
            metrics=self.metrics.to_leyline(),
            is_healthy=self.is_healthy,
            is_improving=self.metrics.improvement_since_stage_start > 0,
            needs_attention=not self.is_healthy or self.metrics.isolation_violations > 0,
        )


# =============================================================================
# Quality Gates
# =============================================================================

# Minimum counterfactual contribution required for fossilization.
# Prevents "free rider" seeds that provide negligible value from becoming permanent.
# DRL rationale: seeds must provide economically significant contribution to justify
# their parameter cost. A 1% threshold ensures measurable causal impact.
MIN_FOSSILIZE_CONTRIBUTION = 1.0  # 1% minimum causal contribution


class QualityGates:
    """Quality gate checks for stage transitions.

    Each gate validates that a seed is ready for the next stage.
    """

    def __init__(
        self,
        min_training_improvement: float = 0.5,
        min_blending_epochs: int = 3,
        max_isolation_violations: int = 10,
        min_probation_stability: float = 0.95,
        min_seed_gradient_ratio: float = DEFAULT_GRADIENT_RATIO_THRESHOLD,
    ):
        self.min_training_improvement = min_training_improvement
        self.min_blending_epochs = min_blending_epochs
        self.max_isolation_violations = max_isolation_violations
        self.min_probation_stability = min_probation_stability
        self.min_seed_gradient_ratio = min_seed_gradient_ratio

    def check_gate(self, state: SeedState, target_stage: SeedStage) -> GateResult:
        """Check if seed passes the gate for target stage."""

        match self._get_gate_level(target_stage):
            case GateLevel.G0:
                return self._check_g0(state)
            case GateLevel.G1:
                return self._check_g1(state)
            case GateLevel.G2:
                return self._check_g2(state)
            case GateLevel.G3:
                return self._check_g3(state)
            case GateLevel.G5:
                return self._check_g5(state)
            case gate:
                # Default: pass
                return GateResult(gate=gate, passed=True, score=1.0)

    def _get_gate_level(self, target_stage: SeedStage) -> GateLevel:
        """Map target stage to gate level."""
        mapping = {
            SeedStage.GERMINATED: GateLevel.G0,
            SeedStage.TRAINING: GateLevel.G1,
            SeedStage.BLENDING: GateLevel.G2,
            SeedStage.PROBATIONARY: GateLevel.G3,  # Was G4, now G3 (direct from BLENDING)
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
        """G2: Blending readiness – global improvement + seed readiness + gradient activity."""
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

        # Gradient-based seed activity: detect if seed is actively learning
        # vs. riding host improvements. Ratio is parameter-normalized EMA.
        if state.metrics.seed_gradient_norm_ratio >= self.min_seed_gradient_ratio:
            checks_passed.append(f"seed_gradient_active_{state.metrics.seed_gradient_norm_ratio:.2f}")
            gradient_ok = True
        else:
            checks_failed.append(f"seed_gradient_low_{state.metrics.seed_gradient_norm_ratio:.2f}")
            gradient_ok = False

        passed = perf_ok and isolation_ok and seed_ok and gradient_ok
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
        """G3: Probation readiness - blending completed with stable integration."""
        checks_passed = []
        checks_failed = []

        # Check blending duration
        if state.metrics.epochs_in_current_stage >= self.min_blending_epochs:
            checks_passed.append("blending_complete")
        else:
            checks_failed.append(f"blending_incomplete_{state.metrics.epochs_in_current_stage}")

        # Check alpha reached target
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

    def _check_g5(self, state: SeedState) -> GateResult:
        """G5: Fossilization readiness - requires counterfactual validation.

        G5 is only reachable from PROBATIONARY stage where counterfactual
        validation is mandatory. No fallback to total_improvement.
        """
        checks_passed = []
        checks_failed = []

        # REQUIRE counterfactual - no fallback
        contribution = state.metrics.counterfactual_contribution
        if contribution is None:
            return GateResult(
                gate=GateLevel.G5,
                passed=False,
                score=0.0,
                checks_passed=[],
                checks_failed=["counterfactual_not_available"],
            )

        # Check contribution meets minimum threshold
        # Prevents zero/negligible contribution seeds from fossilizing
        if contribution >= MIN_FOSSILIZE_CONTRIBUTION:
            checks_passed.append(f"sufficient_contribution_{contribution:.2f}%")
        else:
            checks_failed.append(
                f"insufficient_contribution_{contribution:.2f}%_below_{MIN_FOSSILIZE_CONTRIBUTION}%"
            )

        # Check health
        if state.is_healthy:
            checks_passed.append("healthy")
        else:
            checks_failed.append("unhealthy")

        passed = len(checks_failed) == 0
        return GateResult(
            gate=GateLevel.G5,
            passed=passed,
            score=min(1.0, contribution / 10.0) if contribution > 0 else 0.0,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )


# =============================================================================
# Seed Slot
# =============================================================================

class SeedSlot(nn.Module):
    """A slot in the model where a seed can be attached.

    Manages the full lifecycle of a seed with quality gates.

    Gradient Isolation Strategy:
        There are two gradient paths through a slot during BLENDING+:

        1. DIRECT PATH (host ← loss):
           Host receives (1-α) weighted gradients from task loss.
           Always active - enables host backbone to continue learning.

        2. SEED PATH (host ← seed ← loss):
           Controlled by `isolate_gradients` flag at seed INPUT.
           - True: seed gradients don't backprop into host params
           - False: host co-adapts to support seed representations

        Topology-aware defaults at TRAINING → BLENDING transition:
        - CNN: isolate_gradients=True (stable spatial hierarchies)
        - Transformer: isolate_gradients=False (co-adaptation benefits)

        Diagram:
            Loss
              │
              ▼
            Output = lerp(host, seed, α)
              │                    │
              │ (1-α) gradient     │ α gradient
              ▼                    ▼
            Host Features         Seed Features
              ▲                    │
              │ (blocked if        │
              │  isolate_gradients)│
              └────────────────────┘

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
        self.device = torch.device(device) if isinstance(device, str) else device
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

        # Cached shape probes to avoid per-germinate allocation
        # Keys: "cnn" or "transformer", values: (device, tensor)
        self._shape_probe_cache: dict[str, tuple[torch.device, torch.Tensor]] = {}

    def _get_shape_probe(self, topology: str) -> torch.Tensor:
        """Get cached shape probe for topology, creating if needed."""
        cached = self._shape_probe_cache.get(topology)

        if cached is not None:
            cached_device, cached_tensor = cached
            # Use direct device comparison instead of string
            if cached_device == self.device:
                return cached_tensor

        # Create new probe for this topology/device
        if topology == "cnn":
            probe = torch.randn(
                1,
                self.channels,
                CNN_SHAPE_PROBE_SPATIAL,
                CNN_SHAPE_PROBE_SPATIAL,
                device=self.device,
            )
        else:
            probe = torch.randn(
                2,
                TRANSFORMER_SHAPE_PROBE_SEQ_LEN,
                self.channels,
                device=self.device,
            )

        # Store device as torch.device, not string
        self._shape_probe_cache[topology] = (self.device, probe)
        return probe

    def to(self, *args, **kwargs) -> "SeedSlot":
        """Transfer slot and any active seed to device."""
        old_device = self.device  # Track before move
        super().to(*args, **kwargs)

        # Update device tracking (query from parameters after move)
        try:
            actual_device = next(self.parameters()).device
            self.device = actual_device
        except StopIteration:
            # Infer from args if no parameters
            for arg in args:
                if isinstance(arg, (str, torch.device)):
                    self.device = torch.device(arg) if isinstance(arg, str) else arg
                    break

        # Only clear cache if device actually changed
        if self.device != old_device:
            self._shape_probe_cache.clear()

        return self

    @property
    def is_active(self) -> bool:
        """Check if slot has an active seed."""
        return self.seed is not None and self.state is not None

    @property
    def alpha(self) -> float:
        """Current blending alpha."""
        return self.state.alpha if self.state else 0.0

    @contextmanager
    def force_alpha(self, value: float):
        """Temporarily override alpha for counterfactual evaluation.

        Used for differential validation to measure true seed contribution
        by comparing real output (current alpha) vs host-only (alpha=0).

        Warning:
            NOT THREAD-SAFE. Do not use during concurrent forward passes
            or with DataParallel/DistributedDataParallel. Use model.eval()
            and single-threaded validation only.

            Nested calls are NOT supported - the inner override will be
            clobbered when the outer context exits.

        Args:
            value: Alpha value to force (typically 0.0 for host-only baseline)

        Yields:
            Context where alpha is temporarily overridden
        """
        if self.state is None:
            # No active seed - alpha is effectively 0, nothing to override
            yield
            return

        # Store previous alpha and override
        prev_alpha = self.state.alpha
        self.state.alpha = value
        try:
            yield
        finally:
            self.state.alpha = prev_alpha

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
        shape_probe = self._get_shape_probe(topology)
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

        # Capture param counts once for gradient normalization (G2 gate)
        # This enables scale-invariant comparison across different host/seed sizes
        self.state.metrics.seed_param_count = sum(
            p.numel() for p in self.seed.parameters() if p.requires_grad
        )
        if host_module is not None:
            self.state.metrics.host_param_count = sum(
                p.numel() for p in host_module.parameters() if p.requires_grad
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
            blending_delta = metrics.blending_delta
            counterfactual = metrics.counterfactual_contribution
            epochs_total = metrics.epochs_total
            epochs_in_stage = metrics.epochs_in_current_stage
            blueprint_id = self.state.blueprint_id
            seed_id = self.state.seed_id

            if self.state.transition(target_stage):
                # Stage-specific gradient isolation hooks:
                # - GERMINATED → TRAINING: enable Incubator isolation so the seed
                #   sees detached host features during its training phase.
                # - TRAINING → BLENDING: topology-aware isolation decision
                #   (CNNs keep isolation, Transformers allow co-adaptation)
                if old_stage == SeedStage.GERMINATED and target_stage == SeedStage.TRAINING:
                    self.isolate_gradients = True
                elif old_stage == SeedStage.TRAINING and target_stage == SeedStage.BLENDING:
                    # Topology-aware gradient isolation:
                    # - CNNs: keep isolation (host learns from loss, not seed feedback)
                    #   Rationale: CNNs have rigid spatial hierarchies where co-adaptation
                    #   risks destabilizing learned features.
                    # - Transformers: allow co-adaptation (host receives seed gradients)
                    #   Rationale: Transformers benefit from host adjusting to seed
                    #   representations during blending.
                    topology = self.task_config.topology if self.task_config else "cnn"
                    self.isolate_gradients = (topology == "cnn")
                    # Snapshot accuracy at blending start for true causal attribution
                    # This is when the seed starts actually affecting network output
                    self.state.metrics.accuracy_at_blending_start = self.state.metrics.current_val_accuracy

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
                            "blending_delta": blending_delta,
                            "counterfactual": counterfactual,  # True causal attribution
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

    def cull(self, reason: str = "") -> bool:
        """Cull the current seed.

        FOSSILIZED seeds cannot be culled - they are permanent by design.
        The PROBATIONARY stage exists as the last decision point before
        permanent integration. A future pruning subsystem will handle
        removal of non-performant fossilized nodes.

        Returns:
            True if cull succeeded, False if seed is uncullable (FOSSILIZED)
        """
        if not self.state:
            return False

        # FOSSILIZED seeds are permanent - cannot be culled
        if self.state.stage == SeedStage.FOSSILIZED:
            return False

        # Capture metrics before transition clears state
        improvement = self.state.metrics.total_improvement
        blending_delta = self.state.metrics.blending_delta
        counterfactual = self.state.metrics.counterfactual_contribution
        epochs_total = self.state.metrics.epochs_total
        epochs_in_stage = self.state.metrics.epochs_in_current_stage
        blueprint_id = self.state.blueprint_id
        seed_id = self.state.seed_id

        old_stage = self.state.stage
        if not self.state.transition(SeedStage.CULLED):
            # Transition failed (shouldn't happen for non-FOSSILIZED)
            return False

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
                "blending_delta": blending_delta,
                "counterfactual": counterfactual,  # True causal attribution
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
        return True

    def capture_gradient_telemetry(self) -> None:
        """Calculate gradient norms via the isolation monitor and update internal metrics.

        CRITICAL: Call this from Tolaria after loss.backward() to enable the G2 gate.
        Without this, seed_gradient_norm_ratio remains 0.0 and G2 always fails.

        Performance note: This triggers a device-to-host sync (GPU pipeline stall).
        Use a stride in the training loop (e.g., every 10 steps) to minimize overhead.
        The EMA smoothing makes sparse sampling acceptable for gate decisions.
        """
        # Fast exit if no monitor (fast_mode) or no active seed
        if self.isolation_monitor is None or not self.is_active:
            return

        if not self.state:
            return

        # Ask monitor to calculate gradient norms
        _, stats = self.isolation_monitor.check_isolation()

        host_norm = stats.get("host_grad_norm", 0.0)
        seed_norm = stats.get("seed_grad_norm", 0.0)

        # Update isolation violation count if host received unexpected gradients
        if self.isolate_gradients and host_norm > 1e-6:
            self.state.metrics.isolation_violations += 1

        # Compute parameter-normalized seed gradient ratio
        # Formula: (seed_norm / host_norm) * sqrt(host_params / seed_params)
        # This measures per-parameter gradient intensity, scale-invariant across architectures
        #
        # Edge case handling (DRL Expert review 2025-12-09):
        # - When host_norm < epsilon: host has no gradients, ratio is meaningless (set to 0)
        # - Upper bound clamp (10.0): prevents astronomically large ratios from dominating metrics
        if host_norm < GRADIENT_EPSILON:
            # Host has no gradients - cannot compute meaningful ratio
            # This can happen with frozen host, gradient accumulation before backward, etc.
            raw_ratio = 0.0
        else:
            raw_ratio = seed_norm / host_norm

        # Apply parameter normalization if counts are available
        host_params = self.state.metrics.host_param_count
        seed_params = self.state.metrics.seed_param_count
        if host_params > 0 and seed_params > 0:
            # sqrt(host/seed) scales up small seeds, scales down large seeds
            normalization_factor = (host_params / seed_params) ** 0.5
            ratio = raw_ratio * normalization_factor
        else:
            # Fallback to raw ratio if param counts unavailable (e.g., fast_mode)
            ratio = raw_ratio

        # Clamp to reasonable range to prevent outliers from skewing G2 gate decisions
        self.state.metrics.seed_gradient_norm_ratio = min(MAX_GRADIENT_RATIO, ratio)

        # Update EMA of gradient norm for monitoring
        current_avg = self.state.metrics.gradient_norm_avg
        self.state.metrics.gradient_norm_avg = (
            GRADIENT_EMA_DECAY * current_avg + (1 - GRADIENT_EMA_DECAY) * seed_norm
        )

    def forward(self, host_features: torch.Tensor) -> torch.Tensor:
        """Process features through this slot.

        torch.compile behavior:
        Dynamo creates specialized graphs for each stage/config combination
        (~6-8 graphs total). Guard failures occur at stage transitions (once
        per epoch), triggering recompilation. After warmup, execution stays
        within a single specialized graph per epoch with no recompilation.

        DO NOT use @torch.compiler.disable here - it completely opts out of
        compilation, which is worse than allowing graph specialization. The
        stage-dependent control flow causes specialization overhead during
        warmup, but steady-state performance benefits from end-to-end fusion.
        """
        # blend_with_isolation imported at module level for torch.compile compatibility

        # 1. Early exit if there is no active seed or the lifecycle
        #    stage is inactive (CULLED/EMBARGOED/RESETTING).
        if not self.is_active or not is_active_stage(self.state.stage):
            return host_features

        # 2. Compute seed features. For Incubator/Training we must detach the
        #    host input so seed gradients do not flow back into the host.
        seed_input = host_features.detach() if self.isolate_gradients else host_features
        seed_features = self.seed(seed_input)

        # 3. INCUBATOR MODE (TRAINING stage, alpha == 0.0)
        #
        # Straight-Through Estimator:
        #   forward:  host + (seed - seed.detach()) == host
        #   backward: d loss / d seed_params == d loss / d seed_features
        #
        # This lets the seed see the error signal without changing the
        # host activations. With isolate_gradients=True, host gradients
        # are also identical to the no-seed case.
        if self.state.stage == SeedStage.TRAINING and self.alpha == 0.0:
            if _DEBUG_STE:
                assert seed_features.requires_grad, (
                    "STE requires seed_features to have requires_grad=True for gradient flow"
                )
            return ste_forward(host_features, seed_features)

        # 4. BLENDING and later stages: standard lerp with proper gradient flow.
        #
        # Gradient isolation strategy (updated 2025-12-10):
        # - DIRECT PATH (host ← loss): Host receives (1-α) weighted gradients.
        #   Always active - enables host backbone to continue learning.
        # - SEED PATH (host ← seed ← loss): Controlled by isolate_gradients
        #   at the seed INPUT (line 1024), not here in the blend output.
        #   CNNs: blocked (isolate_gradients=True) to prevent co-adaptation
        #   Transformers: allowed (isolate_gradients=False) for co-adaptation
        #
        # Impact on credit assignment: Transformer improvements during BLENDING+
        # may come from seed OR host adaptation. Use counterfactual_contribution
        # (not blending_delta) for causal attribution.
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

    def _sync_gate_decision(self, gate_result: GateResult) -> None:
        """Ensure all DDP ranks agree on lifecycle transitions (Unanimous Consensus).

        We use ReduceOp.MIN (Logical AND) for all transitions.
        - Advancement: Everyone must be ready.
        - Culling: If one rank votes to keep, we wait (conservative).

        This prevents Architecture Divergence, where ranks typically crash
        if parameter shapes mismatch during the next forward pass.

        COLLECTIVE OPERATION: All ranks MUST call this in identical order
        for each gate check, or deadlock will occur.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return

        # 1 = Passed, 0 = Failed; MIN gives unanimous consensus
        decision = torch.tensor([int(gate_result.passed)], device=self.device, dtype=torch.int)
        torch.distributed.all_reduce(decision, op=torch.distributed.ReduceOp.MIN)

        is_passed = bool(decision.item())

        if gate_result.passed and not is_passed:
            gate_result.message += " (Vetoed by DDP consensus)"
            gate_result.checks_failed.append("ddp_veto")

        gate_result.passed = is_passed

    def step_epoch(self) -> None:
        """Advance lifecycle mechanically once per epoch (Kasmina timekeeper)."""
        if not self.state:
            return

        stage = self.state.stage

        # GERMINATED → TRAINING: immediate advance (no dwell required)
        if stage == SeedStage.GERMINATED:
            gate_result = self.gates.check_gate(self.state, SeedStage.TRAINING)
            self._sync_gate_decision(gate_result)
            if not gate_result.passed:
                return

            old_stage = self.state.stage
            ok = self.state.transition(SeedStage.TRAINING)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition {self.state.stage} → TRAINING"
                )
            # Enable Incubator isolation so seed sees detached host features
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
            self._sync_gate_decision(gate_result)
            if not gate_result.passed:
                return

            old_stage = self.state.stage
            ok = self.state.transition(SeedStage.BLENDING)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition {self.state.stage} → BLENDING"
                )
            # Topology-aware gradient isolation at TRAINING → BLENDING:
            # - CNNs: keep isolation (host learns from loss, not seed feedback)
            # - Transformers: allow co-adaptation (host receives seed gradients)
            topology = self.task_config.topology if self.task_config else "cnn"
            self.isolate_gradients = (topology == "cnn")
            # Snapshot accuracy at blending start for true causal attribution
            # This is when the seed starts actually affecting network output
            self.state.metrics.accuracy_at_blending_start = self.state.metrics.current_val_accuracy
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data={"from": old_stage.name, "to": self.state.stage.name},
            )
            # Use explicit task_config.blending_steps if provided, otherwise default to 5.
            total_steps = 5
            if self.task_config is not None:
                configured_steps = self.task_config.blending_steps
                if isinstance(configured_steps, int) and configured_steps > 0:
                    total_steps = configured_steps
            self.start_blending(total_steps=total_steps, temperature=1.0)
            return

        # BLENDING → PROBATIONARY when alpha ramp completes and gate passes
        if stage == SeedStage.BLENDING:
            self.state.blending_steps_done += 1

            if self.alpha_schedule is not None:
                self.update_alpha_for_step(self.state.blending_steps_done)

            if self.state.blending_steps_done >= self.state.blending_steps_total:
                self.set_alpha(1.0)  # Ensure fully blended
                gate_result = self.gates.check_gate(self.state, SeedStage.PROBATIONARY)
                self._sync_gate_decision(gate_result)
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
            return

        # PROBATIONARY → FOSSILIZED or CULLED
        # This is the final decision point: prove worth or be removed
        if stage == SeedStage.PROBATIONARY:
            # Calculate probation timeout (default 5 epochs, or 10% of max_epochs)
            # Minimum of 5 epochs ensures sufficient time for counterfactual validation
            # (DRL Expert review 2025-12-09: 3 epochs was insufficient for reliable metrics)
            max_probation_epochs = 5
            if self.task_config:
                max_probation_epochs = max(5, int(self.task_config.max_epochs * 0.1))

            # Path 1: Check if we should FOSSILIZE (success case)
            # G5 requires counterfactual validation - no shortcuts
            if self.state.metrics.counterfactual_contribution is not None:
                gate_result = self.gates.check_gate(self.state, SeedStage.FOSSILIZED)
                self._sync_gate_decision(gate_result)
                if gate_result.passed:
                    old_stage = self.state.stage
                    ok = self.state.transition(SeedStage.FOSSILIZED)
                    if not ok:
                        raise RuntimeError(
                            f"Illegal lifecycle transition {self.state.stage} → FOSSILIZED"
                        )
                    self._emit_telemetry(
                        TelemetryEventType.SEED_STAGE_CHANGED,
                        data={"from": old_stage.name, "to": self.state.stage.name},
                    )
                    self._emit_telemetry(
                        TelemetryEventType.SEED_FOSSILIZED,
                        data={
                            "seed_id": self.state.seed_id,
                            "blueprint_id": self.state.blueprint_id,
                            "contribution": self.state.metrics.counterfactual_contribution,
                            "epochs_total": self.state.metrics.epochs_total,
                        },
                    )
                    return
                elif self.state.metrics.counterfactual_contribution <= 0:
                    # Negative contribution = seed hurts performance, cull immediately
                    self.cull(reason="negative_counterfactual")
                    return

            # Path 2: Check for timeout (failure case)
            # Seed failed to prove its worth within the probation window
            if self.state.metrics.epochs_in_current_stage >= max_probation_epochs:
                self.cull(reason="probation_timeout")
                return

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
        """Persist SeedState, alpha schedule, and gradient isolation in checkpoints."""
        return {
            "seed_state": self.state,
            "alpha_schedule": self.alpha_schedule,
            "isolate_gradients": self.isolate_gradients,
        }

    def set_extra_state(self, state: dict) -> None:
        """Restore SeedState, alpha schedule, and gradient isolation from checkpoints."""
        self.state = state.get("seed_state")
        self.alpha_schedule = state.get("alpha_schedule")
        # Default to True (safe/isolated) if not present in old checkpoints
        self.isolate_gradients = state.get("isolate_gradients", True)


__all__ = [
    "SeedMetrics",
    "SeedState",
    "QualityGates",
    "SeedSlot",
]
