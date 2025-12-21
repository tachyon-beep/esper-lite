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
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, ClassVar, TYPE_CHECKING

import torch
import torch.nn as nn

from esper.kasmina.isolation import GradientHealthMonitor, blend_with_isolation, ste_forward
from esper.kasmina.alpha_controller import AlphaController
from esper.leyline.alpha import AlphaCurve, AlphaMode

from esper.leyline import (
    # Lifecycle
    SeedStage,
    VALID_TRANSITIONS,
    is_valid_transition,
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
    # Gate Thresholds (from leyline - single source of truth)
    DEFAULT_MIN_FOSSILIZE_CONTRIBUTION,
    DEFAULT_GRADIENT_RATIO_THRESHOLD,
    DEFAULT_MIN_PROBATION_STABILITY,
    DEFAULT_GRADIENT_EMA_DECAY,
    # Blending default
    DEFAULT_BLENDING_TOTAL_STEPS,
    # QualityGates thresholds
    DEFAULT_MIN_TRAINING_IMPROVEMENT,
    DEFAULT_MIN_BLENDING_EPOCHS,
    DEFAULT_ALPHA_COMPLETE_THRESHOLD,
    DEFAULT_MAX_PROBATION_EPOCHS,
)

if TYPE_CHECKING:
    from esper.simic.features import TaskConfig

# Debug flag for STE gradient assertions (set ESPER_DEBUG_STE=1 to enable)
_DEBUG_STE = os.environ.get("ESPER_DEBUG_STE", "").lower() in ("1", "true", "yes")

# Gradient telemetry constants (hyperparameters imported from leyline, internals stay local)
# Epsilon for numerical stability in gradient ratio computation
GRADIENT_EPSILON: float = 1e-8
# Maximum gradient ratio to prevent outliers from skewing G2 gate decisions.
# Value of 10.0 corresponds to "seed has 100x higher per-parameter gradient intensity
# than host" after sqrt normalization - an extreme value indicating either a very
# small seed or numerical anomaly.
MAX_GRADIENT_RATIO: float = 10.0

# Probation stage constants (internal implementation details)
PROBATION_HISTORY_MAXLEN: int = 100  # Rolling window for stage history
# MAX_PROBATION_EPOCHS now imported from leyline as DEFAULT_MAX_PROBATION_EPOCHS

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

    gradient_norm_avg: float = 0.0

    # Note on alpha semantics (DRL Expert review 2025-12-17):
    # current_alpha represents "blending progress" (step/total_steps), not actual
    # blend values. For GatedBlend, actual per-sample alpha is learned and input-
    # dependent. The agent controls blending TIMELINE, not per-sample gates.
    # This is intentional for credit assignment - observations should reflect
    # controllable state, not emergent gate behavior.
    current_alpha: float = 0.0
    alpha_ramp_step: int = 0

    # Counterfactual contribution (set by vectorized training when available)
    # This is the TRUE causal attribution: real_acc - baseline_acc(alpha=0)
    counterfactual_contribution: float | None = None

    # Flag to distinguish "never reached blending" from "started blending at 0% accuracy"
    _blending_started: bool = False

    # Gradient-based seed activity metric (parameter-normalized)
    # Formula: (seed_norm / host_norm) * sqrt(host_params / seed_params)
    # This measures per-parameter gradient intensity, scale-invariant across architectures
    seed_gradient_norm_ratio: float = 0.0

    # Parameter counts for normalization (set once at germination)
    host_param_count: int = 0
    seed_param_count: int = 0

    # Auto-cull tracking for degenerate policy detection
    # (DRL Expert review 2025-12-17: policies could learn to rely on environment
    # cleanup rather than proactive culling, creating reward hacking via WAIT spam)
    auto_culled: bool = False
    auto_cull_reason: str = ""

    # Known auto-cull reasons (for distinguishing explicit vs auto culls)
    AUTO_CULL_REASONS: ClassVar[frozenset[str]] = frozenset({
        "negative_counterfactual",  # Safety auto-cull: seed hurts performance
        "probation_timeout",        # Timeout auto-cull: no decision made in time
    })

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
    def total_improvement_normalized(self) -> float:
        """Normalized total improvement in [-1, 1] range.

        Raw improvement is in percentage points (0-100 scale), which creates
        scale mismatch with other observation features like alpha (0-1).
        This normalization clamps to ±10 percentage points and scales to [-1, 1]
        for stable PPO value function learning.
        """
        raw = self.total_improvement
        clamped = max(-10.0, min(10.0, raw))
        return clamped / 10.0

    @property
    def improvement_since_stage_start_normalized(self) -> float:
        """Normalized stage improvement in [-1, 1] range.

        See total_improvement_normalized for rationale.
        """
        raw = self.improvement_since_stage_start
        clamped = max(-10.0, min(10.0, raw))
        return clamped / 10.0

    @property
    def blending_delta(self) -> float:
        """Accuracy change since blending started (includes host drift).

        WARNING: This metric is for TELEMETRY/LOGGING only, NOT for RL signals.

        This is NOT causal attribution - it measures the total accuracy change
        during BLENDING stages, which conflates host training gains with seed
        impact. For true causal attribution, use counterfactual_contribution
        (real_acc - baseline_acc with alpha=0).

        DO NOT use this for:
        - Reward shaping (use counterfactual_contribution instead)
        - Observation features (use counterfactual_contribution instead)
        - Gate decisions (G5 uses counterfactual_contribution)

        Returns 0 if seed never reached BLENDING.
        """
        if not self._blending_started:
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
            counterfactual_contribution=self.counterfactual_contribution,
            seed_gradient_norm_ratio=self.seed_gradient_norm_ratio,
            seed_param_count=self.seed_param_count,
            host_param_count=self.host_param_count,
            gradient_norm_avg=self.gradient_norm_avg,
            current_alpha=self.current_alpha,
            alpha_ramp_step=self.alpha_ramp_step,
        )

    def to_dict(self) -> dict:
        """Convert to primitive dict for serialization."""
        return {
            "epochs_total": self.epochs_total,
            "epochs_in_current_stage": self.epochs_in_current_stage,
            "initial_val_accuracy": self.initial_val_accuracy,
            "current_val_accuracy": self.current_val_accuracy,
            "best_val_accuracy": self.best_val_accuracy,
            "accuracy_at_stage_start": self.accuracy_at_stage_start,
            "accuracy_at_blending_start": self.accuracy_at_blending_start,
            "gradient_norm_avg": self.gradient_norm_avg,
            "current_alpha": self.current_alpha,
            "alpha_ramp_step": self.alpha_ramp_step,
            "counterfactual_contribution": self.counterfactual_contribution,
            "_blending_started": self._blending_started,
            "seed_gradient_norm_ratio": self.seed_gradient_norm_ratio,
            "host_param_count": self.host_param_count,
            "seed_param_count": self.seed_param_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeedMetrics":
        """Reconstruct from primitive dict."""
        metrics = cls()
        metrics.epochs_total = data.get("epochs_total", 0)
        metrics.epochs_in_current_stage = data.get("epochs_in_current_stage", 0)
        metrics.initial_val_accuracy = data.get("initial_val_accuracy", 0.0)
        metrics.current_val_accuracy = data.get("current_val_accuracy", 0.0)
        metrics.best_val_accuracy = data.get("best_val_accuracy", 0.0)
        metrics.accuracy_at_stage_start = data.get("accuracy_at_stage_start", 0.0)
        metrics.accuracy_at_blending_start = data.get("accuracy_at_blending_start", 0.0)
        metrics.gradient_norm_avg = data.get("gradient_norm_avg", 0.0)
        metrics.current_alpha = data.get("current_alpha", 0.0)
        metrics.alpha_ramp_step = data.get("alpha_ramp_step", 0)
        metrics.counterfactual_contribution = data.get("counterfactual_contribution")
        metrics._blending_started = data.get("_blending_started", False)
        metrics.seed_gradient_norm_ratio = data.get("seed_gradient_norm_ratio", 0.0)
        metrics.host_param_count = data.get("host_param_count", 0)
        metrics.seed_param_count = data.get("seed_param_count", 0)
        return metrics


# =============================================================================
# Seed State
# =============================================================================

@dataclass(kw_only=True, slots=True)
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

    # Alpha controller state (persisted; future: enables partial holds + prune schedules)
    alpha_controller: AlphaController = field(default_factory=AlphaController)

    # Blending progress tracking
    blending_steps_done: int = 0
    blending_steps_total: int = 0
    blend_tempo_epochs: int = 5  # Default to STANDARD (5 epochs)

    # Flags
    is_healthy: bool = True
    is_paused: bool = False

    # History (bounded to prevent unbounded memory growth in long-running training)
    stage_history: deque = field(default_factory=lambda: deque(maxlen=PROBATION_HISTORY_MAXLEN))

    # Telemetry (initialized in __post_init__)
    telemetry: SeedTelemetry | None = field(default=None)

    def __post_init__(self):
        """Initialize telemetry with seed identity."""
        if self.telemetry is None:
            self.telemetry = SeedTelemetry(
                seed_id=self.seed_id,
                blueprint_id=self.blueprint_id,
            )
        # Keep controller alpha in sync with the scalar alpha used throughout Kasmina.
        self.alpha_controller.alpha = self.alpha

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

        IMPORTANT: accuracy_delta is stage-aware:
        - TRAINING/GERMINATED (alpha=0): Always 0.0 because seed cannot affect output
        - BLENDING+ (alpha>0): Stage-relative improvement (proxy for causal contribution)
        """
        from datetime import timezone

        self.telemetry.accuracy = self.metrics.current_val_accuracy

        # Stage-aware accuracy_delta: seeds with alpha=0 have zero causal impact
        # TRAINING and GERMINATED seeds are learning but not contributing to output
        if self.stage in (SeedStage.TRAINING, SeedStage.GERMINATED, SeedStage.DORMANT):
            self.telemetry.accuracy_delta = 0.0
        else:
            # BLENDING, PROBATIONARY, FOSSILIZED - seed is contributing via alpha
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

        # Compute blending velocity during BLENDING stage
        if self.stage == SeedStage.BLENDING:
            self.telemetry.blend_tempo_epochs = self.blend_tempo_epochs
            epochs_in_blend = self.metrics.epochs_in_current_stage
            if epochs_in_blend > 0:
                self.telemetry.blending_velocity = self.alpha / epochs_in_blend
            else:
                self.telemetry.blending_velocity = 0.0

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
            telemetry=replace(self.telemetry) if self.telemetry is not None else None,
            is_healthy=self.is_healthy,
            is_improving=self.metrics.improvement_since_stage_start > 0,
            needs_attention=not self.is_healthy,
        )

    def to_dict(self) -> dict:
        """Convert to primitive dict for PyTorch 2.9 weights_only=True serialization."""
        return {
            "seed_id": self.seed_id,
            "blueprint_id": self.blueprint_id,
            "slot_id": self.slot_id,
            "stage": self.stage.value,  # Enum -> int
            "previous_stage": self.previous_stage.value if self.previous_stage is not None else None,
            "stage_entered_at": self.stage_entered_at.isoformat(),  # datetime -> str
            "alpha": self.alpha,
            "stage_history": [
                (stage.value, ts.isoformat()) for stage, ts in self.stage_history
            ],  # deque of (Enum, datetime) -> list of (int, str)
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "telemetry": self.telemetry.to_dict() if self.telemetry else None,
            "alpha_controller": self.alpha_controller.to_dict(),
            "blending_steps_done": self.blending_steps_done,
            "blending_steps_total": self.blending_steps_total,
            "blend_tempo_epochs": self.blend_tempo_epochs,
            "is_healthy": self.is_healthy,
            "is_paused": self.is_paused,
            "previous_epochs_in_stage": self.previous_epochs_in_stage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeedState":
        """Reconstruct from primitive dict."""
        from datetime import datetime
        from collections import deque

        state = cls(
            seed_id=data["seed_id"],
            blueprint_id=data["blueprint_id"],
            slot_id=data["slot_id"],
            stage=SeedStage(data["stage"]),
            previous_stage=SeedStage(data["previous_stage"]) if data.get("previous_stage") is not None else None,
        )
        state.stage_entered_at = datetime.fromisoformat(data["stage_entered_at"])
        state.alpha = data.get("alpha", 0.0)
        state.alpha_controller = AlphaController.from_dict(data["alpha_controller"])
        # Scalar alpha is the source of truth at runtime; keep controller synced.
        state.alpha_controller.alpha = state.alpha
        state.stage_history = deque(
            (
                (SeedStage(stage), datetime.fromisoformat(ts))
                for stage, ts in data.get("stage_history", [])
            ),
            maxlen=PROBATION_HISTORY_MAXLEN,
        )
        if data.get("metrics"):
            state.metrics = SeedMetrics.from_dict(data["metrics"])
        if data.get("telemetry"):
            from esper.leyline.telemetry import SeedTelemetry
            state.telemetry = SeedTelemetry.from_dict(data["telemetry"])
        state.blending_steps_done = data.get("blending_steps_done", 0)
        state.blending_steps_total = data.get("blending_steps_total", 0)
        state.blend_tempo_epochs = data.get("blend_tempo_epochs", 5)
        state.is_healthy = data.get("is_healthy", True)
        state.is_paused = data.get("is_paused", False)
        state.previous_epochs_in_stage = data.get("previous_epochs_in_stage", 0)
        return state


# =============================================================================
# Quality Gates
# =============================================================================

# MIN_FOSSILIZE_CONTRIBUTION imported from leyline as DEFAULT_MIN_FOSSILIZE_CONTRIBUTION


class QualityGates:
    """Quality gate checks for stage transitions.

    Each gate validates that a seed is ready for the next stage.
    """

    def __init__(
        self,
        min_training_improvement: float = DEFAULT_MIN_TRAINING_IMPROVEMENT,
        min_blending_epochs: int = DEFAULT_MIN_BLENDING_EPOCHS,
        min_probation_stability: float = DEFAULT_MIN_PROBATION_STABILITY,
        min_seed_gradient_ratio: float = DEFAULT_GRADIENT_RATIO_THRESHOLD,
    ):
        self.min_training_improvement = min_training_improvement
        self.min_blending_epochs = min_blending_epochs
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
        """G2: Blending readiness – global improvement + seed readiness + gradient activity.

        NOTE: Gradient isolation is enforced structurally via detach() at the seed
        input boundary. There is no numeric "violation" detection - the structural
        guarantee is absolute. Host gradients from the direct loss path are EXPECTED
        and do NOT indicate isolation failures.
        """
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

        passed = perf_ok and seed_ok and gradient_ok
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

        # Check alpha reached target (from leyline)
        if state.alpha >= DEFAULT_ALPHA_COMPLETE_THRESHOLD:
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
        if contribution >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION:
            checks_passed.append(f"sufficient_contribution_{contribution:.2f}%")
        else:
            checks_failed.append(
                f"insufficient_contribution_{contribution:.2f}%_below_{DEFAULT_MIN_FOSSILIZE_CONTRIBUTION}%"
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
        self.telemetry_lifecycle_only: bool = False
        self.telemetry_inner_epoch: int | None = None
        self.telemetry_global_epoch: int | None = None
        self.task_config = task_config

        self.seed: nn.Module | None = None
        self.state: SeedState | None = None
        self.alpha_schedule = None
        self.isolate_gradients: bool = False

        # Only create isolation monitor if not in fast mode
        self.isolation_monitor = None

        # Cached shape probes to avoid per-germinate allocation
        # Keys: (topology, channels), values: (device, tensor)
        self._shape_probe_cache: dict[tuple[str, int], tuple[torch.device, torch.Tensor]] = {}

        # Cached alpha tensor to avoid per-forward allocation when alpha_schedule is None
        # Invalidated when alpha changes via set_alpha() or device changes via to()
        self._cached_alpha_tensor: torch.Tensor | None = None

        # Preallocated buffer for DDP gate synchronization to avoid per-call tensor creation
        # Only used when torch.distributed is available and initialized
        self._ddp_sync_buffer: torch.Tensor | None = None

        # Pending async gradient stats (tensor-based, no .item() sync yet)
        # Used by capture_gradient_telemetry_async() / finalize_gradient_telemetry() pair
        self._pending_gradient_stats: dict | None = None

    def _get_shape_probe(self, topology: str) -> torch.Tensor:
        """Get cached shape probe for topology, creating if needed."""
        # Include channels in key to handle slot reuse/reconfiguration (BUG-014)
        key = (topology, self.channels)
        cached = self._shape_probe_cache.get(key)

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
        self._shape_probe_cache[key] = (self.device, probe)
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
            # Infer from args/kwargs if no parameters (DORMANT state with no seed)
            # Check positional args first
            device_found = False
            for arg in args:
                if isinstance(arg, (str, torch.device)):
                    self.device = torch.device(arg) if isinstance(arg, str) else arg
                    device_found = True
                    break
            # Also check kwargs['device'] (e.g., .to(device='cuda:0'))
            if not device_found and 'device' in kwargs:
                device_arg = kwargs['device']
                if device_arg is not None:
                    self.device = torch.device(device_arg) if isinstance(device_arg, str) else device_arg

        # Only clear caches if device actually changed
        if self.device != old_device:
            self._shape_probe_cache.clear()
            self._cached_alpha_tensor = None  # Alpha tensor has device affinity
            self._ddp_sync_buffer = None  # DDP buffer has device affinity

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

        This method temporarily disables any active alpha_schedule to ensure
        forward() uses the forced value. This is essential for correct
        counterfactual attribution during BLENDING stage.

        Warning:
            NOT THREAD-SAFE. Do not use during concurrent forward passes
            or with DataParallel/DistributedDataParallel. Use model.eval()
            and single-threaded validation only.

            Nested calls are NOT supported - the inner override will be
            clobbered when the outer context exits.

        Note:
            When using torch.compile, this will cause graph specialization
            for the alpha_schedule=None path. This is acceptable as
            counterfactual evaluation runs once per epoch in eval mode.

        Args:
            value: Alpha value to force (typically 0.0 for host-only baseline)

        Yields:
            Context where alpha is temporarily overridden
        """
        if self.state is None:
            # No active seed - alpha is effectively 0, nothing to override
            yield
            return

        # Store previous state
        prev_alpha = self.state.alpha
        prev_schedule = self.alpha_schedule

        # Invalidate cache ensures forward() picks up the forced value
        self._cached_alpha_tensor = None

        # Override alpha AND disable schedule to force forward() to use state.alpha
        self.state.alpha = value
        self.state.alpha_controller.alpha = value
        self.alpha_schedule = None

        try:
            yield
        finally:
            self.state.alpha = prev_alpha
            self.state.alpha_controller.alpha = prev_alpha
            self.alpha_schedule = prev_schedule
            # Invalidate cache again to restore original behavior
            self._cached_alpha_tensor = None

    # TODO: [FUTURE ENHANCEMENT] - DDP support for force_alpha
    # Current implementation mutates instance state which is incompatible with
    # DistributedDataParallel. For DDP-safe counterfactual evaluation, consider:
    # 1. A separate "counterfactual forward" method that takes alpha as parameter
    # 2. Rank-local state override with barrier synchronization
    # 3. Functional approach that doesn't mutate module state

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
        blend_algorithm_id: str = "sigmoid",
        blend_tempo_epochs: int = 5,
    ) -> SeedState:
        """Germinate a new seed in this slot.

        Args:
            blueprint_id: Blueprint to instantiate (e.g., "norm", "attention")
            seed_id: Optional unique identifier for the seed
            host_module: Host network for gradient isolation (optional)
            blend_algorithm_id: Blending algorithm ("linear", "sigmoid", "gated")
            blend_tempo_epochs: Number of epochs for blending (3, 5, or 8)
        """
        from esper.kasmina.blueprints import BlueprintRegistry

        # Store blend settings for later use in start_blending()
        self._blend_algorithm_id = blend_algorithm_id
        self._blend_tempo_epochs = blend_tempo_epochs

        if self.is_active and not is_failure_stage(self.state.stage):
            raise RuntimeError(f"Slot {self.slot_id} already has active seed")

        # Default to "cnn" when no TaskConfig is provided to match legacy CNN tests.
        topology = self.task_config.topology if self.task_config is not None else "cnn"
        if topology not in ("cnn", "transformer"):
            raise ValueError(f"Unknown topology '{topology}' for SeedSlot.germinate")
        try:
            self.seed = BlueprintRegistry.create(topology, blueprint_id, self.channels)
        except ValueError as exc:
            available = BlueprintRegistry.list_for_topology(topology)
            names = [s.name for s in available]
            raise ValueError(
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

        # Store blend tempo in state
        self.state.blend_tempo_epochs = blend_tempo_epochs

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

        # Register for isolation monitoring
        # (DRL Expert review 2025-12-17: Always register regardless of fast_mode to keep
        # gradient stats fresh for observation vectors. fast_mode should only affect
        # telemetry emission, not core metric computation like seed_gradient_norm_ratio.)
        if host_module is not None:
            if self.isolation_monitor is None:
                self.isolation_monitor = GradientHealthMonitor()
            self.isolation_monitor.register(host_module, self.seed)

        self._emit_telemetry(
            TelemetryEventType.SEED_GERMINATED,
            data={
                "blueprint_id": blueprint_id,
                "seed_id": seed_id,
                "params": sum(p.numel() for p in self.seed.parameters() if p.requires_grad),
                "blend_tempo_epochs": blend_tempo_epochs,
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
        self._emit_telemetry(
            TelemetryEventType.SEED_GATE_EVALUATED,
            data={
                "gate": gate_result.gate.name,
                "passed": gate_result.passed,
                "target_stage": target_stage.name,
                "checks_passed": list(gate_result.checks_passed),
                "checks_failed": list(gate_result.checks_failed),
                "message": gate_result.message,
            },
        )

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
                # Call unified stage entry hook
                self._on_enter_stage(target_stage, old_stage)

                self._emit_telemetry(
                    TelemetryEventType.SEED_STAGE_CHANGED,
                    data={
                        "from": old_stage.name,
                        "to": target_stage.name,
                        "accuracy_delta": improvement,
                        "epochs_in_stage": epochs_in_stage,
                        "epochs_total": epochs_total,
                        "counterfactual": counterfactual,
                    }
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

        # Track auto-cull status for degenerate policy detection
        # Auto-culls happen via environment safety mechanisms, not explicit RL actions
        is_auto_cull = reason in SeedMetrics.AUTO_CULL_REASONS
        self.state.metrics.auto_culled = is_auto_cull
        self.state.metrics.auto_cull_reason = reason if is_auto_cull else ""

        old_stage = self.state.stage
        if not self.state.transition(SeedStage.CULLED):
            # Transition failed (shouldn't happen for non-FOSSILIZED)
            return False

        self._emit_telemetry(
            TelemetryEventType.SEED_STAGE_CHANGED,
            data={
                "from": old_stage.name,
                "to": SeedStage.CULLED.name,
                "accuracy_delta": improvement,
                "epochs_in_stage": epochs_in_stage,
                "epochs_total": epochs_total,
                "counterfactual": counterfactual,
            },
        )
        self._emit_telemetry(
            TelemetryEventType.SEED_CULLED,
            data={
                "reason": reason,
                "auto_culled": is_auto_cull,  # Distinguish explicit vs environment culls
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
        # Clear shape probe cache to prevent memory leak
        # (DRL Expert review 2025-12-17: cache holds intermediate tensors from shape
        # validation; in PPO rollouts with frequent cull/regerminate cycles, this can
        # accumulate significant GPU memory)
        self._shape_probe_cache.clear()
        # Clear cached alpha tensor (invalidate on cull)
        self._cached_alpha_tensor = None
        # Clear pending async gradient stats
        self._pending_gradient_stats = None
        return True

    def capture_gradient_telemetry(self) -> None:
        """Calculate gradient norms via the health monitor and update internal metrics.

        CRITICAL: Call this from Tolaria after loss.backward() to enable the G2 gate.
        Without this, seed_gradient_norm_ratio remains 0.0 and G2 always fails.

        Performance note: Calls compute_gradient_health() which internally performs
        .item() sync. The actual CUDA→CPU transfer happens in materialize_gradient_stats()
        within compute_gradient_health(). For fully async capture, use
        compute_gradient_health_async() and defer materialization.

        Use a stride in the training loop (e.g., every 10 steps) to minimize overhead.
        The EMA smoothing makes sparse sampling acceptable for gate decisions.
        """
        # Fast exit if no monitor (fast_mode) or no active seed
        if self.isolation_monitor is None or not self.is_active:
            return

        if not self.state:
            return

        # Ask monitor to calculate gradient norms for G2 gate health assessment
        stats = self.isolation_monitor.compute_gradient_health()

        host_norm = stats.get("host_grad_norm", 0.0)
        seed_norm = stats.get("seed_grad_norm", 0.0)

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
        # Note: If host_module was not provided to germinate(), host_param_count=0
        # and we fall back to raw (unnormalized) gradient ratio. This is acceptable
        # for single-seed scenarios but may cause inconsistent G2 gate behavior
        # in multi-seed comparisons where seeds have different host architectures.
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
            DEFAULT_GRADIENT_EMA_DECAY * current_avg + (1 - DEFAULT_GRADIENT_EMA_DECAY) * seed_norm
        )

    def capture_gradient_telemetry_async(self) -> bool:
        """Launch async gradient computation without CPU sync.

        Non-blocking version of capture_gradient_telemetry(). Launches GPU kernel
        to compute gradient norms but does NOT call .item() (no CPU-GPU sync).
        Stores tensor-based stats in _pending_gradient_stats.

        Usage pattern for PPO hot path:
            # After backward(), launch async computation
            for slot in seed_slots.values():
                slot.capture_gradient_telemetry_async()

            # ... do other work (optimizer step, etc.) ...

            # After stream.synchronize() or CUDA graph boundary
            for slot in seed_slots.values():
                slot.finalize_gradient_telemetry()

        Returns:
            True if async stats were captured, False if skipped (no monitor/no seed).
        """
        self._pending_gradient_stats = None

        # Fast exit if no monitor or no active seed
        if self.isolation_monitor is None or not self.is_active:
            return False

        if not self.state:
            return False

        # Launch async computation (no .item() sync)
        self._pending_gradient_stats = self.isolation_monitor.compute_gradient_health_async()
        return True

    def finalize_gradient_telemetry(self) -> None:
        """Materialize pending async gradient stats and update metrics.

        Deferred sync version - call AFTER stream.synchronize() or at CUDA graph
        boundary. Performs .item() calls to extract final values and updates
        seed_gradient_norm_ratio and gradient_norm_avg metrics.

        No-op if capture_gradient_telemetry_async() was not called or returned False.
        """
        if self._pending_gradient_stats is None:
            return

        if not self.state:
            self._pending_gradient_stats = None
            return

        # Materialize tensor values (this is where .item() sync happens)
        stats = self.isolation_monitor.materialize_gradient_stats(self._pending_gradient_stats)
        self._pending_gradient_stats = None

        # Same ratio computation logic as capture_gradient_telemetry()
        host_norm = stats.get("host_grad_norm", 0.0)
        seed_norm = stats.get("seed_grad_norm", 0.0)

        if host_norm < GRADIENT_EPSILON:
            raw_ratio = 0.0
        else:
            raw_ratio = seed_norm / host_norm

        # Apply parameter normalization if counts are available
        host_params = self.state.metrics.host_param_count
        seed_params = self.state.metrics.seed_param_count
        if host_params > 0 and seed_params > 0:
            normalization_factor = (host_params / seed_params) ** 0.5
            ratio = raw_ratio * normalization_factor
        else:
            ratio = raw_ratio

        # Clamp and update metrics
        self.state.metrics.seed_gradient_norm_ratio = min(MAX_GRADIENT_RATIO, ratio)

        # Update EMA of gradient norm for monitoring
        current_avg = self.state.metrics.gradient_norm_avg
        self.state.metrics.gradient_norm_avg = (
            DEFAULT_GRADIENT_EMA_DECAY * current_avg + (1 - DEFAULT_GRADIENT_EMA_DECAY) * seed_norm
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
        #
        #    CHANNELS_LAST WORKAROUND (BUG-005): When using channels_last memory
        #    format with isolate_gradients=True, PyTorch segfaults during backward.
        #    The bug affects BOTH the STE path (TRAINING) and the blend path
        #    (BLENDING+). The root cause is the combination of non-contiguous
        #    tensors (channels_last) with detach() in the autograd graph.
        #
        #    The fix is to make host_features contiguous BEFORE detach, so that
        #    the entire computation and its autograd graph use contiguous tensors.
        if self.isolate_gradients and not host_features.is_contiguous():
            # Make contiguous to avoid channels_last + detach segfault (BUG-005)
            host_features = host_features.contiguous()

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

        # Get alpha from blend algorithm's unified interface
        # All BlendAlgorithm subclasses implement get_alpha_for_blend(x) -> Tensor
        if self.alpha_schedule is not None:
            alpha = self.alpha_schedule.get_alpha_for_blend(host_features)
        else:
            # Use cached alpha tensor to avoid per-forward allocation overhead
            # Cache is invalidated in set_alpha() when value changes
            if (self._cached_alpha_tensor is None or
                self._cached_alpha_tensor.device != host_features.device or
                self._cached_alpha_tensor.dtype != host_features.dtype):
                self._cached_alpha_tensor = torch.tensor(
                    self.alpha, device=host_features.device, dtype=host_features.dtype
                )
            alpha = self._cached_alpha_tensor

        return blend_with_isolation(host_features, seed_features, alpha)

    def get_parameters(self):
        """Get trainable parameters of the seed."""
        if self.seed is None:
            return iter([])
        return self.seed.parameters()

    def set_alpha(self, alpha: float) -> None:
        """Set the blending alpha.

        Invalidates the cached alpha tensor to ensure forward() uses the new value.
        """
        if self.state:
            new_alpha = max(0.0, min(1.0, alpha))
            # Only invalidate cache if alpha actually changed
            if self.state.alpha != new_alpha:
                self._cached_alpha_tensor = None
            self.state.alpha = new_alpha
            self.state.alpha_controller.alpha = new_alpha
            self.state.metrics.current_alpha = self.state.alpha

    def start_blending(self, total_steps: int) -> None:
        """Initialize blending with selected algorithm.

        Uses blend_algorithm_id set during germinate(). Falls back to sigmoid
        if not specified.
        """
        from esper.kasmina.blending import BlendCatalog

        algorithm_id = getattr(self, "_blend_algorithm_id", "sigmoid")

        # Initialize blending progress tracking
        if self.state:
            self.state.blending_steps_total = total_steps
            self.state.blending_steps_done = 0
            match algorithm_id:
                case "linear":
                    curve = AlphaCurve.LINEAR
                case "sigmoid":
                    curve = AlphaCurve.SIGMOID
                case "gated":
                    curve = AlphaCurve.LINEAR
                case _:
                    raise ValueError(
                        f"Unknown blend algorithm: {algorithm_id}. "
                        f"Valid options: linear, sigmoid, gated"
                    )
            # Record the controller state even if we still use alpha_schedule for now.
            self.state.alpha_controller = AlphaController(alpha=self.state.alpha)
            self.state.alpha_controller.retarget(
                alpha_target=1.0,
                alpha_steps_total=total_steps,
                alpha_curve=curve,
            )

        topology = self.task_config.topology if self.task_config else "cnn"

        # Create blend algorithm with appropriate kwargs
        if algorithm_id == "gated":
            # GatedBlend needs channels, topology, and total_steps
            self.alpha_schedule = BlendCatalog.create(
                algorithm_id, channels=self.channels, topology=topology, total_steps=total_steps
            )
            # Move gated blend to same device as seed
            if isinstance(self.alpha_schedule, nn.Module):
                self.alpha_schedule = self.alpha_schedule.to(self.device)
        elif algorithm_id in ("linear", "sigmoid"):
            self.alpha_schedule = BlendCatalog.create(
                algorithm_id, total_steps=total_steps
            )
        else:
            # Exhaustive match above should prevent this, but keep a sanity fallback.
            raise AssertionError(f"Unhandled blend algorithm: {algorithm_id!r}")

    def _on_enter_stage(self, new_stage: SeedStage, old_stage: SeedStage) -> None:
        """Handle stage entry logic uniformly for both advance_stage() and step_epoch().

        This ensures consistent behavior regardless of which method triggers the transition.
        """
        if new_stage == SeedStage.TRAINING and old_stage == SeedStage.GERMINATED:
            # Enable Incubator isolation so seed sees detached host features
            self.isolate_gradients = True

        elif new_stage == SeedStage.BLENDING and old_stage == SeedStage.TRAINING:
            # Topology-aware gradient isolation:
            # - CNNs: keep isolation (host learns from loss, not seed feedback)
            # - Transformers: allow co-adaptation (host receives seed gradients)
            topology = self.task_config.topology if self.task_config else "cnn"
            self.isolate_gradients = (topology == "cnn")

            # Snapshot accuracy at blending start for true causal attribution
            if self.state:
                self.state.metrics.accuracy_at_blending_start = self.state.metrics.current_val_accuracy
                self.state.metrics._blending_started = True

            # Initialize blending schedule
            # Priority: stored tempo > TaskConfig > DEFAULT_BLENDING_TOTAL_STEPS
            total_steps = getattr(self, '_blend_tempo_epochs', None)
            if total_steps is None:
                total_steps = DEFAULT_BLENDING_TOTAL_STEPS
                if self.task_config is not None:
                    configured_steps = self.task_config.blending_steps
                    if isinstance(configured_steps, int) and configured_steps > 0:
                        total_steps = configured_steps
            self.start_blending(total_steps=total_steps)

        elif new_stage == SeedStage.PROBATIONARY and old_stage == SeedStage.BLENDING:
            # Clean up blending resources
            self._on_blending_complete()

    def _on_blending_complete(self) -> None:
        """Clean up after BLENDING stage completes.

        Discards alpha_schedule (no longer needed after full integration).
        Sets state.alpha = 1.0 (permanently fully blended).
        """
        self.alpha_schedule = None
        if self.state:
            self.set_alpha(1.0)
            self.state.alpha_controller.alpha_start = 1.0
            self.state.alpha_controller.alpha_target = 1.0
            self.state.alpha_controller.alpha_mode = AlphaMode.HOLD
            self.state.alpha_controller.alpha_steps_done = self.state.alpha_controller.alpha_steps_total

    def update_alpha_for_step(self, step: int) -> float:
        """Update alpha based on schedule."""
        if self.alpha_schedule is not None:
            self.alpha_schedule.step(step)
            alpha = self.alpha_schedule.get_alpha(step)

            self.set_alpha(alpha)
            if self.state:
                self.state.metrics.alpha_ramp_step = step
                self.state.alpha_controller.alpha_steps_done = step
                if self.state.alpha_controller.alpha_steps_total <= 0:
                    self.state.alpha_controller.alpha_steps_total = self.state.blending_steps_total
                if step >= self.state.alpha_controller.alpha_steps_total:
                    self.state.alpha_controller.alpha_mode = AlphaMode.HOLD
            return alpha
        return self.alpha

    def _sync_gate_decision(self, gate_result: GateResult) -> GateResult:
        """Ensure all DDP ranks agree on lifecycle transitions (Unanimous Consensus).

        We use ReduceOp.MIN (Logical AND) for all transitions.
        - Advancement: Everyone must be ready.
        - Culling: If one rank votes to keep, we wait (conservative).

        This prevents Architecture Divergence, where ranks typically crash
        if parameter shapes mismatch during the next forward pass.

        COLLECTIVE OPERATION: All ranks MUST call this in identical order
        for each gate check, or deadlock will occur.

        Returns:
            New GateResult with synchronized pass/fail decision (immutable pattern).
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return gate_result

        # Use preallocated buffer to avoid per-call tensor creation overhead
        # Buffer is lazily created on first DDP sync and invalidated on device change
        if self._ddp_sync_buffer is None:
            self._ddp_sync_buffer = torch.zeros(1, device=self.device, dtype=torch.int)

        # 1 = Passed, 0 = Failed; MIN gives unanimous consensus
        self._ddp_sync_buffer.fill_(int(gate_result.passed))
        torch.distributed.all_reduce(self._ddp_sync_buffer, op=torch.distributed.ReduceOp.MIN)

        is_passed = bool(self._ddp_sync_buffer.item())

        if gate_result.passed and not is_passed:
            # Vetoed by DDP consensus - create new result with updated status
            return GateResult(
                gate=gate_result.gate,
                passed=False,
                score=gate_result.score,
                checks_passed=gate_result.checks_passed,
                checks_failed=gate_result.checks_failed + ["ddp_veto"],
                message=gate_result.message + " (Vetoed by DDP consensus)",
            )

        # Either already failed locally or passed consensus
        return GateResult(
            gate=gate_result.gate,
            passed=is_passed,
            score=gate_result.score,
            checks_passed=gate_result.checks_passed,
            checks_failed=gate_result.checks_failed,
            message=gate_result.message,
        )

    def step_epoch(self) -> bool:
        """Advance lifecycle mechanically once per epoch (Kasmina timekeeper).

        Returns:
            True if an auto-cull occurred (environment safety mechanism),
            False otherwise. Used by the RL system to apply auto-cull penalties
            that prevent degenerate WAIT-spam policies.
        """
        if not self.state:
            return False

        stage = self.state.stage

        # GERMINATED → TRAINING: immediate advance (no dwell required)
        if stage == SeedStage.GERMINATED:
            gate_result = self.gates.check_gate(self.state, SeedStage.TRAINING)
            gate_result = self._sync_gate_decision(gate_result)
            if not gate_result.passed:
                return False

            old_stage = self.state.stage
            # Capture metrics before transition
            metrics = self.state.metrics
            epochs_in_stage = metrics.epochs_in_current_stage
            epochs_total = metrics.epochs_total
            improvement = metrics.total_improvement

            ok = self.state.transition(SeedStage.TRAINING)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition {self.state.stage} → TRAINING"
                )
            # Call unified stage entry hook
            self._on_enter_stage(SeedStage.TRAINING, old_stage)
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data={
                    "from": old_stage.name,
                    "to": self.state.stage.name,
                    "accuracy_delta": improvement,
                    "epochs_in_stage": epochs_in_stage,
                    "epochs_total": epochs_total,
                },
            )
            return False

        # TRAINING → BLENDING when dwell satisfied and gate passes
        if stage == SeedStage.TRAINING:
            dwell_epochs = 1
            if self.task_config:
                dwell_epochs = max(
                    1, int(self.task_config.max_epochs * self.task_config.train_to_blend_fraction)
                )
            if self.state.metrics.epochs_in_current_stage < dwell_epochs:
                return False

            gate_result = self.gates.check_gate(self.state, SeedStage.BLENDING)
            gate_result = self._sync_gate_decision(gate_result)
            if not gate_result.passed:
                return False

            old_stage = self.state.stage
            # Capture metrics before transition
            metrics = self.state.metrics
            epochs_in_stage = metrics.epochs_in_current_stage
            epochs_total = metrics.epochs_total
            improvement = metrics.total_improvement

            ok = self.state.transition(SeedStage.BLENDING)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition {self.state.stage} → BLENDING"
                )
            # Call unified stage entry hook
            self._on_enter_stage(SeedStage.BLENDING, old_stage)
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data={
                    "from": old_stage.name,
                    "to": self.state.stage.name,
                    "accuracy_delta": improvement,
                    "epochs_in_stage": epochs_in_stage,
                    "epochs_total": epochs_total,
                },
            )
            return False

        # BLENDING → PROBATIONARY when alpha ramp completes and gate passes
        if stage == SeedStage.BLENDING:
            self.state.blending_steps_done += 1

            if self.alpha_schedule is not None:
                self.update_alpha_for_step(self.state.blending_steps_done)

            if self.state.blending_steps_done >= self.state.blending_steps_total:
                self.set_alpha(1.0)  # Ensure fully blended
                gate_result = self.gates.check_gate(self.state, SeedStage.PROBATIONARY)
                gate_result = self._sync_gate_decision(gate_result)
                if not gate_result.passed:
                    return False
                old_stage = self.state.stage
                # Capture metrics before transition
                metrics = self.state.metrics
                epochs_in_stage = metrics.epochs_in_current_stage
                epochs_total = metrics.epochs_total
                improvement = metrics.total_improvement
                counterfactual = metrics.counterfactual_contribution

                ok = self.state.transition(SeedStage.PROBATIONARY)
                if not ok:
                    raise RuntimeError(
                        f"Illegal lifecycle transition {self.state.stage} → PROBATIONARY"
                    )
                # Call unified stage entry hook
                self._on_enter_stage(SeedStage.PROBATIONARY, old_stage)
                self._emit_telemetry(
                    TelemetryEventType.SEED_STAGE_CHANGED,
                    data={
                        "from": old_stage.name,
                        "to": self.state.stage.name,
                        "accuracy_delta": improvement,
                        "epochs_in_stage": epochs_in_stage,
                        "epochs_total": epochs_total,
                        "counterfactual": counterfactual,
                    },
                )
            return False

        # PROBATIONARY: Decision point for Tamiyo
        # Fossilization requires explicit FOSSILIZE action - NO auto-advance
        # (DRL Expert review 2025-12-10: auto-fossilize violated credit assignment)
        # Only handle safety auto-culls and timeout
        if stage == SeedStage.PROBATIONARY:
            # Calculate probation timeout (default DEFAULT_MAX_PROBATION_EPOCHS, or 10% of max_epochs)
            # Minimum of 5 epochs ensures sufficient time for counterfactual validation
            # (DRL Expert review 2025-12-09: 3 epochs was insufficient for reliable metrics)
            max_probation_epochs = DEFAULT_MAX_PROBATION_EPOCHS
            if self.task_config:
                max_probation_epochs = max(5, int(self.task_config.max_epochs * 0.1))

            # Safety auto-cull: negative counterfactual means seed actively hurts performance
            # This is a safety mechanism, not a decision bypass - Tamiyo should learn to
            # cull these earlier via the attribution penalty, but we don't let obviously
            # harmful seeds persist indefinitely
            if self.state.metrics.counterfactual_contribution is not None:
                if self.state.metrics.counterfactual_contribution <= 0:
                    self.cull(reason="negative_counterfactual")
                    return True  # Auto-cull occurred

            # Timeout: Tamiyo failed to decide in time
            # The escalating WAIT penalty in rewards.py creates pressure to decide sooner
            if self.state.metrics.epochs_in_current_stage >= max_probation_epochs:
                self.cull(reason="probation_timeout")
                return True  # Auto-cull occurred

        return False  # No auto-cull

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
        if self.on_telemetry is None:
            return
        if self.fast_mode and not self.telemetry_lifecycle_only:
            return

        payload = dict(data) if data else {}
        if self.state is not None:
            payload.setdefault("alpha", self.state.alpha)
        if self.telemetry_inner_epoch is not None:
            payload.setdefault("inner_epoch", self.telemetry_inner_epoch)
        if self.telemetry_global_epoch is not None:
            payload.setdefault("global_epoch", self.telemetry_global_epoch)
        if (
            self.state is not None
            and self.state.telemetry is not None
            and self.state.telemetry.epoch > 0
        ):
            payload.setdefault("seed_gradient_norm_ratio", self.state.metrics.seed_gradient_norm_ratio)
            payload.setdefault("gradient_health", self.state.telemetry.gradient_health)
            payload.setdefault("has_vanishing", self.state.telemetry.has_vanishing)
            payload.setdefault("has_exploding", self.state.telemetry.has_exploding)

        event = TelemetryEvent(
            event_type=event_type,
            seed_id=self.state.seed_id if self.state else None,
            slot_id=self.slot_id,
            data=payload,
        )
        self.on_telemetry(event)

    def get_extra_state(self) -> dict:
        """Persist SeedState for PyTorch 2.9+ weights_only=True compatibility.

        Returns only primitive types (dict, list, str, int, float, bool, None).
        The alpha_schedule nn.Module weights are saved via state_dict(), not here.
        """
        state_dict = {
            "isolate_gradients": self.isolate_gradients,
        }

        if self.state is not None:
            state_dict["seed_state"] = self.state.to_dict()

        # Alpha schedule: save config only, not the nn.Module
        # The nn.Module weights are saved in state_dict() automatically
        if self.alpha_schedule is not None:
            state_dict["alpha_schedule_config"] = {
                "algorithm_id": getattr(self.alpha_schedule, "algorithm_id", None),
                "total_steps": getattr(self.alpha_schedule, "total_steps", None),
                "current_step": getattr(self.alpha_schedule, "_current_step", 0),
            }
        else:
            state_dict["alpha_schedule_config"] = None

        return state_dict

    def set_extra_state(self, state: dict) -> None:
        """Restore SeedState from primitive dict."""
        self.isolate_gradients = state.get("isolate_gradients", False)

        if state.get("seed_state"):
            self.state = SeedState.from_dict(state["seed_state"])

        # Alpha schedule reconstruction
        # The nn.Module weights are restored via load_state_dict() automatically
        # because PyTorch 2.x includes dynamically assigned modules in state_dict.
        # We only need to restore config and ensure the correct algorithm type.
        if state.get("alpha_schedule_config"):
            config = state["alpha_schedule_config"]
            if config.get("algorithm_id") and self.state and self.state.stage == SeedStage.BLENDING:
                # CRITICAL: Restore algorithm_id BEFORE start_blending()
                # Without this, start_blending() defaults to "sigmoid" and
                # GatedBlend weights become orphaned "unexpected_keys".
                # See: docs/plans/2025-12-16-tolaria-kasmina-remediation.md
                self._blend_algorithm_id = config["algorithm_id"]
                self.start_blending(total_steps=config.get("total_steps", 10))
                # Restore step count (_current_step guaranteed to exist on all BlendAlgorithm instances)
                self.alpha_schedule._current_step = config.get("current_step", 0)
                # Re-restore blending_steps_done (start_blending resets it to 0)
                self.state.blending_steps_done = state["seed_state"].get("blending_steps_done", 0)
                # Re-restore alpha controller (start_blending resets it)
                self.state.alpha_controller = AlphaController.from_dict(state["seed_state"]["alpha_controller"])
                self.state.alpha_controller.alpha = self.state.alpha


__all__ = [
    "SeedMetrics",
    "SeedState",
    "QualityGates",
    "SeedSlot",
]
