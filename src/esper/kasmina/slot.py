"""Kasmina Slot - Seed lifecycle management.

The SeedSlot manages a single seed module through its lifecycle:
germination -> training -> blending -> fossilization/pruning.

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

DDP Symmetry Requirements
-------------------------
When using DistributedDataParallel (DDP), ALL ranks MUST:

1. Have identical SeedSlot configurations (same slots, same stages)
2. Execute the same forward() code path on each iteration
3. Call advance_stage() / step_epoch() in identical order

Violation causes gradient bucket mismatches → NCCL deadlock or shape errors.

Key mechanisms for DDP safety:
- _sync_gate_decision(): Broadcasts rank-0's gate decision to all ranks
- Symmetric lifecycle: All slots advance stages simultaneously
- Same seeds: All ranks germinate the same blueprint in the same slot

WARNING: force_alpha() context manager is NOT DDP-safe (mutates local state).
For counterfactual evaluation under DDP, use a separate non-DDP model replica.
"""

from __future__ import annotations

import os
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar, Generator, TYPE_CHECKING

import torch
import torch.nn as nn

from esper.kasmina.blend_ops import blend_gate, blend_multiply
from esper.kasmina.isolation import GradientHealthMonitor, blend_with_isolation, ste_forward
from esper.kasmina.alpha_controller import AlphaController
from esper.leyline.alpha import AlphaAlgorithm, AlphaCurve, AlphaMode

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
    # Telemetry Payloads
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedGateEvaluatedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
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
    DEFAULT_MIN_GRADIENT_HEALTH_FOR_BLENDING,
    DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE,
    # Blueprint lookup
    BLUEPRINT_ID_TO_INDEX,
)

if TYPE_CHECKING:
    from esper.kasmina.blending import AlphaScheduleProtocol
    from esper.tamiyo.policy.features import TaskConfig

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

    # Contribution velocity tracking (EMA of contribution changes over time)
    # Used for UI telemetry to show trend direction
    _prev_contribution: float | None = None
    contribution_velocity: float = 0.0

    # Flag to distinguish "never reached blending" from "started blending at 0% accuracy"
    _blending_started: bool = False

    # Gradient-based seed activity metric (parameter-normalized)
    # Formula: (seed_norm / host_norm) * sqrt(host_params / seed_params)
    # This measures per-parameter gradient intensity, scale-invariant across architectures
    # NOTE: None = never measured (distinct from 0.0 which means measured but inactive)
    seed_gradient_norm_ratio: float | None = None

    # Parameter counts for normalization (set once at germination)
    host_param_count: int = 0
    seed_param_count: int = 0

    # Auto-prune tracking for degenerate policy detection
    # (DRL Expert review 2025-12-17: policies could learn to rely on environment
    # cleanup rather than proactive pruning, creating reward hacking via WAIT spam)
    auto_pruned: bool = False
    auto_prune_reason: str = ""

    # Inter-slot interaction tracking (set by counterfactual engine)
    # These scaffolding metrics are reset at the START of each epoch's counterfactual phase
    interaction_sum: float = 0.0  # Σ I_ij for all j ≠ i (total synergy from interactions)
    boost_received: float = 0.0  # max(I_ij) for j ≠ i (strongest interaction partner)
    upstream_alpha_sum: float = 0.0  # Σ alpha_j for slots j < i (position-aware blending)
    downstream_alpha_sum: float = 0.0  # Σ alpha_j for slots j > i (position-aware blending)

    # Known auto-prune reasons (catastrophic safety only; HOLDING auto-prunes removed in Phase 4).
    AUTO_PRUNE_REASONS: ClassVar[frozenset[str]] = frozenset({
        "governor_nan",
        "governor_lobotomy",
        "governor_divergence",
        "governor_rollback",
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
        # NOTE: Leyline's SeedMetrics contract treats this as a float; internally we
        # use None to represent "never measured" for fail-loud gate behavior.
        seed_gradient_norm_ratio = (
            0.0 if self.seed_gradient_norm_ratio is None else float(self.seed_gradient_norm_ratio)
        )
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
            seed_gradient_norm_ratio=seed_gradient_norm_ratio,
            seed_param_count=self.seed_param_count,
            host_param_count=self.host_param_count,
            gradient_norm_avg=self.gradient_norm_avg,
            current_alpha=self.current_alpha,
            alpha_ramp_step=self.alpha_ramp_step,
            interaction_sum=self.interaction_sum,
            boost_received=self.boost_received,
            upstream_alpha_sum=self.upstream_alpha_sum,
            downstream_alpha_sum=self.downstream_alpha_sum,
        )

    # Schema version for SeedMetrics serialization
    # Increment when fields are added/removed/changed
    _SCHEMA_VERSION: ClassVar[int] = 2  # v2: added _prev_contribution, contribution_velocity

    def to_dict(self) -> dict[str, Any]:
        """Convert to primitive dict for serialization."""
        return {
            "_schema_version": self._SCHEMA_VERSION,
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
            "_prev_contribution": self._prev_contribution,
            "contribution_velocity": self.contribution_velocity,
            "_blending_started": self._blending_started,
            "seed_gradient_norm_ratio": self.seed_gradient_norm_ratio,
            "host_param_count": self.host_param_count,
            "seed_param_count": self.seed_param_count,
            "interaction_sum": self.interaction_sum,
            "boost_received": self.boost_received,
            "upstream_alpha_sum": self.upstream_alpha_sum,
            "downstream_alpha_sum": self.downstream_alpha_sum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedMetrics":
        """Reconstruct from primitive dict.

        Raises KeyError if required fields are missing (no silent defaults).
        Optional fields (counterfactual_contribution, _prev_contribution,
        contribution_velocity) may be absent in early-lifecycle checkpoints.
        """
        # Check schema version - fail fast on mismatch or missing
        schema_version = data["_schema_version"]  # Required - KeyError if missing
        if schema_version != cls._SCHEMA_VERSION:
            raise ValueError(
                f"SeedMetrics schema version mismatch: "
                f"expected {cls._SCHEMA_VERSION}, got {schema_version}"
            )

        metrics = cls()
        # Required fields - KeyError if missing (no silent defaults)
        metrics.epochs_total = data["epochs_total"]
        metrics.epochs_in_current_stage = data["epochs_in_current_stage"]
        metrics.initial_val_accuracy = data["initial_val_accuracy"]
        metrics.current_val_accuracy = data["current_val_accuracy"]
        metrics.best_val_accuracy = data["best_val_accuracy"]
        metrics.accuracy_at_stage_start = data["accuracy_at_stage_start"]
        metrics.accuracy_at_blending_start = data["accuracy_at_blending_start"]
        metrics.gradient_norm_avg = data["gradient_norm_avg"]
        metrics.current_alpha = data["current_alpha"]
        metrics.alpha_ramp_step = data["alpha_ramp_step"]
        metrics._blending_started = data["_blending_started"]
        metrics.seed_gradient_norm_ratio = data["seed_gradient_norm_ratio"]
        metrics.host_param_count = data["host_param_count"]
        metrics.seed_param_count = data["seed_param_count"]
        metrics.interaction_sum = data["interaction_sum"]
        metrics.boost_received = data["boost_received"]
        metrics.upstream_alpha_sum = data["upstream_alpha_sum"]
        metrics.downstream_alpha_sum = data["downstream_alpha_sum"]

        # Optional fields - these can legitimately be None/missing
        # counterfactual_contribution: None until counterfactual engine runs
        metrics.counterfactual_contribution = data.get("counterfactual_contribution")
        # _prev_contribution: None until second counterfactual measurement
        metrics._prev_contribution = data.get("_prev_contribution")
        # contribution_velocity: 0.0 until enough history exists
        metrics.contribution_velocity = data.get("contribution_velocity", 0.0)

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

    # Blend operator / gating mode (persisted; Phase 3+).
    alpha_algorithm: AlphaAlgorithm = AlphaAlgorithm.ADD

    blend_tempo_epochs: int = 5  # Default to STANDARD (5 epochs)

    # Flags
    is_healthy: bool = True
    is_paused: bool = False

    # History (bounded to prevent unbounded memory growth in long-running training)
    stage_history: deque[tuple[SeedStage, datetime]] = field(default_factory=lambda: deque(maxlen=PROBATION_HISTORY_MAXLEN))

    # Telemetry (initialized in __post_init__)
    telemetry: SeedTelemetry | None = field(default=None)

    def __post_init__(self) -> None:
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
        gradient_norm: float | None = None,
        gradient_health: float | None = None,
        has_vanishing: bool | None = None,
        has_exploding: bool | None = None,
        epoch: int = 0,
        max_epochs: int = 25,
    ) -> None:
        """Sync telemetry from metrics + optional gradient signals.

        Call this once per epoch after validation to update telemetry.
        SeedMetrics remains the source of truth for accuracy/epoch data.

        Gradient parameters are optional - when None, gradient-related telemetry
        fields are left at their default values (no gradient data available).
        This separates the concern of accuracy telemetry (always available)
        from gradient telemetry (only when gradient stats are collected).

        IMPORTANT: accuracy_delta is stage-aware:
        - TRAINING/GERMINATED (alpha=0): Always 0.0 because seed cannot affect output
        - BLENDING+ (alpha>0): Stage-relative improvement (proxy for causal contribution)
        """
        from datetime import timezone

        if self.telemetry is None:
            return

        self.telemetry.accuracy = self.metrics.current_val_accuracy

        # Stage-aware accuracy_delta: seeds with alpha=0 have zero causal impact
        # TRAINING and GERMINATED seeds are learning but not contributing to output
        if self.stage in (SeedStage.TRAINING, SeedStage.GERMINATED, SeedStage.DORMANT):
            self.telemetry.accuracy_delta = 0.0
        else:
            # BLENDING, HOLDING, FOSSILIZED - seed is contributing via alpha
            self.telemetry.accuracy_delta = self.metrics.improvement_since_stage_start
        self.telemetry.epochs_in_stage = self.metrics.epochs_in_current_stage
        self.telemetry.stage = self.stage.value
        self.telemetry.alpha = self.alpha
        self.telemetry.alpha_target = self.alpha_controller.alpha_target
        self.telemetry.alpha_mode = int(self.alpha_controller.alpha_mode)
        self.telemetry.alpha_steps_total = self.alpha_controller.alpha_steps_total
        self.telemetry.alpha_steps_done = self.alpha_controller.alpha_steps_done
        self.telemetry.time_to_target = max(
            self.alpha_controller.alpha_steps_total - self.alpha_controller.alpha_steps_done,
            0,
        )
        if self.alpha_controller.alpha_mode == AlphaMode.HOLD or self.alpha_controller.alpha_steps_total == 0:
            self.telemetry.alpha_velocity = 0.0
        else:
            self.telemetry.alpha_velocity = (
                (self.alpha_controller.alpha_target - self.alpha_controller.alpha_start)
                / self.alpha_controller.alpha_steps_total
            )
        self.telemetry.alpha_algorithm = int(self.alpha_algorithm)

        # Only update gradient telemetry when data is provided
        # (separates accuracy telemetry from gradient telemetry concerns)
        if gradient_norm is not None:
            self.telemetry.gradient_norm = gradient_norm
        if gradient_health is not None:
            self.telemetry.gradient_health = gradient_health
        if has_vanishing is not None:
            self.telemetry.has_vanishing = has_vanishing
        if has_exploding is not None:
            self.telemetry.has_exploding = has_exploding

        self.telemetry.epoch = epoch
        self.telemetry.max_epochs = max_epochs
        self.telemetry.captured_at = datetime.now(timezone.utc)

        # Tempo is chosen at germination and should be observable regardless of stage.
        self.telemetry.blend_tempo_epochs = self.blend_tempo_epochs

        # Compute blending velocity during BLENDING stage only.
        if self.stage == SeedStage.BLENDING:
            epochs_in_blend = self.metrics.epochs_in_current_stage
            self.telemetry.blending_velocity = (
                (self.alpha / epochs_in_blend) if epochs_in_blend > 0 else 0.0
            )
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
        controller = self.alpha_controller
        alpha_steps_total = int(controller.alpha_steps_total)
        alpha_steps_done = int(controller.alpha_steps_done)
        time_to_target = max(alpha_steps_total - alpha_steps_done, 0)
        if controller.alpha_mode == AlphaMode.HOLD or alpha_steps_total == 0:
            alpha_velocity = 0.0
        else:
            alpha_velocity = (controller.alpha_target - controller.alpha_start) / alpha_steps_total

        return SeedStateReport(
            seed_id=self.seed_id,
            slot_id=self.slot_id,
            blueprint_id=self.blueprint_id,
            blueprint_index=BLUEPRINT_ID_TO_INDEX.get(self.blueprint_id, -1),
            stage=self.stage,
            previous_stage=self.previous_stage,
            previous_epochs_in_stage=self.previous_epochs_in_stage,
            stage_entered_at=self.stage_entered_at,
            alpha_mode=int(controller.alpha_mode),
            alpha_target=controller.alpha_target,
            alpha_steps_total=alpha_steps_total,
            alpha_steps_done=alpha_steps_done,
            time_to_target=time_to_target,
            alpha_velocity=alpha_velocity,
            alpha_algorithm=int(self.alpha_algorithm),
            blend_tempo_epochs=self.blend_tempo_epochs,
            metrics=self.metrics.to_leyline(),
            telemetry=replace(self.telemetry) if self.telemetry is not None else None,
            is_healthy=self.is_healthy,
            is_improving=self.metrics.improvement_since_stage_start > 0,
            needs_attention=not self.is_healthy,
        )

    def to_dict(self) -> dict[str, Any]:
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
            "alpha_algorithm": int(self.alpha_algorithm),
            "blend_tempo_epochs": self.blend_tempo_epochs,
            "is_healthy": self.is_healthy,
            "is_paused": self.is_paused,
            "previous_epochs_in_stage": self.previous_epochs_in_stage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedState":
        """Reconstruct from primitive dict.

        Raises KeyError/ValueError if required fields are missing (no silent defaults).
        """
        from datetime import datetime
        from collections import deque

        # Required field with explicit error message
        if "alpha_algorithm" not in data:
            raise ValueError(
                "Checkpoint seed_state is missing required 'alpha_algorithm' (Phase 3+)."
            )

        # Required field with explicit error message
        if "alpha_controller" not in data or not isinstance(data["alpha_controller"], dict):
            raise ValueError(
                "Checkpoint seed_state is missing required 'alpha_controller' (Phase 1+). "
                "Pre-Phase-1 checkpoints are not supported for resume."
            )

        state = cls(
            seed_id=data["seed_id"],  # Required - KeyError if missing
            blueprint_id=data["blueprint_id"],  # Required
            slot_id=data["slot_id"],  # Required
            stage=SeedStage(data["stage"]),  # Required
            previous_stage=SeedStage(data["previous_stage"]) if data["previous_stage"] is not None else SeedStage.UNKNOWN,
            alpha_algorithm=AlphaAlgorithm(int(data["alpha_algorithm"])),
        )
        state.stage_entered_at = datetime.fromisoformat(data["stage_entered_at"])  # Required
        state.alpha = data["alpha"]  # Required - no silent default to 0.0
        state.alpha_controller = AlphaController.from_dict(data["alpha_controller"])
        # Scalar alpha is the source of truth at runtime; keep controller synced.
        state.alpha_controller.alpha = state.alpha
        state.stage_history = deque(
            (
                (SeedStage(stage), datetime.fromisoformat(ts))
                for stage, ts in data["stage_history"]  # Required - no silent default to []
            ),
            maxlen=PROBATION_HISTORY_MAXLEN,
        )
        # metrics and telemetry are optional - they may not exist for newly created seeds
        if data.get("metrics"):
            state.metrics = SeedMetrics.from_dict(data["metrics"])
        if data.get("telemetry"):
            from esper.leyline.telemetry import SeedTelemetry
            state.telemetry = SeedTelemetry.from_dict(data["telemetry"])
        state.blend_tempo_epochs = data["blend_tempo_epochs"]  # Required
        state.is_healthy = data["is_healthy"]  # Required
        state.is_paused = data["is_paused"]  # Required
        state.previous_epochs_in_stage = data["previous_epochs_in_stage"]  # Required
        return state


# =============================================================================
# Quality Gates
# =============================================================================

# MIN_FOSSILIZE_CONTRIBUTION imported from leyline as DEFAULT_MIN_FOSSILIZE_CONTRIBUTION


class QualityGates:
    """Quality gate checks for stage transitions.

    Each gate validates that a seed is ready for the next stage.

    Args:
        permissive: If True, gates only enforce structural requirements
            (alpha completion, health checks) and let Tamiyo learn quality
            thresholds through reward signals. If False (default), gates
            enforce quality thresholds that prevent low-quality seeds from
            advancing. Set permissive=True for RL training where we want
            the policy to learn from mistakes rather than being prevented
            from making them.
    """

    def __init__(
        self,
        min_training_improvement: float = DEFAULT_MIN_TRAINING_IMPROVEMENT,
        min_blending_epochs: int = DEFAULT_MIN_BLENDING_EPOCHS,
        min_probation_stability: float = DEFAULT_MIN_PROBATION_STABILITY,
        min_seed_gradient_ratio: float = DEFAULT_GRADIENT_RATIO_THRESHOLD,
        min_gradient_health_for_blending: float = DEFAULT_MIN_GRADIENT_HEALTH_FOR_BLENDING,
        permissive: bool = False,
    ):
        self.min_training_improvement = min_training_improvement
        self.min_blending_epochs = min_blending_epochs
        self.min_probation_stability = min_probation_stability
        self.min_seed_gradient_ratio = min_seed_gradient_ratio
        self.min_gradient_health_for_blending = min_gradient_health_for_blending
        self.permissive = permissive

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
        """Map target stage to gate level.

        Raises:
            ValueError: If target_stage has no gate mapping.
        """
        mapping = {
            SeedStage.GERMINATED: GateLevel.G0,
            SeedStage.TRAINING: GateLevel.G1,
            SeedStage.BLENDING: GateLevel.G2,
            SeedStage.HOLDING: GateLevel.G3,  # Was G4, now G3 (direct from BLENDING)
            SeedStage.FOSSILIZED: GateLevel.G5,
        }
        if target_stage not in mapping:
            raise ValueError(
                f"No gate defined for target stage {target_stage.name}. "
                f"Valid gated stages: {', '.join(s.name for s in mapping.keys())}"
            )
        return mapping[target_stage]

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

        In permissive mode, only checks that seed has trained at least 1 epoch.
        This lets Tamiyo learn quality thresholds through reward signals.
        """
        checks_passed = []
        checks_failed = []

        improvement = state.metrics.improvement_since_stage_start

        # PERMISSIVE MODE: Skip quality checks but enforce SAFETY gates.
        # Safety gates ensure the seed won't destabilize the host when blended.
        # Per PyTorch expert review 2025-01-09:
        # 1. Minimum epochs to exit "initial chaos" phase of training
        # 2. No exploding gradients (hard requirement - never negotiable)
        # 3. Gradient health above threshold (indicates stable learning)
        if self.permissive:
            # Check 1: Minimum training epochs
            min_epochs = self.min_blending_epochs
            if state.metrics.epochs_in_current_stage >= min_epochs:
                checks_passed.append(f"trained_{state.metrics.epochs_in_current_stage}_epochs")
                epochs_ok = True
            else:
                checks_failed.append(
                    f"insufficient_training_{state.metrics.epochs_in_current_stage}_of_{min_epochs}"
                )
                epochs_ok = False

            # Check 2: No exploding gradients (HARD REQUIREMENT)
            # Exploding gradients indicate unbounded dynamics that will compound
            # catastrophically when the seed's alpha is ramped up during blending.
            telemetry = state.telemetry
            if telemetry is not None and telemetry.has_exploding:
                checks_failed.append("exploding_gradients")
                exploding_ok = False
            else:
                checks_passed.append("no_exploding_gradients")
                exploding_ok = True

            # Check 3: Gradient health above safety threshold
            # Low gradient health indicates the seed may destabilize the host.
            gradient_health_ok = True  # Default to OK if no telemetry
            if telemetry is not None:
                health = telemetry.gradient_health
                if health >= self.min_gradient_health_for_blending:
                    checks_passed.append(f"gradient_health_{health:.2f}")
                else:
                    checks_failed.append(
                        f"gradient_health_low_{health:.2f}_need_{self.min_gradient_health_for_blending:.2f}"
                    )
                    gradient_health_ok = False

            passed = epochs_ok and exploding_ok and gradient_health_ok
            return GateResult(
                gate=GateLevel.G2,
                passed=passed,
                score=min(1.0, improvement / 5.0) if improvement > 0 else 0.0,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                message=f"Permissive G2 Safety: {'PASS' if passed else 'FAIL'}",
            )

        # STRICT MODE: Full quality checks
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
        # NOTE: None means gradient stats were never collected (training loop coupling issue)
        ratio = state.metrics.seed_gradient_norm_ratio
        if ratio is None:
            checks_failed.append("gradient_stats_never_measured")
            gradient_ok = False
        elif ratio >= self.min_seed_gradient_ratio:
            checks_passed.append(f"seed_gradient_active_{ratio:.2f}")
            gradient_ok = True
        else:
            checks_failed.append(f"seed_gradient_low_{ratio:.2f}")
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
        """G3: Holding readiness - blending completed with stable integration.

        In permissive mode, only checks alpha completion (structural requirement).
        The min_blending_epochs check is skipped to let Tamiyo learn timing.
        """
        checks_passed = []
        checks_failed = []
        controller = state.alpha_controller
        eps = 1e-6

        # STRUCTURAL CHECKS (always enforced, both modes):
        # BLENDING -> HOLDING is only legal at full amplitude (alpha_target==1.0).
        if controller.alpha_target >= 1.0 - eps:
            checks_passed.append("alpha_target_full")
            alpha_target_ok = True
        else:
            checks_failed.append(f"alpha_target_not_full_{controller.alpha_target:.2f}")
            alpha_target_ok = False

        # Completion is defined by reaching the controller target and entering HOLD.
        if controller.alpha_mode == AlphaMode.HOLD and abs(controller.alpha - controller.alpha_target) <= eps:
            checks_passed.append("alpha_target_reached")
            alpha_reached_ok = True
        else:
            checks_failed.append(
                f"alpha_not_at_target_{controller.alpha:.2f}_target_{controller.alpha_target:.2f}_mode_{controller.alpha_mode.name}"
            )
            alpha_reached_ok = False

        # PERMISSIVE MODE: Only structural alpha checks
        if self.permissive:
            passed = alpha_target_ok and alpha_reached_ok
            return GateResult(
                gate=GateLevel.G3,
                passed=passed,
                score=controller.alpha,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                message="Permissive G3: alpha completion only",
            )

        # STRICT MODE: Also check blending duration
        if state.metrics.epochs_in_current_stage >= self.min_blending_epochs:
            checks_passed.append("blending_complete")
        else:
            checks_failed.append(f"blending_incomplete_{state.metrics.epochs_in_current_stage}")

        passed = len(checks_failed) == 0
        return GateResult(
            gate=GateLevel.G3,
            passed=passed,
            score=controller.alpha,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def _check_g5(self, state: SeedState) -> GateResult:
        """G5: Fossilization readiness - requires counterfactual validation.

        G5 is only reachable from HOLDING stage where counterfactual
        validation is mandatory. No fallback to total_improvement.

        In permissive mode, only checks:
        - Counterfactual is available (needed for measurement)
        - Seed is healthy (safety - don't fossilize broken seeds)
        The contribution threshold is skipped to let Tamiyo learn.
        """
        checks_passed = []
        checks_failed = []

        # REQUIRE counterfactual - no fallback (both modes)
        # We need counterfactual for reward computation even in permissive mode
        contribution = state.metrics.counterfactual_contribution
        if contribution is None:
            return GateResult(
                gate=GateLevel.G5,
                passed=False,
                score=0.0,
                checks_passed=[],
                checks_failed=["counterfactual_not_available"],
            )

        # Log contribution for telemetry (both modes)
        checks_passed.append(f"counterfactual_available_{contribution:.2f}%")

        # SAFETY CHECK: Health (both modes - never fossilize a broken seed)
        if state.is_healthy:
            checks_passed.append("healthy")
            health_ok = True
        else:
            checks_failed.append("unhealthy")
            health_ok = False

        # PERMISSIVE MODE: Skip contribution threshold, let Tamiyo learn
        if self.permissive:
            passed = health_ok  # Only health matters
            return GateResult(
                gate=GateLevel.G5,
                passed=passed,
                score=min(1.0, contribution / 10.0) if contribution > 0 else 0.0,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                message=f"Permissive G5: {contribution:.2f}% contribution (no threshold)",
            )

        # STRICT MODE: Check contribution meets minimum threshold
        if contribution >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION:
            checks_passed.append(f"sufficient_contribution_{contribution:.2f}%")
        else:
            checks_failed.append(
                f"insufficient_contribution_{contribution:.2f}%_below_{DEFAULT_MIN_FOSSILIZE_CONTRIBUTION}%"
            )

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

    # Schema version for extra_state serialization (checkpoint compatibility)
    _EXTRA_STATE_VERSION: ClassVar[int] = 1

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
        # Auto-forward gates: stage transitions can be advanced automatically by step_epoch()
        # when the corresponding gate passes (configured by Simic TrainingConfig).
        self.auto_forward_gates: frozenset[GateLevel] = frozenset()
        self.task_config = task_config
        self._resolved_topology: str | None = None

        self.seed: nn.Module | None = None
        self.state: SeedState | None = None
        self.alpha_schedule: AlphaScheduleProtocol | None = None
        self.isolate_gradients: bool = False

        # Only create isolation monitor if not in fast mode
        self.isolation_monitor: GradientHealthMonitor | None = None

        # Cached shape probes to avoid per-germinate allocation
        # Keys: (topology, channels), values: (device, tensor)
        self._shape_probe_cache: dict[tuple[str, int], tuple[torch.device, torch.Tensor]] = {}

        # Cached alpha tensor to avoid per-forward allocation when alpha_schedule is None
        # Invalidated when alpha changes via set_alpha() or device changes via to()
        self._cached_alpha_tensor: torch.Tensor | None = None

        # Pending async gradient stats (tensor-based, no .item() sync yet)
        # Used by capture_gradient_telemetry_async() / finalize_gradient_telemetry() pair
        self._pending_gradient_stats: dict[str, Any] | None = None

        # Phase 3: BLEND_OUT freeze tracking (mandatory invariant during DOWN schedules).
        # We record which params were trainable so we can restore on exit.
        self._blend_out_freeze_active: bool = False
        self._blend_out_frozen_params: list[nn.Parameter] = []

        # Scheduled prune bookkeeping (reason/initiator preserved until completion).
        self._pending_prune_reason: str | None = None
        self._pending_prune_initiator: str | None = None

        # Blending state (initialized to None, set when blending starts)
        self._blend_algorithm_id: str | None = None
        self._blend_alpha_target: float | None = None
        self._blend_tempo_epochs: int | None = None

    def _resolve_topology(self) -> str:
        """Resolve topology for blueprint + blending behavior.

        Source of truth priority:
        1) TaskConfig.topology (explicit, required for RL)
        2) _resolved_topology (captured at germination time by MorphogeneticModel/host)
        3) Default "cnn" (keeps slot-only unit tests lightweight)
        """
        if self.task_config is not None:
            return self.task_config.topology
        if self._resolved_topology is not None:
            return self._resolved_topology
        return "cnn"

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

    def to(self, *args: Any, **kwargs: Any) -> "SeedSlot":
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
    def force_alpha(self, value: float) -> Generator[None, None, None]:
        """Temporarily override alpha for counterfactual evaluation.

        Used for differential validation to measure true seed contribution
        by comparing real output (current alpha) vs host-only (alpha=0).

        For `AlphaAlgorithm.GATE`, we preserve `alpha_schedule` because
        `forward()` requires it; forcing `alpha=0.0` still yields host-only
        output because amplitude is zero. For non-gated algorithms,
        `alpha_schedule` should be absent and is cleared defensively.

        Warning:
            NOT THREAD-SAFE. Do not use during concurrent forward passes
            or with DataParallel/DistributedDataParallel. Use model.eval()
            and single-threaded validation only.

            Nested calls are NOT supported - the inner override will be
            clobbered when the outer context exits.

        Note:
            When using torch.compile, this will cause graph specialization
            for the forced-alpha path (and for non-gated algorithms,
            the alpha_schedule=None path). This is acceptable as
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
        prev_algorithm = self.state.alpha_algorithm
        prev_schedule = self.alpha_schedule

        # Invalidate cache ensures forward() picks up the forced value
        self._cached_alpha_tensor = None

        # Override alpha (and invalidate cached alpha tensor).
        self.state.alpha = value
        self.state.alpha_controller.alpha = value
        if prev_algorithm != AlphaAlgorithm.GATE:
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
        """Return trainable params of the active seed subsystem (seed + optional gate)."""
        if self.seed is None:
            return 0
        total = 0
        if self.seed is not None:
            for param in self.seed.parameters():
                if param.requires_grad:
                    total += param.numel()
        if self.alpha_schedule is not None and isinstance(self.alpha_schedule, nn.Module):
            for param in self.alpha_schedule.parameters():
                if param.requires_grad:
                    total += param.numel()
        return total

    def germinate(
        self,
        blueprint_id: str,
        seed_id: str | None = None,
        host_module: nn.Module | None = None,
        blend_algorithm_id: str = "sigmoid",
        blend_tempo_epochs: int = 5,
        alpha_algorithm: AlphaAlgorithm = AlphaAlgorithm.ADD,
        alpha_target: float | None = None,
        topology: str | None = None,
    ) -> SeedState:
        """Germinate a new seed in this slot.

        Args:
            blueprint_id: Blueprint to instantiate (e.g., "norm", "attention")
            seed_id: Optional unique identifier for the seed
            host_module: Host network for gradient isolation (optional)
            blend_algorithm_id: Blending algorithm ("linear", "sigmoid", "gated")
            blend_tempo_epochs: Number of epochs for blending (3, 5, or 8)
            alpha_algorithm: Blend operator / gating mode (ADD, MULTIPLY, or GATE).
            alpha_target: Initial blend target (defaults to full amplitude).
            topology: Optional topology override ("cnn" or "transformer").
                Used by MorphogeneticModel when TaskConfig is not provided.
        """
        from esper.kasmina.blueprints import BlueprintRegistry

        # Store blend settings for later use in start_blending()
        self._blend_algorithm_id = blend_algorithm_id
        self._blend_tempo_epochs = blend_tempo_epochs
        if alpha_target is None:
            self._blend_alpha_target = 1.0
        else:
            if not (0.0 < alpha_target <= 1.0):
                raise ValueError("alpha_target must be within (0, 1].")
            self._blend_alpha_target = float(alpha_target)

        resolved_alpha_algorithm = alpha_algorithm
        if blend_algorithm_id == "gated":
            if alpha_algorithm != AlphaAlgorithm.GATE:
                raise ValueError("blend_algorithm_id='gated' requires alpha_algorithm=GATE")
        elif alpha_algorithm == AlphaAlgorithm.GATE:
            raise ValueError("alpha_algorithm=GATE requires blend_algorithm_id='gated'")

        # Phase 4 contract: PRUNED/EMBARGOED/RESETTING are first-class and keep
        # state even after physical removal (seed=None). The slot is unavailable
        # for germination until it fully returns to DORMANT (state cleared).
        if self.state is not None or self.seed is not None:
            stage_name = self.state.stage.name if self.state is not None else "UNKNOWN"
            raise RuntimeError(
                f"Slot {self.slot_id} is unavailable for germination (stage={stage_name})"
            )

        if self.task_config is not None:
            topology_from_task = self.task_config.topology
            if topology is not None and topology != topology_from_task:
                raise ValueError(
                    f"Topology mismatch for slot {self.slot_id}: task_config.topology={topology_from_task!r} "
                    f"but germinate() received topology={topology!r}."
                )
            resolved_topology = topology_from_task
        else:
            # Default to "cnn" when no TaskConfig is provided to match slot-only tests.
            resolved_topology = topology if topology is not None else "cnn"
        if resolved_topology not in ("cnn", "transformer"):
            raise ValueError(f"Unknown topology '{resolved_topology}' for SeedSlot.germinate")
        self._resolved_topology = resolved_topology
        try:
            self.seed = BlueprintRegistry.create(resolved_topology, blueprint_id, self.channels)
        except ValueError as exc:
            available = BlueprintRegistry.list_for_topology(resolved_topology)
            names = [s.name for s in available]
            raise ValueError(
                f"Blueprint '{blueprint_id}' not available for topology '{resolved_topology}'. Available: {names}"
            ) from exc
        self.seed = self.seed.to(self.device)
        if resolved_alpha_algorithm == AlphaAlgorithm.MULTIPLY:
            from esper.kasmina.blueprints.initialization import zero_init_final_layer

            zero_init_final_layer(self.seed, allow_missing=True)

        # Validate shape: ensure seed preserves feature shape in a host-agnostic way
        # without mutating host BatchNorm statistics. Smoke test only.
        shape_probe = self._get_shape_probe(resolved_topology)
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
            alpha_algorithm=resolved_alpha_algorithm,
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
            data=SeedGerminatedPayload(
                slot_id=self.slot_id,
                env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                blueprint_id=blueprint_id,
                params=sum(p.numel() for p in self.seed.parameters() if p.requires_grad),
                alpha=self.state.alpha if self.state else 0.0,
                blend_tempo_epochs=blend_tempo_epochs,
                alpha_curve=self.state.alpha_controller.alpha_curve.name,
                # Optional gradient health fields - will be zero/false at germination
                grad_ratio=0.0,
                has_vanishing=False,
                has_exploding=False,
                epochs_in_stage=0,
            )
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

        assert self.state is not None  # is_active guarantees this

        if target_stage is not None and is_failure_stage(target_stage):
            raise ValueError(
                f"advance_stage() cannot target failure stage {target_stage.name}; "
                "use prune() and the cooldown pipeline instead."
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

        # Check gate (local evaluation)
        gate_result = self.gates.check_gate(self.state, target_stage)

        # Synchronize gate decision across DDP ranks (rank-0 authoritative)
        # This prevents BUG-030 architecture divergence from unsynced lifecycle transitions
        gate_result = self._sync_gate_decision(gate_result)

        self._emit_telemetry(
            TelemetryEventType.SEED_GATE_EVALUATED,
            data=SeedGateEvaluatedPayload(
                slot_id=self.slot_id,
                env_id=-1,  # Sentinel - replaced by emit_with_env_context in simic
                gate=gate_result.gate.name,
                passed=gate_result.passed,
                target_stage=target_stage.name,
                checks_passed=tuple(gate_result.checks_passed),
                checks_failed=tuple(gate_result.checks_failed),
                message=gate_result.message,
            ),
        )

        if gate_result.passed:
            old_stage = self.state.stage

            # Capture metrics before transition resets stage counters
            metrics = self.state.metrics
            improvement = metrics.total_improvement
            counterfactual = metrics.counterfactual_contribution
            epochs_total = metrics.epochs_total
            epochs_in_stage = metrics.epochs_in_current_stage
            blueprint_id = self.state.blueprint_id

            if self.state.transition(target_stage):
                # Call unified stage entry hook
                self._on_enter_stage(target_stage, old_stage)

                self._emit_telemetry(
                    TelemetryEventType.SEED_STAGE_CHANGED,
                    data=SeedStageChangedPayload(
                        slot_id=self.slot_id,
                        env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                        from_stage=old_stage.name,
                        to_stage=target_stage.name,
                        alpha=self.state.alpha,
                        accuracy_delta=improvement,
                        epochs_in_stage=epochs_in_stage,
                        alpha_curve=self.state.alpha_controller.alpha_curve.name,
                        # Optional gradient health fields
                        grad_ratio=self._telemetry_grad_ratio(),
                        has_vanishing=(
                            self.state.telemetry.has_vanishing
                            if self.state.telemetry and self.state.telemetry.epoch > 0
                            else False
                        ),
                        has_exploding=(
                            self.state.telemetry.has_exploding
                            if self.state.telemetry and self.state.telemetry.epoch > 0
                            else False
                        ),
                    )
                )

                # Handle special stage entry logic
                if target_stage == SeedStage.FOSSILIZED:
                    self._emit_telemetry(
                        TelemetryEventType.SEED_FOSSILIZED,
                        data=SeedFossilizedPayload(
                            slot_id=self.slot_id,
                            env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                            blueprint_id=blueprint_id,
                            improvement=improvement,
                            params_added=sum(
                                p.numel() for p in (self.seed.parameters() if self.seed is not None else []) if p.requires_grad
                            ),
                            alpha=self.state.alpha,
                            epochs_total=epochs_total,
                            counterfactual=counterfactual,
                        )
                    )
            else:
                gate_result = GateResult(
                    gate=gate_result.gate,
                    passed=False,
                    checks_failed=["transition_failed"],
                )

        return gate_result

    def prune(self, reason: str = "", *, initiator: str = "policy") -> bool:
        """Prune the current seed immediately (PRUNE_INSTANT).

        FOSSILIZED seeds cannot be pruned - they are permanent by design.
        The HOLDING stage exists as the last decision point before
        permanent integration. A future pruning subsystem will handle
        removal of non-performant fossilized nodes.

        Returns:
            True if prune succeeded, False if seed is unprunable (FOSSILIZED)
        """
        if not self.state:
            self._pending_prune_reason = None
            self._pending_prune_initiator = None
            return False

        # FOSSILIZED seeds are permanent - cannot be pruned
        if self.state.stage == SeedStage.FOSSILIZED:
            self._pending_prune_reason = None
            self._pending_prune_initiator = None
            return False

        # Capture metrics before transition clears state
        improvement = self.state.metrics.total_improvement
        counterfactual = self.state.metrics.counterfactual_contribution
        epochs_total = self.state.metrics.epochs_total
        epochs_in_stage = self.state.metrics.epochs_in_current_stage
        blueprint_id = self.state.blueprint_id

        # Track auto-prune status for degenerate policy detection.
        # Auto-prunes happen via environment safety mechanisms, not explicit RL actions.
        is_auto_prune = reason in SeedMetrics.AUTO_PRUNE_REASONS
        self.state.metrics.auto_pruned = is_auto_prune
        self.state.metrics.auto_prune_reason = reason if is_auto_prune else ""

        old_stage = self.state.stage
        if not self.state.transition(SeedStage.PRUNED):
            # Transition failed (shouldn't happen for non-FOSSILIZED)
            return False

        # =========================================================================
        # CAPTURE TELEMETRY DATA BEFORE FREEING MEMORY
        # =========================================================================
        # Capture all data needed for telemetry payloads upfront. This allows us
        # to free memory (self.seed = None) before emitting telemetry, ensuring
        # GPU memory is reclaimed even if telemetry has issues.
        alpha_for_telemetry = self.state.alpha
        alpha_curve_name = self.state.alpha_controller.alpha_curve.name
        grad_ratio = self._telemetry_grad_ratio()
        has_vanishing = (
            self.state.telemetry.has_vanishing
            if self.state.telemetry and self.state.telemetry.epoch > 0
            else False
        )
        has_exploding = (
            self.state.telemetry.has_exploding
            if self.state.telemetry and self.state.telemetry.epoch > 0
            else False
        )

        # =========================================================================
        # FREE MEMORY FIRST (BEFORE TELEMETRY)
        # =========================================================================
        # Release the seed nn.Module before any fallible operations. This ensures
        # GPU memory is reclaimed even if telemetry callbacks have issues.
        # Defense-in-depth: _emit_telemetry is now fault-tolerant, but we still
        # free memory first as a safety measure.
        self.seed = None
        self._shape_probe_cache.clear()
        self._cached_alpha_tensor = None

        # =========================================================================
        # EMIT TELEMETRY (NOW FAULT-TOLERANT)
        # =========================================================================
        self._emit_telemetry(
            TelemetryEventType.SEED_STAGE_CHANGED,
            data=SeedStageChangedPayload(
                slot_id=self.slot_id,
                env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                from_stage=old_stage.name,
                to_stage=SeedStage.PRUNED.name,
                alpha=alpha_for_telemetry,
                accuracy_delta=improvement,
                epochs_in_stage=epochs_in_stage,
                alpha_curve=alpha_curve_name,
                # Optional gradient health fields
                grad_ratio=grad_ratio,
                has_vanishing=has_vanishing,
                has_exploding=has_exploding,
            ),
        )
        self._emit_telemetry(
            TelemetryEventType.SEED_PRUNED,
            data=SeedPrunedPayload(
                slot_id=self.slot_id,
                env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                reason=reason,
                blueprint_id=blueprint_id,
                improvement=improvement,
                auto_pruned=is_auto_prune,
                epochs_total=epochs_total,
                counterfactual=counterfactual,
                initiator=initiator,
            )
        )

        # =========================================================================
        # FINAL CLEANUP
        # =========================================================================
        # Phase 4 contract: keep state after physical removal so PRUNED/EMBARGOED/
        # RESETTING are observable to masks + telemetry (anti-thrashing).
        #
        # Prune completion is equivalent to alpha==0 and controller HOLD at 0.
        self.set_alpha(0.0)
        self.state.alpha_controller.alpha_start = 0.0
        self.state.alpha_controller.alpha_target = 0.0
        self.state.alpha_controller.alpha_steps_total = 0
        self.state.alpha_controller.alpha_steps_done = 0
        self.state.alpha_controller.alpha_mode = AlphaMode.HOLD
        self.alpha_schedule = None
        self.isolate_gradients = False
        if self.isolation_monitor is not None:
            self.isolation_monitor.reset()
        # Clear pending async gradient stats
        self._pending_gradient_stats = None
        # Clear any BLEND_OUT freeze tracking (avoid keeping param refs alive)
        self._blend_out_freeze_active = False
        self._blend_out_frozen_params.clear()
        self._pending_prune_reason = None
        self._pending_prune_initiator = None
        return True

    def schedule_prune(
        self,
        *,
        steps: int,
        curve: AlphaCurve | None = None,
        steepness: float = 12.0,
        reason: str = "",
        initiator: str = "policy",
    ) -> bool:
        """Schedule a prune by ramping alpha down to 0 over N controller ticks."""
        if not self.state:
            return False

        if self.state.stage == SeedStage.FOSSILIZED:
            return False

        if steps <= 0 or self.alpha <= 0.0:
            return self.prune(reason=reason, initiator=initiator)

        if self.state.alpha_controller.alpha_mode != AlphaMode.HOLD:
            return False

        if self.state.stage == SeedStage.HOLDING:
            old_stage = self.state.stage
            metrics = self.state.metrics
            epochs_in_stage = metrics.epochs_in_current_stage
            improvement = metrics.total_improvement

            if not self.state.transition(SeedStage.BLENDING):
                raise RuntimeError(
                    f"Illegal lifecycle transition {old_stage} → BLENDING"
                )
            self._on_enter_stage(SeedStage.BLENDING, old_stage)
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data=SeedStageChangedPayload(
                    slot_id=self.slot_id,
                    env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                    from_stage=old_stage.name,
                    to_stage=SeedStage.BLENDING.name,
                    alpha=self.state.alpha,
                    accuracy_delta=improvement,
                    epochs_in_stage=epochs_in_stage,
                    alpha_curve=self.state.alpha_controller.alpha_curve.name,
                    # Optional gradient health fields
                    grad_ratio=self._telemetry_grad_ratio(),
                    has_vanishing=(
                        self.state.telemetry.has_vanishing
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                    has_exploding=(
                        self.state.telemetry.has_exploding
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                ),
            )
        elif self.state.stage != SeedStage.BLENDING:
            return self.prune(reason=reason, initiator=initiator)

        self._pending_prune_reason = reason
        self._pending_prune_initiator = initiator
        self.state.alpha_controller.retarget(
            alpha_target=0.0,
            alpha_steps_total=steps,
            alpha_curve=curve,
            alpha_steepness=steepness,
        )
        self._set_blend_out_freeze(True)
        return True

    def set_alpha_target(
        self,
        *,
        alpha_target: float,
        steps: int,
        curve: AlphaCurve | None = None,
        steepness: float = 12.0,
        alpha_algorithm: AlphaAlgorithm | None = None,
        initiator: str = "policy",
    ) -> bool:
        """Retarget alpha to a non-zero target from HOLD mode.

        Returns False for invalid requests (e.g., non-HOLD controller, zero target,
        or unsupported lifecycle stage).
        """
        if not self.state:
            return False

        if self.state.stage == SeedStage.FOSSILIZED:
            return False

        if alpha_target <= 0.0:
            return False

        controller = self.state.alpha_controller
        if controller.alpha_mode != AlphaMode.HOLD:
            return False

        if self.state.stage not in (SeedStage.BLENDING, SeedStage.HOLDING):
            return False

        # HOLDING is full-amplitude only; partial targets must re-enter BLENDING.
        if self.state.stage == SeedStage.HOLDING and alpha_target < 1.0 - 1e-6:
            old_stage = self.state.stage
            metrics = self.state.metrics
            epochs_in_stage = metrics.epochs_in_current_stage
            improvement = metrics.total_improvement

            if not self.state.transition(SeedStage.BLENDING):
                raise RuntimeError(
                    f"Illegal lifecycle transition {old_stage} → BLENDING"
                )
            self._on_enter_stage(SeedStage.BLENDING, old_stage)
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data=SeedStageChangedPayload(
                    slot_id=self.slot_id,
                    env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                    from_stage=old_stage.name,
                    to_stage=SeedStage.BLENDING.name,
                    alpha=self.state.alpha,
                    accuracy_delta=improvement,
                    epochs_in_stage=epochs_in_stage,
                    alpha_curve=self.state.alpha_controller.alpha_curve.name,
                    # Optional gradient health fields
                    grad_ratio=self._telemetry_grad_ratio(),
                    has_vanishing=(
                        self.state.telemetry.has_vanishing
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                    has_exploding=(
                        self.state.telemetry.has_exploding
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                ),
            )

        # Update alpha algorithm if requested (HOLD-only changes).
        if alpha_algorithm is not None and alpha_algorithm != self.state.alpha_algorithm:
            if alpha_algorithm == AlphaAlgorithm.GATE:
                if self.alpha_schedule is None:
                    from esper.kasmina.blending import BlendCatalog

                    topology = self._resolve_topology()
                    total_steps = max(1, int(steps))
                    # BlendCatalog.create returns BlendAlgorithm which satisfies AlphaScheduleProtocol
                    created_schedule = BlendCatalog.create(
                        "gated",
                        channels=self.channels,
                        topology=topology,
                        total_steps=total_steps,
                    )
                    if isinstance(created_schedule, nn.Module):
                        self.alpha_schedule = created_schedule.to(self.device)  # type: ignore[assignment]
                    else:
                        self.alpha_schedule = created_schedule
            else:
                # Non-gated algorithms must not carry a gate schedule.
                self.alpha_schedule = None
            self.state.alpha_algorithm = alpha_algorithm

        controller.retarget(
            alpha_target=alpha_target,
            alpha_steps_total=steps,
            alpha_curve=curve,
            alpha_steepness=steepness,
        )

        # Freeze learnable params when blending down.
        self._set_blend_out_freeze(controller.alpha_mode == AlphaMode.DOWN)  # type: ignore[comparison-overlap]
        return True

    def capture_gradient_telemetry(self) -> None:
        """Calculate gradient norms via the health monitor and update internal metrics.

        CRITICAL: Call this from Tolaria after loss.backward() to enable the G2 gate.
        Without this, seed_gradient_norm_ratio remains None and G2 fails with
        checks_failed=["gradient_stats_never_measured"] in strict gate mode.

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

        # materialize_gradient_stats() guarantees these keys exist
        host_norm = stats["host_grad_norm"]
        seed_norm = stats["seed_grad_norm"]

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
        assert self.isolation_monitor is not None  # Checked above
        stats = self.isolation_monitor.materialize_gradient_stats(self._pending_gradient_stats)
        self._pending_gradient_stats = None

        # Same ratio computation logic as capture_gradient_telemetry()
        # materialize_gradient_stats() guarantees these keys exist
        host_norm = stats["host_grad_norm"]
        seed_norm = stats["seed_grad_norm"]

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

    def forward(self, host_features: torch.Tensor, alpha_override: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass applying the seed's transformation to host features.

        Args:
            host_features: Input features from the host backbone.
            alpha_override: Optional tensor of shape [B, 1, 1, 1] (or similar) to override
                this slot's scalar alpha for fused multi-configuration passes.

        Returns:
            Blended feature map.
        """
        # 1. NO-OP: Slot is dormant or in failure stage - return host features unchanged
        if not self.is_active or self.seed is None or self.state is None or not is_active_stage(self.state.stage):
            return host_features

        # 2. ISOLATION: Detach host input into the seed path if requested.
        #    This prevents seed gradients from backpropagating into the host.
        #    We preserve memory format (often: channels_last) to preserve host CNN throughput.
        #
        # PERF FIX (BUG-005): Preserve channels_last output under isolation.
        # Avoid coercing host_features to contiguous_format; instead feed the seed a
        # contiguous_format detached copy so backward never sees channels_last + detach.
        if self.isolate_gradients:
            seed_input = (
                host_features.detach()
                if host_features.is_contiguous()
                else host_features.contiguous().detach()
            )
        else:
            seed_input = host_features
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
        if alpha_override is None and self.state.stage == SeedStage.TRAINING and self.alpha == 0.0:
            if _DEBUG_STE:
                assert seed_features.requires_grad, (
                    "STE requires seed_features to have requires_grad=True for gradient flow"
                )
            return ste_forward(host_features, seed_features)

        # 4. BLENDING and later stages: standard lerp with proper gradient flow.
        
        # PERF: Use persistent alpha tensor to avoid per-forward allocations.
        # If alpha_override is provided, we use it directly (already a tensor).
        if alpha_override is not None:
            alpha_amplitude = alpha_override
        else:
            # Cache alpha tensor for torch.compile-friendly blending.
            # The cache is invalidated when alpha changes (see set_alpha, advance_stage, etc.)
            # so we only need to check device/dtype here - no fill_() which breaks compile.
            if (
                self._cached_alpha_tensor is None
                or self._cached_alpha_tensor.device != host_features.device
                or self._cached_alpha_tensor.dtype != host_features.dtype
            ):
                self._cached_alpha_tensor = torch.tensor(
                    self.alpha, device=host_features.device, dtype=host_features.dtype
                )
            alpha_amplitude = self._cached_alpha_tensor

        match self.state.alpha_algorithm:
            case AlphaAlgorithm.ADD:
                if self.alpha_schedule is not None:
                    raise RuntimeError("alpha_schedule is reserved for GATE only (Phase 3+).")
                return blend_with_isolation(host_features, seed_features, alpha_amplitude)
            case AlphaAlgorithm.MULTIPLY:
                if self.alpha_schedule is not None:
                    raise RuntimeError("alpha_schedule is reserved for GATE only (Phase 3+).")
                return blend_multiply(
                    host_features,
                    seed_features,
                    alpha_amplitude,
                    seed_input=seed_input,
                )
            case AlphaAlgorithm.GATE:
                if self.alpha_schedule is None:
                    raise RuntimeError("alpha_schedule is required when alpha_algorithm=GATE.")
                # Isolation contract: use the same input reference that the seed sees.
                gate = self.alpha_schedule.get_alpha_for_blend(seed_input)
                return blend_gate(host_features, seed_features, alpha_amplitude, gate)
            case algo:
                raise ValueError(f"Unknown alpha_algorithm: {algo!r}")

    def get_parameters(self) -> Generator[torch.nn.Parameter, None, None]:
        """Yield trainable parameters for the seed (and alpha_schedule when present)."""
        if self.seed is not None:
            yield from self.seed.parameters()
        if self.alpha_schedule is not None and isinstance(self.alpha_schedule, nn.Module):
            yield from self.alpha_schedule.parameters()

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

    def _set_blend_out_freeze(self, enabled: bool) -> None:
        """Freeze/unfreeze learnable params during BLEND_OUT (alpha_mode == DOWN).

        Freeze semantics are param-only (requires_grad=False). The forward graph
        MUST remain intact to preserve "ghost gradients" to the host.
        """
        if enabled == self._blend_out_freeze_active:
            return

        if enabled:
            self._blend_out_frozen_params.clear()
            for module in (self.seed, self.alpha_schedule):
                if module is None:
                    continue
                # alpha_schedule may be a Protocol, check if it's actually a Module
                if isinstance(module, nn.Module):
                    for param in module.parameters():
                        if param.requires_grad:
                            param.requires_grad_(False)
                            self._blend_out_frozen_params.append(param)
            self._blend_out_freeze_active = True
            return

        # Restore exactly the params that were trainable before freeze.
        for param in self._blend_out_frozen_params:
            param.requires_grad_(True)
        self._blend_out_frozen_params.clear()
        self._blend_out_freeze_active = False

    def start_blending(self, total_steps: int, steepness: float = 12.0) -> None:
        """Initialize blending with selected algorithm.

        Uses blend_algorithm_id set during germinate(). Falls back to sigmoid
        if not specified.

        Phase 2 contract: alpha amplitude is scheduled exclusively by
        AlphaController. alpha_schedule is reserved for per-sample gating
        (currently: "gated").
        """
        from esper.kasmina.blending import BlendCatalog

        algorithm_id = self._blend_algorithm_id or "sigmoid"

        # Initialize blending progress tracking
        if self.state:
            # Phase 3 contract: alpha_schedule is reserved for per-sample gating only.
            # We treat "gated" as enabling the GATE alpha_algorithm.
            if algorithm_id == "gated":
                self.state.alpha_algorithm = AlphaAlgorithm.GATE
            elif self.state.alpha_algorithm == AlphaAlgorithm.GATE:
                raise ValueError("alpha_algorithm=GATE requires blend_algorithm_id='gated'")

            match algorithm_id:
                case "linear":
                    curve = AlphaCurve.LINEAR
                case "sigmoid":
                    curve = AlphaCurve.SIGMOID
                case "gated":
                    # Design Decision: Enforce LINEAR amplitude ramp for gated blending.
                    #
                    # Rationale:
                    # 1. Avoids "Double Dynamics": The learned gate `gate(x)` is already non-linear
                    #    and starts near zero. Adding a Sigmoid amplitude ramp (which also stays
                    #    near zero) creates a "compound silence" that starves the gate of gradients.
                    # 2. Predictable Ceiling: A linear ramp provides a steady, predictable increase
                    #    in the *maximum possible influence* (allowance), putting the onus on the
                    #    gate network to modulate the effective alpha.
                    #
                    # Future: If early-phase instability occurs, we may expose curve selection,
                    # but Linear is the safer default to prevent "silent death" of gates.
                    curve = AlphaCurve.LINEAR
                case _:
                    raise ValueError(
                        f"Unknown blend algorithm: {algorithm_id}. "
                        f"Valid options: linear, sigmoid, gated"
                    )
            # Alpha amplitude scheduling is handled by AlphaController.
            alpha_target = self._blend_alpha_target
            if alpha_target is None:
                alpha_target = 1.0
            self.state.alpha_controller = AlphaController(alpha=self.state.alpha)
            self.state.alpha_controller.retarget(
                alpha_target=alpha_target,
                alpha_steps_total=total_steps,
                alpha_curve=curve,
                alpha_steepness=steepness,
            )

        topology = self._resolve_topology()

        # Create blend algorithm with appropriate kwargs.
        #
        # Phase 2: linear/sigmoid are curves for AlphaController only, so we do
        # not create an alpha_schedule module for them. We keep alpha_schedule
        # only for per-sample gating so its parameters are tracked and checkpointed.
        self.alpha_schedule = None
        if algorithm_id == "gated":
            # BlendCatalog.create returns BlendAlgorithm which satisfies AlphaScheduleProtocol
            created_schedule = BlendCatalog.create(
                algorithm_id, channels=self.channels, topology=topology, total_steps=total_steps
            )
            if isinstance(created_schedule, nn.Module):
                self.alpha_schedule = created_schedule.to(self.device)  # type: ignore[assignment]
            else:
                self.alpha_schedule = created_schedule

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
            topology = self._resolve_topology()
            self.isolate_gradients = (topology == "cnn")

            # Snapshot accuracy at blending start for true causal attribution
            if self.state:
                self.state.metrics.accuracy_at_blending_start = self.state.metrics.current_val_accuracy
                self.state.metrics._blending_started = True

            # Initialize blending schedule
            # Priority: stored tempo > TaskConfig > DEFAULT_BLENDING_TOTAL_STEPS
            total_steps = self._blend_tempo_epochs
            if total_steps is None:
                total_steps = DEFAULT_BLENDING_TOTAL_STEPS
                if self.task_config is not None:
                    configured_steps = self.task_config.blending_steps
                    if isinstance(configured_steps, int) and configured_steps > 0:
                        total_steps = configured_steps
            self.start_blending(total_steps=total_steps)

        elif new_stage == SeedStage.HOLDING and old_stage == SeedStage.BLENDING:
            # Clean up blending resources
            self._on_blending_complete()

    def _on_blending_complete(self) -> None:
        """Clean up after BLENDING stage completes.

        Discards alpha_schedule for non-gated algorithms.
        Sets state.alpha = 1.0 (permanently fully blended).
        """
        if self.state is None:
            self.alpha_schedule = None
            return

        if self.state.alpha_algorithm != AlphaAlgorithm.GATE:
            self.alpha_schedule = None

        self.set_alpha(1.0)
        self.state.alpha_controller.alpha_start = 1.0
        self.state.alpha_controller.alpha_target = 1.0
        self.state.alpha_controller.alpha_mode = AlphaMode.HOLD
        self.state.alpha_controller.alpha_steps_done = self.state.alpha_controller.alpha_steps_total

    def _sync_gate_decision(self, gate_result: GateResult) -> GateResult:
        """Ensure all DDP ranks agree on lifecycle transitions via rank-0 broadcast.

        Rank 0 makes the authoritative gate decision and broadcasts it to all ranks.
        This design (vs. all_reduce consensus) avoids JANK-003 deadlocks when ranks
        have seeds at different stages, because rank-0's decision is always used
        regardless of other ranks' local evaluations.

        This prevents Architecture Divergence, where ranks typically crash
        if parameter shapes mismatch during the next forward pass.

        COLLECTIVE OPERATION: All ranks MUST call this in identical order
        for each slot's gate check, or deadlock will occur. The training loop
        must ensure symmetric advance_stage() calls across ranks.

        Returns:
            GateResult from rank 0, broadcast to all ranks.
        """
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return gate_result

        rank = torch.distributed.get_rank()

        # Serialize GateResult for broadcast
        # We broadcast a minimal dict to avoid pickle overhead on full dataclass
        if rank == 0:
            sync_data = {
                "gate": gate_result.gate.value,  # GateLevel enum value
                "passed": gate_result.passed,
                "score": gate_result.score,
                "checks_passed": list(gate_result.checks_passed),
                "checks_failed": list(gate_result.checks_failed),
                "message": gate_result.message,
            }
        else:
            sync_data = None

        # broadcast_object_list requires a list; rank 0 sends, others receive
        object_list: list[dict[str, Any] | None] = [sync_data]
        torch.distributed.broadcast_object_list(object_list, src=0)
        synced_data = object_list[0]

        # Reconstruct GateResult from broadcast data
        assert synced_data is not None, "Broadcast should always succeed with rank 0 data"
        synced_result = GateResult(
            gate=GateLevel(int(synced_data["gate"])),
            passed=bool(synced_data["passed"]),
            score=float(synced_data["score"]),
            checks_passed=list(synced_data["checks_passed"]),
            checks_failed=list(synced_data["checks_failed"]),
            message=str(synced_data["message"]),
        )

        # Track divergence for debugging (non-rank-0 had different local result)
        if rank != 0 and gate_result.passed != synced_result.passed:
            divergence_note = f" (local={gate_result.passed}, synced from rank0={synced_result.passed})"
            synced_result = GateResult(
                gate=synced_result.gate,
                passed=synced_result.passed,
                score=synced_result.score,
                checks_passed=synced_result.checks_passed,
                checks_failed=synced_result.checks_failed,
                message=synced_result.message + divergence_note,
            )

        return synced_result

    def step_epoch(self) -> None:
        """Advance lifecycle mechanics once per epoch (Kasmina timekeeper).

        Auto-prune events are signaled via `state.metrics.auto_pruned`.
        Callers should typically check (and clear) this flag immediately
        AFTER calling `step_epoch()` to catch both:
        - governor/rollback prunes (flag set outside `step_epoch()`)
        - scheduled prune completion (flag set inside `step_epoch()`)

        By default, stage advancement is explicit via advance_stage(); step_epoch
        only ticks alpha schedules and cooldown transitions. When auto_forward_gates
        is configured, step_epoch will also advance through those gated transitions.
        """
        if not self.state:
            return

        stage = self.state.stage

        # Phase 3 invariant: during BLEND_OUT (alpha_mode == DOWN) we freeze all
        # learnable params that could "fight" decay, but keep the forward graph
        # intact so host gradients still flow ("ghost gradients").
        self._set_blend_out_freeze(self.state.alpha_controller.alpha_mode == AlphaMode.DOWN)

        # Auto-forward: GERMINATED -> TRAINING (G1)
        if stage == SeedStage.GERMINATED:
            if GateLevel.G1 in self.auto_forward_gates:
                self.advance_stage(SeedStage.TRAINING)
            return

        # Auto-forward: TRAINING -> BLENDING (G2)
        # NOTE: Both permissive and strict modes require min_blending_epochs.
        # In permissive mode, G2 gate also checks gradient health and no exploding.
        if stage == SeedStage.TRAINING:
            if GateLevel.G2 in self.auto_forward_gates:
                min_epochs = self.gates.min_blending_epochs
                if self.state.metrics.epochs_in_current_stage >= min_epochs:
                    self.advance_stage(SeedStage.BLENDING)
            return

        # BLENDING: tick alpha controller and enforce scheduled prune completion
        if stage == SeedStage.BLENDING:
            controller = self.state.alpha_controller
            reached_target = controller.step()
            self.set_alpha(controller.alpha)
            self.state.metrics.alpha_ramp_step = controller.alpha_steps_done

            target_reached = (
                reached_target
                or (
                    controller.alpha_mode == AlphaMode.HOLD
                    and abs(controller.alpha - controller.alpha_target) <= 1e-6
                )
            )

            if target_reached:
                if controller.alpha_target <= 1e-6:
                    reason = self._pending_prune_reason or "scheduled_prune"
                    initiator = self._pending_prune_initiator or "policy"
                    self.prune(reason=reason, initiator=initiator)
                    return

                if GateLevel.G3 in self.auto_forward_gates:
                    if self.gates.permissive or (
                        self.state.metrics.epochs_in_current_stage
                        >= self.gates.min_blending_epochs
                    ):
                        self.advance_stage(SeedStage.HOLDING)
            return

        # HOLDING: Decision point for Tamiyo
        # Fossilization requires explicit FOSSILIZE action - NO auto-advance
        # (DRL Expert review 2025-12-10: auto-fossilize violated credit assignment)
        # Phase 4: No auto-prunes here; policy/governor must decide explicitly.
        if stage == SeedStage.HOLDING:
            return

        # Phase 4: cooldown pipeline after pruning (seed removed, state retained).
        #
        # IMPORTANT: In PRUNED/EMBARGOED/RESETTING the seed is not active, so
        # SeedMetrics.record_accuracy() is not called. We therefore advance the
        # stage tick counters here (per step_epoch call) so the dwell completes.
        if stage == SeedStage.PRUNED:
            old_stage = self.state.stage
            metrics = self.state.metrics
            epochs_in_stage = metrics.epochs_in_current_stage

            ok = self.state.transition(SeedStage.EMBARGOED)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition {old_stage} → EMBARGOED"
                )
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data=SeedStageChangedPayload(
                    slot_id=self.slot_id,
                    env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                    from_stage=old_stage.name,
                    to_stage=self.state.stage.name,
                    alpha=self.state.alpha,
                    accuracy_delta=0.0,
                    epochs_in_stage=epochs_in_stage,
                    alpha_curve=self.state.alpha_controller.alpha_curve.name,
                    # Optional gradient health fields
                    grad_ratio=self._telemetry_grad_ratio(),
                    has_vanishing=(
                        self.state.telemetry.has_vanishing
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                    has_exploding=(
                        self.state.telemetry.has_exploding
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                ),
            )
            return

        if stage == SeedStage.EMBARGOED:
            # Advance dwell tick (not tied to validation accuracy).
            self.state.metrics.epochs_total += 1
            self.state.metrics.epochs_in_current_stage += 1

            embargo_epochs = DEFAULT_EMBARGO_EPOCHS_AFTER_PRUNE
            if self.state.metrics.epochs_in_current_stage < embargo_epochs:
                return

            old_stage = self.state.stage
            metrics = self.state.metrics
            epochs_in_stage = metrics.epochs_in_current_stage

            ok = self.state.transition(SeedStage.RESETTING)
            if not ok:
                raise RuntimeError(
                    f"Illegal lifecycle transition {old_stage} → RESETTING"
                )
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data=SeedStageChangedPayload(
                    slot_id=self.slot_id,
                    env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                    from_stage=old_stage.name,
                    to_stage=self.state.stage.name,
                    alpha=self.state.alpha,
                    accuracy_delta=0.0,
                    epochs_in_stage=epochs_in_stage,
                    alpha_curve=self.state.alpha_controller.alpha_curve.name,
                    # Optional gradient health fields
                    grad_ratio=self._telemetry_grad_ratio(),
                    has_vanishing=(
                        self.state.telemetry.has_vanishing
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                    has_exploding=(
                        self.state.telemetry.has_exploding
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                ),
            )
            return

        if stage == SeedStage.RESETTING:
            self.state.metrics.epochs_total += 1
            self.state.metrics.epochs_in_current_stage += 1

            # Resetting is a short cleanup dwell; keep it 1 tick for now.
            if self.state.metrics.epochs_in_current_stage < 1:
                return

            old_stage = self.state.stage
            metrics = self.state.metrics
            epochs_in_stage = metrics.epochs_in_current_stage

            # Emit explicit "back to DORMANT" transition for telemetry/UI, then
            # clear state so slot_reports treat the slot as empty.
            self._emit_telemetry(
                TelemetryEventType.SEED_STAGE_CHANGED,
                data=SeedStageChangedPayload(
                    slot_id=self.slot_id,
                    env_id=-1,  # Sentinel - will be replaced by emit_with_env_context
                    from_stage=old_stage.name,
                    to_stage=SeedStage.DORMANT.name,
                    alpha=self.state.alpha,
                    accuracy_delta=0.0,
                    epochs_in_stage=epochs_in_stage,
                    alpha_curve=self.state.alpha_controller.alpha_curve.name,
                    # Optional gradient health fields
                    grad_ratio=self._telemetry_grad_ratio(),
                    has_vanishing=(
                        self.state.telemetry.has_vanishing
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                    has_exploding=(
                        self.state.telemetry.has_exploding
                        if self.state.telemetry and self.state.telemetry.epoch > 0
                        else False
                    ),
                ),
            )

            # Invariant: when a slot returns to DORMANT (state cleared), it must be
            # physically empty (seed removed). Normal prune() clears the seed at
            # PRUNED entry, but rollbacks can restore lifecycle state without
            # removing an experimental seed module (BUG-ROLLBACK-ORPHAN).
            self.seed = None
            self.alpha_schedule = None
            self._shape_probe_cache.clear()
            self._cached_alpha_tensor = None
            if self.isolation_monitor is not None:
                self.isolation_monitor.reset()
            self._pending_gradient_stats = None
            self._blend_out_freeze_active = False
            self._blend_out_frozen_params.clear()
            self.state = None

    def get_state_report(self) -> SeedStateReport | None:
        """Get current state as Leyline report."""
        if not self.state:
            return None
        return self.state.to_report()

    def _telemetry_grad_ratio(self) -> float:
        """Return a float grad_ratio for telemetry payloads.

        Internal SeedMetrics uses None to mean "never measured". Telemetry payload
        schemas use float defaults, so we coalesce None -> 0.0.
        """
        if self.state is None or self.state.telemetry is None or self.state.telemetry.epoch <= 0:
            return 0.0
        ratio = self.state.metrics.seed_gradient_norm_ratio
        if ratio is None:
            return 0.0
        return float(ratio)

    def _emit_telemetry(
        self,
        event_type: TelemetryEventType,
        data: SeedGerminatedPayload | SeedStageChangedPayload | SeedGateEvaluatedPayload | SeedFossilizedPayload | SeedPrunedPayload,
    ) -> None:
        """Emit a telemetry event with a typed payload.

        Skipped entirely in fast_mode for zero overhead in PPO rollouts.
        All payloads are typed dataclasses - dicts are not supported.

        SAFETY INVARIANT: Telemetry is observability, not correctness.
        If the callback raises (disk full, serialization error, etc.),
        we log to stderr but do NOT abort the calling operation. This
        ensures safety-critical paths (e.g., Governor rollback) complete
        even when telemetry subsystems fail. See governor.py for the
        same pattern applied to Governor's own telemetry emission.
        """
        if self.on_telemetry is None:
            return
        if self.fast_mode and not self.telemetry_lifecycle_only:
            return

        event = TelemetryEvent(
            event_type=event_type,
            seed_id=self.state.seed_id if self.state else None,
            slot_id=self.slot_id,
            epoch=self.telemetry_global_epoch,
            data=data,
        )
        try:
            self.on_telemetry(event)
        except Exception as e:
            # Telemetry callback failed - log but don't abort.
            # This prevents observability failures from blocking
            # safety-critical operations like Governor rollback.
            import sys
            print(
                f"WARNING: Seed telemetry emission failed for {event_type.name} "
                f"(slot={self.slot_id}): {e}",
                file=sys.stderr,
            )

    def get_extra_state(self) -> dict[str, Any]:
        """Persist SeedState for PyTorch 2.9+ weights_only=True compatibility.

        Returns only primitive types (dict, list, str, int, float, bool, None).
        The alpha_schedule nn.Module weights are saved via state_dict(), not here.
        """
        state_dict: dict[str, Any] = {
            "_extra_state_version": self._EXTRA_STATE_VERSION,
            "isolate_gradients": self.isolate_gradients,
            "blend_algorithm_id": self._blend_algorithm_id,
            "blend_tempo_epochs": self._blend_tempo_epochs,
            "blend_alpha_target": self._blend_alpha_target,
            "resolved_topology": self._resolved_topology,
        }

        # Always include seed_state key for checkpoint symmetry.
        # DORMANT slots have state=None, which is valid and must roundtrip correctly.
        state_dict["seed_state"] = self.state.to_dict() if self.state is not None else None

        # Alpha schedule: save config only, not the nn.Module
        # The nn.Module weights are saved in state_dict() automatically
        if self.alpha_schedule is not None:
            # Contract: all alpha_schedule objects must satisfy AlphaScheduleProtocol
            alpha_schedule_config: dict[str, Any] = {
                "algorithm_id": self.alpha_schedule.algorithm_id,
                "total_steps": self.alpha_schedule.total_steps,
                "current_step": self.alpha_schedule._current_step,
            }
            state_dict["alpha_schedule_config"] = alpha_schedule_config
        else:
            state_dict["alpha_schedule_config"] = None

        return state_dict

    def set_extra_state(self, state: dict[str, Any]) -> None:
        """Restore SeedSlot extra state from checkpoint.

        All fields saved by get_extra_state() are required. Missing fields
        indicate a corrupt or incompatible checkpoint and will raise KeyError.

        Raises:
            KeyError: If required field is missing (corrupt checkpoint).
            ValueError: If schema version mismatch or invalid field values.
        """
        # Schema version validation (fail-fast on incompatible checkpoints)
        version = state["_extra_state_version"]
        if version != self._EXTRA_STATE_VERSION:
            raise ValueError(
                f"SeedSlot extra_state schema mismatch: expected v{self._EXTRA_STATE_VERSION}, "
                f"got v{version}. Checkpoint may be from incompatible version."
            )

        # Required fields - KeyError if missing (corrupt checkpoint)
        self.isolate_gradients = state["isolate_gradients"]
        self._blend_algorithm_id = state["blend_algorithm_id"]
        self._blend_tempo_epochs = state["blend_tempo_epochs"]
        self._blend_alpha_target = state["blend_alpha_target"]
        self._resolved_topology = state["resolved_topology"]

        # seed_state is a required key; value may be None for DORMANT slots
        seed_state = state["seed_state"]
        if seed_state is not None:
            self.state = SeedState.from_dict(seed_state)

        # Alpha schedule reconstruction
        # The nn.Module weights are restored via load_state_dict() automatically
        # because PyTorch 2.x includes dynamically assigned modules in state_dict.
        # We only need to restore config and ensure the correct algorithm type.
        alpha_config = state["alpha_schedule_config"]
        if alpha_config is not None:
            algorithm_id = alpha_config["algorithm_id"]
            total_steps = alpha_config["total_steps"]
            current_step = alpha_config["current_step"]

            if self.state is None:
                raise ValueError("Checkpoint contains alpha_schedule_config but seed_state is missing.")
            if algorithm_id != "gated":
                raise ValueError(
                    "Checkpoint contains legacy alpha_schedule_config for "
                    f"algorithm_id={algorithm_id!r}. "
                    "Phase 2+ only supports alpha_schedule_config for 'gated'."
                )
            if self.state.alpha_algorithm != AlphaAlgorithm.GATE:
                raise ValueError(
                    "Checkpoint contains alpha_schedule_config for 'gated' but "
                    f"alpha_algorithm={self.state.alpha_algorithm!r}."
                )

            # CRITICAL: Restore algorithm_id BEFORE start_blending()
            # Without this, start_blending() defaults to "sigmoid" and
            # GatedBlend weights become orphaned "unexpected_keys".
            # See: docs/plans/2025-12-16-tolaria-kasmina-remediation.md
            self._blend_algorithm_id = algorithm_id
            self.start_blending(total_steps=total_steps)

            if self.alpha_schedule is None:
                raise RuntimeError(
                    "start_blending() did not create alpha_schedule. Checkpoint may be corrupt."
                )

            self.alpha_schedule._current_step = current_step

            # Re-restore alpha controller (start_blending resets it)
            alpha_controller = state["seed_state"]["alpha_controller"]
            if not isinstance(alpha_controller, dict):
                raise ValueError(
                    "Checkpoint seed_state is missing required 'alpha_controller' (Phase 1+). "
                    "Pre-Phase-1 checkpoints are not supported for resume."
                )
            self.state.alpha_controller = AlphaController.from_dict(alpha_controller)
            self.state.alpha_controller.alpha = self.state.alpha

        # Ensure BLEND_OUT freeze invariant holds immediately after checkpoint load.
        if self.state is not None:
            self._set_blend_out_freeze(self.state.alpha_controller.alpha_mode == AlphaMode.DOWN)


__all__ = [
    "SeedMetrics",
    "SeedState",
    "QualityGates",
    "SeedSlot",
]
