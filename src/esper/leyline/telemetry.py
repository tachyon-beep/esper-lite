"""Leyline Telemetry Contracts - Event definitions for system monitoring.

These are CONTRACTS only - the actual emission and handling
is done by domain-specific modules and Nissa.

Each domain emits events using these contracts:
- Kasmina: Seed lifecycle events
- Tamiyo: Decision events
- Simic: Training progress events
- Nissa: System health events
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any
from uuid import uuid4

from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.stages import SeedStage

# =============================================================================
# Type Conversion Helpers
# =============================================================================

from typing import TypeVar

_T = TypeVar("_T")


def _ensure_tuple(value: list[_T] | tuple[_T, ...]) -> tuple[_T, ...]:
    """Convert list to tuple for JSON deserialization.

    JSON doesn't distinguish tuples from lists, so after deserialization
    tuple fields will be lists. This helper normalizes them back to tuples.
    """
    return tuple(value) if isinstance(value, list) else value


# Feature normalization constants for RL observation space
# These define the expected ranges for seed telemetry values
_GRADIENT_NORM_MAX: float = 10.0  # 99th percentile typical gradient norm
_EPOCHS_IN_STAGE_MAX: int = 50  # Typical max epochs in single stage
_ACCURACY_DELTA_SCALE: float = 10.0  # Scale factor for accuracy deltas
_ALPHA_MODE_MAX: int = max(mode.value for mode in AlphaMode)
_ALPHA_ALGO_MIN: int = min(algo.value for algo in AlphaAlgorithm)
_ALPHA_ALGO_MAX: int = max(algo.value for algo in AlphaAlgorithm)


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class TelemetryEventType(Enum):
    """Types of telemetry events."""

    # Lifecycle events
    SEED_GERMINATED = auto()
    SEED_STAGE_CHANGED = auto()
    SEED_GATE_EVALUATED = auto()
    SEED_FOSSILIZED = auto()
    SEED_PRUNED = auto()

    # Training events
    EPOCH_COMPLETED = auto()  # Per-env epoch (has env_id in data)
    BATCH_EPOCH_COMPLETED = auto()  # Batch-level epoch summary + progress (commit barrier)
    PLATEAU_DETECTED = auto()
    DEGRADATION_DETECTED = auto()
    IMPROVEMENT_DETECTED = auto()
    TAMIYO_INITIATED = auto()  # Host stabilized, germination now allowed

    # Health events
    # TODO: [DEAD CODE] - ISOLATION_VIOLATION is defined but never emitted or handled.
    # Appears to be planned functionality that was never implemented. Delete or implement.
    ISOLATION_VIOLATION = auto()
    GRADIENT_ANOMALY = auto()
    PERFORMANCE_DEGRADATION = auto()

    # === NEW: PPO Training Events (Ops Normal) ===
    PPO_UPDATE_COMPLETED = auto()
    MEMORY_WARNING = auto()
    REWARD_HACKING_SUSPECTED = auto()
    # TODO: [DEAD CODE] - REWARD_COMPUTED is defined, tested, and handled by Karn,
    # but NEVER emitted in production. Reward data flows via ANALYTICS_SNAPSHOT instead.
    # Either emit this event from vectorized.py or remove it. See: risk assessment 2024-12-24.
    REWARD_COMPUTED = auto()  # Per-step reward breakdown for debugging

    # === NEW: Debug Events (triggered by anomalies) ===
    RATIO_EXPLOSION_DETECTED = auto()
    RATIO_COLLAPSE_DETECTED = auto()
    VALUE_COLLAPSE_DETECTED = auto()
    GRADIENT_PATHOLOGY_DETECTED = auto()
    NUMERICAL_INSTABILITY_DETECTED = auto()

    # === Governor Events (Tolaria) ===
    # TODO: [DEAD CODE] - GOVERNOR_PANIC is defined and has console formatting in nissa/output.py,
    # but Governor only emits GOVERNOR_ROLLBACK. Either emit this or remove handler code.
    GOVERNOR_PANIC = auto()           # Vital signs check failed
    GOVERNOR_ROLLBACK = auto()        # Emergency rollback executed
    # TODO: [DEAD CODE] - GOVERNOR_SNAPSHOT is defined but never emitted or handled.
    # Appears to be planned functionality that was never implemented. Delete or implement.
    GOVERNOR_SNAPSHOT = auto()        # LKG checkpoint saved

    # === Training Progress Events ===
    TRAINING_STARTED = auto()         # Training run initialized
    # TODO: [DEAD CODE] - CHECKPOINT_SAVED is defined and has console formatting,
    # but checkpoint saves in vectorized.py don't emit this event. Either emit or remove.
    CHECKPOINT_SAVED = auto()         # Model checkpoint saved
    CHECKPOINT_LOADED = auto()        # Model checkpoint restored

    # === Counterfactual Attribution Events ===
    COUNTERFACTUAL_COMPUTED = auto()  # Per-slot counterfactual contribution measured
    COUNTERFACTUAL_MATRIX_COMPUTED = auto()  # Full factorial matrix for env

    # === Analytics Events ===
    ANALYTICS_SNAPSHOT = auto()       # Full state snapshot for dashboard sync


@dataclass
class TelemetryEvent:
    """A telemetry event for observability."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: TelemetryEventType = TelemetryEventType.EPOCH_COMPLETED
    timestamp: datetime = field(default_factory=_utc_now)

    # Context
    seed_id: str | None = None
    slot_id: str | None = None
    epoch: int | None = None
    group_id: str = "default"  # A/B testing group identifier (e.g., "A", "B")

    # Event data - TYPED PAYLOAD (replaces untyped dict)
    # See: docs/plans/2025-12-25-typed-telemetry-payloads-design.md
    message: str = ""
    data: "TelemetryPayload | None" = None

    # Severity
    severity: str = "info"  # debug, info, warning, error, critical


# TODO: [DEAD CODE] - PerformanceBudgets and DEFAULT_BUDGETS are defined but never used
# anywhere in production. Either integrate into training pipeline or delete.
# See: architectural risk assessment 2024-12-24.
@dataclass(frozen=True)
class PerformanceBudgets:
    """Performance budget constants."""

    # Timing budgets (milliseconds)
    epoch_budget_ms: float = 18.0
    blending_budget_ms: float = 5.0
    gate_check_budget_ms: float = 2.0

    # Memory budgets (GB)
    seed_memory_budget_gb: float = 2.0
    total_memory_budget_gb: float = 12.0

    # Serialization budgets
    max_message_size_bytes: int = 280
    serialization_budget_us: float = 80.0


# Default budgets
DEFAULT_BUDGETS = PerformanceBudgets()


# =============================================================================
# Seed Telemetry Snapshot
# =============================================================================

@dataclass(slots=True)
class SeedTelemetry:
    """Per-seed telemetry snapshot - the seed's 'local picture'.

    Contract between seed implementations (Kasmina/Simic) and
    decision-makers (Tamiyo). Designed for:
    - Single seed (current): one instance
    - Multi-seed (future): collection managed by registry
    - Hierarchical (stretch): tactical aggregates for strategic

    Note: Uses slots=True for memory efficiency in multi-seed scenarios.
    """

    seed_id: str
    blueprint_id: str = ""
    layer_id: str = ""

    # Health signals (lightweight, always collected)
    gradient_norm: float = 0.0
    gradient_health: float = 1.0  # 0-1, higher is healthier
    has_vanishing: bool = False
    has_exploding: bool = False

    # Progress signals
    accuracy: float = 0.0  # percentage (0-100)
    accuracy_delta: float = 0.0  # positive = improving
    epochs_in_stage: int = 0

    # Stage context
    stage: int = 1  # SeedStage enum value (1-10); feature scaling clamps >=10 to 1.0
    alpha: float = 0.0  # blending weight (0-1)

    # Alpha controller context
    alpha_target: float = 0.0
    alpha_mode: int = 0
    alpha_steps_total: int = 0
    alpha_steps_done: int = 0
    time_to_target: int = 0
    alpha_velocity: float = 0.0
    alpha_algorithm: int = AlphaAlgorithm.ADD.value

    # Temporal context
    epoch: int = 0
    max_epochs: int = 25

    # Timestamp for staleness detection
    captured_at: datetime = field(default_factory=_utc_now)

    # Tempo telemetry
    blend_tempo_epochs: int = 5
    blending_velocity: float = 0.0  # d(alpha) / d(epoch)

    def to_features(self) -> list[float]:
        """Convert to 26-dim feature vector for RL policies.

        Features (schema v1, one-hot stage encoding):
        - [0-9]: Stage one-hot (10 dims) - categorical stage representation
        - [10]: gradient_norm (normalized to [0, 1])
        - [11]: gradient_health ([0, 1])
        - [12]: has_vanishing (0 or 1)
        - [13]: has_exploding (0 or 1)
        - [14]: epochs_in_stage (normalized)
        - [15]: accuracy (normalized to [0, 1])
        - [16]: accuracy_delta (clamped to [-1, 1])
        - [17]: alpha ([0, 1])
        - [18]: epoch/max_epochs (temporal position)
        - [19]: alpha_target ([0, 1])
        - [20]: alpha_mode (normalized)
        - [21]: alpha_steps_total (normalized)
        - [22]: alpha_steps_done (normalized)
        - [23]: time_to_target (normalized)
        - [24]: alpha_velocity (clamped to [-1, 1])
        - [25]: alpha_algorithm (normalized)
        """
        from esper.leyline.stage_schema import stage_to_one_hot, VALID_STAGE_VALUES

        steps_den = max(self.max_epochs, 1)
        alpha_algo_range = max(_ALPHA_ALGO_MAX - _ALPHA_ALGO_MIN, 1)

        # Stage one-hot (10 dims) - handles gap at value 5
        if self.stage in VALID_STAGE_VALUES:
            stage_one_hot = stage_to_one_hot(self.stage)
        else:
            # Fallback for invalid stage: all zeros (should not happen after Phase 0 validation)
            from esper.leyline.stage_schema import NUM_STAGES
            stage_one_hot = [0.0] * NUM_STAGES

        return stage_one_hot + [
            min(self.gradient_norm, _GRADIENT_NORM_MAX) / _GRADIENT_NORM_MAX,
            self.gradient_health,
            float(self.has_vanishing),
            float(self.has_exploding),
            min(self.epochs_in_stage, _EPOCHS_IN_STAGE_MAX) / _EPOCHS_IN_STAGE_MAX,
            self.accuracy / 100.0,
            max(-1.0, min(1.0, self.accuracy_delta / _ACCURACY_DELTA_SCALE)),
            self.alpha,
            self.epoch / max(self.max_epochs, 1),  # temporal position
            self.alpha_target,
            self.alpha_mode / max(_ALPHA_MODE_MAX, 1),
            min(self.alpha_steps_total, steps_den) / steps_den,
            min(self.alpha_steps_done, steps_den) / steps_den,
            min(self.time_to_target, steps_den) / steps_den,
            max(-1.0, min(1.0, self.alpha_velocity)),
            (self.alpha_algorithm - _ALPHA_ALGO_MIN) / alpha_algo_range,
        ]

    @classmethod
    def feature_dim(cls) -> int:
        """Return current feature vector dimension.

        Schema v1: 10 (stage one-hot) + 16 (other features) = 26
        """
        from esper.leyline.stage_schema import NUM_STAGES
        return NUM_STAGES + 16  # 10 + 16 = 26

    def to_dict(self) -> dict[str, Any]:
        """Convert to primitive dict for serialization."""
        from esper.leyline.stage_schema import STAGE_SCHEMA_VERSION
        return {
            "schema_version": STAGE_SCHEMA_VERSION,
            "seed_id": self.seed_id,
            "blueprint_id": self.blueprint_id,
            "layer_id": self.layer_id,
            "gradient_norm": self.gradient_norm,
            "gradient_health": self.gradient_health,
            "has_vanishing": self.has_vanishing,
            "has_exploding": self.has_exploding,
            "accuracy": self.accuracy,
            "accuracy_delta": self.accuracy_delta,
            "epochs_in_stage": self.epochs_in_stage,
            "stage": self.stage,
            "alpha": self.alpha,
            "alpha_target": self.alpha_target,
            "alpha_mode": self.alpha_mode,
            "alpha_steps_total": self.alpha_steps_total,
            "alpha_steps_done": self.alpha_steps_done,
            "time_to_target": self.time_to_target,
            "alpha_velocity": self.alpha_velocity,
            "alpha_algorithm": self.alpha_algorithm,
            "epoch": self.epoch,
            "max_epochs": self.max_epochs,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "blend_tempo_epochs": self.blend_tempo_epochs,
            "blending_velocity": self.blending_velocity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedTelemetry":
        """Reconstruct from primitive dict.

        Raises KeyError/ValueError if required fields are missing (no silent defaults).
        """
        from datetime import datetime
        from esper.leyline.stage_schema import STAGE_SCHEMA_VERSION

        # Validate schema version if present (fail-fast on mismatch)
        schema_version = data.get("schema_version")
        if schema_version is not None and schema_version != STAGE_SCHEMA_VERSION:
            raise ValueError(
                f"SeedTelemetry schema version mismatch: "
                f"expected {STAGE_SCHEMA_VERSION}, got {schema_version}. "
                f"Telemetry from incompatible version cannot be loaded."
            )

        # Required: stage - validate type and enum membership
        stage_raw = data["stage"]  # KeyError if missing
        if not isinstance(stage_raw, int) or isinstance(stage_raw, bool):
            raise ValueError(
                f"SeedTelemetry.stage must be an int SeedStage.value, got {stage_raw!r}"
            )
        try:
            stage = SeedStage(stage_raw).value
        except ValueError as e:
            raise ValueError(
                f"SeedTelemetry.stage must be a valid SeedStage.value, got {stage_raw!r}"
            ) from e

        # Required: alpha_mode - validate type and enum membership
        alpha_mode_raw = data["alpha_mode"]  # KeyError if missing
        if not isinstance(alpha_mode_raw, int) or isinstance(alpha_mode_raw, bool):
            raise ValueError(
                f"SeedTelemetry.alpha_mode must be an int AlphaMode.value, got {alpha_mode_raw!r}"
            )
        try:
            alpha_mode = AlphaMode(alpha_mode_raw).value
        except ValueError as e:
            raise ValueError(
                f"SeedTelemetry.alpha_mode must be a valid AlphaMode.value, got {alpha_mode_raw!r}"
            ) from e

        # Required: alpha_algorithm - validate type and enum membership
        alpha_algorithm_raw = data["alpha_algorithm"]  # KeyError if missing
        if not isinstance(alpha_algorithm_raw, int) or isinstance(alpha_algorithm_raw, bool):
            raise ValueError(
                f"SeedTelemetry.alpha_algorithm must be an int AlphaAlgorithm.value, got {alpha_algorithm_raw!r}"
            )
        try:
            alpha_algorithm = AlphaAlgorithm(alpha_algorithm_raw).value
        except ValueError as e:
            raise ValueError(
                f"SeedTelemetry.alpha_algorithm must be a valid AlphaAlgorithm.value, got {alpha_algorithm_raw!r}"
            ) from e

        return cls(
            # Required fields - KeyError if missing
            seed_id=data["seed_id"],
            blueprint_id=data["blueprint_id"],
            layer_id=data["layer_id"],
            gradient_norm=data["gradient_norm"],
            gradient_health=data["gradient_health"],
            has_vanishing=data["has_vanishing"],
            has_exploding=data["has_exploding"],
            accuracy=data["accuracy"],
            accuracy_delta=data["accuracy_delta"],
            epochs_in_stage=data["epochs_in_stage"],
            stage=stage,
            alpha=data["alpha"],
            alpha_target=data["alpha_target"],
            alpha_mode=alpha_mode,
            alpha_steps_total=data["alpha_steps_total"],
            alpha_steps_done=data["alpha_steps_done"],
            time_to_target=data["time_to_target"],
            alpha_velocity=data["alpha_velocity"],
            alpha_algorithm=alpha_algorithm,
            epoch=data["epoch"],
            max_epochs=data["max_epochs"],
            captured_at=datetime.fromisoformat(data["captured_at"]),
            blend_tempo_epochs=data["blend_tempo_epochs"],
            blending_velocity=data["blending_velocity"],
        )


# =============================================================================
# Typed Telemetry Payloads
# =============================================================================
# These replace the untyped dict[str, Any] in TelemetryEvent.data
# Each payload has:
# - REQUIRED fields (no default) - KeyError if missing
# - OPTIONAL fields (with default) - legitimately nullable
# See: docs/plans/2025-12-25-typed-telemetry-payloads-design.md


@dataclass(slots=True, frozen=True)
class TrainingStartedPayload:
    """Payload for TRAINING_STARTED event. Emitted once at training start."""

    # REQUIRED - training fails without these
    n_envs: int
    max_epochs: int
    task: str
    host_params: int  # Must be post-materialization
    slot_ids: tuple[str, ...]
    seed: int
    n_episodes: int
    lr: float
    clip_ratio: float
    entropy_coef: float
    param_budget: int
    policy_device: str
    env_devices: tuple[str, ...]

    # OPTIONAL - legitimate defaults
    episode_id: str = ""
    resume_path: str = ""
    reward_mode: str = ""
    start_episode: int = 0
    entropy_anneal: dict[str, float] | None = None

    # Distributed training (PyTorch expert recommendation)
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    # AMP config (PyTorch expert recommendation)
    amp_enabled: bool = False
    amp_dtype: str | None = None  # "float16" or "bfloat16"

    # torch.compile config (PyTorch expert recommendation)
    compile_enabled: bool = False
    compile_backend: str | None = None
    compile_mode: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingStartedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            n_envs=data["n_envs"],
            max_epochs=data["max_epochs"],
            task=data["task"],
            host_params=data["host_params"],
            slot_ids=_ensure_tuple(data["slot_ids"]),
            seed=data["seed"],
            n_episodes=data["n_episodes"],
            lr=data["lr"],
            clip_ratio=data["clip_ratio"],
            entropy_coef=data["entropy_coef"],
            param_budget=data["param_budget"],
            policy_device=data["policy_device"],
            env_devices=_ensure_tuple(data["env_devices"]),
            # Optional fields with defaults
            episode_id=data.get("episode_id", ""),
            resume_path=data.get("resume_path", ""),
            reward_mode=data.get("reward_mode", ""),
            start_episode=data.get("start_episode", 0),
            entropy_anneal=data.get("entropy_anneal"),
            world_size=data.get("world_size", 1),
            rank=data.get("rank", 0),
            local_rank=data.get("local_rank", 0),
            amp_enabled=data.get("amp_enabled", False),
            amp_dtype=data.get("amp_dtype"),
            compile_enabled=data.get("compile_enabled", False),
            compile_backend=data.get("compile_backend"),
            compile_mode=data.get("compile_mode"),
        )


@dataclass(slots=True, frozen=True)
class EpochCompletedPayload:
    """Payload for EPOCH_COMPLETED event. Emitted per environment per epoch."""

    # REQUIRED
    env_id: int
    val_accuracy: float
    val_loss: float
    inner_epoch: int

    # OPTIONAL - training metrics (None = not computed, 0.0 = computed as zero)
    train_loss: float | None = None
    train_accuracy: float | None = None
    host_grad_norm: float | None = None

    # OPTIONAL - per-seed telemetry snapshots
    seeds: dict[str, dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpochCompletedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            env_id=data["env_id"],
            val_accuracy=data["val_accuracy"],
            val_loss=data["val_loss"],
            inner_epoch=data.get("inner_epoch", data.get("epoch", 0)),
            train_loss=data.get("train_loss"),
            train_accuracy=data.get("train_accuracy"),
            host_grad_norm=data.get("grad_norm"),  # Note: dict uses "grad_norm"
            seeds=data.get("seeds"),
        )


@dataclass(slots=True, frozen=True)
class BatchEpochCompletedPayload:
    """Payload for BATCH_EPOCH_COMPLETED event. Emitted at episode boundary."""

    # REQUIRED (DRL expert: essential for metric normalization)
    episodes_completed: int
    batch_idx: int
    avg_accuracy: float
    avg_reward: float
    total_episodes: int
    n_envs: int

    # OPTIONAL - resume-aware metadata
    start_episode: int = 0  # Resume offset (episode where this run started)
    requested_episodes: int = 0  # Total episodes requested by user
    rolling_accuracy: float = 0.0
    env_accuracies: tuple[float, ...] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchEpochCompletedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        env_accuracies = data.get("env_accuracies")
        if env_accuracies is not None:
            env_accuracies = _ensure_tuple(env_accuracies)

        return cls(
            episodes_completed=data["episodes_completed"],
            batch_idx=data["batch_idx"],
            avg_accuracy=data["avg_accuracy"],
            avg_reward=data["avg_reward"],
            total_episodes=data["total_episodes"],
            n_envs=data["n_envs"],
            start_episode=data.get("start_episode", 0),
            requested_episodes=data.get("requested_episodes", 0),
            rolling_accuracy=data.get("rolling_accuracy", 0.0),
            env_accuracies=env_accuracies,
        )


@dataclass(slots=True, frozen=True)
class PPOUpdatePayload:
    """Payload for PPO_UPDATE_COMPLETED event. Emitted after each PPO update."""

    # REQUIRED - core PPO health metrics
    policy_loss: float
    value_loss: float
    entropy: float
    grad_norm: float  # Use float('inf') for AMP overflow
    kl_divergence: float
    clip_fraction: float
    nan_grad_count: int  # DRL expert: fail-fast on NaN

    # OPTIONAL - explained_variance can be NaN early training
    explained_variance: float | None = None

    # OPTIONAL - extended diagnostics
    entropy_loss: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    ratio_mean: float = 1.0
    ratio_min: float = 1.0
    ratio_max: float = 1.0
    ratio_std: float = 0.0
    lr: float | None = None
    entropy_coef: float | None = None

    # Gradient health (PyTorch expert: inf separate from nan)
    inf_grad_count: int = 0
    dead_layers: int = 0
    exploding_layers: int = 0
    layer_gradient_health: dict[str, float] | None = None
    entropy_collapsed: bool = False

    # AMP diagnostics (PyTorch expert recommendation)
    loss_scale: float | None = None
    amp_overflow_detected: bool = False
    update_skipped: bool = False

    # Timing
    update_time_ms: float = 0.0
    early_stop_epoch: int | None = None

    # Multi-head entropy (optional, only for factored policies)
    # These are averaged entropy/gradient values per action head
    head_slot_entropy: float | None = None
    head_blueprint_entropy: float | None = None
    head_slot_grad_norm: float | None = None
    head_blueprint_grad_norm: float | None = None
    head_style_entropy: float | None = None
    head_tempo_entropy: float | None = None
    head_alpha_target_entropy: float | None = None
    head_alpha_speed_entropy: float | None = None
    head_alpha_curve_entropy: float | None = None
    head_op_entropy: float | None = None

    # PPO inner loop context
    inner_epoch: int = 0
    batch: int = 0

    # Skipped update flag (for buffer rollback scenarios)
    skipped: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PPOUpdatePayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            policy_loss=data["policy_loss"],
            value_loss=data["value_loss"],
            entropy=data["entropy"],
            grad_norm=data["grad_norm"],
            kl_divergence=data["kl_divergence"],
            clip_fraction=data["clip_fraction"],
            nan_grad_count=data.get("nan_grad_count", 0),
            # Optional fields
            explained_variance=data.get("explained_variance"),
            entropy_loss=data.get("entropy_loss", 0.0),
            advantage_mean=data.get("advantage_mean", 0.0),
            advantage_std=data.get("advantage_std", 0.0),
            ratio_mean=data.get("ratio_mean", 1.0),
            ratio_min=data.get("ratio_min", 1.0),
            ratio_max=data.get("ratio_max", 1.0),
            ratio_std=data.get("ratio_std", 0.0),
            lr=data.get("lr"),
            entropy_coef=data.get("entropy_coef"),
            inf_grad_count=data.get("inf_grad_count", 0),
            dead_layers=data.get("dead_layers", 0),
            exploding_layers=data.get("exploding_layers", 0),
            layer_gradient_health=data.get("layer_gradient_health"),
            entropy_collapsed=data.get("entropy_collapsed", False),
            loss_scale=data.get("loss_scale"),
            amp_overflow_detected=data.get("amp_overflow_detected", False),
            update_skipped=data.get("update_skipped", False),
            update_time_ms=data.get("update_time_ms", 0.0),
            early_stop_epoch=data.get("early_stop_epoch"),
            head_slot_entropy=data.get("head_slot_entropy"),
            head_blueprint_entropy=data.get("head_blueprint_entropy"),
            head_slot_grad_norm=data.get("head_slot_grad_norm"),
            head_blueprint_grad_norm=data.get("head_blueprint_grad_norm"),
            head_style_entropy=data.get("head_style_entropy"),
            head_tempo_entropy=data.get("head_tempo_entropy"),
            head_alpha_target_entropy=data.get("head_alpha_target_entropy"),
            head_alpha_speed_entropy=data.get("head_alpha_speed_entropy"),
            head_alpha_curve_entropy=data.get("head_alpha_curve_entropy"),
            head_op_entropy=data.get("head_op_entropy"),
            inner_epoch=data.get("inner_epoch", 0),
            batch=data.get("batch", 0),
            skipped=data.get("skipped", False),
        )

    @classmethod
    def skipped_update(cls) -> "PPOUpdatePayload":
        """Factory for skipped PPO updates (buffer rollback)."""
        return cls(
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
            skipped=True,
        )


@dataclass(slots=True, frozen=True)
class RewardComputedPayload:
    """Payload for REWARD_COMPUTED event. Emitted per RL step."""

    # REQUIRED (DRL expert: value_estimate and action_confidence essential)
    env_id: int
    total_reward: float
    action_name: str
    value_estimate: float
    action_confidence: float

    # OPTIONAL - reward component breakdown (None = not computed, 0.0 = computed as zero)
    base_acc_delta: float | None = None
    bounded_attribution: float | None = None
    seed_contribution: float | None = None
    compute_rent: float | None = None
    alpha_shock: float | None = None
    ratio_penalty: float | None = None
    stage_bonus: float | None = None
    fossilize_terminal_bonus: float | None = None
    blending_warning: float | None = None
    holding_warning: float | None = None
    val_acc: float | None = None

    # Decision context
    slot_states: dict[str, dict[str, Any]] | None = None
    host_accuracy: float | None = None
    alternatives: list[tuple[str, float]] | None = None
    decision_entropy: float | None = None
    ab_group: str | None = None
    action_slot: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardComputedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            env_id=data["env_id"],
            total_reward=data["total_reward"],
            action_name=data["action_name"],
            value_estimate=data["value_estimate"],
            action_confidence=data["action_confidence"],
            # Optional fields (None = not provided, preserves "not computed" vs "zero")
            base_acc_delta=data.get("base_acc_delta"),
            bounded_attribution=data.get("bounded_attribution"),
            seed_contribution=data.get("seed_contribution"),
            compute_rent=data.get("compute_rent"),
            alpha_shock=data.get("alpha_shock"),
            ratio_penalty=data.get("ratio_penalty"),
            stage_bonus=data.get("stage_bonus"),
            fossilize_terminal_bonus=data.get("fossilize_terminal_bonus"),
            blending_warning=data.get("blending_warning"),
            holding_warning=data.get("holding_warning"),
            val_acc=data.get("val_acc"),
            slot_states=data.get("slot_states"),
            host_accuracy=data.get("host_accuracy"),
            alternatives=data.get("alternatives"),
            decision_entropy=data.get("decision_entropy"),
            ab_group=data.get("ab_group"),
            action_slot=data.get("action_slot"),
        )


@dataclass(slots=True, frozen=True)
class SeedGerminatedPayload:
    """Payload for SEED_GERMINATED event.

    Note: env_id may be -1 when emitted from slots (Kasmina), which don't
    know their environment context. The sentinel is replaced by the actual
    env_id in emit_with_env_context (simic/telemetry/emitters.py).
    """

    # REQUIRED
    slot_id: str
    env_id: int  # -1 = sentinel (replaced by emit_with_env_context)
    blueprint_id: str
    params: int

    # OPTIONAL
    alpha: float = 0.0
    grad_ratio: float = 0.0
    has_vanishing: bool = False
    has_exploding: bool = False
    epochs_in_stage: int = 0
    blend_tempo_epochs: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedGerminatedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            slot_id=data["slot_id"],
            env_id=data["env_id"],
            blueprint_id=data["blueprint_id"],
            params=data["params"],
            alpha=data.get("alpha", 0.0),
            grad_ratio=data.get("grad_ratio", 0.0),
            has_vanishing=data.get("has_vanishing", False),
            has_exploding=data.get("has_exploding", False),
            epochs_in_stage=data.get("epochs_in_stage", 0),
            blend_tempo_epochs=data.get("blend_tempo_epochs", 5),
        )


@dataclass(slots=True, frozen=True)
class SeedStageChangedPayload:
    """Payload for SEED_STAGE_CHANGED event.

    Note: env_id may be -1 when emitted from slots (Kasmina), which don't
    know their environment context. The sentinel is replaced by the actual
    env_id in emit_with_env_context (simic/telemetry/emitters.py).
    """

    # REQUIRED
    slot_id: str
    env_id: int  # -1 = sentinel (replaced by emit_with_env_context)
    from_stage: str
    to_stage: str

    # OPTIONAL
    alpha: float | None = None
    accuracy_delta: float = 0.0
    epochs_in_stage: int = 0
    grad_ratio: float = 0.0
    has_vanishing: bool = False
    has_exploding: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedStageChangedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            slot_id=data["slot_id"],
            env_id=data["env_id"],
            from_stage=data["from"],
            to_stage=data["to"],
            alpha=data.get("alpha"),
            accuracy_delta=data.get("accuracy_delta", 0.0),
            epochs_in_stage=data.get("epochs_in_stage", 0),
            grad_ratio=data.get("grad_ratio", 0.0),
            has_vanishing=data.get("has_vanishing", False),
            has_exploding=data.get("has_exploding", False),
        )


@dataclass(slots=True, frozen=True)
class SeedFossilizedPayload:
    """Payload for SEED_FOSSILIZED event.

    Note: env_id may be -1 when emitted from slots (Kasmina), which don't
    know their environment context. The sentinel is replaced by the actual
    env_id in emit_with_env_context (simic/telemetry/emitters.py).
    """

    # REQUIRED
    slot_id: str
    env_id: int  # -1 = sentinel (replaced by emit_with_env_context)
    blueprint_id: str
    improvement: float
    params_added: int

    # OPTIONAL (None = not computed, 0.0 = computed as zero)
    alpha: float = 1.0
    epochs_total: int = 0
    counterfactual: float | None = None
    blending_delta: float | None = None  # Accuracy change during blending stage

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedFossilizedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            slot_id=data["slot_id"],
            env_id=data["env_id"],
            blueprint_id=data["blueprint_id"],
            improvement=data["improvement"],
            params_added=data["params_added"],
            alpha=data.get("alpha", 1.0),
            epochs_total=data.get("epochs_total", 0),
            counterfactual=data.get("counterfactual"),
            blending_delta=data.get("blending_delta"),
        )


@dataclass(slots=True, frozen=True)
class SeedPrunedPayload:
    """Payload for SEED_PRUNED event.

    Note: env_id may be -1 when emitted from slots (Kasmina), which don't
    know their environment context. The sentinel is replaced by the actual
    env_id in emit_with_env_context (simic/telemetry/emitters.py).
    """

    # REQUIRED
    slot_id: str
    env_id: int  # -1 = sentinel (replaced by emit_with_env_context)
    reason: str

    # OPTIONAL (None = not computed, 0.0 = computed as zero)
    blueprint_id: str | None = None
    improvement: float = 0.0
    auto_pruned: bool = False
    epochs_total: int = 0
    counterfactual: float | None = None
    blending_delta: float | None = None  # Accuracy change during blending stage
    initiator: str = "policy"  # Who initiated the prune: "policy", "governor", "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedPrunedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            slot_id=data["slot_id"],
            env_id=data["env_id"],
            reason=data["reason"],
            blueprint_id=data.get("blueprint_id"),
            improvement=data.get("improvement", 0.0),
            auto_pruned=data.get("auto_pruned", False),
            epochs_total=data.get("epochs_total", 0),
            counterfactual=data.get("counterfactual"),
            blending_delta=data.get("blending_delta"),
            initiator=data.get("initiator", "policy"),
        )


@dataclass(slots=True, frozen=True)
class SeedGateEvaluatedPayload:
    """Payload for SEED_GATE_EVALUATED events.

    Note: env_id may be -1 when emitted from slots (Kasmina), which don't
    know their environment context. The sentinel is replaced by the actual
    env_id in emit_with_env_context (simic/telemetry/emitters.py).
    """

    # REQUIRED
    slot_id: str
    env_id: int  # -1 = sentinel (replaced by emit_with_env_context)
    gate: str  # Gate name (e.g., "G1", "G2", "G3", "G4", "G5")
    passed: bool
    target_stage: str
    checks_passed: tuple[str, ...]
    checks_failed: tuple[str, ...]

    # OPTIONAL
    message: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedGateEvaluatedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            slot_id=data["slot_id"],
            env_id=data["env_id"],
            gate=data["gate"],
            passed=data["passed"],
            target_stage=data["target_stage"],
            checks_passed=_ensure_tuple(data["checks_passed"]),
            checks_failed=_ensure_tuple(data["checks_failed"]),
            message=data.get("message"),
        )


@dataclass(slots=True, frozen=True)
class CounterfactualMatrixPayload:
    """Payload for COUNTERFACTUAL_MATRIX_COMPUTED event."""

    # REQUIRED
    env_id: int
    slot_ids: tuple[str, ...]
    configs: tuple[dict[str, Any], ...]

    # OPTIONAL
    strategy: str = "unavailable"
    compute_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CounterfactualMatrixPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            env_id=data["env_id"],
            slot_ids=_ensure_tuple(data["slot_ids"]),
            configs=_ensure_tuple(data["configs"]),
            strategy=data.get("strategy", "unavailable"),
            compute_time_ms=data.get("compute_time_ms", 0.0),
        )


@dataclass(slots=True, frozen=True)
class AnalyticsSnapshotPayload:
    """Payload for ANALYTICS_SNAPSHOT event. Used for dashboard sync.

    Supported kinds:
    - "action_distribution": batch-level action counts
    - "last_action": per-step decision context
    - "throughput": per-env performance metrics
    - "reward_summary": compact reward breakdown
    - "mask_hit_rates": per-head mask statistics
    - "batch_stats": training batch metrics (console output)
    - "summary_table": formatted analytics tables (console output)
    - "heuristic_config": heuristic mode run configuration
    - "heuristic_episode": heuristic mode episode completion
    - "heuristic_warning": heuristic mode telemetry warning
    - "shapley_computed": Shapley value attribution results
    """

    # REQUIRED
    kind: str

    # OPTIONAL - depends on kind
    action_counts: dict[str, int] | None = None

    # For kind="last_action", includes decision context
    env_id: int | None = None
    total_reward: float | None = None
    action_name: str | None = None  # The op name (e.g., "WAIT", "GERMINATE")
    action_confidence: float | None = None
    value_estimate: float | None = None
    slot_id: str | None = None
    blueprint_id: str | None = None
    style: str | None = None
    blend_id: str | None = None
    tempo_idx: int | None = None
    alpha_target: float | None = None
    alpha_speed: str | None = None
    alpha_curve: str | None = None
    alpha_algorithm: str | None = None
    alpha_algorithm_selected: str | None = None
    action_success: bool | None = None
    # Per-head mask flags (True = action was forced by mask)
    op_masked: bool | None = None
    slot_masked: bool | None = None
    blueprint_masked: bool | None = None
    style_masked: bool | None = None
    tempo_masked: bool | None = None
    alpha_target_masked: bool | None = None
    alpha_speed_masked: bool | None = None
    alpha_curve_masked: bool | None = None

    # For kind="throughput", includes performance metrics
    batch: int | None = None
    episodes_completed: int | None = None
    fps: float | None = None
    step_time_ms: float | None = None
    dataloader_wait_ms: float | None = None

    # For kind="reward_summary", includes reward breakdown
    summary: dict[str, float] | None = None

    # For kind="mask_hit_rates", includes per-head mask stats
    mask_hits: dict[str, int] | None = None
    mask_total: dict[str, int] | None = None

    # For kind="batch_stats", includes training metrics (console output)
    inner_epoch: int | None = None
    accuracy: float | None = None
    host_accuracy: float | None = None
    entropy: float | None = None
    kl_divergence: float | None = None
    value_variance: float | None = None
    seeds_created: int | None = None
    seeds_fossilized: int | None = None
    skipped_update: bool | None = None

    # For kind="summary_table", includes formatted tables (console output)
    summary_table: str | None = None
    scoreboard_tables: dict[int, str] | None = None

    # For kind="heuristic_config", includes heuristic run configuration
    mode: str | None = None  # "heuristic"
    task: str | None = None
    topology: str | None = None
    device: str | None = None
    slots: tuple[str, ...] | None = None
    episodes: int | None = None
    max_epochs: int | None = None
    max_batches: int | None = None
    min_fossilize_improvement: float | None = None
    telemetry_lifecycle_only: bool | None = None
    telemetry_level: str | None = None

    # For kind="heuristic_episode", includes episode completion summary
    episode_id: str | None = None
    episode: int | None = None
    episodes_total: int | None = None
    base_seed: int | None = None
    final_accuracy: float | None = None
    # total_reward already defined above for last_action

    # For kind="shapley_computed", includes Shapley value estimates
    shapley_values: dict[str, dict[str, float]] | None = None
    num_slots: int | None = None
    # epoch already defined via batch/inner_epoch

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalyticsSnapshotPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            kind=data["kind"],
            action_counts=data.get("action_counts"),
            env_id=data.get("env_id"),
            total_reward=data.get("total_reward"),
            action_name=data.get("action_name"),
            action_confidence=data.get("action_confidence"),
            value_estimate=data.get("value_estimate"),
            slot_id=data.get("slot_id"),
            blueprint_id=data.get("blueprint_id"),
            style=data.get("style"),
            blend_id=data.get("blend_id"),
            tempo_idx=data.get("tempo_idx"),
            alpha_target=data.get("alpha_target"),
            alpha_speed=data.get("alpha_speed"),
            alpha_curve=data.get("alpha_curve"),
            alpha_algorithm=data.get("alpha_algorithm"),
            alpha_algorithm_selected=data.get("alpha_algorithm_selected"),
            action_success=data.get("action_success"),
            op_masked=data.get("op_masked"),
            slot_masked=data.get("slot_masked"),
            blueprint_masked=data.get("blueprint_masked"),
            style_masked=data.get("style_masked"),
            tempo_masked=data.get("tempo_masked"),
            alpha_target_masked=data.get("alpha_target_masked"),
            alpha_speed_masked=data.get("alpha_speed_masked"),
            alpha_curve_masked=data.get("alpha_curve_masked"),
            batch=data.get("batch"),
            episodes_completed=data.get("episodes_completed"),
            fps=data.get("fps"),
            step_time_ms=data.get("step_time_ms"),
            dataloader_wait_ms=data.get("dataloader_wait_ms"),
            summary=data.get("summary"),
            mask_hits=data.get("mask_hits"),
            mask_total=data.get("mask_total"),
            inner_epoch=data.get("inner_epoch"),
            accuracy=data.get("accuracy"),
            host_accuracy=data.get("host_accuracy"),
            entropy=data.get("entropy"),
            kl_divergence=data.get("kl_divergence"),
            value_variance=data.get("value_variance"),
            seeds_created=data.get("seeds_created"),
            seeds_fossilized=data.get("seeds_fossilized"),
            skipped_update=data.get("skipped_update"),
            summary_table=data.get("summary_table"),
            scoreboard_tables=data.get("scoreboard_tables"),
            # Heuristic mode fields
            mode=data.get("mode"),
            task=data.get("task"),
            topology=data.get("topology"),
            device=data.get("device"),
            slots=tuple(data["slots"]) if data.get("slots") else None,
            episodes=data.get("episodes"),
            max_epochs=data.get("max_epochs"),
            max_batches=data.get("max_batches"),
            min_fossilize_improvement=data.get("min_fossilize_improvement"),
            telemetry_lifecycle_only=data.get("telemetry_lifecycle_only"),
            telemetry_level=data.get("telemetry_level"),
            episode_id=data.get("episode_id"),
            episode=data.get("episode"),
            episodes_total=data.get("episodes_total"),
            base_seed=data.get("base_seed"),
            final_accuracy=data.get("final_accuracy"),
            # Shapley fields
            shapley_values=data.get("shapley_values"),
            num_slots=data.get("num_slots"),
        )


# =============================================================================
# Telemetry Payload Type Union
# =============================================================================
# All telemetry event payloads are strongly typed dataclasses.
# See: docs/plans/2025-12-25-typed-telemetry-payloads-design.md

TelemetryPayload = (
    TrainingStartedPayload
    | EpochCompletedPayload
    | BatchEpochCompletedPayload
    | PPOUpdatePayload
    | RewardComputedPayload
    | SeedGerminatedPayload
    | SeedStageChangedPayload
    | SeedGateEvaluatedPayload
    | SeedFossilizedPayload
    | SeedPrunedPayload
    | CounterfactualMatrixPayload
    | AnalyticsSnapshotPayload
)
