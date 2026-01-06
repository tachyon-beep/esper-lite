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

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

if TYPE_CHECKING:
    from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry
    from esper.simic.telemetry.observation_stats import ObservationStatsTelemetry
from uuid import uuid4

from esper.leyline.alpha import AlphaAlgorithm, AlphaMode
from esper.leyline.factored_actions import NUM_OPS
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

    # === NEW: Debug Events (triggered by anomalies) ===
    RATIO_EXPLOSION_DETECTED = auto()
    RATIO_COLLAPSE_DETECTED = auto()
    VALUE_COLLAPSE_DETECTED = auto()
    GRADIENT_PATHOLOGY_DETECTED = auto()
    NUMERICAL_INSTABILITY_DETECTED = auto()

    # === Governor Events (Tolaria) ===
    GOVERNOR_ROLLBACK = auto()        # Emergency rollback executed

    # === Training Progress Events ===
    TRAINING_STARTED = auto()         # Training run initialized
    CHECKPOINT_LOADED = auto()        # Model checkpoint restored

    # === Counterfactual Attribution Events ===
    COUNTERFACTUAL_MATRIX_COMPUTED = auto()  # Full factorial matrix for env

    # === Analytics Events ===
    ANALYTICS_SNAPSHOT = auto()       # Full state snapshot for dashboard sync

    # === Episode Events ===
    EPISODE_OUTCOME = auto()  # Multi-objective outcome for Pareto analysis


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

    def __post_init__(self) -> None:
        # Normalize JSON/deserialized events where event_type arrives as a string.
        event_type_raw = cast(Any, self.event_type)
        if type(event_type_raw) is str:
            self.event_type = TelemetryEventType[event_type_raw]


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

    # Alpha curve shape (from AlphaCurveAction enum name).
    # Always present because policy always samples a curve; causal relevance
    # is handled by advantage masking in simic/agent/advantages.py.
    alpha_curve: str = "LINEAR"

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
            "alpha_curve": self.alpha_curve,
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
            alpha_curve=data["alpha_curve"],
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
    max_batches: int  # Total PPO update rounds in run (from CLI rounds)
    task: str
    host_params: int  # Must be post-materialization
    slot_ids: tuple[str, ...]
    seed: int
    n_episodes: int  # Total env episodes in run (n_envs * rounds)
    lr: float
    clip_ratio: float
    entropy_coef: float
    param_budget: int
    policy_device: str
    env_devices: tuple[str, ...]

    # REQUIRED - training context
    reward_mode: str  # e.g. "shaped", "sparse", "minimal", "simplified"

    # OPTIONAL - legitimate defaults
    episode_id: str = ""
    resume_path: str = ""
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
            max_batches=data["max_batches"],
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
            # Required field
            reward_mode=data["reward_mode"],
            # Optional fields with defaults
            episode_id=data.get("episode_id", ""),
            resume_path=data.get("resume_path", ""),
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
class CheckpointLoadedPayload:
    """Payload for CHECKPOINT_LOADED event. Emitted when training resumes from checkpoint."""

    # REQUIRED
    path: str  # Path to the checkpoint file
    start_episode: int  # Episode number to resume from

    # OPTIONAL
    source: str | None = None  # Human-readable source description (e.g., "best checkpoint")
    avg_accuracy: float | None = None  # Accuracy at checkpoint time (if available)


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

    # OPTIONAL - observation space health metrics
    observation_stats: "ObservationStatsTelemetry | None" = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpochCompletedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        observation_stats_data = data["observation_stats"]
        observation_stats: ObservationStatsTelemetry | None
        if observation_stats_data is None:
            observation_stats = None
        else:
            from esper.simic.telemetry.observation_stats import ObservationStatsTelemetry
            observation_stats = ObservationStatsTelemetry.from_dict(observation_stats_data)

        return cls(
            env_id=data["env_id"],
            val_accuracy=data["val_accuracy"],
            val_loss=data["val_loss"],
            inner_epoch=data["inner_epoch"],
            train_loss=data.get("train_loss"),
            train_accuracy=data.get("train_accuracy"),
            host_grad_norm=data.get("host_grad_norm"),
            seeds=data.get("seeds"),
            observation_stats=observation_stats,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for telemetry serialization.

        PERF: Avoids dataclasses.asdict() deep-copying nested dict payloads (e.g., per-seed telemetry).
        """
        observation_stats = (
            self.observation_stats.to_dict()
            if self.observation_stats is not None
            else None
        )
        return {
            "env_id": self.env_id,
            "val_accuracy": self.val_accuracy,
            "val_loss": self.val_loss,
            "inner_epoch": self.inner_epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "host_grad_norm": self.host_grad_norm,
            "seeds": self.seeds,
            "observation_stats": observation_stats,
        }


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
class TrendDetectedPayload:
    """Payload for trend detection events (PLATEAU/DEGRADATION/IMPROVEMENT_DETECTED).

    Emitted when rolling accuracy crosses threshold between batches.
    All three event types use the same payload structure.
    """

    # REQUIRED - identifies where in training this occurred
    batch_idx: int
    episodes_completed: int

    # REQUIRED - the trend detection data
    rolling_delta: float  # Change from previous rolling avg
    rolling_avg_accuracy: float  # Current rolling avg
    prev_rolling_avg_accuracy: float  # Previous rolling avg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrendDetectedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            batch_idx=data["batch_idx"],
            episodes_completed=data["episodes_completed"],
            rolling_delta=data["rolling_delta"],
            rolling_avg_accuracy=data["rolling_avg_accuracy"],
            prev_rolling_avg_accuracy=data["prev_rolling_avg_accuracy"],
        )


@dataclass(slots=True, frozen=True)
class PPOUpdatePayload:
    """Payload for PPO_UPDATE_COMPLETED event. Emitted after each PPO update."""

    # REQUIRED - core PPO health metrics
    policy_loss: float
    value_loss: float
    entropy: float
    grad_norm: float  # Post-clip gradient norm (typically ~1.0 when clipping active)
    kl_divergence: float
    clip_fraction: float
    nan_grad_count: int  # DRL expert: fail-fast on NaN

    # BUG FIX: Pre-clip gradient norm captures actual gradient magnitude BEFORE clipping.
    # Critical for detecting gradient explosion (pre_clip >> 1.0) vs healthy gradients.
    # Previously this was lost because clip_grad_norm_() return value was discarded.
    pre_clip_grad_norm: float = 0.0

    # OPTIONAL - explained_variance can be NaN early training
    explained_variance: float | None = None

    # OPTIONAL - extended diagnostics
    entropy_loss: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_skewness: float = 0.0  # Asymmetry: >0 right-skewed, <0 left-skewed
    advantage_kurtosis: float = 0.0  # Tail heaviness: >0 heavy tails (more outliers), <0 light tails
    advantage_positive_ratio: float = 0.5  # Fraction positive (healthy: 0.4-0.6)

    # Pre-normalization advantage stats (critical for diagnosing value collapse)
    # If pre_norm_std is tiny but pre_clip_grad_norm is huge → normalization amplifying noise
    # If pre_norm_std is healthy but grad is huge → raw return scale or value mismatch
    pre_norm_advantage_mean: float = 0.0
    pre_norm_advantage_std: float = 0.0

    # Return statistics (for diagnosing value loss scale)
    return_mean: float = 0.0
    return_std: float = 0.0

    # Value function quality metrics (TELE-220 to TELE-228)
    # These measure value network calibration and return distribution shape
    v_return_correlation: float = 0.0
    td_error_mean: float = 0.0
    td_error_std: float = 0.0
    bellman_error: float = 0.0
    return_p10: float = 0.0
    return_p50: float = 0.0
    return_p90: float = 0.0
    return_variance: float = 0.0
    return_skewness: float = 0.0

    # Value target scale: the std used to normalize returns before value loss
    # Tracks raw return variance; normalized_returns = valid_returns / value_target_scale
    value_target_scale: float = 1.0

    ratio_mean: float = 1.0
    ratio_min: float = 1.0
    ratio_max: float = 1.0
    ratio_std: float = 0.0
    # Log prob extremes (NaN predictor: <-50 warning, <-100 critical)
    log_prob_min: float = 0.0
    log_prob_max: float = 0.0
    lr: float | None = None
    entropy_coef: float | None = None

    # Gradient health (PyTorch expert: inf separate from nan)
    inf_grad_count: int = 0
    dead_layers: int = 0
    exploding_layers: int = 0
    layer_gradient_health: dict[str, float] | None = None
    entropy_collapsed: bool = False

    # Per-head NaN/Inf detection (for indicator lights with latch behavior)
    # Keys are HEAD_NAMES: op, slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve
    head_nan_detected: dict[str, bool] | None = None
    head_inf_detected: dict[str, bool] | None = None

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
    head_style_grad_norm: float | None = None
    head_tempo_grad_norm: float | None = None
    head_alpha_target_grad_norm: float | None = None
    head_alpha_speed_grad_norm: float | None = None
    head_alpha_curve_grad_norm: float | None = None
    head_op_grad_norm: float | None = None
    head_style_entropy: float | None = None
    head_tempo_entropy: float | None = None
    head_alpha_target_entropy: float | None = None
    head_alpha_speed_entropy: float | None = None
    head_alpha_curve_entropy: float | None = None
    head_op_entropy: float | None = None

    # Per-head PPO ratio max (Policy V2 - multi-head ratio explosion detection)
    # Individual head ratios can look healthy while joint ratio exceeds clip range
    head_slot_ratio_max: float = 1.0
    head_blueprint_ratio_max: float = 1.0
    head_style_ratio_max: float = 1.0
    head_tempo_ratio_max: float = 1.0
    head_alpha_target_ratio_max: float = 1.0
    head_alpha_speed_ratio_max: float = 1.0
    head_alpha_curve_ratio_max: float = 1.0
    head_op_ratio_max: float = 1.0
    joint_ratio_max: float = 1.0  # Product of per-head ratios (computed in log-space)

    # PPO inner loop context
    inner_epoch: int = 0  # Host training epoch (1-150), NOT PPO update iteration
    batch: int = 0

    # BUG FIX: Track actual number of PPO gradient updates that occurred in this batch.
    # Previously inner_epoch was misused for this (always showed max_epochs=150).
    # This may be less than ppo_updates_per_batch if early-stopped on KL divergence.
    ppo_updates_count: int = 1

    # Skipped update flag (for buffer rollback scenarios)
    skipped: bool = False

    # Value function statistics (for divergence detection)
    value_mean: float = 0.0
    value_std: float = 0.0
    value_min: float = 0.0
    value_max: float = 0.0

    # === Op-Conditioned Q-Values (Policy V2) ===
    # Q(s, op) vector aligned to LifecycleOp/NUM_OPS ordering.
    op_q_values: tuple[float, ...] = field(
        default_factory=lambda: tuple(float("nan") for _ in range(NUM_OPS))
    )
    op_valid_mask: tuple[bool, ...] = field(
        default_factory=lambda: tuple(False for _ in range(NUM_OPS))
    )

    # Q-value analysis metrics
    q_variance: float = 0.0  # Variance across ops (low = critic ignoring op conditioning)
    q_spread: float = 0.0    # max(Q) - min(Q) across ops

    # === Gradient Quality Metrics (per DRL expert review) ===
    # Directional clip: WHERE clipping occurs (not WHETHER policy improved)
    clip_fraction_positive: float = 0.0  # r > 1+ε (probability increases capped)
    clip_fraction_negative: float = 0.0  # r < 1-ε (probability decreases capped)
    # Gradient CV: coefficient of variation = std/|mean|
    # Low (<0.5) = high signal quality, High (>2.0) = noisy gradients
    gradient_cv: float = 0.0

    # === Infrastructure Metrics (per PyTorch expert review) ===
    # CUDA memory collected every N batches to amortize sync overhead
    cuda_memory_allocated_gb: float = 0.0  # torch.cuda.memory_allocated()
    cuda_memory_reserved_gb: float = 0.0   # torch.cuda.memory_reserved()
    cuda_memory_peak_gb: float = 0.0       # torch.cuda.max_memory_allocated()
    cuda_memory_fragmentation: float = 0.0 # 1 - (allocated/reserved), >0.3 = pressure
    dataloader_wait_ratio: float = 0.0    # Fraction of step time spent waiting on data

    # === LSTM Hidden State Health (B7-DRL-04) ===
    # Tracks LSTM hidden state stability after PPO updates (BPTT can corrupt states)
    # h = hidden state, c = cell state
    # NOTE: Total L2 norms scale with sqrt(numel) and are NOT batch-size invariant.
    # Use *_rms and *_env_rms_* for scale-free health signals.
    lstm_h_l2_total: float | None = None
    lstm_c_l2_total: float | None = None
    lstm_h_rms: float | None = None
    lstm_c_rms: float | None = None
    lstm_h_env_rms_mean: float | None = None
    lstm_h_env_rms_max: float | None = None
    lstm_c_env_rms_mean: float | None = None
    lstm_c_env_rms_max: float | None = None
    lstm_h_max: float | None = None   # Max absolute value in h
    lstm_c_max: float | None = None   # Max absolute value in c
    lstm_has_nan: bool = False  # NaN detected in hidden state
    lstm_has_inf: bool = False  # Inf detected in hidden state

    def __post_init__(self) -> None:
        if len(self.op_q_values) != NUM_OPS:
            raise ValueError(
                f"Expected op_q_values length {NUM_OPS}, got {len(self.op_q_values)}."
            )
        if len(self.op_valid_mask) != NUM_OPS:
            raise ValueError(
                f"Expected op_valid_mask length {NUM_OPS}, got {len(self.op_valid_mask)}."
            )

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
            # DRL-03 fix: nan_grad_count is required (fail-fast on missing)
            nan_grad_count=data["nan_grad_count"],
            # Always emitted - fail loudly if missing
            pre_clip_grad_norm=data["pre_clip_grad_norm"],
            # Conditionally optional
            explained_variance=data.get("explained_variance"),
            entropy_loss=data.get("entropy_loss", 0.0),  # Legacy: hardcoded to 0
            # Advantage stats - always emitted
            advantage_mean=data["advantage_mean"],
            advantage_std=data["advantage_std"],
            advantage_skewness=data["advantage_skewness"],
            advantage_kurtosis=data["advantage_kurtosis"],
            advantage_positive_ratio=data["advantage_positive_ratio"],
            # Ratio stats - always emitted
            ratio_mean=data["ratio_mean"],
            ratio_min=data["ratio_min"],
            ratio_max=data["ratio_max"],
            ratio_std=data["ratio_std"],
            # Log prob extremes - always emitted
            log_prob_min=data["log_prob_min"],
            log_prob_max=data["log_prob_max"],
            lr=data.get("lr"),
            entropy_coef=data.get("entropy_coef"),
            inf_grad_count=data.get("inf_grad_count", 0),
            dead_layers=data.get("dead_layers", 0),
            exploding_layers=data.get("exploding_layers", 0),
            layer_gradient_health=data.get("layer_gradient_health"),
            # Always emitted
            entropy_collapsed=data["entropy_collapsed"],
            # Per-head NaN/Inf detection (optional)
            head_nan_detected=data.get("head_nan_detected"),
            head_inf_detected=data.get("head_inf_detected"),
            # Conditionally optional (AMP-related)
            loss_scale=data.get("loss_scale"),
            amp_overflow_detected=data.get("amp_overflow_detected", False),
            update_skipped=data.get("update_skipped", False),
            # Always emitted
            update_time_ms=data["update_time_ms"],
            early_stop_epoch=data.get("early_stop_epoch"),
            head_slot_entropy=data.get("head_slot_entropy"),
            head_blueprint_entropy=data.get("head_blueprint_entropy"),
            head_slot_grad_norm=data.get("head_slot_grad_norm"),
            head_blueprint_grad_norm=data.get("head_blueprint_grad_norm"),
            head_style_grad_norm=data.get("head_style_grad_norm"),
            head_tempo_grad_norm=data.get("head_tempo_grad_norm"),
            head_alpha_target_grad_norm=data.get("head_alpha_target_grad_norm"),
            head_alpha_speed_grad_norm=data.get("head_alpha_speed_grad_norm"),
            head_alpha_curve_grad_norm=data.get("head_alpha_curve_grad_norm"),
            head_op_grad_norm=data.get("head_op_grad_norm"),
            head_style_entropy=data.get("head_style_entropy"),
            head_tempo_entropy=data.get("head_tempo_entropy"),
            head_alpha_target_entropy=data.get("head_alpha_target_entropy"),
            head_alpha_speed_entropy=data.get("head_alpha_speed_entropy"),
            head_alpha_curve_entropy=data.get("head_alpha_curve_entropy"),
            head_op_entropy=data.get("head_op_entropy"),
            # Per-head ratio max
            head_slot_ratio_max=data.get("head_slot_ratio_max", 1.0),
            head_blueprint_ratio_max=data.get("head_blueprint_ratio_max", 1.0),
            head_style_ratio_max=data.get("head_style_ratio_max", 1.0),
            head_tempo_ratio_max=data.get("head_tempo_ratio_max", 1.0),
            head_alpha_target_ratio_max=data.get("head_alpha_target_ratio_max", 1.0),
            head_alpha_speed_ratio_max=data.get("head_alpha_speed_ratio_max", 1.0),
            head_alpha_curve_ratio_max=data.get("head_alpha_curve_ratio_max", 1.0),
            head_op_ratio_max=data.get("head_op_ratio_max", 1.0),
            joint_ratio_max=data.get("joint_ratio_max", 1.0),
            # Always emitted
            inner_epoch=data["inner_epoch"],
            batch=data["batch"],
            ppo_updates_count=data["ppo_updates_count"],
            # Conditionally optional (buffer rollback)
            skipped=data.get("skipped", False),
            # Value function statistics - always emitted
            value_mean=data["value_mean"],
            value_std=data["value_std"],
            value_min=data["value_min"],
            value_max=data["value_max"],
            # Q-values
            op_q_values=_ensure_tuple(data["op_q_values"]),
            op_valid_mask=_ensure_tuple(data["op_valid_mask"]),
            q_variance=data["q_variance"],
            q_spread=data["q_spread"],
            # Gradient quality metrics - always emitted
            clip_fraction_positive=data["clip_fraction_positive"],
            clip_fraction_negative=data["clip_fraction_negative"],
            gradient_cv=data["gradient_cv"],
            # Infrastructure metrics
            cuda_memory_allocated_gb=data.get("cuda_memory_allocated_gb", 0.0),
            cuda_memory_reserved_gb=data.get("cuda_memory_reserved_gb", 0.0),
            cuda_memory_peak_gb=data.get("cuda_memory_peak_gb", 0.0),
            cuda_memory_fragmentation=data.get("cuda_memory_fragmentation", 0.0),
            dataloader_wait_ratio=data["dataloader_wait_ratio"],
            # LSTM health metrics (B7-DRL-04)
            lstm_h_l2_total=data.get("lstm_h_l2_total"),
            lstm_c_l2_total=data.get("lstm_c_l2_total"),
            lstm_h_rms=data.get("lstm_h_rms"),
            lstm_c_rms=data.get("lstm_c_rms"),
            lstm_h_env_rms_mean=data.get("lstm_h_env_rms_mean"),
            lstm_h_env_rms_max=data.get("lstm_h_env_rms_max"),
            lstm_c_env_rms_mean=data.get("lstm_c_env_rms_mean"),
            lstm_c_env_rms_max=data.get("lstm_c_env_rms_max"),
            lstm_h_max=data.get("lstm_h_max"),
            lstm_c_max=data.get("lstm_c_max"),
            lstm_has_nan=data.get("lstm_has_nan", False),
            lstm_has_inf=data.get("lstm_has_inf", False),
            # Pre-normalization advantage stats (for diagnosing value collapse)
            # Always emitted - fail loudly if missing
            pre_norm_advantage_mean=data["pre_norm_advantage_mean"],
            pre_norm_advantage_std=data["pre_norm_advantage_std"],
            # Return statistics (for diagnosing value loss scale)
            # Always emitted - fail loudly if missing
            return_mean=data["return_mean"],
            return_std=data["return_std"],
        )

    @classmethod
    def skipped_update(cls) -> "PPOUpdatePayload":
        """Factory for skipped PPO updates (buffer rollback)."""
        nan = float("nan")
        return cls(
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            nan_grad_count=0,
            op_q_values=tuple(nan for _ in range(NUM_OPS)),
            op_valid_mask=tuple(False for _ in range(NUM_OPS)),
            q_variance=nan,
            q_spread=nan,
            skipped=True,
        )


@dataclass(slots=True, frozen=True)
class TamiyoInitiatedPayload:
    """Payload for TAMIYO_INITIATED event.

    Emitted when host network stabilizes and germination becomes allowed.
    This marks the transition from warmup phase to active seed training.
    """

    # REQUIRED
    env_id: int
    epoch: int
    stable_count: int  # Number of consecutive stable epochs
    stabilization_epochs: int  # Required epochs for stabilization
    val_loss: float  # Validation loss at stabilization

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TamiyoInitiatedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            env_id=data["env_id"],
            epoch=data["epoch"],
            stable_count=data["stable_count"],
            stabilization_epochs=data["stabilization_epochs"],
            val_loss=data["val_loss"],
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
    alpha_curve: str = "LINEAR"

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
            alpha_curve=data["alpha_curve"],
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
    # Alpha curve - always present (policy always samples), but only causally
    # relevant during BLENDING. See simic/agent/advantages.py for causal masking.
    alpha_curve: str = "LINEAR"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeedStageChangedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            slot_id=data["slot_id"],
            env_id=data["env_id"],
            from_stage=data["from_stage"],
            to_stage=data["to_stage"],
            alpha=data.get("alpha"),
            accuracy_delta=data.get("accuracy_delta", 0.0),
            epochs_in_stage=data.get("epochs_in_stage", 0),
            grad_ratio=data.get("grad_ratio", 0.0),
            has_vanishing=data.get("has_vanishing", False),
            has_exploding=data.get("has_exploding", False),
            alpha_curve=data["alpha_curve"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for telemetry serialization.

        PERF: Avoids dataclasses.asdict() deep-copy overhead and enforces a
        stable JSON schema that matches the dataclass field names.
        """
        return {
            "slot_id": self.slot_id,
            "env_id": self.env_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "alpha": self.alpha,
            "accuracy_delta": self.accuracy_delta,
            "epochs_in_stage": self.epochs_in_stage,
            "grad_ratio": self.grad_ratio,
            "has_vanishing": self.has_vanishing,
            "has_exploding": self.has_exploding,
            "alpha_curve": self.alpha_curve,
        }


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
class HeadTelemetry:
    """Per-head confidence and entropy values for factored action heads.

    Confidence = P(chosen_action | valid_mask) via exp(log_prob).
    This is the probability among valid actions, properly handling masking.

    Entropy measures how spread out the distribution is (higher = more uncertain).
    """

    # Per-head confidence (probability of chosen action)
    op_confidence: float = 0.0
    slot_confidence: float = 0.0
    blueprint_confidence: float = 0.0
    style_confidence: float = 0.0
    tempo_confidence: float = 0.0
    alpha_target_confidence: float = 0.0
    alpha_speed_confidence: float = 0.0
    curve_confidence: float = 0.0

    # Per-head entropy (distribution spread - higher means more uncertain)
    op_entropy: float = 0.0
    slot_entropy: float = 0.0
    blueprint_entropy: float = 0.0
    style_entropy: float = 0.0
    tempo_entropy: float = 0.0
    alpha_target_entropy: float = 0.0
    alpha_speed_entropy: float = 0.0
    curve_entropy: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadTelemetry":
        """Parse from dict. All fields have defaults of 0.0."""
        return cls(
            op_confidence=data.get("op_confidence", 0.0),
            slot_confidence=data.get("slot_confidence", 0.0),
            blueprint_confidence=data.get("blueprint_confidence", 0.0),
            style_confidence=data.get("style_confidence", 0.0),
            tempo_confidence=data.get("tempo_confidence", 0.0),
            alpha_target_confidence=data.get("alpha_target_confidence", 0.0),
            alpha_speed_confidence=data.get("alpha_speed_confidence", 0.0),
            curve_confidence=data.get("curve_confidence", 0.0),
            op_entropy=data.get("op_entropy", 0.0),
            slot_entropy=data.get("slot_entropy", 0.0),
            blueprint_entropy=data.get("blueprint_entropy", 0.0),
            style_entropy=data.get("style_entropy", 0.0),
            tempo_entropy=data.get("tempo_entropy", 0.0),
            alpha_target_entropy=data.get("alpha_target_entropy", 0.0),
            alpha_speed_entropy=data.get("alpha_speed_entropy", 0.0),
            curve_entropy=data.get("curve_entropy", 0.0),
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for telemetry serialization (avoids dataclasses.asdict deep copy)."""
        return {
            "op_confidence": self.op_confidence,
            "slot_confidence": self.slot_confidence,
            "blueprint_confidence": self.blueprint_confidence,
            "style_confidence": self.style_confidence,
            "tempo_confidence": self.tempo_confidence,
            "alpha_target_confidence": self.alpha_target_confidence,
            "alpha_speed_confidence": self.alpha_speed_confidence,
            "curve_confidence": self.curve_confidence,
            "op_entropy": self.op_entropy,
            "slot_entropy": self.slot_entropy,
            "blueprint_entropy": self.blueprint_entropy,
            "style_entropy": self.style_entropy,
            "tempo_entropy": self.tempo_entropy,
            "alpha_target_entropy": self.alpha_target_entropy,
            "alpha_speed_entropy": self.alpha_speed_entropy,
            "curve_entropy": self.curve_entropy,
        }


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
    # Per-head telemetry (typed dataclass, not raw dict)
    head_telemetry: HeadTelemetry | None = None
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
    # Reward component breakdown (for kind="last_action")
    base_acc_delta: float | None = None  # Legacy shaped signal from accuracy improvement
    bounded_attribution: float | None = None  # Contribution-primary attribution signal
    compute_rent: float | None = None  # Cost of active seeds (always negative)
    # Additional reward components for RewardHealthPanel (Sanctum)
    stage_bonus: float | None = None  # PBRS shaping bonus for lifecycle stages
    ratio_penalty: float | None = None  # Anti-gaming penalty for extreme ratios
    alpha_shock: float | None = None  # Convex penalty on alpha deltas
    # Full reward components dataclass (replaces individual fields)
    reward_components: "RewardComponentsTelemetry | None" = None
    # Observation space health (for early NaN detection)
    observation_stats: "ObservationStatsTelemetry | None" = None
    # Decision context for TamiyoBrain Decision Cards
    slot_states: dict[str, str] | None = None  # slot_id -> "Training 12%" or "Empty"
    alternatives: list[tuple[str, float]] | None = None  # Top-2 alternative (action, prob)
    decision_entropy: float | None = None  # -sum(p*log(p)) for action distribution

    # For kind="throughput", includes performance metrics
    batch: int | None = None
    episodes_completed: int | None = None
    fps: float | None = None
    step_time_ms: float | None = None
    dataloader_wait_ms: float | None = None

    # For kind="reward_summary", includes reward breakdown
    summary: dict[str, float] | None = None

    # Scaffold hindsight credit debugging fields (Phase 3.2)
    # Populated when kind="reward_summary" and scaffolding occurred
    hindsight_credit: float | None = None  # Credit applied (post-cap)
    scaffold_count: int | None = None  # Number of scaffolds that contributed
    avg_scaffold_delay: float | None = None  # Average epochs since scaffolding interactions

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
            # Scaffold hindsight credit fields (Phase 3.2)
            hindsight_credit=data.get("hindsight_credit"),
            scaffold_count=data.get("scaffold_count"),
            avg_scaffold_delay=data.get("avg_scaffold_delay"),
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
            # Reward components (nested dataclass)
            reward_components=cls._parse_reward_components(data.get("reward_components")),
            # Observation stats (nested dataclass)
            observation_stats=cls._parse_observation_stats(data.get("observation_stats")),
            # Head telemetry (nested dataclass)
            head_telemetry=cls._parse_head_telemetry(data.get("head_telemetry")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for telemetry serialization.

        PERF: Avoids dataclasses.asdict() deep-copying large, already-JSONable containers
        like slot_states, alternatives, scoreboard_tables, and shapley_values.
        """
        payload: dict[str, Any] = {}
        for dc_field in dataclasses.fields(self):
            value = getattr(self, dc_field.name)
            if dc_field.name == "head_telemetry":
                payload[dc_field.name] = value.to_dict() if value is not None else None
            elif dc_field.name == "reward_components":
                payload[dc_field.name] = value.to_dict() if value is not None else None
            elif dc_field.name == "observation_stats":
                payload[dc_field.name] = value.to_dict() if value is not None else None
            else:
                payload[dc_field.name] = value
        return payload

    @staticmethod
    def _parse_reward_components(
        data: dict[str, Any] | None,
    ) -> "RewardComponentsTelemetry | None":
        """Parse reward_components from dict if present.

        Uses late import to avoid circular dependency at module load time.
        """
        if data is None:
            return None
        from esper.simic.rewards.reward_telemetry import RewardComponentsTelemetry

        return RewardComponentsTelemetry.from_dict(data)

    @staticmethod
    def _parse_head_telemetry(data: dict[str, Any] | None) -> HeadTelemetry | None:
        """Parse head_telemetry from dict if present."""
        if data is None:
            return None
        return HeadTelemetry.from_dict(data)

    @staticmethod
    def _parse_observation_stats(
        data: dict[str, Any] | None,
    ) -> "ObservationStatsTelemetry | None":
        """Parse observation_stats from dict if present.

        Uses late import to avoid circular dependency at module load time.
        """
        if data is None:
            return None
        from esper.simic.telemetry.observation_stats import ObservationStatsTelemetry

        return ObservationStatsTelemetry.from_dict(data)


@dataclass(slots=True, frozen=True)
class AnomalyDetectedPayload:
    """Payload for all anomaly detection events.

    Used by:
    - RATIO_EXPLOSION_DETECTED
    - RATIO_COLLAPSE_DETECTED
    - VALUE_COLLAPSE_DETECTED
    - NUMERICAL_INSTABILITY_DETECTED
    - GRADIENT_ANOMALY
    - GRADIENT_PATHOLOGY_DETECTED

    All anomaly events share the same payload structure to eliminate
    isinstance(event.data, dict) patterns and provide type safety.
    """

    # REQUIRED - core anomaly metadata
    anomaly_type: str  # The specific anomaly type name (e.g., "ratio_explosion")
    episode: int  # Episode number when anomaly detected
    batch: int  # Batch number within episode
    inner_epoch: int  # Max epochs for the episode
    total_episodes: int  # Total episodes in training run

    # OPTIONAL - anomaly details
    detail: str = ""  # Human-readable description of the anomaly

    # OPTIONAL - debug fields (only populated when collect_debug=True)
    gradient_stats: tuple[dict[str, Any], ...] | None = None  # Per-layer gradient statistics
    stability: dict[str, Any] | None = None  # Numerical stability report
    ratio_diagnostic: dict[str, Any] | None = None  # PPO ratio diagnostic info

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnomalyDetectedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        # Convert list to tuple for gradient_stats (JSON deserialization compatibility)
        gradient_stats = data.get("gradient_stats")
        if gradient_stats is not None:
            gradient_stats = _ensure_tuple(gradient_stats)

        return cls(
            anomaly_type=data["anomaly_type"],
            episode=data["episode"],
            batch=data["batch"],
            inner_epoch=data["inner_epoch"],
            total_episodes=data["total_episodes"],
            detail=data.get("detail", ""),
            gradient_stats=gradient_stats,
            stability=data.get("stability"),
            ratio_diagnostic=data.get("ratio_diagnostic"),
        )


@dataclass(slots=True, frozen=True)
class PerformanceDegradationPayload:
    """Payload for PERFORMANCE_DEGRADATION event.

    Emitted when current accuracy drops significantly below rolling average,
    indicating potential training instability or policy collapse.
    """

    # REQUIRED
    env_id: int
    current_acc: float
    rolling_avg_acc: float
    drop_percent: float  # Relative drop as percentage (0-100)
    threshold_percent: float  # Threshold that was exceeded

    # OPTIONAL
    training_progress: float = 0.0  # Progress through training (0.0 to 1.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceDegradationPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            env_id=data["env_id"],
            current_acc=data["current_acc"],
            rolling_avg_acc=data["rolling_avg_acc"],
            drop_percent=data["drop_percent"],
            threshold_percent=data["threshold_percent"],
            training_progress=data.get("training_progress", 0.0),
        )


@dataclass(slots=True, frozen=True)
class MemoryWarningPayload:
    """Payload for MEMORY_WARNING event.

    Emitted when GPU memory utilization exceeds the configured threshold.
    """

    # REQUIRED
    gpu_utilization: float  # 0.0 to 1.0
    gpu_allocated_gb: float
    gpu_total_gb: float
    threshold: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryWarningPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            gpu_utilization=data["gpu_utilization"],
            gpu_allocated_gb=data["gpu_allocated_gb"],
            gpu_total_gb=data["gpu_total_gb"],
            threshold=data["threshold"],
        )


RewardHackingPattern = Literal[
    "attribution_ratio",
    "ransomware_signature",
]


@dataclass(slots=True, frozen=True)
class RewardHackingSuspectedPayload:
    """Payload for REWARD_HACKING_SUSPECTED event.

    Two patterns share the same event type:
    - attribution_ratio: seed claims an implausible share of improvement
    - ransomware_signature: seed claims high contribution while system degrades
    """

    # REQUIRED
    pattern: RewardHackingPattern
    slot_id: str
    seed_id: str
    seed_contribution: float
    total_improvement: float

    # OPTIONAL - attribution_ratio fields
    ratio: float | None = None
    threshold: float | None = None

    # OPTIONAL - ransomware_signature fields
    contribution_threshold: float | None = None
    degradation_threshold: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardHackingSuspectedPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            pattern=data["pattern"],
            slot_id=data["slot_id"],
            seed_id=data["seed_id"],
            seed_contribution=data["seed_contribution"],
            total_improvement=data["total_improvement"],
            ratio=data.get("ratio"),
            threshold=data.get("threshold"),
            contribution_threshold=data.get("contribution_threshold"),
            degradation_threshold=data.get("degradation_threshold"),
        )


@dataclass(slots=True, frozen=True)
class EpisodeOutcomePayload:
    """Payload for EPISODE_OUTCOME event.

    Captures multi-objective outcomes for Pareto analysis:
    - final_accuracy: Task performance (higher = better)
    - param_ratio: Parameter efficiency (lower = better)
    - stability_score: Training stability (higher = better)

    Note: env_id may be -1 when emitted from slots (Kasmina), which don't
    know their environment context. The sentinel is replaced by the actual
    env_id in emit_with_env_context (simic/telemetry/emitters.py).
    """

    # REQUIRED
    env_id: int  # -1 = sentinel (replaced by emit_with_env_context)
    episode_idx: int
    final_accuracy: float
    param_ratio: float  # total_params / host_params
    num_fossilized: int
    num_contributing_fossilized: int  # Seeds that contributed to learning
    episode_reward: float  # Total reward for the episode
    stability_score: float  # 1 - variance(recent_losses)
    reward_mode: str  # "shaped", "simplified", etc.

    # Episode diagnostics (TELE-610)
    episode_length: int = 0  # Steps in this episode (usually max_epochs)
    outcome_type: str = "unknown"  # "success", "timeout", "early_termination"
    germinate_count: int = 0  # GERMINATE actions this episode
    prune_count: int = 0  # PRUNE actions this episode
    fossilize_count: int = 0  # FOSSILIZE actions this episode

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeOutcomePayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            env_id=data["env_id"],
            episode_idx=data["episode_idx"],
            final_accuracy=data["final_accuracy"],
            param_ratio=data["param_ratio"],
            num_fossilized=data["num_fossilized"],
            num_contributing_fossilized=data["num_contributing_fossilized"],
            episode_reward=data["episode_reward"],
            stability_score=data["stability_score"],
            reward_mode=data["reward_mode"],
            episode_length=data.get("episode_length", 0),
            outcome_type=data.get("outcome_type", "unknown"),
            germinate_count=data.get("germinate_count", 0),
            prune_count=data.get("prune_count", 0),
            fossilize_count=data.get("fossilize_count", 0),
        )


# Valid panic reasons from TolariaGovernor
GovernorPanicReason = Literal[
    "governor_nan",        # NaN or Inf detected in loss
    "governor_lobotomy",   # Loss below random guessing threshold
    "governor_divergence", # Loss exceeding statistical threshold
    "governor_rollback",   # Default fallback reason
]


@dataclass(slots=True, frozen=True)
class GovernorRollbackPayload:
    """Payload for GOVERNOR_ROLLBACK telemetry events.

    Emitted when the TolariaGovernor detects catastrophic instability
    and initiates a rollback to the last known good state.

    Two emission contexts:
    1. Initial panic detection (has loss_at_panic, loss_threshold, etc.)
    2. State dict key mismatch warning (has missing_keys, unexpected_keys)
    """

    # REQUIRED - always present
    env_id: int
    device: str
    reason: str

    # Panic context (present for initial rollback trigger)
    loss_at_panic: float | None = None
    loss_threshold: float | None = None
    consecutive_panics: int | None = None
    panic_reason: GovernorPanicReason | None = None

    # State dict mismatch context (present for key mismatch warnings)
    missing_keys: list[str] | None = None
    unexpected_keys: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GovernorRollbackPayload":
        """Parse from dict. Raises KeyError on missing required fields."""
        return cls(
            env_id=data["env_id"],
            device=data["device"],
            reason=data["reason"],
            loss_at_panic=data.get("loss_at_panic"),
            loss_threshold=data.get("loss_threshold"),
            consecutive_panics=data.get("consecutive_panics"),
            panic_reason=data.get("panic_reason"),
            missing_keys=list(data["missing_keys"]) if data.get("missing_keys") else None,
            unexpected_keys=list(data["unexpected_keys"]) if data.get("unexpected_keys") else None,
        )


# =============================================================================
# Telemetry Payload Type Union
# =============================================================================
# All telemetry event payloads are strongly typed dataclasses.
# See: docs/plans/2025-12-25-typed-telemetry-payloads-design.md

TelemetryPayload = (
    TrainingStartedPayload
    | CheckpointLoadedPayload
    | EpochCompletedPayload
    | BatchEpochCompletedPayload
    | TrendDetectedPayload
    | PPOUpdatePayload
    | MemoryWarningPayload
    | RewardHackingSuspectedPayload
    | TamiyoInitiatedPayload
    | SeedGerminatedPayload
    | SeedStageChangedPayload
    | SeedGateEvaluatedPayload
    | SeedFossilizedPayload
    | SeedPrunedPayload
    | CounterfactualMatrixPayload
    | AnalyticsSnapshotPayload
    | AnomalyDetectedPayload
    | PerformanceDegradationPayload
    | EpisodeOutcomePayload
    | GovernorRollbackPayload
)


# =============================================================================
# Telemetry Callback Type Alias
# =============================================================================

TelemetryCallback = Callable[[TelemetryEvent], None]
"""Type alias for telemetry event callbacks.

Used by components that accept a telemetry emission callback, e.g.:
    telemetry_cb: TelemetryCallback | None = None
"""
