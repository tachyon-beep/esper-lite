"""Karn Store - Data models for research telemetry.

Three-tiered data model for adaptive fidelity:
- Tier 1: EpisodeContext (once per run) - reproducibility
- Tier 2: EpochSnapshot (every epoch) - analysis workhorse
- Tier 3: DenseTrace (on trigger) - deep diagnostics

These are CONTRACTS for telemetry data. Emitters populate them,
analytics consume them, and outputs serialize them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import deque

from esper.karn.constants import AnomalyThresholds
from esper.leyline import SeedStage
from esper.karn.ingest import (
    coerce_bool_or_none,
    coerce_datetime,
    coerce_float,
    coerce_float_dict,
    coerce_float_or_none,
    coerce_int,
    coerce_path,
    coerce_seed_stage,
    coerce_str_or_none,
    filter_dataclass_kwargs,
)

__all__ = [
    # Tier 1
    "EpisodeContext",
    "HostBaseline",
    # Tier 2
    "SlotSnapshot",
    "HostSnapshot",
    "RewardComponents",
    "PolicySnapshot",
    "AdvantageStats",
    "RatioStats",
    "EpochSnapshot",
    # Tier 3
    "DenseTraceTrigger",
    "BatchMetrics",
    "GateEvaluationTrace",
    "DenseTrace",
    # Pareto Analysis
    "EpisodeOutcome",
    # Store
    "TelemetryStore",
]


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


_logger = logging.getLogger(__name__)


# =============================================================================
# Tier 1: Episode Context (Once Per Run)
# =============================================================================


@dataclass(frozen=True)
class EpisodeContext:
    """Everything needed to reproduce a training run.

    Captured once at episode start. Immutable to prevent mid-run mutations.
    """

    # Identity
    episode_id: str  # UUID for this run
    timestamp: datetime = field(default_factory=_utc_now)
    git_commit: str = ""  # Exact code version (empty if not in git repo)

    # Determinism
    base_seed: int = 42
    torch_seed: int = 42
    numpy_seed: int = 42

    # Architecture
    host_architecture: str = ""  # "cnn_3block" / "gpt_6layer"
    host_params: int = 0  # Param count before any seeds
    slot_config: tuple[tuple[str, int], ...] = ()  # (("r0c0", 64), ("r0c1", 128), ...)

    # Training config
    max_epochs: int = 75
    task_type: str = "classification"  # "classification" / "lm"
    reward_mode: str = "shaped"  # "shaped" / "sparse" / "minimal"

    # Hyperparameters (frozen dict as tuple of pairs for hashability)
    hyperparameters: tuple[tuple[str, Any], ...] = ()


@dataclass
class HostBaseline:
    """Host-only baseline for comparison.

    Captured at episode start (epoch 0, before any germination).
    Final values captured at episode end for counterfactual comparison.
    """

    # Initial state (captured at start)
    initial_loss: float = 0.0
    initial_accuracy: float = 0.0
    initial_checkpoint_path: Path | None = None  # Path to saved state

    # Final state (captured at end, for host-only counterfactual)
    final_host_only_loss: float | None = None
    final_host_only_accuracy: float | None = None


# =============================================================================
# Tier 2: Epoch Snapshots (Every Epoch)
# =============================================================================


@dataclass
class SlotSnapshot:
    """Per-slot state at a single epoch."""

    slot_id: str  # "r0c0", "r0c1", "r0c2"

    # Lifecycle state
    stage: SeedStage = SeedStage.DORMANT
    epochs_in_stage: int = 0
    seed_id: str | None = None  # Current seed identifier
    blueprint_id: str | None = None  # "conv_light", "attention", etc.

    # Parameters
    seed_params: int = 0  # Parameter count of this seed
    alpha: float = 0.0  # Blend coefficient [0, 1]

    # Attribution (key metrics)
    counterfactual_contribution: float | None = None  # acc_with - acc_without
    total_improvement: float | None = None  # acc_now - acc_at_germination
    improvement_this_epoch: float = 0.0  # delta from last epoch

    # Gate status
    last_gate_attempted: str | None = None  # "G2", "G3", etc.
    last_gate_passed: bool | None = None
    last_gate_reason: str | None = None  # Why it passed/failed

    # Traffic proxy (basic utilization signal)
    activation_magnitude: float = 0.0  # Mean absolute activation through seed path


@dataclass
class HostSnapshot:
    """Host network state at a single epoch."""

    epoch: int = 0

    # Performance
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0

    # Parameter accounting
    host_params: int = 0  # Fixed host parameters
    total_seed_params: int = 0  # Sum across all active seeds
    total_params: int = 0  # host + seeds
    fossilized_params: int = 0  # Permanently added params

    # Gradient health
    host_grad_norm: float = 0.0
    seed_grad_norms: dict[str, float] = field(default_factory=dict)  # Per-slot
    grad_isolation_leakage: float | None = None  # If monitoring isolation


@dataclass
class RewardComponents:
    """Breakdown of reward computation for debugging."""

    total: float = 0.0
    accuracy_delta: float = 0.0
    bounded_attribution: float | None = None  # For contribution mode
    compute_rent: float = 0.0
    alpha_shock: float = 0.0
    blending_warning: float = 0.0
    holding_warning: float = 0.0
    ratio_penalty: float = 0.0
    terminal_bonus: float = 0.0
    stage_bonus: float = 0.0


@dataclass
class PolicySnapshot:
    """Policy state at a single epoch."""

    # Observation (what the agent saw)
    observation_dim: int = 0
    observation_summary: dict[str, float] = field(default_factory=dict)  # Key stats

    # Action (what the agent did)
    action_op: str = ""  # "WAIT", "GERMINATE", "PRUNE", etc.
    action_slot: str | None = None
    action_blueprint: str | None = None
    action_was_masked: bool = False  # Was this action forced by mask?

    # Log probs per head (for factored actions)
    action_log_probs: dict[str, float] = field(default_factory=dict)

    # Value estimates
    value_estimate: float = 0.0
    advantage: float | None = None  # Computed during PPO update

    # Reward breakdown
    reward_total: float = 0.0
    reward_components: RewardComponents = field(default_factory=RewardComponents)

    # PPO diagnostics (from expert review P0/P1)
    kl_divergence: float | None = None  # KL(old || new)
    explained_variance: float | None = None  # 1 - Var(returns - values) / Var(returns)
    entropy: float | None = None  # Policy entropy (exploration health)


@dataclass
class AdvantageStats:
    """Aggregate advantage statistics for PPO debugging (P0 from expert review)."""

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    fraction_clipped: float = 0.0  # |A| > clip_threshold


@dataclass
class RatioStats:
    """PPO ratio statistics for debugging (P0 from expert review)."""

    mean: float = 0.0
    fraction_clipped_high: float = 0.0  # ratio > 1 + eps
    fraction_clipped_low: float = 0.0  # ratio < 1 - eps
    per_head_clip_rates: dict[str, float] = field(default_factory=dict)


@dataclass
class EpochSnapshot:
    """Complete snapshot of training state at a single epoch.

    This is the workhorse tier for analysis — captured every epoch.
    """

    epoch: int = 0
    timestamp: datetime = field(default_factory=_utc_now)

    # Host state
    host: HostSnapshot = field(default_factory=HostSnapshot)

    # Per-slot state
    slots: dict[str, SlotSnapshot] = field(default_factory=dict)

    # Policy state (if RL-controlled)
    policy: PolicySnapshot | None = None

    # PPO update stats (if update happened this epoch)
    advantage_stats: AdvantageStats | None = None
    ratio_stats: RatioStats | None = None

    # DDP forward compatibility (P0 from expert review)
    rank: int = 0
    world_size: int = 1
    is_reduced: bool = False  # True if aggregated across ranks


# =============================================================================
# Tier 3: Dense Traces (Adaptive / On Trigger)
# =============================================================================


@dataclass
class DenseTraceTrigger:
    """Conditions that trigger dense trace capture."""

    # Stage transitions (always interesting)
    stage_transition: bool = True

    # Anomalies
    loss_spike_threshold: float = AnomalyThresholds.LOSS_SPIKE_MULTIPLIER
    accuracy_drop_threshold: float = AnomalyThresholds.ACCURACY_DROP_POINTS
    gradient_explosion: float = AnomalyThresholds.GRADIENT_EXPLOSION_MULTIPLIER

    # Gate events
    gate_failure: bool = True  # Any G0-G5 gate fails

    # Policy anomalies
    value_collapse: bool = True  # Critic predicts same value everywhere
    entropy_collapse: bool = True  # Policy becomes deterministic

    # Manual override
    force_dense: bool = False  # Research mode toggle


@dataclass
class BatchMetrics:
    """Per-batch metrics for dense trace."""

    epoch: int = 0
    batch_idx: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    host_grad_norm: float = 0.0
    seed_grad_norms: dict[str, float] = field(default_factory=dict)
    isolation_leakage: float | None = None


@dataclass
class GateEvaluationTrace:
    """Detailed trace of gate evaluation (for debugging gate failures)."""

    gate_id: str = ""  # "G0", "G1", etc.
    slot_id: str = ""
    passed: bool = False
    reason: str = ""
    metrics_at_evaluation: dict[str, float] = field(default_factory=dict)
    thresholds_used: dict[str, float] = field(default_factory=dict)


@dataclass
class DenseTrace:
    """Dense diagnostic trace triggered by anomaly or stage transition.

    Captures per-batch metrics and gradient details for a window of epochs.
    """

    trigger_reason: str = ""  # "loss_spike", "gate_failure", etc.
    window_start_epoch: int = 0
    window_end_epoch: int = 0
    timestamp: datetime = field(default_factory=_utc_now)

    # Per-batch metrics
    batch_metrics: list[BatchMetrics] = field(default_factory=list)

    # Gradient details
    gradient_histograms: dict[str, list[float]] = field(default_factory=dict)
    gradient_flow: dict[str, float] = field(default_factory=dict)

    # Activation statistics
    activation_means: dict[str, float] = field(default_factory=dict)
    activation_stds: dict[str, float] = field(default_factory=dict)
    dead_neuron_counts: dict[str, int] = field(default_factory=dict)

    # Gate internals (if gate event)
    gate_evaluation_details: GateEvaluationTrace | None = None


# =============================================================================
# Episode Outcome (Pareto Analysis)
# =============================================================================


@dataclass(frozen=True)
class EpisodeOutcome:
    """Multi-objective outcome for Pareto analysis.

    Captures the key metrics we're optimizing:
    - final_accuracy: Task performance (higher = better)
    - param_ratio: Parameter efficiency (lower = better)
    - stability_score: Training stability (higher = better)
    """

    env_idx: int
    episode_idx: int
    final_accuracy: float
    param_ratio: float  # total_params / host_params
    num_fossilized: int
    num_contributing_fossilized: int  # Seeds that contributed to learning
    episode_reward: float  # Total reward for the episode
    stability_score: float  # 1 - variance(recent_losses)
    reward_mode: str  # "shaped", "simplified", etc.
    timestamp: datetime = field(default_factory=_utc_now)

    def dominates(self, other: "EpisodeOutcome") -> bool:
        """Pareto dominance check.

        Returns True if self dominates other (better or equal on all objectives,
        strictly better on at least one).

        Objectives (higher is better):
        - final_accuracy
        - stability_score

        Objectives (lower is better):
        - param_ratio
        """
        # Check: self >= other on all objectives
        geq_accuracy = self.final_accuracy >= other.final_accuracy
        geq_stability = self.stability_score >= other.stability_score
        leq_ratio = self.param_ratio <= other.param_ratio

        all_geq = geq_accuracy and geq_stability and leq_ratio

        # Check: self > other on at least one objective
        gt_accuracy = self.final_accuracy > other.final_accuracy
        gt_stability = self.stability_score > other.stability_score
        lt_ratio = self.param_ratio < other.param_ratio

        any_gt = gt_accuracy or gt_stability or lt_ratio

        return all_geq and any_gt

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "env_idx": self.env_idx,
            "episode_idx": self.episode_idx,
            "reward_mode": self.reward_mode,
            "final_accuracy": self.final_accuracy,
            "param_ratio": self.param_ratio,
            "stability_score": self.stability_score,
            "num_fossilized": self.num_fossilized,
            "num_contributing_fossilized": self.num_contributing_fossilized,
            "episode_reward": self.episode_reward,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Telemetry Store (In-Memory)
# =============================================================================


@dataclass
class TelemetryStore:
    """In-memory store for current episode telemetry.

    Provides bounded storage with automatic eviction for long runs.
    """

    # Tier 1
    context: EpisodeContext | None = None
    baseline: HostBaseline | None = None

    # Tier 2 (bounded history)
    epoch_snapshots: deque[EpochSnapshot] = field(
        default_factory=lambda: deque(maxlen=500)  # Keep last 500 epochs
    )

    # Tier 3 (bounded)
    dense_traces: deque[DenseTrace] = field(
        default_factory=lambda: deque(maxlen=20)  # Keep last 20 traces
    )

    # Current epoch being built
    current_epoch: EpochSnapshot | None = None

    def start_episode(self, context: EpisodeContext) -> None:
        """Initialize store for new episode."""
        self.context = context
        self.baseline = HostBaseline()
        self.epoch_snapshots.clear()
        self.dense_traces.clear()
        self.current_epoch = None

    def start_epoch(self, epoch: int) -> EpochSnapshot:
        """Start building a new epoch snapshot."""
        self.current_epoch = EpochSnapshot(epoch=epoch)
        return self.current_epoch

    def commit_epoch(self) -> None:
        """Commit current epoch to history."""
        if self.current_epoch:
            self.epoch_snapshots.append(self.current_epoch)
            self.current_epoch = None

    def add_dense_trace(self, trace: DenseTrace) -> None:
        """Add a dense trace (auto-evicts oldest if at capacity)."""
        self.dense_traces.append(trace)

    def get_recent_epochs(self, n: int = 10) -> list[EpochSnapshot]:
        """Get the n most recent epoch snapshots."""
        return list(self.epoch_snapshots)[-n:]

    @property
    def latest_epoch(self) -> EpochSnapshot | None:
        """Get the most recent committed epoch."""
        return self.epoch_snapshots[-1] if self.epoch_snapshots else None

    # =========================================================================
    # TODO: [FUTURE] Store-Based Analytics
    # =========================================================================
    #
    # The following research queries operate on stored epoch snapshots.
    # Candidates for future TelemetryStore methods or a companion Analytics class.
    #
    # Trajectory Analysis:
    #   - accuracy_trajectory() -> list[TrajectoryPoint]: Val accuracy over epochs
    #   - loss_trajectory() -> list[TrajectoryPoint]: Val loss over epochs
    #   - gradient_norm_trajectory() -> list[TrajectoryPoint]: Host grad norms
    #
    # Convergence Detection:
    #   - detect_convergence(window, threshold) -> ConvergenceInfo: Plateau detection
    #   - best_epoch() -> (epoch, accuracy): Peak performance point
    #   - improvement_rate(window) -> list[float]: Rolling accuracy delta
    #
    # Per-Slot Aggregation:
    #   - slot_summary(slot_id) -> SlotSummary: Stats for specific slot position
    #   - slot_contributions() -> dict[slot_id, list[float]]: Counterfactual over time
    #   - compare_slots() -> dict[slot_id, metrics]: Cross-slot comparison
    #
    # Stage Duration Analysis:
    #   - stage_durations(slot_id) -> dict[stage, list[int]]: Time in each stage
    #   - mean_stage_duration(stage) -> float: Average across all slots
    #
    # Episode Summary:
    #   - episode_summary() -> EpisodeSummary: Comprehensive end-of-run stats
    #
    # Implementation note: Can be added as methods here or as a separate
    # StoreAnalytics class that wraps TelemetryStore.
    # =========================================================================

    def export_jsonl(self, path: Path | str) -> int:
        """Export store contents to JSONL file.

        Args:
            path: Path to output JSONL file

        Returns:
            Number of records written
        """
        import json
        from dataclasses import asdict, is_dataclass

        def serialize(obj: Any) -> Any:
            """Serialize dataclass or primitive to JSON-safe dict."""
            if is_dataclass(obj) and not isinstance(obj, type):
                return asdict(obj)
            elif isinstance(obj, deque):
                return list(obj)
            return obj

        def json_default(obj: Any) -> str:
            """Handle non-serializable types for json.dumps."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            # hasattr AUTHORIZED by John on 2025-12-14 15:00:00 UTC
            # Justification: Serialization - handle Enum values in JSON export
            if hasattr(obj, "name") and hasattr(obj, "value"):
                return str(obj.name)  # Serialize enum as name string
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        path = Path(path)
        count = 0

        with open(path, "w") as f:
            # Write context
            if self.context:
                f.write(json.dumps({"type": "context", "data": serialize(self.context)}, default=json_default) + "\n")
                count += 1

            # Write baseline
            if self.baseline:
                f.write(json.dumps({"type": "baseline", "data": serialize(self.baseline)}, default=json_default) + "\n")
                count += 1

            # Write epochs
            for epoch in self.epoch_snapshots:
                f.write(json.dumps({"type": "epoch", "data": serialize(epoch)}, default=json_default) + "\n")
                count += 1

            # Write dense traces
            for trace in self.dense_traces:
                f.write(json.dumps({"type": "dense_trace", "data": serialize(trace)}, default=json_default) + "\n")
                count += 1

        return count

    @classmethod
    def import_jsonl(cls, path: Path | str) -> "TelemetryStore":
        """Import store contents from JSONL file.

        Args:
            path: Path to JSONL file created by export_jsonl()

        Returns:
            TelemetryStore populated from file
        """
        import json

        path = Path(path)
        store = cls()

        def _coerce_tuple_pairs(value: Any, *, field: str) -> tuple[tuple[str, Any], ...]:
            if value is None:
                return ()
            seq: tuple[Any, ...] | list[Any]
            if isinstance(value, tuple):
                seq = value
            elif isinstance(value, list):
                seq = value
            else:
                _logger.warning("Invalid %s=%r (expected list/tuple of pairs); using empty tuple", field, value)
                return ()

            result_list: list[tuple[str, Any]] = []
            for item in seq:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    result_list.append((str(item[0]), item[1]))
                else:
                    _logger.warning("Invalid %s item=%r (expected pair); skipping", field, item)
            return tuple(result_list)

        def _coerce_slot_config(value: Any) -> tuple[tuple[str, int], ...]:
            if value is None:
                return ()
            seq: tuple[Any, ...] | list[Any]
            if isinstance(value, tuple):
                seq = value
            elif isinstance(value, list):
                seq = value
            else:
                _logger.warning("Invalid slot_config=%r (expected list/tuple of pairs); using empty tuple", value)
                return ()
            result_list: list[tuple[str, int]] = []
            for item in seq:
                if not (isinstance(item, (list, tuple)) and len(item) == 2):
                    _logger.warning("Invalid slot_config item=%r (expected pair); skipping", item)
                    continue
                slot_id = str(item[0])
                width = coerce_int(item[1], field="slot_config.width", default=0, minimum=0)
                result_list.append((slot_id, width))
            return tuple(result_list)

        def _parse_episode_context(raw: dict[str, Any]) -> EpisodeContext:
            data = filter_dataclass_kwargs(EpisodeContext, raw, context="EpisodeContext")
            ts = coerce_datetime(data.get("timestamp"), field="EpisodeContext.timestamp", default=None)
            if ts is None:
                data.pop("timestamp", None)
            else:
                data["timestamp"] = ts

            data["base_seed"] = coerce_int(data.get("base_seed"), field="EpisodeContext.base_seed", default=42, minimum=0)
            data["torch_seed"] = coerce_int(data.get("torch_seed"), field="EpisodeContext.torch_seed", default=42, minimum=0)
            data["numpy_seed"] = coerce_int(data.get("numpy_seed"), field="EpisodeContext.numpy_seed", default=42, minimum=0)
            data["max_epochs"] = coerce_int(data.get("max_epochs"), field="EpisodeContext.max_epochs", default=75, minimum=1)

            data["hyperparameters"] = _coerce_tuple_pairs(data.get("hyperparameters"), field="EpisodeContext.hyperparameters")
            data["slot_config"] = _coerce_slot_config(data.get("slot_config"))
            return EpisodeContext(**data)

        def _parse_host_baseline(raw: dict[str, Any]) -> HostBaseline:
            data = filter_dataclass_kwargs(HostBaseline, raw, context="HostBaseline")
            if "initial_checkpoint_path" in data:
                data["initial_checkpoint_path"] = coerce_path(
                    data.get("initial_checkpoint_path"),
                    field="HostBaseline.initial_checkpoint_path",
                )
            return HostBaseline(**data)

        def _parse_slot_snapshot(raw: dict[str, Any]) -> SlotSnapshot:
            data = filter_dataclass_kwargs(SlotSnapshot, raw, context="SlotSnapshot")
            data["stage"] = coerce_seed_stage(data.get("stage"), field="SlotSnapshot.stage", default=SeedStage.DORMANT)
            data["epochs_in_stage"] = coerce_int(
                data.get("epochs_in_stage"), field="SlotSnapshot.epochs_in_stage", default=0, minimum=0
            )
            data["seed_params"] = coerce_int(data.get("seed_params"), field="SlotSnapshot.seed_params", default=0, minimum=0)
            data["alpha"] = coerce_float(data.get("alpha"), field="SlotSnapshot.alpha", default=0.0)
            data["counterfactual_contribution"] = coerce_float_or_none(
                data.get("counterfactual_contribution"), field="SlotSnapshot.counterfactual_contribution"
            )
            data["total_improvement"] = coerce_float_or_none(
                data.get("total_improvement"), field="SlotSnapshot.total_improvement"
            )
            data["improvement_this_epoch"] = coerce_float(
                data.get("improvement_this_epoch"), field="SlotSnapshot.improvement_this_epoch", default=0.0
            )
            data["activation_magnitude"] = coerce_float(
                data.get("activation_magnitude"), field="SlotSnapshot.activation_magnitude", default=0.0
            )
            data["seed_id"] = coerce_str_or_none(data.get("seed_id"), field="SlotSnapshot.seed_id")
            data["blueprint_id"] = coerce_str_or_none(data.get("blueprint_id"), field="SlotSnapshot.blueprint_id")
            return SlotSnapshot(**data)

        def _parse_host_snapshot(raw: dict[str, Any]) -> HostSnapshot:
            data = filter_dataclass_kwargs(HostSnapshot, raw, context="HostSnapshot")
            data["epoch"] = coerce_int(data.get("epoch"), field="HostSnapshot.epoch", default=0, minimum=0)
            data["train_loss"] = coerce_float(data.get("train_loss"), field="HostSnapshot.train_loss", default=0.0)
            data["train_accuracy"] = coerce_float(
                data.get("train_accuracy"), field="HostSnapshot.train_accuracy", default=0.0
            )
            data["val_loss"] = coerce_float(data.get("val_loss"), field="HostSnapshot.val_loss", default=0.0)
            data["val_accuracy"] = coerce_float(data.get("val_accuracy"), field="HostSnapshot.val_accuracy", default=0.0)
            data["host_params"] = coerce_int(data.get("host_params"), field="HostSnapshot.host_params", default=0, minimum=0)
            data["total_seed_params"] = coerce_int(
                data.get("total_seed_params"), field="HostSnapshot.total_seed_params", default=0, minimum=0
            )
            data["total_params"] = coerce_int(data.get("total_params"), field="HostSnapshot.total_params", default=0, minimum=0)
            data["fossilized_params"] = coerce_int(
                data.get("fossilized_params"), field="HostSnapshot.fossilized_params", default=0, minimum=0
            )
            data["host_grad_norm"] = coerce_float(data.get("host_grad_norm"), field="HostSnapshot.host_grad_norm", default=0.0)
            data["seed_grad_norms"] = coerce_float_dict(
                data.get("seed_grad_norms"), field="HostSnapshot.seed_grad_norms"
            )
            data["grad_isolation_leakage"] = coerce_float_or_none(
                data.get("grad_isolation_leakage"), field="HostSnapshot.grad_isolation_leakage"
            )
            return HostSnapshot(**data)

        def _parse_reward_components(raw: dict[str, Any]) -> RewardComponents:
            data = filter_dataclass_kwargs(RewardComponents, raw, context="RewardComponents")
            for key in (
                "total",
                "accuracy_delta",
                "compute_rent",
                "alpha_shock",
                "blending_warning",
                "holding_warning",
                "ratio_penalty",
                "terminal_bonus",
            ):
                data[key] = coerce_float(data.get(key), field=f"RewardComponents.{key}", default=0.0)
            data["bounded_attribution"] = coerce_float_or_none(
                data.get("bounded_attribution"), field="RewardComponents.bounded_attribution"
            )
            return RewardComponents(**data)

        def _parse_policy_snapshot(raw: dict[str, Any]) -> PolicySnapshot:
            data = filter_dataclass_kwargs(PolicySnapshot, raw, context="PolicySnapshot")
            data["observation_dim"] = coerce_int(
                data.get("observation_dim"), field="PolicySnapshot.observation_dim", default=0, minimum=0
            )
            if "observation_summary" in data:
                data["observation_summary"] = coerce_float_dict(
                    data.get("observation_summary"), field="PolicySnapshot.observation_summary"
                )
            data["action_op"] = coerce_str_or_none(data.get("action_op"), field="PolicySnapshot.action_op") or ""
            data["action_slot"] = coerce_str_or_none(data.get("action_slot"), field="PolicySnapshot.action_slot")
            data["action_blueprint"] = coerce_str_or_none(
                data.get("action_blueprint"), field="PolicySnapshot.action_blueprint"
            )

            masked = coerce_bool_or_none(data.get("action_was_masked"), field="PolicySnapshot.action_was_masked")
            data["action_was_masked"] = False if masked is None else masked

            if "action_log_probs" in data:
                data["action_log_probs"] = coerce_float_dict(
                    data.get("action_log_probs"), field="PolicySnapshot.action_log_probs"
                )

            data["value_estimate"] = coerce_float(data.get("value_estimate"), field="PolicySnapshot.value_estimate", default=0.0)
            data["advantage"] = coerce_float_or_none(data.get("advantage"), field="PolicySnapshot.advantage")
            data["reward_total"] = coerce_float(data.get("reward_total"), field="PolicySnapshot.reward_total", default=0.0)

            reward_components_raw = data.get("reward_components")
            if isinstance(reward_components_raw, dict):
                data["reward_components"] = _parse_reward_components(reward_components_raw)
            else:
                data.pop("reward_components", None)

            data["kl_divergence"] = coerce_float_or_none(data.get("kl_divergence"), field="PolicySnapshot.kl_divergence")
            data["explained_variance"] = coerce_float_or_none(
                data.get("explained_variance"), field="PolicySnapshot.explained_variance"
            )
            data["entropy"] = coerce_float_or_none(data.get("entropy"), field="PolicySnapshot.entropy")
            return PolicySnapshot(**data)

        def _parse_advantage_stats(raw: dict[str, Any]) -> AdvantageStats:
            data = filter_dataclass_kwargs(AdvantageStats, raw, context="AdvantageStats")
            for key in ("mean", "std", "min", "max", "fraction_clipped"):
                data[key] = coerce_float(data.get(key), field=f"AdvantageStats.{key}", default=0.0)
            return AdvantageStats(**data)

        def _parse_ratio_stats(raw: dict[str, Any]) -> RatioStats:
            data = filter_dataclass_kwargs(RatioStats, raw, context="RatioStats")
            for key in ("mean", "fraction_clipped_high", "fraction_clipped_low"):
                data[key] = coerce_float(data.get(key), field=f"RatioStats.{key}", default=0.0)
            data["per_head_clip_rates"] = coerce_float_dict(
                data.get("per_head_clip_rates"), field="RatioStats.per_head_clip_rates"
            )
            return RatioStats(**data)

        def _parse_epoch_snapshot(raw: dict[str, Any]) -> EpochSnapshot:
            data = filter_dataclass_kwargs(EpochSnapshot, raw, context="EpochSnapshot")
            ts = coerce_datetime(data.get("timestamp"), field="EpochSnapshot.timestamp", default=None)
            if ts is None:
                data.pop("timestamp", None)
            else:
                data["timestamp"] = ts

            data["epoch"] = coerce_int(data.get("epoch"), field="EpochSnapshot.epoch", default=0, minimum=0)
            data["rank"] = coerce_int(data.get("rank"), field="EpochSnapshot.rank", default=0, minimum=0)
            data["world_size"] = coerce_int(data.get("world_size"), field="EpochSnapshot.world_size", default=1, minimum=1)

            reduced = coerce_bool_or_none(data.get("is_reduced"), field="EpochSnapshot.is_reduced")
            data["is_reduced"] = False if reduced is None else reduced

            host_raw = data.get("host")
            if isinstance(host_raw, dict):
                data["host"] = _parse_host_snapshot(host_raw)

            slots_raw = data.get("slots")
            if isinstance(slots_raw, dict):
                data["slots"] = {k: _parse_slot_snapshot(v) for k, v in slots_raw.items() if isinstance(v, dict)}

            policy_raw = data.get("policy")
            if isinstance(policy_raw, dict):
                data["policy"] = _parse_policy_snapshot(policy_raw)
            else:
                data["policy"] = None

            advantage_raw = data.get("advantage_stats")
            if isinstance(advantage_raw, dict):
                data["advantage_stats"] = _parse_advantage_stats(advantage_raw)
            else:
                data["advantage_stats"] = None

            ratio_raw = data.get("ratio_stats")
            if isinstance(ratio_raw, dict):
                data["ratio_stats"] = _parse_ratio_stats(ratio_raw)
            else:
                data["ratio_stats"] = None

            return EpochSnapshot(**data)

        def _parse_batch_metrics(raw: dict[str, Any]) -> BatchMetrics:
            data = filter_dataclass_kwargs(BatchMetrics, raw, context="BatchMetrics")
            data["epoch"] = coerce_int(data.get("epoch"), field="BatchMetrics.epoch", default=0, minimum=0)
            data["batch_idx"] = coerce_int(data.get("batch_idx"), field="BatchMetrics.batch_idx", default=0, minimum=0)
            data["loss"] = coerce_float(data.get("loss"), field="BatchMetrics.loss", default=0.0)
            data["accuracy"] = coerce_float(data.get("accuracy"), field="BatchMetrics.accuracy", default=0.0)
            data["host_grad_norm"] = coerce_float(
                data.get("host_grad_norm"), field="BatchMetrics.host_grad_norm", default=0.0
            )
            data["seed_grad_norms"] = coerce_float_dict(data.get("seed_grad_norms"), field="BatchMetrics.seed_grad_norms")
            data["isolation_leakage"] = coerce_float_or_none(
                data.get("isolation_leakage"), field="BatchMetrics.isolation_leakage"
            )
            return BatchMetrics(**data)

        def _parse_gate_evaluation_trace(raw: dict[str, Any]) -> GateEvaluationTrace:
            data = filter_dataclass_kwargs(GateEvaluationTrace, raw, context="GateEvaluationTrace")
            data["gate_id"] = coerce_str_or_none(data.get("gate_id"), field="GateEvaluationTrace.gate_id") or ""
            data["slot_id"] = coerce_str_or_none(data.get("slot_id"), field="GateEvaluationTrace.slot_id") or ""
            passed = coerce_bool_or_none(data.get("passed"), field="GateEvaluationTrace.passed")
            data["passed"] = False if passed is None else passed
            data["reason"] = coerce_str_or_none(data.get("reason"), field="GateEvaluationTrace.reason") or ""
            data["metrics_at_evaluation"] = coerce_float_dict(
                data.get("metrics_at_evaluation"), field="GateEvaluationTrace.metrics_at_evaluation"
            )
            data["thresholds_used"] = coerce_float_dict(
                data.get("thresholds_used"), field="GateEvaluationTrace.thresholds_used"
            )
            return GateEvaluationTrace(**data)

        def _parse_dense_trace(raw: dict[str, Any]) -> DenseTrace:
            data = filter_dataclass_kwargs(DenseTrace, raw, context="DenseTrace")
            ts = coerce_datetime(data.get("timestamp"), field="DenseTrace.timestamp", default=None)
            if ts is None:
                data.pop("timestamp", None)
            else:
                data["timestamp"] = ts

            data["window_start_epoch"] = coerce_int(
                data.get("window_start_epoch"), field="DenseTrace.window_start_epoch", default=0, minimum=0
            )
            data["window_end_epoch"] = coerce_int(
                data.get("window_end_epoch"), field="DenseTrace.window_end_epoch", default=0, minimum=0
            )

            batch_metrics_raw = data.get("batch_metrics")
            if isinstance(batch_metrics_raw, list):
                data["batch_metrics"] = [
                    _parse_batch_metrics(item)
                    for item in batch_metrics_raw
                    if isinstance(item, dict)
                ]
            else:
                data.pop("batch_metrics", None)

            gate_raw = data.get("gate_evaluation_details")
            if isinstance(gate_raw, dict):
                data["gate_evaluation_details"] = _parse_gate_evaluation_trace(gate_raw)
            elif gate_raw is not None:
                data.pop("gate_evaluation_details", None)

            return DenseTrace(**data)

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_type = record.get("type")
                data = record.get("data", {})

                if record_type == "context":
                    if isinstance(data, dict):
                        store.context = _parse_episode_context(data)
                elif record_type == "baseline":
                    if isinstance(data, dict):
                        store.baseline = _parse_host_baseline(data)
                elif record_type == "epoch":
                    if isinstance(data, dict):
                        store.epoch_snapshots.append(_parse_epoch_snapshot(data))
                elif record_type == "dense_trace":
                    if isinstance(data, dict):
                        store.dense_traces.append(_parse_dense_trace(data))
                else:
                    _logger.debug("Ignoring unknown record type in %s: %r", path, record_type)

        return store

    @classmethod
    def import_from_nissa_dir(cls, dir_path: Path | str) -> "TelemetryStore":
        """Import events from a Nissa DirectoryOutput folder.

        Nissa DirectoryOutput creates timestamped folders with events.jsonl.
        This method reads those events and reconstructs a TelemetryStore.

        Args:
            dir_path: Path to Nissa output directory (contains events.jsonl)

        Returns:
            TelemetryStore populated from the events
        """
        import json

        dir_path = Path(dir_path)
        events_file = dir_path / "events.jsonl"

        if not events_file.exists():
            raise FileNotFoundError(f"No events.jsonl found in {dir_path}")

        store = cls()
        current_epoch_num = -1

        with open(events_file) as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                event_type = record.get("event_type", "")
                data = record.get("data", {})
                epoch = record.get("epoch") or data.get("epoch", 0)

                # Reconstruct store from events
                if event_type == "TRAINING_STARTED":
                    # Convert hyperparams dict to tuple of pairs for frozen dataclass
                    hyperparams = data.get("hyperparams", {})
                    hyperparameters = tuple(hyperparams.items()) if isinstance(hyperparams, dict) else ()
                    store.context = EpisodeContext(
                        episode_id=data.get("episode_id", "imported"),
                        base_seed=data.get("seed", 42),
                        task_type=data.get("task", "classification"),
                        reward_mode=data.get("reward_mode", "shaped"),
                        max_epochs=data.get("max_epochs", 75),
                        hyperparameters=hyperparameters,
                    )
                elif event_type == "EPOCH_COMPLETED":
                    if epoch != current_epoch_num:
                        if store.current_epoch:
                            store.commit_epoch()
                        store.start_epoch(epoch)
                        current_epoch_num = epoch
                    if store.current_epoch:
                        store.current_epoch.host.val_loss = data.get("val_loss", 0.0)
                        store.current_epoch.host.val_accuracy = data.get("val_accuracy", 0.0)
                elif event_type == "REWARD_COMPUTED":
                    # Legacy event type (kept for backwards compat with old JSONL files)
                    if store.current_epoch and not store.current_epoch.policy:
                        store.current_epoch.policy = PolicySnapshot()
                    if store.current_epoch and store.current_epoch.policy:
                        store.current_epoch.policy.reward_total = data.get("total_reward", 0.0)
                        store.current_epoch.policy.action_op = data.get("action_name", "")
                elif event_type == "ANALYTICS_SNAPSHOT":
                    # New event type: handle kind="last_action" for policy data
                    kind = data.get("kind")
                    if kind == "last_action" and store.current_epoch:
                        if not store.current_epoch.policy:
                            store.current_epoch.policy = PolicySnapshot()
                        policy = store.current_epoch.policy
                        if "total_reward" in data:
                            policy.reward_total = data.get("total_reward", 0.0)
                        if "action_name" in data:
                            policy.action_op = data.get("action_name", "")
                        if "value_estimate" in data:
                            policy.value_estimate = data.get("value_estimate", 0.0)

        # Commit final epoch
        if store.current_epoch:
            store.commit_epoch()

        return store
