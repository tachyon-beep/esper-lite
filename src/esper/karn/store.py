"""Karn Store - Data models for research telemetry.

Three-tiered data model for adaptive fidelity:
- Tier 1: EpisodeContext (once per run) - reproducibility
- Tier 2: EpochSnapshot (every epoch) - analysis workhorse
- Tier 3: DenseTrace (on trigger) - deep diagnostics

These are CONTRACTS for telemetry data. Emitters populate them,
analytics consume them, and outputs serialize them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import deque

from esper.karn.constants import AnomalyThresholds
from esper.leyline import SeedStage


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


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
    blending_warning: float = 0.0
    probation_warning: float = 0.0
    ratio_penalty: float = 0.0
    terminal_bonus: float = 0.0


@dataclass
class PolicySnapshot:
    """Policy state at a single epoch."""

    # Observation (what the agent saw)
    observation_dim: int = 0
    observation_summary: dict[str, float] = field(default_factory=dict)  # Key stats

    # Action (what the agent did)
    action_op: str = ""  # "WAIT", "GERMINATE", "CULL", etc.
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

    def export_jsonl(self, path: Path | str) -> int:
        """Export store contents to JSONL file.

        Args:
            path: Path to output JSONL file

        Returns:
            Number of records written
        """
        import json
        from dataclasses import asdict, is_dataclass

        def serialize(obj):
            """Serialize dataclass or primitive to JSON-safe dict."""
            if is_dataclass(obj) and not isinstance(obj, type):
                return asdict(obj)
            elif isinstance(obj, deque):
                return list(obj)
            return obj

        def json_default(obj):
            """Handle non-serializable types for json.dumps."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            # hasattr AUTHORIZED by John on 2025-12-14 15:00:00 UTC
            # Justification: Serialization - handle Enum values in JSON export
            if hasattr(obj, "name") and hasattr(obj, "value"):
                return obj.name  # Serialize enum as name string
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

        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_type = record.get("type")
                data = record.get("data", {})

                if record_type == "context":
                    store.context = EpisodeContext(**data)
                elif record_type == "baseline":
                    store.baseline = HostBaseline(**data)
                elif record_type == "epoch":
                    # Reconstruct nested dataclasses
                    if "host" in data:
                        data["host"] = HostSnapshot(**data["host"])
                    if "policy" in data and data["policy"]:
                        data["policy"] = PolicySnapshot(**data["policy"])
                    if "slots" in data:
                        data["slots"] = {k: SlotSnapshot(**v) for k, v in data["slots"].items()}
                    store.epoch_snapshots.append(EpochSnapshot(**data))
                elif record_type == "dense_trace":
                    store.dense_traces.append(DenseTrace(**data))

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
                    if store.current_epoch and not store.current_epoch.policy:
                        store.current_epoch.policy = PolicySnapshot()
                    if store.current_epoch and store.current_epoch.policy:
                        store.current_epoch.policy.reward_total = data.get("total_reward", 0.0)
                        store.current_epoch.policy.action_op = data.get("action_name", "")

        # Commit final epoch
        if store.current_epoch:
            store.commit_epoch()

        return store
