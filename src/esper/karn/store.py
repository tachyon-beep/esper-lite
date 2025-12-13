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
from enum import Enum, auto
from pathlib import Path
from typing import Any
from collections import deque


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
    slot_config: tuple[tuple[str, int], ...] = ()  # (("early", 64), ("mid", 128), ...)

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


class SeedStage(Enum):
    """Seed lifecycle stages (mirrors Kasmina's SeedStage)."""

    DORMANT = 1
    GERMINATED = 2
    BLENDING = 3
    PROBATION = 4
    INTEGRATED = 5
    FOSSILIZED = 6
    CULLED = 7


@dataclass
class SlotSnapshot:
    """Per-slot state at a single epoch."""

    slot_id: str  # "early", "mid", "late"

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
    loss_spike_threshold: float = 2.0  # > 2x rolling average
    accuracy_drop_threshold: float = 5.0  # > 5% drop epoch-over-epoch
    gradient_explosion: float = 100.0  # grad_norm > 100x typical

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
