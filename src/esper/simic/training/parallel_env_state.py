"""Parallel environment state for vectorized PPO training.

This module defines the ParallelEnvState dataclass which holds all state
for a single parallel training environment, including:
- Model and optimizer references
- Episode tracking (rewards, action counts)
- Pre-allocated GPU accumulators (to avoid per-epoch allocation churn)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, DefaultDict

import torch

from esper.leyline import LifecycleOp

if TYPE_CHECKING:
    from torch.amp.grad_scaler import GradScaler
    from esper.tolaria import TolariaGovernor
    from esper.karn.health import HealthMonitor
    from esper.simic.attribution import CounterfactualHelper
    from esper.kasmina.host import MorphogeneticModel
    from esper.tamiyo.tracker import SignalTracker
    from esper.leyline import TelemetryCallback


@dataclass(slots=True)
class ParallelEnvState:
    """State for a single parallel environment with CUDA stream for async execution.

    DataLoaders are now SHARED via SharedBatchIterator - batches are pre-split
    and data is pre-moved to each env's device with non_blocking=True.
    """
    model: "MorphogeneticModel"
    host_optimizer: torch.optim.Optimizer
    signal_tracker: "SignalTracker"  # Tracks training signals (accuracy, loss trends)
    governor: "TolariaGovernor"  # Fail-safe watchdog for catastrophic failure detection
    needs_governor_snapshot: bool = False  # Flag to trigger snapshot after fossilization
    health_monitor: "HealthMonitor | None" = None  # System health monitoring (GPU memory warnings)
    counterfactual_helper: "CounterfactualHelper | None" = None  # Shapley value analysis at episode end
    seed_optimizers: dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    env_device: str = "cpu"  # Device this env runs on
    stream: torch.cuda.Stream | None = None  # CUDA stream for async execution
    augment_generator: torch.Generator | None = None  # RNG for GPU augmentations
    scaler: "GradScaler | None" = None  # Per-env AMP scaler for FP16 mixed precision
    seeds_created: int = 0
    seeds_fossilized: int = 0  # Total seeds fossilized this episode
    contributing_fossilized: int = 0  # Seeds with total_improvement >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
    # Action counters for episode diagnostics (TELE-610)
    germinate_count: int = 0
    prune_count: int = 0
    fossilize_count: int = 0
    episode_rewards: list[float] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)
    successful_action_counts: dict[str, int] = field(default_factory=dict)
    action_enum: type | None = None
    # Metrics for current batch step
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    # "Committed" accuracy: evaluation with only fossilized seeds enabled.
    committed_val_acc: float = 0.0
    committed_acc_history: list[float] = field(default_factory=list)
    # Ransomware-resistant reward: track accuracy at germination for progress calculation
    acc_at_germination: dict[str, float] = field(default_factory=dict)
    # Escrow attribution ledger (RewardMode.ESCROW): per-slot unrealised credit balance.
    escrow_credit: dict[str, float] = field(default_factory=dict)
    # Maximum accuracy achieved during episode (for sparse reward)
    host_max_acc: float = 0.0
    # Pre-allocated accumulators to avoid per-epoch tensor allocation churn
    train_loss_accum: torch.Tensor | None = None
    train_correct_accum: torch.Tensor | None = None
    val_loss_accum: torch.Tensor | None = None
    val_correct_accum: torch.Tensor | None = None
    # Per-slot counterfactual accumulators for multi-slot reward attribution
    cf_correct_accums: dict[str, torch.Tensor] = field(default_factory=dict)
    cf_totals: dict[str, int] = field(default_factory=dict)
    # "All disabled" counterfactual for true pair synergy measurement
    cf_all_disabled_accum: torch.Tensor | None = None
    cf_all_disabled_total: int = 0
    # Pair counterfactual accumulators for 3-4 seeds (key: tuple of slot indices)
    cf_pair_accums: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    cf_pair_totals: dict[tuple[int, int], int] = field(default_factory=dict)
    telemetry_cb: "TelemetryCallback | None" = None  # Callback wired when telemetry is enabled
    # Per-slot EMA tracking for seed gradient ratio (for G2 gate)
    # Smooths per-step ratio noise with momentum=0.9
    gradient_ratio_ema: dict[str, float] = field(default_factory=dict)
    # Pending auto-prune penalty to be applied on next reward computation
    # (DRL Expert review 2025-12-17: prevents degenerate WAIT-spam policies
    # that rely on environment cleanup rather than proactive lifecycle management)
    pending_auto_prune_penalty: float = 0.0
    # Previous alpha/param snapshots for convex shock penalty (Phase 5)
    prev_slot_alphas: dict[str, float] = field(default_factory=dict)
    prev_slot_params: dict[str, int] = field(default_factory=dict)
    # Scaffold hindsight credit tracking (Phase 3.2)
    # Maps scaffold_slot -> list of (boost_given, beneficiary_slot, epoch_of_boost)
    # Using defaultdict to auto-create empty lists on first access
    scaffold_boost_ledger: DefaultDict[str, list[tuple[float, str, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Pending hindsight credit to add to next transition (BEFORE normalization)
    pending_hindsight_credit: float = 0.0
    # Pre-computed autocast decision for hot path performance
    # Avoids repeated device type checks and amp flag evaluation per batch
    autocast_enabled: bool = False

    # === Obs V3 Action Feedback (Phase 2a½) ===
    # last_action_success: True = no prior action to fail (first step has none)
    # last_action_op: LifecycleOp.WAIT.value (0) = neutral "no action yet"
    last_action_success: bool = True
    last_action_op: int = 0  # LifecycleOp.WAIT.value, avoid lambda for slots=True

    # === Obs V3 Gradient Health Tracking (Phase 2a½) ===
    # Maps slot_id -> previous epoch's gradient_health value
    # Initial value for new slots: 1.0 (assume healthy)
    gradient_health_prev: dict[str, float] = field(default_factory=dict)

    # === Obs V3 Counterfactual Freshness Tracking (Phase 2a½) ===
    # Maps slot_id -> epochs since last counterfactual measurement
    # Initial value for new slots: 0 (fresh when slot germinates)
    epochs_since_counterfactual: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Initialize counters with LifecycleOp names (WAIT, GERMINATE, SET_ALPHA_TARGET, PRUNE, FOSSILIZE, ADVANCE)
        # since factored actions use op.name for counting, not flat action enum names
        if not self.action_counts:
            base_counts = {op.name: 0 for op in LifecycleOp}
            self.action_counts = base_counts.copy()
            self.successful_action_counts = base_counts.copy()

    def init_accumulators(self, slots: list[str]) -> None:
        """Initialize pre-allocated accumulators on the environment's device.

        Args:
            slots: List of enabled slot IDs for per-slot counterfactual tracking
        """
        self.train_loss_accum = torch.zeros(1, device=self.env_device)
        self.train_correct_accum = torch.zeros(1, device=self.env_device)
        self.val_loss_accum = torch.zeros(1, device=self.env_device)
        self.val_correct_accum = torch.zeros(1, device=self.env_device)
        # Per-slot counterfactual accumulators for multi-slot reward attribution
        self.cf_correct_accums: dict[str, torch.Tensor] = {
            slot_id: torch.zeros(1, device=self.env_device) for slot_id in slots
        }
        self.cf_totals: dict[str, int] = {slot_id: 0 for slot_id in slots}
        # "All disabled" accumulator for true pair synergy measurement
        self.cf_all_disabled_accum = torch.zeros(1, device=self.env_device)
        self.cf_all_disabled_total = 0
        # Pair accumulators for 3-4 seeds (all C(n,2) pairs)
        n = len(slots)
        self.cf_pair_accums = {}
        self.cf_pair_totals = {}
        if 3 <= n <= 4:
            for i in range(n):
                for j in range(i + 1, n):
                    self.cf_pair_accums[(i, j)] = torch.zeros(1, device=self.env_device)
                    self.cf_pair_totals[(i, j)] = 0

    def zero_accumulators(self) -> None:
        """Zero accumulators at the start of each epoch (faster than reallocating).

        Note: Assumes init_accumulators() was called. Guards added for mypy.
        """
        if self.train_loss_accum is not None:
            self.train_loss_accum.zero_()
        if self.train_correct_accum is not None:
            self.train_correct_accum.zero_()
        if self.val_loss_accum is not None:
            self.val_loss_accum.zero_()
        if self.val_correct_accum is not None:
            self.val_correct_accum.zero_()
        # Zero per-slot counterfactual accumulators
        for slot_id in self.cf_correct_accums:
            self.cf_correct_accums[slot_id].zero_()
        for slot_id in self.cf_totals:
            self.cf_totals[slot_id] = 0
        # Zero "all disabled" accumulator
        if self.cf_all_disabled_accum is not None:
            self.cf_all_disabled_accum.zero_()
        self.cf_all_disabled_total = 0
        # Zero pair accumulators
        for pair_key in self.cf_pair_accums:
            self.cf_pair_accums[pair_key].zero_()
        for pair_key in self.cf_pair_totals:
            self.cf_pair_totals[pair_key] = 0

    def reset_episode_state(self, slots: list[str]) -> None:
        """Reset per-episode state when reusing env instances."""
        self.seeds_created = 0
        self.seeds_fossilized = 0
        self.contributing_fossilized = 0
        # Reset action counters for episode diagnostics (TELE-610)
        self.germinate_count = 0
        self.prune_count = 0
        self.fossilize_count = 0
        self.episode_rewards.clear()

        base_counts = {op.name: 0 for op in LifecycleOp}
        self.action_counts = base_counts.copy()
        self.successful_action_counts = base_counts.copy()

        self.seed_optimizers.clear()
        self.acc_at_germination.clear()
        self.committed_val_acc = 0.0
        self.committed_acc_history.clear()
        self.escrow_credit = {slot_id: 0.0 for slot_id in slots}
        self.host_max_acc = 0.0
        self.pending_auto_prune_penalty = 0.0
        self.prev_slot_alphas = {slot_id: 0.0 for slot_id in slots}
        self.prev_slot_params = {slot_id: 0 for slot_id in slots}
        self.gradient_ratio_ema = {slot_id: 0.0 for slot_id in slots}
        self.scaffold_boost_ledger.clear()
        self.pending_hindsight_credit = 0.0

        # Reset Obs V3 action feedback (Phase 2a½)
        self.last_action_success = True  # No prior action to fail
        self.last_action_op = LifecycleOp.WAIT.value  # Neutral op

        # Clear Obs V3 per-slot tracking (Phase 2a½)
        # These are populated per-slot when seeds germinate:
        # - gradient_health_prev[slot_id] = 1.0 (assume healthy)
        # - epochs_since_counterfactual[slot_id] = 0 (fresh)
        self.gradient_health_prev.clear()
        self.epochs_since_counterfactual.clear()

        self.signal_tracker.reset()
        self.governor.reset()
        self.needs_governor_snapshot = False
        if self.health_monitor is not None:
            self.health_monitor.reset()
        if self.counterfactual_helper is not None:
            self.counterfactual_helper.reset()

        if self.train_loss_accum is None:
            self.init_accumulators(slots)
        else:
            self.zero_accumulators()

    def init_obs_v3_slot_tracking(self, slot_id: str) -> None:
        """Initialize Obs V3 per-slot tracking for a newly germinated seed."""
        self.gradient_health_prev[slot_id] = 1.0
        self.epochs_since_counterfactual[slot_id] = 0

    def clear_obs_v3_slot_tracking(self, slot_id: str) -> None:
        """Clear Obs V3 per-slot tracking when a slot becomes empty."""
        if slot_id in self.gradient_health_prev:
            del self.gradient_health_prev[slot_id]
        if slot_id in self.epochs_since_counterfactual:
            del self.epochs_since_counterfactual[slot_id]


__all__ = ["ParallelEnvState"]
