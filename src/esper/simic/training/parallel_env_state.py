"""Parallel environment state for vectorized PPO training.

This module defines the ParallelEnvState dataclass which holds all state
for a single parallel training environment, including:
- Model and optimizer references
- Episode tracking (rewards, action counts)
- Pre-allocated GPU accumulators (to avoid per-epoch allocation churn)
- LSTM hidden state for recurrent policies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from esper.leyline.factored_actions import LifecycleOp

if TYPE_CHECKING:
    from esper.tolaria import TolariaGovernor


@dataclass
class ParallelEnvState:
    """State for a single parallel environment with CUDA stream for async execution.

    DataLoaders are now SHARED via SharedBatchIterator - batches are pre-split
    and data is pre-moved to each env's device with non_blocking=True.
    """
    model: nn.Module
    host_optimizer: torch.optim.Optimizer
    signal_tracker: any  # SignalTracker from tamiyo
    governor: "TolariaGovernor"  # Fail-safe watchdog for catastrophic failure detection
    seed_optimizers: dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    env_device: str = "cuda:0"  # Device this env runs on
    stream: torch.cuda.Stream | None = None  # CUDA stream for async execution
    seeds_created: int = 0
    seeds_fossilized: int = 0  # Total seeds fossilized this episode
    contributing_fossilized: int = 0  # Seeds with total_improvement >= DEFAULT_MIN_FOSSILIZE_CONTRIBUTION
    episode_rewards: list = field(default_factory=list)
    action_counts: dict = field(default_factory=dict)
    successful_action_counts: dict = field(default_factory=dict)
    action_enum: type | None = None
    # Metrics for current batch step
    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    # Ransomware-resistant reward: track accuracy at germination for progress calculation
    acc_at_germination: dict[str, float] = field(default_factory=dict)
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
    # LSTM hidden state for recurrent policy
    # Shape: (h, c) where each is [num_layers, 1, hidden_dim] for this single env
    # (Batched to [num_layers, num_envs, hidden_dim] during forward pass)
    # None = fresh episode (initialized on first action selection)
    lstm_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    telemetry_cb: any = None  # Callback wired when telemetry is enabled
    # Per-slot EMA tracking for seed gradient ratio (for G2 gate)
    # Smooths per-step ratio noise with momentum=0.9
    gradient_ratio_ema: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Initialize counters with LifecycleOp names (WAIT, GERMINATE, FOSSILIZE, CULL)
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

    def zero_accumulators(self) -> None:
        """Zero accumulators at the start of each epoch (faster than reallocating)."""
        self.train_loss_accum.zero_()
        self.train_correct_accum.zero_()
        self.val_loss_accum.zero_()
        self.val_correct_accum.zero_()
        # Zero per-slot counterfactual accumulators
        for slot_id in self.cf_correct_accums:
            self.cf_correct_accums[slot_id].zero_()
        for slot_id in self.cf_totals:
            self.cf_totals[slot_id] = 0


__all__ = ["ParallelEnvState"]
