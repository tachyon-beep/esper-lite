"""Sanctum Schema - Complete state objects matching existing Rich TUI.

These dataclasses mirror ALL state tracked by karn/tui.py for 1:1 port.
Reference: src/esper/karn/tui.py (EnvState, SeedState, TUIState, GPUStats)

CRITICAL FIXES APPLIED:
1. Added total_actions to TamiyoState (for action % calculation)
2. Added fossilized_params to EnvState (for scoreboard param counts)
3. Documented ALL reward_components keys explicitly
4. Documented action normalization (GERMINATE_* → GERMINATE)
5. Documented EnvState update methods (add_reward, add_accuracy, add_action, _update_status)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CounterfactualConfig:
    """Single configuration result from factorial evaluation.

    Represents one row in the counterfactual matrix:
    e.g., seed_mask=(True, False, True) means slots 0 and 2 enabled.
    """
    seed_mask: tuple[bool, ...]  # Which seeds are enabled
    accuracy: float = 0.0  # Validation accuracy for this config


@dataclass
class CounterfactualSnapshot:
    """Full factorial counterfactual matrix for an environment.

    Contains all 2^n configurations for n active seeds.
    Used to compute marginal contributions and interaction terms.
    """
    slot_ids: tuple[str, ...] = ()  # ("r0c0", "r0c1", "r0c2")
    configs: list[CounterfactualConfig] = field(default_factory=list)
    strategy: str = "unavailable"  # "full_factorial" or "unavailable"
    compute_time_ms: float = 0.0

    @property
    def baseline_accuracy(self) -> float:
        """Accuracy with all seeds disabled."""
        for cfg in self.configs:
            if not any(cfg.seed_mask):
                return cfg.accuracy
        return 0.0

    @property
    def combined_accuracy(self) -> float:
        """Accuracy with all seeds enabled."""
        for cfg in self.configs:
            if all(cfg.seed_mask):
                return cfg.accuracy
        return 0.0

    def get_accuracy(self, mask: tuple[bool, ...]) -> float | None:
        """Get accuracy for a specific seed configuration."""
        for cfg in self.configs:
            if cfg.seed_mask == mask:
                return cfg.accuracy
        return None

    def individual_contributions(self) -> dict[str, float]:
        """Compute each seed's solo contribution over baseline."""
        baseline = self.baseline_accuracy
        result = {}
        n = len(self.slot_ids)
        for i, slot_id in enumerate(self.slot_ids):
            mask = tuple(j == i for j in range(n))
            acc = self.get_accuracy(mask)
            if acc is not None:
                result[slot_id] = acc - baseline
        return result

    def pair_contributions(self) -> dict[tuple[str, str], float]:
        """Compute each pair's contribution over baseline."""
        baseline = self.baseline_accuracy
        result = {}
        n = len(self.slot_ids)
        for i in range(n):
            for j in range(i + 1, n):
                mask = tuple(k == i or k == j for k in range(n))
                acc = self.get_accuracy(mask)
                if acc is not None:
                    pair = (self.slot_ids[i], self.slot_ids[j])
                    result[pair] = acc - baseline
        return result

    def total_synergy(self) -> float:
        """Compute total synergy: combined - baseline - sum(individual contributions)."""
        baseline = self.baseline_accuracy
        combined = self.combined_accuracy
        individuals = self.individual_contributions()
        expected = baseline + sum(individuals.values())
        return combined - expected


@dataclass
class SeedState:
    """State of a single seed slot.

    Reference: tui.py lines 87-100 (SeedState dataclass)
    """
    slot_id: str
    stage: str = "DORMANT"
    blueprint_id: str | None = None
    alpha: float = 0.0
    accuracy_delta: float = 0.0
    seed_params: int = 0
    grad_ratio: float = 0.0
    # Gradient health flags - shown as ▼ (vanishing) and ▲ (exploding)
    has_vanishing: bool = False
    has_exploding: bool = False
    # Stage progress - shown as "e5" in slot cell
    epochs_in_stage: int = 0
    # Fossilization/cull context (P1/P2 telemetry gap fix)
    improvement: float = 0.0  # Accuracy improvement when fossilized
    cull_reason: str = ""  # Why seed was culled (e.g., "gradient_explosion", "stagnation")
    auto_culled: bool = False  # True if system auto-culled vs policy decision
    epochs_total: int = 0  # Total epochs seed was alive
    counterfactual: float = 0.0  # Causal attribution score
    # Blend tempo - Tamiyo's chosen integration speed (3=FAST, 5=STANDARD, 8=SLOW)
    blend_tempo_epochs: int = 5


@dataclass
class GPUStats:
    """Per-GPU statistics for multi-GPU support.

    Reference: tui.py GPUStats dataclass
    """
    device_id: int = 0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    utilization: float = 0.0
    temperature: float = 0.0


@dataclass
class EnvState:
    """Per-environment state for multi-env tracking.

    Reference: tui.py lines 124-267 (EnvState dataclass)

    KEY METHODS:
    - add_reward(reward, epoch): Add reward and update best tracking
    - add_accuracy(accuracy, epoch, episode): Add accuracy, update best/status
    - add_action(action_name): Track action (normalizes GERMINATE_* → GERMINATE)
    - _update_status(prev_acc, curr_acc): Update env status based on metrics
    """
    env_id: int
    current_epoch: int = 0
    host_accuracy: float = 0.0
    host_loss: float = 0.0
    host_params: int = 0

    # Seed slots
    seeds: dict[str, SeedState] = field(default_factory=dict)
    active_seed_count: int = 0
    fossilized_count: int = 0
    culled_count: int = 0

    # FIX: Added fossilized_params for scoreboard display (total params in FOSSILIZED seeds)
    fossilized_params: int = 0

    # Seed graveyard: per-blueprint lifecycle tracking
    blueprint_spawns: dict[str, int] = field(default_factory=dict)  # blueprint -> spawn count
    blueprint_culls: dict[str, int] = field(default_factory=dict)   # blueprint -> cull count
    blueprint_fossilized: dict[str, int] = field(default_factory=dict)  # blueprint -> fossilized count

    # Reward component breakdown (from REWARD_COMPUTED telemetry)
    # Uses RewardComponents dataclass for type safety. Populated by aggregator.
    reward_components: "RewardComponents" = field(default_factory=lambda: RewardComponents())

    # Counterfactual matrix (from COUNTERFACTUAL_MATRIX_COMPUTED telemetry)
    counterfactual_matrix: CounterfactualSnapshot = field(
        default_factory=CounterfactualSnapshot
    )

    # History for sparklines (maxlen=50)
    reward_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Best tracking
    best_reward: float = float('-inf')
    best_reward_epoch: int = 0
    best_accuracy: float = 0.0
    best_accuracy_epoch: int = 0
    best_accuracy_episode: int = 0
    best_seeds: dict[str, SeedState] = field(default_factory=dict)

    # Per-env action tracking
    # ACTION NORMALIZATION: add_action() normalizes factored actions:
    #   GERMINATE_CONV_LIGHT → GERMINATE
    #   GERMINATE_DENSE_HEAVY → GERMINATE
    #   FOSSILIZE_R0C0 → FOSSILIZE
    #   CULL_R1C1 → CULL
    action_history: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    action_counts: dict[str, int] = field(default_factory=lambda: {
        "WAIT": 0, "GERMINATE": 0, "CULL": 0, "FOSSILIZE": 0
    })
    total_actions: int = 0

    # Status tracking
    status: str = "initializing"
    last_update: datetime | None = None
    epochs_since_improvement: int = 0

    # A/B test cohort (for color coding)
    # Captured from REWARD_COMPUTED event's ab_group field
    # Values: "shaped", "simplified", "sparse", or None if not A/B testing
    reward_mode: str | None = None

    @property
    def current_reward(self) -> float:
        """Get most recent reward."""
        return self.reward_history[-1] if self.reward_history else 0.0

    @property
    def mean_reward(self) -> float:
        """Mean reward over history."""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    @property
    def growth_ratio(self) -> float:
        """Ratio of mutated model size to original host size.

        Shows mutation overhead: (host_params + fossilized_params) / host_params
        A growth_ratio of 1.2 means the model is 20% larger due to fossilized seeds.
        """
        if self.host_params == 0:
            return 1.0
        return (self.host_params + self.fossilized_params) / self.host_params

    def add_reward(self, reward: float, epoch: int) -> None:
        """Add reward and update best tracking."""
        self.reward_history.append(reward)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_reward_epoch = epoch

    def add_accuracy(self, accuracy: float, epoch: int, episode: int = 0) -> None:
        """Add accuracy and update best/status tracking."""
        prev_acc = self.accuracy_history[-1] if self.accuracy_history else 0.0
        self.accuracy_history.append(accuracy)
        self.host_accuracy = accuracy

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_accuracy_epoch = epoch
            self.best_accuracy_episode = episode
            self.epochs_since_improvement = 0
            # Snapshot contributing seeds when new best is achieved
            # Include permanent (FOSSILIZED) and provisional (PROBATIONARY, BLENDING)
            _contributing_stages = {"FOSSILIZED", "PROBATIONARY", "BLENDING"}
            self.best_seeds = {
                slot_id: SeedState(
                    slot_id=seed.slot_id,
                    stage=seed.stage,
                    blueprint_id=seed.blueprint_id,
                    alpha=seed.alpha,
                    accuracy_delta=seed.accuracy_delta,
                    seed_params=seed.seed_params,
                    grad_ratio=seed.grad_ratio,
                    has_vanishing=seed.has_vanishing,
                    has_exploding=seed.has_exploding,
                    epochs_in_stage=seed.epochs_in_stage,
                )
                for slot_id, seed in self.seeds.items()
                if seed.stage in _contributing_stages
            }
        else:
            self.epochs_since_improvement += 1

        self._update_status(prev_acc, accuracy)

    def add_action(self, action_name: str) -> None:
        """Track action taken.

        ACTION NORMALIZATION: Normalizes factored germination actions to base types:
        - GERMINATE_CONV_LIGHT → GERMINATE
        - GERMINATE_DENSE_HEAVY → GERMINATE
        - FOSSILIZE_R0C0 → FOSSILIZE
        - CULL_R1C1 → CULL
        - WAIT → WAIT (unchanged)
        """
        self.action_history.append(action_name)

        # Normalize factored actions
        normalized = action_name
        if action_name.startswith("GERMINATE"):
            normalized = "GERMINATE"
        elif action_name.startswith("FOSSILIZE"):
            normalized = "FOSSILIZE"
        elif action_name.startswith("CULL"):
            normalized = "CULL"
        elif action_name.startswith("WAIT"):
            normalized = "WAIT"

        if normalized in self.action_counts:
            self.action_counts[normalized] += 1
            self.total_actions += 1

    def _update_status(self, prev_acc: float, curr_acc: float) -> None:
        """Update env status based on metrics.

        Status values: initializing, healthy, excellent, stalled, degraded
        """
        if self.epochs_since_improvement > 10:
            self.status = "stalled"
        elif curr_acc < prev_acc - 1.0:
            self.status = "degraded"
        elif curr_acc > 80.0:
            self.status = "excellent"
        elif self.current_epoch > 0:
            self.status = "healthy"


@dataclass
class TamiyoState:
    """Tamiyo policy agent state - ALL metrics from existing TUI.

    Reference: tui.py TUIState policy metrics + _render_tamiyo_brain()

    FIX: Added total_actions field (required for action percentage calculation)
    FIX: Added advantage stats, entropy_coef, gradient health fields from aggregator
    """
    # Policy health (Health panel)
    entropy: float = 0.0
    clip_fraction: float = 0.0
    kl_divergence: float = 0.0
    explained_variance: float = 0.0

    # Losses (Losses panel)
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    grad_norm: float = 0.0

    # Vitals (Vitals panel)
    learning_rate: float | None = None
    entropy_coef: float = 0.0  # Entropy coefficient (adaptive)
    ratio_mean: float = 1.0
    ratio_min: float = 1.0
    ratio_max: float = 1.0
    ratio_std: float = 0.0  # Standard deviation of ratio

    # Advantage statistics (from PPO update)
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0

    # Gradient health (shown in Vitals)
    dead_layers: int = 0
    exploding_layers: int = 0
    nan_grad_count: int = 0  # NaN gradient count
    layer_gradient_health: float = 1.0  # GradHP percentage (0-1)
    entropy_collapsed: bool = False  # Entropy collapse detected

    # Performance timing (for throughput monitoring)
    update_time_ms: float = 0.0  # PPO update duration in milliseconds
    early_stop_epoch: int | None = None  # KL early stopping triggered at this epoch

    # Per-head entropy and gradient norms (for multi-head policy)
    head_slot_entropy: float = 0.0  # Entropy for slot action head
    head_slot_grad_norm: float = 0.0  # Gradient norm for slot head
    head_blueprint_entropy: float = 0.0  # Entropy for blueprint action head
    head_blueprint_grad_norm: float = 0.0  # Gradient norm for blueprint head

    # PPO inner loop context
    inner_epoch: int = 0  # Current inner optimization epoch
    ppo_batch: int = 0  # Current batch within PPO update (use ppo_batch to avoid conflict with any existing 'batch')

    # Action distribution (Actions panel)
    action_counts: dict[str, int] = field(default_factory=dict)
    # FIX: Added total_actions for percentage calculation in TamiyoBrain
    total_actions: int = 0

    # PPO data received flag
    ppo_data_received: bool = False

    # Last decision snapshot (captured from REWARD_COMPUTED events)
    # Deprecated: Use recent_decisions instead
    last_decision: "DecisionSnapshot | None" = None

    # Recent decisions list (up to 3, each visible for at least 10 seconds)
    recent_decisions: list["DecisionSnapshot"] = field(default_factory=list)


@dataclass
class SystemVitals:
    """System resource metrics - ALL from existing TUI.

    Reference: tui.py _render_esper_status() and _update_system_stats()
    """
    # Multi-GPU support
    gpu_stats: dict[int, GPUStats] = field(default_factory=dict)

    # Convenience fields for single-GPU access (populated from gpu_stats[0])
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0

    # FIX: CPU was collected but never displayed in old TUI
    cpu_percent: float = 0.0

    # RAM
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0

    # Throughput
    epochs_per_second: float = 0.0
    batches_per_hour: float = 0.0
    steps_per_second: float = 0.0

    # Host network
    host_params: int = 0

    @property
    def has_memory_alarm(self) -> bool:
        """Check if any device exceeds 90% memory usage."""
        # Check RAM
        if self.ram_total_gb > 0 and (self.ram_used_gb / self.ram_total_gb) > 0.90:
            return True
        # Check GPUs
        for stats in self.gpu_stats.values():
            if stats.memory_total_gb > 0:
                usage = stats.memory_used_gb / stats.memory_total_gb
                if usage > 0.90:
                    return True
        # Fallback single GPU
        if self.gpu_memory_total_gb > 0:
            usage = self.gpu_memory_used_gb / self.gpu_memory_total_gb
            if usage > 0.90:
                return True
        return False

    @property
    def memory_alarm_devices(self) -> list[str]:
        """Get list of devices exceeding 90% memory usage."""
        devices = []
        # Check RAM
        if self.ram_total_gb > 0 and (self.ram_used_gb / self.ram_total_gb) > 0.90:
            devices.append("RAM")
        # Check multi-GPU stats
        for device, stats in self.gpu_stats.items():
            if stats.memory_total_gb > 0:
                usage = stats.memory_used_gb / stats.memory_total_gb
                if usage > 0.90:
                    devices.append(f"cuda:{device}")
        # Fallback to single GPU if no gpu_stats but fallback fields are populated
        if not self.gpu_stats and self.gpu_memory_total_gb > 0:
            usage = self.gpu_memory_used_gb / self.gpu_memory_total_gb
            if usage > 0.90:
                devices.append("cuda:0")
        return devices


@dataclass
class RewardComponents:
    """Esper-specific reward signal breakdown.

    Reference: tui.py _render_reward_components() lines 1513-1586

    ALL KEYS DOCUMENTED (Esper-specific components):
    - base_acc_delta: Legacy shaped signal based on accuracy improvement
    - bounded_attribution: Contribution-primary attribution signal (replaces seed_contribution)
    - seed_contribution: Seed contribution percentage (older format, may coexist)
    - compute_rent: Cost of active seeds (always negative)
    - ratio_penalty: Penalty for extreme policy ratios (negative if triggered)
    - stage_bonus: Bonus for reaching advanced lifecycle stages (BLENDING+)
    - fossilize_terminal_bonus: Large terminal bonus for successful fossilization
    - blending_warning: Warning signal during blending phase (negative)
    - probation_warning: Warning signal during probationary period
    - val_acc: Validation accuracy context (not a reward component, metadata)
    """
    # Total reward
    total: float = 0.0

    # Base delta (legacy shaped signal)
    base_acc_delta: float = 0.0

    # Attribution (contribution-primary)
    bounded_attribution: float = 0.0
    seed_contribution: float = 0.0

    # Costs
    compute_rent: float = 0.0
    ratio_penalty: float = 0.0

    # Bonuses
    stage_bonus: float = 0.0
    fossilize_terminal_bonus: float = 0.0

    # Warnings
    blending_warning: float = 0.0
    probation_warning: float = 0.0

    # Context
    env_id: int = 0
    val_acc: float = 0.0
    last_action: str = ""


@dataclass
class DecisionSnapshot:
    """Snapshot of a single Tamiyo decision for display.

    Captures what Tamiyo saw, what she chose, and the outcome.
    Used for the "Last Decision" section of TamiyoBrain.

    Stable carousel behavior:
    - Each decision stays visible for at least 30 seconds
    - Only the oldest unpinned decision can be replaced
    - Pinned decisions never get replaced
    """
    timestamp: datetime
    slot_states: dict[str, str]  # slot_id -> "Training 12%" or "Empty"
    host_accuracy: float
    chosen_action: str  # "GERMINATE", "WAIT", "CULL", "FOSSILIZE"
    chosen_slot: str | None  # Target slot for action (None for WAIT)
    confidence: float  # Action probability (0-1)
    expected_value: float  # Value estimate before action
    actual_reward: float | None  # Actual reward received (None if pending)
    alternatives: list[tuple[str, float]]  # [(action_name, probability), ...]
    # Unique ID for click-to-pin targeting
    decision_id: str = ""
    # Pinned decisions never get replaced
    pinned: bool = False


@dataclass
class RunConfig:
    """Training run configuration captured at TRAINING_STARTED.

    Stores hyperparameters and config for display in run header.
    """
    seed: int | None = None  # Random seed for reproducibility
    n_episodes: int = 0  # Total episodes to train
    lr: float = 0.0  # Initial learning rate
    clip_ratio: float = 0.2  # PPO clip ratio
    entropy_coef: float = 0.01  # Initial entropy coefficient
    param_budget: int = 0  # Seed parameter budget
    resume_path: str = ""  # Checkpoint resume path (empty if fresh run)
    entropy_anneal: dict = field(default_factory=dict)  # Entropy schedule config


@dataclass
class BestRunRecord:
    """Historical record of a best run for the leaderboard.

    Captured at batch end when an env achieves a new personal best.
    Shows both the peak accuracy and where the run ended up.

    Reference: tui.py lines 103-114 (BestRunRecord dataclass)
    """
    env_id: int
    episode: int  # Batch/episode number (0-indexed)
    peak_accuracy: float  # Best accuracy achieved during this run
    final_accuracy: float  # Accuracy at the end of the batch
    epoch: int = 0  # Epoch within episode when best was achieved
    seeds: dict[str, SeedState] = field(default_factory=dict)  # Seeds at peak
    growth_ratio: float = 1.0  # Model size ratio: (host + fossilized) / host


@dataclass
class EventLogEntry:
    """Single event log entry for display in Event Log panel.

    Reference: Used by aggregator to build event_log list
    """
    timestamp: str  # Formatted as HH:MM:SS
    event_type: str  # REWARD_COMPUTED, SEED_GERMINATED, etc.
    env_id: int | None  # None for global events (PPO, BATCH)
    message: str  # Formatted message for display
    episode: int = 0  # Episode number for grouping
    relative_time: str = ""  # "(2s ago)" relative time string


@dataclass
class SanctumSnapshot:
    """Complete snapshot of Sanctum state for rendering.

    This is the single source of truth passed to all widgets.
    """
    # Per-env state
    envs: dict[int, EnvState] = field(default_factory=dict)

    # Policy agent state
    tamiyo: TamiyoState = field(default_factory=TamiyoState)

    # System metrics
    vitals: SystemVitals = field(default_factory=SystemVitals)

    # Focused env reward breakdown
    rewards: RewardComponents = field(default_factory=RewardComponents)

    # Slot configuration (dynamic based on config)
    slot_ids: list[str] = field(default_factory=list)

    # Run metadata
    current_episode: int = 0
    current_batch: int = 0
    current_epoch: int = 0
    max_epochs: int = 0
    run_id: str = ""
    task_name: str = ""
    run_config: RunConfig = field(default_factory=RunConfig)
    start_time: datetime | None = None

    # Connection and timing (used by aggregator)
    connected: bool = False
    runtime_seconds: float = 0.0
    staleness_seconds: float = float('inf')
    captured_at: str = ""  # ISO timestamp
    total_events_received: int = 0  # Debug: total events received by backend
    poll_count: int = 0  # Debug: number of UI poll cycles
    training_thread_alive: bool | None = None  # Debug: is training thread running?

    # Aggregates (computed from envs)
    aggregate_mean_accuracy: float = 0.0
    aggregate_mean_reward: float = 0.0

    # Batch-level aggregates (from BATCH_COMPLETED)
    batch_avg_reward: float = 0.0  # Average reward for last batch
    batch_total_episodes: int = 0  # Total episodes in training run

    # Rolling average history (mean accuracy across all envs over time)
    mean_accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Event log (most recent last)
    event_log: list[EventLogEntry] = field(default_factory=list)

    # Best runs leaderboard (top 10 by peak accuracy)
    best_runs: list[BestRunRecord] = field(default_factory=list)

    # Timestamps for staleness detection
    last_ppo_update: datetime | None = None
    last_reward_update: datetime | None = None

    # Focused env for detail panel
    focused_env_id: int = 0

    @property
    def is_stale(self) -> bool:
        """Check if data is stale (>5s since last update).

        STALENESS THRESHOLD: 5 seconds (matches Overwatch behavior)
        """
        return self.staleness_seconds > 5.0


def make_sparkline(values: list[float] | deque[float], width: int = 8) -> str:
    """Create a sparkline from values.

    Reference: tui.py _make_sparkline_static() lines 103-121
    """
    if not values:
        return "─" * width

    values_list = list(values)[-width:]
    min_val = min(values_list)
    max_val = max(values_list)
    range_val = max_val - min_val if max_val != min_val else 1.0

    blocks = "▁▂▃▄▅▆▇█"
    result = ""
    for v in values_list:
        normalized = (v - min_val) / range_val
        idx = min(int(normalized * (len(blocks) - 1)), len(blocks) - 1)
        result += blocks[idx]

    return result.ljust(width, "─")
