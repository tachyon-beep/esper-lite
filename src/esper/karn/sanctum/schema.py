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


@dataclass(slots=True)
class SeedState:
    """State of a single seed slot.

    Uses slots=True for memory efficiency (saves ~100 bytes per instance).
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
    # Fossilization/prune context (P1/P2 telemetry gap fix)
    improvement: float = 0.0  # Accuracy improvement when fossilized
    prune_reason: str = ""  # Why seed was pruned (e.g., "gradient_explosion", "stagnation")
    auto_pruned: bool = False  # True if system auto-pruned vs policy decision
    epochs_total: int = 0  # Total epochs seed was alive
    counterfactual: float = 0.0  # Causal attribution score
    # Blend tempo - Tamiyo's chosen integration speed (3=FAST, 5=STANDARD, 8=SLOW)
    blend_tempo_epochs: int = 5
    # Alpha curve shape - always present, but only displayed during BLENDING
    # (when the curve is causally active). See design rationale in plan.
    alpha_curve: str = "LINEAR"

    # Inter-slot interaction metrics (from counterfactual engine)
    # These show how this seed synergizes with others in the ensemble
    contribution_velocity: float = 0.0  # EMA of contribution changes (trend direction)
    interaction_sum: float = 0.0  # Σ I_ij for all j ≠ i (total synergy from interactions)
    boost_received: float = 0.0  # max(I_ij) for j ≠ i (strongest interaction partner)
    upstream_alpha_sum: float = 0.0  # Σ alpha_j for slots j < i (position-aware blending)
    downstream_alpha_sum: float = 0.0  # Σ alpha_j for slots j > i (position-aware blending)


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
    pruned_count: int = 0

    # FIX: Added fossilized_params for scoreboard display (total params in FOSSILIZED seeds)
    fossilized_params: int = 0

    # Seed graveyard: per-blueprint lifecycle tracking
    blueprint_spawns: dict[str, int] = field(default_factory=dict)  # blueprint -> spawn count
    blueprint_prunes: dict[str, int] = field(default_factory=dict)   # blueprint -> prune count
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
    #   PRUNE_R1C1 → PRUNE
    #   ADVANCE_R1C1 → ADVANCE
    action_history: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    action_counts: dict[str, int] = field(default_factory=lambda: {
        "WAIT": 0,
        "GERMINATE": 0,
        "SET_ALPHA_TARGET": 0,
        "PRUNE": 0,
        "FOSSILIZE": 0,
        "ADVANCE": 0,
    })
    total_actions: int = 0

    # Gaming rate tracking (for per-env reward health)
    # Resets each episode to show recent behavior
    gaming_trigger_count: int = 0   # Steps where ratio_penalty or alpha_shock fired
    total_reward_steps: int = 0     # Total steps with reward computed

    # Status tracking
    status: str = "initializing"
    last_update: datetime | None = None
    epochs_since_improvement: int = 0

    # Hysteresis counter for status changes (prevents flicker)
    # Status only changes after 3 consecutive epochs meeting condition
    stall_counter: int = 0
    degraded_counter: int = 0

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
    def gaming_rate(self) -> float:
        """Fraction of steps with anti-gaming penalties (ratio_penalty or alpha_shock)."""
        if self.total_reward_steps == 0:
            return 0.0
        return self.gaming_trigger_count / self.total_reward_steps

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
        self.current_epoch = epoch

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_accuracy_epoch = epoch
            self.best_accuracy_episode = episode
            self.epochs_since_improvement = 0
            # Snapshot contributing seeds when new best is achieved
            # Include permanent (FOSSILIZED) and provisional (HOLDING, BLENDING)
            _contributing_stages = {"FOSSILIZED", "HOLDING", "BLENDING"}
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
        - SET_ALPHA_TARGET_R0C0 → SET_ALPHA_TARGET
        - FOSSILIZE_R0C0 → FOSSILIZE
        - PRUNE_R1C1 → PRUNE
        - ADVANCE_R1C1 → ADVANCE
        - WAIT → WAIT (unchanged)
        """
        self.action_history.append(action_name)

        # Normalize factored actions
        normalized = action_name
        if action_name.startswith("GERMINATE"):
            normalized = "GERMINATE"
        elif action_name.startswith("SET_ALPHA_TARGET"):
            normalized = "SET_ALPHA_TARGET"
        elif action_name.startswith("FOSSILIZE"):
            normalized = "FOSSILIZE"
        elif action_name.startswith("PRUNE"):
            normalized = "PRUNE"
        elif action_name.startswith("ADVANCE"):
            normalized = "ADVANCE"
        elif action_name.startswith("WAIT"):
            normalized = "WAIT"

        if normalized in self.action_counts:
            self.action_counts[normalized] += 1
            self.total_actions += 1

    def _update_status(self, prev_acc: float, curr_acc: float) -> None:
        """Update env status based on metrics with hysteresis.

        Status values: initializing, healthy, excellent, stalled, degraded

        Hysteresis: Status only changes after 3 consecutive epochs meeting
        the condition. This prevents flicker from transient spikes.
        """
        HYSTERESIS_THRESHOLD = 3

        # Check stall condition (>10 epochs without improvement)
        if self.epochs_since_improvement > 10:
            self.stall_counter += 1
            if self.stall_counter >= HYSTERESIS_THRESHOLD:
                self.status = "stalled"
        else:
            self.stall_counter = 0  # Reset on improvement

        # Check degraded condition (accuracy dropped >1%)
        if curr_acc < prev_acc - 1.0:
            self.degraded_counter += 1
            if self.degraded_counter >= HYSTERESIS_THRESHOLD:
                self.status = "degraded"
        elif curr_acc > prev_acc:  # Only reset on IMPROVEMENT
            self.degraded_counter = 0
        # Note: if stable (curr_acc == prev_acc), counter persists

        # Positive status updates (no hysteresis needed - immediate feedback is good)
        if self.epochs_since_improvement == 0:  # Just improved
            if curr_acc > 80.0:
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
    # Post-normalization stats (should be ~0 mean, ~1 std if normalization working)
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    # Pre-normalization stats (raw learning signal magnitude)
    advantage_raw_mean: float = 0.0
    advantage_raw_std: float = 0.0

    # Gradient health (shown in Vitals)
    dead_layers: int = 0
    exploding_layers: int = 0
    nan_grad_count: int = 0  # NaN gradient count
    layer_gradient_health: dict[str, float] | None = None  # Per-layer gradient health metrics
    entropy_collapsed: bool = False  # Entropy collapse detected

    # Performance timing (for throughput monitoring)
    update_time_ms: float = 0.0  # PPO update duration in milliseconds
    early_stop_epoch: int | None = None  # KL early stopping triggered at this epoch

    # Per-head entropy (for multi-head policy collapse detection)
    head_slot_entropy: float = 0.0
    head_blueprint_entropy: float = 0.0
    head_style_entropy: float = 0.0
    head_tempo_entropy: float = 0.0
    head_alpha_target_entropy: float = 0.0
    head_alpha_speed_entropy: float = 0.0
    head_alpha_curve_entropy: float = 0.0
    head_op_entropy: float = 0.0

    # Per-head gradient norms (for multi-head gradient health)
    head_slot_grad_norm: float = 0.0
    head_blueprint_grad_norm: float = 0.0
    head_style_grad_norm: float = 0.0
    head_tempo_grad_norm: float = 0.0
    head_alpha_target_grad_norm: float = 0.0
    head_alpha_speed_grad_norm: float = 0.0
    head_alpha_curve_grad_norm: float = 0.0
    head_op_grad_norm: float = 0.0

    # Episode return tracking (PRIMARY RL METRIC - per DRL review)
    episode_return_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    current_episode_return: float = 0.0
    current_episode: int = 0  # Current episode number for return history display

    # History for trend sparklines (last 10 values)
    policy_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    value_loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    grad_norm_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    entropy_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    explained_variance_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    kl_divergence_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))
    clip_fraction_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    # PPO inner loop context
    inner_epoch: int = 0  # Current inner optimization epoch
    ppo_batch: int = 0  # Current batch within PPO update (use ppo_batch to avoid conflict with any existing 'batch')

    # Action distribution (Actions panel)
    action_counts: dict[str, int] = field(default_factory=dict)
    # FIX: Added total_actions for percentage calculation in TamiyoBrain
    total_actions: int = 0

    # PPO data received flag
    ppo_data_received: bool = False

    # Recent decisions list (up to 3, each visible for at least 10 seconds)
    recent_decisions: list["DecisionSnapshot"] = field(default_factory=list)

    # A/B testing identification (None when not in A/B mode)
    group_id: str | None = None


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
    cpu_percent: float | None = 0.0

    # RAM
    ram_used_gb: float | None = 0.0
    ram_total_gb: float | None = 0.0

    # Throughput
    epochs_per_second: float = 0.0
    batches_per_hour: float = 0.0

    # Host network
    host_params: int = 0

    @property
    def has_memory_alarm(self) -> bool:
        """Check if any device exceeds 90% memory usage."""
        # Check RAM
        if (self.ram_total_gb is not None and self.ram_used_gb is not None and
            self.ram_total_gb > 0 and (self.ram_used_gb / self.ram_total_gb) > 0.90):
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
        if (self.ram_total_gb is not None and self.ram_used_gb is not None and
            self.ram_total_gb > 0 and (self.ram_used_gb / self.ram_total_gb) > 0.90):
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
    - alpha_shock: Convex penalty on alpha deltas (negative if triggered)
    - ratio_penalty: Penalty for extreme policy ratios (negative if triggered)
    - stage_bonus: Bonus for reaching advanced lifecycle stages (BLENDING+)
    - fossilize_terminal_bonus: Large terminal bonus for successful fossilization
    - blending_warning: Warning signal during blending phase (negative)
    - holding_warning: Warning signal during holding period
    - hindsight_credit: Retroactive credit when beneficiary fossilizes (blue bonus)
    - scaffold_count: Number of scaffolds that contributed (debugging)
    - avg_scaffold_delay: Average epochs since scaffolding interactions (debugging)
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
    alpha_shock: float = 0.0
    ratio_penalty: float = 0.0

    # Bonuses
    stage_bonus: float = 0.0
    fossilize_terminal_bonus: float = 0.0

    # Warnings
    blending_warning: float = 0.0
    holding_warning: float = 0.0

    # Hindsight credit (scaffold contribution bonus) - Phase 3.2
    hindsight_credit: float = 0.0
    scaffold_count: int = 0  # Number of scaffolds that contributed
    avg_scaffold_delay: float = 0.0  # Average epochs since scaffolding

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
    chosen_action: str  # "GERMINATE", "ADVANCE", "SET_ALPHA_TARGET", "PRUNE", "FOSSILIZE", "WAIT"
    chosen_slot: str | None  # Target slot for action (None for WAIT)
    confidence: float  # Action probability (0-1)
    expected_value: float  # Value estimate before action
    actual_reward: float | None  # Actual reward received (None if pending)
    alternatives: list[tuple[str, float]]  # [(action_name, probability), ...]
    # Unique ID for click-to-pin targeting
    decision_id: str = ""
    # Pinned decisions never get replaced
    pinned: bool = False
    # Environment ID that made this decision (for TD advantage tracking)
    env_id: int = 0

    # Per-decision metrics (per DRL review)
    # Note: expected_value (above) contains V(s), no need for separate value_estimate
    value_residual: float = 0.0   # r - V(s): immediate reward minus value estimate
    td_advantage: float | None = None  # r + γV(s') - V(s): true TD(0) advantage (None until next step)

    # Decision-specific entropy (per DRL review - more useful than policy entropy)
    decision_entropy: float = 0.0  # -sum(p*log(p)) for this action distribution

    # Head choice details for GERMINATE actions (per DRL/UX specialist review)
    # These enable decision cards to show sub-decisions like 'bpnt:conv_l(87%) tmp:STD'
    chosen_blueprint: str | None = None  # e.g., "conv_light", "attention"
    chosen_tempo: str | None = None  # "FAST", "STANDARD", "SLOW"
    chosen_style: str | None = None  # "LINEAR_ADD", "GATED_GATE", etc. (replaces alpha_curve)
    blueprint_confidence: float = 0.0  # Probability of chosen blueprint
    tempo_confidence: float = 0.0  # Probability of chosen tempo
    op_confidence: float = 0.0  # Probability of chosen operation (per DRL review)


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
    entropy_anneal: dict[str, float] = field(default_factory=dict)  # Entropy schedule config


@dataclass
class BestRunRecord:
    """Historical record of a best run for the leaderboard.

    Captured at batch end when an env achieves a new personal best.
    Shows both the peak accuracy and where the run ended up.

    INTERACTIVE FEATURES:
    - Left-click: Opens historical detail modal showing full env state at peak
    - Right-click: Pins the record (prevents removal from leaderboard)

    Reference: tui.py lines 103-114 (BestRunRecord dataclass)
    """
    env_id: int
    episode: int  # Batch/episode number (0-indexed)
    peak_accuracy: float  # Best accuracy achieved during this run
    final_accuracy: float  # Accuracy at the end of the batch
    epoch: int = 0  # Epoch within episode when best was achieved
    seeds: dict[str, SeedState] = field(default_factory=dict)  # Seeds at peak
    slot_ids: list[str] = field(default_factory=list)  # All slot IDs (for showing DORMANT)
    growth_ratio: float = 1.0  # Model size ratio: (host + fossilized) / host

    # === Interactive features (like DecisionSnapshot) ===
    record_id: str = ""  # Unique ID for click targeting
    pinned: bool = False  # Pinned records never get removed from leaderboard

    # === Full env snapshot at peak (for historical detail view) ===
    # Reward breakdown at peak
    reward_components: "RewardComponents | None" = None
    # Counterfactual matrix at peak (if available)
    counterfactual_matrix: "CounterfactualSnapshot | None" = None
    # Recent actions leading to peak (last 10)
    action_history: list[str] = field(default_factory=list)
    # Reward/accuracy history up to peak
    reward_history: list[float] = field(default_factory=list)
    accuracy_history: list[float] = field(default_factory=list)
    # Host metrics at peak
    host_loss: float = 0.0
    host_params: int = 0
    # Lifecycle counts at peak
    fossilized_count: int = 0
    pruned_count: int = 0
    # A/B cohort (for dot indicator)
    reward_mode: str | None = None
    # Seed graveyard: per-blueprint lifecycle stats at time of snapshot
    blueprint_spawns: dict[str, int] = field(default_factory=dict)
    blueprint_fossilized: dict[str, int] = field(default_factory=dict)
    blueprint_prunes: dict[str, int] = field(default_factory=dict)


@dataclass
class EventLogEntry:
    """Single event log entry for display in Event Log panel.

    Reference: Used by aggregator to build event_log list

    Design: message is GENERIC (e.g., "Germinated", "Stage changed").
    Specific values (slot_id, reward, blueprint) go in metadata dict
    for display in detail modal. This allows proper rollup by event_type.
    """
    timestamp: str  # Formatted as HH:MM:SS
    event_type: str  # REWARD_COMPUTED, SEED_GERMINATED, etc.
    env_id: int | None  # None for global events (PPO, BATCH)
    message: str  # Generic message (specific values in metadata)
    episode: int = 0  # Episode number for grouping
    relative_time: str = ""  # "(2s ago)" relative time string
    # Structured metadata for detail view (slot_id, reward, blueprint, etc.)
    metadata: dict[str, str | int | float] = field(default_factory=dict)


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
    max_batches: int = 100  # Maximum batches per episode
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

    # Batch-level aggregates (from BATCH_EPOCH_COMPLETED)
    batch_avg_reward: float = 0.0  # Average reward for last batch
    batch_total_episodes: int = 0  # Total episodes in training run

    # Rolling average history (mean accuracy across all envs over time)
    mean_accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Event log (most recent last)
    event_log: list[EventLogEntry] = field(default_factory=list)

    # Best runs leaderboard (top 10 by peak accuracy)
    best_runs: list[BestRunRecord] = field(default_factory=list)

    # Cumulative counts (never reset, tracks entire training run)
    cumulative_fossilized: int = 0
    cumulative_pruned: int = 0
    # Cumulative graveyard (per-blueprint lifecycle stats across entire run)
    cumulative_blueprint_spawns: dict[str, int] = field(default_factory=dict)
    cumulative_blueprint_fossilized: dict[str, int] = field(default_factory=dict)
    cumulative_blueprint_prunes: dict[str, int] = field(default_factory=dict)

    # Aggregate slot state across all environments (for TamiyoBrain slot summary)
    # Keys: DORMANT, GERMINATED, TRAINING, BLENDING, HOLDING, FOSSILIZED
    slot_stage_counts: dict[str, int] = field(default_factory=dict)
    total_slots: int = 0  # n_envs * n_slots_per_env
    active_slots: int = 0  # Slots not in DORMANT state
    avg_epochs_in_stage: float = 0.0  # Mean epochs_in_stage for non-dormant slots

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


# Metric-specific thresholds (per DRL review)
# Higher threshold = more change needed to trigger improving/warning
TREND_THRESHOLDS: dict[str, float] = {
    "episode_return": 0.15,   # 15% - returns vary naturally
    "entropy": 0.08,          # 8% - entropy changes are meaningful
    "policy_loss": 0.20,      # 20% - policy loss is noisy
    "value_loss": 0.20,       # 20% - value loss is noisy
    "kl_divergence": 0.30,    # 30% - KL varies widely
    "clip_fraction": 0.30,    # 30% - clip fraction is variable
    "grad_norm": 0.25,        # 25% - gradients vary batch-to-batch
    "expl_var": 0.15,         # 15% - explained variance
    "default": 0.15,          # 15% - fallback
}


def detect_trend(
    values: list[float] | deque[float],
    metric_name: str = "default",
    metric_type: str = "loss",
) -> str:
    """Detect trend pattern in metric values with RL-appropriate thresholds.

    Uses 10-sample windows (not 5) because RL metrics are inherently noisy.
    Uses variance ratio for volatility (not CV) per DRL review.

    Args:
        values: Recent metric values (oldest first).
        metric_name: Specific metric for threshold lookup.
        metric_type: "loss" (lower=better) or "accuracy" (higher=better).

    Returns:
        Trend label: "improving", "stable", "volatile", "warning"
    """
    if len(values) < 5:
        return "stable"

    values_list = list(values)

    # Use 10-sample windows for RL (per DRL review)
    window_size = min(10, len(values_list) // 2)
    if window_size < 3:
        return "stable"

    recent = values_list[-window_size:]
    # Use the window immediately before recent (non-overlapping)
    if len(values_list) >= 2 * window_size:
        older = values_list[-2*window_size:-window_size]
    else:
        # Not enough data for proper comparison - use what we have
        older = values_list[:window_size]

    if not recent or not older:
        return "stable"

    recent_mean = sum(recent) / len(recent)
    older_mean = sum(older) / len(older)

    # Volatility check: variance ratio > 3x (per DRL review, not CV > 50%)
    recent_var = sum((v - recent_mean) ** 2 for v in recent) / len(recent)
    older_var = sum((v - older_mean) ** 2 for v in older) / len(older)

    VOLATILITY_EPSILON = 1e-8
    if older_var > VOLATILITY_EPSILON and recent_var / older_var > 3.0:
        return "volatile"

    # Get metric-specific threshold
    threshold_pct = TREND_THRESHOLDS.get(metric_name, TREND_THRESHOLDS["default"])
    change_threshold = threshold_pct * abs(older_mean) if older_mean != 0 else 0.01
    change = recent_mean - older_mean

    if metric_type == "loss":
        # For loss: decreasing is good, increasing is bad
        if change <= -change_threshold:  # Include boundary
            return "improving"
        elif change >= change_threshold:  # Include boundary
            return "warning"
    else:
        # For accuracy/return: increasing is good, decreasing is bad
        if change >= change_threshold:  # Include boundary
            return "improving"
        elif change <= -change_threshold:  # Include boundary
            return "warning"

    return "stable"


def trend_to_indicator(trend: str) -> tuple[str, str]:
    """Convert trend label to display indicator and style.

    Per UX review: No brackets, just single char with color.

    Returns:
        Tuple of (indicator_string, rich_style)
    """
    indicators = {
        "improving": ("^", "green"),
        "stable": ("-", "dim"),
        "volatile": ("~", "yellow"),
        "warning": ("v", "red"),
    }
    return indicators.get(trend, ("-", "dim"))


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
