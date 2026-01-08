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

import numpy as np

from esper.leyline import HEAD_NAMES, NUM_OPS


def compute_entropy_velocity(entropy_history: deque[float] | list[float]) -> float:
    """Compute rate of entropy change (d(entropy)/d(batch)).

    Uses numpy linear regression over last 10 samples for performance and stability.

    Args:
        entropy_history: Recent entropy values (oldest first).

    Returns:
        Velocity in entropy units per batch. Negative = declining.
    """
    if len(entropy_history) < 5:
        return 0.0

    values = np.array(list(entropy_history)[-10:])
    n = len(values)
    x = np.arange(n)

    # Least squares slope using numpy (10x faster than pure Python)
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def compute_correlation(
    x_values: deque[float] | list[float],
    y_values: deque[float] | list[float],
) -> float:
    """Compute Pearson correlation between two metric histories.

    Returns:
        Correlation coefficient (-1 to +1), or 0.0 if insufficient data
        or zero variance (to avoid NaN).
    """
    if len(x_values) < 5 or len(y_values) < 5:
        return 0.0

    x = list(x_values)[-10:]
    y = list(y_values)[-10:]

    n = min(len(x), len(y))
    x, y = x[-n:], y[-n:]

    # Re-check after alignment (n could be < 5 if sequences have different lengths)
    if n < 5:
        return 0.0

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    numerator: float = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    x_var: float = sum((xi - x_mean) ** 2 for xi in x)
    y_var: float = sum((yi - y_mean) ** 2 for yi in y)

    # Check variance product BEFORE square root for better numerical stability
    EPSILON = 1e-10
    variance_product = x_var * y_var
    if variance_product < EPSILON * EPSILON:
        return 0.0

    denominator: float = variance_product ** 0.5
    return numerator / denominator


def compute_collapse_risk(
    entropy_history: deque[float] | list[float],
    critical_threshold: float = 0.3,
    warning_threshold: float = 0.5,
    max_healthy_entropy: float = 1.39,
    previous_risk: float = 0.0,
    hysteresis: float = 0.08,
) -> float:
    """Compute entropy collapse risk score (0.0 to 1.0).

    Risk is based on:
    - Current distance from critical threshold (proximity)
    - Velocity (rate of decline)
    - Hysteresis to prevent risk score flapping

    Args:
        entropy_history: Recent entropy values (oldest first).
        critical_threshold: Entropy below this is collapsed.
        warning_threshold: Entropy below this is concerning.
        max_healthy_entropy: Expected healthy entropy level (default ~ln(4)).
        previous_risk: Previous risk score for hysteresis.
        hysteresis: Risk change must exceed this to update.

    Returns:
        0.0 = no risk, 1.0 = imminent/active collapse
    """
    if len(entropy_history) < 5:
        return 0.0

    values = list(entropy_history)
    current = values[-1]
    velocity = compute_entropy_velocity(entropy_history)

    # Already collapsed
    if current <= critical_threshold:
        return 1.0

    # Calculate proximity-based risk (being near critical is risky even if stable)
    # max_healthy_entropy is the expected healthy entropy level (default ~ln(4))
    denominator = max_healthy_entropy - critical_threshold
    if denominator <= 0:
        # Invalid threshold configuration - treat as critical proximity
        proximity = 1.0
    else:
        proximity = 1.0 - (current - critical_threshold) / denominator
        proximity = max(0.0, min(1.0, proximity))
    proximity_risk = proximity * 0.3  # Cap proximity contribution at 0.3

    # Rising or stable entropy = minimal risk (just proximity)
    EPSILON = 1e-6
    if velocity >= -EPSILON:
        base_risk = proximity_risk
    else:
        # Declining entropy - calculate time to collapse
        distance = current - critical_threshold
        time_to_collapse = distance / abs(velocity)

        # Time-based risk thresholds
        if time_to_collapse > 100:
            time_risk = 0.1
        elif time_to_collapse > 50:
            time_risk = 0.25
        elif time_to_collapse > 20:
            time_risk = 0.5
        elif time_to_collapse > 10:
            time_risk = 0.7
        else:
            time_risk = 0.9

        # Combine time and proximity risks
        if current < warning_threshold:
            base_risk = 0.5 * time_risk + 0.5 * proximity_risk
        else:
            base_risk = 0.7 * time_risk + 0.3 * proximity_risk

    # Apply hysteresis to prevent flapping
    if abs(base_risk - previous_risk) < hysteresis:
        return previous_risk

    return min(1.0, max(0.0, base_risk))


@dataclass
class SeedLifecycleStats:
    """Seed lifecycle aggregate metrics for TamiyoBrain display.

    Tracks germination, pruning, and fossilization rates over the training run.
    Used by SlotsPanel to show lifecycle health beyond current slot states.
    """
    # Cumulative counts (entire run)
    germination_count: int = 0
    prune_count: int = 0
    fossilize_count: int = 0

    # Current active count
    active_count: int = 0
    total_slots: int = 0

    # Rates (per episode) - computed from counts / current_episode
    germination_rate: float = 0.0  # Germinations per episode
    prune_rate: float = 0.0  # Prunes per episode
    fossilize_rate: float = 0.0  # Fossilizations per episode

    # Quality metrics
    blend_success_rate: float = 0.0  # fossilized / (fossilized + pruned)
    avg_lifespan_epochs: float = 0.0  # Mean epochs a seed lives before terminal state

    # Trend indicators (computed from rate history)
    germination_trend: str = "stable"  # "rising", "stable", "falling"
    prune_trend: str = "stable"
    fossilize_trend: str = "stable"


@dataclass
class ObservationStats:
    """Observation space health metrics.

    Tracks feature statistics to catch input distribution issues
    before they propagate to NaN gradients.
    """
    # Per-feature-group statistics (mean/std across batch)
    slot_features_mean: float = 0.0
    slot_features_std: float = 0.0
    host_features_mean: float = 0.0
    host_features_std: float = 0.0
    context_features_mean: float = 0.0  # Epoch progress, action history, etc.
    context_features_std: float = 0.0

    # Outlier detection
    outlier_pct: float = 0.0  # Fraction of observations outside 3σ (rendered as X.X%)

    # Saturation / clipping indicators (computed on NORMALIZED obs)
    near_clip_pct: float = 0.0  # Fraction with |x_norm| >= 0.9*clip
    clip_pct: float = 0.0  # Fraction clamped at |x_norm| == clip

    # Numerical health
    nan_count: int = 0  # NaN values detected in observations
    inf_count: int = 0  # Inf values detected in observations
    nan_pct: float = 0.0  # Fraction of NaNs in raw obs tensor
    inf_pct: float = 0.0  # Fraction of Infs in raw obs tensor

    # Normalization drift (running stats divergence)
    normalization_drift: float = 0.0  # How much running mean/std has shifted

    # Batch context
    batch_size: int = 0  # Number of environments in the batch


@dataclass
class EpisodeStats:
    """Episode-level aggregate metrics.

    Provides insight into episode structure beyond just returns.
    Helps diagnose timeout issues, early termination, and success rates.
    """
    # Episode length statistics
    length_mean: float = 0.0
    length_std: float = 0.0
    length_min: int = 0
    length_max: int = 0

    # Outcome tracking (over recent N episodes)
    total_episodes: int = 0
    episodes_per_second: float = 0.0  # Throughput: inner episodes completed per second
    timeout_count: int = 0  # Episodes that hit max steps
    success_count: int = 0  # Episodes that achieved goal
    early_termination_count: int = 0  # Episodes terminated early (failure/reset)

    # Derived rates
    timeout_rate: float = 0.0  # timeout_count / total_episodes
    success_rate: float = 0.0  # success_count / total_episodes
    early_termination_rate: float = 0.0  # early_termination_count / total_episodes

    # Steps per action type (insight into action efficiency)
    steps_per_germinate: float = 0.0  # Avg steps between GERMINATE actions
    steps_per_prune: float = 0.0  # Avg steps between PRUNE actions
    steps_per_fossilize: float = 0.0  # Avg steps between FOSSILIZE actions

    # === DRL Diagnostic Metrics (per DRL expert review) ===
    # These replace the useless Length/Outcomes metrics for fixed-length episodes

    # Action entropy: Policy sharpness indicator (0=deterministic, 1=uniform random)
    # Normalized Shannon entropy: H = -sum(p(a)*log(p(a))) / log(|A|)
    # Good: 0.3-0.5 (converging), Bad high: >0.8 (random), Bad low: <0.1 (collapsed)
    action_entropy: float = 0.0

    # Yield rate: Seed efficiency (fossilizations / germinations)
    # Good: 0.4-0.7 (healthy churn), Bad low: <0.2 (thrashing), Bad high: >0.9 (too conservative)
    yield_rate: float = 0.0

    # Slot utilization: Capacity usage (active_slots / max_slots)
    # Good: 0.4-0.8, Bad low: <0.2 (WAIT spam), Bad high: 1.0 constant (germinate spam)
    slot_utilization: float = 0.0

    # Completion trend
    completion_trend: str = "stable"  # "improving", "stable", "declining"


@dataclass
class ShapleyEstimate:
    """Shapley value estimate for a single slot.

    Contains the mean contribution and uncertainty bounds.
    """
    mean: float = 0.0  # Expected marginal contribution
    std: float = 0.0   # Standard deviation (uncertainty)
    n_samples: int = 0  # Number of permutation samples used


@dataclass
class ShapleySnapshot:
    """Shapley value attribution for all slots in an environment.

    Computed via permutation sampling at PPO batch boundaries.
    More accurate than simple ablation but more expensive.
    """
    slot_ids: tuple[str, ...] = ()  # ("r0c0", "r0c1", "r0c2")
    values: dict[str, ShapleyEstimate] = field(default_factory=dict)
    epoch: int = 0  # PPO batch index (1-based) when computed
    timestamp: datetime | None = None

    def get_mean(self, slot_id: str) -> float:
        """Get mean Shapley value for a slot."""
        if slot_id in self.values:
            return self.values[slot_id].mean
        return 0.0

    def get_significance(self, slot_id: str, z: float = 1.96) -> bool:
        """Check if slot's contribution is statistically significant (95% CI)."""
        if slot_id not in self.values:
            return False
        est = self.values[slot_id]
        return abs(est.mean) > z * est.std if est.std > 0 else est.mean != 0

    def ranked_slots(self) -> list[tuple[str, float]]:
        """Return slots ranked by mean contribution (descending)."""
        return sorted(
            [(slot_id, est.mean) for slot_id, est in self.values.items()],
            key=lambda x: x[1],
            reverse=True,
        )


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

    # Shapley value attribution (from ANALYTICS_SNAPSHOT kind="shapley_computed")
    # More accurate than counterfactual marginals, computed at episode boundaries
    shapley_snapshot: ShapleySnapshot = field(default_factory=ShapleySnapshot)

    # History for sparklines (maxlen=50)
    reward_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Reward tracking
    cumulative_reward: float = 0.0  # Sum of all rewards received (for entire episode)

    # Best tracking
    best_reward: float = float('-inf')
    best_reward_epoch: int = 0
    best_accuracy: float = 0.0
    best_accuracy_epoch: int = 0
    best_accuracy_episode: int = 0
    best_seeds: dict[str, SeedState] = field(default_factory=dict)

    # Snapshot volatile state at peak accuracy (for historical detail modal)
    # These capture the state AT THE MOMENT best_accuracy was achieved,
    # not the state at batch end (which would be stale/incorrect).
    best_reward_components: RewardComponents | None = None
    best_counterfactual_matrix: CounterfactualSnapshot | None = None
    best_shapley_snapshot: ShapleySnapshot | None = None
    best_action_history: list[str] = field(default_factory=list)

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
    # Execution feedback for last action (invalid slot+op combos are rejected at runtime)
    last_action_success: bool = True

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
    # Captured from TRAINING_STARTED payload.reward_mode
    # Values: "shaped", "simplified", "sparse", etc.
    reward_mode: str | None = None

    # Governor rollback state (catastrophic failure indicator)
    # UI flashes the env row for a few seconds after rollback_timestamp is set.
    rolled_back: bool = False
    rollback_reason: str = ""  # "nan", "lobotomy", "divergence"
    rollback_timestamp: datetime | None = None

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
        """Add reward and update best/cumulative tracking."""
        self.reward_history.append(reward)
        self.cumulative_reward += reward
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

            # Snapshot volatile state at peak (for historical detail modal)
            # Deep copy to preserve state at this moment
            if isinstance(self.reward_components, RewardComponents):
                self.best_reward_components = RewardComponents(
                    **self.reward_components.__dict__
                )
            else:
                self.best_reward_components = None

            if self.counterfactual_matrix and self.counterfactual_matrix.slot_ids:
                self.best_counterfactual_matrix = CounterfactualSnapshot(
                    slot_ids=self.counterfactual_matrix.slot_ids,
                    configs=list(self.counterfactual_matrix.configs),
                    strategy=self.counterfactual_matrix.strategy,
                    compute_time_ms=self.counterfactual_matrix.compute_time_ms,
                )
            else:
                self.best_counterfactual_matrix = None

            # NOTE: best_shapley_snapshot is NOT captured here.
            # Shapley values are computed at episode END, so at peak accuracy time
            # the shapley_snapshot contains stale data from the previous episode.
            # The aggregator updates best_shapley_snapshot when shapley_computed
            # telemetry arrives, which is the correct time to capture it.

            # Snapshot action history (last 10 actions leading to peak)
            self.best_action_history = list(self.action_history)
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
class ValueFunctionMetrics:
    """Value function quality diagnostics for DRL training health.

    Per DRL expert review: Value function quality is THE primary diagnostic
    for RL training failures. Low V-Return correlation means advantage
    estimates are garbage, regardless of how healthy policy metrics look.

    Grouped separately to prevent TamiyoState bloat.
    """

    # V-Return Correlation (PRIMARY VALUE METRIC)
    # Pearson correlation between V(s) predictions and actual returns
    # Low (<0.5) = value network isn't learning, advantages are noise
    # High (>0.8) = value network is well-calibrated
    v_return_correlation: float = 0.0

    # TD Error Statistics
    # TD error = r + γV(s') - V(s) (true TD(0) error)
    # High mean = biased value estimates (target network staleness)
    # High std = noisy gradient targets (normal in early training)
    td_error_mean: float = 0.0
    td_error_std: float = 0.0

    # Bellman Error (|V(s) - (r + γV(s'))|²)
    # Spikes often PRECEDE NaN losses - early warning signal
    bellman_error: float = 0.0

    # Return Distribution Percentiles (per DRL expert - catches bimodal policies)
    # If p90 - p10 is huge, policy is inconsistent (some episodes succeed, others fail)
    return_p10: float = 0.0
    return_p50: float = 0.0  # Median (more robust than mean)
    return_p90: float = 0.0

    # Return distribution shape (for quick diagnosis)
    return_skewness: float = 0.0  # >0 = right-skewed (few big wins)
    return_variance: float = 0.0  # High = inconsistent policy

    # Historical tracking for correlation computation
    # These are populated by aggregator from DecisionSnapshot td_advantage values
    value_predictions: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    actual_returns: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    td_errors: deque[float] = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class GradientQualityMetrics:
    """Gradient quality diagnostics for DRL training health.

    Grouped separately to prevent TamiyoState bloat (per code review).

    Note: Uses Coefficient of Variation (CV) not SNR per DRL expert review.
    The original plan had inverted SNR (var/mean² is noise-to-signal).
    CV = sqrt(var)/|mean| is standard and self-explanatory.
    """
    # Gradient Coefficient of Variation (per DRL expert - replaces inverted SNR)
    # Low CV (<0.5) = high signal quality, High CV (>2.0) = noisy gradients
    gradient_cv: float = 0.0

    # Directional Clip Fraction (per DRL expert recommendation)
    # These track WHERE clipping occurs, not WHETHER policy improved:
    # clip+ = r > 1+ε (probability increases were capped)
    # clip- = r < 1-ε (probability decreases were capped)
    # Asymmetry indicates directional policy drift; symmetric high values are normal
    clip_fraction_positive: float = 0.0
    clip_fraction_negative: float = 0.0

    # Note: param_update_magnitude REMOVED per PyTorch expert review
    # - Conflates gradient magnitude with learning rate
    # - Existing grad_norm, dead_layers, exploding_layers already provide this signal

    # Note: minibatch_gradient_variance REMOVED per PyTorch expert review
    # - Not applicable to recurrent PPO (single-batch processing due to LSTM coherence)


@dataclass
class InfrastructureMetrics:
    """PyTorch infrastructure health metrics.

    Grouped separately to prevent TamiyoState bloat (per code review).
    Collected every N batches to amortize CPU-GPU sync overhead.
    """
    # CUDA Memory (PyTorch expert recommendation)
    cuda_memory_allocated_gb: float = 0.0   # torch.cuda.memory_allocated()
    cuda_memory_reserved_gb: float = 0.0    # torch.cuda.memory_reserved()
    cuda_memory_peak_gb: float = 0.0        # torch.cuda.max_memory_allocated()
    cuda_memory_fragmentation: float = 0.0  # 1 - (allocated/reserved), >0.3 = pressure
    dataloader_wait_ratio: float = 0.0      # Fraction of step time spent waiting on data

    # torch.compile Status (captured at training start - static session metadata)
    # Note: graph_break_count/compile_healthy removed - not accessible via PyTorch API
    # Compile issues will surface in throughput metrics (fps, step_time_ms)
    compile_enabled: bool = False
    compile_backend: str = ""    # "inductor", "eager", etc.
    compile_mode: str = ""       # "default", "reduce-overhead", "max-autotune"

    @property
    def memory_usage_percent(self) -> float:
        """Memory usage as percentage for compact display."""
        if self.cuda_memory_reserved_gb <= 0:
            return 0.0
        return (self.cuda_memory_allocated_gb / self.cuda_memory_reserved_gb) * 100


@dataclass
class TamiyoState:
    """Tamiyo policy agent state - ALL metrics from existing TUI.

    Reference: tui.py TUIState policy metrics + Tamiyo panel renderer

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
    # NaN = no data (std too low or no valid advantages); 0 = symmetric/normal
    advantage_skewness: float = float("nan")  # >0 right-skewed (few big wins), <0 left-skewed (few big losses)
    advantage_kurtosis: float = float("nan")  # >0 heavy tails (outliers), <0 light tails; >3 is super-Gaussian
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    # Pre-normalization stats (raw learning signal magnitude)
    advantage_raw_mean: float = 0.0
    advantage_raw_std: float = 0.0
    # Advantage distribution health (NaN = no data)
    advantage_positive_ratio: float = float("nan")  # Fraction of positive advantages (healthy: 0.4-0.6)

    # Log probability extremes (NaN predictor)
    # Values < -50 warning, < -100 critical (numerical underflow imminent)
    # NaN = no data (no PPO updates yet); 0.0 = deterministic action (valid but rare)
    log_prob_min: float = float("nan")  # Most negative log prob this update
    log_prob_max: float = float("nan")  # Highest log prob (should be <= 0)

    # D5: Slot Saturation Diagnostics
    # Track decision agency to understand PPO stability under capacity pressure.
    # When slots saturate, action space collapses to WAIT-only (forced steps).
    decision_density: float = 1.0  # Fraction with agency (1 - forced_step_ratio), higher = healthier
    forced_step_ratio: float = 0.0  # Fraction of forced steps (lower = healthier)
    advantage_std_floored: bool = False  # True if std clamped to floor (degenerate batch)
    pre_norm_advantage_std: float | None = None  # Raw advantage std (before normalization)
    decision_density_history: deque[float] = field(default_factory=lambda: deque(maxlen=10))

    # Gradient health (shown in Vitals)
    dead_layers: int = 0
    exploding_layers: int = 0
    nan_grad_count: int = 0  # NaN gradient count
    inf_grad_count: int = 0  # Inf gradient count
    # Per-head NaN/Inf latch (indicator lights - once True, stays True for entire run)
    # Pre-populated with all HEAD_NAMES keys to enable direct access without .get()
    # Keys: slot, blueprint, style, tempo, alpha_target, alpha_speed, alpha_curve, op
    head_nan_latch: dict[str, bool] = field(
        default_factory=lambda: {head: False for head in HEAD_NAMES}
    )
    head_inf_latch: dict[str, bool] = field(
        default_factory=lambda: {head: False for head in HEAD_NAMES}
    )
    layer_gradient_health: dict[str, float] | None = None  # Per-layer gradient health metrics
    entropy_collapsed: bool = False  # Entropy collapse detected

    # LSTM hidden state health (B7-DRL-04)
    # LSTM states can become corrupted during BPTT - tracked for early warning
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
    lstm_has_nan: bool = False  # NaN detected in LSTM hidden state
    lstm_has_inf: bool = False  # Inf detected in LSTM hidden state

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

    # Previous gradient norms (Policy V2 - for trend detection)
    # Enables distinguishing transient spikes from sustained gradient issues
    head_slot_grad_norm_prev: float = 0.0
    head_blueprint_grad_norm_prev: float = 0.0
    head_style_grad_norm_prev: float = 0.0
    head_tempo_grad_norm_prev: float = 0.0
    head_alpha_target_grad_norm_prev: float = 0.0
    head_alpha_speed_grad_norm_prev: float = 0.0
    head_alpha_curve_grad_norm_prev: float = 0.0
    head_op_grad_norm_prev: float = 0.0

    # Per-head PPO ratios (Policy V2 - multi-head ratio explosion detection)
    # Individual head ratios can look healthy while joint ratio exceeds clip range
    head_slot_ratio_max: float = 1.0
    head_blueprint_ratio_max: float = 1.0
    head_style_ratio_max: float = 1.0
    head_tempo_ratio_max: float = 1.0
    head_alpha_target_ratio_max: float = 1.0
    head_alpha_speed_ratio_max: float = 1.0
    head_alpha_curve_ratio_max: float = 1.0
    head_op_ratio_max: float = 1.0
    joint_ratio_max: float = 1.0  # Product of per-head ratios

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
    action_counts: dict[str, int] = field(default_factory=dict)  # Current batch only
    total_actions: int = 0  # Current batch only
    # Cumulative action counts across all batches (for "total run" display)
    cumulative_action_counts: dict[str, int] = field(default_factory=dict)
    cumulative_total_actions: int = 0

    # PPO data received flag
    ppo_data_received: bool = False

    # Recent decisions list (up to MAX_DECISIONS=8, expires at 2 minutes)
    recent_decisions: list["DecisionSnapshot"] = field(default_factory=list)

    # A/B testing identification (None when not in A/B mode)
    group_id: str | None = None

    # Entropy prediction (computed from entropy_history)
    entropy_velocity: float = 0.0          # d(entropy)/d(batch), negative = declining
    collapse_risk_score: float = 0.0       # 0.0-1.0, >0.7 = high risk
    _previous_risk: float = 0.0            # For hysteresis (not serialized)

    # Entropy-clip correlation (for policy collapse pattern detection)
    # Negative correlation + low entropy + high clip = COLLAPSE RISK
    entropy_clip_correlation: float = 0.0

    # Value function statistics (for divergence detection)
    value_mean: float = 0.0
    value_std: float = 0.0
    value_min: float = 0.0
    value_max: float = 0.0
    initial_value_spread: float | None = None  # Set after warmup for relative thresholds

    # Op-conditioned Q-values (Policy V2 - Q(s,op) architecture)
    # Vector aligns to LifecycleOp/NUM_OPS ordering.
    op_q_values: tuple[float, ...] = field(
        default_factory=lambda: tuple(float("nan") for _ in range(NUM_OPS))
    )
    op_valid_mask: tuple[bool, ...] = field(
        default_factory=lambda: tuple(False for _ in range(NUM_OPS))
    )
    # Q-value analysis metrics
    q_variance: float = 0.0  # Variance across ops (low = critic ignoring op conditioning)
    q_spread: float = 0.0  # max(Q) - min(Q) across ops

    # Action feedback (Policy V2 - added to observations for credit assignment)
    last_action_success: bool = True  # Whether previous action executed successfully
    last_action_op: str = "WAIT"  # Previous operation for context

    # === Nested Metric Groups (per code review - prevents schema bloat) ===
    infrastructure: InfrastructureMetrics = field(default_factory=InfrastructureMetrics)
    gradient_quality: GradientQualityMetrics = field(default_factory=GradientQualityMetrics)
    value_function: ValueFunctionMetrics = field(default_factory=ValueFunctionMetrics)


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
    - stable_val_acc: Stable accuracy used for escrow/progress gating (min over a short window)
    - escrow_credit_prev: Prior escrow credit state (RewardMode.ESCROW)
    - escrow_credit_target: Target credit computed from stable accuracy (RewardMode.ESCROW)
    - escrow_delta: Per-step escrow reward payout (RewardMode.ESCROW)
    - escrow_credit_next: Next escrow credit state after applying delta (RewardMode.ESCROW)
    - escrow_forfeit: Terminal clawback for non-fossilized escrow credit (RewardMode.ESCROW)
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

    # Escrow attribution (RewardMode.ESCROW)
    escrow_credit_prev: float = 0.0
    escrow_credit_target: float = 0.0
    escrow_delta: float = 0.0
    escrow_credit_next: float = 0.0
    escrow_forfeit: float = 0.0

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
    stable_val_acc: float | None = None
    last_action: str = ""


@dataclass
class DecisionSnapshot:
    """Snapshot of a single Tamiyo decision for display.

    Captures what Tamiyo saw, what she chose, and the outcome.
    Used for the "Last Decision" section of TamiyoBrain.
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
    action_success: bool | None = None  # Whether action executed successfully (None if unknown)
    # Unique ID for click targeting
    decision_id: str = ""
    # Environment ID that made this decision (for TD advantage tracking)
    env_id: int = 0
    # Episode number for telemetry search
    episode: int = 0
    # Training context when decision was made
    epoch: int = 0
    batch: int = 0

    # Per-decision metrics (per DRL review)
    # Note: expected_value (above) contains V(s), no need for separate value_estimate
    value_residual: float = 0.0   # r - V(s): immediate reward minus value estimate
    td_advantage: float | None = None  # r + γV(s') - V(s): true TD(0) advantage (None until next step)

    # Decision-specific entropy (per DRL review - more useful than policy entropy)
    decision_entropy: float = 0.0  # -sum(p*log(p)) for this action distribution

    # Head choice details for factored action heads (per DRL/UX specialist review)
    # These enable decision cards to show sub-decisions like 'bpnt:conv_l(87%) tmp:STD'
    # See leyline/factored_actions.py for full head specification
    chosen_blueprint: str | None = None  # e.g., "conv_light", "attention"
    chosen_tempo: str | None = None  # "FAST", "STANDARD", "SLOW"
    chosen_style: str | None = None  # "LINEAR_ADD", "GATED_GATE", etc.
    chosen_curve: str | None = None  # "LINEAR", "COSINE", "SIGMOID", etc. (alpha curve shape)
    chosen_alpha_target: str | None = None  # "HALF", "SEVENTY", "FULL" (target alpha amplitude)
    chosen_alpha_speed: str | None = None  # "INSTANT", "FAST", "MEDIUM", "SLOW" (ramp speed)

    # Per-head confidence values (probability of chosen option within each head)
    op_confidence: float = 0.0  # Probability of chosen operation
    slot_confidence: float = 0.0  # Probability of chosen slot
    blueprint_confidence: float = 0.0  # Probability of chosen blueprint
    style_confidence: float = 0.0  # Probability of chosen style
    tempo_confidence: float = 0.0  # Probability of chosen tempo
    alpha_target_confidence: float = 0.0  # Probability of chosen alpha target
    alpha_speed_confidence: float = 0.0  # Probability of chosen alpha speed
    curve_confidence: float = 0.0  # Probability of chosen curve

    # Per-head entropy values (distribution spread - higher means more uncertain)
    # Useful for diagnosing policy collapse (entropy -> 0) or exploration issues
    op_entropy: float = 0.0
    slot_entropy: float = 0.0
    blueprint_entropy: float = 0.0
    style_entropy: float = 0.0
    tempo_entropy: float = 0.0
    alpha_target_entropy: float = 0.0
    alpha_speed_entropy: float = 0.0
    curve_entropy: float = 0.0


@dataclass
class RunConfig:
    """Training run configuration captured at TRAINING_STARTED.

    Stores hyperparameters and config for display in run header.
    """
    seed: int | None = None  # Random seed for reproducibility
    n_episodes: int = 0  # Total env episodes to train
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
    """
    env_id: int
    episode: int  # Batch/episode number (0-indexed)
    peak_accuracy: float  # Best accuracy achieved during this run
    final_accuracy: float  # Accuracy at the end of the batch
    epoch: int = 0  # Epoch within episode when best was achieved
    seeds: dict[str, SeedState] = field(default_factory=dict)  # Seeds at peak
    slot_ids: list[str] = field(default_factory=list)  # All slot IDs (for showing DORMANT)
    growth_ratio: float = 1.0  # Model size ratio: (host + fossilized) / host

    record_id: str = ""  # Unique ID for click targeting

    # === Full env snapshot at peak (for historical detail view) ===
    # Reward breakdown at peak
    reward_components: "RewardComponents | None" = None
    # Counterfactual matrix at peak (if available)
    counterfactual_matrix: "CounterfactualSnapshot | None" = None
    # Shapley attribution snapshot at peak (if available)
    shapley_snapshot: "ShapleySnapshot | None" = None
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
    max_batches: int = 0  # Total PPO update rounds in run (from CLI)
    current_epoch: int = 0
    max_epochs: int = 0
    run_id: str = ""
    task_name: str = ""
    reward_mode: str | None = None  # Primary reward mode for this policy group
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
    cumulative_germinated: int = 0  # Total germinations across run
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

    # === New Diagnostic Metrics (Panel Expansion) ===
    # Seed lifecycle aggregate stats (for SlotsPanel expansion)
    seed_lifecycle: SeedLifecycleStats = field(default_factory=SeedLifecycleStats)

    # Observation health stats (for HealthStatusPanel expansion)
    observation_stats: ObservationStats = field(default_factory=ObservationStats)

    # Episode-level metrics (for new EpisodeMetricsPanel)
    episode_stats: EpisodeStats = field(default_factory=EpisodeStats)

    # Focused env for detail panel
    focused_env_id: int = 0

    # Last action target (for row highlighting in EnvOverview)
    last_action_env_id: int | None = None  # None if no actions yet
    last_action_timestamp: datetime | None = None  # For hysteresis (5s decay)

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
