"""Sanctum Telemetry Aggregator - Transforms event stream into SanctumSnapshot.

Maintains stateful accumulation of telemetry events to build
real-time SanctumSnapshot objects for the Sanctum TUI.

Thread-safe: Uses threading.Lock to protect state during concurrent
access from training thread (process_event) and UI thread (get_snapshot).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import psutil

_logger = logging.getLogger(__name__)

from esper.karn.sanctum.schema import (
    SanctumSnapshot,
    EnvState,
    SeedState,
    TamiyoState,
    SystemVitals,
    GPUStats,
    RewardComponents,
    EventLogEntry,
    BestRunRecord,
    DecisionSnapshot,
    RunConfig,
    CounterfactualConfig,
    CounterfactualSnapshot,
)
from esper.leyline import (
    DEFAULT_GAMMA,
    TrainingStartedPayload,
    EpochCompletedPayload,
    BatchEpochCompletedPayload,
    PPOUpdatePayload,
    RewardComputedPayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    CounterfactualMatrixPayload,
    AnalyticsSnapshotPayload,
)

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent


@dataclass
class PendingDecision:
    """Data needed to compute TD advantage when next V(s') arrives."""
    decision: DecisionSnapshot
    reward: float
    value_s: float  # V(s) at decision time
    env_id: int


# Action normalization map: factored action names -> base action
ACTION_NORMALIZATION = {
    "GERMINATE_CONV_LIGHT": "GERMINATE",
    "GERMINATE_CONV_HEAVY": "GERMINATE",
    "GERMINATE_ATTENTION": "GERMINATE",
    "GERMINATE_MLP": "GERMINATE",
    "FOSSILIZE_G0": "FOSSILIZE",
    "FOSSILIZE_G1": "FOSSILIZE",
    "FOSSILIZE_G2": "FOSSILIZE",
}


def normalize_action(action: str) -> str:
    """Normalize factored action names to base action.

    Args:
        action: Action name, possibly factored (e.g., "GERMINATE_CONV_LIGHT").

    Returns:
        Base action name (e.g., "GERMINATE").
    """
    normalized = ACTION_NORMALIZATION.get(action, action)
    if normalized == action:
        if action.startswith("GERMINATE"):
            return "GERMINATE"
        if action.startswith("SET_ALPHA_TARGET"):
            return "SET_ALPHA_TARGET"
        if action.startswith("FOSSILIZE"):
            return "FOSSILIZE"
        if action.startswith("PRUNE"):
            return "PRUNE"
        if action.startswith("WAIT"):
            return "WAIT"
    return normalized


# Maximum decisions to keep for display
# Must match TamiyoBrain._get_max_decision_cards() upper bound (8)
MAX_DECISIONS = 8


@dataclass
class SanctumAggregator:
    """Aggregates telemetry events into SanctumSnapshot state.

    Thread-safe: process_event() and get_snapshot() can be called
    from different threads safely due to internal locking.

    Handles ALL telemetry events from existing TUIOutput:
    - TRAINING_STARTED: Initialize run context
    - EPOCH_COMPLETED: Update per-env accuracy/loss
    - PPO_UPDATE_COMPLETED: Update Tamiyo policy metrics
    - REWARD_COMPUTED: Update per-env reward components
    - SEED_GERMINATED: Add seed to env
    - SEED_STAGE_CHANGED: Update seed stage
    - SEED_FOSSILIZED: Increment fossilized count
    - SEED_PRUNED: Increment pruned count
    - BATCH_EPOCH_COMPLETED: Update episode/throughput

    Usage:
        agg = SanctumAggregator(num_envs=16)

        # From backend thread
        agg.process_event(event)

        # From UI thread
        snapshot = agg.get_snapshot()
    """

    num_envs: int = 16
    max_event_log: int = 100
    max_history: int = 50

    # Run context
    _run_id: str = ""
    _task_name: str = ""
    _max_epochs: int = 75
    _start_time: float = field(default_factory=time.time)
    _connected: bool = False
    _last_event_ts: float = 0.0

    # Progress tracking
    _current_episode: int = 0
    _current_epoch: int = 0
    _batches_completed: int = 0
    _host_params: int = 0  # Baseline host model params (for growth_ratio)
    _reward_mode: str = ""  # A/B test cohort (shaped, simplified, sparse)
    _current_batch: int = 0  # Current batch index (from BATCH_EPOCH_COMPLETED)
    _batch_avg_accuracy: float = 0.0  # Batch-level average accuracy
    _batch_rolling_accuracy: float = 0.0  # Rolling average for trend display
    _batch_avg_reward: float = 0.0  # Batch average reward
    _batch_total_episodes: int = 0  # Total episodes in run
    _run_config: "RunConfig" = field(default_factory=lambda: RunConfig())

    # Cumulative counts (never reset, tracks entire training run)
    _cumulative_fossilized: int = 0
    _cumulative_pruned: int = 0
    # Cumulative graveyard (per-blueprint lifecycle stats across entire run)
    _cumulative_blueprint_spawns: dict[str, int] = field(default_factory=dict)
    _cumulative_blueprint_fossilized: dict[str, int] = field(default_factory=dict)
    _cumulative_blueprint_prunes: dict[str, int] = field(default_factory=dict)

    # Per-env state: env_id -> EnvState
    _envs: dict[int, EnvState] = field(default_factory=dict)

    # Tamiyo state
    _tamiyo: TamiyoState = field(default_factory=TamiyoState)

    # System vitals
    _vitals: SystemVitals = field(default_factory=SystemVitals)
    _gpu_devices: list[str] = field(default_factory=list)

    # Event log
    _event_log: deque[EventLogEntry] = field(default_factory=lambda: deque(maxlen=100))

    # Focused env for reward panel
    _focused_env_id: int = 0

    # Historical best runs leaderboard (updated at batch end)
    _best_runs: list[BestRunRecord] = field(default_factory=list)

    # Slot configuration (dynamic - populated from training config or observed seeds)
    _slot_ids: list[str] = field(default_factory=list)
    # Lock slot_ids after TRAINING_STARTED provides authoritative list
    _slot_ids_locked: bool = False

    # Rolling average history (mean accuracy across all envs, updated per epoch)
    _mean_accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Pending decisions per env (for delayed TD advantage computation)
    # When we get V(s') for env, we compute td_advantage for its pending decision
    _pending_decisions: dict[int, PendingDecision] = field(default_factory=dict)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        """Initialize state."""
        self._envs = {}
        self._event_log = deque(maxlen=self.max_event_log)
        self._tamiyo = TamiyoState()
        self._vitals = SystemVitals()
        self._gpu_devices = []
        self._best_runs = []
        self._slot_ids = []
        self._slot_ids_locked = False
        self._mean_accuracy_history = deque(maxlen=self.max_history)
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._run_config = RunConfig()

        # Cumulative counters (never reset)
        self._cumulative_fossilized = 0
        self._cumulative_pruned = 0
        self._cumulative_blueprint_spawns = {}
        self._cumulative_blueprint_fossilized = {}
        self._cumulative_blueprint_prunes = {}

        # Pre-create env states
        for i in range(self.num_envs):
            self._ensure_env(i)

    def process_event(self, event: "TelemetryEvent") -> None:
        """Process a telemetry event and update internal state.

        Args:
            event: The telemetry event to process.
        """
        with self._lock:
            self._process_event_unlocked(event)

    def _process_event_unlocked(self, event: "TelemetryEvent") -> None:
        """Process event without locking (caller must hold lock)."""
        # Update last event timestamp
        if event.timestamp:
            self._last_event_ts = event.timestamp.timestamp()
        else:
            self._last_event_ts = time.time()

        # Get event type name
        # hasattr AUTHORIZED by operator on 2025-12-18 15:00:00 UTC
        # Justification: Serialization - handle both enum and string event_type values
        event_type = (
            event.event_type.name
            if hasattr(event.event_type, "name")
            else str(event.event_type)
        )

        # Log event
        self._add_event_log(event, event_type)

        # Route to handler
        if event_type == "TRAINING_STARTED":
            self._handle_training_started(event)
        elif event_type == "EPOCH_COMPLETED":
            self._handle_epoch_completed(event)
        elif event_type == "PPO_UPDATE_COMPLETED":
            self._handle_ppo_update(event)
        elif event_type == "REWARD_COMPUTED":
            self._handle_reward_computed(event)
        elif event_type.startswith("SEED_"):
            self._handle_seed_event(event, event_type)
        elif event_type == "BATCH_EPOCH_COMPLETED":
            self._handle_batch_epoch_completed(event)
        elif event_type == "COUNTERFACTUAL_MATRIX_COMPUTED":
            self._handle_counterfactual_matrix(event)
        elif event_type == "ANALYTICS_SNAPSHOT":
            self._handle_analytics_snapshot(event)

    def get_snapshot(self) -> SanctumSnapshot:
        """Get current SanctumSnapshot.

        Returns:
            Complete snapshot of current aggregator state.
        """
        with self._lock:
            return self._get_snapshot_unlocked()

    def toggle_decision_pin(self, decision_id: str) -> bool:
        """Toggle pin status for a decision by ID.

        Args:
            decision_id: The decision_id to toggle.

        Returns:
            New pin status (True if now pinned, False if unpinned).
        """
        with self._lock:
            for decision in self._tamiyo.recent_decisions:
                if decision.decision_id == decision_id:
                    decision.pinned = not decision.pinned
                    return decision.pinned
        return False

    def toggle_best_run_pin(self, record_id: str) -> bool:
        """Toggle pin status for a best run record by ID.

        Pinned records are never removed from the leaderboard, even when
        newer records with higher accuracy are added.

        Args:
            record_id: The record_id to toggle.

        Returns:
            New pin status (True if now pinned, False if unpinned).
        """
        with self._lock:
            for record in self._best_runs:
                if record.record_id == record_id:
                    record.pinned = not record.pinned
                    return record.pinned
        return False

    def _get_snapshot_unlocked(self) -> SanctumSnapshot:
        """Get snapshot without locking (caller must hold lock)."""
        now = time.time()
        now_dt = datetime.now(timezone.utc)
        staleness = now - self._last_event_ts if self._last_event_ts else float("inf")
        runtime = now - self._start_time if self._connected else 0.0

        # Update system vitals
        self._update_system_vitals()

        # Aggregate action counts from per-step reward telemetry when available.
        # If debug REWARD_COMPUTED telemetry is disabled, fall back to
        # ANALYTICS_SNAPSHOT(action_distribution) which populates self._tamiyo directly.
        aggregated_actions: dict[str, int] = {
            "WAIT": 0,
            "GERMINATE": 0,
            "SET_ALPHA_TARGET": 0,
            "PRUNE": 0,
            "FOSSILIZE": 0,
        }
        total_actions = 0
        for env in self._envs.values():
            for action, count in env.action_counts.items():
                aggregated_actions[action] = aggregated_actions.get(action, 0) + count
            total_actions += env.total_actions
        if total_actions > 0:
            self._tamiyo.action_counts = aggregated_actions
            self._tamiyo.total_actions = total_actions

        # Carousel rotation: expire ONE oldest unpinned decision per cycle if > 30s old
        # This runs every snapshot (250ms), creating natural stagger as each decision
        # expires based on its individual timestamp, not batch replacement.
        decisions = self._tamiyo.recent_decisions
        for i in range(len(decisions) - 1, -1, -1):  # Iterate oldest-first
            d = decisions[i]
            if not d.pinned:
                age = (now_dt - d.timestamp).total_seconds()
                if age > 30.0:
                    decisions.pop(i)
                    break  # Only expire ONE per cycle for smooth rotation
        self._tamiyo.recent_decisions = decisions[:MAX_DECISIONS]

        # Get focused env's reward components for the detail panel
        focused_rewards = RewardComponents()
        if self._focused_env_id in self._envs:
            focused_env = self._envs[self._focused_env_id]
            if isinstance(focused_env.reward_components, RewardComponents):
                focused_rewards = focused_env.reward_components

        # Aggregate mean metrics for EnvOverview Σ row
        accuracies = [e.host_accuracy for e in self._envs.values() if e.accuracy_history]
        rewards = [e.current_reward for e in self._envs.values() if e.reward_history]
        mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        # Aggregate slot states across all environments for TamiyoBrain slot summary
        slot_stage_counts: dict[str, int] = {
            "DORMANT": 0,
            "GERMINATED": 0,
            "TRAINING": 0,
            "BLENDING": 0,
            "HOLDING": 0,
            "FOSSILIZED": 0,
        }
        total_epochs_in_stage = 0
        non_dormant_count = 0

        for env in self._envs.values():
            for slot_id in self._slot_ids:
                seed = env.seeds.get(slot_id)
                if seed is None:
                    slot_stage_counts["DORMANT"] += 1
                else:
                    stage = seed.stage
                    if stage in slot_stage_counts:
                        slot_stage_counts[stage] += 1
                    else:
                        # Handle transition states (PRUNED, EMBARGOED, RESETTING) as DORMANT
                        slot_stage_counts["DORMANT"] += 1
                    if stage != "DORMANT":
                        total_epochs_in_stage += seed.epochs_in_stage
                        non_dormant_count += 1

        total_slots = len(self._envs) * len(self._slot_ids)
        active_slots = total_slots - slot_stage_counts["DORMANT"]
        avg_epochs = total_epochs_in_stage / non_dormant_count if non_dormant_count > 0 else 0.0

        return SanctumSnapshot(
            # Run context
            run_id=self._run_id,
            task_name=self._task_name,
            run_config=self._run_config,
            current_episode=self._current_episode,
            current_batch=self._current_batch or self._batches_completed,
            current_epoch=self._current_epoch,
            max_epochs=self._max_epochs,
            runtime_seconds=runtime,
            connected=self._connected,
            staleness_seconds=staleness,
            captured_at=datetime.now(timezone.utc).isoformat(),
            aggregate_mean_accuracy=mean_accuracy,
            aggregate_mean_reward=mean_reward,
            # Slot configuration for dynamic columns
            slot_ids=list(self._slot_ids),
            # Per-env state
            envs=dict(self._envs),
            focused_env_id=self._focused_env_id,
            # Focused env's reward breakdown (for RewardComponents widget)
            rewards=focused_rewards,
            # Tamiyo state
            tamiyo=self._tamiyo,
            # System vitals
            vitals=self._vitals,
            # Event log
            event_log=list(self._event_log),
            # Historical best runs
            best_runs=list(self._best_runs),
            # Rolling mean accuracy history
            mean_accuracy_history=deque(self._mean_accuracy_history, maxlen=50),
            # Batch-level aggregates
            batch_avg_reward=self._batch_avg_reward,
            batch_total_episodes=self._batch_total_episodes,
            # Cumulative counts (never reset, tracks entire training run)
            cumulative_fossilized=self._cumulative_fossilized,
            cumulative_pruned=self._cumulative_pruned,
            cumulative_blueprint_spawns=dict(self._cumulative_blueprint_spawns),
            cumulative_blueprint_fossilized=dict(self._cumulative_blueprint_fossilized),
            cumulative_blueprint_prunes=dict(self._cumulative_blueprint_prunes),
            # Aggregate slot state across all environments
            slot_stage_counts=slot_stage_counts,
            total_slots=total_slots,
            active_slots=active_slots,
            avg_epochs_in_stage=avg_epochs,
        )

    # =========================================================================
    # Event Handlers (matching TUIOutput behavior 1:1)
    # =========================================================================

    def _handle_training_started(self, event: "TelemetryEvent") -> None:
        """Handle TRAINING_STARTED event."""
        if not isinstance(event.data, TrainingStartedPayload):
            _logger.warning(
                "Expected TrainingStartedPayload for TRAINING_STARTED, got %s",
                type(event.data).__name__,
            )
            return

        payload = event.data
        self._run_id = payload.episode_id
        self._task_name = payload.task
        self._max_epochs = payload.max_epochs
        self._connected = True
        self._start_time = time.time()
        self._current_episode = payload.start_episode
        self._current_epoch = 0

        # Capture GPU devices
        all_devices = []
        if payload.policy_device:
            all_devices.append(payload.policy_device)
        for dev in payload.env_devices:
            if dev not in all_devices:
                all_devices.append(dev)
        self._gpu_devices = all_devices

        # Initialize num_envs from event
        self.num_envs = payload.n_envs

        # Capture host params baseline (for growth_ratio calculation)
        self._host_params = payload.host_params
        self._vitals.host_params = self._host_params

        # Capture reward mode for A/B test cohort display
        self._reward_mode = payload.reward_mode

        # Capture hyperparameters for run header display
        self._run_config = RunConfig(
            seed=payload.seed,
            n_episodes=payload.n_episodes,
            lr=payload.lr,
            clip_ratio=payload.clip_ratio,
            entropy_coef=payload.entropy_coef,
            param_budget=payload.param_budget,
            resume_path=payload.resume_path,
            entropy_anneal=payload.entropy_anneal or {},
        )

        # Capture slot configuration from event
        self._slot_ids = list(payload.slot_ids)
        self._slot_ids_locked = True  # Lock - don't add more dynamically

        # Reset and recreate env states
        self._envs.clear()
        for i in range(self.num_envs):
            self._ensure_env(i)

        # Reset Tamiyo state
        self._tamiyo = TamiyoState()

        # Reset best runs for new training session
        self._best_runs = []

    def _handle_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle EPOCH_COMPLETED event (per-env only).

        Only processes per-env EPOCH_COMPLETED events (with explicit env_id).
        Batch-level events now use BATCH_EPOCH_COMPLETED, but we still skip
        any event missing env_id as a defensive measure.
        """
        if not isinstance(event.data, EpochCompletedPayload):
            _logger.warning(
                "Expected EpochCompletedPayload for EPOCH_COMPLETED, got %s",
                type(event.data).__name__,
            )
            return

        payload = event.data

        env_id = payload.env_id
        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Update accuracy
        val_acc = payload.val_accuracy
        val_loss = payload.val_loss
        inner_epoch = payload.inner_epoch

        env.host_loss = val_loss
        env.current_epoch = inner_epoch

        # Update global epoch
        self._current_epoch = inner_epoch

        # Update per-seed telemetry from EPOCH_COMPLETED event
        # Note: seeds is dict[str, dict[str, Any]] - inner dicts remain untyped
        if payload.seeds is None:
            _logger.warning("EPOCH_COMPLETED missing seeds telemetry")
            seeds_data = {}
        else:
            seeds_data = payload.seeds
        for slot_id, seed_telemetry in seeds_data.items():
            # Ensure seed exists
            if slot_id not in env.seeds:
                env.seeds[slot_id] = SeedState(slot_id=slot_id)
            seed = env.seeds[slot_id]

            # Update from telemetry (using .get() on inner dict is intentional)
            seed.stage = seed_telemetry.get("stage", seed.stage)
            seed.blueprint_id = seed_telemetry.get("blueprint_id", seed.blueprint_id)
            seed.accuracy_delta = seed_telemetry.get("accuracy_delta", seed.accuracy_delta)
            seed.epochs_in_stage = seed_telemetry.get("epochs_in_stage", seed.epochs_in_stage)
            seed.alpha = seed_telemetry.get("alpha", seed.alpha)
            seed.grad_ratio = seed_telemetry.get("grad_ratio", seed.grad_ratio)
            seed.has_vanishing = seed_telemetry.get("has_vanishing", seed.has_vanishing)
            seed.has_exploding = seed_telemetry.get("has_exploding", seed.has_exploding)

            # Track slot_ids dynamically (only if not locked by TRAINING_STARTED)
            if not self._slot_ids_locked and slot_id not in self._slot_ids and slot_id != "unknown":
                self._slot_ids.append(slot_id)
                self._slot_ids.sort()

        # Update accuracy AFTER seeds are refreshed
        episode_id = self._current_episode + env_id
        env.add_accuracy(val_acc, inner_epoch, episode=episode_id)

        env.last_update = datetime.now(timezone.utc)

        # Update rolling mean accuracy across all envs
        accuracies = [e.host_accuracy for e in self._envs.values() if e.host_accuracy > 0]
        if accuracies:
            mean_acc = sum(accuracies) / len(accuracies)
            self._mean_accuracy_history.append(mean_acc)

    def _handle_ppo_update(self, event: "TelemetryEvent") -> None:
        """Handle PPO_UPDATE_COMPLETED event."""
        if not isinstance(event.data, PPOUpdatePayload):
            _logger.warning(
                "Expected PPOUpdatePayload for PPO_UPDATE_COMPLETED, got %s",
                type(event.data).__name__,
            )
            return

        payload = event.data

        if payload.skipped:
            return

        # Mark that we've received PPO data (enables TamiyoBrain display)
        self._tamiyo.ppo_data_received = True

        # A/B testing group identification
        group_id = event.group_id
        if group_id and group_id != "default":
            self._tamiyo.group_id = group_id

        # Update Tamiyo state with all PPO metrics AND append to history
        policy_loss = payload.policy_loss
        self._tamiyo.policy_loss = policy_loss
        self._tamiyo.policy_loss_history.append(policy_loss)

        value_loss = payload.value_loss
        self._tamiyo.value_loss = value_loss
        self._tamiyo.value_loss_history.append(value_loss)

        entropy = payload.entropy
        self._tamiyo.entropy = entropy
        self._tamiyo.entropy_history.append(entropy)

        # Optional field with None - use explicit check to distinguish None from 0.0
        explained_variance = (
            payload.explained_variance if payload.explained_variance is not None else 0.0
        )
        self._tamiyo.explained_variance = explained_variance
        self._tamiyo.explained_variance_history.append(explained_variance)

        grad_norm = payload.grad_norm
        self._tamiyo.grad_norm = grad_norm
        self._tamiyo.grad_norm_history.append(grad_norm)

        kl_divergence = payload.kl_divergence
        self._tamiyo.kl_divergence = kl_divergence
        self._tamiyo.kl_divergence_history.append(kl_divergence)

        clip_fraction = payload.clip_fraction
        self._tamiyo.clip_fraction = clip_fraction
        self._tamiyo.clip_fraction_history.append(clip_fraction)

        # Other losses (no history tracking) - has default
        self._tamiyo.entropy_loss = payload.entropy_loss

        # Advantage stats - have defaults
        self._tamiyo.advantage_mean = payload.advantage_mean
        self._tamiyo.advantage_std = payload.advantage_std

        # Ratio statistics (PPO importance sampling ratios) - have defaults
        self._tamiyo.ratio_mean = payload.ratio_mean
        self._tamiyo.ratio_min = payload.ratio_min
        self._tamiyo.ratio_max = payload.ratio_max
        self._tamiyo.ratio_std = payload.ratio_std

        # Learning rate and entropy coefficient - optional with None
        if payload.lr is not None:
            self._tamiyo.learning_rate = payload.lr

        if payload.entropy_coef is not None:
            self._tamiyo.entropy_coef = payload.entropy_coef

        # Gradient health - have defaults
        self._tamiyo.dead_layers = payload.dead_layers
        self._tamiyo.exploding_layers = payload.exploding_layers
        self._tamiyo.nan_grad_count = payload.nan_grad_count
        if payload.layer_gradient_health is not None:
            self._tamiyo.layer_gradient_health = payload.layer_gradient_health
        self._tamiyo.entropy_collapsed = payload.entropy_collapsed

        # Performance timing - have defaults
        self._tamiyo.update_time_ms = payload.update_time_ms
        self._tamiyo.early_stop_epoch = payload.early_stop_epoch

        # Per-head entropy and gradient norms - optional with None
        if payload.head_slot_entropy is not None:
            self._tamiyo.head_slot_entropy = payload.head_slot_entropy
        if payload.head_slot_grad_norm is not None:
            self._tamiyo.head_slot_grad_norm = payload.head_slot_grad_norm
        if payload.head_blueprint_entropy is not None:
            self._tamiyo.head_blueprint_entropy = payload.head_blueprint_entropy
        if payload.head_blueprint_grad_norm is not None:
            self._tamiyo.head_blueprint_grad_norm = payload.head_blueprint_grad_norm

        # Per-head entropies (for heatmap visualization) - optional with None
        if payload.head_style_entropy is not None:
            self._tamiyo.head_style_entropy = payload.head_style_entropy
        if payload.head_tempo_entropy is not None:
            self._tamiyo.head_tempo_entropy = payload.head_tempo_entropy
        if payload.head_alpha_target_entropy is not None:
            self._tamiyo.head_alpha_target_entropy = payload.head_alpha_target_entropy
        if payload.head_alpha_speed_entropy is not None:
            self._tamiyo.head_alpha_speed_entropy = payload.head_alpha_speed_entropy
        if payload.head_alpha_curve_entropy is not None:
            self._tamiyo.head_alpha_curve_entropy = payload.head_alpha_curve_entropy
        if payload.head_op_entropy is not None:
            self._tamiyo.head_op_entropy = payload.head_op_entropy

        # PPO inner loop context - have defaults
        self._tamiyo.inner_epoch = payload.inner_epoch
        self._tamiyo.ppo_batch = payload.batch

    def _handle_reward_computed(self, event: "TelemetryEvent") -> None:
        """Handle REWARD_COMPUTED event with per-env routing."""
        if not isinstance(event.data, RewardComputedPayload):
            _logger.warning(
                "Expected RewardComputedPayload for REWARD_COMPUTED, got %s",
                type(event.data).__name__,
            )
            return

        payload = event.data
        env_id = payload.env_id
        epoch = event.epoch or 0

        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Update reward tracking
        total_reward = payload.total_reward
        env.reward_history.append(total_reward)
        env.current_epoch = epoch

        # Update action tracking (with normalization)
        action_name = normalize_action(payload.action_name)
        env.action_history.append(action_name)
        env.action_counts[action_name] = env.action_counts.get(action_name, 0) + 1
        env.total_actions += 1

        # Capture A/B test cohort
        if payload.ab_group:
            env.reward_mode = payload.ab_group

        # Store reward component breakdown - provide defaults for None values
        env.reward_components = RewardComponents(
            base_acc_delta=payload.base_acc_delta if payload.base_acc_delta is not None else 0.0,
            bounded_attribution=payload.bounded_attribution if payload.bounded_attribution is not None else 0.0,
            seed_contribution=payload.seed_contribution if payload.seed_contribution is not None else 0.0,
            compute_rent=payload.compute_rent if payload.compute_rent is not None else 0.0,
            alpha_shock=payload.alpha_shock if payload.alpha_shock is not None else 0.0,
            ratio_penalty=payload.ratio_penalty if payload.ratio_penalty is not None else 0.0,
            stage_bonus=payload.stage_bonus if payload.stage_bonus is not None else 0.0,
            fossilize_terminal_bonus=payload.fossilize_terminal_bonus if payload.fossilize_terminal_bonus is not None else 0.0,
            blending_warning=payload.blending_warning if payload.blending_warning is not None else 0.0,
            holding_warning=payload.holding_warning if payload.holding_warning is not None else 0.0,
            val_acc=payload.val_acc if payload.val_acc is not None else env.host_accuracy,
            total=total_reward,
            last_action=action_name,
            env_id=env_id,
        )

        # Capture decision snapshot
        now_dt = event.timestamp or datetime.now(timezone.utc)
        value_s = payload.value_estimate

        # Compute TD advantage for previous decision from this env
        if env_id in self._pending_decisions:
            pending = self._pending_decisions[env_id]
            td_adv = pending.reward + DEFAULT_GAMMA * value_s - pending.value_s
            pending.decision.td_advantage = td_adv
            del self._pending_decisions[env_id]

        # Convert slot_states from dict[str, dict[str, Any]] to dict[str, str] for display
        slot_states_display: dict[str, str] = {}
        if payload.slot_states:
            for slot_id, state_dict in payload.slot_states.items():
                # Format: "Training 12%" or "Empty" etc.
                slot_states_display[slot_id] = str(state_dict)  # Simple string conversion for now

        # Alternatives are already list[tuple[str, float]] from payload
        alternatives_list: list[tuple[str, float]] = list(payload.alternatives) if payload.alternatives else []

        decision = DecisionSnapshot(
            timestamp=now_dt,
            slot_states=slot_states_display,
            host_accuracy=payload.host_accuracy or env.host_accuracy,
            chosen_action=action_name,
            chosen_slot=payload.action_slot,
            confidence=payload.action_confidence,
            expected_value=value_s,
            actual_reward=total_reward,
            alternatives=alternatives_list,
            decision_id=str(uuid.uuid4())[:8],
            decision_entropy=payload.decision_entropy if payload.decision_entropy is not None else 0.0,
            env_id=env_id,
            value_residual=total_reward - value_s,
            td_advantage=None,
        )

        # Store as pending for TD advantage computation on next decision
        self._pending_decisions[env_id] = PendingDecision(
            decision=decision,
            reward=total_reward,
            value_s=value_s,
            env_id=env_id,
        )

        # Add decision if room available
        decisions = self._tamiyo.recent_decisions
        if len(decisions) < MAX_DECISIONS:
            decisions.insert(0, decision)
            self._tamiyo.recent_decisions = decisions

        # Track focused env for reward panel
        self._focused_env_id = env_id

        env.last_update = datetime.now(timezone.utc)

    def _handle_seed_event(self, event: "TelemetryEvent", event_type: str) -> None:
        """Handle seed lifecycle events with per-env tracking."""
        # Type-safe payload access with type switching based on event_type
        if event_type == "SEED_GERMINATED" and isinstance(event.data, SeedGerminatedPayload):
            germinated_payload = event.data
            slot_id = event.slot_id or germinated_payload.slot_id
            env_id = germinated_payload.env_id

            self._ensure_env(env_id)
            env = self._envs[env_id]

            # Dynamically track slot_ids
            if not self._slot_ids_locked and slot_id and slot_id not in self._slot_ids and slot_id != "unknown":
                self._slot_ids.append(slot_id)
                self._slot_ids.sort()

            # Get or create seed state
            if slot_id not in env.seeds:
                env.seeds[slot_id] = SeedState(slot_id=slot_id)
            seed = env.seeds[slot_id]

            # Update from payload - all have defaults
            seed.stage = "GERMINATED"
            seed.blueprint_id = germinated_payload.blueprint_id
            seed.seed_params = germinated_payload.params
            seed.grad_ratio = germinated_payload.grad_ratio
            seed.has_vanishing = germinated_payload.has_vanishing
            seed.has_exploding = germinated_payload.has_exploding
            seed.epochs_in_stage = germinated_payload.epochs_in_stage
            seed.blend_tempo_epochs = germinated_payload.blend_tempo_epochs
            seed.alpha = germinated_payload.alpha
            env.active_seed_count += 1

            # Track blueprint spawn for graveyard
            env.blueprint_spawns[seed.blueprint_id] = (
                env.blueprint_spawns.get(seed.blueprint_id, 0) + 1
            )
            self._cumulative_blueprint_spawns[seed.blueprint_id] = (
                self._cumulative_blueprint_spawns.get(seed.blueprint_id, 0) + 1
            )

        elif event_type == "SEED_STAGE_CHANGED" and isinstance(event.data, SeedStageChangedPayload):
            stage_changed_payload = event.data
            slot_id = event.slot_id or stage_changed_payload.slot_id
            env_id = stage_changed_payload.env_id

            self._ensure_env(env_id)
            env = self._envs[env_id]

            # Dynamically track slot_ids
            if not self._slot_ids_locked and slot_id and slot_id not in self._slot_ids and slot_id != "unknown":
                self._slot_ids.append(slot_id)
                self._slot_ids.sort()

            # Get or create seed state
            if slot_id not in env.seeds:
                env.seeds[slot_id] = SeedState(slot_id=slot_id)
            seed = env.seeds[slot_id]

            # Update from payload - note: from_stage and to_stage field names
            seed.stage = stage_changed_payload.to_stage
            seed.grad_ratio = stage_changed_payload.grad_ratio
            seed.has_vanishing = stage_changed_payload.has_vanishing
            seed.has_exploding = stage_changed_payload.has_exploding
            seed.epochs_in_stage = stage_changed_payload.epochs_in_stage
            seed.accuracy_delta = stage_changed_payload.accuracy_delta
            if stage_changed_payload.alpha is not None:
                seed.alpha = stage_changed_payload.alpha

        elif event_type == "SEED_FOSSILIZED" and isinstance(event.data, SeedFossilizedPayload):
            fossilized_payload = event.data
            slot_id = event.slot_id or fossilized_payload.slot_id
            env_id = fossilized_payload.env_id

            self._ensure_env(env_id)
            env = self._envs[env_id]

            # Dynamically track slot_ids
            if not self._slot_ids_locked and slot_id and slot_id not in self._slot_ids and slot_id != "unknown":
                self._slot_ids.append(slot_id)
                self._slot_ids.sort()

            # Get or create seed state
            if slot_id not in env.seeds:
                env.seeds[slot_id] = SeedState(slot_id=slot_id)
            seed = env.seeds[slot_id]

            # Update from payload
            seed.stage = "FOSSILIZED"
            seed.improvement = fossilized_payload.improvement
            seed.blueprint_id = fossilized_payload.blueprint_id
            seed.epochs_total = fossilized_payload.epochs_total
            seed.counterfactual = fossilized_payload.counterfactual if fossilized_payload.counterfactual is not None else 0.0
            seed.alpha = fossilized_payload.alpha
            env.fossilized_params += fossilized_payload.params_added
            env.fossilized_count += 1
            self._cumulative_fossilized += 1
            env.active_seed_count = max(0, env.active_seed_count - 1)

            # Track blueprint fossilization for graveyard
            env.blueprint_fossilized[seed.blueprint_id] = (
                env.blueprint_fossilized.get(seed.blueprint_id, 0) + 1
            )
            self._cumulative_blueprint_fossilized[seed.blueprint_id] = (
                self._cumulative_blueprint_fossilized.get(seed.blueprint_id, 0) + 1
            )

        elif event_type == "SEED_PRUNED" and isinstance(event.data, SeedPrunedPayload):
            pruned_payload = event.data
            slot_id = event.slot_id or pruned_payload.slot_id
            env_id = pruned_payload.env_id

            self._ensure_env(env_id)
            env = self._envs[env_id]

            # Dynamically track slot_ids
            if not self._slot_ids_locked and slot_id and slot_id not in self._slot_ids and slot_id != "unknown":
                self._slot_ids.append(slot_id)
                self._slot_ids.sort()

            # Get or create seed state
            if slot_id not in env.seeds:
                env.seeds[slot_id] = SeedState(slot_id=slot_id)
            seed = env.seeds[slot_id]

            # Update from payload
            seed.prune_reason = pruned_payload.reason
            seed.improvement = pruned_payload.improvement
            seed.auto_pruned = pruned_payload.auto_pruned
            seed.epochs_total = pruned_payload.epochs_total
            seed.counterfactual = pruned_payload.counterfactual if pruned_payload.counterfactual is not None else 0.0
            pruned_blueprint = pruned_payload.blueprint_id or seed.blueprint_id
            seed.blueprint_id = pruned_blueprint

            # Track blueprint prune for graveyard
            if pruned_blueprint:
                env.blueprint_prunes[pruned_blueprint] = (
                    env.blueprint_prunes.get(pruned_blueprint, 0) + 1
                )
                self._cumulative_blueprint_prunes[pruned_blueprint] = (
                    self._cumulative_blueprint_prunes.get(pruned_blueprint, 0) + 1
                )

            # Mark PRUNED
            seed.stage = "PRUNED"
            seed.seed_params = 0
            seed.alpha = 0.0
            seed.accuracy_delta = 0.0
            seed.grad_ratio = 0.0
            seed.has_vanishing = False
            seed.has_exploding = False
            seed.epochs_in_stage = 0
            env.pruned_count += 1
            self._cumulative_pruned += 1
            env.active_seed_count = max(0, env.active_seed_count - 1)

    def _handle_batch_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle BATCH_EPOCH_COMPLETED event (episode completion)."""
        if not isinstance(event.data, BatchEpochCompletedPayload):
            _logger.warning(
                "Expected BatchEpochCompletedPayload for BATCH_EPOCH_COMPLETED, got %s",
                type(event.data).__name__,
            )
            return

        payload = event.data

        self._current_episode = payload.episodes_completed
        self._batches_completed += 1

        # Capture batch-level aggregates - all required fields
        self._current_batch = payload.batch_idx
        self._batch_avg_accuracy = payload.avg_accuracy
        self._batch_rolling_accuracy = payload.rolling_accuracy
        self._batch_avg_reward = payload.avg_reward
        self._batch_total_episodes = payload.total_episodes

        # Episode Return
        self._tamiyo.current_episode_return = payload.avg_reward
        self._tamiyo.episode_return_history.append(payload.avg_reward)

        # Calculate throughput
        now = time.time()
        elapsed = now - self._start_time
        if elapsed > 0:
            total_epochs = self._current_episode * self._max_epochs
            self._vitals.epochs_per_second = total_epochs / elapsed
            self._vitals.batches_per_hour = (self._batches_completed / elapsed) * 3600

        # Capture best_runs
        n_envs = payload.n_envs
        episode_start = self._current_episode - n_envs
        for env in self._envs.values():
            if env.best_accuracy > 0:
                base_params = env.host_params
                seed_params_total = sum(
                    int(seed.seed_params or 0) for seed in env.best_seeds.values()
                )
                growth_ratio = (
                    (base_params + seed_params_total) / base_params
                    if base_params > 0
                    else 1.0
                )
                # Create full snapshot for historical detail view
                record = BestRunRecord(
                    env_id=env.env_id,
                    episode=episode_start + env.env_id,
                    peak_accuracy=env.best_accuracy,
                    final_accuracy=env.host_accuracy,
                    epoch=env.best_accuracy_epoch,
                    seeds={k: SeedState(**v.__dict__) for k, v in env.best_seeds.items()},
                    slot_ids=list(self._slot_ids),  # All slots for showing DORMANT in detail
                    growth_ratio=growth_ratio,
                    # Interactive features
                    record_id=str(uuid.uuid4())[:8],
                    pinned=False,
                    # Full env snapshot at peak
                    reward_components=RewardComponents(**env.reward_components.__dict__)
                        if env.reward_components else None,
                    counterfactual_matrix=CounterfactualSnapshot(
                        slot_ids=env.counterfactual_matrix.slot_ids,
                        configs=list(env.counterfactual_matrix.configs),
                        strategy=env.counterfactual_matrix.strategy,
                        compute_time_ms=env.counterfactual_matrix.compute_time_ms,
                    ) if env.counterfactual_matrix and env.counterfactual_matrix.slot_ids else None,
                    action_history=list(env.action_history),
                    reward_history=list(env.reward_history),
                    accuracy_history=list(env.accuracy_history),
                    host_loss=env.host_loss,
                    host_params=env.host_params,
                    fossilized_count=env.fossilized_count,
                    pruned_count=env.pruned_count,
                    reward_mode=env.reward_mode,
                    # Seed graveyard: per-blueprint lifecycle stats
                    blueprint_spawns=dict(env.blueprint_spawns),
                    blueprint_fossilized=dict(env.blueprint_fossilized),
                    blueprint_prunes=dict(env.blueprint_prunes),
                )
                self._best_runs.append(record)

        # Sort by peak accuracy descending
        self._best_runs.sort(key=lambda r: r.peak_accuracy, reverse=True)
        # Keep pinned records + top 10 unpinned
        pinned = [r for r in self._best_runs if r.pinned]
        unpinned = [r for r in self._best_runs if not r.pinned][:10]
        self._best_runs = sorted(pinned + unpinned, key=lambda r: r.peak_accuracy, reverse=True)

        # Reset per-env state for next episode
        # ALL episode-scoped fields must reset here
        for env in self._envs.values():
            # Seed state
            env.seeds.clear()
            env.active_seed_count = 0
            env.fossilized_count = 0
            env.pruned_count = 0
            env.fossilized_params = 0

            # Epoch/progress tracking
            env.current_epoch = 0
            env.host_accuracy = 0.0
            env.epochs_since_improvement = 0
            env.status = "initializing"

            # History (fresh sparklines each episode)
            env.reward_history.clear()
            env.accuracy_history.clear()

            # Best tracking (fresh per episode)
            env.best_reward = float('-inf')
            env.best_reward_epoch = 0
            env.best_accuracy = 0.0
            env.best_accuracy_epoch = 0
            env.best_accuracy_episode = 0
            env.best_seeds.clear()

            # Action tracking (fresh distribution each episode)
            env.action_history.clear()
            env.action_counts = {
                "WAIT": 0,
                "GERMINATE": 0,
                "SET_ALPHA_TARGET": 0,
                "PRUNE": 0,
                "FOSSILIZE": 0,
            }
            env.total_actions = 0

            # Reward components (stale from last step)
            env.reward_components = RewardComponents()

            # Counterfactual matrix (stale from previous episode)
            env.counterfactual_matrix = CounterfactualSnapshot()

            # Graveyard stats (per-episode)
            env.blueprint_spawns.clear()
            env.blueprint_prunes.clear()
            env.blueprint_fossilized.clear()

    def _handle_counterfactual_matrix(self, event: "TelemetryEvent") -> None:
        """Handle COUNTERFACTUAL_MATRIX_COMPUTED event."""
        if not isinstance(event.data, CounterfactualMatrixPayload):
            _logger.warning(
                "Expected CounterfactualMatrixPayload for COUNTERFACTUAL_MATRIX_COMPUTED, got %s",
                type(event.data).__name__,
            )
            return

        payload = event.data
        env_id = payload.env_id

        env = self._envs.get(env_id)
        if env is None:
            return

        # Parse configs
        configs = [
            CounterfactualConfig(
                seed_mask=tuple(cfg.get("seed_mask", [])),
                accuracy=cfg.get("accuracy", 0.0),
            )
            for cfg in payload.configs
        ]

        env.counterfactual_matrix = CounterfactualSnapshot(
            slot_ids=payload.slot_ids,
            configs=configs,
            strategy=payload.strategy,
            compute_time_ms=payload.compute_time_ms,
        )

    def _handle_analytics_snapshot(self, event: "TelemetryEvent") -> None:
        """Handle ANALYTICS_SNAPSHOT events used by Sanctum/Tamiyo UI."""
        if not isinstance(event.data, AnalyticsSnapshotPayload):
            _logger.warning(
                "Expected AnalyticsSnapshotPayload for ANALYTICS_SNAPSHOT, got %s",
                type(event.data).__name__,
            )
            return

        payload = event.data
        kind = payload.kind

        # Batch-level action distribution
        if kind == "action_distribution":
            if payload.action_counts:
                counts = {str(action): int(count) for action, count in payload.action_counts.items()}
                self._tamiyo.action_counts = counts
                self._tamiyo.total_actions = sum(counts.values())
            return

        # Per-step "last_action" snapshots - treat like REWARD_COMPUTED
        if kind == "last_action" and payload.action_confidence is not None:
            self._handle_reward_computed(event)
            return

    # =========================================================================
    # Helpers
    # =========================================================================

    def _ensure_env(self, env_id: int) -> None:
        """Ensure EnvState exists for env_id."""
        if env_id not in self._envs:
            self._envs[env_id] = EnvState(
                env_id=env_id,
                host_params=self._host_params,  # Set baseline for growth_ratio
                reward_mode=self._reward_mode,  # A/B test cohort for colored pip
                accuracy_history=deque(maxlen=self.max_history),
                reward_history=deque(maxlen=self.max_history),
                action_history=deque(maxlen=self.max_history),
            )

    def _add_event_log(self, event: "TelemetryEvent", event_type: str) -> None:
        """Add event to log with generic message and structured metadata.

        Messages are kept generic for proper rollup grouping.
        Specific values (slot_id, reward, blueprint) go in metadata.
        """
        timestamp = event.timestamp or datetime.now(timezone.utc)

        # Calculate relative time
        now = datetime.now(timezone.utc)
        age_seconds = (now - timestamp).total_seconds()
        if age_seconds < 60:
            relative_time = f"({age_seconds:.0f}s)"
        elif age_seconds < 3600:
            relative_time = f"({age_seconds/60:.0f}m)"
        else:
            relative_time = f"({age_seconds/3600:.0f}h)"

        # Generic message + structured metadata based on event type
        metadata: dict[str, str | int | float] = {}
        env_id: int | None = None

        if event_type == "REWARD_COMPUTED":
            if isinstance(event.data, RewardComputedPayload):
                message = "Reward computed"
                metadata["action"] = normalize_action(event.data.action_name)
                metadata["reward"] = event.data.total_reward
                env_id = event.data.env_id
            else:
                message = "Reward computed (unknown payload)"
        elif event_type.startswith("SEED_"):
            if event_type == "SEED_GERMINATED" and isinstance(event.data, SeedGerminatedPayload):
                message = "Germinated"
                slot_id = event.slot_id or event.data.slot_id
                metadata["slot_id"] = slot_id
                metadata["blueprint"] = event.data.blueprint_id
                env_id = event.data.env_id
            elif event_type == "SEED_STAGE_CHANGED" and isinstance(event.data, SeedStageChangedPayload):
                message = "Stage changed"
                slot_id = event.slot_id or event.data.slot_id
                metadata["slot_id"] = slot_id
                metadata["from"] = event.data.from_stage
                metadata["to"] = event.data.to_stage
                env_id = event.data.env_id
            elif event_type == "SEED_FOSSILIZED" and isinstance(event.data, SeedFossilizedPayload):
                message = "Fossilized"
                slot_id = event.slot_id or event.data.slot_id
                metadata["slot_id"] = slot_id
                metadata["improvement"] = event.data.improvement
                env_id = event.data.env_id
            elif event_type == "SEED_PRUNED" and isinstance(event.data, SeedPrunedPayload):
                message = "Pruned"
                slot_id = event.slot_id or event.data.slot_id
                metadata["slot_id"] = slot_id
                metadata["reason"] = event.data.reason
                env_id = event.data.env_id
            else:
                message = event_type.replace("SEED_", "")
        elif event_type == "PPO_UPDATE_COMPLETED":
            if isinstance(event.data, PPOUpdatePayload):
                if event.data.skipped:
                    message = "PPO skipped"
                    metadata["reason"] = "buffer rollback"
                else:
                    message = "PPO update"
                    metadata["entropy"] = event.data.entropy
                    metadata["clip_fraction"] = event.data.clip_fraction
            else:
                message = "PPO update (unknown payload)"
        elif event_type == "BATCH_EPOCH_COMPLETED":
            if isinstance(event.data, BatchEpochCompletedPayload):
                message = "Batch complete"
                metadata["batch"] = event.data.batch_idx
                metadata["episodes"] = event.data.episodes_completed
            else:
                message = "Batch complete (unknown payload)"
        else:
            message = event.message or event_type

        self._event_log.append(EventLogEntry(
            timestamp=timestamp.strftime("%H:%M:%S"),
            event_type=event_type,
            env_id=env_id,
            message=message,
            episode=self._current_episode,
            relative_time=relative_time,
            metadata=metadata,
        ))

    def _update_system_vitals(self) -> None:
        """Update system vitals (CPU, RAM, GPU)."""
        # CPU (THE FIX - was collected but never displayed)
        try:
            self._vitals.cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            pass

        # RAM
        try:
            mem = psutil.virtual_memory()
            self._vitals.ram_used_gb = mem.used / (1024**3)
            self._vitals.ram_total_gb = mem.total / (1024**3)
        except Exception:
            pass

        # GPU stats (multi-GPU support)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_stats: dict[int, GPUStats] = {}
                for i, device in enumerate(self._gpu_devices):
                    if device.startswith("cuda"):
                        device_idx = int(device.split(":")[-1]) if ":" in device else i
                        try:
                            # Use reserved memory (allocator footprint) for OOM risk visibility.
                            mem_reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
                            props = torch.cuda.get_device_properties(device_idx)
                            mem_total = props.total_memory / (1024**3)

                            gpu_stats[device_idx] = GPUStats(
                                device_id=device_idx,
                                memory_used_gb=mem_reserved,
                                memory_total_gb=mem_total,
                                # Only set when actual utilization is available (e.g., via NVML).
                                utilization=0.0,
                            )
                        except Exception:
                            pass
                self._vitals.gpu_stats = gpu_stats
                if 0 in gpu_stats:
                    stats0 = gpu_stats[0]
                    self._vitals.gpu_memory_used_gb = stats0.memory_used_gb
                    self._vitals.gpu_memory_total_gb = stats0.memory_total_gb
                    self._vitals.gpu_utilization = stats0.utilization
                    self._vitals.gpu_temperature = stats0.temperature
        except ImportError:
            pass
