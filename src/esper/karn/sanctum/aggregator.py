"""Sanctum Telemetry Aggregator - Transforms event stream into SanctumSnapshot.

Maintains stateful accumulation of telemetry events to build
real-time SanctumSnapshot objects for the Sanctum TUI.

Thread-safe: Uses threading.Lock to protect state during concurrent
access from training thread (process_event) and UI thread (get_snapshot).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import psutil

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

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent


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

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
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
        )

    # =========================================================================
    # Event Handlers (matching TUIOutput behavior 1:1)
    # =========================================================================

    def _handle_training_started(self, event: "TelemetryEvent") -> None:
        """Handle TRAINING_STARTED event."""
        data = event.data or {}
        self._run_id = data.get("episode_id", "")
        self._task_name = data.get("task", "")
        self._max_epochs = data.get("max_epochs", 75)
        self._connected = True
        self._start_time = time.time()
        start_episode = data.get("start_episode", 0)
        self._current_episode = int(start_episode) if isinstance(start_episode, (int, float)) else 0
        self._current_epoch = 0

        # Capture GPU devices
        env_devices = data.get("env_devices", [])
        policy_device = data.get("policy_device", "")
        all_devices = []
        if policy_device:
            all_devices.append(policy_device)
        for dev in env_devices:
            if dev not in all_devices:
                all_devices.append(dev)
        self._gpu_devices = all_devices

        # Initialize num_envs from event
        n_envs = data.get("n_envs", self.num_envs)
        self.num_envs = n_envs

        # Capture host params baseline (for growth_ratio calculation)
        self._host_params = data.get("host_params", 0)
        self._vitals.host_params = self._host_params

        # Capture reward mode for A/B test cohort display
        self._reward_mode = data.get("reward_mode", "")

        # Capture hyperparameters for run header display
        self._run_config = RunConfig(
            seed=data.get("seed"),
            n_episodes=data.get("n_episodes", 0),
            lr=data.get("lr", 0.0),
            clip_ratio=data.get("clip_ratio", 0.2),
            entropy_coef=data.get("entropy_coef", 0.01),
            param_budget=data.get("param_budget", 0),
            resume_path=data.get("resume_path", ""),
            entropy_anneal=data.get("entropy_anneal") or {},
        )

        # Capture slot configuration from event (if provided)
        # Falls back to default 2x2 grid if not specified
        slot_ids = data.get("slot_ids")
        if slot_ids and isinstance(slot_ids, (list, tuple)):
            self._slot_ids = list(slot_ids)
            self._slot_ids_locked = True  # Lock - don't add more dynamically
        else:
            # Default to 2x2 grid (r0c0, r0c1, r1c0, r1c1)
            from esper.leyline.slot_id import format_slot_id
            self._slot_ids = [format_slot_id(r, c) for r in range(2) for c in range(2)]
            self._slot_ids_locked = False  # Allow dynamic discovery

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
        data = event.data or {}

        # Belt-and-suspenders: skip events without env_id
        # Batch-level events now use BATCH_EPOCH_COMPLETED, but if any legacy
        # EPOCH_COMPLETED without env_id arrives, don't corrupt env 0's tracking
        if "env_id" not in data:
            return

        env_id = data["env_id"]

        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Update accuracy
        val_acc = data.get("val_accuracy", 0.0)
        val_loss = data.get("val_loss", 0.0)
        inner_epoch = data.get("inner_epoch", data.get("epoch", 0))

        env.host_loss = val_loss
        env.current_epoch = inner_epoch

        # Update global epoch
        self._current_epoch = inner_epoch

        # Update per-seed telemetry from EPOCH_COMPLETED event
        # This provides per-tick accuracy_delta updates for all active seeds
        seeds_data = data.get("seeds", {})
        for slot_id, seed_telemetry in seeds_data.items():
            # Ensure seed exists
            if slot_id not in env.seeds:
                env.seeds[slot_id] = SeedState(slot_id=slot_id)
            seed = env.seeds[slot_id]

            # Update from telemetry
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

        # Update accuracy AFTER seeds are refreshed so best_seeds snapshots are accurate.
        # Episode numbering: BATCH_EPOCH_COMPLETED increments episodes_completed by envs_this_batch,
        # so the absolute episode for this env is base + env_id.
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
        data = event.data or {}

        if data.get("skipped"):
            return

        # Mark that we've received PPO data (enables TamiyoBrain display)
        self._tamiyo.ppo_data_received = True

        # A/B testing group identification
        # Filter out "default" - that's the single-policy default value
        group_id = event.group_id
        if group_id and group_id != "default":
            self._tamiyo.group_id = group_id

        # Update Tamiyo state with all PPO metrics AND append to history
        policy_loss = data.get("policy_loss", 0.0)
        self._tamiyo.policy_loss = policy_loss
        self._tamiyo.policy_loss_history.append(policy_loss)

        value_loss = data.get("value_loss", 0.0)
        self._tamiyo.value_loss = value_loss
        self._tamiyo.value_loss_history.append(value_loss)

        entropy = data.get("entropy", 0.0)
        self._tamiyo.entropy = entropy
        self._tamiyo.entropy_history.append(entropy)

        explained_variance = data.get("explained_variance", 0.0)
        self._tamiyo.explained_variance = explained_variance
        self._tamiyo.explained_variance_history.append(explained_variance)

        grad_norm = data.get("grad_norm", 0.0)
        self._tamiyo.grad_norm = grad_norm
        self._tamiyo.grad_norm_history.append(grad_norm)

        kl_divergence = data.get("kl_divergence", 0.0)
        self._tamiyo.kl_divergence = kl_divergence
        self._tamiyo.kl_divergence_history.append(kl_divergence)

        clip_fraction = data.get("clip_fraction", 0.0)
        self._tamiyo.clip_fraction = clip_fraction
        self._tamiyo.clip_fraction_history.append(clip_fraction)

        # Other losses (no history tracking)
        self._tamiyo.entropy_loss = data.get("entropy_loss", 0.0)

        # Advantage stats
        self._tamiyo.advantage_mean = data.get("advantage_mean", 0.0)
        self._tamiyo.advantage_std = data.get("advantage_std", 0.0)

        # Ratio statistics (PPO importance sampling ratios)
        self._tamiyo.ratio_mean = data.get("ratio_mean", 1.0)
        self._tamiyo.ratio_min = data.get("ratio_min", 1.0)
        self._tamiyo.ratio_max = data.get("ratio_max", 1.0)
        self._tamiyo.ratio_std = data.get("ratio_std", 0.0)

        # Learning rate and entropy coefficient
        lr = data.get("lr")
        if lr is not None:
            self._tamiyo.learning_rate = lr

        entropy_coef = data.get("entropy_coef")
        if entropy_coef is not None:
            self._tamiyo.entropy_coef = entropy_coef

        # Gradient health
        self._tamiyo.dead_layers = data.get("dead_layers", 0)
        self._tamiyo.exploding_layers = data.get("exploding_layers", 0)
        self._tamiyo.nan_grad_count = data.get("nan_grad_count", 0)
        layer_health = data.get("layer_gradient_health")
        if layer_health is not None:
            self._tamiyo.layer_gradient_health = layer_health
        self._tamiyo.entropy_collapsed = data.get("entropy_collapsed", False)

        # Performance timing
        self._tamiyo.update_time_ms = data.get("update_time_ms", 0.0)
        self._tamiyo.early_stop_epoch = data.get("early_stop_epoch")

        # Per-head entropy and gradient norms
        self._tamiyo.head_slot_entropy = data.get("head_slot_entropy", self._tamiyo.head_slot_entropy)
        self._tamiyo.head_slot_grad_norm = data.get("head_slot_grad_norm", self._tamiyo.head_slot_grad_norm)
        self._tamiyo.head_blueprint_entropy = data.get("head_blueprint_entropy", self._tamiyo.head_blueprint_entropy)
        self._tamiyo.head_blueprint_grad_norm = data.get("head_blueprint_grad_norm", self._tamiyo.head_blueprint_grad_norm)

        # Per-head entropies (for heatmap visualization)
        # These are optional - only present when neural network emits them
        self._tamiyo.head_style_entropy = data.get("head_style_entropy", self._tamiyo.head_style_entropy)
        self._tamiyo.head_tempo_entropy = data.get("head_tempo_entropy", self._tamiyo.head_tempo_entropy)
        self._tamiyo.head_alpha_target_entropy = data.get("head_alpha_target_entropy", self._tamiyo.head_alpha_target_entropy)
        self._tamiyo.head_alpha_speed_entropy = data.get("head_alpha_speed_entropy", self._tamiyo.head_alpha_speed_entropy)
        self._tamiyo.head_alpha_curve_entropy = data.get("head_alpha_curve_entropy", self._tamiyo.head_alpha_curve_entropy)
        self._tamiyo.head_op_entropy = data.get("head_op_entropy", self._tamiyo.head_op_entropy)

        # PPO inner loop context
        self._tamiyo.inner_epoch = data.get("inner_epoch", 0)
        self._tamiyo.ppo_batch = data.get("batch", 0)

    def _handle_reward_computed(self, event: "TelemetryEvent") -> None:
        """Handle REWARD_COMPUTED event with per-env routing."""
        data = event.data or {}
        env_id = data.get("env_id", 0)
        epoch = event.epoch or 0

        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Update reward tracking
        total_reward = data.get("total_reward", 0.0)
        env.reward_history.append(total_reward)
        env.current_epoch = epoch

        # Update action tracking (with normalization)
        action_name = normalize_action(data.get("action_name", "WAIT"))
        env.action_history.append(action_name)
        env.action_counts[action_name] = env.action_counts.get(action_name, 0) + 1
        env.total_actions += 1

        # Capture A/B test cohort (for color coding in TUI)
        # ab_group is emitted by vectorized training when --ab-test is used
        ab_group = data.get("ab_group")
        if ab_group:
            env.reward_mode = ab_group

        # Store reward component breakdown
        env.reward_components = RewardComponents(
            base_acc_delta=data.get("base_acc_delta", 0.0),
            bounded_attribution=data.get("bounded_attribution", 0.0),
            seed_contribution=data.get("seed_contribution", 0.0),
            compute_rent=data.get("compute_rent", 0.0),
            alpha_shock=data.get("alpha_shock", 0.0),
            ratio_penalty=data.get("ratio_penalty", 0.0),
            stage_bonus=data.get("stage_bonus", 0.0),
            fossilize_terminal_bonus=data.get("fossilize_terminal_bonus", 0.0),
            blending_warning=data.get("blending_warning", 0.0),
            holding_warning=data.get("holding_warning", 0.0),
            val_acc=data.get("val_acc", env.host_accuracy),
            total=total_reward,
            last_action=action_name,
            env_id=env_id,
        )

        # Capture decision snapshot
        # Only capture if we have decision data (action_confidence present)
        if "action_confidence" in data:
            now_dt = event.timestamp or datetime.now(timezone.utc)
            decision = DecisionSnapshot(
                timestamp=now_dt,
                slot_states=data.get("slot_states", {}),
                host_accuracy=data.get("host_accuracy", env.host_accuracy),
                chosen_action=action_name,
                chosen_slot=data.get("action_slot"),
                confidence=data.get("action_confidence", 0.0),
                expected_value=data.get("value_estimate", 0.0),
                actual_reward=total_reward,
                alternatives=data.get("alternatives", []),
                decision_id=str(uuid.uuid4())[:8],  # Short unique ID for pinning
            )

            # Add decision if room available (expiration handled by build_snapshot)
            # Decisions expire after 30s via build_snapshot() carousel logic,
            # creating natural stagger as each expires on its own timestamp.
            decisions = self._tamiyo.recent_decisions
            if len(decisions) < MAX_DECISIONS:
                decisions.insert(0, decision)
                self._tamiyo.recent_decisions = decisions

            # Also keep last_decision for backwards compatibility
            self._tamiyo.last_decision = decision

        # Track focused env for reward panel
        self._focused_env_id = env_id

        env.last_update = datetime.now(timezone.utc)

    def _handle_seed_event(self, event: "TelemetryEvent", event_type: str) -> None:
        """Handle seed lifecycle events with per-env tracking."""
        data = event.data or {}
        slot_id = event.slot_id or data.get("slot_id", "unknown")
        env_id = data.get("env_id", 0)

        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Dynamically track slot_ids (only if not locked by TRAINING_STARTED)
        if not self._slot_ids_locked and slot_id and slot_id not in self._slot_ids and slot_id != "unknown":
            self._slot_ids.append(slot_id)
            # Keep sorted for consistent column order
            self._slot_ids.sort()

        # Get or create seed state
        if slot_id not in env.seeds:
            env.seeds[slot_id] = SeedState(slot_id=slot_id)
        seed = env.seeds[slot_id]

        # Capture alpha from ALL seed events
        if "alpha" in data:
            seed.alpha = data["alpha"]

        if event_type == "SEED_GERMINATED":
            seed.stage = "GERMINATED"
            seed.blueprint_id = data.get("blueprint_id")
            seed.seed_params = data.get("params", seed.seed_params)
            seed.grad_ratio = data.get("grad_ratio", seed.grad_ratio)
            seed.has_vanishing = data.get("has_vanishing", False)
            seed.has_exploding = data.get("has_exploding", False)
            seed.epochs_in_stage = data.get("epochs_in_stage", 0)
            seed.blend_tempo_epochs = data.get("blend_tempo_epochs", 5)
            env.active_seed_count += 1
            # Track blueprint spawn for graveyard (per-episode and cumulative)
            if seed.blueprint_id:
                env.blueprint_spawns[seed.blueprint_id] = (
                    env.blueprint_spawns.get(seed.blueprint_id, 0) + 1
                )
                self._cumulative_blueprint_spawns[seed.blueprint_id] = (
                    self._cumulative_blueprint_spawns.get(seed.blueprint_id, 0) + 1
                )

        elif event_type == "SEED_STAGE_CHANGED":
            seed.stage = data.get("to", seed.stage)
            seed.grad_ratio = data.get("grad_ratio", seed.grad_ratio)
            seed.has_vanishing = data.get("has_vanishing", seed.has_vanishing)
            seed.has_exploding = data.get("has_exploding", seed.has_exploding)
            seed.epochs_in_stage = data.get("epochs_in_stage", 0)
            # Capture accuracy_delta from enhanced telemetry
            if "accuracy_delta" in data:
                seed.accuracy_delta = data["accuracy_delta"]

        elif event_type == "SEED_FOSSILIZED":
            seed.stage = "FOSSILIZED"
            # Capture fossilization context (P1/P2 telemetry gap fix)
            seed.improvement = data.get("improvement", 0.0)
            seed.blueprint_id = data.get("blueprint_id") or seed.blueprint_id
            seed.epochs_total = data.get("epochs_total", 0)
            seed.counterfactual = data.get("counterfactual", 0.0)
            # Preserve blend_tempo_epochs for fossilized display (already set at germination)
            env.fossilized_params += int(data.get("params_added", 0) or 0)
            env.fossilized_count += 1
            self._cumulative_fossilized += 1  # Never resets
            env.active_seed_count = max(0, env.active_seed_count - 1)
            # Track blueprint fossilization for graveyard (per-episode and cumulative)
            if seed.blueprint_id:
                env.blueprint_fossilized[seed.blueprint_id] = (
                    env.blueprint_fossilized.get(seed.blueprint_id, 0) + 1
                )
                self._cumulative_blueprint_fossilized[seed.blueprint_id] = (
                    self._cumulative_blueprint_fossilized.get(seed.blueprint_id, 0) + 1
                )

        elif event_type == "SEED_PRUNED":
            # Capture prune context (P1/P2 telemetry gap fix).
            #
            # Phase 4 contract: PRUNED/EMBARGOED/RESETTING are first-class states
            # after physical removal (seed module is gone), so we must NOT reset
            # the slot to DORMANT here. Kasmina will emit subsequent
            # SEED_STAGE_CHANGED events through the cooldown pipeline and finally
            # return to DORMANT when the slot is reusable.
            seed.prune_reason = data.get("reason", "")
            seed.improvement = data.get("improvement", 0.0)
            seed.auto_pruned = data.get("auto_pruned", False)
            seed.epochs_total = data.get("epochs_total", 0)
            seed.counterfactual = data.get("counterfactual", 0.0)
            pruned_blueprint = data.get("blueprint_id") or seed.blueprint_id
            seed.blueprint_id = pruned_blueprint

            # Track blueprint prune for graveyard (per-episode and cumulative).
            if pruned_blueprint:
                env.blueprint_prunes[pruned_blueprint] = (
                    env.blueprint_prunes.get(pruned_blueprint, 0) + 1
                )
                self._cumulative_blueprint_prunes[pruned_blueprint] = (
                    self._cumulative_blueprint_prunes.get(pruned_blueprint, 0) + 1
                )

            # Mark PRUNED (seed physically removed, slot unavailable until cooldown completes).
            seed.stage = "PRUNED"
            seed.seed_params = 0
            seed.alpha = 0.0
            seed.accuracy_delta = 0.0
            seed.grad_ratio = 0.0
            seed.has_vanishing = False
            seed.has_exploding = False
            seed.epochs_in_stage = 0
            env.pruned_count += 1
            self._cumulative_pruned += 1  # Never resets
            env.active_seed_count = max(0, env.active_seed_count - 1)

    def _handle_batch_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle BATCH_EPOCH_COMPLETED event (episode completion)."""
        data = event.data or {}

        episodes_completed = data.get("episodes_completed")
        if isinstance(episodes_completed, (int, float)):
            self._current_episode = int(episodes_completed)
        self._batches_completed += 1

        # Capture batch-level aggregates (P1 telemetry gap fix)
        self._current_batch = data.get("batch_idx", self._batches_completed)
        self._batch_avg_accuracy = data.get("avg_accuracy", 0.0)
        self._batch_rolling_accuracy = data.get("rolling_accuracy", 0.0)
        self._batch_avg_reward = data.get("avg_reward", 0.0)
        self._batch_total_episodes = data.get("total_episodes", 0)

        # Episode Return (PRIMARY RL METRIC per DRL review)
        # avg_reward from BATCH_EPOCH_COMPLETED is the average episode return
        episode_return = data.get("avg_reward", 0.0)
        if episode_return != 0.0:
            self._tamiyo.current_episode_return = episode_return
            self._tamiyo.episode_return_history.append(episode_return)

        # Calculate throughput
        now = time.time()
        elapsed = now - self._start_time
        if elapsed > 0:
            total_epochs = self._current_episode * self._max_epochs
            self._vitals.epochs_per_second = total_epochs / elapsed
            self._vitals.batches_per_hour = (self._batches_completed / elapsed) * 3600

        # Capture best_runs from current episode (before reset clears env.best_accuracy)
        # Allow multiple entries per env to reflect distinct episodes; keep top 10 overall.
        # PINNING: Pinned records are never removed from the leaderboard.
        n_envs = data.get("n_envs")
        if not isinstance(n_envs, int) or n_envs <= 0:
            env_accuracies = data.get("env_accuracies")
            if isinstance(env_accuracies, (list, tuple)):
                n_envs = len(env_accuracies)
            else:
                n_envs = self.num_envs
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
        data = event.data or {}
        env_id = data.get("env_id")

        if env_id is None:
            return

        env = self._envs.get(env_id)
        if env is None:
            return

        # Parse configs
        slot_ids = tuple(data.get("slot_ids", []))
        raw_configs = data.get("configs", [])

        configs = [
            CounterfactualConfig(
                seed_mask=tuple(cfg.get("seed_mask", [])),
                accuracy=cfg.get("accuracy", 0.0),
            )
            for cfg in raw_configs
        ]

        env.counterfactual_matrix = CounterfactualSnapshot(
            slot_ids=slot_ids,
            configs=configs,
            strategy=data.get("strategy", "unavailable"),
            compute_time_ms=data.get("compute_time_ms", 0.0),
        )

    def _handle_analytics_snapshot(self, event: "TelemetryEvent") -> None:
        """Handle ANALYTICS_SNAPSHOT events used by Sanctum/Tamiyo UI."""
        data = event.data or {}
        kind = data.get("kind")

        # Batch-level action distribution is emitted at ops_normal and provides
        # Tamiyo panel coverage even when debug REWARD_COMPUTED is disabled.
        if kind == "action_distribution":
            action_counts = data.get("action_counts", {})
            if isinstance(action_counts, dict):
                counts = {str(action): int(count) for action, count in action_counts.items()}
                self._tamiyo.action_counts = counts
                self._tamiyo.total_actions = sum(counts.values())
            return

        # Vectorized PPO emits per-step "last_action" snapshots at ops_normal.
        # When these include decision fields (total_reward/action_confidence/etc),
        # treat them like REWARD_COMPUTED so Sanctum can populate:
        # - Recent Decisions carousel (TamiyoBrain)
        # - RewardComponents panel
        if kind == "last_action" and "action_confidence" in data:
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
        data = event.data or {}
        env_id = data.get("env_id")
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

        if event_type == "REWARD_COMPUTED":
            message = "Reward computed"
            metadata["action"] = normalize_action(data.get("action_name", "?"))
            metadata["reward"] = data.get("total_reward", 0.0)
        elif event_type.startswith("SEED_"):
            slot_id = event.slot_id or data.get("slot_id", "?")
            metadata["slot_id"] = slot_id
            if event_type == "SEED_GERMINATED":
                message = "Germinated"
                metadata["blueprint"] = data.get("blueprint_id", "?")
            elif event_type == "SEED_STAGE_CHANGED":
                message = "Stage changed"
                metadata["from"] = data.get("from", "?")
                metadata["to"] = data.get("to", "?")
            elif event_type == "SEED_FOSSILIZED":
                message = "Fossilized"
                metadata["improvement"] = data.get("improvement", 0.0)
            elif event_type == "SEED_PRUNED":
                message = "Pruned"
                metadata["reason"] = data.get("reason", "")
            else:
                message = event_type.replace("SEED_", "")
        elif event_type == "PPO_UPDATE_COMPLETED":
            if data.get("skipped"):
                message = "PPO skipped"
                metadata["reason"] = "buffer rollback"
            else:
                message = "PPO update"
                metadata["entropy"] = data.get("entropy", 0.0)
                metadata["clip_fraction"] = data.get("clip_fraction", 0.0)
        elif event_type == "BATCH_EPOCH_COMPLETED":
            message = "Batch complete"
            metadata["batch"] = data.get("batch_idx", 0)
            metadata["episodes"] = data.get("episodes_completed", 0)
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
