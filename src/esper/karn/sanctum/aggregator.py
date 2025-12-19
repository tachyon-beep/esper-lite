"""Sanctum Telemetry Aggregator - Transforms event stream into SanctumSnapshot.

Maintains stateful accumulation of telemetry events to build
real-time SanctumSnapshot objects for the Sanctum TUI.

Thread-safe: Uses threading.Lock to protect state during concurrent
access from training thread (process_event) and UI thread (get_snapshot).
"""

from __future__ import annotations

import threading
import time
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
    "CULL_PROBATION": "CULL",
    "CULL_STAGNATION": "CULL",
    "CULL_ACCURACY": "CULL",
}


def normalize_action(action: str) -> str:
    """Normalize factored action names to base action.

    Args:
        action: Action name, possibly factored (e.g., "GERMINATE_CONV_LIGHT").

    Returns:
        Base action name (e.g., "GERMINATE").
    """
    return ACTION_NORMALIZATION.get(action, action)


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
    - SEED_CULLED: Increment culled count
    - BATCH_COMPLETED: Update episode/throughput

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
    _current_batch: int = 0  # Current batch index (from BATCH_COMPLETED)
    _batch_avg_accuracy: float = 0.0  # Batch-level average accuracy
    _batch_rolling_accuracy: float = 0.0  # Rolling average for trend display
    _batch_avg_reward: float = 0.0  # Batch average reward
    _batch_total_episodes: int = 0  # Total episodes in run
    _run_config: "RunConfig" = field(default_factory=lambda: RunConfig())

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
        self._mean_accuracy_history = deque(maxlen=self.max_history)
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._run_config = RunConfig()

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
        elif event_type == "BATCH_COMPLETED":
            self._handle_batch_completed(event)

    def get_snapshot(self) -> SanctumSnapshot:
        """Get current SanctumSnapshot.

        Returns:
            Complete snapshot of current aggregator state.
        """
        with self._lock:
            return self._get_snapshot_unlocked()

    def _get_snapshot_unlocked(self) -> SanctumSnapshot:
        """Get snapshot without locking (caller must hold lock)."""
        now = time.time()
        staleness = now - self._last_event_ts if self._last_event_ts else float("inf")
        runtime = now - self._start_time if self._connected else 0.0

        # Update system vitals
        self._update_system_vitals()

        # Get focused env's reward components for the detail panel
        focused_rewards = RewardComponents()
        if self._focused_env_id in self._envs:
            focused_env = self._envs[self._focused_env_id]
            if isinstance(focused_env.reward_components, RewardComponents):
                focused_rewards = focused_env.reward_components

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
        )

    # =========================================================================
    # Event Handlers (matching TUIOutput behavior 1:1)
    # =========================================================================

    def _handle_training_started(self, event: "TelemetryEvent") -> None:
        """Handle TRAINING_STARTED event."""
        data = event.data or {}
        self._run_id = data.get("run_id", data.get("episode_id", ""))
        self._task_name = data.get("task", "")
        self._max_epochs = data.get("max_epochs", 75)
        self._connected = True
        self._start_time = time.time()
        self._current_episode = 0
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
        else:
            # Default to 2x2 grid (r0c0, r0c1, r1c0, r1c1)
            from esper.leyline.slot_id import format_slot_id
            self._slot_ids = [format_slot_id(r, c) for r in range(2) for c in range(2)]

        # Reset and recreate env states
        self._envs.clear()
        for i in range(self.num_envs):
            self._ensure_env(i)

        # Reset Tamiyo state
        self._tamiyo = TamiyoState()

        # Reset best runs for new training session
        self._best_runs = []

    def _handle_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle EPOCH_COMPLETED event."""
        data = event.data or {}
        env_id = data.get("env_id", 0)

        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Update accuracy
        val_acc = data.get("val_accuracy", 0.0)
        val_loss = data.get("val_loss", 0.0)
        inner_epoch = data.get("inner_epoch", data.get("epoch", 0))

        # Capture previous accuracy for status update
        prev_acc = env.accuracy_history[-1] if env.accuracy_history else 0.0

        env.host_accuracy = val_acc
        env.host_loss = val_loss
        env.current_epoch = inner_epoch

        # Update global epoch
        self._current_epoch = inner_epoch

        # Add to accuracy history
        env.accuracy_history.append(val_acc)

        # Track best accuracy
        if val_acc > env.best_accuracy:
            env.epochs_since_improvement = 0
            env.best_accuracy = val_acc
            env.best_accuracy_epoch = inner_epoch
            env.best_accuracy_episode = self._current_episode
            # Snapshot seeds at best accuracy
            env.best_seeds = {k: SeedState(**v.__dict__) for k, v in env.seeds.items()}
        else:
            env.epochs_since_improvement += 1

        # Update env status based on accuracy changes
        env._update_status(prev_acc, val_acc)

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

            # Track slot_ids dynamically
            if slot_id not in self._slot_ids and slot_id != "unknown":
                self._slot_ids.append(slot_id)
                self._slot_ids.sort()

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

        # Update Tamiyo state with all PPO metrics
        self._tamiyo.entropy = data.get("entropy", 0.0)
        self._tamiyo.clip_fraction = data.get("clip_fraction", 0.0)
        self._tamiyo.explained_variance = data.get("explained_variance", 0.0)
        self._tamiyo.kl_divergence = data.get("kl_divergence", 0.0)

        # Losses
        self._tamiyo.policy_loss = data.get("policy_loss", 0.0)
        self._tamiyo.value_loss = data.get("value_loss", 0.0)
        self._tamiyo.entropy_loss = data.get("entropy_loss", 0.0)
        self._tamiyo.grad_norm = data.get("grad_norm", 0.0)

        # Advantage stats
        self._tamiyo.advantage_mean = data.get("advantage_mean", 0.0)
        self._tamiyo.advantage_std = data.get("advantage_std", 0.0)

        # Ratio statistics (PPO importance sampling ratios)
        self._tamiyo.ratio_mean = data.get("ratio_mean", 1.0)
        self._tamiyo.ratio_min = data.get("ratio_min", 1.0)
        self._tamiyo.ratio_max = data.get("ratio_max", 1.0)
        self._tamiyo.ratio_std = data.get("ratio_std", 0.0)

        # Learning rate and entropy coefficient
        self._tamiyo.learning_rate = data.get("lr")
        self._tamiyo.entropy_coef = data.get("entropy_coef", 0.0)

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
        self._tamiyo.head_slot_entropy = data.get("head_slot_entropy", 0.0)
        self._tamiyo.head_slot_grad_norm = data.get("head_slot_grad_norm", 0.0)
        self._tamiyo.head_blueprint_entropy = data.get("head_blueprint_entropy", 0.0)
        self._tamiyo.head_blueprint_grad_norm = data.get("head_blueprint_grad_norm", 0.0)

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
            ratio_penalty=data.get("ratio_penalty", 0.0),
            stage_bonus=data.get("stage_bonus", 0.0),
            fossilize_terminal_bonus=data.get("fossilize_terminal_bonus", 0.0),
            blending_warning=data.get("blending_warning", 0.0),
            probation_warning=data.get("probation_warning", 0.0),
            val_acc=data.get("val_acc", env.host_accuracy),
            total=total_reward,
            last_action=action_name,
            env_id=env_id,
        )

        # Capture decision snapshot (NEW)
        # Only capture if we have decision data (action_confidence present)
        if "action_confidence" in data:
            self._tamiyo.last_decision = DecisionSnapshot(
                timestamp=event.timestamp or datetime.now(timezone.utc),
                slot_states=data.get("slot_states", {}),
                host_accuracy=data.get("host_accuracy", env.host_accuracy),
                chosen_action=action_name,
                chosen_slot=data.get("action_slot"),
                confidence=data.get("action_confidence", 0.0),
                expected_value=data.get("value_estimate", 0.0),
                actual_reward=total_reward,
                alternatives=data.get("alternatives", []),
            )

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

        # Dynamically track slot_ids as we observe them
        if slot_id and slot_id not in self._slot_ids and slot_id != "unknown":
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
            env.active_seed_count += 1

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
            env.fossilized_params += int(data.get("params_added", 0) or 0)
            env.fossilized_count += 1
            env.active_seed_count = max(0, env.active_seed_count - 1)

        elif event_type == "SEED_CULLED":
            # Capture cull context before resetting (P1/P2 telemetry gap fix)
            seed.cull_reason = data.get("reason", "")
            seed.improvement = data.get("improvement", 0.0)
            seed.auto_culled = data.get("auto_culled", False)
            seed.epochs_total = data.get("epochs_total", 0)
            seed.counterfactual = data.get("counterfactual", 0.0)
            seed.blueprint_id = data.get("blueprint_id") or seed.blueprint_id

            # Reset slot to DORMANT
            seed.stage = "DORMANT"
            seed.seed_params = 0
            seed.blueprint_id = None
            seed.alpha = 0.0
            seed.accuracy_delta = 0.0
            seed.grad_ratio = 0.0
            seed.has_vanishing = False
            seed.has_exploding = False
            seed.epochs_in_stage = 0
            env.culled_count += 1
            env.active_seed_count = max(0, env.active_seed_count - 1)

    def _handle_batch_completed(self, event: "TelemetryEvent") -> None:
        """Handle BATCH_COMPLETED event (episode completion)."""
        data = event.data or {}

        # Capture current episode BEFORE updating for best_runs check
        # (best_accuracy_episode was set using the old value during EPOCH_COMPLETED)
        current_ep = self._current_episode

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

        # Calculate throughput
        now = time.time()
        elapsed = now - self._start_time
        if elapsed > 0:
            total_epochs = self._current_episode * self._max_epochs
            self._vitals.epochs_per_second = total_epochs / elapsed
            self._vitals.batches_per_hour = (self._batches_completed / elapsed) * 3600

        # Capture best run records for envs that improved during this batch
        # Do this BEFORE resetting seed state so we can snapshot the seeds
        for env in self._envs.values():
            # Check if this env achieved a new best during this episode
            if env.best_accuracy_episode == current_ep and env.best_accuracy > 0:
                # Compute absolute episode for human-readable display
                # e.g., batch 3 with 8 envs and env_id=0 → absolute episode 25
                absolute_ep = current_ep * self.num_envs + env.env_id + 1
                record = BestRunRecord(
                    env_id=env.env_id,
                    episode=current_ep,
                    peak_accuracy=env.best_accuracy,
                    final_accuracy=env.host_accuracy,
                    absolute_episode=absolute_ep,
                    seeds={k: SeedState(**v.__dict__) for k, v in env.best_seeds.items()},
                )
                # Remove existing record ONLY if same env in same episode (duplicate event)
                # Different episodes are different training runs, keep both!
                self._best_runs = [
                    r for r in self._best_runs
                    if not (r.env_id == env.env_id and r.episode == current_ep)
                ]
                self._best_runs.append(record)

        # Sort by peak accuracy descending, keep top 10
        self._best_runs.sort(key=lambda r: r.peak_accuracy, reverse=True)
        self._best_runs = self._best_runs[:10]

        # Reset per-env seed state for next episode
        for env in self._envs.values():
            env.seeds.clear()
            env.active_seed_count = 0
            env.fossilized_count = 0
            env.culled_count = 0
            env.fossilized_params = 0

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
        """Add event to log with formatting, episode, and relative time."""
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

        # Format message based on event type
        if event_type == "REWARD_COMPUTED":
            action = normalize_action(data.get("action_name", "?"))
            total = data.get("total_reward", 0.0)
            message = f"{action} r={total:+.3f}"
        elif event_type.startswith("SEED_"):
            slot_id = event.slot_id or data.get("slot_id", "?")
            if event_type == "SEED_GERMINATED":
                blueprint = data.get("blueprint_id", "?")
                message = f"{slot_id} germinated ({blueprint})"
            elif event_type == "SEED_STAGE_CHANGED":
                message = f"{slot_id} {data.get('from', '?')}->{data.get('to', '?')}"
            elif event_type == "SEED_FOSSILIZED":
                improvement = data.get("improvement", 0.0)
                message = f"{slot_id} fossilized" + (f" (+{improvement:.1f}%)" if improvement else "")
            elif event_type == "SEED_CULLED":
                reason = data.get("reason", "")
                message = f"{slot_id} culled" + (f" ({reason})" if reason else "")
            else:
                message = slot_id
        elif event_type == "PPO_UPDATE_COMPLETED":
            if data.get("skipped"):
                message = "skipped (buffer rollback)"
            else:
                ent = data.get("entropy", 0.0)
                clip = data.get("clip_fraction", 0.0)
                message = f"ent={ent:.3f} clip={clip:.3f}"
        elif event_type == "BATCH_COMPLETED":
            batch = data.get("batch_idx", "?")
            eps = data.get("episodes_completed", "?")
            message = f"batch={batch} ep={eps}"
        else:
            message = event.message or event_type

        self._event_log.append(EventLogEntry(
            timestamp=timestamp.strftime("%H:%M:%S"),
            event_type=event_type,
            env_id=env_id,
            message=message,
            episode=self._current_episode,
            relative_time=relative_time,
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
                self._vitals.gpu_stats = {}
                for i, device in enumerate(self._gpu_devices):
                    if device.startswith("cuda"):
                        device_idx = int(device.split(":")[-1]) if ":" in device else i
                        try:
                            mem_allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
                            mem_reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)
                            props = torch.cuda.get_device_properties(device_idx)
                            mem_total = props.total_memory / (1024**3)

                            self._vitals.gpu_stats[device] = GPUStats(
                                device_id=device,
                                memory_used_gb=mem_allocated,
                                memory_total_gb=mem_total,
                                utilization=mem_allocated / mem_total if mem_total > 0 else 0.0,
                            )
                        except Exception:
                            pass
        except ImportError:
            pass
