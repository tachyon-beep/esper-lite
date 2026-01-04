"""Sanctum Telemetry Aggregator - Transforms event stream into SanctumSnapshot.

Maintains stateful accumulation of telemetry events to build
real-time SanctumSnapshot objects for the Sanctum TUI.

Thread-safe: Uses threading.Lock to protect state during concurrent
access from training thread (process_event) and UI thread (get_snapshot).
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
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
    ShapleySnapshot,
    ShapleyEstimate,
    SeedLifecycleStats,
    ObservationStats,
    EpisodeStats,
    compute_entropy_velocity,
    compute_collapse_risk,
    compute_correlation,
)
from esper.karn.sanctum.snapshot_copy import copy_snapshot
from esper.karn.constants import TUIThresholds
from esper.karn.sanctum.widgets.reward_health import RewardHealthData
from esper.karn.pareto import extract_pareto_frontier, compute_hypervolume_2d
from esper.leyline import (
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_GAMMA,
    TrainingStartedPayload,
    EpochCompletedPayload,
    BatchEpochCompletedPayload,
    PPOUpdatePayload,
    SeedGerminatedPayload,
    SeedStageChangedPayload,
    SeedGateEvaluatedPayload,
    SeedFossilizedPayload,
    SeedPrunedPayload,
    CounterfactualMatrixPayload,
    AnalyticsSnapshotPayload,
    EpisodeOutcomePayload,
    GovernorRollbackPayload,
    TEMPO_NAMES,
)

if TYPE_CHECKING:
    from esper.leyline import EpisodeOutcome, TelemetryEvent

_logger = logging.getLogger(__name__)


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


def detect_rate_trend(history: deque[float]) -> str:
    """Detect trend in rate history (rising/stable/falling).

    Compares recent 5 samples to older 5 samples.
    """
    if len(history) < 5:
        return "stable"

    recent = list(history)[-5:]
    older = list(history)[-10:-5] if len(history) >= 10 else list(history)[:5]

    if not older:
        return "stable"

    recent_mean = sum(recent) / len(recent)
    older_mean = sum(older) / len(older)

    # 20% change threshold
    threshold = 0.2 * max(abs(older_mean), 0.01)
    diff = recent_mean - older_mean

    if diff > threshold:
        return "rising"
    elif diff < -threshold:
        return "falling"
    return "stable"


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
        if action.startswith("ADVANCE"):
            return "ADVANCE"
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
    - ANALYTICS_SNAPSHOT(last_action): Update per-env reward/decision tracking
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
    _max_epochs: int = DEFAULT_EPISODE_LENGTH
    _max_batches: int = 100
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
    _cumulative_germinated: int = 0
    # Cumulative action counts across all batches
    _cumulative_action_counts: dict[str, int] = field(default_factory=dict)
    _cumulative_total_actions: int = 0
    # Cumulative graveyard (per-blueprint lifecycle stats across entire run)
    _cumulative_blueprint_spawns: dict[str, int] = field(default_factory=dict)
    _cumulative_blueprint_fossilized: dict[str, int] = field(default_factory=dict)
    _cumulative_blueprint_prunes: dict[str, int] = field(default_factory=dict)

    # Seed lifecycle tracking for trends
    _seed_lifespan_history: deque[int] = field(default_factory=lambda: deque(maxlen=100))
    _germination_rate_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _prune_rate_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _fossilize_rate_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))

    # Per-env state: env_id -> EnvState
    _envs: dict[int, EnvState] = field(default_factory=dict)

    # Tamiyo state
    _tamiyo: TamiyoState = field(default_factory=TamiyoState)

    # System vitals
    _vitals: SystemVitals = field(default_factory=SystemVitals)
    _gpu_devices: list[str] = field(default_factory=list)
    _gpu_total_memory_gb: dict[int, float] = field(default_factory=dict)

    # Event log
    _event_log: deque[EventLogEntry] = field(default_factory=lambda: deque(maxlen=100))

    # Focused env for reward panel
    _focused_env_id: int = 0

    # Last action target (for EnvOverview row highlighting)
    _last_action_env_id: int | None = None
    _last_action_timestamp: datetime | None = None

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
        self._observation_stats = ObservationStats()  # Updated when telemetry provides stats
        self._vitals = SystemVitals()
        self._gpu_devices = []
        self._gpu_total_memory_gb = {}
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
        self._cumulative_germinated = 0
        self._cumulative_action_counts = {}
        self._cumulative_total_actions = 0
        self._cumulative_blueprint_spawns = {}
        self._cumulative_blueprint_fossilized = {}
        self._cumulative_blueprint_prunes = {}

        # Seed lifecycle tracking
        self._seed_lifespan_history = deque(maxlen=100)
        self._germination_rate_history = deque(maxlen=20)
        self._prune_rate_history = deque(maxlen=20)
        self._fossilize_rate_history = deque(maxlen=20)

        # Episode outcomes for Pareto analysis (reward health panel)
        self._episode_outcomes: list[EpisodeOutcome] = []

        # Episode diagnostics (TELE-610)
        self._episode_lengths: deque[int] = deque(maxlen=100)  # Rolling window
        self._success_count: int = 0
        self._timeout_count: int = 0
        self._total_germinate: int = 0
        self._total_prune: int = 0
        self._total_fossilize: int = 0
        self._recent_outcomes: deque[bool] = deque(maxlen=20)  # For trend detection (True=success)

        # Sliding window for instantaneous throughput (episodes per second)
        # Stores (timestamp, episode_count) tuples for rate calculation
        self._episode_completion_times: deque[tuple[float, int]] = deque(maxlen=100)

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

        # Get event type name - handle both Enum (with .name) and string
        event_type_raw = event.event_type
        event_type = event_type_raw.name if hasattr(event_type_raw, "name") else event_type_raw

        # Log event
        self._add_event_log(event, event_type)

        # Route to handler
        if event_type == "TRAINING_STARTED":
            self._handle_training_started(event)
        elif event_type == "EPOCH_COMPLETED":
            self._handle_epoch_completed(event)
        elif event_type == "PPO_UPDATE_COMPLETED":
            self._handle_ppo_update(event)
        elif event_type.startswith("SEED_"):
            self._handle_seed_event(event, event_type)
        elif event_type == "BATCH_EPOCH_COMPLETED":
            self._handle_batch_epoch_completed(event)
        elif event_type == "COUNTERFACTUAL_MATRIX_COMPUTED":
            self._handle_counterfactual_matrix(event)
        elif event_type == "ANALYTICS_SNAPSHOT":
            self._handle_analytics_snapshot(event)
        elif event_type == "EPISODE_OUTCOME":
            self._handle_episode_outcome(event)
        elif event_type == "GOVERNOR_ROLLBACK":
            self._handle_governor_rollback(event)

    def get_snapshot(self) -> SanctumSnapshot:
        """Get current SanctumSnapshot.

        Returns:
            Complete snapshot of current aggregator state.
        """
        with self._lock:
            return self._get_snapshot_unlocked()

    def _compute_instantaneous_throughput(self, now: float) -> float:
        """Compute episodes per second using sliding window.

        Uses recent episode completion timestamps to calculate instantaneous
        throughput, unaffected by warmup time. Falls back to 0.0 if insufficient
        data (need at least 2 data points spanning > 1 second).

        Args:
            now: Current timestamp (time.time()).

        Returns:
            Episodes per second based on recent completions.
        """
        if len(self._episode_completion_times) < 2:
            return 0.0

        # Get oldest and newest timestamps in window
        oldest_time, oldest_count = self._episode_completion_times[0]
        newest_time, newest_count = self._episode_completion_times[-1]

        # Calculate time span and episode delta
        time_span = newest_time - oldest_time
        episode_delta = newest_count - oldest_count

        # Need meaningful time span to avoid division issues
        if time_span < 1.0:
            return 0.0

        return episode_delta / time_span

    def _get_snapshot_unlocked(self) -> SanctumSnapshot:
        """Get snapshot without locking (caller must hold lock)."""
        now = time.time()
        now_dt = datetime.now(timezone.utc)
        staleness = now - self._last_event_ts if self._last_event_ts else float("inf")
        runtime = now - self._start_time if self._connected else 0.0

        # Update system vitals
        self._update_system_vitals()

        # Aggregate action counts from per-step ANALYTICS_SNAPSHOT(last_action) telemetry.
        # Falls back to ANALYTICS_SNAPSHOT(action_distribution) which populates
        # self._tamiyo directly if per-step telemetry is unavailable.
        aggregated_actions: dict[str, int] = {
            "WAIT": 0,
            "GERMINATE": 0,
            "SET_ALPHA_TARGET": 0,
            "PRUNE": 0,
            "FOSSILIZE": 0,
            "ADVANCE": 0,
        }
        total_actions = 0
        for env in self._envs.values():
            for action, count in env.action_counts.items():
                aggregated_actions[action] = aggregated_actions.get(action, 0) + count
            total_actions += env.total_actions
        if total_actions > 0:
            self._tamiyo.action_counts = aggregated_actions
            self._tamiyo.total_actions = total_actions
        # Always update cumulative counts (even if current batch is zero)
        self._tamiyo.cumulative_action_counts = dict(self._cumulative_action_counts)
        self._tamiyo.cumulative_total_actions = self._cumulative_total_actions

        # Carousel rotation: expire ONE oldest decision per cycle if > 2min old
        # This runs every snapshot (250ms), creating natural stagger as each decision
        # expires based on its individual timestamp, not batch replacement.
        # Must match MAX_DISPLAY_AGE_S in ActionHeadsPanel (120 seconds).
        decisions = self._tamiyo.recent_decisions
        for i in range(len(decisions) - 1, -1, -1):  # Iterate oldest-first
            d = decisions[i]
            age = (now_dt - d.timestamp).total_seconds()
            if age > 120.0:
                decisions.pop(i)
                break  # Only expire ONE per cycle for smooth rotation
        self._tamiyo.recent_decisions = decisions[:MAX_DECISIONS]

        # Get most interesting reward components for the detail panel
        # Priority: find env with non-zero bonuses/penalties, else use focused env
        best_reward = RewardComponents()
        best_score = 0.0
        for env in self._envs.values():
            rc = env.reward_components
            if not isinstance(rc, RewardComponents):
                continue
            # Score based on interesting activity (bonuses/penalties)
            score = (
                abs(rc.alpha_shock)
                + abs(rc.ratio_penalty)
                + abs(rc.stage_bonus)
                + abs(rc.fossilize_terminal_bonus)
                + abs(rc.hindsight_credit)
            )
            if score > best_score:
                best_score = score
                best_reward = rc
        # Fallback to focused env if no interesting activity found
        if best_score == 0.0 and self._focused_env_id in self._envs:
            focused_env = self._envs[self._focused_env_id]
            if isinstance(focused_env.reward_components, RewardComponents):
                best_reward = focused_env.reward_components

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

        # Compute seed lifecycle stats
        blend_success = (
            self._cumulative_fossilized / max(1, self._cumulative_fossilized + self._cumulative_pruned)
            if (self._cumulative_fossilized + self._cumulative_pruned) > 0
            else 0.0
        )
        avg_lifespan = (
            sum(self._seed_lifespan_history) / len(self._seed_lifespan_history)
            if self._seed_lifespan_history
            else 0.0
        )
        current_ep = max(1, self._current_episode)

        seed_lifecycle = SeedLifecycleStats(
            germination_count=self._cumulative_germinated,
            prune_count=self._cumulative_pruned,
            fossilize_count=self._cumulative_fossilized,
            active_count=active_slots,
            total_slots=total_slots,
            germination_rate=self._cumulative_germinated / current_ep,
            prune_rate=self._cumulative_pruned / current_ep,
            fossilize_rate=self._cumulative_fossilized / current_ep,
            blend_success_rate=blend_success,
            avg_lifespan_epochs=avg_lifespan,
            germination_trend=detect_rate_trend(self._germination_rate_history),
            prune_trend=detect_rate_trend(self._prune_rate_history),
            fossilize_trend=detect_rate_trend(self._fossilize_rate_history),
        )

        # Observation stats from telemetry (updated when present in ANALYTICS_SNAPSHOT)
        observation_stats = self._observation_stats

        # Episode stats (TELE-610)
        total = self._current_episode
        if total > 0 and self._episode_lengths:
            lengths = list(self._episode_lengths)
            length_mean = sum(lengths) / len(lengths)
            length_std = (sum((x - length_mean) ** 2 for x in lengths) / len(lengths)) ** 0.5
            length_min = min(lengths)
            length_max = max(lengths)

            success_rate = self._success_count / total
            timeout_rate = self._timeout_count / total

            # Trend detection (DRL expert: compare rolling windows, not single samples)
            outcomes = list(self._recent_outcomes)
            if len(outcomes) >= 10:
                recent = outcomes[-10:]  # Last 10 episodes
                older = outcomes[-20:-10] if len(outcomes) >= 20 else outcomes[:len(outcomes)//2]
                recent_rate = sum(recent) / len(recent)
                older_rate = sum(older) / len(older) if older else recent_rate
                if recent_rate > older_rate + 0.1:
                    trend = "improving"
                elif recent_rate < older_rate - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"  # Not enough data for trend

            # Action efficiency
            total_steps = sum(lengths)
            steps_per_germinate = total_steps / max(1, self._total_germinate)
            steps_per_prune = total_steps / max(1, self._total_prune)
            steps_per_fossilize = total_steps / max(1, self._total_fossilize)

            # === DRL Diagnostic Metrics ===
            # Action entropy: Normalized Shannon entropy of cumulative action distribution
            # H = -sum(p(a)*log(p(a))) / log(|A|), normalized to [0, 1]
            action_entropy = 0.0
            if self._cumulative_action_counts:
                total_actions = sum(self._cumulative_action_counts.values())
                if total_actions > 0:
                    num_actions = len(self._cumulative_action_counts)
                    if num_actions > 1:
                        entropy_sum = 0.0
                        for count in self._cumulative_action_counts.values():
                            if count > 0:
                                p = count / total_actions
                                entropy_sum -= p * math.log(p)
                        # Normalize by max entropy (log of action count)
                        action_entropy = entropy_sum / math.log(num_actions)

            # Yield rate: fossilizations / germinations (seed efficiency)
            yield_rate = 0.0
            if self._cumulative_germinated > 0:
                yield_rate = self._cumulative_fossilized / self._cumulative_germinated

            # Slot utilization: active_slots / total_slots
            slot_utilization = active_slots / total_slots if total_slots > 0 else 0.0

            # Episodes per second (instantaneous throughput via sliding window)
            # Uses recent episode completions rather than cumulative average to show
            # current rate, unaffected by warmup time
            episodes_per_second = self._compute_instantaneous_throughput(now)

            episode_stats = EpisodeStats(
                total_episodes=total,
                episodes_per_second=episodes_per_second,
                length_mean=length_mean,
                length_std=length_std,
                length_min=length_min,
                length_max=length_max,
                success_count=self._success_count,
                timeout_count=self._timeout_count,
                success_rate=success_rate,
                timeout_rate=timeout_rate,
                early_termination_rate=0.0,  # Not applicable for fixed-length episodes
                steps_per_germinate=steps_per_germinate,
                steps_per_prune=steps_per_prune,
                steps_per_fossilize=steps_per_fossilize,
                action_entropy=action_entropy,
                yield_rate=yield_rate,
                slot_utilization=slot_utilization,
                completion_trend=trend,
            )
        else:
            # Minimal stats - still compute DRL metrics from cumulative data
            action_entropy = 0.0
            if self._cumulative_action_counts:
                total_actions = sum(self._cumulative_action_counts.values())
                if total_actions > 0:
                    num_actions = len(self._cumulative_action_counts)
                    if num_actions > 1:
                        entropy_sum = 0.0
                        for count in self._cumulative_action_counts.values():
                            if count > 0:
                                p = count / total_actions
                                entropy_sum -= p * math.log(p)
                        action_entropy = entropy_sum / math.log(num_actions)

            yield_rate = 0.0
            if self._cumulative_germinated > 0:
                yield_rate = self._cumulative_fossilized / self._cumulative_germinated

            slot_utilization = active_slots / total_slots if total_slots > 0 else 0.0

            # Episodes per second (instantaneous throughput via sliding window)
            episodes_per_second = self._compute_instantaneous_throughput(now)

            episode_stats = EpisodeStats(
                total_episodes=total,
                episodes_per_second=episodes_per_second,
                action_entropy=action_entropy,
                yield_rate=yield_rate,
                slot_utilization=slot_utilization,
            )

        snapshot = SanctumSnapshot(
            # Run context
            run_id=self._run_id,
            task_name=self._task_name,
            run_config=self._run_config,
            current_episode=self._current_episode,
            current_batch=self._current_batch or self._batches_completed,
            current_epoch=self._current_epoch,
            max_epochs=self._max_epochs,
            max_batches=self._max_batches,
            runtime_seconds=runtime,
            connected=self._connected,
            staleness_seconds=staleness,
            captured_at=datetime.now(timezone.utc).isoformat(),
            aggregate_mean_accuracy=mean_accuracy,
            aggregate_mean_reward=mean_reward,
            # Slot configuration for dynamic columns
            slot_ids=self._slot_ids,
            # Per-env state
            envs=self._envs,
            focused_env_id=self._focused_env_id,
            # Last action target (for EnvOverview row highlighting)
            last_action_env_id=self._last_action_env_id,
            last_action_timestamp=self._last_action_timestamp,
            # Best reward components for detail panel (most interesting activity)
            rewards=best_reward,
            # Tamiyo state
            tamiyo=self._tamiyo,
            # System vitals
            vitals=self._vitals,
            # Event log
            event_log=list(self._event_log),
            # Historical best runs
            best_runs=self._best_runs,
            # Rolling mean accuracy history
            mean_accuracy_history=self._mean_accuracy_history,
            # Batch-level aggregates
            batch_avg_reward=self._batch_avg_reward,
            batch_total_episodes=self._batch_total_episodes,
            # Cumulative counts (never reset, tracks entire training run)
            cumulative_fossilized=self._cumulative_fossilized,
            cumulative_pruned=self._cumulative_pruned,
            cumulative_blueprint_spawns=self._cumulative_blueprint_spawns,
            cumulative_blueprint_fossilized=self._cumulative_blueprint_fossilized,
            cumulative_blueprint_prunes=self._cumulative_blueprint_prunes,
            # Aggregate slot state across all environments
            slot_stage_counts=slot_stage_counts,
            total_slots=total_slots,
            active_slots=active_slots,
            avg_epochs_in_stage=avg_epochs,
            # New diagnostic metrics
            seed_lifecycle=seed_lifecycle,
            observation_stats=observation_stats,
            episode_stats=episode_stats,
            cumulative_germinated=self._cumulative_germinated,
        )
        return copy_snapshot(snapshot)

    # =========================================================================
    # Event Handlers (matching TUIOutput behavior 1:1)
    # =========================================================================

    def _handle_training_started(self, event: "TelemetryEvent") -> None:
        """Handle TRAINING_STARTED event."""
        if not isinstance(event.data, TrainingStartedPayload):
            raise TypeError(
                "TRAINING_STARTED requires TrainingStartedPayload, got "
                f"{type(event.data).__name__}"
            )

        payload = event.data
        self._run_id = payload.episode_id
        self._task_name = payload.task
        self._max_epochs = payload.max_epochs
        self._max_batches = payload.max_batches
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
        # entropy_anneal is optional telemetry (None = not configured)
        entropy_anneal_config = payload.entropy_anneal or {}

        self._run_config = RunConfig(
            seed=payload.seed,
            n_episodes=payload.n_episodes,
            lr=payload.lr,
            clip_ratio=payload.clip_ratio,
            entropy_coef=payload.entropy_coef,
            param_budget=payload.param_budget,
            resume_path=payload.resume_path,
            entropy_anneal=entropy_anneal_config,
        )

        # Capture slot configuration from event
        self._slot_ids = list(payload.slot_ids)
        self._slot_ids_locked = True  # Lock - don't add more dynamically

        # Reset and recreate env states
        self._envs.clear()
        for i in range(self.num_envs):
            self._ensure_env(i)

        # Reset sliding window for throughput calculation
        self._episode_completion_times.clear()

        # Reset Tamiyo state
        self._tamiyo = TamiyoState()

        # Compile status (static configuration from training start)
        self._tamiyo.infrastructure.compile_enabled = payload.compile_enabled
        # MED-04 fix: Use explicit None check - empty string is falsy but valid
        self._tamiyo.infrastructure.compile_backend = payload.compile_backend if payload.compile_backend is not None else ""
        self._tamiyo.infrastructure.compile_mode = payload.compile_mode if payload.compile_mode is not None else ""

        # Reset best runs for new training session
        self._best_runs = []

    def _handle_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle EPOCH_COMPLETED event (per-env only).

        Only processes per-env EPOCH_COMPLETED events (with explicit env_id).
        Batch-level events now use BATCH_EPOCH_COMPLETED, but we still skip
        any event missing env_id as a defensive measure.
        """
        if not isinstance(event.data, EpochCompletedPayload):
            raise TypeError(
                "EPOCH_COMPLETED requires EpochCompletedPayload, got "
                f"{type(event.data).__name__}"
            )

        payload = event.data

        env_id = payload.env_id
        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Clear rollback state - training has resumed for this env
        if env.rolled_back:
            env.rolled_back = False
            env.rollback_reason = ""

        # Clear stale Shapley data at episode start
        # Shapley is computed at episode END, so displaying previous episode's
        # data during a new episode is misleading. Clear it so the panel shows
        # "unavailable" until fresh data arrives.
        inner_epoch = payload.inner_epoch
        if inner_epoch == 0:
            env.shapley_snapshot = ShapleySnapshot()

        # Update accuracy
        val_acc = payload.val_accuracy
        val_loss = payload.val_loss

        env.host_loss = val_loss
        env.current_epoch = inner_epoch

        # Update global epoch
        self._current_epoch = inner_epoch

        # Update per-seed telemetry from EPOCH_COMPLETED event
        # Note: seeds telemetry is optional; inner dicts remain untyped.
        seeds_data = payload.seeds or {}
        for slot_id, seed_telemetry in seeds_data.items():
            # Ensure seed exists
            if slot_id not in env.seeds:
                env.seeds[slot_id] = SeedState(slot_id=slot_id)
            seed = env.seeds[slot_id]

            # HIGH-03 fix: Direct access for core fields - fail-fast if telemetry contract changes
            # Core lifecycle fields (always emitted by simic)
            seed.stage = seed_telemetry["stage"]
            seed.blueprint_id = seed_telemetry["blueprint_id"]
            seed.accuracy_delta = seed_telemetry["accuracy_delta"]
            seed.epochs_in_stage = seed_telemetry["epochs_in_stage"]
            seed.alpha = seed_telemetry["alpha"]
            # Gradient health fields (always emitted)
            seed.grad_ratio = seed_telemetry["grad_ratio"]
            seed.has_vanishing = seed_telemetry["has_vanishing"]
            seed.has_exploding = seed_telemetry["has_exploding"]
            # Inter-slot interaction metrics (optional - only present when counterfactual engine active)
            # These use .get() with None because absence is semantically meaningful ("not computed")
            if "contribution_velocity" in seed_telemetry:
                seed.contribution_velocity = seed_telemetry["contribution_velocity"]
            if "interaction_sum" in seed_telemetry:
                seed.interaction_sum = seed_telemetry["interaction_sum"]
            if "boost_received" in seed_telemetry:
                seed.boost_received = seed_telemetry["boost_received"]
            if "upstream_alpha_sum" in seed_telemetry:
                seed.upstream_alpha_sum = seed_telemetry["upstream_alpha_sum"]
            if "downstream_alpha_sum" in seed_telemetry:
                seed.downstream_alpha_sum = seed_telemetry["downstream_alpha_sum"]

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
            raise TypeError(
                "PPO_UPDATE_COMPLETED requires PPOUpdatePayload, got "
                f"{type(event.data).__name__}"
            )

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
        self._tamiyo.advantage_skewness = payload.advantage_skewness
        self._tamiyo.advantage_kurtosis = payload.advantage_kurtosis
        self._tamiyo.advantage_positive_ratio = payload.advantage_positive_ratio

        # Log prob extremes (NaN predictor) - have defaults
        self._tamiyo.log_prob_min = payload.log_prob_min
        self._tamiyo.log_prob_max = payload.log_prob_max

        # Ratio statistics (PPO importance sampling ratios) - have defaults
        self._tamiyo.ratio_mean = payload.ratio_mean
        self._tamiyo.ratio_min = payload.ratio_min
        self._tamiyo.ratio_max = payload.ratio_max
        self._tamiyo.ratio_std = payload.ratio_std

        # Per-head ratio max (Policy V2 - multi-head ratio explosion detection)
        self._tamiyo.head_slot_ratio_max = payload.head_slot_ratio_max
        self._tamiyo.head_blueprint_ratio_max = payload.head_blueprint_ratio_max
        self._tamiyo.head_style_ratio_max = payload.head_style_ratio_max
        self._tamiyo.head_tempo_ratio_max = payload.head_tempo_ratio_max
        self._tamiyo.head_alpha_target_ratio_max = payload.head_alpha_target_ratio_max
        self._tamiyo.head_alpha_speed_ratio_max = payload.head_alpha_speed_ratio_max
        self._tamiyo.head_alpha_curve_ratio_max = payload.head_alpha_curve_ratio_max
        self._tamiyo.head_op_ratio_max = payload.head_op_ratio_max
        self._tamiyo.joint_ratio_max = payload.joint_ratio_max

        # Learning rate and entropy coefficient - optional with None
        if payload.lr is not None:
            self._tamiyo.learning_rate = payload.lr

        if payload.entropy_coef is not None:
            self._tamiyo.entropy_coef = payload.entropy_coef

        # Gradient health - have defaults
        self._tamiyo.dead_layers = payload.dead_layers
        self._tamiyo.exploding_layers = payload.exploding_layers
        self._tamiyo.nan_grad_count = payload.nan_grad_count
        self._tamiyo.inf_grad_count = payload.inf_grad_count
        if payload.layer_gradient_health is not None:
            self._tamiyo.layer_gradient_health = payload.layer_gradient_health
        self._tamiyo.entropy_collapsed = payload.entropy_collapsed

        # LSTM hidden state health (B7-DRL-04)
        # None values indicate no LSTM in the policy (non-recurrent architecture)
        self._tamiyo.lstm_h_l2_total = payload.lstm_h_l2_total
        self._tamiyo.lstm_c_l2_total = payload.lstm_c_l2_total
        self._tamiyo.lstm_h_rms = payload.lstm_h_rms
        self._tamiyo.lstm_c_rms = payload.lstm_c_rms
        self._tamiyo.lstm_h_env_rms_mean = payload.lstm_h_env_rms_mean
        self._tamiyo.lstm_h_env_rms_max = payload.lstm_h_env_rms_max
        self._tamiyo.lstm_c_env_rms_mean = payload.lstm_c_env_rms_mean
        self._tamiyo.lstm_c_env_rms_max = payload.lstm_c_env_rms_max
        self._tamiyo.lstm_h_max = payload.lstm_h_max
        self._tamiyo.lstm_c_max = payload.lstm_c_max
        self._tamiyo.lstm_has_nan = payload.lstm_has_nan
        self._tamiyo.lstm_has_inf = payload.lstm_has_inf

        # Per-head NaN/Inf OR-latch (once True, stays True for entire run)
        if payload.head_nan_detected:
            for head, detected in payload.head_nan_detected.items():
                if detected:
                    self._tamiyo.head_nan_latch[head] = True
        if payload.head_inf_detected:
            for head, detected in payload.head_inf_detected.items():
                if detected:
                    self._tamiyo.head_inf_latch[head] = True

        # Performance timing - have defaults
        self._tamiyo.update_time_ms = payload.update_time_ms
        self._tamiyo.early_stop_epoch = payload.early_stop_epoch

        # Per-head entropy and gradient norms - optional with None
        if payload.head_slot_entropy is not None:
            self._tamiyo.head_slot_entropy = payload.head_slot_entropy
        if payload.head_slot_grad_norm is not None:
            self._tamiyo.head_slot_grad_norm_prev = self._tamiyo.head_slot_grad_norm
            self._tamiyo.head_slot_grad_norm = payload.head_slot_grad_norm
        if payload.head_blueprint_entropy is not None:
            self._tamiyo.head_blueprint_entropy = payload.head_blueprint_entropy
        if payload.head_blueprint_grad_norm is not None:
            self._tamiyo.head_blueprint_grad_norm_prev = self._tamiyo.head_blueprint_grad_norm
            self._tamiyo.head_blueprint_grad_norm = payload.head_blueprint_grad_norm
        if payload.head_style_grad_norm is not None:
            self._tamiyo.head_style_grad_norm_prev = self._tamiyo.head_style_grad_norm
            self._tamiyo.head_style_grad_norm = payload.head_style_grad_norm
        if payload.head_tempo_grad_norm is not None:
            self._tamiyo.head_tempo_grad_norm_prev = self._tamiyo.head_tempo_grad_norm
            self._tamiyo.head_tempo_grad_norm = payload.head_tempo_grad_norm
        if payload.head_alpha_target_grad_norm is not None:
            self._tamiyo.head_alpha_target_grad_norm_prev = self._tamiyo.head_alpha_target_grad_norm
            self._tamiyo.head_alpha_target_grad_norm = payload.head_alpha_target_grad_norm
        if payload.head_alpha_speed_grad_norm is not None:
            self._tamiyo.head_alpha_speed_grad_norm_prev = self._tamiyo.head_alpha_speed_grad_norm
            self._tamiyo.head_alpha_speed_grad_norm = payload.head_alpha_speed_grad_norm
        if payload.head_alpha_curve_grad_norm is not None:
            self._tamiyo.head_alpha_curve_grad_norm_prev = self._tamiyo.head_alpha_curve_grad_norm
            self._tamiyo.head_alpha_curve_grad_norm = payload.head_alpha_curve_grad_norm
        if payload.head_op_grad_norm is not None:
            self._tamiyo.head_op_grad_norm_prev = self._tamiyo.head_op_grad_norm
            self._tamiyo.head_op_grad_norm = payload.head_op_grad_norm

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

        # Value function statistics (for divergence detection)
        self._tamiyo.value_mean = payload.value_mean
        self._tamiyo.value_std = payload.value_std
        self._tamiyo.value_min = payload.value_min
        self._tamiyo.value_max = payload.value_max

        # Value function quality metrics (TELE-220 to TELE-228)
        # Update the nested ValueFunctionMetrics dataclass
        vf = self._tamiyo.value_function
        vf.v_return_correlation = payload.v_return_correlation
        vf.td_error_mean = payload.td_error_mean
        vf.td_error_std = payload.td_error_std
        vf.bellman_error = payload.bellman_error
        vf.return_p10 = payload.return_p10
        vf.return_p50 = payload.return_p50
        vf.return_p90 = payload.return_p90
        vf.return_variance = payload.return_variance
        vf.return_skewness = payload.return_skewness

        # Op-conditioned Q-values (Policy V2)
        self._tamiyo.q_germinate = payload.q_germinate
        self._tamiyo.q_advance = payload.q_advance
        self._tamiyo.q_fossilize = payload.q_fossilize
        self._tamiyo.q_prune = payload.q_prune
        self._tamiyo.q_wait = payload.q_wait
        self._tamiyo.q_set_alpha = payload.q_set_alpha
        self._tamiyo.q_variance = payload.q_variance
        self._tamiyo.q_spread = payload.q_spread

        # Set initial spread after warmup for relative thresholds
        WARMUP_BATCHES = 50
        if self._tamiyo.initial_value_spread is None and self._current_batch >= WARMUP_BATCHES:
            spread = self._tamiyo.value_max - self._tamiyo.value_min
            if spread > 0.1:  # Only set if non-trivial
                self._tamiyo.initial_value_spread = spread

        # Compute entropy velocity and collapse risk (after entropy_history is updated)
        self._tamiyo.entropy_velocity = compute_entropy_velocity(
            self._tamiyo.entropy_history
        )
        self._tamiyo.collapse_risk_score = compute_collapse_risk(
            self._tamiyo.entropy_history,
            critical_threshold=TUIThresholds.ENTROPY_CRITICAL,
            warning_threshold=TUIThresholds.ENTROPY_WARNING,
            max_healthy_entropy=TUIThresholds.ENTROPY_MAX,
            previous_risk=self._tamiyo._previous_risk,
        )
        self._tamiyo._previous_risk = self._tamiyo.collapse_risk_score

        # Compute entropy-clip correlation (for policy collapse pattern detection)
        self._tamiyo.entropy_clip_correlation = compute_correlation(
            self._tamiyo.entropy_history,
            self._tamiyo.clip_fraction_history,
        )

        # Gradient quality metrics (nested dataclass)
        self._tamiyo.gradient_quality.clip_fraction_positive = payload.clip_fraction_positive
        self._tamiyo.gradient_quality.clip_fraction_negative = payload.clip_fraction_negative
        self._tamiyo.gradient_quality.gradient_cv = payload.gradient_cv

        # Infrastructure metrics (nested dataclass)
        self._tamiyo.infrastructure.cuda_memory_allocated_gb = payload.cuda_memory_allocated_gb
        self._tamiyo.infrastructure.cuda_memory_reserved_gb = payload.cuda_memory_reserved_gb
        self._tamiyo.infrastructure.cuda_memory_peak_gb = payload.cuda_memory_peak_gb
        self._tamiyo.infrastructure.cuda_memory_fragmentation = payload.cuda_memory_fragmentation

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
            seed.alpha_curve = germinated_payload.alpha_curve
            env.active_seed_count += 1
            self._cumulative_germinated += 1

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
            seed.alpha_curve = stage_changed_payload.alpha_curve

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
            # Track lifespan for lifecycle stats
            if fossilized_payload.epochs_total > 0:
                self._seed_lifespan_history.append(fossilized_payload.epochs_total)
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
            # Track lifespan for lifecycle stats
            if pruned_payload.epochs_total > 0:
                self._seed_lifespan_history.append(pruned_payload.epochs_total)
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
        elif event_type == "SEED_GATE_EVALUATED" and isinstance(event.data, SeedGateEvaluatedPayload):
            gate_payload = event.data
            slot_id = event.slot_id or gate_payload.slot_id
            env_id = gate_payload.env_id

            self._ensure_env(env_id)
            env = self._envs[env_id]

            # Dynamically track slot_ids (covers tests / partial telemetry streams).
            if (
                not self._slot_ids_locked
                and slot_id
                and slot_id not in self._slot_ids
                and slot_id != "unknown"
            ):
                self._slot_ids.append(slot_id)
                self._slot_ids.sort()

            # Ensure the slot exists so gate activity can be inspected later.
            if slot_id not in env.seeds:
                env.seeds[slot_id] = SeedState(slot_id=slot_id)
        else:
            expected: dict[str, type] = {
                "SEED_GERMINATED": SeedGerminatedPayload,
                "SEED_STAGE_CHANGED": SeedStageChangedPayload,
                "SEED_GATE_EVALUATED": SeedGateEvaluatedPayload,
                "SEED_FOSSILIZED": SeedFossilizedPayload,
                "SEED_PRUNED": SeedPrunedPayload,
            }
            if event_type in expected:
                raise TypeError(
                    f"{event_type} requires {expected[event_type].__name__}, got "
                    f"{type(event.data).__name__}"
                )
            raise NotImplementedError(f"Unhandled seed event type: {event_type}")

    def _handle_batch_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle BATCH_EPOCH_COMPLETED event (episode completion)."""
        if not isinstance(event.data, BatchEpochCompletedPayload):
            raise TypeError(
                "BATCH_EPOCH_COMPLETED requires BatchEpochCompletedPayload, got "
                f"{type(event.data).__name__}"
            )

        payload = event.data

        self._current_episode = payload.episodes_completed
        self._batches_completed += 1

        # Record timestamp for instantaneous throughput calculation
        self._episode_completion_times.append((time.time(), payload.episodes_completed))

        # Capture batch-level aggregates - all required fields
        self._current_batch = payload.batch_idx
        self._batch_avg_accuracy = payload.avg_accuracy
        self._batch_rolling_accuracy = payload.rolling_accuracy
        self._batch_avg_reward = payload.avg_reward
        self._batch_total_episodes = payload.total_episodes

        # Episode Return
        self._tamiyo.current_episode_return = payload.avg_reward
        self._tamiyo.episode_return_history.append(payload.avg_reward)

        # Compute per-episode lifecycle rates for trend tracking
        if self._current_episode > 0:
            germ_rate = self._cumulative_germinated / self._current_episode
            prune_rate = self._cumulative_pruned / self._current_episode
            foss_rate = self._cumulative_fossilized / self._current_episode
            self._germination_rate_history.append(germ_rate)
            self._prune_rate_history.append(prune_rate)
            self._fossilize_rate_history.append(foss_rate)

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
                    seeds={k: replace(v) for k, v in env.best_seeds.items()},
                    slot_ids=list(self._slot_ids),  # All slots for showing DORMANT in detail
                    growth_ratio=growth_ratio,
                    record_id=str(uuid.uuid4())[:8],
                    # Full env snapshot at peak (captured by EnvState.add_accuracy())
                    reward_components=env.best_reward_components,
                    counterfactual_matrix=env.best_counterfactual_matrix,
                    shapley_snapshot=env.best_shapley_snapshot,
                    action_history=env.best_action_history,
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
        # Keep top 10 records
        self._best_runs = self._best_runs[:10]

        # Accumulate batch action counts into cumulative totals BEFORE resetting
        for env in self._envs.values():
            for action, count in env.action_counts.items():
                self._cumulative_action_counts[action] = self._cumulative_action_counts.get(action, 0) + count
            self._cumulative_total_actions += env.total_actions

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
            env.cumulative_reward = 0.0

            # Best tracking (fresh per episode)
            env.best_reward = float('-inf')
            env.best_reward_epoch = 0
            env.best_accuracy = 0.0
            env.best_accuracy_epoch = 0
            env.best_accuracy_episode = 0
            env.best_seeds.clear()

            # Volatile state snapshots (fresh per episode)
            env.best_reward_components = None
            env.best_counterfactual_matrix = None
            env.best_action_history = []

            # Action tracking (fresh distribution each episode)
            env.action_history.clear()
            env.action_counts = {
                "WAIT": 0,
                "GERMINATE": 0,
                "SET_ALPHA_TARGET": 0,
                "PRUNE": 0,
                "FOSSILIZE": 0,
                "ADVANCE": 0,
            }
            env.total_actions = 0

            # Gaming rate tracking (fresh per episode)
            env.gaming_trigger_count = 0
            env.total_reward_steps = 0

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
            raise TypeError(
                "COUNTERFACTUAL_MATRIX_COMPUTED requires CounterfactualMatrixPayload, got "
                f"{type(event.data).__name__}"
            )

        payload = event.data
        env_id = payload.env_id

        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Parse configs - seed_mask and accuracy MUST exist (created by emitter)
        configs = [
            CounterfactualConfig(
                seed_mask=tuple(cfg["seed_mask"]),
                accuracy=cfg["accuracy"],
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
            raise TypeError(
                "ANALYTICS_SNAPSHOT requires AnalyticsSnapshotPayload, got "
                f"{type(event.data).__name__}"
            )

        payload = event.data
        kind = payload.kind

        # Update observation stats if present (can be on any kind of ANALYTICS_SNAPSHOT)
        if payload.observation_stats is not None:
            obs = payload.observation_stats
            self._observation_stats = ObservationStats(
                slot_features_mean=obs.slot_features_mean,
                slot_features_std=obs.slot_features_std,
                host_features_mean=obs.host_features_mean,
                host_features_std=obs.host_features_std,
                context_features_mean=obs.context_features_mean,
                context_features_std=obs.context_features_std,
                outlier_pct=obs.outlier_pct,
                nan_count=obs.nan_count,
                inf_count=obs.inf_count,
                normalization_drift=obs.normalization_drift,
                batch_size=obs.batch_size,
            )

        # Batch-level action distribution
        if kind == "action_distribution":
            if payload.action_counts:
                counts = {str(action): int(count) for action, count in payload.action_counts.items()}
                self._tamiyo.action_counts = counts
                self._tamiyo.total_actions = sum(counts.values())
            return

        # Per-env Shapley value attribution (computed at episode boundaries)
        if kind == "shapley_computed":
            env_id = payload.env_id
            if env_id is not None and payload.shapley_values is not None:
                self._ensure_env(env_id)
                env = self._envs[env_id]

                # Convert telemetry dict to ShapleySnapshot
                values: dict[str, ShapleyEstimate] = {}
                slot_ids: list[str] = []
                for slot_id, metrics in payload.shapley_values.items():
                    slot_ids.append(slot_id)
                    values[slot_id] = ShapleyEstimate(
                        mean=metrics.get("mean", 0.0),
                        std=metrics.get("std", 0.0),
                        n_samples=int(metrics.get("n_samples", 0)),
                    )

                env.shapley_snapshot = ShapleySnapshot(
                    slot_ids=tuple(slot_ids),
                    values=values,
                    epoch=payload.batch or 0,
                    timestamp=datetime.now(timezone.utc),
                )

                # Also update best_shapley_snapshot for historical view.
                # Shapley is computed at episode end, and since envs reset each batch,
                # this IS the correct attribution data for this episode's peak accuracy.
                # We copy rather than alias to preserve the snapshot if shapley_snapshot
                # gets cleared at next episode start.
                env.best_shapley_snapshot = ShapleySnapshot(
                    slot_ids=tuple(slot_ids),
                    values=dict(values),
                    epoch=payload.batch or 0,
                    timestamp=env.shapley_snapshot.timestamp,
                )
            return

        # Per-step "last_action" snapshots - update env state and decision tracking
        if kind == "last_action" and payload.action_confidence is not None:
            env_id = payload.env_id
            if env_id is None:
                raise ValueError("ANALYTICS_SNAPSHOT(kind=last_action) missing env_id")

            self._ensure_env(env_id)
            env = self._envs[env_id]
            epoch = event.epoch if event.epoch is not None else 0

            # Update reward tracking (DRL-01 fix: explicit None check)
            total_reward = payload.total_reward if payload.total_reward is not None else 0.0
            env.reward_history.append(total_reward)
            env.cumulative_reward += total_reward
            env.current_epoch = epoch

            # Update action tracking (with normalization)
            action_name = normalize_action(payload.action_name or "UNKNOWN")
            env.action_history.append(action_name)
            env.action_counts[action_name] = env.action_counts.get(action_name, 0) + 1
            env.total_actions += 1

            # Track last action target for EnvOverview row highlighting (with timestamp for hysteresis)
            self._last_action_env_id = env_id
            self._last_action_timestamp = datetime.now(timezone.utc)

            # Update reward component breakdown from typed dataclass
            rc = payload.reward_components
            if rc is not None:
                env.reward_components.base_acc_delta = rc.base_acc_delta
                # bounded_attribution is None for LOSS family (not computed) - leave at default
                if rc.bounded_attribution is not None:
                    env.reward_components.bounded_attribution = rc.bounded_attribution
                # seed_contribution is None for non-contribution modes - leave at default
                if rc.seed_contribution is not None:
                    env.reward_components.seed_contribution = rc.seed_contribution
                env.reward_components.compute_rent = rc.compute_rent
                env.reward_components.stage_bonus = rc.stage_bonus
                env.reward_components.ratio_penalty = rc.ratio_penalty
                env.reward_components.alpha_shock = rc.alpha_shock
                env.reward_components.hindsight_credit = rc.hindsight_credit
                # Wiring fix: these fields were defined in schema but never populated
                env.reward_components.fossilize_terminal_bonus = rc.fossilize_terminal_bonus
                env.reward_components.blending_warning = rc.blending_warning
                env.reward_components.holding_warning = rc.holding_warning
                env.reward_components.total = rc.total_reward
                env.reward_components.last_action = action_name
                env.reward_components.env_id = env_id
                env.reward_components.val_acc = env.host_accuracy
                # New fields now available
                if rc.num_fossilized_seeds is not None:
                    env.reward_components.scaffold_count = rc.num_fossilized_seeds
                # avg_scaffold_delay is not in RewardComponentsTelemetry, leave as default

            # Track gaming rate (for per-env reward health)
            env.total_reward_steps += 1
            if env.reward_components.ratio_penalty != 0 or env.reward_components.alpha_shock != 0:
                env.gaming_trigger_count += 1

            # Create decision snapshot
            now_dt = event.timestamp or datetime.now(timezone.utc)
            # DRL-02 fix: explicit None check (0.0 is a valid value estimate)
            value_s = payload.value_estimate if payload.value_estimate is not None else 0.0

            # Compute TD advantage for previous decision from this env
            if env_id in self._pending_decisions:
                pending = self._pending_decisions[env_id]
                td_adv = pending.reward + DEFAULT_GAMMA * value_s - pending.value_s
                pending.decision.td_advantage = td_adv
                del self._pending_decisions[env_id]

            # Map tempo index to tempo name (FAST/STANDARD/SLOW)
            chosen_tempo: str | None = None
            if payload.tempo_idx is not None and 0 <= payload.tempo_idx < len(TEMPO_NAMES):
                chosen_tempo = TEMPO_NAMES[payload.tempo_idx]

            # Map alpha_target float to enum name (HALF/SEVENTY/FULL)
            chosen_alpha_target: str | None = None
            if payload.alpha_target is not None:
                # MED-05 fix: Validate alpha_target is in map, log warning if not
                alpha_target_map = {0.5: "HALF", 0.7: "SEVENTY", 1.0: "FULL"}
                chosen_alpha_target = alpha_target_map.get(payload.alpha_target)
                if chosen_alpha_target is None:
                    _logger.warning("Unknown alpha_target value: %s (expected 0.5, 0.7, or 1.0)", payload.alpha_target)

            # Extract HeadTelemetry if present
            ht = payload.head_telemetry

            decision = DecisionSnapshot(
                timestamp=now_dt,
                slot_states=payload.slot_states or {},
                host_accuracy=env.host_accuracy,
                chosen_action=action_name,
                chosen_slot=payload.slot_id,
                confidence=payload.action_confidence,
                expected_value=value_s,
                actual_reward=total_reward,
                alternatives=payload.alternatives or [],
                decision_id=str(uuid.uuid4())[:8],
                decision_entropy=payload.decision_entropy or 0.0,
                env_id=env_id,
                epoch=self._current_epoch,
                batch=self._current_batch,
                value_residual=total_reward - value_s,
                td_advantage=None,
                # Head choice fields (from AnalyticsSnapshotPayload)
                chosen_blueprint=payload.blueprint_id,
                chosen_tempo=chosen_tempo,
                chosen_style=payload.style,
                chosen_curve=payload.alpha_curve,
                chosen_alpha_target=chosen_alpha_target,
                chosen_alpha_speed=payload.alpha_speed,
                # Per-head confidence values (from HeadTelemetry)
                op_confidence=ht.op_confidence if ht else 0.0,
                slot_confidence=ht.slot_confidence if ht else 0.0,
                blueprint_confidence=ht.blueprint_confidence if ht else 0.0,
                style_confidence=ht.style_confidence if ht else 0.0,
                tempo_confidence=ht.tempo_confidence if ht else 0.0,
                alpha_target_confidence=ht.alpha_target_confidence if ht else 0.0,
                alpha_speed_confidence=ht.alpha_speed_confidence if ht else 0.0,
                curve_confidence=ht.curve_confidence if ht else 0.0,
                # Per-head entropy values (from HeadTelemetry)
                op_entropy=ht.op_entropy if ht else 0.0,
                slot_entropy=ht.slot_entropy if ht else 0.0,
                blueprint_entropy=ht.blueprint_entropy if ht else 0.0,
                style_entropy=ht.style_entropy if ht else 0.0,
                tempo_entropy=ht.tempo_entropy if ht else 0.0,
                alpha_target_entropy=ht.alpha_target_entropy if ht else 0.0,
                alpha_speed_entropy=ht.alpha_speed_entropy if ht else 0.0,
                curve_entropy=ht.curve_entropy if ht else 0.0,
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

            # Update last action tracking for Sequence section display
            self._tamiyo.last_action_op = action_name
            if payload.action_success is not None:
                self._tamiyo.last_action_success = payload.action_success

            return

    def _handle_episode_outcome(self, event: "TelemetryEvent") -> None:
        """Handle incoming episode outcome events."""
        from esper.leyline import EpisodeOutcome

        data = event.data
        if data is None:
            raise TypeError("EPISODE_OUTCOME event missing data payload")

        if not isinstance(data, EpisodeOutcomePayload):
            raise TypeError(
                "EPISODE_OUTCOME requires EpisodeOutcomePayload, got "
                f"{type(data).__name__}"
            )

        outcome = EpisodeOutcome(
            env_id=data.env_id,
            episode_idx=data.episode_idx,
            final_accuracy=data.final_accuracy,
            param_ratio=data.param_ratio,
            num_fossilized=data.num_fossilized,
            num_contributing_fossilized=data.num_contributing_fossilized,
            episode_reward=data.episode_reward,
            stability_score=data.stability_score,
            reward_mode=data.reward_mode,
        )

        self._episode_outcomes.append(outcome)

        # Keep only last 100 outcomes to bound memory
        if len(self._episode_outcomes) > 100:
            self._episode_outcomes = self._episode_outcomes[-100:]

        # TELE-610: Track episode diagnostics
        self._episode_lengths.append(data.episode_length)
        is_success = data.outcome_type == "success"
        self._recent_outcomes.append(is_success)
        if is_success:
            self._success_count += 1
        elif data.outcome_type == "timeout":
            self._timeout_count += 1
        self._total_germinate += data.germinate_count
        self._total_prune += data.prune_count
        self._total_fossilize += data.fossilize_count

    def _handle_governor_rollback(self, event: "TelemetryEvent") -> None:
        """Handle GOVERNOR_ROLLBACK event - catastrophic failure indicator.

        Sets rollback state on the affected env, which triggers a red alert
        overlay in the env row widget. The flag is cleared when the next
        EPOCH_COMPLETED event arrives for that env (training resumed).
        """
        if not isinstance(event.data, GovernorRollbackPayload):
            raise TypeError(
                "GOVERNOR_ROLLBACK requires GovernorRollbackPayload, got "
                f"{type(event.data).__name__}"
            )

        payload = event.data
        env_id = payload.env_id
        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Set rollback state - will show red alert until training resumes
        env.rolled_back = True
        env.rollback_reason = payload.reason
        env.rollback_timestamp = event.timestamp or datetime.now(timezone.utc)

        _logger.info(
            "Governor rollback for env %d: %s",
            env_id,
            payload.reason,
        )

    def _compute_hypervolume(self) -> float:
        """Compute hypervolume indicator from recent episode outcomes."""
        if not self._episode_outcomes:
            return 0.0

        frontier = extract_pareto_frontier(self._episode_outcomes)
        ref_point = (0.0, 1.0)  # (min_accuracy, max_param_ratio)
        return compute_hypervolume_2d(frontier, ref_point)

    def compute_reward_health(self) -> RewardHealthData:
        """Compute reward health metrics from recent telemetry."""
        with self._lock:
            # Collect latest reward components from all envs
            components = [
                env.reward_components for env in self._envs.values()
                if env.reward_components and env.reward_components.total != 0
            ]

            if not components:
                return RewardHealthData()

            # PBRS proxy: stage_bonus is the PBRS shaping reward
            pbrs_total = sum(abs(c.stage_bonus) for c in components)
            reward_total = sum(abs(c.total) for c in components)
            pbrs_fraction = pbrs_total / max(1e-8, reward_total)

            # Anti-gaming trigger rate
            gaming_steps = sum(
                1 for c in components
                if c.ratio_penalty != 0 or c.alpha_shock != 0
            )
            gaming_rate = gaming_steps / max(1, len(components))

            # Get latest EV from Tamiyo PPO state
            ev = self._tamiyo.explained_variance

            # Hypervolume from episode outcomes
            hv = self._compute_hypervolume()

            return RewardHealthData(
                pbrs_fraction=pbrs_fraction,
                anti_gaming_trigger_rate=gaming_rate,
                ev_explained=ev,
                hypervolume=hv,
            )

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
        metadata: dict[str, str | int | float] = {"event_id": event.event_id}
        env_id: int | None = None

        if event_type.startswith("SEED_"):
            if event_type == "SEED_GERMINATED" and isinstance(event.data, SeedGerminatedPayload):
                message = "Germinated"
                # Event envelope slot_id is authoritative (set at emission time)
                slot_id = event.slot_id if event.slot_id else event.data.slot_id
                metadata["slot_id"] = slot_id
                metadata["blueprint"] = event.data.blueprint_id
                env_id = event.data.env_id
            elif event_type == "SEED_GATE_EVALUATED" and isinstance(event.data, SeedGateEvaluatedPayload):
                message = "Gate evaluated"
                slot_id = event.slot_id if event.slot_id else event.data.slot_id
                metadata["slot_id"] = slot_id
                metadata["gate"] = event.data.gate
                metadata["target_stage"] = event.data.target_stage
                metadata["result"] = "PASS" if event.data.passed else "FAIL"
                if event.data.checks_failed:
                    metadata["failed_checks"] = len(event.data.checks_failed)
                if event.data.message:
                    metadata["detail"] = event.data.message
                env_id = event.data.env_id
            elif event_type == "SEED_STAGE_CHANGED" and isinstance(event.data, SeedStageChangedPayload):
                message = "Stage changed"
                # Event envelope slot_id is authoritative (set at emission time)
                slot_id = event.slot_id if event.slot_id else event.data.slot_id
                metadata["slot_id"] = slot_id
                metadata["from"] = event.data.from_stage
                metadata["to"] = event.data.to_stage
                env_id = event.data.env_id
            elif event_type == "SEED_FOSSILIZED" and isinstance(event.data, SeedFossilizedPayload):
                message = "Fossilized"
                # Event envelope slot_id is authoritative (set at emission time)
                slot_id = event.slot_id if event.slot_id else event.data.slot_id
                metadata["slot_id"] = slot_id
                metadata["improvement"] = event.data.improvement
                env_id = event.data.env_id
            elif event_type == "SEED_PRUNED" and isinstance(event.data, SeedPrunedPayload):
                message = "Pruned"
                # Event envelope slot_id is authoritative (set at emission time)
                slot_id = event.slot_id if event.slot_id else event.data.slot_id
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
        except Exception as e:
            _logger.warning("Failed to collect CPU vitals: %s", e)
            self._vitals.cpu_percent = None  # Explicit unavailable state

        # RAM
        try:
            mem = psutil.virtual_memory()
            self._vitals.ram_used_gb = mem.used / (1024**3)
            self._vitals.ram_total_gb = mem.total / (1024**3)
        except Exception as e:
            _logger.warning("Failed to collect RAM vitals: %s", e)
            self._vitals.ram_used_gb = None  # Explicit unavailable state
            self._vitals.ram_total_gb = None  # Explicit unavailable state

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
                            if device_idx in self._gpu_total_memory_gb:
                                mem_total = self._gpu_total_memory_gb[device_idx]
                            else:
                                props = torch.cuda.get_device_properties(device_idx)
                                mem_total = props.total_memory / (1024**3)
                                self._gpu_total_memory_gb[device_idx] = mem_total

                            gpu_stats[device_idx] = GPUStats(
                                device_id=device_idx,
                                memory_used_gb=mem_reserved,
                                memory_total_gb=mem_total,
                                # Only set when actual utilization is available (e.g., via NVML).
                                utilization=0.0,
                            )
                        except Exception as e:
                            _logger.warning("Failed to collect GPU stats for device %d: %s", device_idx, e)
                self._vitals.gpu_stats = gpu_stats
                if 0 in gpu_stats:
                    stats0 = gpu_stats[0]
                    self._vitals.gpu_memory_used_gb = stats0.memory_used_gb
                    self._vitals.gpu_memory_total_gb = stats0.memory_total_gb
                    self._vitals.gpu_utilization = stats0.utilization
                    self._vitals.gpu_temperature = stats0.temperature
        except ImportError as e:
            _logger.warning("Failed to import torch for GPU vitals: %s", e)
            self._vitals.gpu_stats = {}  # Explicit unavailable state
