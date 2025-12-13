"""Karn TUI - Terminal User Interface for PPO Training Monitoring.

Rich-based live TUI that displays PPO training metrics in a structured dashboard.
Implements the Nissa OutputBackend protocol for seamless integration with the
telemetry hub.

Usage:
    from esper.karn import TUIOutput

    hub = get_hub()
    tui = TUIOutput()
    hub.add_backend(tui)

    # TUI displays live metrics during training
    # Press 'q' to quit (when running standalone)
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import BarColumn, Progress, TextColumn

if TYPE_CHECKING:
    from esper.leyline.telemetry import TelemetryEvent

_logger = logging.getLogger(__name__)


# =============================================================================
# Health Status Thresholds (from DRL Expert requirements)
# =============================================================================

class HealthStatus(Enum):
    """Health status for color coding."""
    OK = "green"
    WARNING = "yellow"
    CRITICAL = "red"


@dataclass
class ThresholdConfig:
    """Thresholds for red flag detection."""

    # Entropy thresholds (healthy starts near ln(4) ≈ 1.39)
    entropy_critical: float = 0.3
    entropy_warning: float = 0.5
    entropy_max: float = 1.39  # ln(4) for 4 actions

    # Clip fraction thresholds (target 0.1-0.2)
    clip_critical: float = 0.3
    clip_warning: float = 0.25

    # Explained variance (value learning quality)
    explained_var_critical: float = 0.5
    explained_var_warning: float = 0.7

    # Gradient norm
    grad_norm_critical: float = 10.0
    grad_norm_warning: float = 5.0

    # KL divergence (policy change magnitude)
    kl_warning: float = 0.05

    # Action distribution (WAIT dominance)
    wait_warning: float = 0.7  # > 70% WAIT is suspicious


# =============================================================================
# Metric Tracking State
# =============================================================================

@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode: int = 0
    reward: float = 0.0
    final_accuracy: float = 0.0
    episode_length: int = 0


@dataclass
class SeedState:
    """Current state of a seed slot."""
    slot_id: str = ""
    stage: str = "DORMANT"
    blueprint_id: str | None = None
    alpha: float = 0.0
    accuracy_delta: float = 0.0


def _make_sparkline_static(values: list[float], width: int = 8) -> str:
    """Create a sparkline from values (static helper)."""
    if not values:
        return "─" * width

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1.0

    blocks = "▁▂▃▄▅▆▇█"
    values = values[-width:]

    result = ""
    for v in values:
        normalized = (v - min_val) / range_val
        idx = min(int(normalized * (len(blocks) - 1)), len(blocks) - 1)
        result += blocks[idx]

    return result.ljust(width, "─")


@dataclass
class EnvState:
    """Per-environment state for multi-env tracking."""
    env_id: int = 0
    current_epoch: int = 0
    host_accuracy: float = 0.0
    host_loss: float = 0.0
    seeds: dict[str, SeedState] = field(default_factory=dict)
    active_seed_count: int = 0
    fossilized_count: int = 0
    culled_count: int = 0

    # Per-env reward tracking
    reward_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    best_reward: float = float('-inf')
    best_reward_epoch: int = 0

    # Per-env accuracy tracking
    accuracy_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    best_accuracy: float = 0.0
    best_accuracy_epoch: int = 0

    # Per-env action tracking
    action_history: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    action_counts: dict[str, int] = field(default_factory=lambda: {
        "WAIT": 0, "GERMINATE": 0, "CULL": 0, "FOSSILIZE": 0
    })
    total_actions: int = 0

    # Status tracking
    status: str = "initializing"
    last_update: datetime | None = None
    epochs_since_improvement: int = 0

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
    def reward_sparkline(self) -> str:
        """Generate sparkline from reward history."""
        return _make_sparkline_static(list(self.reward_history), width=8)

    @property
    def accuracy_sparkline(self) -> str:
        """Generate sparkline from accuracy history."""
        return _make_sparkline_static(list(self.accuracy_history), width=8)

    def add_reward(self, reward: float, epoch: int) -> None:
        """Add reward and update best tracking."""
        self.reward_history.append(reward)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_reward_epoch = epoch

    def add_accuracy(self, accuracy: float, epoch: int) -> None:
        """Add accuracy and update best/status tracking."""
        prev_acc = self.accuracy_history[-1] if self.accuracy_history else 0.0
        self.accuracy_history.append(accuracy)
        self.host_accuracy = accuracy

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_accuracy_epoch = epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        self._update_status(prev_acc, accuracy)

    def add_action(self, action_name: str) -> None:
        """Track action taken."""
        self.action_history.append(action_name)
        if action_name in self.action_counts:
            self.action_counts[action_name] += 1
            self.total_actions += 1

    def _update_status(self, prev_acc: float, curr_acc: float) -> None:
        """Update env status based on metrics."""
        if self.epochs_since_improvement > 10:
            self.status = "stalled"
        elif curr_acc < prev_acc - 1.0:
            self.status = "degraded"
        elif curr_acc > 80.0:
            self.status = "excellent"
        elif self.current_epoch > 0:
            self.status = "healthy"


@dataclass
class TUIState:
    """Thread-safe state for the TUI display."""

    # Episode tracking
    current_episode: int = 0
    current_epoch: int = 0

    # Reward metrics
    current_reward: float = 0.0
    reward_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    best_reward: float = float('-inf')
    best_episode: int = 0

    # Host metrics (aggregated across all envs)
    host_accuracy: float = 0.0
    host_accuracy_delta: float = 0.0
    host_loss: float = 0.0
    best_accuracy: float = 0.0  # Best accuracy seen across all envs
    best_accuracy_episode: int = 0  # Episode when best was achieved

    # Policy health (P0 Critical)
    entropy: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    kl_divergence: float = 0.0

    # Losses
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy_loss: float = 0.0
    grad_norm: float = 0.0

    # Action distribution
    action_counts: dict[str, int] = field(default_factory=lambda: {
        "WAIT": 0, "GERMINATE": 0, "CULL": 0, "FOSSILIZE": 0
    })
    total_actions: int = 0

    # Reward components
    reward_components: dict[str, float] = field(default_factory=dict)

    # Event log for TUI display (P1 - TUI/Nissa integration)
    event_log: deque[tuple[str, str, str]] = field(
        default_factory=lambda: deque(maxlen=100)
    )  # (timestamp, event_type, formatted_message)
    event_log_min_severity: str = "info"

    # Per-environment state (for multi-env vectorized training)
    env_states: dict[int, EnvState] = field(default_factory=dict)
    n_envs: int = 1  # Number of environments (updated from TRAINING_STARTED)

    # Aggregate seed stats (summed across all envs)
    active_seed_count: int = 0
    fossilized_count: int = 0
    culled_count: int = 0

    # Advantage stats
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0

    # Episode history for sparkline
    episode_rewards: deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Red flags
    reward_hacking_detected: bool = False

    # Performance metrics
    start_time: datetime | None = None
    last_batch_time: datetime | None = None
    batches_completed: int = 0
    epochs_completed: int = 0
    epochs_per_second: float = 0.0
    batches_per_hour: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0

    # Lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update_reward(self, reward: float) -> None:
        """Thread-safe reward update."""
        with self._lock:
            self.current_reward = reward
            self.reward_history.append(reward)
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_episode = self.current_episode

    @property
    def mean_reward_100(self) -> float:
        """Mean reward over last 100 episodes."""
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    @property
    def std_reward_100(self) -> float:
        """Std of reward over last 100 episodes."""
        if len(self.reward_history) < 2:
            return 0.0
        mean = self.mean_reward_100
        variance = sum((r - mean) ** 2 for r in self.reward_history) / len(self.reward_history)
        return variance ** 0.5

    def get_action_percentages(self) -> dict[str, float]:
        """Get action distribution as percentages."""
        if self.total_actions == 0:
            return {k: 0.0 for k in self.action_counts}
        return {k: v / self.total_actions * 100 for k, v in self.action_counts.items()}

    def get_or_create_env(self, env_id: int) -> EnvState:
        """Get or create an environment state."""
        if env_id not in self.env_states:
            self.env_states[env_id] = EnvState(env_id=env_id)
        return self.env_states[env_id]

    def update_aggregate_seed_counts(self) -> None:
        """Recalculate aggregate seed counts from all envs."""
        self.active_seed_count = sum(e.active_seed_count for e in self.env_states.values())
        self.fossilized_count = sum(e.fossilized_count for e in self.env_states.values())
        self.culled_count = sum(e.culled_count for e in self.env_states.values())

    # =========================================================================
    # Aggregate Properties (cross-env calculations)
    # =========================================================================

    # Track which env last emitted reward (for detail view focus)
    last_reward_env_id: int = 0

    @property
    def aggregate_mean_reward(self) -> float:
        """Mean of current rewards across all envs."""
        if not self.env_states:
            return 0.0
        rewards = [e.current_reward for e in self.env_states.values()]
        return sum(rewards) / len(rewards) if rewards else 0.0

    @property
    def aggregate_mean_accuracy(self) -> float:
        """Mean of current accuracies across all envs."""
        if not self.env_states:
            return 0.0
        accs = [e.host_accuracy for e in self.env_states.values()]
        return sum(accs) / len(accs) if accs else 0.0

    @property
    def aggregate_best_accuracy(self) -> tuple[float, int, int]:
        """Best accuracy across all envs: (accuracy, env_id, epoch)."""
        if not self.env_states:
            return (0.0, -1, 0)
        best_env = max(self.env_states.values(), key=lambda e: e.best_accuracy)
        return (best_env.best_accuracy, best_env.env_id, best_env.best_accuracy_epoch)

    @property
    def aggregate_action_counts(self) -> dict[str, int]:
        """Sum action counts across all envs."""
        totals: dict[str, int] = {"WAIT": 0, "GERMINATE": 0, "CULL": 0, "FOSSILIZE": 0}
        for env in self.env_states.values():
            for action, count in env.action_counts.items():
                totals[action] = totals.get(action, 0) + count
        return totals

    @property
    def aggregate_total_actions(self) -> int:
        """Total actions across all envs."""
        return sum(e.total_actions for e in self.env_states.values())

    @property
    def envs_by_status(self) -> dict[str, list[int]]:
        """Group env IDs by status."""
        by_status: dict[str, list[int]] = {}
        for env_id, env in self.env_states.items():
            if env.status not in by_status:
                by_status[env.status] = []
            by_status[env.status].append(env_id)
        return by_status


# =============================================================================
# TUI Output Backend
# =============================================================================

class TUIOutput:
    """Rich-based TUI output backend for PPO training monitoring.

    Implements the Nissa OutputBackend protocol (emit, close) and displays
    live metrics in a structured terminal dashboard.

    Args:
        thresholds: Health status thresholds for color coding.
        force_layout: Force a specific layout mode ('compact', 'standard', 'wide')
                     instead of auto-detecting from terminal width. Currently unused
                     but reserved for future multi-layout support.
    """

    def __init__(
        self,
        thresholds: ThresholdConfig | None = None,
        force_layout: str | None = None,
    ):
        self.thresholds = thresholds or ThresholdConfig()
        self.state = TUIState()
        self.console = Console()
        self._live: Live | None = None
        self._started = False
        self._lock = threading.Lock()
        self._force_layout = force_layout  # Reserved for future layout modes

    def start(self) -> None:
        """Start the live display."""
        if self._started:
            return

        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=4,
            screen=False,  # Don't use alternate screen - allows scrollback
        )
        self._live.start()
        self._started = True

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None
        self._started = False

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit a telemetry event to update the TUI.

        Thread-safe - can be called from training loop.
        All events are logged to the event log panel; specific events also update metrics.
        """
        # Auto-start on first event
        if not self._started:
            self.start()

        # Get event type as string
        # hasattr AUTHORIZED by John on 2025-12-14 04:00:00 UTC
        # Justification: Serialization - handle both enum and string event_type values
        event_type = (
            event.event_type.name
            if hasattr(event.event_type, "name")
            else str(event.event_type)
        )

        # Buffer ALL events to the event log (P1 - TUI/Nissa integration)
        formatted = self._format_event_for_log(event)
        if formatted:
            self.state.event_log.append(formatted)

        # Route to appropriate handler for metrics updates
        with self._lock:
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

        # Update the display
        if self._live:
            self._live.update(self._render())

    def close(self) -> None:
        """Close the TUI backend."""
        self.stop()

    # =========================================================================
    # Event Log Formatting (P1 - TUI/Nissa integration)
    # =========================================================================

    _SEVERITY_ORDER = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}

    def _format_event_for_log(self, event: "TelemetryEvent") -> tuple[str, str, str] | None:
        """Format event for log display, returns (timestamp, event_type, message) or None if filtered."""

        # Filter by severity
        severity = getattr(event, "severity", "info")
        if self._SEVERITY_ORDER.get(severity, 1) < self._SEVERITY_ORDER.get(self.state.event_log_min_severity, 1):
            return None

        timestamp = event.timestamp.strftime("%H:%M:%S") if event.timestamp else datetime.now().strftime("%H:%M:%S")

        # hasattr AUTHORIZED by John on 2025-12-14 12:00:00 UTC
        # Justification: Serialization - handle both enum and string event_type values
        event_type = (
            event.event_type.name
            if hasattr(event.event_type, "name")
            else str(event.event_type)
        )

        data = event.data or {}

        # Format message based on event type
        if "EPOCH" in event_type:
            loss = data.get("val_loss", "?")
            acc = data.get("val_accuracy", "?")
            epoch = event.epoch or data.get("epoch", "?")
            if isinstance(loss, float) and isinstance(acc, float):
                msg = f"epoch={epoch} loss={loss:.4f} acc={acc:.2%}"
            else:
                msg = f"epoch={epoch}"
        elif event_type == "REWARD_COMPUTED":
            action = data.get("action_name", "?")
            total = data.get("total_reward", 0.0)
            msg = f"{action} r={total:+.3f}"
        elif event_type.startswith("SEED_"):
            seed_id = event.seed_id or "?"
            if event_type == "SEED_GERMINATED":
                blueprint_id = data.get("blueprint_id", "?")
                msg = f"{seed_id} germinated ({blueprint_id})"
            elif event_type == "SEED_STAGE_CHANGED":
                from_stage = data.get("from", "?")
                to_stage = data.get("to", "?")
                msg = f"{seed_id} {from_stage}->{to_stage}"
            elif event_type == "SEED_FOSSILIZED":
                improvement = data.get("improvement", 0)
                msg = f"{seed_id} fossilized (+{improvement:.2f}%)"
            elif event_type == "SEED_CULLED":
                reason = data.get("reason", "")
                msg = f"{seed_id} culled" + (f" ({reason})" if reason else "")
            else:
                msg = f"{seed_id}"
        elif event_type == "PPO_UPDATE_COMPLETED":
            if data.get("skipped"):
                msg = "skipped (buffer rollback)"
            else:
                entropy = data.get("entropy", 0.0)
                clip = data.get("clip_fraction", 0.0)
                msg = f"ent={entropy:.3f} clip={clip:.3f}"
        elif event_type == "CHECKPOINT_SAVED":
            path = data.get("path", "?")
            msg = f"saved {Path(path).name}" if path != "?" else "saved"
        elif event_type == "CHECKPOINT_LOADED":
            path = data.get("path", "?")
            msg = f"loaded {Path(path).name}" if path != "?" else "loaded"
        elif event_type.startswith("GOVERNOR_"):
            detail = data.get("detail", event_type)
            msg = str(detail)
        elif event_type == "TRAINING_STARTED":
            task = data.get("task", "?")
            max_epochs = data.get("max_epochs", "?")
            msg = f"task={task} epochs={max_epochs}"
        elif event_type == "BATCH_COMPLETED":
            batch = data.get("batch", "?")
            msg = f"batch {batch} complete"
        else:
            msg = event.message or ""

        return (timestamp, event_type, msg)

    def _render_event_log(self, max_lines: int = 12) -> Panel:
        """Render the event log panel."""
        # Get recent events
        events = list(self.state.event_log)[-max_lines:]

        if not events:
            content = Text("Waiting for events...", style="dim")
        else:
            lines = []
            for timestamp, event_type, msg in events:
                # Color code by event type
                if "ERROR" in event_type or "PANIC" in event_type or "COLLAPSE" in event_type:
                    style = "red"
                elif "WARNING" in event_type or "ROLLBACK" in event_type:
                    style = "yellow"
                elif "FOSSILIZED" in event_type:
                    style = "green"
                elif "CULLED" in event_type:
                    style = "red dim"
                elif "GERMINATED" in event_type:
                    style = "cyan"
                elif "CHECKPOINT" in event_type:
                    style = "blue"
                else:
                    style = "white"

                # Truncate message if too long
                max_msg_len = 50
                if len(msg) > max_msg_len:
                    msg = msg[:max_msg_len-3] + "..."

                line = Text()
                line.append(f"{timestamp} ", style="dim")
                line.append(f"{event_type:<25} ", style=style)
                line.append(msg, style="white")
                lines.append(line)

            content = Group(*lines)

        return Panel(
            content,
            title="[bold]Event Log[/bold]",
            border_style="cyan",
        )

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _handle_training_started(self, event: "TelemetryEvent") -> None:
        """Handle TRAINING_STARTED event."""
        data = event.data or {}
        self.state.current_episode = data.get("episode", 0)
        self.state.current_epoch = 0
        self.state.n_envs = data.get("n_envs", 1)

        # Initialize performance tracking
        self.state.start_time = datetime.now()
        self.state.batches_completed = 0
        self.state.best_accuracy = 0.0
        self.state.best_accuracy_episode = 0

        # Pre-create env states
        for i in range(self.state.n_envs):
            self.state.get_or_create_env(i)

        # Initial system stats
        self._update_system_stats()

    def _handle_epoch_completed(self, event: "TelemetryEvent") -> None:
        """Handle EPOCH_COMPLETED event."""
        data = event.data or {}
        self.state.current_epoch = event.epoch or data.get("epoch", 0)
        self.state.host_accuracy = data.get("val_accuracy", 0.0)
        self.state.host_loss = data.get("val_loss", 0.0)

        # Calculate accuracy delta from train vs val
        train_acc = data.get("train_accuracy", 0.0)
        self.state.host_accuracy_delta = self.state.host_accuracy - train_acc

    def _handle_ppo_update(self, event: "TelemetryEvent") -> None:
        """Handle PPO_UPDATE_COMPLETED event."""
        data = event.data or {}

        if data.get("skipped"):
            return

        # P0 Critical metrics
        self.state.entropy = data.get("entropy", 0.0)
        self.state.clip_fraction = data.get("clip_fraction", 0.0)
        self.state.explained_variance = data.get("explained_variance", 0.0)
        self.state.kl_divergence = data.get("kl_divergence", 0.0)

        # Losses
        self.state.policy_loss = data.get("policy_loss", 0.0)
        self.state.value_loss = data.get("value_loss", 0.0)
        self.state.entropy_loss = data.get("entropy_loss", 0.0)
        self.state.grad_norm = data.get("grad_norm", 0.0)

        # Advantage stats
        self.state.advantage_mean = data.get("advantage_mean", 0.0)
        self.state.advantage_std = data.get("advantage_std", 0.0)
        self.state.advantage_min = data.get("advantage_min", 0.0)
        self.state.advantage_max = data.get("advantage_max", 0.0)

    def _handle_reward_computed(self, event: "TelemetryEvent") -> None:
        """Handle REWARD_COMPUTED event with per-env routing."""
        data = event.data or {}
        env_id = data.get("env_id", 0)
        epoch = event.epoch or 0

        # Get or create per-env state
        env_state = self.state.get_or_create_env(env_id)

        # Update per-env reward tracking
        total_reward = data.get("total_reward", 0.0)
        env_state.add_reward(total_reward, epoch)

        # Update per-env action tracking
        action_name = data.get("action_name", "WAIT")
        env_state.add_action(action_name)

        # Update per-env accuracy if provided
        val_acc = data.get("val_acc")
        if val_acc is not None:
            env_state.add_accuracy(val_acc, epoch)

        env_state.current_epoch = epoch
        env_state.last_update = datetime.now()

        # Update global state for header display
        self.state.reward_components = {
            "accuracy_delta": data.get("base_acc_delta", 0.0),
            "bounded_attr": data.get("bounded_attribution", 0.0),
            "compute_rent": data.get("compute_rent", 0.0),
            "blending_warn": data.get("blending_warning", 0.0),
            "probation_warn": data.get("probation_warning", 0.0),
            "terminal_bonus": data.get("fossilize_terminal_bonus", 0.0),
        }
        self.state.last_reward_env_id = env_id

        # Red flag: reward hacking detection
        acc_delta = data.get("base_acc_delta", 0.0)
        if acc_delta < 0 and total_reward > 0:
            self.state.reward_hacking_detected = True

        # Update global metrics for backward compatibility
        self.state.current_reward = total_reward
        self.state.host_accuracy = data.get("val_acc", self.state.host_accuracy)

    def _handle_seed_event(self, event: "TelemetryEvent", event_type: str) -> None:
        """Handle seed lifecycle events with per-env tracking."""
        data = event.data or {}
        slot_id = event.slot_id or data.get("slot_id", "unknown")
        env_id = data.get("env_id", 0)

        # Get or create env state
        env_state = self.state.get_or_create_env(env_id)

        # Get or create seed state within this env
        if slot_id not in env_state.seeds:
            env_state.seeds[slot_id] = SeedState(slot_id=slot_id)

        seed = env_state.seeds[slot_id]

        if event_type == "SEED_GERMINATED":
            seed.stage = "GERMINATED"
            seed.blueprint_id = data.get("blueprint_id")
            env_state.active_seed_count += 1
        elif event_type == "SEED_STAGE_CHANGED":
            seed.stage = data.get("to", seed.stage)
            seed.alpha = data.get("alpha", seed.alpha)
        elif event_type == "SEED_FOSSILIZED":
            seed.stage = "FOSSILIZED"
            env_state.fossilized_count += 1
            env_state.active_seed_count = max(0, env_state.active_seed_count - 1)
        elif event_type == "SEED_CULLED":
            seed.stage = "CULLED"
            env_state.culled_count += 1
            env_state.active_seed_count = max(0, env_state.active_seed_count - 1)

        # Update aggregate counts
        self.state.update_aggregate_seed_counts()

    def _handle_batch_completed(self, event: "TelemetryEvent") -> None:
        """Handle BATCH_COMPLETED event (episode completion)."""
        data = event.data or {}
        self.state.current_episode = data.get("episodes_completed", self.state.current_episode)
        self.state.batches_completed += 1

        # Track best accuracy across all envs
        current_acc = data.get("rolling_avg_accuracy", data.get("avg_accuracy", 0.0))
        self.state.host_accuracy = current_acc
        if current_acc > self.state.best_accuracy:
            self.state.best_accuracy = current_acc
            self.state.best_accuracy_episode = self.state.current_episode

        # Calculate throughput
        now = datetime.now()
        if self.state.start_time:
            elapsed = (now - self.state.start_time).total_seconds()
            if elapsed > 0:
                # epochs_completed = episodes * epochs_per_episode
                total_epochs = data.get("total_epochs", self.state.batches_completed * 75)
                self.state.epochs_per_second = total_epochs / elapsed
                self.state.batches_per_hour = (self.state.batches_completed / elapsed) * 3600

        self.state.last_batch_time = now

        # Add to episode rewards for sparkline
        avg_reward = data.get("avg_reward", 0.0)
        self.state.episode_rewards.append(avg_reward)

        # Update GPU stats if available
        self._update_system_stats()

    # =========================================================================
    # Rendering
    # =========================================================================

    def _render(self) -> Layout:
        """Render the full TUI layout with event log.

        Returns Layout directly (not wrapped in Panel) to avoid
        off-by-one height issues from extra border lines.
        """
        layout = Layout()

        # Create main sections with event log and performance stats
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="bottom", size=12),
            Layout(name="footer", size=3),
        )

        # Split bottom into event log (left) and performance stats (right)
        layout["bottom"].split_row(
            Layout(name="event_log", ratio=2),
            Layout(name="perf_stats", ratio=1),
        )

        # Split main into two columns
        layout["main"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        # Left column: rewards and seeds
        layout["left"].split_column(
            Layout(name="rewards", ratio=1),
            Layout(name="seeds", ratio=1),
            Layout(name="reward_components", ratio=1),
        )

        # Right column: policy health and actions
        layout["right"].split_column(
            Layout(name="policy_health", ratio=1),
            Layout(name="actions", ratio=1),
            Layout(name="losses", ratio=1),
        )

        # Render each section
        layout["header"].update(self._render_header())
        layout["rewards"].update(self._render_rewards())
        layout["seeds"].update(self._render_seeds())
        layout["reward_components"].update(self._render_reward_components())
        layout["policy_health"].update(self._render_policy_health())
        layout["actions"].update(self._render_actions())
        layout["losses"].update(self._render_losses())
        layout["event_log"].update(self._render_event_log(max_lines=8))
        layout["perf_stats"].update(self._render_performance())
        layout["footer"].update(self._render_footer())

        return layout

    def _render_header(self) -> Panel:
        """Render the header with episode info and multi-env summary."""
        text = Text()
        text.append(f"Episode: ", style="dim")
        text.append(f"{self.state.current_episode}", style="bold cyan")
        text.append(f"  |  Batches: ", style="dim")
        text.append(f"{self.state.batches_completed}", style="bold cyan")
        text.append(f"  |  Best Acc: ", style="dim")
        text.append(f"{self.state.best_accuracy:.1f}%", style="bold green")
        text.append(f" (ep {self.state.best_accuracy_episode})", style="dim")
        text.append(f"  |  Current: ", style="dim")
        text.append(f"{self.state.host_accuracy:.1f}%", style="cyan")

        # Add reward hacking warning
        if self.state.reward_hacking_detected:
            text.append("  |  ", style="dim")
            text.append("REWARD HACKING SUSPECTED", style="bold red blink")

        return Panel(
            text,
            title="[bold blue]ESPER-LITE PPO Training Monitor[/bold blue]",
            border_style="blue",
        )

    def _render_rewards(self) -> Panel:
        """Render the rewards panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        # Current reward
        table.add_row("Current:", f"{self.state.current_reward:+.2f}")

        # Mean and std
        mean = self.state.mean_reward_100
        std = self.state.std_reward_100
        table.add_row(
            "Mean(100):",
            f"{mean:+.2f}  Std: {std:.2f}"
        )

        # Best
        table.add_row(
            "Best:",
            f"{self.state.best_reward:+.2f} (ep {self.state.best_episode})"
        )

        # Sparkline of recent rewards
        sparkline = self._make_sparkline(list(self.state.episode_rewards))
        table.add_row("History:", sparkline)

        return Panel(table, title="[bold]REWARDS[/bold]", border_style="cyan")

    def _render_policy_health(self) -> Panel:
        """Render the policy health panel with status indicators."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_column("Bar", width=12)
        table.add_column("Status", justify="center")

        # Entropy
        entropy_pct = min(self.state.entropy / self.thresholds.entropy_max, 1.0) * 100
        entropy_status = self._get_entropy_status(self.state.entropy)
        table.add_row(
            "Entropy:",
            f"{self.state.entropy:.2f}",
            self._make_bar(entropy_pct),
            self._status_text(entropy_status)
        )

        # Clip fraction
        clip_pct = min(self.state.clip_fraction / 0.5, 1.0) * 100
        clip_status = self._get_clip_status(self.state.clip_fraction)
        table.add_row(
            "Clip Frac:",
            f"{self.state.clip_fraction:.2f}",
            self._make_bar(clip_pct),
            self._status_text(clip_status)
        )

        # KL divergence
        kl_pct = min(self.state.kl_divergence / 0.1, 1.0) * 100
        kl_status = self._get_kl_status(self.state.kl_divergence)
        table.add_row(
            "KL Div:",
            f"{self.state.kl_divergence:.4f}",
            self._make_bar(kl_pct),
            self._status_text(kl_status)
        )

        # Explained variance
        ev_pct = self.state.explained_variance * 100
        ev_status = self._get_explained_var_status(self.state.explained_variance)
        table.add_row(
            "Expl Var:",
            f"{self.state.explained_variance:.2f}",
            self._make_bar(max(0, ev_pct)),
            self._status_text(ev_status)
        )

        return Panel(table, title="[bold]POLICY HEALTH[/bold]", border_style="cyan")

    # Stage name abbreviations for compact multi-env display
    _STAGE_ABBREV: dict[str, str] = {
        "GERMINATED": "Germ",
        "TRAINING": "Train",
        "BLENDING": "Blend",
        "PROBATIONARY": "Prob",
        "FOSSILIZED": "Fossil",
        "RESETTING": "Reset",
        "EMBARGOED": "Embg",
    }

    def _render_single_env_seeds(self, env_state: EnvState) -> Table:
        """Render seed state table for a single environment."""
        table = Table(show_header=False, box=None, padding=(0, 0), expand=True)
        table.add_column("Info", style="dim", ratio=1)
        table.add_column("Value", justify="right", ratio=1)

        # Count stages
        stage_counts: dict[str, int] = {}
        for seed in env_state.seeds.values():
            stage_counts[seed.stage] = stage_counts.get(seed.stage, 0) + 1

        # Compact stage display with readable abbreviations
        for stage, count in sorted(stage_counts.items()):
            if stage not in ("DORMANT", "CULLED"):
                marker = self._get_stage_marker(stage)
                abbrev = self._STAGE_ABBREV.get(stage, stage[:5])
                table.add_row(f"{marker}{abbrev}", f"{count}")

        table.add_row("Fossil", f"{env_state.fossilized_count}")
        table.add_row("Culled", f"{env_state.culled_count}")

        return table

    def _render_seeds(self) -> Panel:
        """Render the seed state panel with per-env breakdown."""
        n_envs = max(1, self.state.n_envs)

        if n_envs == 1:
            # Single env: simple layout
            env_state = self.state.get_or_create_env(0)
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Info", style="dim")
            table.add_column("Value", justify="right")

            table.add_row("Active:", f"{self.state.active_seed_count} seeds")

            stage_counts: dict[str, int] = {}
            for seed in env_state.seeds.values():
                stage_counts[seed.stage] = stage_counts.get(seed.stage, 0) + 1

            for stage, count in sorted(stage_counts.items()):
                if stage not in ("DORMANT", "CULLED"):
                    marker = self._get_stage_marker(stage)
                    table.add_row(f"  {marker} {stage}", f"({count})")

            table.add_row("Fossilized:", f"{self.state.fossilized_count}")
            table.add_row("Culled:", f"{self.state.culled_count}")

            return Panel(table, title="[bold]SEED STATE[/bold]", border_style="cyan")

        # Multi-env: per-env columns
        main_table = Table(show_header=True, box=None, padding=(0, 1), expand=True)

        # Add header columns for each env
        for i in range(n_envs):
            main_table.add_column(f"Env{i}", justify="center", style="cyan")

        # Build rows by collecting data from each env
        env_tables = [
            self._render_single_env_seeds(self.state.get_or_create_env(i))
            for i in range(n_envs)
        ]

        main_table.add_row(*env_tables)

        # Add aggregate totals
        agg_text = Text()
        agg_text.append(f"Total: ", style="dim")
        agg_text.append(f"{self.state.active_seed_count}A ", style="green")
        agg_text.append(f"{self.state.fossilized_count}F ", style="blue")
        agg_text.append(f"{self.state.culled_count}C", style="red")

        content = Group(main_table, agg_text)
        return Panel(content, title="[bold]SEED STATE[/bold]", border_style="cyan")

    def _render_actions(self) -> Panel:
        """Render the action distribution panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Action", style="dim")
        table.add_column("Bar", width=15)
        table.add_column("Pct", justify="right")

        percentages = self.state.get_action_percentages()

        for action, pct in sorted(percentages.items(), key=lambda x: -x[1]):
            # Color based on action type
            action_style = {
                "WAIT": "dim",
                "GERMINATE": "green",
                "CULL": "red",
                "FOSSILIZE": "blue",
            }.get(action, "white")

            # Warning if WAIT is too high
            pct_style = "yellow bold" if action == "WAIT" and pct > self.thresholds.wait_warning * 100 else ""

            table.add_row(
                Text(action, style=action_style),
                self._make_bar(pct),
                Text(f"{pct:.0f}%", style=pct_style)
            )

        return Panel(table, title="[bold]ACTION DISTRIBUTION[/bold]", border_style="cyan")

    def _render_reward_components(self) -> Panel:
        """Render the reward components breakdown."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Component", style="dim")
        table.add_column("Value", justify="right")

        components = self.state.reward_components

        # Display key components
        if "accuracy_delta" in components:
            val = components["accuracy_delta"]
            style = "green" if val >= 0 else "red"
            table.add_row("Accuracy Δ:", Text(f"{val:+.2f}", style=style))

        if "bounded_attr" in components and components["bounded_attr"]:
            val = components["bounded_attr"]
            style = "green" if val >= 0 else "red"
            table.add_row("Attribution:", Text(f"{val:+.2f}", style=style))

        if "compute_rent" in components:
            val = components["compute_rent"]
            style = "red" if val < 0 else "dim"
            table.add_row("Compute Rent:", Text(f"{val:+.2f}", style=style))

        if "terminal_bonus" in components and components["terminal_bonus"]:
            val = components["terminal_bonus"]
            table.add_row("Terminal:", Text(f"{val:+.2f}", style="blue"))

        # Warnings
        warnings_shown = False
        if "blending_warn" in components and components["blending_warn"] < 0:
            table.add_row("Blending Warn:", Text(f"{components['blending_warn']:.2f}", style="yellow"))
            warnings_shown = True
        if "probation_warn" in components and components["probation_warn"] < 0:
            table.add_row("Probation Warn:", Text(f"{components['probation_warn']:.2f}", style="yellow"))
            warnings_shown = True

        # Total
        if not warnings_shown:
            table.add_row("", "")
        table.add_row(
            "───────────",
            "───────"
        )
        total = self.state.current_reward
        style = "bold green" if total >= 0 else "bold red"
        table.add_row("Total:", Text(f"{total:+.2f}", style=style))

        return Panel(table, title="[bold]REWARD COMPONENTS[/bold]", border_style="cyan")

    def _render_losses(self) -> Panel:
        """Render the losses panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Loss", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Policy:", f"{self.state.policy_loss:.4f}")
        table.add_row("Value:", f"{self.state.value_loss:.4f}")
        table.add_row("Entropy:", f"{self.state.entropy_loss:.4f}")
        table.add_row("Total:", f"{self.state.policy_loss + self.state.value_loss + self.state.entropy_loss:.4f}")

        table.add_row("", "")

        # Gradient norm with status
        grad_status = self._get_grad_norm_status(self.state.grad_norm)
        grad_style = {
            HealthStatus.OK: "green",
            HealthStatus.WARNING: "yellow",
            HealthStatus.CRITICAL: "red bold",
        }[grad_status]
        table.add_row("Grad Norm:", Text(f"{self.state.grad_norm:.2f}", style=grad_style))

        # Advantage stats
        table.add_row("", "")
        table.add_row("Adv Mean:", f"{self.state.advantage_mean:+.3f}")
        table.add_row("Adv Std:", f"{self.state.advantage_std:.3f}")

        return Panel(table, title="[bold]LOSSES[/bold]", border_style="cyan")

    def _render_footer(self) -> Panel:
        """Render the footer with key bindings hint."""
        text = Text(justify="center")
        text.append("[q] ", style="bold cyan")
        text.append("Quit  ", style="dim")
        text.append("[Ctrl+C] ", style="bold cyan")
        text.append("Stop Training", style="dim")
        return Panel(text, border_style="dim")

    def _render_performance(self) -> Panel:
        """Render performance statistics panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        # Throughput
        table.add_row("Epochs/sec:", f"{self.state.epochs_per_second:.2f}")
        table.add_row("Batches/hr:", f"{self.state.batches_per_hour:.0f}")

        # Runtime
        if self.state.start_time:
            elapsed = datetime.now() - self.state.start_time
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            table.add_row("Runtime:", f"{hours}h {minutes}m {seconds}s")
        else:
            table.add_row("Runtime:", "-")

        table.add_row("", "")

        # GPU stats
        if self.state.gpu_memory_total_gb > 0:
            gpu_pct = (self.state.gpu_memory_used_gb / self.state.gpu_memory_total_gb) * 100
            mem_style = "red" if gpu_pct > 90 else "yellow" if gpu_pct > 75 else "green"
            table.add_row(
                "GPU Mem:",
                Text(f"{self.state.gpu_memory_used_gb:.1f}/{self.state.gpu_memory_total_gb:.1f}GB", style=mem_style)
            )
        else:
            table.add_row("GPU Mem:", "-")

        if self.state.gpu_utilization > 0:
            util_style = "green" if self.state.gpu_utilization > 80 else "yellow" if self.state.gpu_utilization > 50 else "dim"
            table.add_row("GPU Util:", Text(f"{self.state.gpu_utilization:.0f}%", style=util_style))

        if self.state.gpu_temperature > 0:
            temp_style = "red" if self.state.gpu_temperature > 85 else "yellow" if self.state.gpu_temperature > 75 else "green"
            table.add_row("GPU Temp:", Text(f"{self.state.gpu_temperature:.0f}°C", style=temp_style))

        # CPU/RAM
        if self.state.cpu_percent > 0:
            table.add_row("CPU:", f"{self.state.cpu_percent:.0f}%")

        if self.state.ram_total_gb > 0:
            ram_pct = (self.state.ram_used_gb / self.state.ram_total_gb) * 100
            ram_style = "red" if ram_pct > 90 else "yellow" if ram_pct > 75 else "dim"
            table.add_row(
                "RAM:",
                Text(f"{self.state.ram_used_gb:.1f}/{self.state.ram_total_gb:.0f}GB", style=ram_style)
            )

        return Panel(table, title="[bold]PERFORMANCE[/bold]", border_style="cyan")

    # =========================================================================
    # System Monitoring
    # =========================================================================

    def _update_system_stats(self) -> None:
        """Update GPU and CPU statistics."""
        # GPU stats via PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                # Memory
                self.state.gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                self.state.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                # Try to get utilization and temp via pynvml
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.state.gpu_utilization = util.gpu
                    self.state.gpu_temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    pass  # pynvml not available
        except Exception:
            pass  # No GPU or error

        # CPU/RAM stats
        try:
            import psutil
            self.state.cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            self.state.ram_used_gb = mem.used / (1024**3)
            self.state.ram_total_gb = mem.total / (1024**3)
        except Exception:
            pass  # psutil not available

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _make_sparkline(self, values: list[float], width: int = 20) -> str:
        """Create a sparkline from values."""
        if not values:
            return "─" * width

        # Normalize to 0-1
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1.0

        # Use unicode block characters
        blocks = "▁▂▃▄▅▆▇█"

        # Take last `width` values
        values = values[-width:]

        result = ""
        for v in values:
            normalized = (v - min_val) / range_val
            idx = min(int(normalized * (len(blocks) - 1)), len(blocks) - 1)
            result += blocks[idx]

        # Pad with empty if not enough values
        result = result.ljust(width, "─")
        return result

    def _make_bar(self, percentage: float, width: int = 10) -> str:
        """Create a progress bar."""
        filled = int(percentage / 100 * width)
        return "█" * filled + "░" * (width - filled)

    def _status_text(self, status: HealthStatus) -> Text:
        """Create status text with color."""
        if status == HealthStatus.OK:
            return Text("OK", style="green")
        elif status == HealthStatus.WARNING:
            return Text("WARN", style="yellow")
        else:
            return Text("CRIT", style="red bold")

    def _get_stage_marker(self, stage: str) -> str:
        """Get a marker for seed stage."""
        markers = {
            "GERMINATED": "●",
            "BLENDING": "◐",
            "PROBATION": "◑",
            "INTEGRATED": "◉",
            "FOSSILIZED": "★",
        }
        return markers.get(stage, "○")

    def _get_entropy_status(self, entropy: float) -> HealthStatus:
        """Get health status for entropy."""
        if entropy < self.thresholds.entropy_critical:
            return HealthStatus.CRITICAL
        elif entropy < self.thresholds.entropy_warning:
            return HealthStatus.WARNING
        return HealthStatus.OK

    def _get_clip_status(self, clip_fraction: float) -> HealthStatus:
        """Get health status for clip fraction."""
        if clip_fraction > self.thresholds.clip_critical:
            return HealthStatus.CRITICAL
        elif clip_fraction > self.thresholds.clip_warning:
            return HealthStatus.WARNING
        return HealthStatus.OK

    def _get_kl_status(self, kl_div: float) -> HealthStatus:
        """Get health status for KL divergence."""
        if kl_div > self.thresholds.kl_warning:
            return HealthStatus.WARNING
        return HealthStatus.OK

    def _get_explained_var_status(self, ev: float) -> HealthStatus:
        """Get health status for explained variance."""
        if ev < self.thresholds.explained_var_critical:
            return HealthStatus.CRITICAL
        elif ev < self.thresholds.explained_var_warning:
            return HealthStatus.WARNING
        return HealthStatus.OK

    def _get_grad_norm_status(self, grad_norm: float) -> HealthStatus:
        """Get health status for gradient norm."""
        if grad_norm > self.thresholds.grad_norm_critical:
            return HealthStatus.CRITICAL
        elif grad_norm > self.thresholds.grad_norm_warning:
            return HealthStatus.WARNING
        return HealthStatus.OK


__all__ = ["TUIOutput", "TUIState", "ThresholdConfig", "HealthStatus"]
