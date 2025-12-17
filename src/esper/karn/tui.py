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
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from esper.karn.contracts import KarnSlotConfig, SlotConfigProtocol, TelemetryEventLike

_logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Enums
# =============================================================================

class ThresholdConfig:
    """Thresholds for red flag detection (delegates to constants)."""

    # Import at module level
    from esper.karn.constants import TUIThresholds

    # Entropy thresholds
    entropy_critical: float = TUIThresholds.ENTROPY_CRITICAL
    entropy_warning: float = TUIThresholds.ENTROPY_WARNING
    entropy_max: float = TUIThresholds.ENTROPY_MAX

    # Clip fraction thresholds
    clip_critical: float = TUIThresholds.CLIP_CRITICAL
    clip_warning: float = TUIThresholds.CLIP_WARNING

    # Explained variance
    explained_var_critical: float = TUIThresholds.EXPLAINED_VAR_CRITICAL
    explained_var_warning: float = TUIThresholds.EXPLAINED_VAR_WARNING

    # Gradient norm
    grad_norm_critical: float = TUIThresholds.GRAD_NORM_CRITICAL
    grad_norm_warning: float = TUIThresholds.GRAD_NORM_WARNING

    # KL divergence
    kl_warning: float = TUIThresholds.KL_WARNING

    # Action distribution
    wait_warning: float = TUIThresholds.WAIT_DOMINANCE_WARNING


class HealthStatus(Enum):
    """Health status for color coding."""
    OK = "green"
    WARNING = "yellow"
    CRITICAL = "red"


# =============================================================================
# Per-Environment State Classes
# =============================================================================

@dataclass
class SeedState:
    """Current state of a seed slot."""
    slot_id: str = ""
    stage: str = "DORMANT"
    blueprint_id: str | None = None
    alpha: float = 0.0
    accuracy_delta: float = 0.0
    seed_params: int = 0


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
    host_params: int = 0
    seeds: dict[str, SeedState] = field(default_factory=dict)
    active_seed_count: int = 0
    fossilized_count: int = 0
    culled_count: int = 0

    # Most recent reward component breakdown for this env (from REWARD_COMPUTED telemetry)
    reward_components: dict[str, float | None] = field(default_factory=dict)

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
    fossilized_params: int = 0

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

        # Normalize factored germination actions (e.g., GERMINATE_CONV_LIGHT) to GERMINATE
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
        """Update env status based on metrics."""
        if self.epochs_since_improvement > 10:
            self.status = "stalled"
        elif curr_acc < prev_acc - 1.0:
            self.status = "degraded"
        elif curr_acc > 80.0:
            self.status = "excellent"
        elif self.current_epoch > 0:
            self.status = "healthy"

    @property
    def total_seed_params(self) -> int:
        """Approximate total seed params (active + fossilized)."""
        total = self.fossilized_params
        for seed in self.seeds.values():
            if seed.stage not in ("CULLED", "DORMANT"):
                total += seed.seed_params
        return total


# =============================================================================
# TUI State
# =============================================================================


@dataclass
class TUIState:
    """Thread-safe state for the TUI display."""

    # Slot configuration (accepts any SlotConfigProtocol-compatible object)
    slot_config: SlotConfigProtocol = field(default_factory=KarnSlotConfig.default)

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
    max_epochs: int = 75  # Inner epochs per batch (from TRAINING_STARTED)

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
        counts = self.aggregate_action_counts if self.env_states else self.action_counts
        total = self.aggregate_total_actions if self.env_states else self.total_actions
        if total == 0:
            return {k: 0.0 for k in counts}
        return {k: v / total * 100 for k, v in counts.items()}

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
        slot_config: Slot configuration for dynamic action spaces. Accepts any
            object implementing SlotConfigProtocol. Defaults to
            KarnSlotConfig.default() (3-slot configuration).
        force_layout: Force a specific layout mode ('compact', 'standard', 'wide')
                     instead of auto-detecting from terminal width. Currently unused
                     but reserved for future multi-layout support.

    Note:
        The ``slot_config`` parameter accepts any object implementing
        ``esper.karn.contracts.SlotConfigProtocol``. Both
        ``esper.leyline.slot_config.SlotConfig`` and
        ``esper.karn.contracts.KarnSlotConfig`` implement this protocol. The
        TUI is decoupled from Leyline to enable standalone use or testing.
    """

    def __init__(
        self,
        thresholds: ThresholdConfig | None = None,
        slot_config: SlotConfigProtocol | None = None,
        force_layout: str | None = None,
    ):
        self.thresholds = thresholds or ThresholdConfig()
        self.state = TUIState(slot_config=slot_config or KarnSlotConfig.default())
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

    def emit(self, event: TelemetryEventLike) -> None:
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

    def _format_event_for_log(self, event: TelemetryEventLike) -> tuple[str, str, str] | None:
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

        # Extract env_id for prefix (multi-env tracking)
        env_id = data.get("env_id")
        env_prefix = f"[{env_id}] " if env_id is not None else ""

        # Format message based on event type
        if "EPOCH" in event_type:
            loss = data.get("val_loss", "?")
            acc = data.get("val_accuracy", "?")
            epoch = event.epoch or data.get("epoch", "?")
            inner_epoch = data.get("inner_epoch")
            batch = data.get("batch")

            if isinstance(acc, (int, float)):
                acc_str = f"{acc:.2%}" if 0.0 <= float(acc) <= 1.0 else f"{acc:.1f}%"
            else:
                acc_str = str(acc)

            prefix = f"epoch={epoch}"
            if batch is not None:
                prefix += f" b={batch}"
            if inner_epoch is not None:
                prefix += f" in={inner_epoch}"

            if isinstance(loss, (int, float)) and isinstance(acc, (int, float)):
                msg = f"{prefix} loss={float(loss):.4f} acc={acc_str}"
            else:
                msg = prefix
        elif event_type == "REWARD_COMPUTED":
            action = data.get("action_name", "?")
            total = data.get("total_reward", 0.0)
            msg = f"{env_prefix}{action} r={total:+.3f}"
        elif event_type.startswith("SEED_"):
            seed_id = event.seed_id or "?"
            if event_type == "SEED_GERMINATED":
                blueprint_id = data.get("blueprint_id", "?")
                msg = f"{env_prefix}{seed_id} germinated ({blueprint_id})"
            elif event_type == "SEED_STAGE_CHANGED":
                from_stage = data.get("from", "?")
                to_stage = data.get("to", "?")
                msg = f"{env_prefix}{seed_id} {from_stage}->{to_stage}"
            elif event_type == "SEED_FOSSILIZED":
                improvement = data.get("improvement", 0)
                msg = f"{env_prefix}{seed_id} fossilized (+{improvement:.2f}%)"
            elif event_type == "SEED_CULLED":
                reason = data.get("reason", "")
                msg = f"{env_prefix}{seed_id} culled" + (f" ({reason})" if reason else "")
            else:
                msg = f"{env_prefix}{seed_id}"
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
            batch_idx = data.get("batch_idx", "?")
            episodes = data.get("episodes_completed", "?")
            total = data.get("total_episodes", "?")
            rolling_acc = data.get("rolling_accuracy")
            avg_reward = data.get("avg_reward")

            parts = [f"batch={batch_idx}", f"ep={episodes}/{total}"]
            if isinstance(rolling_acc, (int, float)):
                parts.append(f"acc={rolling_acc:.1f}%")
            if isinstance(avg_reward, (int, float)):
                parts.append(f"r={avg_reward:+.2f}")
            msg = " ".join(parts)
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

    def _handle_training_started(self, event: TelemetryEventLike) -> None:
        """Handle TRAINING_STARTED event."""
        data = event.data or {}
        self.state.current_episode = data.get("episode", 0)
        self.state.current_epoch = 0
        self.state.n_envs = data.get("n_envs", 1)
        self.state.max_epochs = data.get("max_epochs", self.state.max_epochs)
        self.state.reward_hacking_detected = False
        self.state.current_reward = 0.0
        self.state.reward_history.clear()
        self.state.best_reward = float("-inf")
        self.state.best_episode = 0
        self.state.episode_rewards.clear()
        self.state.last_reward_env_id = 0

        # Initialize performance tracking
        self.state.start_time = datetime.now()
        self.state.batches_completed = 0
        self.state.best_accuracy = 0.0
        self.state.best_accuracy_episode = 0

        # Reset multi-env state for a new run
        self.state.env_states.clear()

        # Pre-create env states
        for i in range(self.state.n_envs):
            self.state.get_or_create_env(i)

        # Initial system stats
        self._update_system_stats()

    def _handle_epoch_completed(self, event: TelemetryEventLike) -> None:
        """Handle EPOCH_COMPLETED event."""
        data = event.data or {}
        self.state.current_epoch = event.epoch or data.get("epoch", 0)
        self.state.host_accuracy = data.get("val_accuracy", 0.0)
        self.state.host_loss = data.get("val_loss", 0.0)

        # Calculate accuracy delta from train vs val
        train_acc = data.get("train_accuracy", 0.0)
        self.state.host_accuracy_delta = self.state.host_accuracy - train_acc

    def _handle_ppo_update(self, event: TelemetryEventLike) -> None:
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

    def _handle_reward_computed(self, event: TelemetryEventLike) -> None:
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

        # Store reward component breakdown per environment
        env_state.reward_components = {
            "base_acc_delta": data.get("base_acc_delta", 0.0),
            "seed_contribution": data.get("seed_contribution"),
            "bounded_attribution": data.get("bounded_attribution"),
            "progress_since_germination": data.get("progress_since_germination"),
            "attribution_discount": data.get("attribution_discount", 1.0),
            "ratio_penalty": data.get("ratio_penalty", 0.0),
            "compute_rent": data.get("compute_rent", 0.0),
            "blending_warning": data.get("blending_warning", 0.0),
            "probation_warning": data.get("probation_warning", 0.0),
            "stage_bonus": data.get("stage_bonus", 0.0),
            "pbrs_bonus": data.get("pbrs_bonus", 0.0),
            "action_shaping": data.get("action_shaping", 0.0),
            "terminal_bonus": data.get("terminal_bonus", 0.0),
            "fossilize_terminal_bonus": data.get("fossilize_terminal_bonus", 0.0),
            "growth_ratio": data.get("growth_ratio", 0.0),
            "val_acc": data.get("val_acc", env_state.host_accuracy),
        }
        self.state.last_reward_env_id = env_id

        # Red flag: reward hacking detection
        acc_delta = data.get("base_acc_delta", 0.0)
        if acc_delta < 0 and total_reward > 0:
            self.state.reward_hacking_detected = True

    def _handle_seed_event(self, event: TelemetryEventLike, event_type: str) -> None:
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
            seed.seed_params = data.get("params", seed.seed_params)
            env_state.active_seed_count += 1
        elif event_type == "SEED_STAGE_CHANGED":
            seed.stage = data.get("to", seed.stage)
            seed.alpha = data.get("alpha", seed.alpha)
        elif event_type == "SEED_FOSSILIZED":
            seed.stage = "FOSSILIZED"
            # params_added moves into host; keep count for total params estimate
            env_state.fossilized_params += int(data.get("params_added", 0) or 0)
            env_state.fossilized_count += 1
            env_state.active_seed_count = max(0, env_state.active_seed_count - 1)
        elif event_type == "SEED_CULLED":
            seed.stage = "CULLED"
            seed.seed_params = 0
            env_state.culled_count += 1
            env_state.active_seed_count = max(0, env_state.active_seed_count - 1)

        # Update aggregate counts
        self.state.update_aggregate_seed_counts()

    def _handle_batch_completed(self, event: TelemetryEventLike) -> None:
        """Handle BATCH_COMPLETED event (episode completion)."""
        data = event.data or {}
        episodes_completed = data.get("episodes_completed")
        if isinstance(episodes_completed, (int, float)):
            self.state.current_episode = int(episodes_completed)
        self.state.batches_completed += 1

        # Track best accuracy across all envs
        current_acc = data.get("rolling_accuracy", 0.0)
        self.state.host_accuracy = current_acc
        if current_acc > self.state.best_accuracy:
            self.state.best_accuracy = current_acc
            self.state.best_accuracy_episode = self.state.current_episode

        # Track rewards at the batch/episode level
        avg_reward = data.get("avg_reward", 0.0)
        self.state.update_reward(avg_reward)

        # Calculate throughput
        now = datetime.now()
        if self.state.start_time:
            elapsed = (now - self.state.start_time).total_seconds()
            if elapsed > 0:
                episodes_completed = self.state.current_episode if self.state.current_episode > 0 else self.state.batches_completed
                total_epochs = episodes_completed * self.state.max_epochs
                self.state.epochs_completed = total_epochs
                self.state.epochs_per_second = total_epochs / elapsed
                self.state.batches_per_hour = (self.state.batches_completed / elapsed) * 3600

        self.state.last_batch_time = now

        # Add to episode rewards for sparkline
        self.state.episode_rewards.append(avg_reward)

        # Update GPU stats if available
        self._update_system_stats()

        # Reset per-env seed state for the next batch/episode
        for env_state in self.state.env_states.values():
            env_state.seeds.clear()
            env_state.active_seed_count = 0
            env_state.fossilized_count = 0
            env_state.culled_count = 0
            env_state.fossilized_params = 0
        self.state.update_aggregate_seed_counts()

    # =========================================================================
    # Rendering
    # =========================================================================

    def _render(self) -> Layout:
        """Render the full TUI layout with event log.

        Returns Layout directly (not wrapped in Panel) to avoid
        off-by-one height issues from extra border lines.
        """
        layout = Layout()

        # Unified layout with env cards always visible
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="env_cards", ratio=2),
            Layout(name="main", size=4),
            Layout(name="bottom", ratio=2),
            Layout(name="footer", size=2),
        )
        layout["env_cards"].update(self._render_env_cards())

        # Split bottom into event log (left) and performance stats (right)
        layout["bottom"].split_row(
            Layout(name="event_log", ratio=2),
            Layout(name="perf_stats", ratio=1),
        )

        # Main: combined policy stats panel (actions + health + losses + rewards)
        layout["main"].update(self._render_policy_stats())

        # Render each section
        layout["header"].update(self._render_header())
        layout["event_log"].update(self._render_event_log(max_lines=8))
        layout["perf_stats"].update(self._render_performance())
        layout["footer"].update(self._render_footer())

        return layout

    def _render_header(self) -> Panel:
        """Render the header with episode info and multi-env summary."""
        text = Text()
        text.append("Episode: ", style="dim")
        text.append(f"{self.state.current_episode}", style="bold cyan")
        text.append("  |  Batches: ", style="dim")
        text.append(f"{self.state.batches_completed}", style="bold cyan")
        text.append("  |  Best Acc: ", style="dim")
        text.append(f"{self.state.best_accuracy:.1f}%", style="bold green")
        text.append(f" (ep {self.state.best_accuracy_episode})", style="dim")
        text.append("  |  Current: ", style="dim")
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

    def _render_rewards_table(self) -> Table:
        """Render rewards summary as a table (for combined panels)."""
        table = Table(show_header=False, box=None, padding=(0, 0))
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

        return table

    def _render_rewards(self) -> Panel:
        """Render the rewards panel."""
        return Panel(self._render_rewards_table(), title="[bold]REWARDS[/bold]", border_style="cyan")

    def _render_env_rewards(self) -> Panel:
        """Render a per-environment reward components summary (all envs at once)."""
        if not self.state.env_states:
            return Panel(
                Text("Waiting for reward telemetry…", style="dim"),
                title="[bold]ENV REWARDS[/bold]",
                border_style="cyan",
            )

        def _fmt_component(value: float | None) -> Text:
            if not isinstance(value, (int, float)):
                return Text("─", style="dim")
            v = float(value)
            if v > 0:
                style = "green"
            elif v < 0:
                style = "red"
            else:
                style = "dim"
            return Text(f"{v:+.2f}", style=style)

        def _fmt_total(value: float) -> Text:
            if value > 0:
                style = "bold green"
            elif value < 0:
                style = "bold red"
            else:
                style = "dim"
            return Text(f"{value:+.2f}", style=style)

        def _fmt_warning(value: float | None) -> Text:
            if not isinstance(value, (int, float)):
                return Text("─", style="dim")
            v = float(value)
            if v < 0:
                style = "yellow"
            elif v > 0:
                style = "green"
            else:
                style = "dim"
            return Text(f"{v:+.2f}", style=style)

        table = Table(show_header=True, box=None, padding=(0, 0), expand=True)
        table.add_column("Env", style="cyan", justify="center", width=3)
        table.add_column("Total", justify="right", width=7)
        table.add_column("ΔAcc", justify="right", width=6)
        table.add_column("Attr", justify="right", width=6)
        table.add_column("Rent", justify="right", width=6)
        table.add_column("Pen", justify="right", width=6)
        table.add_column("Other", justify="right", width=6)
        table.add_column("Warn", justify="right", width=6)

        for env_id in sorted(self.state.env_states.keys()):
            env = self.state.env_states[env_id]
            components = env.reward_components

            base = components.get("base_acc_delta")
            attr = components.get("bounded_attribution")
            rent = components.get("compute_rent")
            penalty = components.get("ratio_penalty")

            other_total = 0.0
            other_any = False
            other_sources = (
                "stage_bonus",
                "pbrs_bonus",
                "action_shaping",
                "terminal_bonus",
                "fossilize_terminal_bonus",
            )
            for key in other_sources:
                v = components.get(key)
                if isinstance(v, (int, float)):
                    other_total += float(v)
                    other_any = True

            warn_total = 0.0
            warn_any = False
            warn_sources = ("blending_warning", "probation_warning")
            for key in warn_sources:
                v = components.get(key)
                if isinstance(v, (int, float)):
                    warn_total += float(v)
                    warn_any = True

            table.add_row(
                str(env_id),
                _fmt_total(env.current_reward),
                _fmt_component(base),
                _fmt_component(attr),
                _fmt_component(rent),
                _fmt_component(penalty),
                _fmt_component(other_total if other_any else None),
                _fmt_warning(warn_total if warn_any else None),
            )

    def _render_env_cards(self) -> Panel:
        """Render compact cards for all environments with per-slot telemetry."""
        if not self.state.env_states:
            return Panel(Text("Waiting for telemetry…", style="dim"), title="[bold]ENV CARDS[/bold]", border_style="cyan")

        cards = []
        for env_id in sorted(self.state.env_states.keys()):
            env = self.state.env_states[env_id]

            # Compact table: summary row with slots rendered horizontally
            table = Table(show_header=True, box=None, padding=(0, 0), expand=True)
            table.add_column("Row", style="dim", width=5)
            table.add_column("Reward", justify="right", width=9)
            table.add_column("Acc", justify="right", width=8)
            table.add_column("Seeds/Params", justify="right", width=14)
            table.add_column("Rent", justify="right", width=6)
            table.add_column("", width=1)
            # Dynamic slot columns based on slot_config
            for slot_id in self.state.slot_config.slot_ids:
                table.add_column(slot_id, justify="left", width=12)
            table.add_column("Last / Status", justify="left", width=16)

            reward = env.current_reward
            rent = env.reward_components.get("compute_rent")
            penalty = env.reward_components.get("ratio_penalty")
            params = getattr(env, "host_params", None)
            last_action = env.action_history[-1] if env.action_history else "—"
            seeds_summary = f"A:{env.active_seed_count} F:{env.fossilized_count} C:{env.culled_count}"
            growth_ratio = env.reward_components.get("growth_ratio")

            # Estimate total params if host_params absent using growth_ratio + seed params
            total_params_display = None
            if isinstance(params, int) and params > 0:
                total_params_display = params + env.total_seed_params
            elif isinstance(growth_ratio, (int, float)) and growth_ratio > 1.0 and env.total_seed_params > 0:
                host_est = int(env.total_seed_params / (growth_ratio - 1))
                env.host_params = host_est
                total_params_display = host_est + env.total_seed_params

            def _slot_summary(slot_name: str) -> str:
                seed = env.seeds.get(slot_name)
                if seed:
                    stage_cell = self._format_slot_cell(env, slot_name)
                    alpha = f"{seed.alpha:.2f}"
                    contrib = f"{seed.accuracy_delta:+.2f}"
                    grad_ratio = getattr(seed, "grad_ratio", None)
                    grad = f"{grad_ratio:+.2f}" if isinstance(grad_ratio, (int, float)) else "─"
                    return f"{stage_cell} α={alpha} Δ={contrib} g={grad}"
                return "[dim]─[/dim]"

            # Build the first row dynamically with all slot summaries
            row_values = [
                "Env",
                f"{reward:+.2f}",
                f"{env.host_accuracy:.1f}%",
                seeds_summary,
                f"{float(rent):+.2f}" if isinstance(rent, (int, float)) else "─",
                "",
            ]
            # Add dynamic slot summaries
            for slot_id in self.state.slot_config.slot_ids:
                row_values.append(_slot_summary(slot_id))
            # Add last action
            row_values.append(last_action)

            table.add_row(*row_values)

            # Secondary stats row within the same card to avoid extra tables
            best_reward = env.best_reward if env.reward_history else 0.0
            mean_reward = env.mean_reward
            reward_spark = env.reward_sparkline
            acc_spark = env.accuracy_sparkline
            status_style = {
                "excellent": "bold green",
                "healthy": "green",
                "stalled": "yellow",
                "degraded": "red",
                "initializing": "dim",
            }.get(env.status, "white")

            penalty_str = f"{float(penalty):+.2f}" if isinstance(penalty, (int, float)) else "─"

            # Build the second row dynamically
            stats_row_values = [
                "Stats",
                f"{mean_reward:+.2f}/{best_reward:+.2f}",
                f"best {env.best_accuracy:.1f}%",
                f"{total_params_display:,}" if isinstance(total_params_display, (int, float)) else seeds_summary,
                penalty_str,
                "",
            ]
            # Fill slot columns with sparklines/status
            # First slot gets reward sparkline, second gets accuracy sparkline, third gets status
            # If more slots exist, fill with empty strings
            num_slots = self.state.slot_config.num_slots
            for i in range(num_slots):
                if i == 0:
                    stats_row_values.append(f"Rwd {reward_spark}")
                elif i == 1:
                    stats_row_values.append(f"Acc {acc_spark}")
                elif i == 2:
                    stats_row_values.append(f"[{status_style}]{env.status.upper()}[/{status_style}]")
                else:
                    stats_row_values.append("")
            # Add last action
            stats_row_values.append(Text(last_action, style="dim"))

            table.add_row(*stats_row_values)

            header = Text()
            header.append(f"Env {env_id}  ", style="bold cyan")
            header.append(f"Acc {env.host_accuracy:.1f}%", style="green" if env.host_accuracy >= env.best_accuracy else "yellow")

            cards.append(Panel(table, title=header, border_style="cyan"))

        return Panel(Group(*cards), title="[bold]ENV CARDS[/bold]", border_style="cyan")

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

    def _render_actions_table(self) -> Table:
        """Render action distribution as a table (for combined panel)."""
        table = Table(show_header=False, box=None, padding=(0, 0))
        table.add_column("Action", style="dim", width=8)
        table.add_column("Bar", width=8)
        table.add_column("%", justify="right", width=4)

        percentages = self.state.get_action_percentages()

        for action, pct in sorted(percentages.items(), key=lambda x: -x[1]):
            action_style = {
                "WAIT": "dim",
                "GERMINATE": "green",
                "CULL": "red",
                "FOSSILIZE": "blue",
            }.get(action, "white")

            pct_style = "yellow bold" if action == "WAIT" and pct > self.thresholds.wait_warning * 100 else ""

            table.add_row(
                Text(action, style=action_style),
                self._make_bar(pct, width=8),
                Text(f"{pct:.0f}%", style=pct_style)
            )

        return table

    def _render_policy_health_table(self) -> Table:
        """Render policy health as a table (for combined panel)."""
        table = Table(show_header=False, box=None, padding=(0, 0))
        table.add_column("Metric", style="dim", width=8)
        table.add_column("Val", justify="right", width=5)
        table.add_column("St", justify="center", width=4)

        # Entropy
        entropy_status = self._get_entropy_status(self.state.entropy)
        table.add_row("Entropy", f"{self.state.entropy:.2f}", self._status_text(entropy_status))

        # Clip fraction
        clip_status = self._get_clip_status(self.state.clip_fraction)
        table.add_row("Clip", f"{self.state.clip_fraction:.2f}", self._status_text(clip_status))

        # KL divergence
        kl_status = self._get_kl_status(self.state.kl_divergence)
        table.add_row("KL", f"{self.state.kl_divergence:.3f}", self._status_text(kl_status))

        # Explained variance
        ev_status = self._get_explained_var_status(self.state.explained_variance)
        table.add_row("ExplVar", f"{self.state.explained_variance:.2f}", self._status_text(ev_status))

        return table

    def _render_losses_table(self) -> Table:
        """Render losses as a table (for combined panel)."""
        table = Table(show_header=False, box=None, padding=(0, 0))
        table.add_column("Loss", style="dim", width=7)
        table.add_column("Value", justify="right", width=8)

        table.add_row("Policy", f"{self.state.policy_loss:.4f}")
        table.add_row("Value", f"{self.state.value_loss:.4f}")
        table.add_row("Entropy", f"{self.state.entropy_loss:.4f}")

        # Gradient norm with status
        grad_status = self._get_grad_norm_status(self.state.grad_norm)
        grad_style = {
            HealthStatus.OK: "green",
            HealthStatus.WARNING: "yellow",
            HealthStatus.CRITICAL: "red bold",
        }[grad_status]
        table.add_row("GradNorm", Text(f"{self.state.grad_norm:.2f}", style=grad_style))

        return table

    def _render_policy_stats(self) -> Panel:
        """Render a compact policy stats strip with minimal height."""
        table = Table(show_header=False, box=None, padding=(0, 0), expand=True)
        table.add_column("Actions", ratio=2)
        table.add_column("Health", ratio=2)
        table.add_column("Losses", ratio=2)
        table.add_column("Rewards", ratio=2)

        # Actions mix as a single line
        percentages = self.state.get_action_percentages()
        action_parts = []
        for action, pct in sorted(percentages.items(), key=lambda x: -x[1]):
            action_short = {
                "WAIT": "W",
                "GERMINATE": "G",
                "CULL": "C",
                "FOSSILIZE": "F",
            }.get(action, action[:1])
            style = {
                "WAIT": "dim",
                "GERMINATE": "green",
                "CULL": "red",
                "FOSSILIZE": "blue",
            }.get(action, "white")
            action_parts.append(f"[{style}]{action_short}{pct:.0f}%[/{style}]")
        actions_line = " ".join(action_parts) if action_parts else "─"

        # Health metrics inline
        entropy_status = self._get_entropy_status(self.state.entropy)
        clip_status = self._get_clip_status(self.state.clip_fraction)
        kl_status = self._get_kl_status(self.state.kl_divergence)
        ev_status = self._get_explained_var_status(self.state.explained_variance)
        health_line = " ".join(
            [
                f"H {self.state.entropy:.2f}{self._status_text(entropy_status)}",
                f"Clip {self.state.clip_fraction:.2f}{self._status_text(clip_status)}",
                f"KL {self.state.kl_divergence:.3f}{self._status_text(kl_status)}",
                f"EV {self.state.explained_variance:.2f}{self._status_text(ev_status)}",
            ]
        )

        # Losses inline
        grad_status = self._get_grad_norm_status(self.state.grad_norm)
        grad_style = {
            HealthStatus.OK: "green",
            HealthStatus.WARNING: "yellow",
            HealthStatus.CRITICAL: "red bold",
        }[grad_status]
        losses_line = " ".join(
            [
                f"P{self.state.policy_loss:.3f}",
                f"V{self.state.value_loss:.3f}",
                f"E{self.state.entropy_loss:.3f}",
                f"G[{grad_style}]{self.state.grad_norm:.2f}[/]",
            ]
        )

        # Rewards inline
        mean = self.state.mean_reward_100
        std = self.state.std_reward_100
        rewards_line = " ".join(
            [
                f"Cur {self.state.current_reward:+.2f}",
                f"Mean100 {mean:+.2f}",
                f"Std {std:.2f}",
                f"Best {self.state.best_reward:+.2f} (ep {self.state.best_episode})",
            ]
        )

        table.add_row(actions_line, health_line, losses_line, rewards_line)

        # Wrap in a tight panel to keep a small footprint
        return Panel(
            table,
            title="[bold]POLICY STATS[/bold]",
            border_style="cyan",
            padding=(0, 0),
            expand=True,
        )

    def _render_reward_components(self) -> Panel:
        """Render the reward components breakdown for the focused environment."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Component", style="dim")
        table.add_column("Value", justify="right")

        env_id = self.state.last_reward_env_id
        env_state = self.state.get_or_create_env(env_id)
        components = env_state.reward_components

        # Header context
        table.add_row("Env:", Text(str(env_id), style="bold cyan"))
        if env_state.action_history:
            table.add_row("Action:", env_state.action_history[-1])
        if "val_acc" in components and isinstance(components.get("val_acc"), (int, float)):
            table.add_row("Val Acc:", f"{float(components['val_acc']):.1f}%")

        table.add_row("", "")

        # Base delta (legacy shaped signal)
        base = components.get("base_acc_delta")
        if isinstance(base, (int, float)):
            style = "green" if float(base) >= 0 else "red"
            table.add_row("ΔAcc:", Text(f"{float(base):+,.2f}", style=style))

        # Attribution (contribution-primary)
        bounded = components.get("bounded_attribution")
        if isinstance(bounded, (int, float)) and float(bounded) != 0.0:
            style = "green" if float(bounded) >= 0 else "red"
            table.add_row("Attr:", Text(f"{float(bounded):+,.2f}", style=style))

        # Compute rent (usually negative)
        rent = components.get("compute_rent")
        if isinstance(rent, (int, float)):
            style = "red" if float(rent) < 0 else "dim"
            table.add_row("Rent:", Text(f"{float(rent):+,.2f}", style=style))

        # Ratio penalty (ransomware / attribution mismatch)
        ratio_penalty = components.get("ratio_penalty")
        if isinstance(ratio_penalty, (int, float)) and float(ratio_penalty) != 0.0:
            style = "red" if float(ratio_penalty) < 0 else "dim"
            table.add_row("Penalty:", Text(f"{float(ratio_penalty):+,.2f}", style=style))

        # Stage / terminal bonuses
        stage_bonus = components.get("stage_bonus")
        if isinstance(stage_bonus, (int, float)) and float(stage_bonus) != 0.0:
            table.add_row("Stage:", Text(f"{float(stage_bonus):+,.2f}", style="blue"))

        fossil_bonus = components.get("fossilize_terminal_bonus")
        if isinstance(fossil_bonus, (int, float)) and float(fossil_bonus) != 0.0:
            table.add_row("Fossil:", Text(f"{float(fossil_bonus):+,.2f}", style="blue"))

        # Warnings
        blending_warn = components.get("blending_warning")
        if isinstance(blending_warn, (int, float)) and float(blending_warn) < 0:
            table.add_row("Blend Warn:", Text(f"{float(blending_warn):.2f}", style="yellow"))

        probation_warn = components.get("probation_warning")
        if isinstance(probation_warn, (int, float)) and float(probation_warn) < 0:
            table.add_row("Prob Warn:", Text(f"{float(probation_warn):.2f}", style="yellow"))

        # Total (last computed reward for this env)
        table.add_row("", "")
        table.add_row("───────────", "───────")
        total = env_state.current_reward
        style = "bold green" if total >= 0 else "bold red"
        table.add_row("Total:", Text(f"{total:+.2f}", style=style))

        return Panel(
            table,
            title=f"[bold]REWARD COMPONENTS (env {env_id})[/bold]",
            border_style="cyan",
        )

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

    # Stage color mapping for slot display
    _STAGE_STYLES: dict[str, str] = {
        "DORMANT": "dim",
        "GERMINATED": "bright_cyan",
        "TRAINING": "yellow",
        "BLENDING": "magenta",
        "PROBATIONARY": "bright_blue",
        "FOSSILIZED": "bright_green",
        "CULLED": "red",
        "RESETTING": "dim",
        "EMBARGOED": "red",
    }

    # Short stage names for compact display
    _STAGE_SHORT: dict[str, str] = {
        "DORMANT": "Dorm",
        "GERMINATED": "Germ",
        "TRAINING": "Train",
        "BLENDING": "Blend",
        "PROBATIONARY": "Prob",
        "FOSSILIZED": "Foss",
        "CULLED": "Cull",
        "RESETTING": "Reset",
        "EMBARGOED": "Embg",
    }

    def _format_slot_cell(self, env: EnvState, slot_name: str) -> str:
        """Format a slot cell showing stage and blueprint.

        Returns styled string like "[cyan]Train:conv[/cyan]" or "[dim]─[/dim]".
        """
        seed = env.seeds.get(slot_name)
        if not seed or seed.stage == "DORMANT":
            return "[dim]─[/dim]"

        stage_short = self._STAGE_SHORT.get(seed.stage, seed.stage[:4])
        style = self._STAGE_STYLES.get(seed.stage, "white")

        # Get blueprint abbreviation (first 4 chars)
        blueprint = seed.blueprint_id or "?"
        if len(blueprint) > 4:
            blueprint = blueprint[:4]

        return f"[{style}]{stage_short}:{blueprint}[/{style}]"

    def _render_env_overview(self) -> Panel:
        """Render per-environment overview table.

        Shows a row per environment with key metrics, slot states, and status.
        Only rendered when n_envs > 1.
        """
        table = Table(show_header=True, box=None, padding=(0, 1), expand=True)

        # Columns: ID, Step, Accuracy, Sparklines, Reward, Components, Slots, Status
        table.add_column("Env", style="cyan", justify="center", width=3)
        table.add_column("Step", justify="right", width=7)
        table.add_column("Acc", justify="right", width=6)
        table.add_column("▁▃▅", justify="left", width=8)  # Sparkline
        table.add_column("Reward", justify="right", width=7)
        table.add_column("▁▃▅", justify="left", width=8)  # Sparkline
        table.add_column("ΔAcc", justify="right", width=6)
        table.add_column("Rent", justify="right", width=6)
        # Dynamic slot columns based on slot_config
        for slot_id in self.state.slot_config.slot_ids:
            table.add_column(slot_id, justify="center", width=10)
        table.add_column("Status", justify="center", width=9)

        # Status color mapping
        status_styles = {
            "excellent": "bold green",
            "healthy": "green",
            "initializing": "dim",
            "stalled": "yellow",
            "degraded": "red",
        }

        for env_id in sorted(self.state.env_states.keys()):
            env = self.state.env_states[env_id]

            step_str = f"{env.current_epoch}/{self.state.max_epochs}"

            # Accuracy with delta indicator
            acc_str = f"{env.host_accuracy:.1f}%"
            if env.best_accuracy > 0:
                if env.host_accuracy >= env.best_accuracy:
                    acc_str = f"[green]{acc_str}[/green]"
                elif env.epochs_since_improvement > 5:
                    acc_str = f"[yellow]{acc_str}[/yellow]"

            # Reward
            reward_str = f"{env.current_reward:+.2f}"
            if env.current_reward > 0:
                reward_str = f"[green]{reward_str}[/green]"
            elif env.current_reward < -0.5:
                reward_str = f"[red]{reward_str}[/red]"

            # Reward components (from last REWARD_COMPUTED)
            base_delta = env.reward_components.get("base_acc_delta")
            if isinstance(base_delta, (int, float)):
                style = "green" if float(base_delta) >= 0 else "red"
                delta_str = f"[{style}]{float(base_delta):+,.2f}[/{style}]"
            else:
                delta_str = "─"

            rent_val = env.reward_components.get("compute_rent")
            if isinstance(rent_val, (int, float)):
                style = "red" if float(rent_val) < 0 else "dim"
                rent_str = f"[{style}]{float(rent_val):+,.2f}[/{style}]"
            else:
                rent_str = "─"

            # Status with styling
            status_style = status_styles.get(env.status, "white")
            status_str = f"[{status_style}]{env.status.upper()}[/{status_style}]"

            # Build row dynamically with all slots
            row_values = [
                str(env_id),
                step_str,
                acc_str,
                env.accuracy_sparkline,
                reward_str,
                env.reward_sparkline,
                delta_str,
                rent_str,
            ]
            # Add dynamic slot cells
            for slot_id in self.state.slot_config.slot_ids:
                row_values.append(self._format_slot_cell(env, slot_id))
            # Add status
            row_values.append(status_str)

            table.add_row(*row_values)

        # Add aggregate row if multiple envs
        if len(self.state.env_states) > 1:
            best_acc, best_env, best_epoch = self.state.aggregate_best_accuracy
            step = max((e.current_epoch for e in self.state.env_states.values()), default=0)
            deltas = [
                float(e.reward_components.get("base_acc_delta"))
                for e in self.state.env_states.values()
                if isinstance(e.reward_components.get("base_acc_delta"), (int, float))
            ]
            rents = [
                float(e.reward_components.get("compute_rent"))
                for e in self.state.env_states.values()
                if isinstance(e.reward_components.get("compute_rent"), (int, float))
            ]
            mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
            mean_rent = sum(rents) / len(rents) if rents else 0.0

            # Build separator row dynamically
            separator_row = [
                "─" * 2,
                "─" * 5,
                "─" * 5,
                "─" * 6,
                "─" * 6,
                "─" * 6,
                "─" * 5,
                "─" * 5,
            ]
            # Add separator for each slot
            for _ in self.state.slot_config.slot_ids:
                separator_row.append("─" * 8)
            # Add separator for status column
            separator_row.append("─" * 7)
            table.add_row(*separator_row)

            # Count slots by stage across all envs
            stage_counts: dict[str, int] = {}
            for env in self.state.env_states.values():
                for seed in env.seeds.values():
                    if seed.stage != "DORMANT":
                        stage_counts[seed.stage] = stage_counts.get(seed.stage, 0) + 1
            stage_summary = " ".join(
                f"{self._STAGE_SHORT.get(s, s[:3])}:{c}"
                for s, c in sorted(stage_counts.items())
            ) or "─"

            # Build aggregate row dynamically
            agg_row = [
                "[bold]Σ[/bold]",
                f"[dim]{step}/{self.state.max_epochs}[/dim]",
                f"[bold]{self.state.aggregate_mean_accuracy:.1f}%[/bold]",
                "",
                f"[bold]{self.state.aggregate_mean_reward:+.2f}[/bold]",
                "",
                f"[dim]{mean_delta:+.2f}[/dim]" if deltas else "─",
                f"[dim]{mean_rent:+.2f}[/dim]" if rents else "─",
            ]
            # Fill slot columns with stage summary in first slot, empty in others
            num_slots = self.state.slot_config.num_slots
            for i in range(num_slots):
                if i == 0:
                    agg_row.append(f"[dim]{stage_summary}[/dim]")
                else:
                    agg_row.append("")
            # Add best accuracy in status column
            agg_row.append(f"[dim]{best_acc:.1f}%[/dim]")

            table.add_row(*agg_row)

        return Panel(
            table,
            title="[bold]ENVIRONMENT OVERVIEW[/bold]",
            border_style="cyan",
        )

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
