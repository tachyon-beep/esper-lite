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

    # Host metrics
    host_accuracy: float = 0.0
    host_accuracy_delta: float = 0.0
    host_loss: float = 0.0

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

    # Seed states
    seeds: dict[str, SeedState] = field(default_factory=dict)
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


# =============================================================================
# TUI Output Backend
# =============================================================================

class TUIOutput:
    """Rich-based TUI output backend for PPO training monitoring.

    Implements the Nissa OutputBackend protocol (emit, close) and displays
    live metrics in a structured terminal dashboard.
    """

    def __init__(self, thresholds: ThresholdConfig | None = None):
        self.thresholds = thresholds or ThresholdConfig()
        self.state = TUIState()
        self.console = Console()
        self._live: Live | None = None
        self._started = False
        self._lock = threading.Lock()

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

        # Route to appropriate handler
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
        """Handle REWARD_COMPUTED event."""
        data = event.data or {}

        total_reward = data.get("total_reward", 0.0)
        self.state.update_reward(total_reward)

        # Track action distribution
        action_name = data.get("action_name", "WAIT")
        if action_name in self.state.action_counts:
            self.state.action_counts[action_name] += 1
            self.state.total_actions += 1

        # Reward components for breakdown
        self.state.reward_components = {
            "accuracy_delta": data.get("base_acc_delta", 0.0),
            "bounded_attr": data.get("bounded_attribution", 0.0),
            "compute_rent": data.get("compute_rent", 0.0),
            "blending_warn": data.get("blending_warning", 0.0),
            "probation_warn": data.get("probation_warning", 0.0),
            "terminal_bonus": data.get("fossilize_terminal_bonus", 0.0),
        }

        # Red flag: reward hacking detection
        acc_delta = data.get("base_acc_delta", 0.0)
        if acc_delta < 0 and total_reward > 0:
            self.state.reward_hacking_detected = True

        # Host accuracy from reward event
        self.state.host_accuracy = data.get("val_acc", self.state.host_accuracy)

    def _handle_seed_event(self, event: "TelemetryEvent", event_type: str) -> None:
        """Handle seed lifecycle events."""
        data = event.data or {}
        slot_id = event.slot_id or data.get("slot_id", "unknown")

        if slot_id not in self.state.seeds:
            self.state.seeds[slot_id] = SeedState(slot_id=slot_id)

        seed = self.state.seeds[slot_id]

        if event_type == "SEED_GERMINATED":
            seed.stage = "GERMINATED"
            seed.blueprint_id = data.get("blueprint_id")
            self.state.active_seed_count += 1
        elif event_type == "SEED_STAGE_CHANGED":
            seed.stage = data.get("to", seed.stage)
            seed.alpha = data.get("alpha", seed.alpha)
        elif event_type == "SEED_FOSSILIZED":
            seed.stage = "FOSSILIZED"
            self.state.fossilized_count += 1
            self.state.active_seed_count = max(0, self.state.active_seed_count - 1)
        elif event_type == "SEED_CULLED":
            seed.stage = "CULLED"
            self.state.culled_count += 1
            self.state.active_seed_count = max(0, self.state.active_seed_count - 1)

    def _handle_batch_completed(self, event: "TelemetryEvent") -> None:
        """Handle BATCH_COMPLETED event (episode completion)."""
        data = event.data or {}
        self.state.current_episode = data.get("episodes_completed", self.state.current_episode)

        # Add to episode rewards for sparkline
        avg_reward = data.get("avg_reward", 0.0)
        self.state.episode_rewards.append(avg_reward)

    # =========================================================================
    # Rendering
    # =========================================================================

    def _render(self) -> Panel:
        """Render the full TUI layout with event log."""
        layout = Layout()

        # Create main sections with event log
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="event_log", size=14),  # Event log panel
            Layout(name="footer", size=3),
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
        layout["event_log"].update(self._render_event_log(max_lines=10))
        layout["footer"].update(self._render_footer())

        return Panel(
            layout,
            title="[bold blue]ESPER-LITE PPO Training Monitor[/bold blue]",
            border_style="blue",
        )

    def _render_header(self) -> Panel:
        """Render the header with episode info."""
        text = Text()
        text.append(f"Episode: ", style="dim")
        text.append(f"{self.state.current_episode}", style="bold cyan")
        text.append(f"  |  Epoch: ", style="dim")
        text.append(f"{self.state.current_epoch}", style="bold cyan")
        text.append(f"  |  Host Accuracy: ", style="dim")
        text.append(f"{self.state.host_accuracy:.1f}%", style="bold green")

        # Add reward hacking warning
        if self.state.reward_hacking_detected:
            text.append("  |  ", style="dim")
            text.append("REWARD HACKING SUSPECTED", style="bold red blink")

        return Panel(text, border_style="dim")

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

    def _render_seeds(self) -> Panel:
        """Render the seed state panel."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Info", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Active:", f"{self.state.active_seed_count} seeds")

        # Show individual seed states
        stage_counts: dict[str, int] = {}
        for seed in self.state.seeds.values():
            stage_counts[seed.stage] = stage_counts.get(seed.stage, 0) + 1

        for stage, count in sorted(stage_counts.items()):
            if stage not in ("DORMANT", "CULLED"):
                marker = self._get_stage_marker(stage)
                table.add_row(f"  {marker} {stage}", f"({count})")

        table.add_row("", "")
        table.add_row("Fossilized:", f"{self.state.fossilized_count}")
        table.add_row("Culled:", f"{self.state.culled_count}")
        table.add_row("Host Acc:", f"{self.state.host_accuracy:.1f}%")

        return Panel(table, title="[bold]SEED STATE[/bold]", border_style="cyan")

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
        text = Text()
        text.append("[q] ", style="bold")
        text.append("Quit  ", style="dim")
        text.append("[Ctrl+C] ", style="bold")
        text.append("Stop Training", style="dim")
        return Panel(text, border_style="dim")

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
