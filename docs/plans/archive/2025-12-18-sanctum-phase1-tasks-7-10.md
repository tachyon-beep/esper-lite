# Sanctum Phase 1 Implementation Plan: Tasks 7-10

> **Scope:** App wiring, telemetry backend, CLI integration, and final cleanup
> **Depends on:** Tasks 1-6 (schema, app shell, widgets)
> **Reference:** Overwatch patterns in `src/esper/karn/overwatch/`

---

## Task 7: Wire Up Complete App

**Goal:** Connect all widgets to backend and implement snapshot propagation.

**Files:**
- Modify: `src/esper/karn/sanctum/app.py`
- Create: `tests/karn/sanctum/test_app_integration.py`
- Reference: `src/esper/karn/overwatch/app.py`

### Step 1: Define Widget Update Protocol

All Sanctum widgets must implement a consistent update interface:

```python
# src/esper/karn/sanctum/widgets/base.py
"""Base class for Sanctum widgets with update protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from textual.widget import Widget

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class SanctumWidget(Protocol):
    """Protocol for Sanctum widgets that receive snapshot updates."""

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data.

        Args:
            snapshot: The current telemetry snapshot.
        """
        ...
```

### Step 2: Complete App Implementation

```python
# src/esper/karn/sanctum/app.py
"""Sanctum TUI Application - Developer diagnostic interface."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from esper.karn.sanctum.widgets import (
    EnvOverview,
    Scoreboard,
    TamiyoBrain,
    RewardComponents,
    EsperStatus,
    EventLog,
)

if TYPE_CHECKING:
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.schema import SanctumSnapshot


class SanctumApp(App):
    """Sanctum TUI - Developer debugging interface for Esper training.

    Provides deep diagnostic inspection when training misbehaves:
    - Per-environment metrics with sparklines
    - Tamiyo policy health and action distribution
    - Reward component breakdown
    - System vitals with CPU fix

    Args:
        backend: SanctumBackend providing snapshot data.
        num_envs: Number of training environments.
        refresh_rate: Snapshot refresh rate in Hz (default: 4).
    """

    TITLE = "Sanctum - Developer Diagnostics"
    SUB_TITLE = "Esper Training Debugger"
    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next", "Next Panel", show=False),
        Binding("shift+tab", "focus_previous", "Prev Panel", show=False),
        Binding("1", "focus_env(0)", "Env 0", show=False),
        Binding("2", "focus_env(1)", "Env 1", show=False),
        Binding("3", "focus_env(2)", "Env 2", show=False),
        Binding("4", "focus_env(3)", "Env 3", show=False),
        Binding("5", "focus_env(4)", "Env 4", show=False),
        Binding("6", "focus_env(5)", "Env 5", show=False),
        Binding("7", "focus_env(6)", "Env 6", show=False),
        Binding("8", "focus_env(7)", "Env 7", show=False),
        Binding("9", "focus_env(8)", "Env 8", show=False),
        Binding("0", "focus_env(9)", "Env 9", show=False),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("?", "toggle_help", "Help", show=True),
    ]

    def __init__(
        self,
        backend: "SanctumBackend",
        num_envs: int = 16,
        refresh_rate: float = 4.0,
    ):
        """Initialize Sanctum app.

        Args:
            backend: SanctumBackend providing snapshot data.
            num_envs: Number of training environments.
            refresh_rate: Snapshot refresh rate in Hz.
        """
        super().__init__()
        self._backend = backend
        self._num_envs = num_envs
        self._refresh_interval = 1.0 / refresh_rate
        self._focused_env_id: int = 0
        self._snapshot: "SanctumSnapshot" | None = None
        self._lock = threading.Lock()

    def compose(self) -> ComposeResult:
        """Build the Sanctum layout.

        Layout matches existing Rich TUI structure:
        - Top row: EnvOverview (65%) | Scoreboard (35%)
        - Middle row: TamiyoBrain (full width, fixed height)
        - Bottom row: RewardComponents + EventLog (65%) | EsperStatus (35%)
        - Footer: Keybindings
        """
        yield Header()

        with Container(id="sanctum-main"):
            # Top section: Environment Overview and Scoreboard
            with Horizontal(id="top-section"):
                yield EnvOverview(num_envs=self._num_envs, id="env-overview")
                yield Scoreboard(id="scoreboard")

            # Middle section: Tamiyo Brain (full width, fixed height)
            yield TamiyoBrain(id="tamiyo-brain")

            # Bottom section: Reward/Event and Esper Status
            with Horizontal(id="bottom-section"):
                with Vertical(id="bottom-left"):
                    yield RewardComponents(id="reward-components")
                    yield EventLog(id="event-log")
                yield EsperStatus(id="esper-status")

        yield Footer()

    def on_mount(self) -> None:
        """Start refresh timer when app mounts."""
        self.set_interval(self._refresh_interval, self._poll_and_refresh)

    def _poll_and_refresh(self) -> None:
        """Poll backend for new snapshot and refresh all panels.

        Called periodically by set_interval timer.
        Thread-safe: backend.get_snapshot() is thread-safe.
        """
        if self._backend is None:
            return

        # Get snapshot from backend (thread-safe)
        snapshot = self._backend.get_snapshot()

        with self._lock:
            self._snapshot = snapshot

        # Update all widgets
        self._refresh_all_panels(snapshot)

    def _refresh_all_panels(self, snapshot: "SanctumSnapshot") -> None:
        """Refresh all panels with new snapshot data.

        Args:
            snapshot: The current telemetry snapshot.
        """
        # Update each widget - query by ID and call update_snapshot
        try:
            self.query_one("#env-overview", EnvOverview).update_snapshot(snapshot)
        except Exception:
            pass  # Widget may not exist yet during startup

        try:
            self.query_one("#scoreboard", Scoreboard).update_snapshot(snapshot)
        except Exception:
            pass

        try:
            self.query_one("#tamiyo-brain", TamiyoBrain).update_snapshot(snapshot)
        except Exception:
            pass

        try:
            # RewardComponents needs focused env
            reward_widget = self.query_one("#reward-components", RewardComponents)
            reward_widget.update_snapshot(snapshot, env_id=self._focused_env_id)
        except Exception:
            pass

        try:
            self.query_one("#event-log", EventLog).update_snapshot(snapshot)
        except Exception:
            pass

        try:
            self.query_one("#esper-status", EsperStatus).update_snapshot(snapshot)
        except Exception:
            pass

    def action_focus_env(self, env_id: int) -> None:
        """Focus on specific environment for detail panels.

        Args:
            env_id: Environment ID to focus (0-indexed).
        """
        if 0 <= env_id < self._num_envs:
            self._focused_env_id = env_id
            # Immediately refresh reward components with new focus
            if self._snapshot:
                try:
                    reward_widget = self.query_one("#reward-components", RewardComponents)
                    reward_widget.update_snapshot(self._snapshot, env_id=env_id)
                except Exception:
                    pass

    def action_refresh(self) -> None:
        """Manually trigger refresh."""
        self._poll_and_refresh()

    def action_toggle_help(self) -> None:
        """Toggle help display."""
        # Textual built-in help
        pass
```

### Step 3: Add Styles for Layout

```css
/* src/esper/karn/sanctum/styles.tcss */

/* Main container fills screen */
#sanctum-main {
    height: 100%;
    width: 100%;
}

/* Top section: EnvOverview 65%, Scoreboard 35% */
#top-section {
    height: 3fr;
    width: 100%;
}

#env-overview {
    width: 65%;
    border: solid $primary;
}

#scoreboard {
    width: 35%;
    border: solid $primary;
}

/* Middle section: TamiyoBrain full width, fixed height */
#tamiyo-brain {
    height: 11;
    width: 100%;
    border: solid $primary;
}

/* Bottom section: Left 65%, Right 35% */
#bottom-section {
    height: 2fr;
    width: 100%;
}

#bottom-left {
    width: 65%;
}

#reward-components {
    height: 1fr;
    border: solid $primary;
}

#event-log {
    height: 1fr;
    border: solid $primary;
}

#esper-status {
    width: 35%;
    border: solid $primary;
}

/* Panel focus states */
.focusable:focus {
    border: double $accent;
}

/* Staleness indicator */
.stale {
    opacity: 0.6;
}

.stale-warning {
    color: $warning;
}

/* A/B test cohort styling (--ab-test shaped-vs-simplified)
   NOTE: Primary color coding uses inline Rich markup (●) for the pip.
   These CSS classes are available for future use (e.g., row backgrounds).

   Color scheme:
   - shaped: blue (default reward function)
   - simplified: yellow/amber (simplified reward function)
   - sparse: cyan (sparse reward function)
*/
.ab-shaped {
    /* Default - no special styling needed */
}

.ab-simplified {
    /* Yellow tint for simplified reward cohort */
    background: $warning 10%;
}

.ab-sparse {
    /* Cyan tint for sparse reward cohort */
    background: $accent 10%;
}
```

### Step 4: Tests

```python
# tests/karn/sanctum/test_app_integration.py
"""Integration tests for SanctumApp."""

import pytest
from unittest.mock import MagicMock, patch

from esper.karn.sanctum.schema import SanctumSnapshot, TamiyoState, EnvState


class TestSanctumAppIntegration:
    """Test SanctumApp widget wiring."""

    def test_app_creates_all_widgets(self):
        """All required widgets should be created on compose."""
        # Import here to avoid Textual import issues in test collection
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=4)

        # Use Textual's pilot for testing
        async def test_widgets():
            async with app.run_test() as pilot:
                # Verify all widgets exist
                assert app.query_one("#env-overview") is not None
                assert app.query_one("#scoreboard") is not None
                assert app.query_one("#tamiyo-brain") is not None
                assert app.query_one("#reward-components") is not None
                assert app.query_one("#event-log") is not None
                assert app.query_one("#esper-status") is not None

        import asyncio
        asyncio.run(test_widgets())

    def test_snapshot_propagates_to_all_widgets(self):
        """Snapshot updates should reach all widgets."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        snapshot = SanctumSnapshot(
            tamiyo=TamiyoState(entropy=1.2, clip_fraction=0.15),
            envs={0: EnvState(env_id=0, host_accuracy=75.5)},
        )
        mock_backend.get_snapshot.return_value = snapshot

        app = SanctumApp(backend=mock_backend, num_envs=4)

        async def test_propagation():
            async with app.run_test() as pilot:
                # Trigger refresh
                await pilot.press("r")

                # Backend should have been called
                mock_backend.get_snapshot.assert_called()

        import asyncio
        asyncio.run(test_propagation())

    def test_focus_env_updates_reward_panel(self):
        """Pressing 1-9 should focus that env in reward panel."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=16)

        async def test_focus():
            async with app.run_test() as pilot:
                # Press '3' to focus env 2 (0-indexed)
                await pilot.press("3")
                assert app._focused_env_id == 2

                # Press '8' to focus env 7
                await pilot.press("8")
                assert app._focused_env_id == 7

        import asyncio
        asyncio.run(test_focus())

    def test_quit_action_exits_app(self):
        """Pressing 'q' should quit the app."""
        from esper.karn.sanctum.app import SanctumApp

        mock_backend = MagicMock()
        mock_backend.get_snapshot.return_value = SanctumSnapshot()

        app = SanctumApp(backend=mock_backend, num_envs=4)

        async def test_quit():
            async with app.run_test() as pilot:
                await pilot.press("q")
                # App should be exiting
                assert app._exit

        import asyncio
        asyncio.run(test_quit())
```

---

## Task 8: Add Telemetry Backend

**Goal:** Create SanctumBackend that transforms telemetry events into SanctumSnapshot.

**Files:**
- Create: `src/esper/karn/sanctum/aggregator.py`
- Create: `src/esper/karn/sanctum/backend.py`
- Create: `tests/karn/sanctum/test_backend.py`
- Reference: `src/esper/karn/overwatch/aggregator.py`, `src/esper/karn/tui.py`

### Step 1: Create Aggregator

The aggregator maintains state and transforms events into SanctumSnapshot.
It handles ALL events from the existing TUIOutput class.

```python
# src/esper/karn/sanctum/aggregator.py
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

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        """Initialize state."""
        self._envs = {}
        self._event_log = deque(maxlen=self.max_event_log)
        self._tamiyo = TamiyoState()
        self._vitals = SystemVitals()
        self._gpu_devices = []
        self._start_time = time.time()
        self._lock = threading.Lock()

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

        return SanctumSnapshot(
            # Run context
            run_id=self._run_id,
            task_name=self._task_name,
            current_episode=self._current_episode,
            current_epoch=self._current_epoch,
            max_epochs=self._max_epochs,
            runtime_seconds=runtime,
            connected=self._connected,
            staleness_seconds=staleness,
            captured_at=datetime.now(timezone.utc).isoformat(),
            # Per-env state
            envs=dict(self._envs),
            focused_env_id=self._focused_env_id,
            # Tamiyo state
            tamiyo=self._tamiyo,
            # System vitals
            vitals=self._vitals,
            # Event log
            event_log=list(self._event_log),
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

        # Reset and recreate env states
        self._envs.clear()
        for i in range(self.num_envs):
            self._ensure_env(i)

        # Reset Tamiyo state
        self._tamiyo = TamiyoState()

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

        env.last_update = datetime.now(timezone.utc)

    def _handle_ppo_update(self, event: "TelemetryEvent") -> None:
        """Handle PPO_UPDATE_COMPLETED event."""
        data = event.data or {}

        if data.get("skipped"):
            return

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
            bounded_attribution=data.get("bounded_attribution"),
            seed_contribution=data.get("seed_contribution"),
            compute_rent=data.get("compute_rent", 0.0),
            ratio_penalty=data.get("ratio_penalty", 0.0),
            stage_bonus=data.get("stage_bonus", 0.0),
            fossilize_terminal_bonus=data.get("fossilize_terminal_bonus", 0.0),
            blending_warning=data.get("blending_warning", 0.0),
            probation_warning=data.get("probation_warning", 0.0),
            val_acc=data.get("val_acc", env.host_accuracy),
            total_reward=total_reward,
            last_action=action_name,
            env_id=env_id,
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

        elif event_type == "SEED_FOSSILIZED":
            seed.stage = "FOSSILIZED"
            env.fossilized_params += int(data.get("params_added", 0) or 0)
            env.fossilized_count += 1
            env.active_seed_count = max(0, env.active_seed_count - 1)

        elif event_type == "SEED_CULLED":
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

        episodes_completed = data.get("episodes_completed")
        if isinstance(episodes_completed, (int, float)):
            self._current_episode = int(episodes_completed)
        self._batches_completed += 1

        # Calculate throughput
        now = time.time()
        elapsed = now - self._start_time
        if elapsed > 0:
            total_epochs = self._current_episode * self._max_epochs
            self._vitals.epochs_per_second = total_epochs / elapsed
            self._vitals.batches_per_hour = (self._batches_completed / elapsed) * 3600

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
                accuracy_history=deque(maxlen=self.max_history),
                reward_history=deque(maxlen=self.max_history),
                action_history=deque(maxlen=self.max_history),
            )

    def _add_event_log(self, event: "TelemetryEvent", event_type: str) -> None:
        """Add event to log with formatting."""
        data = event.data or {}
        env_id = data.get("env_id")
        timestamp = event.timestamp or datetime.now(timezone.utc)

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
                message = f"{slot_id} fossilized"
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
```

### Step 2: Create Backend

```python
# src/esper/karn/sanctum/backend.py
"""Sanctum Backend - OutputBackend for live telemetry.

Implements Nissa's OutputBackend protocol to receive telemetry
events and update the SanctumAggregator for TUI consumption.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from esper.karn.sanctum.aggregator import SanctumAggregator

if TYPE_CHECKING:
    from esper.leyline import TelemetryEvent
    from esper.karn.sanctum.schema import SanctumSnapshot


class SanctumBackend:
    """OutputBackend that feeds telemetry to Sanctum TUI.

    Thread-safe: emit() can be called from training thread while
    get_snapshot() is called from UI thread (aggregator handles locking).

    Usage:
        from esper.nissa import get_hub
        from esper.karn.sanctum.backend import SanctumBackend

        backend = SanctumBackend(num_envs=16)
        get_hub().add_backend(backend)

        # In UI thread
        snapshot = backend.get_snapshot()
    """

    def __init__(self, num_envs: int = 16, max_event_log: int = 100):
        """Initialize the backend.

        Args:
            num_envs: Expected number of training environments.
            max_event_log: Maximum events to keep in log.
        """
        self._aggregator = SanctumAggregator(
            num_envs=num_envs,
            max_event_log=max_event_log,
        )
        self._started = False

    def start(self) -> None:
        """Start the backend (required by OutputBackend protocol)."""
        self._started = True

    def emit(self, event: "TelemetryEvent") -> None:
        """Emit telemetry event to aggregator.

        Args:
            event: The telemetry event to process.
        """
        if not self._started:
            return
        self._aggregator.process_event(event)

    def close(self) -> None:
        """Close the backend (required by OutputBackend protocol)."""
        self._started = False

    def get_snapshot(self) -> "SanctumSnapshot":
        """Get current SanctumSnapshot for UI rendering.

        Returns:
            Snapshot of current aggregator state.
        """
        return self._aggregator.get_snapshot()
```

### Step 3: Tests

```python
# tests/karn/sanctum/test_backend.py
"""Tests for SanctumBackend and SanctumAggregator."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from esper.karn.sanctum.backend import SanctumBackend
from esper.karn.sanctum.aggregator import SanctumAggregator, normalize_action


class TestActionNormalization:
    """Test action name normalization."""

    def test_factored_germinate_normalizes(self):
        """Factored GERMINATE actions should normalize."""
        assert normalize_action("GERMINATE_CONV_LIGHT") == "GERMINATE"
        assert normalize_action("GERMINATE_CONV_HEAVY") == "GERMINATE"
        assert normalize_action("GERMINATE_ATTENTION") == "GERMINATE"

    def test_factored_cull_normalizes(self):
        """Factored CULL actions should normalize."""
        assert normalize_action("CULL_PROBATION") == "CULL"
        assert normalize_action("CULL_STAGNATION") == "CULL"

    def test_base_actions_unchanged(self):
        """Base actions should remain unchanged."""
        assert normalize_action("WAIT") == "WAIT"
        assert normalize_action("GERMINATE") == "GERMINATE"
        assert normalize_action("FOSSILIZE") == "FOSSILIZE"


class TestSanctumAggregator:
    """Test SanctumAggregator event processing."""

    def test_training_started_initializes_state(self):
        """TRAINING_STARTED should initialize run context."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "TRAINING_STARTED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = {
            "run_id": "test-run-123",
            "task": "mnist",
            "max_epochs": 50,
            "n_envs": 4,
        }

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        assert snapshot.run_id == "test-run-123"
        assert snapshot.task_name == "mnist"
        assert snapshot.max_epochs == 50
        assert snapshot.connected is True
        assert len(snapshot.envs) == 4

    def test_ppo_update_updates_tamiyo_state(self):
        """PPO_UPDATE_COMPLETED should update Tamiyo metrics."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "PPO_UPDATE_COMPLETED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = {
            "entropy": 1.2,
            "clip_fraction": 0.15,
            "kl_divergence": 0.02,
            "explained_variance": 0.8,
            "policy_loss": -0.05,
            "value_loss": 0.1,
            "grad_norm": 2.5,
            "lr": 3e-4,
            "dead_layers": 0,
            "exploding_layers": 1,
        }

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        assert snapshot.tamiyo.entropy == 1.2
        assert snapshot.tamiyo.clip_fraction == 0.15
        assert snapshot.tamiyo.kl_divergence == 0.02
        assert snapshot.tamiyo.explained_variance == 0.8
        assert snapshot.tamiyo.learning_rate == 3e-4
        assert snapshot.tamiyo.dead_layers == 0
        assert snapshot.tamiyo.exploding_layers == 1

    def test_reward_computed_updates_env_state(self):
        """REWARD_COMPUTED should update per-env state."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "REWARD_COMPUTED"
        event.timestamp = datetime.now(timezone.utc)
        event.epoch = 10
        event.data = {
            "env_id": 2,
            "total_reward": 0.5,
            "action_name": "GERMINATE_CONV_LIGHT",
            "base_acc_delta": 0.1,
            "compute_rent": -0.01,
            "val_acc": 75.5,
        }

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[2]
        assert len(env.reward_history) == 1
        assert env.reward_history[0] == 0.5
        assert env.action_counts["GERMINATE"] == 1  # Normalized
        assert env.reward_components.base_acc_delta == 0.1
        assert env.reward_components.compute_rent == -0.01

    def test_seed_germinated_adds_seed(self):
        """SEED_GERMINATED should add seed to env."""
        agg = SanctumAggregator(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "SEED_GERMINATED"
        event.timestamp = datetime.now(timezone.utc)
        event.slot_id = "r0c1"
        event.data = {
            "env_id": 0,
            "blueprint_id": "conv_light",
            "params": 1000,
        }

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        env = snapshot.envs[0]
        assert "r0c1" in env.seeds
        assert env.seeds["r0c1"].stage == "GERMINATED"
        assert env.seeds["r0c1"].blueprint_id == "conv_light"
        assert env.active_seed_count == 1

    def test_seed_fossilized_updates_counts(self):
        """SEED_FOSSILIZED should increment counters."""
        agg = SanctumAggregator(num_envs=4)

        # First germinate
        germ_event = MagicMock()
        germ_event.event_type = MagicMock()
        germ_event.event_type.name = "SEED_GERMINATED"
        germ_event.timestamp = datetime.now(timezone.utc)
        germ_event.slot_id = "r0c0"
        germ_event.data = {"env_id": 0, "blueprint_id": "conv_light"}
        agg.process_event(germ_event)

        # Then fossilize
        foss_event = MagicMock()
        foss_event.event_type = MagicMock()
        foss_event.event_type.name = "SEED_FOSSILIZED"
        foss_event.timestamp = datetime.now(timezone.utc)
        foss_event.slot_id = "r0c0"
        foss_event.data = {"env_id": 0, "params_added": 5000}
        agg.process_event(foss_event)

        snapshot = agg.get_snapshot()
        env = snapshot.envs[0]

        assert env.fossilized_count == 1
        assert env.fossilized_params == 5000
        assert env.active_seed_count == 0  # Decremented
        assert env.seeds["r0c0"].stage == "FOSSILIZED"

    def test_event_log_captures_events(self):
        """Events should be logged."""
        agg = SanctumAggregator(num_envs=4, max_event_log=10)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "REWARD_COMPUTED"
        event.timestamp = datetime.now(timezone.utc)
        event.message = None
        event.data = {
            "env_id": 0,
            "total_reward": 0.5,
            "action_name": "WAIT",
        }

        agg.process_event(event)
        snapshot = agg.get_snapshot()

        assert len(snapshot.event_log) == 1
        assert snapshot.event_log[0].event_type == "REWARD_COMPUTED"
        assert "WAIT" in snapshot.event_log[0].message


class TestSanctumBackend:
    """Test SanctumBackend OutputBackend protocol."""

    def test_backend_implements_protocol(self):
        """Backend should implement start/emit/close."""
        backend = SanctumBackend(num_envs=4)

        assert hasattr(backend, "start")
        assert hasattr(backend, "emit")
        assert hasattr(backend, "close")
        assert hasattr(backend, "get_snapshot")

    def test_emit_ignored_before_start(self):
        """Events should be ignored before start()."""
        backend = SanctumBackend(num_envs=4)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "TRAINING_STARTED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = {"run_id": "test"}

        # Emit before start
        backend.emit(event)
        snapshot = backend.get_snapshot()

        # Should not have processed
        assert snapshot.run_id == ""

    def test_emit_processed_after_start(self):
        """Events should be processed after start()."""
        backend = SanctumBackend(num_envs=4)
        backend.start()

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "TRAINING_STARTED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = {"run_id": "test-run"}

        backend.emit(event)
        snapshot = backend.get_snapshot()

        assert snapshot.run_id == "test-run"

    def test_close_stops_processing(self):
        """Events should be ignored after close()."""
        backend = SanctumBackend(num_envs=4)
        backend.start()
        backend.close()

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.name = "TRAINING_STARTED"
        event.timestamp = datetime.now(timezone.utc)
        event.data = {"run_id": "ignored"}

        backend.emit(event)
        snapshot = backend.get_snapshot()

        assert snapshot.run_id == ""
```

---

## Task 9: Add CLI Integration

**Goal:** Add `--sanctum` flag to train.py, mutually exclusive with `--overwatch`.

**Files:**
- Modify: `src/esper/scripts/train.py`
- Create: `tests/scripts/test_train_sanctum_flag.py`
- Reference: Existing `--overwatch` integration (lines 69-72, 320-348, 433-443)

### Step 1: Add Argument

```python
# In src/esper/scripts/train.py, around line 69-72
# Add after --overwatch argument:

telemetry_parent.add_argument(
    "--sanctum",
    action="store_true",
    help="Launch Sanctum TUI for developer debugging (replaces Rich TUI, mutually exclusive with --overwatch)",
)
```

### Step 2: Add Mutual Exclusion Check

```python
# In main() function, after parsing args (around line 310):

# Check mutual exclusion
if args.overwatch and args.sanctum:
    parser.error("--overwatch and --sanctum are mutually exclusive. Choose one.")

use_overwatch = args.overwatch
use_sanctum = args.sanctum
```

### Step 3: Add Backend Setup

```python
# After the Overwatch backend setup (around line 350):

sanctum_backend = None
if use_sanctum:
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.leyline import DEFAULT_N_ENVS

    # Determine num_envs for Sanctum display (same logic as Overwatch)
    if args.algorithm == "ppo":
        ppo_config_path = Path(args.config) if args.config else None
        if ppo_config_path and ppo_config_path.exists():
            import json
            with open(ppo_config_path) as f:
                config = json.load(f)
            ppo_params = config.get("ppo", {})
            num_envs = ppo_params.get("n_envs", DEFAULT_N_ENVS)
        else:
            num_envs = DEFAULT_N_ENVS
    else:
        num_envs = DEFAULT_N_ENVS

    sanctum_backend = SanctumBackend(num_envs=num_envs)
    hub.add_backend(sanctum_backend)
```

### Step 4: Add App Launch

```python
# In the try block where training runs (around line 433-446):

if use_overwatch:
    # Overwatch mode (existing)
    import threading
    from esper.karn.overwatch import OverwatchApp

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    app = OverwatchApp(backend=overwatch_backend)
    app.run()

elif use_sanctum:
    # Sanctum mode (new)
    import threading
    from esper.karn.sanctum import SanctumApp

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()

    app = SanctumApp(backend=sanctum_backend, num_envs=num_envs)
    app.run()

else:
    # Normal mode: run training directly in main thread
    run_training()
```

### Step 5: Tests

```python
# tests/scripts/test_train_sanctum_flag.py
"""Tests for --sanctum CLI flag."""

import pytest
import subprocess
import sys


class TestSanctumCLIFlag:
    """Test --sanctum CLI integration."""

    def test_sanctum_flag_exists(self):
        """--sanctum flag should be recognized."""
        result = subprocess.run(
            [sys.executable, "-m", "esper.scripts.train", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--sanctum" in result.stdout

    def test_mutual_exclusion_error(self):
        """--sanctum and --overwatch together should error."""
        result = subprocess.run(
            [
                sys.executable, "-m", "esper.scripts.train",
                "ppo", "--overwatch", "--sanctum",
                "--episodes", "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr.lower()

    def test_sanctum_flag_accepted_alone(self):
        """--sanctum alone should be accepted (will fail without config, but flag parses)."""
        # Just test the flag is parsed, not that training runs
        result = subprocess.run(
            [
                sys.executable, "-m", "esper.scripts.train",
                "ppo", "--sanctum", "--help",
            ],
            capture_output=True,
            text=True,
        )
        # Help should work with --sanctum present
        assert result.returncode == 0
```

---

## Task 10: Final Verification and Cleanup

**Goal:** Verify complete port, delete legacy TUI, update exports.

**Files:**
- Delete: `src/esper/karn/tui.py`
- Modify: `src/esper/karn/__init__.py`
- Modify: `src/esper/karn/sanctum/__init__.py`
- Create: `tests/karn/sanctum/test_final_verification.py`

### Step 1: Verification Checklist

Before deleting `tui.py`, verify ALL of the following:

```markdown
## Pre-Deletion Verification Checklist

### Unit Tests
- [ ] `pytest tests/karn/sanctum/ -v` - All Sanctum tests pass
- [ ] `pytest tests/karn/ -v` - All Karn tests still pass

### Integration Tests
- [ ] `pytest tests/integration/ -v` - Integration tests pass

### Manual Verification
Launch both TUIs and compare:

1. **Header/Context**
   - [ ] Run ID displays correctly
   - [ ] Task name displays correctly
   - [ ] Episode counter updates
   - [ ] Runtime displays correctly

2. **Environment Overview**
   - [ ] All 16 envs visible
   - [ ] Accuracy updates and sparklines work
   - [ ] Reward updates and sparklines work
   - [ ] Slot cells show stage:blueprint:epochs
   - [ ] Gradient indicators (▼▲) display
   - [ ] Status updates (improving/stagnant)

3. **Scoreboard (Best Runs)**
   - [ ] Global best accuracy shows
   - [ ] Mean best accuracy shows
   - [ ] Fossilized/Culled counts show
   - [ ] Top 10 leaderboard populates
   - [ ] Medals (🥇🥈🥉) display correctly
   - [ ] Seeds at best accuracy show

4. **Tamiyo Brain**
   - [ ] Entropy displays with color coding
   - [ ] Clip fraction displays with color coding
   - [ ] KL divergence displays
   - [ ] Explained variance displays
   - [ ] All losses display
   - [ ] Gradient norm displays
   - [ ] Learning rate displays
   - [ ] Gradient health (GradHP) displays
   - [ ] Action distribution percentages display

5. **Reward Components**
   - [ ] Focused env updates on keypress (1-9)
   - [ ] All 10 reward components display
   - [ ] Color coding correct (green/red/blue/yellow)
   - [ ] Total reward displays

6. **Esper Status**
   - [ ] Seed stage counts display
   - [ ] Host params display (formatted K/M)
   - [ ] Throughput displays (epochs/sec)
   - [ ] Runtime displays
   - [ ] GPU memory displays (multi-GPU if applicable)
   - [ ] GPU utilization displays
   - [ ] RAM displays
   - [ ] **CPU displays (THE FIX)**

7. **Event Log**
   - [ ] Events appear in real-time
   - [ ] Color coding by event type
   - [ ] Scrolls/truncates correctly

8. **Keyboard Navigation**
   - [ ] Tab cycles panels
   - [ ] 1-9 focuses envs
   - [ ] q quits
   - [ ] r refreshes
```

### Step 2: Update Sanctum Exports

```python
# src/esper/karn/sanctum/__init__.py
"""Sanctum - Developer diagnostic TUI for Esper training.

Provides deep inspection when training misbehaves:
- Per-environment metrics with sparklines
- Tamiyo policy health and action distribution
- Reward component breakdown
- System vitals with CPU display

Usage:
    from esper.karn.sanctum import SanctumBackend
    hub.add_backend(SanctumBackend(num_envs=16))

    # Launch TUI
    from esper.karn.sanctum import SanctumApp
    app = SanctumApp(backend=backend)
    app.run()
"""

from esper.karn.sanctum.schema import (
    SanctumSnapshot,
    EnvState,
    SeedState,
    TamiyoState,
    SystemVitals,
    GPUStats,
    RewardComponents,
    EventLogEntry,
)

from esper.karn.sanctum.aggregator import SanctumAggregator
from esper.karn.sanctum.backend import SanctumBackend

# Lazy import for SanctumApp - Textual may not be installed
try:
    from esper.karn.sanctum.app import SanctumApp
except ImportError:
    SanctumApp = None  # type: ignore[misc, assignment]

__all__ = [
    # Schema
    "SanctumSnapshot",
    "EnvState",
    "SeedState",
    "TamiyoState",
    "SystemVitals",
    "GPUStats",
    "RewardComponents",
    "EventLogEntry",
    # Aggregator & Backend
    "SanctumAggregator",
    "SanctumBackend",
    # App (may be None if Textual not installed)
    "SanctumApp",
]
```

### Step 3: Update Karn Exports

After deleting `tui.py`, update `karn/__init__.py`:

```python
# src/esper/karn/__init__.py
# Remove these lines (lines 89-95):
# from esper.karn.tui import (
#     TUIOutput,
#     TUIState,
#     ThresholdConfig,
#     HealthStatus,
# )

# Add Sanctum exports:
from esper.karn.sanctum.backend import SanctumBackend

# Update __all__ to remove TUI exports and add Sanctum:
__all__ = [
    # ... existing exports ...
    # Remove: "TUIOutput", "TUIState", "ThresholdConfig", "HealthStatus",
    # Add:
    "SanctumBackend",
]
```

### Step 4: Delete Legacy TUI

```bash
# Only after ALL verification passes:
git rm src/esper/karn/tui.py
```

### Step 5: Final Tests

```python
# tests/karn/sanctum/test_final_verification.py
"""Final verification tests for Sanctum port completion."""

import pytest


class TestSanctumPortComplete:
    """Verify Sanctum is a complete port of the Rich TUI."""

    def test_tui_py_deleted(self):
        """Legacy tui.py should no longer exist."""
        from pathlib import Path
        tui_path = Path(__file__).parents[3] / "src" / "esper" / "karn" / "tui.py"
        assert not tui_path.exists(), "tui.py should be deleted after Sanctum port"

    def test_sanctum_exports_available(self):
        """Sanctum exports should be available from karn."""
        from esper.karn.sanctum import (
            SanctumSnapshot,
            SanctumBackend,
            SanctumAggregator,
        )
        assert SanctumSnapshot is not None
        assert SanctumBackend is not None
        assert SanctumAggregator is not None

    def test_sanctum_app_importable(self):
        """SanctumApp should be importable (may be None without Textual)."""
        from esper.karn.sanctum import SanctumApp
        # SanctumApp may be None if Textual not installed, but import should work

    def test_cli_flag_works(self):
        """--sanctum flag should be recognized in train.py."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "esper.scripts.train", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--sanctum" in result.stdout
```

---

## Summary

| Task | Description | Tests | Key Implementation |
|------|-------------|-------|-------------------|
| 7 | Wire up complete app | 4 | Widget protocol, refresh loop, focus management |
| 8 | Add telemetry backend | 12+ | SanctumAggregator handles ALL TUIOutput events |
| 9 | Add CLI integration | 3 | `--sanctum` flag, mutual exclusion with `--overwatch` |
| 10 | Final verification | 4 | Checklist, delete tui.py, update exports |

**Total: ~23+ tests across 4 tasks**

**Key patterns from Overwatch reused:**
- Backend delegates to aggregator for thread safety
- Lazy import for App (Textual may not be installed)
- Threading model: training in background, TUI in main thread
- OutputBackend protocol: start(), emit(), close()

**Key differences from Overwatch:**
- More detailed schema (SanctumSnapshot has ALL TUIOutput fields)
- Action normalization (factored actions → base actions)
- CPU display (THE FIX - was collected but never shown)
- Denser information display (developer not operator focus)
