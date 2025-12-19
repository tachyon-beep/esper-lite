# Sanctum Phase 1: Rich to Textual Port

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port the existing Rich-based Karn TUI (`karn/tui.py`) to Textual as `karn/sanctum/`, preserving ALL functionality with minor enhancements from specialist review.

**Key Principle:** This is a **1:1 port**. Every feature in the existing TUI must work in Sanctum. No simplification.

**Architecture:** Create `karn/sanctum/` as peer to `karn/overwatch/`. Port each Rich panel as a Textual widget. Reuse telemetry backend patterns. Delete `karn/tui.py` when port complete.

**Tech Stack:** Python 3.11+, Textual 0.x, pytest, existing Nissa telemetry hub

**Reference Files:**
- Design doc: `docs/plans/2025-12-18-sanctum-design.md`
- Existing TUI: `src/esper/karn/tui.py` (~2100 lines)
- Thresholds: `src/esper/karn/constants.py` (TUIThresholds)
- Overwatch patterns: `src/esper/karn/overwatch/` (for reuse)

---

## Task 1: Create Directory Structure and Complete Schema

**Files:**
- Create: `src/esper/karn/sanctum/__init__.py`
- Create: `src/esper/karn/sanctum/schema.py`
- Create: `src/esper/karn/sanctum/widgets/__init__.py`
- Create: `tests/karn/sanctum/__init__.py`
- Create: `tests/karn/sanctum/test_schema.py`

**Step 1: Create directory structure**

```bash
mkdir -p src/esper/karn/sanctum/widgets
mkdir -p tests/karn/sanctum
touch src/esper/karn/sanctum/__init__.py
touch src/esper/karn/sanctum/widgets/__init__.py
touch tests/karn/sanctum/__init__.py
```

**Step 2: Write schema tests**

Create `tests/karn/sanctum/test_schema.py`:

```python
"""Tests for Sanctum schema - must match all existing TUI state."""
import pytest
from collections import deque

from esper.karn.sanctum.schema import (
    SanctumSnapshot,
    EnvState,
    SeedState,
    TamiyoState,
    SystemVitals,
    RewardComponents,
    GPUStats,
)


class TestSeedState:
    """SeedState must capture all per-slot data from existing TUI."""

    def test_defaults(self):
        seed = SeedState(slot_id="r0c0")
        assert seed.slot_id == "r0c0"
        assert seed.stage == "DORMANT"
        assert seed.alpha == 0.0

    def test_gradient_health_flags(self):
        """Must track vanishing/exploding gradients for ▼▲ indicators."""
        seed = SeedState(
            slot_id="r0c0",
            has_vanishing=True,
            has_exploding=False,
        )
        assert seed.has_vanishing is True
        assert seed.has_exploding is False

    def test_epochs_in_stage(self):
        """Must track epochs for 'e5' display in slot cell."""
        seed = SeedState(slot_id="r0c0", epochs_in_stage=5)
        assert seed.epochs_in_stage == 5


class TestEnvState:
    """EnvState must capture all per-environment data."""

    def test_defaults(self):
        env = EnvState(env_id=0)
        assert env.env_id == 0
        assert env.host_accuracy == 0.0
        assert env.seeds == {}

    def test_history_tracking(self):
        """Must support sparkline generation."""
        env = EnvState(env_id=0)
        env.accuracy_history.append(75.0)
        env.accuracy_history.append(76.5)
        env.reward_history.append(0.5)
        assert len(env.accuracy_history) == 2
        assert len(env.reward_history) == 1

    def test_action_tracking(self):
        """Must track per-env action distribution."""
        env = EnvState(env_id=0)
        env.action_counts["WAIT"] = 10
        env.action_counts["GERMINATE"] = 5
        assert env.action_counts["WAIT"] == 10

    def test_best_seeds_snapshot(self):
        """Must preserve seeds at best accuracy."""
        env = EnvState(env_id=0)
        env.best_seeds["r0c0"] = SeedState(slot_id="r0c0", stage="FOSSILIZED")
        assert "r0c0" in env.best_seeds


class TestTamiyoState:
    """TamiyoState must capture all policy agent metrics."""

    def test_defaults(self):
        tamiyo = TamiyoState()
        assert tamiyo.entropy == 0.0
        assert tamiyo.clip_fraction == 0.0

    def test_learning_rate(self):
        """Must track LR for Vitals display."""
        tamiyo = TamiyoState(learning_rate=3e-4)
        assert tamiyo.learning_rate == 3e-4

    def test_gradient_health_metrics(self):
        """Must track dead/exploding layers and GradHP."""
        tamiyo = TamiyoState(
            dead_layers=2,
            exploding_layers=0,
            layer_gradient_health=0.85,
        )
        assert tamiyo.dead_layers == 2
        assert tamiyo.layer_gradient_health == 0.85

    def test_ratio_stats(self):
        """Must track all ratio statistics including std."""
        tamiyo = TamiyoState(
            ratio_mean=1.02,
            ratio_min=0.8,
            ratio_max=1.5,
            ratio_std=0.15,
        )
        assert tamiyo.ratio_std == 0.15


class TestSystemVitals:
    """SystemVitals must capture all system metrics."""

    def test_cpu_percent(self):
        """CPU was collected but never displayed - must fix."""
        vitals = SystemVitals(cpu_percent=67.0)
        assert vitals.cpu_percent == 67.0

    def test_multi_gpu(self):
        """Must support multiple GPUs."""
        vitals = SystemVitals()
        vitals.gpu_stats[0] = GPUStats(device_id=0, memory_used_gb=12.0)
        vitals.gpu_stats[1] = GPUStats(device_id=1, memory_used_gb=8.0)
        assert len(vitals.gpu_stats) == 2

    def test_throughput(self):
        """Must track epochs/sec and batches/hr."""
        vitals = SystemVitals(
            epochs_per_second=2.5,
            batches_per_hour=150.0,
        )
        assert vitals.epochs_per_second == 2.5


class TestRewardComponents:
    """RewardComponents must match Esper reward breakdown."""

    def test_esper_components(self):
        """Must have Esper-specific reward components."""
        rewards = RewardComponents(
            base_acc_delta=0.5,
            bounded_attribution=0.3,
            compute_rent=-0.1,
            ratio_penalty=-0.05,
            stage_bonus=0.2,
            fossilize_terminal_bonus=1.0,
            blending_warning=-0.1,
            probation_warning=0.0,
        )
        assert rewards.base_acc_delta == 0.5
        assert rewards.compute_rent == -0.1


class TestSanctumSnapshot:
    """SanctumSnapshot aggregates all state."""

    def test_creation(self):
        snapshot = SanctumSnapshot(
            envs={0: EnvState(env_id=0)},
            tamiyo=TamiyoState(),
            vitals=SystemVitals(),
        )
        assert len(snapshot.envs) == 1

    def test_staleness_detection(self):
        """Must detect stale data (>5s since update)."""
        snapshot = SanctumSnapshot()
        assert snapshot.is_stale is True  # No update yet

    def test_slot_config(self):
        """Must track dynamic slot configuration."""
        snapshot = SanctumSnapshot(slot_ids=["r0c0", "r0c1", "r1c0", "r1c1"])
        assert len(snapshot.slot_ids) == 4
```

**Step 3: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py -v
```

Expected: FAIL with "No module named 'esper.karn.sanctum.schema'"

**Step 4: Write complete schema implementation**

Create `src/esper/karn/sanctum/schema.py`:

```python
"""Sanctum Schema - Complete state objects matching existing Rich TUI.

These dataclasses mirror ALL state tracked by karn/tui.py for 1:1 port.
Reference: src/esper/karn/tui.py (EnvState, SeedState, TUIState, GPUStats)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SeedState:
    """State of a single seed slot.

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

    Reference: tui.py lines 124-214 (EnvState dataclass)
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
    culled_count: int = 0

    # Reward component breakdown (from REWARD_COMPUTED telemetry)
    reward_components: dict[str, float | None] = field(default_factory=dict)

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


@dataclass
class TamiyoState:
    """Tamiyo policy agent state - ALL metrics from existing TUI.

    Reference: tui.py TUIState policy metrics + _render_tamiyo_brain()
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
    ratio_mean: float = 1.0
    ratio_min: float = 1.0
    ratio_max: float = 1.0
    ratio_std: float = 0.0  # Standard deviation of ratio

    # Gradient health (shown in Vitals)
    dead_layers: int = 0
    exploding_layers: int = 0
    layer_gradient_health: float = 1.0  # GradHP percentage (0-1)

    # Action distribution (Actions panel)
    action_counts: dict[str, int] = field(default_factory=dict)

    # PPO data received flag
    ppo_data_received: bool = False


@dataclass
class SystemVitals:
    """System resource metrics - ALL from existing TUI.

    Reference: tui.py _render_esper_status() and _update_system_stats()
    """
    # Multi-GPU support
    gpu_stats: dict[int, GPUStats] = field(default_factory=dict)

    # Legacy single-GPU fields (for backward compat)
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0

    # CPU (FIX: was collected but never displayed!)
    cpu_percent: float = 0.0

    # RAM
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0

    # Throughput
    epochs_per_second: float = 0.0
    batches_per_hour: float = 0.0
    steps_per_second: float = 0.0

    # Host network
    host_params: int = 0


@dataclass
class RewardComponents:
    """Esper-specific reward signal breakdown.

    Reference: tui.py _render_reward_components() lines 1513-1586
    These are the ACTUAL Esper reward components, not generic ones.
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
    ratio_penalty: float = 0.0

    # Bonuses
    stage_bonus: float = 0.0
    fossilize_terminal_bonus: float = 0.0

    # Warnings
    blending_warning: float = 0.0
    probation_warning: float = 0.0

    # Context
    env_id: int = 0
    val_acc: float = 0.0
    last_action: str = ""


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
    current_epoch: int = 0
    max_epochs: int = 0
    run_id: str = ""
    task_name: str = ""
    start_time: datetime | None = None

    # Aggregates (computed from envs)
    aggregate_mean_accuracy: float = 0.0
    aggregate_mean_reward: float = 0.0

    # Timestamps for staleness detection
    last_ppo_update: datetime | None = None
    last_reward_update: datetime | None = None

    # Focused env for detail panel
    focused_env_id: int = 0

    @property
    def is_stale(self) -> bool:
        """Check if data is stale (>5s since last update)."""
        if self.last_ppo_update is None:
            return True
        age = (datetime.now() - self.last_ppo_update).total_seconds()
        return age > 5.0

    @property
    def staleness_seconds(self) -> float:
        """Get seconds since last update."""
        if self.last_ppo_update is None:
            return float('inf')
        return (datetime.now() - self.last_ppo_update).total_seconds()


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
```

**Step 5: Run tests to verify they pass**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py -v
```

Expected: All tests PASS

**Step 6: Update sanctum __init__.py**

Create `src/esper/karn/sanctum/__init__.py`:

```python
"""Karn Sanctum - Developer Diagnostic TUI.

Sanctum provides deep inspection of PPO training for debugging
misbehaving runs. It complements Overwatch (operator monitoring)
with detailed diagnostic panels.

Usage:
    python -m esper.scripts.train ppo --sanctum
"""
from esper.karn.sanctum.schema import (
    SanctumSnapshot,
    EnvState,
    SeedState,
    TamiyoState,
    SystemVitals,
    RewardComponents,
    GPUStats,
    make_sparkline,
)

__all__ = [
    "SanctumSnapshot",
    "EnvState",
    "SeedState",
    "TamiyoState",
    "SystemVitals",
    "RewardComponents",
    "GPUStats",
    "make_sparkline",
]
```

**Step 7: Commit**

```bash
git add src/esper/karn/sanctum/ tests/karn/sanctum/
git commit -m "feat(sanctum): add complete schema matching existing TUI

Port ALL state classes from Rich TUI to Sanctum schema:
- SeedState with gradient health flags (▼▲ indicators)
- EnvState with history deques for sparklines
- TamiyoState with LR, ratio_std, dead/exploding layers, GradHP
- SystemVitals with multi-GPU support and CPU fix
- RewardComponents with Esper-specific breakdown
- GPUStats for multi-GPU tracking
- make_sparkline() helper function

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Create Base App Shell

**Files:**
- Create: `src/esper/karn/sanctum/app.py`
- Create: `src/esper/karn/sanctum/styles.tcss`
- Create: `tests/karn/sanctum/test_app.py`

**Step 1: Write app tests**

Create `tests/karn/sanctum/test_app.py`:

```python
"""Tests for Sanctum Textual app."""
import pytest

from esper.karn.sanctum.app import SanctumApp


@pytest.mark.asyncio
async def test_app_launches():
    """App should launch without errors."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        assert app.title == "Esper Sanctum"


@pytest.mark.asyncio
async def test_app_has_main_panels():
    """App should have all required panels."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        # Main panels from existing TUI layout
        assert app.query_one("#env-overview") is not None
        assert app.query_one("#scoreboard") is not None
        assert app.query_one("#tamiyo-brain") is not None
        assert app.query_one("#reward-components") is not None
        assert app.query_one("#esper-status") is not None


@pytest.mark.asyncio
async def test_app_quit_binding():
    """Pressing q should quit the app."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        await pilot.press("q")
        assert app._exit


@pytest.mark.asyncio
async def test_app_focus_navigation():
    """Tab should cycle through focusable panels."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        await pilot.press("tab")
        # Should have moved focus
        assert app.focused is not None
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py -v
```

**Step 3: Write app implementation**

Create `src/esper/karn/sanctum/app.py`:

```python
"""Sanctum Textual Application.

Developer diagnostic TUI for debugging PPO training runs.
Layout matches existing Rich TUI (tui.py _render() method).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class SanctumApp(App):
    """Sanctum diagnostic TUI for Esper training.

    Provides deep inspection of PPO training for debugging.
    Layout mirrors existing Rich TUI for 1:1 port.
    """

    TITLE = "Esper Sanctum"
    SUB_TITLE = "Diagnostic Console"

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("?", "toggle_help", "Help", show=True),
        Binding("tab", "focus_next", "Next Panel", show=False),
        Binding("shift+tab", "focus_previous", "Prev Panel", show=False),
        Binding("1-9", "focus_env", "Focus Env", show=False),
    ]

    def __init__(self, num_envs: int = 16, **kwargs) -> None:
        """Initialize Sanctum app."""
        super().__init__(**kwargs)
        self.num_envs = num_envs
        self._snapshot: SanctumSnapshot | None = None

    def compose(self) -> ComposeResult:
        """Compose the application layout.

        Layout matches Rich TUI _render() method:
        - Top: Header with run info
        - Left: Environment Overview (main table)
        - Right top: Best Runs scoreboard
        - Right middle: Tamiyo Brain (4-column panel)
        - Right bottom: Reward Components + Esper Status
        - Bottom: Footer with keybindings
        """
        yield Header()

        with Container(id="sanctum-main"):
            with Horizontal(id="top-section"):
                # Left: Environment Overview (takes 65% width)
                yield Static("[Environment Overview]", id="env-overview", classes="panel focusable")

                # Right column
                with Vertical(id="right-column"):
                    # Scoreboard (Best Runs)
                    yield Static("[Best Runs]", id="scoreboard", classes="panel focusable")

                    # Tamiyo Brain
                    yield Static("[Tamiyo Brain]", id="tamiyo-brain", classes="panel focusable")

                    # Bottom right: Rewards + Status side by side
                    with Horizontal(id="bottom-right"):
                        yield Static("[Reward Components]", id="reward-components", classes="panel focusable")
                        yield Static("[Esper Status]", id="esper-status", classes="panel focusable")

        yield Footer()

    def action_toggle_help(self) -> None:
        """Toggle help overlay."""
        self.notify("Help: q=quit, Tab=navigate, 1-9=focus env, ?=help")

    def action_focus_env(self, env_id: str) -> None:
        """Focus a specific environment by number."""
        try:
            env_num = int(env_id) - 1  # 1-indexed for user
            if 0 <= env_num < self.num_envs:
                if self._snapshot:
                    self._snapshot.focused_env_id = env_num
                    self._refresh_focused_panels()
        except ValueError:
            pass

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update all widgets with new snapshot."""
        self._snapshot = snapshot
        self._refresh_all_panels()

    def _refresh_all_panels(self) -> None:
        """Refresh all panels with current snapshot."""
        # Will be implemented when widgets are created
        pass

    def _refresh_focused_panels(self) -> None:
        """Refresh panels that depend on focused env."""
        # Will be implemented when widgets are created
        pass
```

**Step 4: Create styles**

Create `src/esper/karn/sanctum/styles.tcss`:

```css
/* Sanctum Diagnostic TUI Styles
 * Matches Rich TUI layout with Textual styling.
 */

#sanctum-main {
    height: 1fr;
    padding: 0 1;
}

#top-section {
    height: 1fr;
}

/* Environment Overview - main table (65% width) */
#env-overview {
    width: 65%;
    border: solid $primary;
    margin-right: 1;
}

#env-overview:focus {
    border: double $accent;
}

/* Right column (35% width) */
#right-column {
    width: 35%;
}

/* Scoreboard - Best Runs */
#scoreboard {
    height: 30%;
    border: solid cyan;
    margin-bottom: 1;
}

#scoreboard:focus {
    border: double $accent;
}

/* Tamiyo Brain - 4-column policy panel */
#tamiyo-brain {
    height: 40%;
    border: solid magenta;
    margin-bottom: 1;
}

#tamiyo-brain:focus {
    border: double $accent;
}

/* Bottom right section */
#bottom-right {
    height: 30%;
}

/* Reward Components */
#reward-components {
    width: 1fr;
    border: solid cyan;
    margin-right: 1;
}

#reward-components:focus {
    border: double $accent;
}

/* Esper Status */
#esper-status {
    width: 1fr;
    border: solid cyan;
}

#esper-status:focus {
    border: double $accent;
}

/* Common panel styling */
.panel {
    padding: 1;
}

.focusable:focus-within {
    border: double $accent;
}

/* Staleness indicator */
.stale {
    opacity: 0.7;
}

.stale::after {
    content: " (stale)";
    color: $warning;
}
```

**Step 5: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py -v
```

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/app.py src/esper/karn/sanctum/styles.tcss tests/karn/sanctum/test_app.py
git commit -m "feat(sanctum): add Textual app shell matching Rich TUI layout

Layout mirrors existing TUI _render():
- Environment Overview (65% left)
- Best Runs scoreboard (right top)
- Tamiyo Brain 4-panel (right middle)
- Reward Components + Esper Status (right bottom)
- Keyboard navigation with focus states

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Port Environment Overview Widget (Table-Based)

**Files:**
- Create: `src/esper/karn/sanctum/widgets/env_overview.py`
- Create: `tests/karn/sanctum/test_env_overview.py`
- Reference: `src/esper/karn/tui.py` lines 1760-1946 (_render_env_overview)

This is the **main display** - a detailed table with per-env rows, NOT a card grid.

**Step 1: Write tests**

Create `tests/karn/sanctum/test_env_overview.py`:

```python
"""Tests for Sanctum EnvOverview widget - must match existing TUI table."""
import pytest

from esper.karn.sanctum.widgets.env_overview import EnvOverview
from esper.karn.sanctum.schema import EnvState, SeedState, SanctumSnapshot


def test_env_overview_creation():
    """EnvOverview should create without errors."""
    widget = EnvOverview()
    assert widget is not None


def test_env_overview_has_correct_columns():
    """Must have all columns from existing TUI."""
    widget = EnvOverview(slot_ids=["r0c0", "r0c1"])
    # Columns: Env, Acc, Reward, Acc sparkline, Rwd sparkline,
    #          ΔAcc, Seed Δ, Rent, [slots...], Last, Status
    expected_base_columns = 8  # Before slots
    expected_slot_columns = 2  # r0c0, r0c1
    expected_end_columns = 2   # Last, Status
    # Total: 8 + 2 + 2 = 12


def test_env_overview_update():
    """EnvOverview should accept snapshot updates."""
    widget = EnvOverview(slot_ids=["r0c0"])
    snapshot = SanctumSnapshot(
        envs={
            0: EnvState(env_id=0, host_accuracy=75.5),
            1: EnvState(env_id=1, host_accuracy=72.0),
        },
        slot_ids=["r0c0"],
    )
    widget.update_snapshot(snapshot)
    assert len(widget._envs) == 2


def test_slot_cell_formatting():
    """Slot cells should show stage:blueprint with gradient indicators."""
    widget = EnvOverview(slot_ids=["r0c0"])
    seed = SeedState(
        slot_id="r0c0",
        stage="TRAINING",
        blueprint_id="conv_light",
        epochs_in_stage=5,
        has_vanishing=True,
    )
    cell = widget._format_slot_cell(seed)
    assert "Train" in cell or "TRAIN" in cell.upper()
    assert "conv_l" in cell  # First 6 chars
    assert "e5" in cell  # Epochs
    # Should have vanishing indicator
    assert "▼" in cell


def test_slot_cell_blending_shows_alpha():
    """BLENDING seeds should show alpha instead of epochs."""
    widget = EnvOverview(slot_ids=["r0c0"])
    seed = SeedState(
        slot_id="r0c0",
        stage="BLENDING",
        blueprint_id="conv_light",
        alpha=0.35,
    )
    cell = widget._format_slot_cell(seed)
    assert "0.3" in cell or "0.35" in cell


def test_aggregate_row():
    """Should have aggregate row with Σ at bottom."""
    widget = EnvOverview(slot_ids=["r0c0"])
    snapshot = SanctumSnapshot(
        envs={i: EnvState(env_id=i, host_accuracy=70 + i) for i in range(4)},
        slot_ids=["r0c0"],
        aggregate_mean_accuracy=72.0,
        aggregate_mean_reward=0.5,
    )
    widget.update_snapshot(snapshot)
    # Widget should render aggregate row


def test_sparklines_render():
    """Accuracy and reward sparklines should render."""
    widget = EnvOverview(slot_ids=["r0c0"])
    env = EnvState(env_id=0)
    for i in range(10):
        env.accuracy_history.append(70 + i)
        env.reward_history.append(0.1 * i)

    acc_sparkline = widget._make_sparkline(list(env.accuracy_history))
    rwd_sparkline = widget._make_sparkline(list(env.reward_history))

    assert len(acc_sparkline) == 8  # Default width
    assert "▁" in acc_sparkline or "█" in acc_sparkline
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py -v
```

**Step 3: Write EnvOverview implementation**

Create `src/esper/karn/sanctum/widgets/env_overview.py`:

```python
"""Environment Overview Widget - Per-environment table.

This is the MAIN display panel showing detailed per-env metrics.
Ported from Rich TUI _render_env_overview() (tui.py:1760-1946).

NOT a card grid - this is a full table with columns for:
Env, Acc, Reward, Sparklines, ΔAcc, Seed Δ, Rent, [slots...], Last, Status
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static, DataTable
from textual.containers import Vertical

from esper.karn.sanctum.schema import (
    EnvState,
    SeedState,
    SanctumSnapshot,
    make_sparkline,
)


class EnvOverview(Static):
    """Per-environment overview table matching Rich TUI layout."""

    DEFAULT_CSS = """
    EnvOverview {
        height: 100%;
    }

    EnvOverview DataTable {
        height: 1fr;
    }
    """

    # Stage display styles
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

    def __init__(self, slot_ids: list[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.slot_ids = slot_ids or []
        self._envs: dict[int, EnvState] = {}
        self._snapshot: SanctumSnapshot | None = None
        self._table: DataTable | None = None

    def compose(self) -> ComposeResult:
        """Create the table."""
        table = DataTable(id="env-table", zebra_stripes=True)

        # Base columns (before slots)
        table.add_column("Env", width=3)
        table.add_column("Acc", width=6)
        table.add_column("Reward", width=12)
        table.add_column("Acc▁▃▅", width=8)
        table.add_column("Rwd▁▃▅", width=8)
        table.add_column("ΔAcc", width=6)
        table.add_column("Seed Δ", width=7)
        table.add_column("Rent", width=5)

        # Dynamic slot columns
        for slot_id in self.slot_ids:
            table.add_column(slot_id, width=12)

        # End columns
        table.add_column("Last", width=6)
        table.add_column("Status", width=8)

        self._table = table
        yield table

    def update_snapshot(self, snapshot: SanctumSnapshot) -> None:
        """Update with new snapshot."""
        self._snapshot = snapshot
        self._envs = snapshot.envs
        if snapshot.slot_ids and snapshot.slot_ids != self.slot_ids:
            self.slot_ids = snapshot.slot_ids
            # Rebuild table columns if slots changed
            self._rebuild_columns()
        self._refresh_display()

    def _rebuild_columns(self) -> None:
        """Rebuild table with new slot columns."""
        if self._table is None:
            return
        self._table.clear(columns=True)

        # Re-add all columns
        self._table.add_column("Env", width=3)
        self._table.add_column("Acc", width=6)
        self._table.add_column("Reward", width=12)
        self._table.add_column("Acc▁▃▅", width=8)
        self._table.add_column("Rwd▁▃▅", width=8)
        self._table.add_column("ΔAcc", width=6)
        self._table.add_column("Seed Δ", width=7)
        self._table.add_column("Rent", width=5)

        for slot_id in self.slot_ids:
            self._table.add_column(slot_id, width=12)

        self._table.add_column("Last", width=6)
        self._table.add_column("Status", width=8)

    def _format_slot_cell(self, seed: SeedState | None) -> str:
        """Format a slot cell showing stage, blueprint, epochs and gradient health.

        Reference: tui.py _format_slot_cell() lines 1724-1758
        """
        if not seed or seed.stage == "DORMANT":
            return "[dim]─[/dim]"

        stage_short = self._STAGE_SHORT.get(seed.stage, seed.stage[:4])
        style = self._STAGE_STYLES.get(seed.stage, "white")

        # Blueprint abbreviation (first 6 chars)
        blueprint = seed.blueprint_id or "?"
        if len(blueprint) > 6:
            blueprint = blueprint[:6]

        # Gradient health indicator
        grad_indicator = ""
        if seed.has_exploding:
            grad_indicator = "[red]▲[/red]"
        elif seed.has_vanishing:
            grad_indicator = "[yellow]▼[/yellow]"

        # BLENDING shows alpha, others show epochs
        if seed.stage == "BLENDING" and seed.alpha > 0:
            base = f"[{style}]{stage_short}:{blueprint} {seed.alpha:.1f}[/{style}]"
        else:
            epochs_str = f" e{seed.epochs_in_stage}" if seed.epochs_in_stage > 0 else ""
            base = f"[{style}]{stage_short}:{blueprint}{epochs_str}[/{style}]"

        return f"{base}{grad_indicator}" if grad_indicator else base

    def _make_sparkline(self, values: list[float], width: int = 8) -> str:
        """Create sparkline from values."""
        return make_sparkline(values, width)

    def _refresh_display(self) -> None:
        """Refresh the table with current data."""
        if self._table is None or self._snapshot is None:
            return

        self._table.clear()

        # Status styles
        status_styles = {
            "excellent": "bold green",
            "healthy": "green",
            "initializing": "dim",
            "stalled": "yellow",
            "degraded": "red",
        }
        status_short = {
            "excellent": "EXCL",
            "healthy": "OK",
            "initializing": "INIT",
            "stalled": "STAL",
            "degraded": "DEGD",
        }

        # Sort envs by env_id
        sorted_envs = sorted(self._envs.values(), key=lambda e: e.env_id)

        for env in sorted_envs:
            row = self._build_env_row(env, status_styles, status_short)
            self._table.add_row(*row, key=f"env-{env.env_id}")

        # Add aggregate row
        if sorted_envs:
            self._add_aggregate_row(sorted_envs)

    def _build_env_row(
        self,
        env: EnvState,
        status_styles: dict[str, str],
        status_short: dict[str, str],
    ) -> list[str]:
        """Build a single env row."""
        # Env ID
        env_id_str = str(env.env_id)

        # Accuracy with color
        acc_str = f"{env.host_accuracy:.1f}%"
        if env.best_accuracy > 0:
            if env.host_accuracy >= env.best_accuracy:
                acc_str = f"[green]{acc_str}[/green]"
            elif env.epochs_since_improvement > 5:
                acc_str = f"[yellow]{acc_str}[/yellow]"

        # Reward with rolling average
        reward_val = env.current_reward
        mean_val = env.mean_reward
        r_style = "green" if reward_val > 0 else "red" if reward_val < -0.5 else "white"
        reward_str = f"[{r_style}]{reward_val:+.2f}[/] [dim]({mean_val:+.2f})[/]"

        # Sparklines
        acc_spark = self._make_sparkline(list(env.accuracy_history))
        rwd_spark = self._make_sparkline(list(env.reward_history))

        # Reward components
        base_delta = env.reward_components.get("base_acc_delta")
        if isinstance(base_delta, (int, float)):
            style = "green" if float(base_delta) >= 0 else "red"
            delta_str = f"[{style}]{float(base_delta):+.2f}[/{style}]"
        else:
            delta_str = "─"

        # Seed contribution
        seed_contrib = env.reward_components.get("seed_contribution")
        bounded_attr = env.reward_components.get("bounded_attribution")
        if isinstance(seed_contrib, (int, float)) and seed_contrib != 0:
            style = "green" if seed_contrib > 0 else "red"
            contrib_str = f"[{style}]{seed_contrib:+.1f}%[/{style}]"
        elif isinstance(bounded_attr, (int, float)) and bounded_attr != 0:
            style = "green" if bounded_attr > 0 else "red"
            contrib_str = f"[{style}]{bounded_attr:+.2f}[/{style}]"
        else:
            contrib_str = "─"

        # Compute rent
        compute_rent = env.reward_components.get("compute_rent")
        if isinstance(compute_rent, (int, float)) and compute_rent != 0:
            rent_str = f"[red]{compute_rent:.2f}[/red]"
        else:
            rent_str = "─"

        # Build row
        row = [
            env_id_str,
            acc_str,
            reward_str,
            acc_spark,
            rwd_spark,
            delta_str,
            contrib_str,
            rent_str,
        ]

        # Slot cells
        for slot_id in self.slot_ids:
            seed = env.seeds.get(slot_id)
            row.append(self._format_slot_cell(seed))

        # Last action
        last_action = env.action_history[-1] if env.action_history else "─"
        last_short = {
            "WAIT": "W",
            "GERMINATE": "G",
            "CULL": "C",
            "FOSSILIZE": "F",
        }.get(last_action, last_action[:1] if last_action != "─" else "─")
        row.append(last_short)

        # Status
        status_style = status_styles.get(env.status, "white")
        status_text = status_short.get(env.status, env.status[:4].upper())
        row.append(f"[{status_style}]{status_text}[/{status_style}]")

        return row

    def _add_aggregate_row(self, envs: list[EnvState]) -> None:
        """Add aggregate summary row."""
        if self._table is None or self._snapshot is None:
            return

        # Compute aggregates
        deltas = [
            float(e.reward_components.get("base_acc_delta"))
            for e in envs
            if isinstance(e.reward_components.get("base_acc_delta"), (int, float))
        ]
        rents = [
            float(e.reward_components.get("compute_rent"))
            for e in envs
            if isinstance(e.reward_components.get("compute_rent"), (int, float))
        ]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        mean_rent = sum(rents) / len(rents) if rents else 0.0
        best_acc = max(e.best_accuracy for e in envs) if envs else 0.0

        # Separator
        sep_row = ["─" * 2, "─" * 5, "─" * 10, "─" * 6, "─" * 6, "─" * 5, "─" * 5, "─" * 4]
        for _ in self.slot_ids:
            sep_row.append("─" * 8)
        sep_row.extend(["─" * 4, "─" * 6])
        self._table.add_row(*sep_row)

        # Aggregate row
        agg_row = [
            "[bold]Σ[/bold]",
            f"[bold]{self._snapshot.aggregate_mean_accuracy:.1f}%[/bold]",
            f"[bold]{self._snapshot.aggregate_mean_reward:+.2f}[/bold]",
            "",  # Acc sparkline
            "",  # Rwd sparkline
            f"[dim]{mean_delta:+.2f}[/dim]" if deltas else "─",
            "─",  # Seed Δ aggregate not meaningful
            f"[dim]{mean_rent:.2f}[/dim]" if rents else "─",
        ]
        for _ in self.slot_ids:
            agg_row.append("")
        agg_row.extend(["", f"[dim]{best_acc:.1f}%[/dim]"])

        self._table.add_row(*agg_row, key="aggregate")
```

**Step 4: Update widgets __init__.py**

```python
"""Sanctum widgets."""
from esper.karn.sanctum.widgets.env_overview import EnvOverview

__all__ = ["EnvOverview"]
```

**Step 5: Run tests**

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py -v
```

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/widgets/ tests/karn/sanctum/test_env_overview.py
git commit -m "feat(sanctum): add EnvOverview table widget (1:1 port)

Port _render_env_overview() from Rich TUI as table, NOT card grid:
- Per-env rows with all metrics
- Dynamic slot columns based on config
- Sparklines for accuracy/reward history
- Slot cells with stage:blueprint, epochs, gradient indicators (▼▲)
- BLENDING shows alpha instead of epochs
- Aggregate row with Σ at bottom
- Status with color coding

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Port Scoreboard Widget (Best Runs)

**Files:**
- Create: `src/esper/karn/sanctum/widgets/scoreboard.py`
- Create: `tests/karn/sanctum/test_scoreboard.py`
- Reference: `src/esper/karn/tui.py` lines 1039-1138

**Step 1: Write tests**

Create `tests/karn/sanctum/test_scoreboard.py`:

```python
"""Tests for Sanctum Scoreboard widget."""
import pytest

from esper.karn.sanctum.widgets.scoreboard import Scoreboard
from esper.karn.sanctum.schema import EnvState, SeedState


def test_scoreboard_creation():
    """Scoreboard should create without errors."""
    board = Scoreboard()
    assert board is not None


def test_scoreboard_sorts_by_best_accuracy():
    """Scoreboard should sort envs by best accuracy descending."""
    board = Scoreboard()
    envs = {
        0: EnvState(env_id=0, best_accuracy=75.0),
        1: EnvState(env_id=1, best_accuracy=85.0),
        2: EnvState(env_id=2, best_accuracy=80.0),
    }
    board.update_envs(envs)
    assert board._sorted_envs[0].env_id == 1
    assert board._sorted_envs[1].env_id == 2
    assert board._sorted_envs[2].env_id == 0


def test_scoreboard_shows_medals():
    """Top 3 should get medal indicators."""
    board = Scoreboard()
    envs = {i: EnvState(env_id=i, best_accuracy=90 - i) for i in range(5)}
    board.update_envs(envs)
    assert board._get_rank_display(0) == "🥇"
    assert board._get_rank_display(1) == "🥈"
    assert board._get_rank_display(2) == "🥉"
    assert board._get_rank_display(3) == "4"


def test_scoreboard_aggregates():
    """Should compute global best, mean, fossilized, culled."""
    board = Scoreboard()
    envs = {
        0: EnvState(env_id=0, best_accuracy=80.0, fossilized_count=2, culled_count=1),
        1: EnvState(env_id=1, best_accuracy=90.0, fossilized_count=1, culled_count=0),
    }
    board.update_envs(envs)
    assert board._global_best == 90.0
    assert board._mean_best == 85.0
    assert board._total_fossilized == 3
    assert board._total_culled == 1


def test_scoreboard_shows_best_seeds():
    """Should show seeds at best accuracy."""
    board = Scoreboard()
    env = EnvState(env_id=0, best_accuracy=85.0)
    env.best_seeds["r0c0"] = SeedState(slot_id="r0c0", blueprint_id="conv_light")
    board.update_envs({0: env})
    # Should display seed blueprint
```

**Step 2: Write Scoreboard implementation**

Create `src/esper/karn/sanctum/widgets/scoreboard.py`:

```python
"""Scoreboard Widget - Best Runs leaderboard.

Displays top 10 environments by best accuracy with medals.
Ported from Rich TUI _render_scoreboard() (tui.py:1039-1138).
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static, DataTable
from textual.containers import Vertical

from esper.karn.sanctum.schema import EnvState


class Scoreboard(Static):
    """Best Runs leaderboard showing top performers."""

    DEFAULT_CSS = """
    Scoreboard {
        height: 100%;
        padding: 1;
    }

    Scoreboard DataTable {
        height: 1fr;
    }
    """

    MEDALS = ["🥇", "🥈", "🥉"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._envs: dict[int, EnvState] = {}
        self._sorted_envs: list[EnvState] = []
        self._table: DataTable | None = None

        # Aggregate stats
        self._global_best: float = 0.0
        self._mean_best: float = 0.0
        self._total_fossilized: int = 0
        self._total_culled: int = 0

    def compose(self) -> ComposeResult:
        """Create the scoreboard layout."""
        # Stats header
        yield Static("", id="scoreboard-stats")

        # Leaderboard table
        table = DataTable(id="scoreboard-table")
        table.add_columns("#", "@Ep", "High", "Cur", "Seeds")
        self._table = table
        yield table

    def update_envs(self, envs: dict[int, EnvState]) -> None:
        """Update with new environment states."""
        self._envs = envs

        # Sort by best accuracy descending, top 10
        self._sorted_envs = sorted(
            envs.values(),
            key=lambda e: e.best_accuracy,
            reverse=True
        )[:10]

        # Compute aggregates
        all_envs = list(envs.values())
        self._total_fossilized = sum(e.fossilized_count for e in all_envs)
        self._total_culled = sum(e.culled_count for e in all_envs)
        best_accs = [e.best_accuracy for e in all_envs if e.best_accuracy > 0]
        self._mean_best = sum(best_accs) / len(best_accs) if best_accs else 0.0
        self._global_best = max(best_accs) if best_accs else 0.0

        self._refresh_display()

    def _get_rank_display(self, rank: int) -> str:
        """Get display string for rank (medal or number)."""
        if rank < len(self.MEDALS):
            return self.MEDALS[rank]
        return str(rank + 1)

    def _refresh_display(self) -> None:
        """Refresh the scoreboard display."""
        # Update stats header
        stats = self.query_one("#scoreboard-stats", Static)
        stats.update(
            f"Global Best: [bold green]{self._global_best:.1f}%[/] | "
            f"Mean: {self._mean_best:.1f}% | "
            f"Foss: [green]{self._total_fossilized}[/] | "
            f"Cull: [red]{self._total_culled}[/]"
        )

        if self._table is None:
            return

        self._table.clear()

        if not self._sorted_envs:
            self._table.add_row("-", "-", "-", "-", "-")
            return

        for i, env in enumerate(self._sorted_envs):
            rank = self._get_rank_display(i)

            # Current vs best styling
            delta = env.host_accuracy - env.best_accuracy
            if delta >= -0.5:
                cur_style = "green"
            elif delta >= -2.0:
                cur_style = "yellow"
            else:
                cur_style = "dim"

            # Seeds at best accuracy
            if env.best_seeds:
                n_seeds = len(env.best_seeds)
                if n_seeds <= 2:
                    seed_parts = [
                        f"[green]{s.blueprint_id[:6] if s.blueprint_id else '?'}[/]"
                        for s in env.best_seeds.values()
                    ]
                    seeds_str = " ".join(seed_parts)
                else:
                    seeds_str = f"[green]{n_seeds} seeds[/]"
            else:
                seeds_str = "─"

            self._table.add_row(
                rank,
                str(env.best_accuracy_episode),
                f"[bold green]{env.best_accuracy:.1f}[/]",
                f"[{cur_style}]{env.host_accuracy:.1f}[/]",
                seeds_str,
            )
```

**Step 3: Update widgets __init__.py**

**Step 4: Run tests and commit**

---

## Task 5: Port Tamiyo Brain Widget (Complete)

**Files:**
- Create: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Create: `tests/karn/sanctum/test_tamiyo_brain.py`
- Reference: `src/esper/karn/tui.py` lines 1140-1430

This is the 4-column policy panel: Health, Losses, Vitals, Actions.

**Step 1: Write tests**

Create `tests/karn/sanctum/test_tamiyo_brain.py`:

```python
"""Tests for Sanctum TamiyoBrain widget - must have ALL metrics."""
import pytest

from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain
from esper.karn.sanctum.schema import TamiyoState
from esper.karn.constants import TUIThresholds


def test_tamiyo_brain_creation():
    """TamiyoBrain should create without errors."""
    brain = TamiyoBrain()
    assert brain is not None


def test_tamiyo_brain_update():
    """TamiyoBrain should accept state updates."""
    brain = TamiyoBrain()
    state = TamiyoState(
        entropy=1.2,
        clip_fraction=0.15,
        kl_divergence=0.02,
        explained_variance=0.85,
    )
    brain.update_state(state)
    assert brain._state.entropy == 1.2


def test_entropy_uses_thresholds():
    """Must use TUIThresholds, not hardcoded values."""
    brain = TamiyoBrain()
    # Test against actual thresholds
    assert brain._get_entropy_status(TUIThresholds.ENTROPY_CRITICAL - 0.01) == "critical"
    assert brain._get_entropy_status(TUIThresholds.ENTROPY_WARNING - 0.01) == "warning"
    assert brain._get_entropy_status(TUIThresholds.ENTROPY_WARNING + 0.1) == "ok"


def test_clip_uses_thresholds():
    """Must use TUIThresholds for clip fraction."""
    brain = TamiyoBrain()
    assert brain._get_clip_status(TUIThresholds.CLIP_CRITICAL + 0.01) == "critical"
    assert brain._get_clip_status(TUIThresholds.CLIP_WARNING + 0.01) == "warning"
    assert brain._get_clip_status(TUIThresholds.CLIP_WARNING - 0.05) == "ok"


def test_learning_rate_display():
    """Must display learning rate in Vitals."""
    brain = TamiyoBrain()
    state = TamiyoState(learning_rate=3e-4)
    brain.update_state(state)
    # LR should be formatted as scientific notation


def test_gradient_health_metrics():
    """Must display dead/exploding layers and GradHP."""
    brain = TamiyoBrain()
    state = TamiyoState(
        dead_layers=2,
        exploding_layers=1,
        layer_gradient_health=0.75,
    )
    brain.update_state(state)
    # Should show dead layers, exploding layers, and GradHP percentage


def test_ratio_stats_all_displayed():
    """Must display ratio mean, min, max, AND std."""
    brain = TamiyoBrain()
    state = TamiyoState(
        ratio_mean=1.02,
        ratio_min=0.8,
        ratio_max=1.5,
        ratio_std=0.15,
    )
    brain.update_state(state)
    # All ratio stats should be shown


def test_waiting_state():
    """Should show waiting message before PPO data received."""
    brain = TamiyoBrain()
    state = TamiyoState(ppo_data_received=False)
    brain.update_state(state)
    # Should show "Waiting for first batch..."
```

**Step 2: Write TamiyoBrain implementation**

Create `src/esper/karn/sanctum/widgets/tamiyo_brain.py`:

```python
"""Tamiyo Brain Widget - Policy agent diagnostics.

4-column panel: Health, Losses, Vitals, Actions.
Ported from Rich TUI _render_tamiyo_brain() (tui.py:1140-1430).

IMPORTANT: Uses TUIThresholds from karn/constants.py, NOT hardcoded values.
"""
from __future__ import annotations

import math
from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Horizontal, Vertical, Grid

from esper.karn.sanctum.schema import TamiyoState
from esper.karn.constants import TUIThresholds


class TamiyoBrain(Static):
    """Tamiyo policy agent diagnostic panel - 4 columns."""

    DEFAULT_CSS = """
    TamiyoBrain {
        height: 100%;
        padding: 1;
    }

    .brain-grid {
        layout: grid;
        grid-size: 4 1;
        grid-gutter: 1;
        height: 100%;
    }

    .brain-section {
        border: solid $primary-darken-2;
        padding: 0 1;
    }

    .brain-section-title {
        text-style: bold;
        color: $text-muted;
        text-align: center;
    }
    """

    # Max entropy for 4-action space (WAIT, GERMINATE, CULL, FOSSILIZE)
    MAX_ENTROPY = math.log(4)  # ≈ 1.386

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = TamiyoState()

    def compose(self) -> ComposeResult:
        """Create the 4-column brain layout."""
        with Grid(classes="brain-grid"):
            # Column 1: Health
            with Vertical(classes="brain-section", id="health-section"):
                yield Static("Health", classes="brain-section-title")
                yield Static("", id="health-content")

            # Column 2: Losses
            with Vertical(classes="brain-section", id="losses-section"):
                yield Static("Losses", classes="brain-section-title")
                yield Static("", id="losses-content")

            # Column 3: Vitals
            with Vertical(classes="brain-section", id="vitals-section"):
                yield Static("Vitals", classes="brain-section-title")
                yield Static("", id="vitals-content")

            # Column 4: Actions
            with Vertical(classes="brain-section", id="actions-section"):
                yield Static("Actions", classes="brain-section-title")
                yield Static("", id="actions-content")

    def update_state(self, state: TamiyoState) -> None:
        """Update with new Tamiyo state."""
        self._state = state
        self._refresh_display()

    # =========================================================================
    # Status methods using TUIThresholds (NOT hardcoded values!)
    # =========================================================================

    def _get_entropy_status(self, entropy: float) -> str:
        """Get status for entropy using TUIThresholds."""
        if entropy < TUIThresholds.ENTROPY_CRITICAL:
            return "critical"
        elif entropy < TUIThresholds.ENTROPY_WARNING:
            return "warning"
        return "ok"

    def _get_clip_status(self, clip: float) -> str:
        """Get status for clip fraction using TUIThresholds."""
        if clip > TUIThresholds.CLIP_CRITICAL:
            return "critical"
        elif clip > TUIThresholds.CLIP_WARNING:
            return "warning"
        return "ok"

    def _get_kl_status(self, kl: float) -> str:
        """Get status for KL divergence."""
        if kl > TUIThresholds.KL_WARNING:
            return "warning"
        return "ok"

    def _get_ev_status(self, ev: float) -> str:
        """Get status for explained variance."""
        if ev < TUIThresholds.EXPLAINED_VAR_CRITICAL:
            return "critical"
        elif ev < TUIThresholds.EXPLAINED_VAR_WARNING:
            return "warning"
        return "ok"

    def _get_grad_norm_status(self, grad_norm: float) -> str:
        """Get status for gradient norm."""
        if grad_norm > TUIThresholds.GRAD_NORM_CRITICAL:
            return "critical"
        elif grad_norm > TUIThresholds.GRAD_NORM_WARNING:
            return "warning"
        return "ok"

    def _status_style(self, status: str) -> str:
        """Get Rich style for status."""
        return {"ok": "green", "warning": "yellow", "critical": "red bold"}.get(status, "dim")

    def _status_text(self, status: str) -> str:
        """Get status indicator text."""
        return {
            "ok": "[green]✓ OK[/]",
            "warning": "[yellow]⚠ WARN[/]",
            "critical": "[red bold]✕ CRIT[/]",
        }.get(status, "")

    # =========================================================================
    # Refresh methods
    # =========================================================================

    def _refresh_display(self) -> None:
        """Refresh all sections."""
        # Check for waiting state
        if not self._state.ppo_data_received:
            self._show_waiting_state()
            return

        self._refresh_health()
        self._refresh_losses()
        self._refresh_vitals()
        self._refresh_actions()

    def _show_waiting_state(self) -> None:
        """Show waiting state before PPO data arrives."""
        for section_id in ["health-content", "losses-content", "vitals-content", "actions-content"]:
            try:
                content = self.query_one(f"#{section_id}", Static)
                content.update("[dim italic]Waiting for PPO data...[/]")
            except Exception:
                pass

    def _refresh_health(self) -> None:
        """Refresh health section."""
        s = self._state

        # DRL Expert: Show entropy as percentage of max
        ent_pct = (s.entropy / self.MAX_ENTROPY) * 100 if self.MAX_ENTROPY > 0 else 0
        ent_status = self._get_entropy_status(s.entropy)

        clip_status = self._get_clip_status(s.clip_fraction)
        kl_status = self._get_kl_status(s.kl_divergence)
        ev_status = self._get_ev_status(s.explained_variance)

        # DRL Expert: Add interpretive hint for ExplVar
        if s.explained_variance < TUIThresholds.EXPLAINED_VAR_CRITICAL:
            ev_hint = "harm"
        elif s.explained_variance < TUIThresholds.EXPLAINED_VAR_WARNING:
            ev_hint = "weak"
        else:
            ev_hint = "ok"

        lines = [
            f"Entropy   [{self._status_style(ent_status)}]{s.entropy:.2f}[/] ({ent_pct:.0f}%)",
            f"Clip      [{self._status_style(clip_status)}]{s.clip_fraction:.3f}[/] {self._status_text(clip_status)}",
            f"KL        {s.kl_divergence:.4f} {self._status_text(kl_status)}",
            f"ExplVar   [{self._status_style(ev_status)}]{s.explained_variance:.2f}[/] ({ev_hint})",
        ]

        content = self.query_one("#health-content", Static)
        content.update("\n".join(lines))

    def _refresh_losses(self) -> None:
        """Refresh losses section."""
        s = self._state

        grad_status = self._get_grad_norm_status(s.grad_norm)

        lines = [
            f"Policy    {s.policy_loss:.4f}",
            f"Value     {s.value_loss:.4f}",
            f"Entropy   {s.entropy_loss:.4f}",
            f"GradNorm  [{self._status_style(grad_status)}]{s.grad_norm:.2f}[/]",
        ]

        content = self.query_one("#losses-content", Static)
        content.update("\n".join(lines))

    def _refresh_vitals(self) -> None:
        """Refresh vitals section."""
        s = self._state

        # Learning rate
        if s.learning_rate is not None:
            lr_str = f"{s.learning_rate:.2e}"
        else:
            lr_str = "─"

        # Ratio stats with color coding (DRL Expert: prefix with π)
        ratio_max = s.ratio_max
        ratio_min = s.ratio_min
        ratio_std = s.ratio_std

        max_style = "red bold" if ratio_max >= 2.0 else "yellow" if ratio_max >= 1.5 else "green"
        min_style = "red bold" if ratio_min <= 0.3 else "yellow" if ratio_min <= 0.5 else "green"
        std_style = "yellow" if ratio_std >= 0.5 else ""

        lines = [
            f"LR        {lr_str}",
            f"π Ratio↑  [{max_style}]{ratio_max:.2f}[/]",
            f"π Ratio↓  [{min_style}]{ratio_min:.2f}[/]",
            f"π Ratio σ [{std_style}]{ratio_std:.3f}[/]" if std_style else f"π Ratio σ {ratio_std:.3f}",
        ]

        # Gradient health (dead/exploding layers, GradHP)
        if s.dead_layers > 0:
            lines.append(f"[yellow bold]Dead: {s.dead_layers} layers[/]")
        if s.exploding_layers > 0:
            lines.append(f"[red bold]Explode: {s.exploding_layers} layers[/]")

        # GradHP (layer gradient health) - renamed from GradHP per DRL expert
        health = s.layer_gradient_health
        health_style = "red bold" if health < 0.5 else "yellow" if health < 0.8 else "green"
        lines.append(f"Grad Health [{health_style}]{health:.0%}[/]")

        content = self.query_one("#vitals-content", Static)
        content.update("\n".join(lines))

    def _refresh_actions(self) -> None:
        """Refresh actions section."""
        s = self._state

        total = sum(s.action_counts.values()) or 1

        # Action order and styling
        action_styles = {
            "WAIT": "dim",
            "GERMINATE": "green",
            "CULL": "red",
            "FOSSILIZE": "blue",
        }

        lines = []
        for action in ["WAIT", "GERMINATE", "CULL", "FOSSILIZE"]:
            count = s.action_counts.get(action, 0)
            pct = (count / total) * 100
            style = action_styles.get(action, "white")

            # Warn if WAIT dominates
            if action == "WAIT" and pct > TUIThresholds.WAIT_DOMINANCE_WARNING * 100:
                style = "yellow bold"

            lines.append(f"[{style}]{action:10} {pct:5.1f}%[/]")

        content = self.query_one("#actions-content", Static)
        content.update("\n".join(lines) if lines else "[dim]No actions yet[/]")
```

**Step 3-6: Update, test, commit**

---

## Task 6: Port Remaining Widgets

**Files:**
- Create: `src/esper/karn/sanctum/widgets/reward_components.py`
- Create: `src/esper/karn/sanctum/widgets/esper_status.py`
- Create: `tests/karn/sanctum/test_remaining_widgets.py`
- Reference: `src/esper/karn/tui.py` lines 1513-1586, 1596-1696

### RewardComponents - Esper-specific breakdown

Must show: base_acc_delta, bounded_attribution, compute_rent, ratio_penalty, stage_bonus, fossilize_terminal_bonus, blending_warning, probation_warning

### EsperStatus - System vitals + seed stage counts

Must show: stage counts (Train/Blend/Prob/Foss), host params, throughput, runtime, GPU stats (multi-GPU), RAM, **CPU (FIX)**

---

## Task 7: Wire Up Complete App

Update `app.py` to use real widgets and propagate snapshot.

---

## Task 8: Add Telemetry Backend

**Files:**
- Create: `src/esper/karn/sanctum/backend.py`
- Reference: `src/esper/karn/tui.py` TUIOutput class (lines 215-500 approximately)

Backend must:
- Implement OutputBackend protocol
- Handle all telemetry events that existing TUI handles
- Update SanctumSnapshot state
- Be thread-safe

---

## Task 9: Add CLI Integration

Add `--sanctum` flag to `train.py`, mutually exclusive with `--overwatch`.

---

## Task 10: Final Verification and Cleanup

1. Run full test suite
2. Manual verification - launch both old TUI and new Sanctum, compare side-by-side
3. Delete `src/esper/karn/tui.py`
4. Update `karn/__init__.py` exports

---

## Summary

| Task | Description | Tests | Key Changes from Original Plan |
|------|-------------|-------|-------------------------------|
| 1 | Complete schema | 15+ | Added ALL fields from existing TUI |
| 2 | App shell | 4 | Layout matches existing TUI |
| 3 | EnvOverview | 7 | TABLE not cards, with sparklines |
| 4 | Scoreboard | 5 | Unchanged |
| 5 | TamiyoBrain | 7 | Uses TUIThresholds, has LR/GradHP |
| 6 | Remaining | 8 | Esper-specific reward components |
| 7 | Wire app | 2 | Unchanged |
| 8 | Backend | 3+ | Match existing event handling |
| 9 | CLI | 1 | Unchanged |
| 10 | Cleanup | - | Unchanged |

**Total: ~50+ tests across 10 tasks**

**Key corrections from original plan:**
1. ✅ EnvOverview is a TABLE, not a card grid
2. ✅ Schema includes ALL existing TUI fields
3. ✅ Reward components are Esper-specific
4. ✅ TamiyoBrain uses TUIThresholds, not hardcoded values
5. ✅ Gradient health metrics (dead/exploding layers, GradHP) included
6. ✅ Sparklines for accuracy/reward history
7. ✅ Multi-GPU support
8. ✅ CPU indicator fix preserved
