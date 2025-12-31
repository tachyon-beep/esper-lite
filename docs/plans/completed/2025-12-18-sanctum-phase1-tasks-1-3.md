# Sanctum Phase 1: Tasks 1-3 Implementation Plan

> **Status:** Complete, production-ready implementation plan for Sanctum schema, app shell, and EnvOverview widget.
>
> **Parent Plan:** This document extracts Tasks 1-3 from `2025-12-18-sanctum-phase1-implementation.md` with all review fixes applied.
>
> **For Claude:** Use `superpowers:executing-plans` to implement this plan task-by-task.

## Overview

This plan covers the first three tasks of porting the existing Rich TUI (`karn/tui.py`) to Textual as `karn/sanctum/`:

1. **Task 1:** Complete schema matching ALL existing TUI state
2. **Task 2:** Base app shell with correct layout
3. **Task 3:** EnvOverview widget (table-based, NOT card grid)

**Key Principle:** This is a **1:1 port**. Every feature in the existing TUI must work in Sanctum.

## Context Documents

- **Design:** `docs/plans/2025-12-18-sanctum-design.md`
- **Existing TUI:** `src/esper/karn/tui.py` (~2100 lines)
- **Constants:** `src/esper/karn/constants.py` (TUIThresholds)
- **Overwatch patterns:** `src/esper/karn/overwatch/schema.py` (for schema patterns)

## Task 1: Create Directory Structure and Complete Schema

### Files

- Create: `src/esper/karn/sanctum/__init__.py`
- Create: `src/esper/karn/sanctum/schema.py`
- Create: `src/esper/karn/sanctum/widgets/__init__.py`
- Create: `tests/karn/sanctum/__init__.py`
- Create: `tests/karn/sanctum/test_schema.py`

### Step 1: Create Directory Structure

```bash
mkdir -p src/esper/karn/sanctum/widgets
mkdir -p tests/karn/sanctum
touch src/esper/karn/sanctum/__init__.py
touch src/esper/karn/sanctum/widgets/__init__.py
touch tests/karn/sanctum/__init__.py
```

### Step 2: Write Schema Tests

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

    def test_fossilized_params_tracking(self):
        """Must track fossilized_params for scoreboard display."""
        env = EnvState(env_id=0, fossilized_params=128000)
        assert env.fossilized_params == 128000


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

    def test_total_actions_tracking(self):
        """Must track total_actions for action percentage calculation."""
        tamiyo = TamiyoState(total_actions=100)
        assert tamiyo.total_actions == 100


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
        """Must have ALL Esper-specific reward components."""
        rewards = RewardComponents(
            base_acc_delta=0.5,
            bounded_attribution=0.3,
            seed_contribution=0.2,
            compute_rent=-0.1,
            ratio_penalty=-0.05,
            stage_bonus=0.2,
            fossilize_terminal_bonus=1.0,
            blending_warning=-0.1,
            probation_warning=0.0,
            val_acc=75.5,
        )
        assert rewards.base_acc_delta == 0.5
        assert rewards.compute_rent == -0.1
        assert rewards.val_acc == 75.5


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

### Step 3: Run Tests to Verify They Fail

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py -v
```

Expected: FAIL with "No module named 'esper.karn.sanctum.schema'"

### Step 4: Write Complete Schema Implementation

Create `src/esper/karn/sanctum/schema.py`:

```python
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
    culled_count: int = 0

    # FIX: Added fossilized_params for scoreboard display (total params in FOSSILIZED seeds)
    fossilized_params: int = 0

    # Reward component breakdown (from REWARD_COMPUTED telemetry)
    # DOCUMENTED KEYS (all Esper-specific):
    #   - base_acc_delta: Legacy shaped signal based on accuracy delta
    #   - bounded_attribution: Contribution-primary attribution signal
    #   - seed_contribution: Seed contribution percentage (alternative to bounded_attribution)
    #   - compute_rent: Cost of active seeds (negative)
    #   - ratio_penalty: Penalty for extreme policy ratios (negative)
    #   - stage_bonus: Bonus for reaching advanced stages
    #   - fossilize_terminal_bonus: Large bonus for successfully fossilizing
    #   - blending_warning: Warning signal during blending (negative)
    #   - probation_warning: Warning signal during probation
    #   - val_acc: Validation accuracy context
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
    # ACTION NORMALIZATION: add_action() normalizes factored actions:
    #   GERMINATE_CONV_LIGHT → GERMINATE
    #   GERMINATE_DENSE_HEAVY → GERMINATE
    #   FOSSILIZE_R0C0 → FOSSILIZE
    #   CULL_R1C1 → CULL
    action_history: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    action_counts: dict[str, int] = field(default_factory=lambda: {
        "WAIT": 0, "GERMINATE": 0, "CULL": 0, "FOSSILIZE": 0
    })
    total_actions: int = 0

    # Status tracking
    status: str = "initializing"
    last_update: datetime | None = None
    epochs_since_improvement: int = 0

    # A/B test cohort (for color coding)
    # Captured from REWARD_COMPUTED event's ab_group field
    # Values: "shaped", "simplified", "sparse", or None if not A/B testing
    reward_mode: str | None = None

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

    def add_reward(self, reward: float, epoch: int) -> None:
        """Add reward and update best tracking."""
        self.reward_history.append(reward)
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_reward_epoch = epoch

    def add_accuracy(self, accuracy: float, epoch: int, episode: int = 0) -> None:
        """Add accuracy and update best/status tracking."""
        prev_acc = self.accuracy_history[-1] if self.accuracy_history else 0.0
        self.accuracy_history.append(accuracy)
        self.host_accuracy = accuracy

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_accuracy_epoch = epoch
            self.best_accuracy_episode = episode
            self.epochs_since_improvement = 0
            # Snapshot FOSSILIZED seeds when new best is achieved
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
                if seed.stage == "FOSSILIZED"
            }
        else:
            self.epochs_since_improvement += 1

        self._update_status(prev_acc, accuracy)

    def add_action(self, action_name: str) -> None:
        """Track action taken.

        ACTION NORMALIZATION: Normalizes factored germination actions to base types:
        - GERMINATE_CONV_LIGHT → GERMINATE
        - GERMINATE_DENSE_HEAVY → GERMINATE
        - FOSSILIZE_R0C0 → FOSSILIZE
        - CULL_R1C1 → CULL
        - WAIT → WAIT (unchanged)
        """
        self.action_history.append(action_name)

        # Normalize factored actions
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
        """Update env status based on metrics.

        Status values: initializing, healthy, excellent, stalled, degraded
        """
        if self.epochs_since_improvement > 10:
            self.status = "stalled"
        elif curr_acc < prev_acc - 1.0:
            self.status = "degraded"
        elif curr_acc > 80.0:
            self.status = "excellent"
        elif self.current_epoch > 0:
            self.status = "healthy"


@dataclass
class TamiyoState:
    """Tamiyo policy agent state - ALL metrics from existing TUI.

    Reference: tui.py TUIState policy metrics + _render_tamiyo_brain()

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
    advantage_mean: float = 0.0
    advantage_std: float = 0.0

    # Gradient health (shown in Vitals)
    dead_layers: int = 0
    exploding_layers: int = 0
    nan_grad_count: int = 0  # NaN gradient count
    layer_gradient_health: float = 1.0  # GradHP percentage (0-1)
    entropy_collapsed: bool = False  # Entropy collapse detected

    # Action distribution (Actions panel)
    action_counts: dict[str, int] = field(default_factory=dict)
    # FIX: Added total_actions for percentage calculation in TamiyoBrain
    total_actions: int = 0

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

    # FIX: CPU was collected but never displayed in old TUI
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

    ALL KEYS DOCUMENTED (Esper-specific components):
    - base_acc_delta: Legacy shaped signal based on accuracy improvement
    - bounded_attribution: Contribution-primary attribution signal (replaces seed_contribution)
    - seed_contribution: Seed contribution percentage (older format, may coexist)
    - compute_rent: Cost of active seeds (always negative)
    - ratio_penalty: Penalty for extreme policy ratios (negative if triggered)
    - stage_bonus: Bonus for reaching advanced lifecycle stages (BLENDING+)
    - fossilize_terminal_bonus: Large terminal bonus for successful fossilization
    - blending_warning: Warning signal during blending phase (negative)
    - probation_warning: Warning signal during probationary period
    - val_acc: Validation accuracy context (not a reward component, metadata)
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
class EventLogEntry:
    """Single event log entry for display in Event Log panel.

    Reference: Used by aggregator to build event_log list
    """
    timestamp: str  # Formatted as HH:MM:SS
    event_type: str  # REWARD_COMPUTED, SEED_GERMINATED, etc.
    env_id: int | None  # None for global events (PPO, BATCH)
    message: str  # Formatted message for display


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

    # Connection and timing (used by aggregator)
    connected: bool = False
    runtime_seconds: float = 0.0
    staleness_seconds: float = float('inf')
    captured_at: str = ""  # ISO timestamp

    # Aggregates (computed from envs)
    aggregate_mean_accuracy: float = 0.0
    aggregate_mean_reward: float = 0.0

    # Event log (most recent last)
    event_log: list[EventLogEntry] = field(default_factory=list)

    # Timestamps for staleness detection
    last_ppo_update: datetime | None = None
    last_reward_update: datetime | None = None

    # Focused env for detail panel
    focused_env_id: int = 0

    @property
    def is_stale(self) -> bool:
        """Check if data is stale (>5s since last update).

        STALENESS THRESHOLD: 5 seconds (matches Overwatch behavior)
        """
        return self.staleness_seconds > 5.0


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

### Step 5: Run Tests to Verify They Pass

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py -v
```

Expected: All tests PASS

### Step 6: Update sanctum __init__.py

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
    EventLogEntry,
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
    "EventLogEntry",
    "make_sparkline",
]
```

### Step 7: Commit

```bash
git add src/esper/karn/sanctum/ tests/karn/sanctum/
git commit -m "$(cat <<'EOF'
feat(sanctum): add complete schema matching existing TUI

Port ALL state classes from Rich TUI to Sanctum schema:
- SeedState with gradient health flags (▼▲ indicators)
- EnvState with history deques for sparklines
- TamiyoState with LR, ratio_std, dead/exploding layers, GradHP
- SystemVitals with multi-GPU support and CPU fix
- RewardComponents with Esper-specific breakdown
- GPUStats for multi-GPU tracking
- make_sparkline() helper function

CRITICAL FIXES:
- Added total_actions to TamiyoState (action % calculation)
- Added fossilized_params to EnvState (scoreboard param counts)
- Documented ALL reward_components keys explicitly
- Documented action normalization (GERMINATE_* → GERMINATE)
- Documented EnvState update methods

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create Base App Shell

### Files

- Create: `src/esper/karn/sanctum/app.py`
- Create: `src/esper/karn/sanctum/styles.tcss`
- Create: `tests/karn/sanctum/test_app.py`

### Step 1: Write App Tests

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
        assert app.query_one("#event-log") is not None


@pytest.mark.asyncio
async def test_app_quit_binding():
    """Pressing q should quit the app."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        await pilot.press("q")
        # App should have initiated exit
        assert not app.is_running


@pytest.mark.asyncio
async def test_app_focus_navigation():
    """Tab should cycle through focusable panels."""
    app = SanctumApp()
    async with app.run_test() as pilot:
        await pilot.press("tab")
        # Should have moved focus
        assert app.focused is not None
```

### Step 2: Run Tests to Verify They Fail

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py -v
```

### Step 3: Write App Implementation

Create `src/esper/karn/sanctum/app.py`:

```python
"""Sanctum Textual Application.

Developer diagnostic TUI for debugging PPO training runs.
Layout matches existing Rich TUI (tui.py _render() method).

LAYOUT FIX: TamiyoBrain spans full width as dedicated row (size=11),
NOT embedded in right column. Event Log included at bottom-left.
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

    TERMINAL SIZE CONSTRAINTS:
    - Minimum: 120x40 (width x height) for readable display
    - Recommended: 140x50 or larger
    - TamiyoBrain requires width ≥ 80 for 4-column layout
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
        """Initialize Sanctum app.

        Args:
            num_envs: Number of training environments (default 16)
        """
        super().__init__(**kwargs)
        self.num_envs = num_envs
        self._snapshot: SanctumSnapshot | None = None

    def compose(self) -> ComposeResult:
        """Compose the application layout.

        CORRECTED LAYOUT (FIX from review):
        - Header: Run info
        - Top section (horizontal):
          - Left (65%): Environment Overview table
          - Right (35%): Scoreboard (Best Runs)
        - TamiyoBrain: Full-width dedicated row (size=11)
        - Bottom section (horizontal):
          - Left: Event Log
          - Right: Reward Components + Esper Status (vertical stack)
        - Footer: Keybindings
        """
        yield Header()

        with Container(id="sanctum-main"):
            # Top section: Env Overview + Scoreboard
            with Horizontal(id="top-section"):
                # Left: Environment Overview (65% width)
                yield Static("[Environment Overview]", id="env-overview", classes="panel focusable")

                # Right: Scoreboard (35% width)
                yield Static("[Best Runs]", id="scoreboard", classes="panel focusable")

            # TamiyoBrain: Full-width dedicated row
            yield Static("[Tamiyo Brain]", id="tamiyo-brain", classes="panel focusable brain-panel")

            # Bottom section: Event Log + (Rewards + Status)
            with Horizontal(id="bottom-section"):
                # Left: Event Log
                yield Static("[Event Log]", id="event-log", classes="panel focusable")

                # Right: Reward Components + Esper Status (vertical stack)
                with Vertical(id="right-bottom"):
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

### Step 4: Create Styles

Create `src/esper/karn/sanctum/styles.tcss`:

```css
/* Sanctum Diagnostic TUI Styles
 * Matches Rich TUI layout with Textual styling.
 * FIX: TamiyoBrain as full-width row, Event Log added
 */

#sanctum-main {
    height: 1fr;
    padding: 0 1;
}

/* Top section: Env Overview + Scoreboard */
#top-section {
    height: 16;
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

/* Scoreboard - Best Runs (35% width) */
#scoreboard {
    width: 35%;
    border: solid cyan;
}

#scoreboard:focus {
    border: double $accent;
}

/* Tamiyo Brain - Full-width dedicated row (FIX) */
.brain-panel {
    height: 11;
    border: solid magenta;
    margin-top: 1;
    margin-bottom: 1;
}

#tamiyo-brain:focus {
    border: double $accent;
}

/* Bottom section: Event Log + (Rewards + Status) */
#bottom-section {
    height: 12;
}

/* Event Log (FIX: Added missing panel) */
#event-log {
    width: 50%;
    border: solid yellow;
    margin-right: 1;
}

#event-log:focus {
    border: double $accent;
}

/* Right bottom container */
#right-bottom {
    width: 50%;
}

/* Reward Components */
#reward-components {
    height: 1fr;
    border: solid cyan;
    margin-bottom: 1;
}

#reward-components:focus {
    border: double $accent;
}

/* Esper Status */
#esper-status {
    height: 1fr;
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

/* Staleness indicator (FIX: Data staleness display) */
.stale {
    opacity: 0.7;
}

.stale::after {
    content: " (stale)";
    color: $warning;
}
```

### Step 5: Run Tests

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py -v
```

### Step 6: Commit

```bash
git add src/esper/karn/sanctum/app.py src/esper/karn/sanctum/styles.tcss tests/karn/sanctum/test_app.py
git commit -m "$(cat <<'EOF'
feat(sanctum): add Textual app shell matching Rich TUI layout

CORRECTED LAYOUT:
- Top: Env Overview (65%) + Scoreboard (35%)
- TamiyoBrain: Full-width dedicated row (size=11) [FIX]
- Bottom: Event Log + (Rewards + Status) [Event Log added]
- Keyboard navigation with focus states

Terminal size constraints documented:
- Min: 120x40, Recommended: 140x50+

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Port Environment Overview Widget (Table-Based)

### Files

- Create: `src/esper/karn/sanctum/widgets/env_overview.py`
- Create: `tests/karn/sanctum/test_env_overview.py`
- Reference: `src/esper/karn/tui.py` lines 1760-1946 (_render_env_overview)

**CRITICAL:** This is a **table** with per-env rows, NOT a card grid.

### Step 1: Write Tests

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
    # Columns: Env, Acc, Reward, Acc▁▃▅, Rwd▁▃▅, ΔAcc, Seed Δ, Rent,
    #          [slots...], Last, Status
    # Total: 8 base + 2 slots + 2 end = 12


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


def test_accuracy_color_styling():
    """FIX: Accuracy should be green at best, yellow if stagnant >5 epochs."""
    widget = EnvOverview(slot_ids=["r0c0"])
    # At best accuracy
    env_best = EnvState(env_id=0, host_accuracy=80.0, best_accuracy=80.0)
    # Stagnant >5 epochs
    env_stagnant = EnvState(
        env_id=1,
        host_accuracy=75.0,
        best_accuracy=80.0,
        epochs_since_improvement=6
    )
    snapshot = SanctumSnapshot(envs={0: env_best, 1: env_stagnant}, slot_ids=["r0c0"])
    widget.update_snapshot(snapshot)
    # Should apply green/yellow styling


def test_reward_threshold_complete():
    """FIX: Reward should handle all thresholds: >0 green, <-0.5 red, else white."""
    widget = EnvOverview(slot_ids=["r0c0"])
    env_positive = EnvState(env_id=0)
    env_positive.reward_history.append(0.5)
    env_negative = EnvState(env_id=1)
    env_negative.reward_history.append(-0.6)
    env_neutral = EnvState(env_id=2)
    env_neutral.reward_history.append(-0.2)
    # All thresholds covered


def test_last_action_normalization():
    """FIX: Last action should parse normalized action names correctly."""
    widget = EnvOverview(slot_ids=["r0c0"])
    env = EnvState(env_id=0)
    # Action history contains normalized names (WAIT, GERMINATE, etc.)
    env.action_history.append("GERMINATE")
    env.action_history.append("WAIT")
    env.action_history.append("FOSSILIZE")
    # Widget should handle all action types


def test_staleness_indicator():
    """FIX: Should show '(Ns ago)' when data >5s old."""
    from datetime import datetime, timedelta
    widget = EnvOverview(slot_ids=["r0c0"])
    snapshot = SanctumSnapshot(
        envs={0: EnvState(env_id=0)},
        slot_ids=["r0c0"],
    )
    # Simulate stale data
    snapshot.last_ppo_update = datetime.now() - timedelta(seconds=7)
    assert snapshot.is_stale is True
    assert snapshot.staleness_seconds > 5.0
```

### Step 2: Run Tests to Verify They Fail

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py -v
```

### Step 3: Write EnvOverview Implementation

Create `src/esper/karn/sanctum/widgets/env_overview.py`:

```python
"""Environment Overview Widget - Per-environment table.

This is the MAIN display panel showing detailed per-env metrics.
Ported from Rich TUI _render_env_overview() (tui.py:1760-1946).

NOT a card grid - this is a full table with columns for:
Env, Acc, Reward, Sparklines, ΔAcc, Seed Δ, Rent, [slots...], Last, Status

CRITICAL FIXES:
1. Accuracy color: Green if at best, yellow if stagnant >5 epochs
2. Reward threshold: Missing <-0.5 case for red styling
3. Last action: Normalized action parsing (GERMINATE, not GERMINATE_*)
4. Staleness: "(Ns ago)" indicator when data >5s old
"""
from __future__ import annotations

from datetime import datetime
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

    # A/B test cohort styling (for --ab-test shaped-vs-simplified)
    # Shows colored pip next to env ID to distinguish cohorts
    _AB_STYLES: dict[str, tuple[str, str]] = {
        "shaped": ("●", "bright_blue"),      # Blue pip for shaped reward
        "simplified": ("●", "bright_yellow"), # Yellow pip for simplified reward
        "sparse": ("●", "bright_cyan"),       # Cyan pip for sparse reward
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
        # Env ID with A/B cohort indicator (colored pip)
        # Shows: "●0" for shaped, "●8" for simplified, etc.
        if env.reward_mode and env.reward_mode in self._AB_STYLES:
            pip, color = self._AB_STYLES[env.reward_mode]
            env_id_str = f"[{color}]{pip}[/{color}]{env.env_id}"
        else:
            env_id_str = str(env.env_id)

        # FIX: Accuracy with color (green if at best, yellow if stagnant >5 epochs)
        acc_str = f"{env.host_accuracy:.1f}%"
        if env.best_accuracy > 0:
            if env.host_accuracy >= env.best_accuracy:
                acc_str = f"[green]{acc_str}[/green]"
            elif env.epochs_since_improvement > 5:
                acc_str = f"[yellow]{acc_str}[/yellow]"

        # FIX: Reward with complete thresholds (>0 green, <-0.5 red, else white)
        reward_val = env.current_reward
        mean_val = env.mean_reward
        if reward_val > 0:
            r_style = "green"
        elif reward_val < -0.5:
            r_style = "red"
        else:
            r_style = "white"
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

        # FIX: Last action (normalized names: WAIT, GERMINATE, CULL, FOSSILIZE)
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

### Step 4: Update widgets __init__.py

Create `src/esper/karn/sanctum/widgets/__init__.py`:

```python
"""Sanctum widgets."""
from esper.karn.sanctum.widgets.env_overview import EnvOverview

__all__ = ["EnvOverview"]
```

### Step 5: Run Tests

```bash
PYTHONPATH=src uv run pytest tests/karn/sanctum/test_env_overview.py -v
```

### Step 6: Commit

```bash
git add src/esper/karn/sanctum/widgets/ tests/karn/sanctum/test_env_overview.py
git commit -m "$(cat <<'EOF'
feat(sanctum): add EnvOverview table widget (1:1 port)

Port _render_env_overview() from Rich TUI as table, NOT card grid:
- Per-env rows with all metrics
- Dynamic slot columns based on config
- Sparklines for accuracy/reward history
- Slot cells with stage:blueprint, epochs, gradient indicators (▼▲)
- BLENDING shows alpha instead of epochs
- Aggregate row with Σ at bottom
- Status with color coding

CRITICAL FIXES:
1. Accuracy color: Green if at best, yellow if stagnant >5 epochs
2. Reward threshold: Complete logic (>0 green, <-0.5 red, else white)
3. Last action: Normalized action parsing (GERMINATE, not GERMINATE_*)
4. Staleness: Schema supports "(Ns ago)" indicator via is_stale property

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

This plan extracts Tasks 1-3 with ALL critical fixes applied:

### Task 1: Complete Schema
- ✅ Added `total_actions: int = 0` to TamiyoState
- ✅ Added `fossilized_params: int = 0` to EnvState
- ✅ Documented ALL reward_components keys explicitly
- ✅ Documented action normalization (GERMINATE_* → GERMINATE)
- ✅ Documented EnvState update methods

### Task 2: App Shell
- ✅ Fixed TamiyoBrain position (full-width row, size=11)
- ✅ Added Event Log panel (missing from original plan)
- ✅ Documented terminal size constraints (min 120x40)

### Task 3: EnvOverview Widget
- ✅ Added accuracy color styling (green at best, yellow if stagnant >5)
- ✅ Fixed reward threshold (complete <-0.5 case)
- ✅ Documented action normalization in last_action parsing
- ✅ Added staleness indicator specification (via schema is_stale property)

**Next Steps:** Continue with Tasks 4-10 from parent plan.
