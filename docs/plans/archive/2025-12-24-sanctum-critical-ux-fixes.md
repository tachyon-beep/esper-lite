# Sanctum TUI Critical UX Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 4 critical UX improvements to make Sanctum an effective "air traffic control" dashboard where anomalies surface automatically.

**Architecture:**
- Add new AnomalyStrip widget between RunHeader and main content
- Implement hysteresis in EnvState status tracking (3-epoch debounce)
- Dynamic border styling based on memory alarm state
- New ThreadDeathModal for prominent crash notification

**Tech Stack:** Textual, Rich, dataclasses

---

## Task 1: Add Hysteresis Counter to EnvState Schema

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py:214-220`
- Test: `tests/karn/sanctum/test_schema.py`

**Step 1: Write the failing test**

```python
def test_env_state_status_hysteresis():
    """Status changes require 3 consecutive epochs of the condition."""
    env = EnvState(env_id=0)

    # Initial state
    assert env.status == "initializing"
    assert env.stall_counter == 0

    # Simulate 11 epochs without improvement (should NOT immediately stall)
    for i in range(11):
        env.add_accuracy(50.0, epoch=i)  # Same accuracy = no improvement

    # After 11 epochs, stall_counter should be 1 (first epoch over threshold)
    # Status should still be healthy until counter reaches 3
    assert env.stall_counter >= 1

    # Continue for 2 more epochs
    env.add_accuracy(50.0, epoch=12)
    env.add_accuracy(50.0, epoch=13)

    # NOW status should be stalled (3 consecutive epochs over threshold)
    assert env.status == "stalled"

    # Improvement resets counter
    env.add_accuracy(60.0, epoch=14)  # Better!
    assert env.stall_counter == 0
    assert env.status == "excellent"  # 60% > threshold
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_env_state_status_hysteresis -v`
Expected: FAIL with "AttributeError: 'EnvState' has no attribute 'stall_counter'"

**Step 3: Add stall_counter field to EnvState**

In `src/esper/karn/sanctum/schema.py`, after line 217 (`epochs_since_improvement: int = 0`), add:

```python
    # Hysteresis counter for status changes (prevents flicker)
    # Status only changes after 3 consecutive epochs meeting condition
    stall_counter: int = 0
    degraded_counter: int = 0
```

**Step 4: Update _update_status to use hysteresis**

Replace the `_update_status` method (lines 322-334):

```python
    def _update_status(self, prev_acc: float, curr_acc: float) -> None:
        """Update env status based on metrics with hysteresis.

        Status values: initializing, healthy, excellent, stalled, degraded

        Hysteresis: Status only changes after 3 consecutive epochs meeting
        the condition. This prevents flicker from transient spikes.
        """
        HYSTERESIS_THRESHOLD = 3

        # Check stall condition (>10 epochs without improvement)
        if self.epochs_since_improvement > 10:
            self.stall_counter += 1
            if self.stall_counter >= HYSTERESIS_THRESHOLD:
                self.status = "stalled"
        else:
            self.stall_counter = 0  # Reset on improvement

        # Check degraded condition (accuracy dropped >1%)
        if curr_acc < prev_acc - 1.0:
            self.degraded_counter += 1
            if self.degraded_counter >= HYSTERESIS_THRESHOLD:
                self.status = "degraded"
        else:
            self.degraded_counter = 0  # Reset on stable/improving

        # Positive status updates (no hysteresis needed - immediate feedback is good)
        if self.epochs_since_improvement == 0:  # Just improved
            if curr_acc > 80.0:
                self.status = "excellent"
            elif self.current_epoch > 0:
                self.status = "healthy"
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_env_state_status_hysteresis -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(sanctum): add status hysteresis to prevent flicker

Status changes now require 3 consecutive epochs meeting the condition.
This prevents alert fatigue from transient accuracy fluctuations.

- Add stall_counter and degraded_counter to EnvState
- Update _update_status with 3-epoch debounce
- Positive status updates remain immediate (good feedback)"
```

---

## Task 2: Create AnomalyStrip Widget

**Files:**
- Create: `src/esper/karn/sanctum/widgets/anomaly_strip.py`
- Modify: `src/esper/karn/sanctum/widgets/__init__.py`
- Test: `tests/karn/sanctum/test_anomaly_strip.py`

**Step 1: Write the failing test**

```python
"""Tests for AnomalyStrip widget."""
import pytest
from esper.karn.sanctum.widgets.anomaly_strip import AnomalyStrip
from esper.karn.sanctum.schema import (
    SanctumSnapshot, EnvState, SeedState, TamiyoState, SystemVitals
)


def test_anomaly_strip_no_anomalies():
    """When everything is OK, show green 'ALL CLEAR'."""
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="healthy")
    snapshot.vitals = SystemVitals()
    snapshot.tamiyo = TamiyoState()

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    # Check that the widget reports no anomalies
    assert strip.has_anomalies is False


def test_anomaly_strip_stalled_envs():
    """Stalled envs should be counted and displayed."""
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="stalled")
    snapshot.envs[1] = EnvState(env_id=1, status="stalled")
    snapshot.envs[2] = EnvState(env_id=2, status="healthy")
    snapshot.vitals = SystemVitals()
    snapshot.tamiyo = TamiyoState()

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    assert strip.stalled_count == 2


def test_anomaly_strip_gradient_issues():
    """Seeds with gradient issues should be counted."""
    snapshot = SanctumSnapshot()
    env = EnvState(env_id=0, status="healthy")
    env.seeds["r0c0"] = SeedState(slot_id="r0c0", has_exploding=True)
    env.seeds["r0c1"] = SeedState(slot_id="r0c1", has_vanishing=True)
    env.seeds["r1c0"] = SeedState(slot_id="r1c0")  # OK
    snapshot.envs[0] = env
    snapshot.vitals = SystemVitals()
    snapshot.tamiyo = TamiyoState()

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    assert strip.gradient_issues == 2


def test_anomaly_strip_ppo_health():
    """PPO issues (entropy collapse, high KL) should be flagged."""
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="healthy")
    snapshot.vitals = SystemVitals()
    snapshot.tamiyo = TamiyoState(entropy_collapsed=True)

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    assert strip.ppo_issues is True


def test_anomaly_strip_memory_pressure():
    """Memory pressure should be flagged."""
    snapshot = SanctumSnapshot()
    snapshot.envs[0] = EnvState(env_id=0, status="healthy")
    vitals = SystemVitals(ram_used_gb=14.5, ram_total_gb=16.0)  # 90.6%
    snapshot.vitals = vitals
    snapshot.tamiyo = TamiyoState()

    strip = AnomalyStrip()
    strip.update_snapshot(snapshot)

    assert strip.has_anomalies is True
    assert strip.memory_alarm is True
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_anomaly_strip.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'esper.karn.sanctum.widgets.anomaly_strip'"

**Step 3: Create AnomalyStrip widget**

Create `src/esper/karn/sanctum/widgets/anomaly_strip.py`:

```python
"""AnomalyStrip widget - surfaces problems automatically.

Shows a single-line summary of all anomalies detected across the system.
When everything is OK, displays "ALL CLEAR" in green.
When problems exist, displays counts with color-coded severity.

Layout:
  ANOMALIES: 2 envs stalled | 1 seed exploding | PPO entropy low | MEM 95%
  -- or --
  ALL CLEAR ✓
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class AnomalyStrip(Static):
    """Single-line anomaly summary widget.

    Surfaces problems automatically so operators don't need to scan.
    Red items are critical, yellow are warnings, green means OK.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        # Computed anomaly counts
        self.stalled_count: int = 0
        self.degraded_count: int = 0
        self.gradient_issues: int = 0
        self.ppo_issues: bool = False
        self.memory_alarm: bool = False

    @property
    def has_anomalies(self) -> bool:
        """True if any anomaly is detected."""
        return (
            self.stalled_count > 0
            or self.degraded_count > 0
            or self.gradient_issues > 0
            or self.ppo_issues
            or self.memory_alarm
        )

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update widget with new snapshot data."""
        self._snapshot = snapshot
        self._compute_anomalies()
        self.refresh()

    def _compute_anomalies(self) -> None:
        """Compute all anomaly counts from snapshot."""
        if self._snapshot is None:
            return

        # Reset counts
        self.stalled_count = 0
        self.degraded_count = 0
        self.gradient_issues = 0
        self.ppo_issues = False
        self.memory_alarm = False

        # Count env status issues
        for env in self._snapshot.envs.values():
            if env.status == "stalled":
                self.stalled_count += 1
            elif env.status == "degraded":
                self.degraded_count += 1

            # Count gradient issues across all seeds
            for seed in env.seeds.values():
                if seed.has_exploding or seed.has_vanishing:
                    self.gradient_issues += 1

        # Check PPO health
        tamiyo = self._snapshot.tamiyo
        if tamiyo.entropy_collapsed:
            self.ppo_issues = True
        # High KL divergence (>0.05) is also a warning
        if tamiyo.kl_divergence > 0.05:
            self.ppo_issues = True

        # Check memory pressure
        self.memory_alarm = self._snapshot.vitals.has_memory_alarm

    def render(self) -> Text:
        """Render the anomaly strip."""
        if self._snapshot is None:
            return Text("Waiting for data...", style="dim")

        if not self.has_anomalies:
            return Text("ALL CLEAR ✓", style="bold green")

        # Build anomaly summary
        parts = []

        if self.stalled_count > 0:
            parts.append(("stalled", self.stalled_count, "yellow"))
        if self.degraded_count > 0:
            parts.append(("degraded", self.degraded_count, "red"))
        if self.gradient_issues > 0:
            label = "grad issue" if self.gradient_issues == 1 else "grad issues"
            parts.append((label, self.gradient_issues, "red"))
        if self.ppo_issues:
            parts.append(("PPO", None, "yellow"))
        if self.memory_alarm:
            parts.append(("MEM", None, "red"))

        result = Text()
        result.append("ANOMALIES: ", style="bold red")

        for i, (label, count, color) in enumerate(parts):
            if i > 0:
                result.append(" | ", style="dim")
            if count is not None:
                result.append(f"{count} {label}", style=color)
            else:
                result.append(f"{label} ⚠", style=color)

        return result
```

**Step 4: Add to widget exports**

In `src/esper/karn/sanctum/widgets/__init__.py`, add:

```python
from esper.karn.sanctum.widgets.anomaly_strip import AnomalyStrip

__all__ = [
    # ... existing exports ...
    "AnomalyStrip",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_anomaly_strip.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/widgets/anomaly_strip.py \
        src/esper/karn/sanctum/widgets/__init__.py \
        tests/karn/sanctum/test_anomaly_strip.py
git commit -m "feat(sanctum): add AnomalyStrip widget for automatic problem surfacing

New widget displays a single-line summary of all detected anomalies:
- Stalled/degraded env counts
- Gradient issues (exploding/vanishing)
- PPO health (entropy collapse, high KL)
- Memory pressure

Shows 'ALL CLEAR ✓' in green when no issues detected."
```

---

## Task 3: Integrate AnomalyStrip into App Layout

**Files:**
- Modify: `src/esper/karn/sanctum/app.py:161-186`
- Modify: `src/esper/karn/sanctum/styles.tcss`
- Test: `tests/karn/sanctum/test_app.py`

**Step 1: Write the failing test**

```python
def test_app_has_anomaly_strip():
    """App should have AnomalyStrip widget between RunHeader and main content."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.widgets import AnomalyStrip
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.aggregator import SanctumAggregator

    aggregator = SanctumAggregator()
    backend = SanctumBackend(aggregator)
    app = SanctumApp(backend=backend, num_envs=4)

    async with app.run_test() as pilot:
        # Query for anomaly strip
        strip = app.query_one("#anomaly-strip", AnomalyStrip)
        assert strip is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::test_app_has_anomaly_strip -v`
Expected: FAIL with "NoMatches"

**Step 3: Add AnomalyStrip to app.py compose()**

In `src/esper/karn/sanctum/app.py`, update imports (line 22-30):

```python
from esper.karn.sanctum.widgets import (
    AnomalyStrip,  # ADD THIS
    EnvDetailScreen,
    EnvOverview,
    EventLog,
    HistoricalEnvDetail,
    RunHeader,
    Scoreboard,
    TamiyoBrain,
)
```

Update compose() method (after `yield RunHeader`, before `with Container`):

```python
    def compose(self) -> ComposeResult:
        """Build the Sanctum layout."""
        yield RunHeader(id="run-header")
        yield AnomalyStrip(id="anomaly-strip")  # ADD THIS LINE

        with Container(id="sanctum-main"):
            # ... rest unchanged ...
```

Update `_refresh_all_panels` to include AnomalyStrip:

```python
        # Update anomaly strip (after run header)
        try:
            self.query_one("#anomaly-strip", AnomalyStrip).update_snapshot(snapshot)
        except NoMatches:
            pass
        except Exception as e:
            self.log.warning(f"Failed to update anomaly-strip: {e}")
```

**Step 4: Add CSS for AnomalyStrip**

In `src/esper/karn/sanctum/styles.tcss`, after `#run-header` block:

```css
/* Anomaly Strip - single line below header */
#anomaly-strip {
    height: 1;
    min-height: 1;
    padding: 0 1;
    margin-bottom: 1;
}

/* When anomalies exist, make it stand out */
#anomaly-strip.has-anomalies {
    background: $error-darken-2;
}
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::test_app_has_anomaly_strip -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/app.py src/esper/karn/sanctum/styles.tcss tests/karn/sanctum/test_app.py
git commit -m "feat(sanctum): integrate AnomalyStrip into app layout

AnomalyStrip now appears between RunHeader and main content.
Automatically updated during refresh cycle."
```

---

## Task 4: Dynamic RunHeader Border on Memory Alarm

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/run_header.py:172-306`
- Test: `tests/karn/sanctum/test_run_header.py`

**Step 1: Write the failing test**

```python
def test_run_header_border_red_on_memory_alarm():
    """RunHeader border should be red when memory alarm is active."""
    from rich.panel import Panel
    from esper.karn.sanctum.widgets.run_header import RunHeader
    from esper.karn.sanctum.schema import SanctumSnapshot, SystemVitals

    # Create snapshot with memory alarm
    snapshot = SanctumSnapshot()
    snapshot.vitals = SystemVitals(ram_used_gb=14.5, ram_total_gb=16.0)  # >90%
    snapshot.connected = True

    header = RunHeader()
    header.update_snapshot(snapshot)

    # Render and check border style
    rendered = header.render()
    assert isinstance(rendered, Panel)
    assert rendered.border_style == "bold red"


def test_run_header_border_blue_normally():
    """RunHeader border should be blue when no memory alarm."""
    from rich.panel import Panel
    from esper.karn.sanctum.widgets.run_header import RunHeader
    from esper.karn.sanctum.schema import SanctumSnapshot, SystemVitals

    # Create snapshot without memory alarm
    snapshot = SanctumSnapshot()
    snapshot.vitals = SystemVitals(ram_used_gb=8.0, ram_total_gb=16.0)  # 50%
    snapshot.connected = True

    header = RunHeader()
    header.update_snapshot(snapshot)

    # Render and check border style
    rendered = header.render()
    assert isinstance(rendered, Panel)
    assert rendered.border_style == "blue"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_run_header.py::test_run_header_border_red_on_memory_alarm -v`
Expected: FAIL (border_style is always "blue")

**Step 3: Update render() to use dynamic border**

In `src/esper/karn/sanctum/widgets/run_header.py`, update the render() method's return statement (around line 300-306):

```python
        # System alarm indicator
        alarm_indicator = self._get_system_alarm_indicator()
        alarm_style = "green" if alarm_indicator == "OK" else "bold red"

        # Dynamic border: red when memory alarm active
        border_style = "bold red" if self._snapshot.vitals.has_memory_alarm else "blue"

        return Panel(
            table,
            title="[bold]RUN STATUS[/bold]",
            subtitle=f"[{alarm_style}]{alarm_indicator}[/]",
            subtitle_align="right",
            border_style=border_style,
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_run_header.py::test_run_header_border_red_on_memory_alarm tests/karn/sanctum/test_run_header.py::test_run_header_border_blue_normally -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/run_header.py tests/karn/sanctum/test_run_header.py
git commit -m "feat(sanctum): RunHeader border turns red on memory alarm

Makes memory pressure unmissable - entire header border changes to
bold red when any device exceeds 90% memory usage."
```

---

## Task 5: Create ThreadDeathModal

**Files:**
- Create: `src/esper/karn/sanctum/widgets/thread_death_modal.py`
- Modify: `src/esper/karn/sanctum/widgets/__init__.py`
- Test: `tests/karn/sanctum/test_thread_death_modal.py`

**Step 1: Write the failing test**

```python
"""Tests for ThreadDeathModal widget."""
import pytest
from textual.screen import ModalScreen


def test_thread_death_modal_is_modal():
    """ThreadDeathModal should be a ModalScreen."""
    from esper.karn.sanctum.widgets.thread_death_modal import ThreadDeathModal

    modal = ThreadDeathModal()
    assert isinstance(modal, ModalScreen)


def test_thread_death_modal_has_dismiss_binding():
    """ThreadDeathModal should be dismissable with Escape."""
    from esper.karn.sanctum.widgets.thread_death_modal import ThreadDeathModal

    modal = ThreadDeathModal()
    binding_keys = [b.key for b in modal.BINDINGS]
    assert "escape" in binding_keys
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_thread_death_modal.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create ThreadDeathModal**

Create `src/esper/karn/sanctum/widgets/thread_death_modal.py`:

```python
"""ThreadDeathModal - Prominent notification when training thread dies.

Shows a large, unmissable modal when the training thread stops.
This is a critical failure mode that requires operator attention.
"""
from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static


class ThreadDeathModal(ModalScreen[None]):
    """Modal shown when training thread dies.

    This is a critical failure notification. The modal is large and
    prominent to ensure the operator notices the crash.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
        Binding("enter", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    ThreadDeathModal {
        align: center middle;
        background: $error-darken-3 90%;
    }

    ThreadDeathModal > #death-container {
        width: 60;
        height: auto;
        max-height: 20;
        background: $error-darken-2;
        border: thick $error;
        padding: 2 4;
    }

    ThreadDeathModal .death-title {
        text-align: center;
        text-style: bold;
        color: $error;
        margin-bottom: 1;
    }

    ThreadDeathModal .death-message {
        text-align: center;
        margin-bottom: 1;
    }

    ThreadDeathModal .death-hint {
        text-align: center;
        color: $text-muted;
    }
    """

    def compose(self):
        """Compose the death modal."""
        with Container(id="death-container"):
            yield Static(
                Text("⚠ TRAINING THREAD DIED ⚠", style="bold red"),
                classes="death-title",
            )
            yield Static(
                Text(
                    "The training thread has stopped unexpectedly.\n"
                    "Check the terminal for stack trace.",
                    style="white",
                ),
                classes="death-message",
            )
            yield Static(
                Text("Press ESC, Q, or Enter to close", style="dim"),
                classes="death-hint",
            )
```

**Step 4: Add to widget exports**

In `src/esper/karn/sanctum/widgets/__init__.py`:

```python
from esper.karn.sanctum.widgets.thread_death_modal import ThreadDeathModal

__all__ = [
    # ... existing ...
    "ThreadDeathModal",
]
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_thread_death_modal.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/widgets/thread_death_modal.py \
        src/esper/karn/sanctum/widgets/__init__.py \
        tests/karn/sanctum/test_thread_death_modal.py
git commit -m "feat(sanctum): add ThreadDeathModal for crash notification

Large, prominent modal shown when training thread dies.
Uses error colors and clear messaging to ensure operator notices."
```

---

## Task 6: Integrate ThreadDeathModal into App

**Files:**
- Modify: `src/esper/karn/sanctum/app.py`
- Test: `tests/karn/sanctum/test_app.py`

**Step 1: Write the failing test**

```python
def test_app_shows_thread_death_modal():
    """App should show ThreadDeathModal when thread dies."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.widgets import ThreadDeathModal
    from esper.karn.sanctum.backend import SanctumBackend
    from esper.karn.sanctum.aggregator import SanctumAggregator
    import threading

    aggregator = SanctumAggregator()
    backend = SanctumBackend(aggregator)

    # Create a thread that immediately stops
    dead_thread = threading.Thread(target=lambda: None)
    dead_thread.start()
    dead_thread.join()  # Wait for it to die

    app = SanctumApp(backend=backend, num_envs=4, training_thread=dead_thread)

    async with app.run_test() as pilot:
        # Trigger a refresh which should detect dead thread
        app._poll_and_refresh()
        await pilot.pause()

        # Check that modal was shown
        assert app._thread_death_shown is True
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::test_app_shows_thread_death_modal -v`
Expected: FAIL

**Step 3: Add thread death detection to app**

In `src/esper/karn/sanctum/app.py`, update imports:

```python
from esper.karn.sanctum.widgets import (
    AnomalyStrip,
    EnvDetailScreen,
    EnvOverview,
    EventLog,
    HistoricalEnvDetail,
    RunHeader,
    Scoreboard,
    TamiyoBrain,
    ThreadDeathModal,  # ADD THIS
)
```

Add tracking flag to `__init__`:

```python
    def __init__(self, ...):
        # ... existing code ...
        self._thread_death_shown = False  # Track if we've shown the modal
```

Add detection to `_poll_and_refresh`:

```python
    def _poll_and_refresh(self) -> None:
        """Poll backend for new snapshot and refresh all panels."""
        self._poll_count += 1

        # ... existing code up to thread_alive check ...

        # Check if training thread died (and we haven't shown modal yet)
        thread_alive = self._training_thread.is_alive() if self._training_thread else None
        snapshot.training_thread_alive = thread_alive

        if thread_alive is False and not self._thread_death_shown:
            self._thread_death_shown = True
            self.push_screen(ThreadDeathModal())
            self.log.error("Training thread died! Showing death modal.")

        # ... rest of existing code ...
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::test_app_shows_thread_death_modal -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/app.py tests/karn/sanctum/test_app.py
git commit -m "feat(sanctum): show ThreadDeathModal when training crashes

Detects when training_thread.is_alive() returns False and shows
prominent modal. Modal only shown once per session."
```

---

## Summary

This plan implements all 4 critical UX fixes:

1. **Status Hysteresis** (Task 1) - Prevents flicker with 3-epoch debounce
2. **AnomalyStrip** (Tasks 2-3) - Surfaces problems automatically
3. **Memory Alarm Border** (Task 4) - Makes pressure unmissable
4. **ThreadDeathModal** (Tasks 5-6) - Prominent crash notification

Total: 6 tasks, ~30 steps, estimated 60-90 minutes implementation time.
