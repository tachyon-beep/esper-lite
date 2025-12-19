# Sanctum UX Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure Sanctum TUI layout to serve as both a teaching tool and operational telemetry dashboard

**Architecture:** Three-phase approach: (1) Layout restructure with panel removal, (2) TamiyoBrain complete redesign with Learning Vitals and Decision Snapshot, (3) EventLog enhancement with color coding and episode grouping. EnvOverview and Scoreboard are SACRED and must not be touched.

**Tech Stack:** Textual (TUI framework), Rich (rendering), Python dataclasses (schema)

---

## Phase 1: Layout Restructure

### Task 1: Update Schema with System Alarm Data

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py:267-296` (SystemVitals)

**Step 1: Write the failing test**

Create test for system alarm threshold detection.

```python
# tests/karn/sanctum/test_schema.py (append)

def test_system_vitals_memory_alarm_threshold():
    """Memory above 90% should trigger alarm state."""
    vitals = SystemVitals(
        gpu_memory_used_gb=9.5,
        gpu_memory_total_gb=10.0,  # 95% usage
        ram_used_gb=30.0,
        ram_total_gb=32.0,  # 93.75% usage
    )
    assert vitals.has_memory_alarm is True
    assert vitals.memory_alarm_devices == ["cuda:0"]


def test_system_vitals_no_alarm_below_threshold():
    """Memory below 90% should not trigger alarm."""
    vitals = SystemVitals(
        gpu_memory_used_gb=7.0,
        gpu_memory_total_gb=10.0,  # 70% usage
        ram_used_gb=20.0,
        ram_total_gb=32.0,  # 62.5% usage
    )
    assert vitals.has_memory_alarm is False
    assert vitals.memory_alarm_devices == []
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_system_vitals_memory_alarm_threshold -v`
Expected: FAIL with "AttributeError: 'SystemVitals' object has no attribute 'has_memory_alarm'"

**Step 3: Write minimal implementation**

```python
# In SystemVitals dataclass, add these properties after existing fields:

    @property
    def has_memory_alarm(self) -> bool:
        """Check if any device exceeds 90% memory usage."""
        # Check RAM
        if self.ram_total_gb > 0 and (self.ram_used_gb / self.ram_total_gb) > 0.90:
            return True
        # Check GPUs
        for stats in self.gpu_stats.values():
            if stats.memory_total_gb > 0:
                usage = stats.memory_used_gb / stats.memory_total_gb
                if usage > 0.90:
                    return True
        # Fallback single GPU
        if self.gpu_memory_total_gb > 0:
            usage = self.gpu_memory_used_gb / self.gpu_memory_total_gb
            if usage > 0.90:
                return True
        return False

    @property
    def memory_alarm_devices(self) -> list[str]:
        """Get list of devices exceeding 90% memory usage."""
        devices = []
        for device, stats in self.gpu_stats.items():
            if stats.memory_total_gb > 0:
                usage = stats.memory_used_gb / stats.memory_total_gb
                if usage > 0.90:
                    devices.append(device)
        return devices
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_system_vitals_memory_alarm_threshold tests/karn/sanctum/test_schema.py::test_system_vitals_no_alarm_below_threshold -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/karn/sanctum/test_schema.py src/esper/karn/sanctum/schema.py
git commit -m "feat(sanctum): add memory alarm threshold detection to SystemVitals

Add has_memory_alarm and memory_alarm_devices properties for exception-based
monitoring. Threshold is 90% for both RAM and GPU memory.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 2: Add System Alarm Indicator to RunHeader

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/run_header.py:139-257` (render method)
- Test: `tests/karn/sanctum/test_run_header.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_run_header.py (append)

def test_run_header_system_alarm_ok():
    """Test OK indicator when no memory alarms."""
    from esper.karn.sanctum.schema import SystemVitals

    snapshot = SanctumSnapshot(
        connected=True,
        staleness_seconds=1.0,
    )
    snapshot.vitals = SystemVitals(
        gpu_memory_used_gb=5.0,
        gpu_memory_total_gb=10.0,  # 50% - OK
    )

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    # Should show OK indicator in title
    assert "OK" in rendered


def test_run_header_system_alarm_triggered():
    """Test alarm indicator when memory threshold exceeded."""
    from esper.karn.sanctum.schema import SystemVitals, GPUStats

    snapshot = SanctumSnapshot(
        connected=True,
        staleness_seconds=1.0,
    )
    snapshot.vitals = SystemVitals(
        gpu_stats={"cuda:0": GPUStats(
            device_id="cuda:0",
            memory_used_gb=9.5,
            memory_total_gb=10.0,  # 95% - ALARM
        )},
    )

    widget = RunHeader()
    widget.update_snapshot(snapshot)
    panel = widget.render()
    rendered = render_to_text(panel)

    # Should show alarm indicator
    assert "cuda:0" in rendered or "RAM" in rendered
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_run_header.py::test_run_header_system_alarm_ok -v`
Expected: FAIL (OK not in rendered output in right position)

**Step 3: Write minimal implementation**

Modify `run_header.py` render method to add alarm indicator to Panel title:

```python
# In RunHeader.render(), replace the final Panel creation:

    def _get_system_alarm_indicator(self) -> str:
        """Get system alarm indicator for header.

        Returns:
            Empty string if OK, or alarm indicator like "[cuda:0] RAM 92%"
        """
        if self._snapshot is None:
            return "OK"

        vitals = self._snapshot.vitals
        if not vitals.has_memory_alarm:
            return "OK"

        # Build alarm indicator
        alarms = []
        for device in vitals.memory_alarm_devices:
            stats = vitals.gpu_stats.get(device)
            if stats and stats.memory_total_gb > 0:
                pct = int((stats.memory_used_gb / stats.memory_total_gb) * 100)
                alarms.append(f"[{device}] RAM {pct}%")

        return " │ ".join(alarms) if alarms else "OK"

# Then in render(), change the Panel creation:
        alarm_indicator = self._get_system_alarm_indicator()
        alarm_style = "green" if alarm_indicator == "OK" else "red bold"

        return Panel(
            table,
            title=f"[bold]RUN STATUS[/bold]",
            subtitle=f"[{alarm_style}]{alarm_indicator}[/]",
            subtitle_align="right",
            border_style="blue",
        )
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_run_header.py::test_run_header_system_alarm_ok tests/karn/sanctum/test_run_header.py::test_run_header_system_alarm_triggered -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/run_header.py tests/karn/sanctum/test_run_header.py
git commit -m "feat(sanctum): add system alarm indicator to RunHeader

Exception-based monitoring: shows OK when healthy, shows device + percentage
when memory exceeds 90% threshold. Appears as subtitle on right side of header.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Delete SystemResources and TrainingHealth Widgets

**Files:**
- Delete: `src/esper/karn/sanctum/widgets/system_resources.py`
- Delete: `src/esper/karn/sanctum/widgets/training_health.py`
- Modify: `src/esper/karn/sanctum/widgets/__init__.py`
- Modify: `src/esper/karn/sanctum/app.py`

**Step 1: Update widget exports**

```python
# src/esper/karn/sanctum/widgets/__init__.py
"""Sanctum TUI widgets."""
from esper.karn.sanctum.widgets.env_detail_screen import EnvDetailScreen
from esper.karn.sanctum.widgets.env_overview import EnvOverview
from esper.karn.sanctum.widgets.event_log import EventLog
from esper.karn.sanctum.widgets.run_header import RunHeader
from esper.karn.sanctum.widgets.scoreboard import Scoreboard
from esper.karn.sanctum.widgets.tamiyo_brain import TamiyoBrain

__all__ = [
    "EnvDetailScreen",
    "EnvOverview",
    "EventLog",
    "RunHeader",
    "Scoreboard",
    "TamiyoBrain",
]
```

**Step 2: Update app.py imports and compose**

```python
# src/esper/karn/sanctum/app.py

# Remove from imports:
# SystemResources, TrainingHealth

# In compose() method, replace bottom section with:
            # Bottom section: Event Log (left) | TamiyoBrain (right)
            with Horizontal(id="bottom-section"):
                # Left side: Event Log (50%)
                yield EventLog(id="event-log")
                # Right side: TamiyoBrain (50%)
                yield TamiyoBrain(id="tamiyo-brain")
```

**Step 3: Remove refresh calls for deleted widgets**

Remove these blocks from `_refresh_all_panels`:

```python
# DELETE these blocks:
        try:
            self.query_one("#system-resources", SystemResources).update_snapshot(snapshot)
        except NoMatches:
            pass

        try:
            self.query_one("#training-health", TrainingHealth).update_snapshot(snapshot)
        except NoMatches:
            pass
```

**Step 4: Delete the widget files**

```bash
rm src/esper/karn/sanctum/widgets/system_resources.py
rm src/esper/karn/sanctum/widgets/training_health.py
```

**Step 5: Run tests to verify nothing breaks**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v --ignore=tests/karn/sanctum/test_remaining_widgets.py`
Expected: PASS (may need to update test_remaining_widgets.py)

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor(sanctum): remove SystemResources and TrainingHealth widgets

These panels are being replaced:
- SystemResources → header alarm indicator (Task 2)
- TrainingHealth → merged into TamiyoBrain (Phase 2)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 4: Update Layout Styles

**Files:**
- Modify: `src/esper/karn/sanctum/styles.tcss`

**Step 1: Update CSS for new layout**

```css
/* Sanctum Diagnostic TUI Styles
 * New layout: Top (EnvOverview + Scoreboard) | Bottom (EventLog + TamiyoBrain)
 */

/* Run Header - fixed height for 2-row display */
#run-header {
    height: 6;
    min-height: 6;
    margin-bottom: 1;
}

#sanctum-main {
    height: 1fr;
    padding: 0 1;
}

/* Top section: Env Overview + Scoreboard (~55% of main area) */
#top-section {
    height: 55%;
    min-height: 15;
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

/* Bottom section: Event Log + TamiyoBrain (~45% of main area) */
#bottom-section {
    height: 45%;
    min-height: 12;
}

/* Event Log - system velocity (50% width) */
#event-log {
    width: 50%;
    height: 1fr;
    border: solid yellow;
    border-title-color: yellow;
    margin-right: 1;
    overflow-y: auto;
    padding: 0 1;
}

#event-log:focus {
    border: double $accent;
}

/* Tamiyo Brain - learning + decisions (50% width) */
#tamiyo-brain {
    width: 50%;
    height: 1fr;
    border: solid magenta;
}

#tamiyo-brain:focus {
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
```

**Step 2: Run the app visually to verify layout**

Run: `PYTHONPATH=src uv run python -m esper.karn.sanctum.app` (or integration test)

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/styles.tcss
git commit -m "style(sanctum): update layout for new 50/50 bottom split

- Top section: EnvOverview (65%) + Scoreboard (35%) - ~55% height
- Bottom section: EventLog (50%) + TamiyoBrain (50%) - ~45% height
- Removed styles for deleted SystemResources and TrainingHealth

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: TamiyoBrain Redesign

### Task 5: Add Decision Snapshot Schema

**Files:**
- Modify: `src/esper/karn/sanctum/schema.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_schema.py (append)

def test_decision_snapshot_creation():
    """Test DecisionSnapshot dataclass creation."""
    from esper.karn.sanctum.schema import DecisionSnapshot

    decision = DecisionSnapshot(
        timestamp=datetime.now(timezone.utc),
        slot_states={"r0c0": "Training 12%", "r0c1": "Empty"},
        host_accuracy=67.0,
        chosen_action="GERMINATE",
        chosen_slot="r0c1",
        confidence=0.73,
        expected_value=0.42,
        actual_reward=0.38,
        alternatives=[("WAIT", 0.15), ("BLEND r0c0", 0.12)],
    )

    assert decision.chosen_action == "GERMINATE"
    assert decision.confidence == 0.73
    assert len(decision.alternatives) == 2
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_decision_snapshot_creation -v`
Expected: FAIL with "ImportError: cannot import name 'DecisionSnapshot'"

**Step 3: Write minimal implementation**

```python
# Add to schema.py after RewardComponents:

@dataclass
class DecisionSnapshot:
    """Snapshot of a single Tamiyo decision for display.

    Captures what Tamiyo saw, what she chose, and the outcome.
    Used for the "Last Decision" section of TamiyoBrain.
    """
    timestamp: datetime
    slot_states: dict[str, str]  # slot_id -> "Training 12%" or "Empty"
    host_accuracy: float
    chosen_action: str  # "GERMINATE", "WAIT", "CULL", "FOSSILIZE"
    chosen_slot: str | None  # Target slot for action (None for WAIT)
    confidence: float  # Action probability (0-1)
    expected_value: float  # Value estimate before action
    actual_reward: float | None  # Actual reward received (None if pending)
    alternatives: list[tuple[str, float]]  # [(action_name, probability), ...]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_schema.py::test_decision_snapshot_creation -v`
Expected: PASS

**Step 5: Add DecisionSnapshot to TamiyoState**

```python
# In TamiyoState dataclass, add:
    # Last decision snapshot (sampled ~1/minute)
    last_decision: DecisionSnapshot | None = None
```

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_schema.py
git commit -m "feat(sanctum): add DecisionSnapshot schema for Tamiyo decisions

Captures: what Tamiyo saw (slot states, host accuracy), what she chose
(action + confidence), expected vs actual reward, and alternatives considered.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Redesign TamiyoBrain Widget - Learning Vitals

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_tamiyo_brain.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_tamiyo_brain.py (append)

@pytest.mark.asyncio
async def test_tamiyo_brain_learning_vitals_section():
    """TamiyoBrain should have Learning Vitals section with action bar and gauges."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            action_counts={"WAIT": 35, "GERMINATE": 25, "CULL": 0, "FOSSILIZE": 40},
            total_actions=100,
            entropy=0.42,
            value_loss=0.08,
            advantage_mean=0.31,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Widget should render Learning Vitals section
        assert widget._snapshot.tamiyo.entropy == 0.42
        assert widget._snapshot.tamiyo.value_loss == 0.08
        assert widget._snapshot.tamiyo.advantage_mean == 0.31


@pytest.mark.asyncio
async def test_tamiyo_brain_action_distribution_bar():
    """Action distribution should render as horizontal stacked bar."""
    app = TamiyoBrainTestApp()
    async with app.run_test():
        widget = app.query_one(TamiyoBrain)
        snapshot = SanctumSnapshot(slot_ids=["R0C0"])
        snapshot.tamiyo = TamiyoState(
            action_counts={"WAIT": 35, "GERMINATE": 25, "CULL": 0, "FOSSILIZE": 40},
            total_actions=100,
            ppo_data_received=True,
        )

        widget.update_snapshot(snapshot)

        # Test the action bar rendering method
        bar = widget._render_action_distribution_bar()
        assert bar is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py::test_tamiyo_brain_learning_vitals_section -v`
Expected: PASS (just tests data) or FAIL if method doesn't exist

**Step 3: Rewrite TamiyoBrain with new layout**

```python
"""TamiyoBrain widget - Policy agent diagnostics (Redesigned).

New layout focuses on answering:
- "What is Tamiyo doing?" (Action distribution bar)
- "Is she learning?" (Entropy, Value Loss gauges)
- "What did she just decide?" (Last Decision snapshot)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.widgets import Static

from esper.karn.constants import TUIThresholds

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


class TamiyoBrain(Static):
    """TamiyoBrain widget - Policy agent diagnostics.

    New two-section layout:
    1. LEARNING VITALS - Action distribution bar + gauges (entropy, value loss, advantage)
    2. LAST DECISION - What Tamiyo saw, chose, and got
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        self._snapshot = snapshot
        self.refresh()

    def render(self) -> Panel:
        if self._snapshot is None:
            return Panel("No data", title="TAMIYO", border_style="magenta")

        if not self._snapshot.tamiyo.ppo_data_received:
            waiting_text = Text(justify="center")
            waiting_text.append("⏳ Waiting for PPO data...\n", style="dim italic")
            waiting_text.append(
                f"Progress: {self._snapshot.current_epoch}/{self._snapshot.max_epochs} epochs",
                style="cyan",
            )
            return Panel(
                waiting_text,
                title="[bold magenta]TAMIYO[/bold magenta]",
                border_style="magenta dim",
            )

        # Main layout: two sections stacked
        main_table = Table.grid(expand=True)
        main_table.add_column(ratio=1)

        # Section 1: Learning Vitals
        vitals_panel = self._render_learning_vitals()
        main_table.add_row(vitals_panel)

        # Section 2: Last Decision (if available)
        decision_panel = self._render_last_decision()
        main_table.add_row(decision_panel)

        return Panel(
            main_table,
            title="[bold magenta]TAMIYO[/bold magenta]",
            border_style="magenta",
        )

    def _render_learning_vitals(self) -> Panel:
        """Render Learning Vitals section with action bar and gauges."""
        tamiyo = self._snapshot.tamiyo

        content = Table.grid(expand=True)
        content.add_column(ratio=1)

        # Row 1: Action distribution bar
        action_bar = self._render_action_distribution_bar()
        content.add_row(action_bar)

        # Row 2: Gauges (Entropy, Value Loss, Advantage)
        gauges = Table.grid(expand=True)
        gauges.add_column(ratio=1)
        gauges.add_column(ratio=1)
        gauges.add_column(ratio=1)

        entropy_gauge = self._render_gauge(
            "Entropy", tamiyo.entropy, 0, 2.0,
            self._get_entropy_label(tamiyo.entropy)
        )
        value_gauge = self._render_gauge(
            "Value Loss", tamiyo.value_loss, 0, 1.0,
            self._get_value_loss_label(tamiyo.value_loss)
        )
        advantage_gauge = self._render_gauge(
            "Advantage", tamiyo.advantage_mean, -1.0, 1.0,
            self._get_advantage_label(tamiyo.advantage_mean)
        )

        gauges.add_row(entropy_gauge, value_gauge, advantage_gauge)
        content.add_row(gauges)

        return Panel(content, title="LEARNING VITALS", border_style="dim")

    def _render_action_distribution_bar(self) -> Text:
        """Render horizontal stacked bar for action distribution."""
        tamiyo = self._snapshot.tamiyo
        total = tamiyo.total_actions

        if total == 0:
            return Text("Actions: [no data]", style="dim")

        # Calculate percentages
        pcts = {a: (c / total) * 100 for a, c in tamiyo.action_counts.items()}

        # Build stacked bar (width 40 chars)
        bar_width = 40
        bar = Text("Actions: [")

        # Color mapping
        colors = {
            "GERMINATE": "green",
            "WAIT": "dim",
            "BLEND": "cyan",  # Blending includes FOSSILIZE transitions
            "FOSSILIZE": "blue",
            "CULL": "red",
        }

        for action in ["GERMINATE", "WAIT", "FOSSILIZE", "CULL"]:
            pct = pcts.get(action, 0)
            width = int((pct / 100) * bar_width)
            if width > 0:
                bar.append("▓" * width, style=colors.get(action, "white"))

        bar.append("]")

        # Add legend
        bar.append("  ")
        for action in ["GERMINATE", "WAIT", "FOSSILIZE"]:
            if pcts.get(action, 0) > 0:
                bar.append(f"{action[:4]} {pcts[action]:.0f}%  ", style=colors.get(action, "white"))

        return bar

    def _render_gauge(self, label: str, value: float, min_val: float, max_val: float, description: str) -> Text:
        """Render a single gauge with label and description."""
        # Normalize to 0-1
        normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        normalized = max(0, min(1, normalized))

        # Build gauge bar (width 12)
        gauge_width = 12
        filled = int(normalized * gauge_width)
        empty = gauge_width - filled

        gauge = Text()
        gauge.append(f"{label}: ", style="dim")
        gauge.append("[")
        gauge.append("█" * filled, style="cyan")
        gauge.append("░" * empty, style="dim")
        gauge.append("]")
        gauge.append(f" {value:.2f}  ", style="cyan")
        gauge.append(f'"{description}"', style="italic dim")

        return gauge

    def _get_entropy_label(self, entropy: float) -> str:
        if entropy < TUIThresholds.ENTROPY_CRITICAL:
            return "Collapsed!"
        elif entropy < TUIThresholds.ENTROPY_WARNING:
            return "Getting decisive"
        else:
            return "Exploring"

    def _get_value_loss_label(self, value_loss: float) -> str:
        if value_loss < 0.1:
            return "Learning well"
        elif value_loss < 0.5:
            return "Still learning"
        else:
            return "Struggling"

    def _get_advantage_label(self, advantage: float) -> str:
        if advantage > 0.2:
            return "Choices working"
        elif advantage > 0:
            return "Slight edge"
        else:
            return "Needs improvement"

    def _render_last_decision(self) -> Panel:
        """Render Last Decision section."""
        tamiyo = self._snapshot.tamiyo
        decision = tamiyo.last_decision

        if decision is None:
            return Panel(
                Text("No decisions captured yet", style="dim italic"),
                title="LAST DECISION",
                border_style="dim",
            )

        # Build decision display
        content = Table.grid(expand=True)
        content.add_column(ratio=1)

        # Time since decision
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        age = (now - decision.timestamp).total_seconds()
        age_str = f"{age:.1f}s ago" if age < 60 else f"{age/60:.0f}m ago"

        # SAW line
        saw_line = Text()
        saw_line.append("SAW:  ", style="bold")
        for slot_id, state in decision.slot_states.items():
            saw_line.append(f"{slot_id}: {state} │ ", style="dim")
        saw_line.append(f"Host: {decision.host_accuracy:.0f}%", style="cyan")
        content.add_row(saw_line)

        # CHOSE line
        chose_line = Text()
        chose_line.append("CHOSE: ", style="bold")
        chose_line.append(f"{decision.chosen_action}", style="green bold")
        if decision.chosen_slot:
            chose_line.append(f" {decision.chosen_slot}", style="cyan")
        chose_line.append(f" ({decision.confidence:.0%})", style="dim")
        content.add_row(chose_line)

        # EXPECTED vs GOT line
        result_line = Text()
        result_line.append("EXPECTED: ", style="dim")
        result_line.append(f"{decision.expected_value:+.2f}", style="cyan")
        result_line.append("  →  GOT: ", style="dim")
        if decision.actual_reward is not None:
            diff = decision.actual_reward - decision.expected_value
            style = "green" if abs(diff) < 0.1 else ("yellow" if diff > 0 else "red")
            result_line.append(f"{decision.actual_reward:+.2f} ", style=style)
            result_line.append("✓" if abs(diff) < 0.1 else "✗", style=style)
        else:
            result_line.append("pending...", style="dim italic")
        content.add_row(result_line)

        # Alternatives line
        if decision.alternatives:
            alt_line = Text()
            alt_line.append("Also: ", style="dim")
            for action, prob in decision.alternatives[:2]:
                alt_line.append(f"{action} ({prob:.0%}), ", style="dim")
            content.add_row(alt_line)

        return Panel(content, title=f"LAST DECISION ({age_str})", border_style="dim")
```

**Step 4: Run tests to verify**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_tamiyo_brain.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/widgets/tamiyo_brain.py tests/karn/sanctum/test_tamiyo_brain.py
git commit -m "feat(sanctum): redesign TamiyoBrain with Learning Vitals and Decision Snapshot

New two-section layout:
1. LEARNING VITALS - horizontal action bar + gauges (entropy, value loss, advantage)
2. LAST DECISION - what Tamiyo saw, chose, expected, and got

Answers key teaching questions:
- What is Tamiyo doing? (action distribution)
- Is she learning? (entropy + value loss gauges)
- What did she just decide and why? (decision snapshot)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Wire Up Decision Capture in Aggregator

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`
- Test: `tests/karn/sanctum/test_backend.py`

**Step 1: Write the failing test**

```python
# tests/karn/sanctum/test_backend.py (append or new file)

def test_aggregator_captures_decision_snapshot():
    """Aggregator should capture DecisionSnapshot from REWARD_COMPUTED events."""
    from esper.karn.sanctum.aggregator import SanctumAggregator
    from esper.leyline import TelemetryEvent, EventType
    from datetime import datetime, timezone

    agg = SanctumAggregator(num_envs=4)

    # Simulate REWARD_COMPUTED with decision data
    event = TelemetryEvent(
        event_type=EventType.REWARD_COMPUTED,
        timestamp=datetime.now(timezone.utc),
        epoch=10,
        data={
            "env_id": 0,
            "total_reward": 0.38,
            "action_name": "GERMINATE",
            "action_slot": "r0c1",
            "action_confidence": 0.73,
            "value_estimate": 0.42,
            "slot_states": {"r0c0": "Training 12%", "r0c1": "Empty"},
            "host_accuracy": 67.0,
            "alternatives": [("WAIT", 0.15), ("BLEND", 0.12)],
        },
    )

    agg.process_event(event)
    snapshot = agg.get_snapshot()

    # Decision should be captured
    decision = snapshot.tamiyo.last_decision
    assert decision is not None
    assert decision.chosen_action == "GERMINATE"
    assert decision.confidence == 0.73
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_backend.py::test_aggregator_captures_decision_snapshot -v`
Expected: FAIL with assertion error (decision is None)

**Step 3: Modify aggregator to capture decisions**

```python
# In aggregator.py, modify _handle_reward_computed:

from esper.karn.sanctum.schema import DecisionSnapshot

    def _handle_reward_computed(self, event: "TelemetryEvent") -> None:
        """Handle REWARD_COMPUTED event with per-env routing and decision capture."""
        data = event.data or {}
        env_id = data.get("env_id", 0)
        epoch = event.epoch or 0

        self._ensure_env(env_id)
        env = self._envs[env_id]

        # Update reward tracking (existing code)
        total_reward = data.get("total_reward", 0.0)
        env.reward_history.append(total_reward)
        env.current_epoch = epoch

        # Update action tracking (existing code)
        action_name = normalize_action(data.get("action_name", "WAIT"))
        env.action_history.append(action_name)
        env.action_counts[action_name] = env.action_counts.get(action_name, 0) + 1
        env.total_actions += 1

        # ... existing reward components code ...

        # Capture decision snapshot (NEW)
        # Only capture if we have decision data and sample ~1 per minute
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

        # Also update global action counts for Tamiyo
        self._tamiyo.action_counts[action_name] = self._tamiyo.action_counts.get(action_name, 0) + 1
        self._tamiyo.total_actions += 1
        self._tamiyo.ppo_data_received = True  # Mark that we have data

        env.last_update = datetime.now(timezone.utc)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_backend.py::test_aggregator_captures_decision_snapshot -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py tests/karn/sanctum/test_backend.py
git commit -m "feat(sanctum): capture decision snapshots in aggregator

Extracts decision data from REWARD_COMPUTED events when available:
- slot_states, host_accuracy (what Tamiyo saw)
- action_name, action_slot, action_confidence (what she chose)
- value_estimate, total_reward (expected vs actual)
- alternatives (runner-up actions)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: EventLog Enhancement

### Task 8: Add Episode Grouping to EventLog

**Files:**
- Modify: `src/esper/karn/sanctum/widgets/event_log.py`
- Modify: `src/esper/karn/sanctum/schema.py` (EventLogEntry)
- Test: `tests/karn/sanctum/test_event_log.py`

**Step 1: Update EventLogEntry schema**

```python
# In schema.py, modify EventLogEntry:

@dataclass
class EventLogEntry:
    """Single event log entry for display in Event Log panel."""
    timestamp: str  # Formatted as HH:MM:SS
    event_type: str  # REWARD_COMPUTED, SEED_GERMINATED, etc.
    env_id: int | None  # None for global events (PPO, BATCH)
    message: str  # Formatted message for display
    episode: int = 0  # Episode number for grouping
    relative_time: str = ""  # "(2s ago)" relative time string
```

**Step 2: Write the failing test**

```python
# tests/karn/sanctum/test_event_log.py (new file)

import pytest
from textual.app import App

from esper.karn.sanctum.schema import EventLogEntry, SanctumSnapshot
from esper.karn.sanctum.widgets.event_log import EventLog


class EventLogTestApp(App):
    def compose(self):
        yield EventLog()


@pytest.mark.asyncio
async def test_event_log_episode_grouping():
    """EventLog should group events by episode with separators."""
    app = EventLogTestApp()
    async with app.run_test():
        widget = app.query_one(EventLog)

        snapshot = SanctumSnapshot()
        snapshot.event_log = [
            EventLogEntry(
                timestamp="12:34:56",
                event_type="SEED_GERMINATED",
                env_id=0,
                message="seed_0a3f germinated",
                episode=5,
                relative_time="(2s)",
            ),
            EventLogEntry(
                timestamp="12:34:51",
                event_type="REWARD_COMPUTED",
                env_id=0,
                message="WAIT r=+0.12",
                episode=5,
                relative_time="(7s)",
            ),
            EventLogEntry(
                timestamp="12:33:12",
                event_type="BATCH_COMPLETED",
                env_id=None,
                message="Episode 4 complete",
                episode=4,
                relative_time="(1m)",
            ),
        ]

        widget.update_snapshot(snapshot)

        # Should have episode grouping
        assert widget._snapshot is not None
        assert len(widget._snapshot.event_log) == 3


@pytest.mark.asyncio
async def test_event_log_color_coding():
    """EventLog should color-code events by type."""
    app = EventLogTestApp()
    async with app.run_test():
        widget = app.query_one(EventLog)

        # Test color mapping
        assert widget._get_event_color("SEED_GERMINATED") == "bright_yellow"
        assert widget._get_event_color("SEED_FOSSILIZED") == "bright_green"
        assert widget._get_event_color("SEED_CULLED") == "bright_red"
        assert widget._get_event_color("REWARD_COMPUTED") == "bright_cyan"
```

**Step 3: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_event_log.py -v`
Expected: FAIL (method doesn't exist or wrong color)

**Step 4: Implement enhanced EventLog**

```python
"""EventLog widget - Recent event feed with color coding and episode grouping.

Enhanced design:
- Full-width rows using all horizontal space
- Color-coded by event type (green=lifecycle, cyan=tamiyo, yellow=warning, red=error)
- Timestamp + relative time "(2s ago)"
- Episode grouping with visual separators
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Group
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import EventLogEntry, SanctumSnapshot


# Event type color mapping
_EVENT_COLORS: dict[str, str] = {
    # Seed lifecycle (green family)
    "SEED_GERMINATED": "bright_yellow",
    "SEED_STAGE_CHANGED": "bright_white",
    "SEED_FOSSILIZED": "bright_green",
    "SEED_CULLED": "bright_red",
    # Tamiyo actions (cyan)
    "REWARD_COMPUTED": "bright_cyan",
    # Training events (blue)
    "TRAINING_STARTED": "bright_green",
    "EPOCH_COMPLETED": "bright_blue",
    "PPO_UPDATE_COMPLETED": "bright_magenta",
    "BATCH_COMPLETED": "bright_blue",
}

# Event type emoji mapping
_EVENT_EMOJI: dict[str, str] = {
    "SEED_GERMINATED": "🌱",
    "SEED_FOSSILIZED": "✅",
    "SEED_CULLED": "⚠️",
    "REWARD_COMPUTED": "📊",
    "BATCH_COMPLETED": "🏆",
}


class EventLog(Static):
    """Event log widget showing recent telemetry events.

    Enhanced features:
    - Full-width rows with timestamp + relative time
    - Color coding by event type
    - Episode grouping with separators
    - Emoji prefixes for quick scanning
    """

    def __init__(self, max_events: int = 20, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_events = max_events
        self._snapshot: SanctumSnapshot | None = None
        self.border_title = "EVENTS"

    def _get_event_color(self, event_type: str) -> str:
        """Get color for event type."""
        return _EVENT_COLORS.get(event_type, "white")

    def _get_event_emoji(self, event_type: str) -> str:
        """Get emoji for event type."""
        return _EVENT_EMOJI.get(event_type, "")

    def render(self):
        """Render the event log with episode grouping."""
        if self._snapshot is None or not self._snapshot.event_log:
            return Text("Waiting for events...", style="dim")

        events = list(self._snapshot.event_log)[-self._max_events:]
        lines = []
        last_episode = None

        for entry in events:
            # Episode separator
            if entry.episode != last_episode and last_episode is not None:
                separator = Text(f"─── Episode {entry.episode} ", style="dim")
                separator.append("─" * 40, style="dim")
                lines.append(separator)
            last_episode = entry.episode

            # Event line
            color = self._get_event_color(entry.event_type)
            emoji = self._get_event_emoji(entry.event_type)

            text = Text()
            text.append(f"{entry.timestamp} ", style="dim")
            if entry.relative_time:
                text.append(f"{entry.relative_time} ", style="dim")
            if emoji:
                text.append(f"{emoji} ")
            if entry.env_id is not None:
                text.append(f"ENV:{entry.env_id:02d} ", style="bright_blue")
            text.append(entry.message, style=color)

            lines.append(text)

        return Group(*lines)

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        self._snapshot = snapshot
        self.refresh()
```

**Step 5: Run tests to verify**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_event_log.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/widgets/event_log.py src/esper/karn/sanctum/schema.py tests/karn/sanctum/test_event_log.py
git commit -m "feat(sanctum): enhance EventLog with episode grouping and color coding

- Full-width rows using all horizontal space
- Color-coded by event type (green=lifecycle, cyan=tamiyo, yellow=warning, red=error)
- Timestamp + relative time display
- Episode grouping with visual separators
- Emoji prefixes for quick scanning

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 9: Update Aggregator for Enhanced Event Logging

**Files:**
- Modify: `src/esper/karn/sanctum/aggregator.py`

**Step 1: Modify _add_event_log to include episode and relative time**

```python
# In aggregator.py, modify _add_event_log:

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

        # Format message (existing code)
        # ... message formatting logic ...

        self._event_log.append(EventLogEntry(
            timestamp=timestamp.strftime("%H:%M:%S"),
            event_type=event_type,
            env_id=env_id,
            message=message,
            episode=self._current_episode,
            relative_time=relative_time,
        ))
```

**Step 2: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/esper/karn/sanctum/aggregator.py
git commit -m "feat(sanctum): add episode and relative time to event log entries

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 10: Final Integration Test

**Files:**
- Test: `tests/karn/sanctum/test_app_integration.py`

**Step 1: Write integration test for new layout**

```python
# tests/karn/sanctum/test_app_integration.py (append)

@pytest.mark.asyncio
async def test_new_layout_structure():
    """Test that new layout has correct panel structure."""
    from esper.karn.sanctum.app import SanctumApp
    from esper.karn.sanctum.backend import SanctumBackend

    backend = SanctumBackend()
    app = SanctumApp(backend=backend, num_envs=4)

    async with app.run_test() as pilot:
        # Should have EnvOverview and Scoreboard in top section
        assert app.query_one("#env-overview") is not None
        assert app.query_one("#scoreboard") is not None

        # Should have EventLog and TamiyoBrain in bottom section
        assert app.query_one("#event-log") is not None
        assert app.query_one("#tamiyo-brain") is not None

        # Should NOT have SystemResources or TrainingHealth
        from textual.css.query import NoMatches
        with pytest.raises(NoMatches):
            app.query_one("#system-resources")
        with pytest.raises(NoMatches):
            app.query_one("#training-health")
```

**Step 2: Run integration test**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app_integration.py::test_new_layout_structure -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v`
Expected: ALL PASS

**Step 4: Final commit**

```bash
git add tests/karn/sanctum/test_app_integration.py
git commit -m "test(sanctum): add integration test for new layout structure

Verifies:
- EnvOverview and Scoreboard present (DO NOT TOUCH panels)
- EventLog and TamiyoBrain in bottom section
- SystemResources and TrainingHealth removed

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Schema: Add memory alarm detection | schema.py, test_schema.py |
| 2 | Header: Add system alarm indicator | run_header.py, test_run_header.py |
| 3 | Delete SystemResources + TrainingHealth | widgets/__init__.py, app.py |
| 4 | Update CSS for new layout | styles.tcss |
| 5 | Schema: Add DecisionSnapshot | schema.py, test_schema.py |
| 6 | Redesign TamiyoBrain | tamiyo_brain.py, test_tamiyo_brain.py |
| 7 | Wire decision capture in aggregator | aggregator.py, test_backend.py |
| 8 | Enhance EventLog | event_log.py, schema.py, test_event_log.py |
| 9 | Update aggregator for event log | aggregator.py |
| 10 | Integration test | test_app_integration.py |

**Total: 10 tasks, 3 phases**
