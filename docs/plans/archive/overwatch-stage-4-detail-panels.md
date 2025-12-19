# Overwatch Stage 4: Detail Panels

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the detail panel placeholder with toggleable Context Panel ("why flagged") and Tamiyo Detail Panel (full agent diagnostics), with proper empty states.

**Architecture:** DetailPanel is a Container that switches between two content modes: ContextPanel (shows selected env's anomaly reasons and slot details) and TamiyoDetailPanel (shows full action distribution, confidence sparkline, exploration bar). Mode switches via `c`/`t` keybindings. Empty states display when no data is available.

**Tech Stack:** Python 3.11, Textual widgets (Container, Static), Rich markup for formatting, CSS styling.

**Prerequisites:**
- Stage 3 complete (RunHeader, TamiyoStrip)
- Branch: `feat/overwatch-textual-ui`

---

## Task 1: Create ContextPanel Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/context_panel.py`
- Create: `tests/karn/overwatch/test_context_panel.py`

**Step 1: Write failing tests for ContextPanel**

```python
# tests/karn/overwatch/test_context_panel.py
"""Tests for ContextPanel widget."""

from __future__ import annotations

from esper.karn.overwatch.schema import (
    EnvSummary,
    SlotChipState,
)


class TestContextPanel:
    """Tests for ContextPanel widget."""

    def test_context_panel_imports(self) -> None:
        """ContextPanel can be imported."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        assert ContextPanel is not None

    def test_context_panel_renders_anomaly_reasons(self) -> None:
        """ContextPanel displays anomaly reasons as bullet list."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=3,
            device_id=1,
            status="WARN",
            anomaly_score=0.72,
            anomaly_reasons=["High gradient ratio (3.2x)", "Memory pressure (94%)"],
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "WHY FLAGGED" in content or "Why Flagged" in content
        assert "High gradient ratio" in content
        assert "Memory pressure" in content
        assert "•" in content or "-" in content  # Bullet points

    def test_context_panel_renders_env_header(self) -> None:
        """ContextPanel shows env ID and status."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=3,
            device_id=1,
            status="CRIT",
            anomaly_score=0.85,
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "Env 3" in content
        assert "CRIT" in content

    def test_context_panel_renders_slot_details(self) -> None:
        """ContextPanel shows selected slot details."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            slots={
                "r0c1": SlotChipState(
                    slot_id="r0c1",
                    stage="BLENDING",
                    blueprint_id="conv_light",
                    alpha=0.78,
                    gate_last="G2",
                    gate_passed=True,
                ),
            },
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "r0c1" in content
        assert "BLENDING" in content or "Blending" in content
        assert "conv_light" in content
        assert "0.78" in content or "78" in content
        assert "G2" in content

    def test_context_panel_renders_metrics(self) -> None:
        """ContextPanel shows env metrics."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            throughput_fps=98.5,
            reward_last=0.42,
            task_metric=0.823,
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "98" in content  # throughput
        assert "0.42" in content or "42" in content  # reward

    def test_context_panel_empty_state(self) -> None:
        """ContextPanel shows empty state when no env selected."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        panel = ContextPanel()

        content = panel.render_content()
        assert "Select" in content or "select" in content or "No env" in content

    def test_context_panel_no_anomaly_reasons(self) -> None:
        """ContextPanel handles env with no anomaly reasons."""
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        env = EnvSummary(
            env_id=0,
            device_id=0,
            status="OK",
            anomaly_score=0.1,
            anomaly_reasons=[],
        )
        panel = ContextPanel()
        panel.update_env(env)

        content = panel.render_content()
        assert "No issues" in content or "Healthy" in content or "OK" in content
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_context_panel.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement ContextPanel widget**

```python
# src/esper/karn/overwatch/widgets/context_panel.py
"""Context Panel Widget.

Displays detailed context for the selected environment:
- "Why Flagged" anomaly reasons
- Slot details (stage, blueprint, alpha, gate history)
- Environment metrics (throughput, reward, task metric)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import EnvSummary


class ContextPanel(Container):
    """Widget displaying context for the selected environment.

    Shows:
    - Environment header (ID, status, anomaly score)
    - "Why Flagged" bullet list of anomaly reasons
    - Slot details for each slot in the env
    - Environment metrics
    """

    DEFAULT_CSS = """
    ContextPanel {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }

    ContextPanel .section-header {
        text-style: bold;
        margin-top: 1;
    }

    ContextPanel .anomaly-reason {
        color: $warning;
    }

    ContextPanel .healthy {
        color: $success;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the context panel."""
        super().__init__(**kwargs)
        self._env: EnvSummary | None = None

    def render_content(self) -> str:
        """Render the panel content."""
        if self._env is None:
            return "[dim]Select an environment (j/k to navigate)[/dim]"

        lines = []
        env = self._env

        # Header
        status_color = self._status_color(env.status)
        lines.append(f"[bold]Env {env.env_id}[/bold] [{status_color}]{env.status}[/{status_color}]")
        lines.append(f"Device: GPU {env.device_id} | Anomaly: {env.anomaly_score:.2f}")
        lines.append("")

        # Why Flagged section
        lines.append("[bold]Why Flagged[/bold]")
        if env.anomaly_reasons:
            for reason in env.anomaly_reasons:
                lines.append(f"  [yellow]•[/yellow] {reason}")
        else:
            lines.append("  [green]No issues detected - Healthy[/green]")
        lines.append("")

        # Metrics section
        lines.append("[bold]Metrics[/bold]")
        lines.append(f"  Throughput: {env.throughput_fps:.1f} fps")
        lines.append(f"  Last Reward: {env.reward_last:.3f}")
        lines.append(f"  Task Metric: {env.task_metric:.3f}")
        lines.append("")

        # Slots section
        if env.slots:
            lines.append("[bold]Slots[/bold]")
            for slot_id, slot in sorted(env.slots.items()):
                gate_str = ""
                if slot.gate_last:
                    gate_icon = "✓" if slot.gate_passed else "✗"
                    gate_str = f" | {slot.gate_last}{gate_icon}"
                lines.append(
                    f"  [{slot.slot_id}] {slot.stage} | {slot.blueprint_id} | "
                    f"α={slot.alpha:.2f}{gate_str}"
                )

        return "\n".join(lines)

    def _status_color(self, status: str) -> str:
        """Get Rich color for status."""
        colors = {
            "OK": "green",
            "INFO": "blue",
            "WARN": "yellow",
            "CRIT": "red",
        }
        return colors.get(status, "white")

    def compose(self) -> ComposeResult:
        """Compose the panel layout."""
        yield Static(self.render_content(), id="context-content")

    def update_env(self, env: EnvSummary | None) -> None:
        """Update with selected environment."""
        self._env = env
        self._refresh_content()

    def _refresh_content(self) -> None:
        """Refresh the displayed content."""
        try:
            self.query_one("#context-content", Static).update(self.render_content())
        except Exception:
            pass  # Widget not mounted yet
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_context_panel.py -v`

Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/context_panel.py tests/karn/overwatch/test_context_panel.py
git commit -m "feat(overwatch): add ContextPanel widget for env details"
```

---

## Task 2: Create TamiyoDetailPanel Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/tamiyo_detail.py`
- Create: `tests/karn/overwatch/test_tamiyo_detail.py`

**Step 1: Write failing tests for TamiyoDetailPanel**

```python
# tests/karn/overwatch/test_tamiyo_detail.py
"""Tests for TamiyoDetailPanel widget."""

from __future__ import annotations

from esper.karn.overwatch.schema import TamiyoState


class TestTamiyoDetailPanel:
    """Tests for TamiyoDetailPanel widget."""

    def test_tamiyo_detail_imports(self) -> None:
        """TamiyoDetailPanel can be imported."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        assert TamiyoDetailPanel is not None

    def test_tamiyo_detail_renders_action_distribution(self) -> None:
        """TamiyoDetailPanel shows action distribution bars."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            action_counts={"GERMINATE": 34, "BLEND": 28, "CULL": 12, "WAIT": 26},
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "GERM" in content or "Germinate" in content
        assert "BLEND" in content or "Blend" in content
        assert "34%" in content or "34" in content
        # Should have visual bars
        assert "█" in content or "▓" in content or "=" in content

    def test_tamiyo_detail_renders_recent_actions_grid(self) -> None:
        """TamiyoDetailPanel shows recent actions in grid format."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            recent_actions=["G", "B", "B", "W", "G", "C", "W", "W", "B", "G"],
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "Recent" in content
        # Should show the action codes
        assert "G" in content
        assert "B" in content
        assert "W" in content

    def test_tamiyo_detail_renders_confidence(self) -> None:
        """TamiyoDetailPanel shows confidence metrics."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            confidence_mean=0.73,
            confidence_min=0.45,
            confidence_max=0.92,
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "Confidence" in content
        assert "73" in content  # mean
        assert "45" in content or "92" in content  # min or max

    def test_tamiyo_detail_renders_exploration(self) -> None:
        """TamiyoDetailPanel shows exploration percentage."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            exploration_pct=0.65,
            entropy=1.5,
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "Exploration" in content or "Entropy" in content
        assert "65" in content or "1.5" in content

    def test_tamiyo_detail_renders_learning_signals(self) -> None:
        """TamiyoDetailPanel shows PPO learning signals with health status."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            kl_divergence=0.015,
            explained_variance=0.75,
            clip_fraction=0.08,
            grad_norm=0.5,
            learning_rate=3e-4,
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        assert "KL" in content
        assert "0.015" in content or "015" in content
        assert "EV" in content or "Explained" in content
        assert "0.75" in content or "75" in content

    def test_tamiyo_detail_health_colors(self) -> None:
        """TamiyoDetailPanel applies health-based colors to signals."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        # Critical entropy (collapsed)
        tamiyo = TamiyoState(
            entropy=0.1,
            entropy_collapsed=True,
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        # Should have red color markup for critical entropy
        assert "red" in content or "CRIT" in content or "⚠" in content

    def test_tamiyo_detail_empty_state(self) -> None:
        """TamiyoDetailPanel shows empty state when no data."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        panel = TamiyoDetailPanel()

        content = panel.render_content()
        assert "Waiting" in content or "warmup" in content or "No data" in content

    def test_tamiyo_detail_sparkline_placeholder(self) -> None:
        """TamiyoDetailPanel shows confidence history as sparkline."""
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        tamiyo = TamiyoState(
            confidence_history=[0.5, 0.6, 0.7, 0.65, 0.8, 0.75, 0.9],
        )
        panel = TamiyoDetailPanel()
        panel.update_tamiyo(tamiyo)

        content = panel.render_content()
        # Should have some visual representation
        assert "▁" in content or "▂" in content or "▃" in content or "History" in content
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_tamiyo_detail.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement TamiyoDetailPanel widget**

```python
# src/esper/karn/overwatch/widgets/tamiyo_detail.py
"""Tamiyo Detail Panel Widget.

Displays comprehensive Tamiyo agent diagnostics:
- Full action distribution with visual bars
- Recent actions grid
- Confidence sparkline with min/max
- Exploration bar
- Learning signals with health indicators
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from esper.karn.overwatch.display_state import (
    kl_health,
    entropy_health,
    ev_health,
)

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TamiyoState


# Sparkline characters (8 levels)
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"

# Action display names
ACTION_NAMES = {
    "GERMINATE": "GERM",
    "BLEND": "BLEND",
    "CULL": "CULL",
    "WAIT": "WAIT",
    "ADVANCE": "ADV",
    "HOLD": "HOLD",
}


def sparkline(values: list[float], width: int = 20) -> str:
    """Generate a sparkline from values.

    Args:
        values: List of values to visualize
        width: Target width (will sample if needed)

    Returns:
        Unicode sparkline string
    """
    if not values:
        return "─" * width

    # Sample if too many values
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    # Normalize to 0-7 range
    min_v = min(values)
    max_v = max(values)
    range_v = max_v - min_v if max_v != min_v else 1

    chars = []
    for v in values:
        idx = int((v - min_v) / range_v * 7)
        idx = max(0, min(7, idx))
        chars.append(SPARKLINE_CHARS[idx])

    return "".join(chars)


def progress_bar(pct: float, width: int = 15) -> str:
    """Generate a progress bar.

    Args:
        pct: Percentage (0-100)
        width: Bar width in characters

    Returns:
        Progress bar string like "████████░░░░░░░"
    """
    filled = int(pct / 100 * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


class TamiyoDetailPanel(Container):
    """Widget displaying comprehensive Tamiyo agent diagnostics.

    Shows:
    - Action distribution with visual percentage bars
    - Recent actions grid (last 10-20 actions)
    - Confidence sparkline with min/max/mean
    - Exploration bar (entropy as % of max)
    - Learning signals (KL, EV, Clip) with health status
    """

    DEFAULT_CSS = """
    TamiyoDetailPanel {
        width: 100%;
        height: 100%;
        padding: 0 1;
        color: #c678dd;  /* Tamiyo magenta */
    }

    TamiyoDetailPanel .section-header {
        text-style: bold;
        margin-top: 1;
    }

    TamiyoDetailPanel .health-ok {
        color: $success;
    }

    TamiyoDetailPanel .health-warn {
        color: $warning;
    }

    TamiyoDetailPanel .health-crit {
        color: $error;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the Tamiyo detail panel."""
        super().__init__(**kwargs)
        self._tamiyo: TamiyoState | None = None

    def render_content(self) -> str:
        """Render the panel content."""
        if self._tamiyo is None:
            return "[dim]Waiting for Tamiyo data (warmup period)...[/dim]"

        lines = []
        t = self._tamiyo

        # Action Distribution section
        lines.append("[bold magenta]Action Distribution[/bold magenta]")
        total = sum(t.action_counts.values()) if t.action_counts else 1
        for action, count in sorted(t.action_counts.items(), key=lambda x: -x[1]):
            pct = (count / total) * 100 if total > 0 else 0
            name = ACTION_NAMES.get(action, action[:4].upper())
            bar = progress_bar(pct, width=12)
            lines.append(f"  {name:5} {bar} {pct:5.1f}%")
        lines.append("")

        # Recent Actions section
        lines.append("[bold magenta]Recent Actions[/bold magenta]")
        if t.recent_actions:
            # Display as grid with colored codes
            action_str = " ".join(f"[{self._action_color(a)}]{a}[/{self._action_color(a)}]"
                                  for a in t.recent_actions[-15:])
            lines.append(f"  {action_str}")
        else:
            lines.append("  [dim]No actions yet[/dim]")
        lines.append("")

        # Confidence section
        lines.append("[bold magenta]Confidence[/bold magenta]")
        lines.append(f"  Mean: {t.confidence_mean*100:.1f}%  "
                     f"Min: {t.confidence_min*100:.1f}%  "
                     f"Max: {t.confidence_max*100:.1f}%")
        if t.confidence_history:
            spark = sparkline(t.confidence_history)
            lines.append(f"  History: {spark}")
        lines.append("")

        # Exploration section
        lines.append("[bold magenta]Exploration[/bold magenta]")
        expl_bar = progress_bar(t.exploration_pct * 100, width=15)
        lines.append(f"  Entropy: {t.entropy:.3f}  [{expl_bar}] {t.exploration_pct*100:.0f}%")
        lines.append("")

        # Learning Signals section
        lines.append("[bold magenta]Learning Signals[/bold magenta]")

        # KL with health
        kl_h = kl_health(t.kl_divergence)
        kl_color = self._health_color(kl_h)
        lines.append(f"  KL Divergence: [{kl_color}]{t.kl_divergence:.4f}[/{kl_color}] ({kl_h.upper()})")

        # EV with health
        ev_h = ev_health(t.explained_variance)
        ev_color = self._health_color(ev_h)
        lines.append(f"  Explained Var: [{ev_color}]{t.explained_variance:.3f}[/{ev_color}] ({ev_h.upper()})")

        # Entropy with health
        ent_h = entropy_health(t.entropy)
        ent_color = self._health_color(ent_h)
        ent_warn = " ⚠ COLLAPSED" if t.entropy_collapsed else ""
        lines.append(f"  Entropy:       [{ent_color}]{t.entropy:.3f}[/{ent_color}] ({ent_h.upper()}){ent_warn}")

        # Other signals
        lines.append(f"  Clip Fraction: {t.clip_fraction:.3f}")
        lines.append(f"  Grad Norm:     {t.grad_norm:.3f}")
        lines.append(f"  Learning Rate: {t.learning_rate:.2e}")

        return "\n".join(lines)

    def _health_color(self, health: str) -> str:
        """Get Rich color for health level."""
        return {"ok": "green", "warn": "yellow", "crit": "red"}.get(health, "white")

    def _action_color(self, action: str) -> str:
        """Get color for action code."""
        colors = {
            "G": "green",      # Germinate
            "B": "magenta",    # Blend
            "C": "red",        # Cull
            "W": "dim",        # Wait
            "A": "blue",       # Advance
            "H": "yellow",     # Hold
        }
        return colors.get(action, "white")

    def compose(self) -> ComposeResult:
        """Compose the panel layout."""
        yield Static(self.render_content(), id="tamiyo-detail-content")

    def update_tamiyo(self, tamiyo: TamiyoState | None) -> None:
        """Update with Tamiyo state."""
        self._tamiyo = tamiyo
        self._refresh_content()

    def _refresh_content(self) -> None:
        """Refresh the displayed content."""
        try:
            self.query_one("#tamiyo-detail-content", Static).update(self.render_content())
        except Exception:
            pass  # Widget not mounted yet
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_tamiyo_detail.py -v`

Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/tamiyo_detail.py tests/karn/overwatch/test_tamiyo_detail.py
git commit -m "feat(overwatch): add TamiyoDetailPanel widget with full diagnostics"
```

---

## Task 3: Create DetailPanel Container

**Files:**
- Create: `src/esper/karn/overwatch/widgets/detail_panel.py`
- Create: `tests/karn/overwatch/test_detail_panel.py`

**Step 1: Write failing tests for DetailPanel**

```python
# tests/karn/overwatch/test_detail_panel.py
"""Tests for DetailPanel container widget."""

from __future__ import annotations

from esper.karn.overwatch.schema import (
    EnvSummary,
    TamiyoState,
)


class TestDetailPanel:
    """Tests for DetailPanel container."""

    def test_detail_panel_imports(self) -> None:
        """DetailPanel can be imported."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        assert DetailPanel is not None

    def test_detail_panel_starts_in_context_mode(self) -> None:
        """DetailPanel starts in context mode by default."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        assert panel.mode == "context"

    def test_detail_panel_switches_to_tamiyo_mode(self) -> None:
        """DetailPanel can switch to tamiyo mode."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        panel.set_mode("tamiyo")
        assert panel.mode == "tamiyo"

    def test_detail_panel_toggle_mode(self) -> None:
        """DetailPanel toggles between modes."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        assert panel.mode == "context"

        panel.toggle_mode("tamiyo")
        assert panel.mode == "tamiyo"

        panel.toggle_mode("tamiyo")  # Toggle same mode hides (back to context)
        assert panel.mode == "context"

        panel.toggle_mode("context")
        assert panel.mode == "context"

    def test_detail_panel_updates_env(self) -> None:
        """DetailPanel forwards env updates to context panel."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        env = EnvSummary(env_id=3, device_id=1, status="WARN")
        panel.update_env(env)

        assert panel._env == env

    def test_detail_panel_updates_tamiyo(self) -> None:
        """DetailPanel forwards tamiyo updates to tamiyo panel."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        tamiyo = TamiyoState(entropy=1.5)
        panel.update_tamiyo(tamiyo)

        assert panel._tamiyo == tamiyo

    def test_detail_panel_mode_property(self) -> None:
        """DetailPanel exposes current mode."""
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        panel = DetailPanel()
        assert panel.mode in ("context", "tamiyo")
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_detail_panel.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement DetailPanel container**

```python
# src/esper/karn/overwatch/widgets/detail_panel.py
"""Detail Panel Container Widget.

Switchable container that shows either:
- ContextPanel: Environment details and "why flagged"
- TamiyoDetailPanel: Full Tamiyo agent diagnostics

Toggle with 'c' for context, 't' for tamiyo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive

from esper.karn.overwatch.widgets.context_panel import ContextPanel
from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import EnvSummary, TamiyoState


PanelMode = Literal["context", "tamiyo"]


class DetailPanel(Container):
    """Switchable detail panel container.

    Displays either:
    - Context mode: Selected env details, anomaly reasons, slot info
    - Tamiyo mode: Full Tamiyo diagnostics, action distribution, confidence

    Toggle modes with set_mode() or toggle_mode().
    """

    DEFAULT_CSS = """
    DetailPanel {
        width: 100%;
        height: 100%;
    }

    DetailPanel > ContextPanel {
        display: block;
    }

    DetailPanel > TamiyoDetailPanel {
        display: none;
    }

    DetailPanel.tamiyo-mode > ContextPanel {
        display: none;
    }

    DetailPanel.tamiyo-mode > TamiyoDetailPanel {
        display: block;
    }
    """

    mode: reactive[PanelMode] = reactive("context")

    def __init__(self, **kwargs) -> None:
        """Initialize the detail panel."""
        super().__init__(**kwargs)
        self._env: EnvSummary | None = None
        self._tamiyo: TamiyoState | None = None

    def compose(self) -> ComposeResult:
        """Compose both panels (visibility controlled by CSS)."""
        yield ContextPanel(id="context-panel")
        yield TamiyoDetailPanel(id="tamiyo-panel")

    def set_mode(self, mode: PanelMode) -> None:
        """Set the panel mode.

        Args:
            mode: "context" or "tamiyo"
        """
        self.mode = mode
        self._update_mode_class()

    def toggle_mode(self, mode: PanelMode) -> None:
        """Toggle to a mode, or back to context if already in that mode.

        Args:
            mode: "context" or "tamiyo"
        """
        if self.mode == mode:
            self.mode = "context"
        else:
            self.mode = mode
        self._update_mode_class()

    def _update_mode_class(self) -> None:
        """Update CSS class based on current mode."""
        if self.mode == "tamiyo":
            self.add_class("tamiyo-mode")
        else:
            self.remove_class("tamiyo-mode")

    def update_env(self, env: EnvSummary | None) -> None:
        """Update the selected environment.

        Args:
            env: Environment summary or None
        """
        self._env = env
        try:
            self.query_one(ContextPanel).update_env(env)
        except Exception:
            pass  # Not mounted yet

    def update_tamiyo(self, tamiyo: TamiyoState | None) -> None:
        """Update the Tamiyo state.

        Args:
            tamiyo: Tamiyo state or None
        """
        self._tamiyo = tamiyo
        try:
            self.query_one(TamiyoDetailPanel).update_tamiyo(tamiyo)
        except Exception:
            pass  # Not mounted yet
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_detail_panel.py -v`

Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/detail_panel.py tests/karn/overwatch/test_detail_panel.py
git commit -m "feat(overwatch): add DetailPanel container with mode switching"
```

---

## Task 4: Update Widget Package Exports

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/__init__.py`
- Modify: `tests/karn/overwatch/test_widgets.py`

**Step 0: Fix unused import in test_widgets.py**

Remove line 5 (`import pytest`) from `tests/karn/overwatch/test_widgets.py` - it's unused and will fail linting.

**Step 1: Add test for new widget exports**

Append to `tests/karn/overwatch/test_widgets.py`:

```python
class TestStage4WidgetExports:
    """Tests for Stage 4 widget exports."""

    def test_context_panel_importable(self) -> None:
        """ContextPanel is importable from package."""
        from esper.karn.overwatch.widgets import ContextPanel

        assert ContextPanel is not None

    def test_tamiyo_detail_panel_importable(self) -> None:
        """TamiyoDetailPanel is importable from package."""
        from esper.karn.overwatch.widgets import TamiyoDetailPanel

        assert TamiyoDetailPanel is not None

    def test_detail_panel_importable(self) -> None:
        """DetailPanel is importable from package."""
        from esper.karn.overwatch.widgets import DetailPanel

        assert DetailPanel is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestStage4WidgetExports -v`

Expected: FAIL with `ImportError`

**Step 3: Update widgets __init__.py**

```python
# src/esper/karn/overwatch/widgets/__init__.py
"""Overwatch TUI Widgets.

Custom Textual widgets for the Overwatch monitoring interface.
"""

from esper.karn.overwatch.widgets.help import HelpOverlay
from esper.karn.overwatch.widgets.slot_chip import SlotChip
from esper.karn.overwatch.widgets.env_row import EnvRow
from esper.karn.overwatch.widgets.flight_board import FlightBoard
from esper.karn.overwatch.widgets.run_header import RunHeader
from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip
from esper.karn.overwatch.widgets.context_panel import ContextPanel
from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel
from esper.karn.overwatch.widgets.detail_panel import DetailPanel

__all__ = [
    "HelpOverlay",
    "SlotChip",
    "EnvRow",
    "FlightBoard",
    "RunHeader",
    "TamiyoStrip",
    "ContextPanel",
    "TamiyoDetailPanel",
    "DetailPanel",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestStage4WidgetExports -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/__init__.py tests/karn/overwatch/test_widgets.py
git commit -m "feat(overwatch): export Stage 4 detail panel widgets"
```

---

## Task 5: Wire DetailPanel into OverwatchApp

**Files:**
- Modify: `src/esper/karn/overwatch/app.py`

**Step 1: Update app.py to use DetailPanel with keybindings**

Replace the placeholder detail panel and add keybindings:

```python
# src/esper/karn/overwatch/app.py
"""Overwatch Textual Application.

Main application class for the Overwatch TUI monitoring interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header, Static

from esper.karn.overwatch.widgets.help import HelpOverlay
from esper.karn.overwatch.widgets.flight_board import FlightBoard
from esper.karn.overwatch.widgets.run_header import RunHeader
from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip
from esper.karn.overwatch.widgets.detail_panel import DetailPanel

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


class OverwatchApp(App):
    """Overwatch TUI for monitoring Esper training runs.

    Provides real-time visibility into:
    - Training environments (Flight Board)
    - Seed lifecycle and health
    - Tamiyo agent decisions
    - System resources

    Usage:
        app = OverwatchApp()
        app.run()

        # Or with replay file:
        app = OverwatchApp(replay_path="training.jsonl")
        app.run()
    """

    TITLE = "Esper Overwatch"
    SUB_TITLE = "Training Monitor"

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("question_mark", "toggle_help", "Help", show=True),
        Binding("escape", "dismiss", "Dismiss", show=False),
        Binding("c", "show_context", "Context", show=True),
        Binding("t", "show_tamiyo", "Tamiyo", show=True),
    ]

    def __init__(
        self,
        replay_path: Path | str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Overwatch app.

        Args:
            replay_path: Optional path to JSONL replay file
            **kwargs: Additional args passed to App
        """
        super().__init__(**kwargs)
        self._replay_path = Path(replay_path) if replay_path else None
        self._snapshot: TuiSnapshot | None = None
        self._help_visible = False

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()

        # Run header (run identity, connection status)
        # NOTE: Keep id="header" for backwards compatibility with existing integration tests
        yield RunHeader(id="header")

        # Tamiyo Strip (PPO vitals, action summary)
        yield TamiyoStrip(id="tamiyo-strip")

        # Main area with flight board and detail panel
        with Container(id="main-area"):
            yield FlightBoard(id="flight-board")
            yield DetailPanel(id="detail-panel")

        # Event feed
        yield Static(
            self._render_event_feed_content(),
            id="event-feed",
        )

        # Help overlay (hidden by default)
        yield HelpOverlay(id="help-overlay", classes="hidden")

        yield Footer()

    def _render_event_feed_content(self) -> str:
        """Render Event Feed placeholder content."""
        if self._snapshot and self._snapshot.event_feed:
            n = len(self._snapshot.event_feed)
            return f"[EVENT FEED] {n} events"
        return "[EVENT FEED] No events"

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # Load initial snapshot if replay file provided
        if self._replay_path:
            self._load_first_snapshot()

        # Set focus to flight board for navigation
        self.query_one(FlightBoard).focus()

    def _load_first_snapshot(self) -> None:
        """Load the first snapshot from replay file."""
        from esper.karn.overwatch.replay import SnapshotReader

        if not self._replay_path or not self._replay_path.exists():
            self.notify(f"Replay file not found: {self._replay_path}", severity="error")
            return

        reader = SnapshotReader(self._replay_path)
        for snapshot in reader:
            self._snapshot = snapshot
            break  # Take first snapshot only

        if self._snapshot:
            self.notify(f"Loaded snapshot from {self._snapshot.captured_at}")
            self._update_all_widgets()
        else:
            self.notify("No snapshots found in replay file", severity="warning")

    def _update_all_widgets(self) -> None:
        """Update all widgets with current snapshot."""
        if self._snapshot is None:
            return

        # Update run header
        self.query_one(RunHeader).update_snapshot(self._snapshot)

        # Update tamiyo strip
        self.query_one(TamiyoStrip).update_snapshot(self._snapshot)

        # Update flight board
        self.query_one(FlightBoard).update_snapshot(self._snapshot)

        # Update detail panel with tamiyo data
        detail_panel = self.query_one(DetailPanel)
        detail_panel.update_tamiyo(self._snapshot.tamiyo)

        # Update context panel with initial env selection
        board = self.query_one(FlightBoard)
        if board.selected_env_id is not None:
            for env in self._snapshot.flight_board:
                if env.env_id == board.selected_env_id:
                    detail_panel.update_env(env)
                    break

        # Update event feed placeholder
        self.query_one("#event-feed", Static).update(self._render_event_feed_content())

    def action_toggle_help(self) -> None:
        """Toggle the help overlay visibility."""
        help_overlay = self.query_one("#help-overlay")
        help_overlay.toggle_class("hidden")
        self._help_visible = not self._help_visible

    def action_dismiss(self) -> None:
        """Dismiss overlays or collapse expanded elements."""
        if self._help_visible:
            self.action_toggle_help()

    def action_show_context(self) -> None:
        """Toggle context panel view."""
        self.query_one(DetailPanel).toggle_mode("context")

    def action_show_tamiyo(self) -> None:
        """Toggle tamiyo detail panel view."""
        self.query_one(DetailPanel).toggle_mode("tamiyo")

    def on_flight_board_env_selected(self, message: FlightBoard.EnvSelected) -> None:
        """Handle env selection in flight board."""
        self._update_detail_panel_env(message.env_id)

    def on_flight_board_env_expanded(self, message: FlightBoard.EnvExpanded) -> None:
        """Handle env expansion in flight board."""
        pass  # Could update detail panel

    def _update_detail_panel_env(self, env_id: int | None) -> None:
        """Update detail panel with selected env info."""
        if env_id is None or self._snapshot is None:
            self.query_one(DetailPanel).update_env(None)
            return

        # Find the env
        env = None
        for e in self._snapshot.flight_board:
            if e.env_id == env_id:
                env = e
                break

        self.query_one(DetailPanel).update_env(env)
```

**Step 2: Run app tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_app.py -v`

Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/esper/karn/overwatch/app.py
git commit -m "feat(overwatch): wire DetailPanel into app with c/t keybindings"
```

---

## Task 6: Integration Tests for Detail Panels

**Files:**
- Modify: `tests/karn/overwatch/test_integration.py`

**Step 0: Fix existing test that will break**

The existing `test_detail_panel_updates_on_selection` test queries `#detail-panel` as a `Static`, but we're replacing it with `DetailPanel`. Update this test in `TestFlightBoardNavigation`:

```python
    @pytest.mark.asyncio
    async def test_detail_panel_updates_on_selection(self, multi_env_replay: Path) -> None:
        """Detail panel updates when env is selected via navigation."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.flight_board import FlightBoard
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        app = OverwatchApp(replay_path=multi_env_replay)

        async with app.run_test() as pilot:
            # Verify flight board has initial selection
            board = app.query_one(FlightBoard)
            assert board.selected_env_id == 2

            # Navigate to env 1
            await pilot.press("j")
            await pilot.pause()

            # Panel should update after navigation - query ContextPanel instead of Static
            context = app.query_one(ContextPanel)
            detail_text = context.render_content()
            assert "Env 1" in detail_text, f"Expected 'Env 1' in detail panel, got: {detail_text}"
```

**Step 1: Add integration tests**

Append to `tests/karn/overwatch/test_integration.py`:

```python
class TestDetailPanelIntegration:
    """Integration tests for DetailPanel functionality."""

    @pytest.fixture
    def detail_replay(self, tmp_path: Path) -> Path:
        """Create replay with detailed env and tamiyo data."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
            SlotChipState,
        )

        path = tmp_path / "detail.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T14:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(
                    action_counts={"GERMINATE": 34, "BLEND": 28, "CULL": 12, "WAIT": 26},
                    recent_actions=["G", "B", "W", "W", "C"],
                    confidence_mean=0.73,
                    confidence_min=0.45,
                    confidence_max=0.92,
                    confidence_history=[0.5, 0.6, 0.7, 0.65, 0.8],
                    exploration_pct=0.65,
                    kl_divergence=0.015,
                    entropy=1.5,
                    explained_variance=0.75,
                ),
                flight_board=[
                    EnvSummary(
                        env_id=3,
                        device_id=1,
                        status="WARN",
                        anomaly_score=0.72,
                        anomaly_reasons=["High gradient ratio (3.2x)", "Memory pressure (94%)"],
                        throughput_fps=102.5,
                        slots={
                            "r0c1": SlotChipState("r0c1", "BLENDING", "conv_light", 0.78, gate_last="G2", gate_passed=True),
                        },
                    ),
                    EnvSummary(env_id=0, device_id=0, status="OK", anomaly_score=0.1),
                ],
            )
            writer.write(snap)
        return path

    @pytest.mark.asyncio
    async def test_detail_panel_shows_context_by_default(self, detail_replay: Path) -> None:
        """DetailPanel starts in context mode."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            panel = app.query_one(DetailPanel)
            assert panel.mode == "context"

    @pytest.mark.asyncio
    async def test_t_key_switches_to_tamiyo_mode(self, detail_replay: Path) -> None:
        """Pressing 't' switches to tamiyo mode."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            panel = app.query_one(DetailPanel)
            assert panel.mode == "context"

            await pilot.press("t")
            assert panel.mode == "tamiyo"

    @pytest.mark.asyncio
    async def test_c_key_switches_to_context_mode(self, detail_replay: Path) -> None:
        """Pressing 'c' switches to context mode."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.detail_panel import DetailPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            panel = app.query_one(DetailPanel)

            # Switch to tamiyo first
            await pilot.press("t")
            assert panel.mode == "tamiyo"

            # Switch back to context
            await pilot.press("c")
            assert panel.mode == "context"

    @pytest.mark.asyncio
    async def test_context_panel_shows_anomaly_reasons(self, detail_replay: Path) -> None:
        """Context panel displays anomaly reasons for selected env."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.context_panel import ContextPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            # Initial selection should be highest anomaly (env 3)
            context = app.query_one(ContextPanel)
            content = context.render_content()

            assert "High gradient ratio" in content
            assert "Memory pressure" in content

    @pytest.mark.asyncio
    async def test_tamiyo_detail_shows_action_distribution(self, detail_replay: Path) -> None:
        """Tamiyo detail panel shows action distribution."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.tamiyo_detail import TamiyoDetailPanel

        app = OverwatchApp(replay_path=detail_replay)

        async with app.run_test() as pilot:
            # Switch to tamiyo mode
            await pilot.press("t")

            tamiyo = app.query_one(TamiyoDetailPanel)
            content = tamiyo.render_content()

            assert "GERM" in content or "Germinate" in content
            assert "34" in content  # 34%
```

**Step 2: Run integration tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_integration.py::TestDetailPanelIntegration -v`

Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add tests/karn/overwatch/test_integration.py
git commit -m "test(overwatch): add integration tests for DetailPanel"
```

---

## Task 7: Run Full Test Suite

**Step 1: Run all Overwatch tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/ -v`

Expected: All tests PASS (150+ tests)

**Step 2: Run linting**

Run: `uv run ruff check src/esper/karn/overwatch/`

Expected: No errors

**Step 3: Final commit if needed**

```bash
git status
# If any uncommitted files:
git add -A
git commit -m "chore(overwatch): Stage 4 complete - Detail Panels"
```

---

## Verification Checklist

- [ ] ContextPanel displays "Why Flagged" with anomaly reasons
- [ ] ContextPanel shows env header (ID, status, anomaly score)
- [ ] ContextPanel shows slot details (stage, blueprint, alpha, gate)
- [ ] ContextPanel shows env metrics (throughput, reward)
- [ ] ContextPanel handles empty state (no env selected)
- [ ] TamiyoDetailPanel shows action distribution with bars
- [ ] TamiyoDetailPanel shows recent actions grid
- [ ] TamiyoDetailPanel shows confidence (mean/min/max/sparkline)
- [ ] TamiyoDetailPanel shows exploration bar
- [ ] TamiyoDetailPanel shows learning signals with health colors
- [ ] TamiyoDetailPanel handles empty state (warmup)
- [ ] `c` key toggles context panel
- [ ] `t` key toggles tamiyo panel
- [ ] Pressing same key twice returns to default mode
- [ ] All tests pass (150+)
- [ ] Linting passes

---

## Files Created/Modified

```
src/esper/karn/overwatch/
├── app.py                    # Modified: wire DetailPanel, add keybindings
└── widgets/
    ├── __init__.py           # Modified: export new widgets
    ├── context_panel.py      # NEW: env context and anomaly display
    ├── tamiyo_detail.py      # NEW: full Tamiyo diagnostics
    └── detail_panel.py       # NEW: switchable container

tests/karn/overwatch/
├── test_context_panel.py     # NEW: ContextPanel tests
├── test_tamiyo_detail.py     # NEW: TamiyoDetailPanel tests
├── test_detail_panel.py      # NEW: DetailPanel tests
├── test_widgets.py           # Modified: add export tests
└── test_integration.py       # Modified: add integration tests
```

---

## Next Stage

After Stage 4 is merged, proceed to **Stage 5: Event Feed + Replay** which will:
- Add EventFeed widget with scrollable log
- Implement event type badges with colors
- Add replay controls (play/pause, step, speed)
- Add event filtering
