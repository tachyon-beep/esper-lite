# Overwatch Stage 3: Header + Tamiyo Strip

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace placeholder Header and Tamiyo Strip with real widgets showing run identity, connection status, PPO vitals, and health indicators with trend arrows.

**Architecture:** Two-widget approach: `RunHeader` (2 rows) shows run identity, connection status, and env health counts; `TamiyoStrip` (2 rows) shows PPO vitals with trend arrows and action distribution. Both widgets update reactively from TuiSnapshot data.

**Tech Stack:** Python 3.11, Textual widgets (Static, Container), reactive properties, CSS styling, Unicode trend arrows (↑↓→).

**Prerequisites:**
- Stage 2 complete (FlightBoard, EnvRow, SlotChip)
- Branch: `feat/overwatch-textual-ui`

---

## Task 1: Create Health Indicator Logic

**Files:**
- Modify: `src/esper/karn/overwatch/display_state.py`
- Create: `tests/karn/overwatch/test_health_indicators.py`

**Step 1: Write failing tests for health indicators**

```python
# tests/karn/overwatch/test_health_indicators.py
"""Tests for health indicator logic."""

from __future__ import annotations

import pytest


class TestTrendArrow:
    """Tests for trend arrow rendering."""

    def test_trend_arrow_rising(self) -> None:
        """Positive trend shows up arrow."""
        from esper.karn.overwatch.display_state import trend_arrow

        assert trend_arrow(0.05) == "↑"

    def test_trend_arrow_falling(self) -> None:
        """Negative trend shows down arrow."""
        from esper.karn.overwatch.display_state import trend_arrow

        assert trend_arrow(-0.05) == "↓"

    def test_trend_arrow_stable(self) -> None:
        """Near-zero trend shows right arrow (stable)."""
        from esper.karn.overwatch.display_state import trend_arrow

        assert trend_arrow(0.001) == "→"
        assert trend_arrow(-0.001) == "→"
        assert trend_arrow(0.0) == "→"

    def test_trend_arrow_custom_threshold(self) -> None:
        """Custom threshold changes sensitivity."""
        from esper.karn.overwatch.display_state import trend_arrow

        # With threshold=0.1, a trend of 0.05 is stable
        assert trend_arrow(0.05, threshold=0.1) == "→"
        # With threshold=0.01, a trend of 0.05 is rising
        assert trend_arrow(0.05, threshold=0.01) == "↑"


class TestHealthLevel:
    """Tests for health level classification."""

    def test_kl_health_ok(self) -> None:
        """Low KL divergence is healthy."""
        from esper.karn.overwatch.display_state import kl_health

        assert kl_health(0.01) == "ok"
        assert kl_health(0.02) == "ok"

    def test_kl_health_warn(self) -> None:
        """Medium KL divergence is warning."""
        from esper.karn.overwatch.display_state import kl_health

        assert kl_health(0.03) == "warn"
        assert kl_health(0.04) == "warn"

    def test_kl_health_crit(self) -> None:
        """High KL divergence is critical."""
        from esper.karn.overwatch.display_state import kl_health

        assert kl_health(0.05) == "crit"
        assert kl_health(0.1) == "crit"

    def test_entropy_health_ok(self) -> None:
        """Good entropy range is healthy."""
        from esper.karn.overwatch.display_state import entropy_health

        assert entropy_health(1.5) == "ok"
        assert entropy_health(2.0) == "ok"

    def test_entropy_health_warn_low(self) -> None:
        """Low entropy (collapsed) is warning."""
        from esper.karn.overwatch.display_state import entropy_health

        assert entropy_health(0.3) == "warn"

    def test_entropy_health_crit_collapsed(self) -> None:
        """Very low entropy is critical (policy collapsed)."""
        from esper.karn.overwatch.display_state import entropy_health

        assert entropy_health(0.1) == "crit"

    def test_ev_health_ok(self) -> None:
        """High explained variance is healthy."""
        from esper.karn.overwatch.display_state import ev_health

        assert ev_health(0.8) == "ok"
        assert ev_health(0.95) == "ok"

    def test_ev_health_warn(self) -> None:
        """Medium explained variance is warning."""
        from esper.karn.overwatch.display_state import ev_health

        assert ev_health(0.5) == "warn"

    def test_ev_health_crit(self) -> None:
        """Low explained variance is critical."""
        from esper.karn.overwatch.display_state import ev_health

        assert ev_health(0.2) == "crit"
        assert ev_health(-0.1) == "crit"


class TestFormatRuntime:
    """Tests for runtime formatting."""

    def test_format_runtime_seconds(self) -> None:
        """Short runtime shows seconds."""
        from esper.karn.overwatch.display_state import format_runtime

        assert format_runtime(45.0) == "45s"

    def test_format_runtime_minutes(self) -> None:
        """Minutes shown with seconds."""
        from esper.karn.overwatch.display_state import format_runtime

        assert format_runtime(125.0) == "2m 5s"

    def test_format_runtime_hours(self) -> None:
        """Hours shown with minutes."""
        from esper.karn.overwatch.display_state import format_runtime

        assert format_runtime(3725.0) == "1h 2m"

    def test_format_runtime_zero(self) -> None:
        """Zero runtime shows 0s."""
        from esper.karn.overwatch.display_state import format_runtime

        assert format_runtime(0.0) == "0s"
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_health_indicators.py -v`

Expected: FAIL with `ImportError` (functions not defined)

**Step 3: Implement health indicator functions**

Add to `src/esper/karn/overwatch/display_state.py`:

```python
# Add these functions after the existing classes

def trend_arrow(value: float, threshold: float = 0.01) -> str:
    """Convert trend value to arrow character.

    Args:
        value: Trend value (positive = rising, negative = falling)
        threshold: Minimum absolute value to show direction

    Returns:
        Arrow character: ↑ (rising), ↓ (falling), → (stable)
    """
    if value > threshold:
        return "↑"
    elif value < -threshold:
        return "↓"
    return "→"


def kl_health(kl: float) -> str:
    """Classify KL divergence health level.

    PPO target KL is typically 0.01-0.02. Higher values indicate
    the policy is changing too fast.

    Args:
        kl: KL divergence value

    Returns:
        Health level: "ok", "warn", or "crit"
    """
    if kl < 0.025:
        return "ok"
    elif kl < 0.05:
        return "warn"
    return "crit"


def entropy_health(entropy: float) -> str:
    """Classify entropy health level.

    Entropy measures exploration. Very low entropy indicates
    the policy has collapsed to deterministic actions.

    Args:
        entropy: Policy entropy value

    Returns:
        Health level: "ok", "warn", or "crit"
    """
    if entropy < 0.2:
        return "crit"  # Policy collapsed
    elif entropy < 0.5:
        return "warn"
    return "ok"


def ev_health(ev: float) -> str:
    """Classify explained variance health level.

    Explained variance measures how well the value function
    predicts returns. Values near 1.0 are good.

    Args:
        ev: Explained variance (-inf to 1.0)

    Returns:
        Health level: "ok", "warn", or "crit"
    """
    if ev < 0.3:
        return "crit"
    elif ev < 0.6:
        return "warn"
    return "ok"


def format_runtime(seconds: float) -> str:
    """Format runtime as human-readable string.

    Args:
        seconds: Runtime in seconds

    Returns:
        Formatted string like "1h 2m", "5m 30s", or "45s"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_health_indicators.py -v`

Expected: All 15 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/display_state.py tests/karn/overwatch/test_health_indicators.py
git commit -m "feat(overwatch): add health indicator and trend arrow logic"
```

---

## Task 2: Create RunHeader Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/run_header.py`
- Create: `tests/karn/overwatch/test_run_header.py`

**Step 1: Write failing tests for RunHeader**

```python
# tests/karn/overwatch/test_run_header.py
"""Tests for RunHeader widget."""

from __future__ import annotations

import pytest

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    ConnectionStatus,
    TamiyoState,
)


@pytest.fixture
def sample_snapshot() -> TuiSnapshot:
    """Create a sample snapshot with run info."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T14:00:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(),
        run_id="exp-001",
        task_name="cifar10",
        episode=5,
        batch=150,
        best_metric=0.823,
        runtime_s=3725.0,
        envs_ok=3,
        envs_warn=1,
        envs_crit=0,
    )


class TestRunHeader:
    """Tests for RunHeader widget."""

    def test_run_header_imports(self) -> None:
        """RunHeader can be imported."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        assert RunHeader is not None

    def test_run_header_renders_run_id(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader displays run ID."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line1()
        assert "exp-001" in content

    def test_run_header_renders_task(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader displays task name."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line1()
        assert "cifar10" in content

    def test_run_header_renders_episode(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader displays episode number."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line1()
        assert "5" in content or "Ep 5" in content or "ep:5" in content.lower()

    def test_run_header_renders_runtime(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader displays formatted runtime."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line1()
        assert "1h" in content  # 3725s = 1h 2m

    def test_run_header_renders_connection_live(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader shows Live when connected with low staleness."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line2()
        assert "Live" in content or "●" in content

    def test_run_header_renders_connection_stale(self) -> None:
        """RunHeader shows Stale when staleness is high."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        snapshot = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T14:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 10.0),  # 10s stale
            tamiyo=TamiyoState(),
        )
        header = RunHeader()
        header.update_snapshot(snapshot)

        content = header.render_line2()
        assert "Stale" in content or "10" in content

    def test_run_header_renders_env_counts(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader shows environment health counts."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line2()
        # Should show OK:3 WARN:1 CRIT:0 or similar
        assert "3" in content  # 3 OK
        assert "1" in content  # 1 WARN

    def test_run_header_renders_best_metric(self, sample_snapshot: TuiSnapshot) -> None:
        """RunHeader shows best metric achieved."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()
        header.update_snapshot(sample_snapshot)

        content = header.render_line2()
        assert "82" in content or "0.82" in content  # best_metric=0.823

    def test_run_header_empty_state(self) -> None:
        """RunHeader handles no snapshot gracefully."""
        from esper.karn.overwatch.widgets.run_header import RunHeader

        header = RunHeader()

        content = header.render_line1()
        assert "Waiting" in content or "No data" in content or "--" in content
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_run_header.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement RunHeader widget**

```python
# src/esper/karn/overwatch/widgets/run_header.py
"""Run Header Widget.

Displays run identity, connection status, and environment health summary
in two rows at the top of the Overwatch TUI.

Row 1: Run ID | Task | Episode | Batch | Runtime
Row 2: Connection | Best Metric | Env Counts (OK/WARN/CRIT)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from esper.karn.overwatch.display_state import format_runtime

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


class RunHeader(Container):
    """Widget displaying run identity and connection status.

    Two-line format:
        exp-001 | cifar10 | Ep 5 | Batch 150 | 1h 2m
        ● Live | Best: 82.3% | OK:3 WARN:1 CRIT:0
    """

    DEFAULT_CSS = """
    RunHeader {
        width: 100%;
        height: 2;
        padding: 0 1;
        background: $surface;
    }

    RunHeader .header-line {
        width: 100%;
        height: 1;
    }

    RunHeader .connection-live {
        color: $success;
    }

    RunHeader .connection-stale {
        color: $warning;
    }

    RunHeader .connection-disconnected {
        color: $error;
    }

    RunHeader .env-ok {
        color: $success;
    }

    RunHeader .env-warn {
        color: $warning;
    }

    RunHeader .env-crit {
        color: $error;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the run header."""
        super().__init__(**kwargs)
        self._snapshot: TuiSnapshot | None = None

    def render_line1(self) -> str:
        """Render first line: run identity."""
        if self._snapshot is None:
            return "-- | Waiting for data..."

        s = self._snapshot
        run_id = s.run_id or "--"
        task = s.task_name or "--"
        runtime = format_runtime(s.runtime_s)

        return f"{run_id} | {task} | Ep {s.episode} | Batch {s.batch} | {runtime}"

    def render_line2(self) -> str:
        """Render second line: connection and health summary."""
        if self._snapshot is None:
            return "○ Disconnected | -- | --"

        s = self._snapshot
        c = s.connection

        # Connection indicator
        if not c.connected:
            conn = "○ Disconnected"
        elif c.staleness_s < 2.0:
            conn = "● Live"
        elif c.staleness_s < 5.0:
            conn = f"● Live ({c.staleness_s:.0f}s)"
        else:
            conn = f"◐ Stale ({c.staleness_s:.0f}s)"

        # Best metric
        best = f"Best: {s.best_metric*100:.1f}%" if s.best_metric > 0 else "Best: --"

        # Env counts
        envs = f"OK:{s.envs_ok} WARN:{s.envs_warn} CRIT:{s.envs_crit}"

        return f"{conn} | {best} | {envs}"

    def compose(self) -> ComposeResult:
        """Compose the header layout."""
        yield Static(self.render_line1(), classes="header-line", id="header-line1")
        yield Static(self.render_line2(), classes="header-line", id="header-line2")

    def update_snapshot(self, snapshot: TuiSnapshot) -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self._refresh_content()

    def _refresh_content(self) -> None:
        """Refresh the displayed content."""
        try:
            self.query_one("#header-line1", Static).update(self.render_line1())
            self.query_one("#header-line2", Static).update(self.render_line2())
        except Exception:
            # Widget not mounted yet
            pass
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_run_header.py -v`

Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/run_header.py tests/karn/overwatch/test_run_header.py
git commit -m "feat(overwatch): add RunHeader widget"
```

---

## Task 3: Create TamiyoStrip Widget

**Files:**
- Create: `src/esper/karn/overwatch/widgets/tamiyo_strip.py`
- Create: `tests/karn/overwatch/test_tamiyo_strip.py`

**Step 1: Write failing tests for TamiyoStrip**

```python
# tests/karn/overwatch/test_tamiyo_strip.py
"""Tests for TamiyoStrip widget."""

from __future__ import annotations

import pytest

from esper.karn.overwatch.schema import (
    TuiSnapshot,
    ConnectionStatus,
    TamiyoState,
)


@pytest.fixture
def tamiyo_snapshot() -> TuiSnapshot:
    """Create a snapshot with Tamiyo data."""
    return TuiSnapshot(
        schema_version=1,
        captured_at="2025-12-18T14:00:00Z",
        connection=ConnectionStatus(True, 1000.0, 0.5),
        tamiyo=TamiyoState(
            kl_divergence=0.015,
            entropy=1.5,
            explained_variance=0.75,
            clip_fraction=0.08,
            grad_norm=0.5,
            learning_rate=3e-4,
            kl_trend=0.002,
            entropy_trend=-0.05,
            ev_trend=0.01,
            entropy_collapsed=False,
            ev_warning=False,
            action_counts={"GERMINATE": 10, "BLEND": 20, "CULL": 5, "WAIT": 65},
            recent_actions=["G", "B", "W", "W", "C"],
        ),
    )


class TestTamiyoStrip:
    """Tests for TamiyoStrip widget."""

    def test_tamiyo_strip_imports(self) -> None:
        """TamiyoStrip can be imported."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        assert TamiyoStrip is not None

    def test_tamiyo_strip_renders_kl(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip displays KL divergence."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_vitals()
        assert "KL" in content
        assert "0.015" in content or "0.02" in content

    def test_tamiyo_strip_renders_entropy(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip displays entropy."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_vitals()
        assert "Ent" in content or "H" in content
        assert "1.5" in content

    def test_tamiyo_strip_renders_ev(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip displays explained variance."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_vitals()
        assert "EV" in content
        assert "0.75" in content or "75" in content

    def test_tamiyo_strip_renders_trend_arrows(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip shows trend arrows."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_vitals()
        # kl_trend=0.002 (stable →), entropy_trend=-0.05 (↓), ev_trend=0.01 (stable/↑)
        assert "↓" in content  # entropy falling

    def test_tamiyo_strip_renders_action_counts(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip shows action distribution."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_actions()
        # Should show action names and counts/percentages
        assert "G" in content or "GERM" in content
        assert "B" in content or "BLEND" in content

    def test_tamiyo_strip_renders_recent_actions(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip shows recent action sequence."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        content = strip.render_actions()
        # recent_actions = ["G", "B", "W", "W", "C"]
        assert "G" in content
        assert "W" in content

    def test_tamiyo_strip_health_coloring(self, tamiyo_snapshot: TuiSnapshot) -> None:
        """TamiyoStrip applies health-based CSS classes."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()
        strip.update_snapshot(tamiyo_snapshot)

        health = strip.get_vitals_health()
        # kl=0.015 is ok, entropy=1.5 is ok, ev=0.75 is ok
        assert health["kl"] == "ok"
        assert health["entropy"] == "ok"
        assert health["ev"] == "ok"

    def test_tamiyo_strip_empty_state(self) -> None:
        """TamiyoStrip handles no snapshot gracefully."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        strip = TamiyoStrip()

        content = strip.render_vitals()
        assert "Waiting" in content or "--" in content

    def test_tamiyo_strip_entropy_collapsed_warning(self) -> None:
        """TamiyoStrip shows warning when entropy is collapsed."""
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        snapshot = TuiSnapshot(
            schema_version=1,
            captured_at="2025-12-18T14:00:00Z",
            connection=ConnectionStatus(True, 1000.0, 0.5),
            tamiyo=TamiyoState(
                entropy=0.1,
                entropy_collapsed=True,
            ),
        )
        strip = TamiyoStrip()
        strip.update_snapshot(snapshot)

        health = strip.get_vitals_health()
        assert health["entropy"] == "crit"
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_tamiyo_strip.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement TamiyoStrip widget**

```python
# src/esper/karn/overwatch/widgets/tamiyo_strip.py
"""Tamiyo Strip Widget.

Displays PPO vitals, trend arrows, and action distribution in two rows
below the header. Color-coded health indicators show policy status.

Row 1: KL↑ 0.015 | Ent↓ 1.5 | EV→ 0.75 | Clip 8% | ∇ 0.5
Row 2: Actions: G:10% B:20% C:5% W:65% | Recent: GBWWC
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static

from esper.karn.overwatch.display_state import (
    trend_arrow,
    kl_health,
    entropy_health,
    ev_health,
)

if TYPE_CHECKING:
    from esper.karn.overwatch.schema import TuiSnapshot


# Action code to display name mapping
ACTION_CODES = {
    "GERMINATE": "G",
    "BLEND": "B",
    "CULL": "C",
    "WAIT": "W",
    "ADVANCE": "A",
    "HOLD": "H",
}


class TamiyoStrip(Container):
    """Widget displaying Tamiyo PPO vitals and action distribution.

    Two-line format with health-colored indicators:
        KL↑ 0.015 | Ent↓ 1.5 | EV→ 0.75 | Clip 8% | ∇ 0.5
        Actions: G:10% B:20% C:5% W:65% | Recent: GBWWC
    """

    DEFAULT_CSS = """
    TamiyoStrip {
        width: 100%;
        height: 2;
        padding: 0 1;
        background: $surface;
        color: #c678dd;  /* Tamiyo magenta */
    }

    TamiyoStrip .strip-line {
        width: 100%;
        height: 1;
    }

    TamiyoStrip .health-ok {
        color: $success;
    }

    TamiyoStrip .health-warn {
        color: $warning;
    }

    TamiyoStrip .health-crit {
        color: $error;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the Tamiyo strip."""
        super().__init__(**kwargs)
        self._snapshot: TuiSnapshot | None = None

    def render_vitals(self) -> str:
        """Render vitals line with trend arrows."""
        if self._snapshot is None or self._snapshot.tamiyo is None:
            return "-- | Waiting for policy data..."

        t = self._snapshot.tamiyo

        # Format each metric with trend arrow
        kl_arrow = trend_arrow(t.kl_trend)
        ent_arrow = trend_arrow(t.entropy_trend)
        ev_arrow = trend_arrow(t.ev_trend)

        kl_str = f"KL{kl_arrow} {t.kl_divergence:.3f}"
        ent_str = f"Ent{ent_arrow} {t.entropy:.2f}"
        ev_str = f"EV{ev_arrow} {t.explained_variance:.2f}"
        clip_str = f"Clip {t.clip_fraction*100:.0f}%"
        grad_str = f"∇ {t.grad_norm:.2f}"

        return f"{kl_str} | {ent_str} | {ev_str} | {clip_str} | {grad_str}"

    def render_actions(self) -> str:
        """Render action distribution and recent actions."""
        if self._snapshot is None or self._snapshot.tamiyo is None:
            return "Actions: -- | Recent: --"

        t = self._snapshot.tamiyo

        # Action distribution as percentages
        total = sum(t.action_counts.values()) if t.action_counts else 1
        parts = []
        for action, count in sorted(t.action_counts.items()):
            code = ACTION_CODES.get(action, action[0])
            pct = (count / total) * 100 if total > 0 else 0
            parts.append(f"{code}:{pct:.0f}%")

        actions_str = " ".join(parts) if parts else "--"

        # Recent actions as compact string
        recent = "".join(t.recent_actions) if t.recent_actions else "--"

        return f"Actions: {actions_str} | Recent: {recent}"

    def get_vitals_health(self) -> dict[str, str]:
        """Get health levels for each vital metric.

        Returns:
            Dict with keys 'kl', 'entropy', 'ev' and values 'ok', 'warn', 'crit'
        """
        if self._snapshot is None or self._snapshot.tamiyo is None:
            return {"kl": "ok", "entropy": "ok", "ev": "ok"}

        t = self._snapshot.tamiyo
        return {
            "kl": kl_health(t.kl_divergence),
            "entropy": entropy_health(t.entropy),
            "ev": ev_health(t.explained_variance),
        }

    def compose(self) -> ComposeResult:
        """Compose the strip layout."""
        yield Static(self.render_vitals(), classes="strip-line", id="tamiyo-vitals")
        yield Static(self.render_actions(), classes="strip-line", id="tamiyo-actions")

    def update_snapshot(self, snapshot: TuiSnapshot) -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        self._refresh_content()

    def _refresh_content(self) -> None:
        """Refresh the displayed content."""
        try:
            self.query_one("#tamiyo-vitals", Static).update(self.render_vitals())
            self.query_one("#tamiyo-actions", Static).update(self.render_actions())
        except Exception:
            # Widget not mounted yet
            pass
```

**Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_tamiyo_strip.py -v`

Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/tamiyo_strip.py tests/karn/overwatch/test_tamiyo_strip.py
git commit -m "feat(overwatch): add TamiyoStrip widget with PPO vitals"
```

---

## Task 4: Update Widget Package Exports

**Files:**
- Modify: `src/esper/karn/overwatch/widgets/__init__.py`
- Modify: `tests/karn/overwatch/test_widgets.py`

**Step 1: Add test for new widget exports**

Append to `tests/karn/overwatch/test_widgets.py`:

```python
class TestStage3WidgetExports:
    """Tests for Stage 3 widget exports."""

    def test_run_header_importable(self) -> None:
        """RunHeader is importable from package."""
        from esper.karn.overwatch.widgets import RunHeader

        assert RunHeader is not None

    def test_tamiyo_strip_importable(self) -> None:
        """TamiyoStrip is importable from package."""
        from esper.karn.overwatch.widgets import TamiyoStrip

        assert TamiyoStrip is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestStage3WidgetExports -v`

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

__all__ = [
    "HelpOverlay",
    "SlotChip",
    "EnvRow",
    "FlightBoard",
    "RunHeader",
    "TamiyoStrip",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_widgets.py::TestStage3WidgetExports -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/overwatch/widgets/__init__.py tests/karn/overwatch/test_widgets.py
git commit -m "feat(overwatch): export RunHeader and TamiyoStrip from package"
```

---

## Task 5: Wire Widgets into OverwatchApp

**Files:**
- Modify: `src/esper/karn/overwatch/app.py`

**Step 1: Update app.py to use real widgets**

Replace the placeholder Static widgets with RunHeader and TamiyoStrip:

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

            yield Static(
                self._render_detail_panel_content(),
                id="detail-panel",
            )

        # Event feed
        yield Static(
            self._render_event_feed_content(),
            id="event-feed",
        )

        # Help overlay (hidden by default)
        yield HelpOverlay(id="help-overlay", classes="hidden")

        yield Footer()

    def _render_detail_panel_content(self) -> str:
        """Render Detail Panel placeholder content."""
        return "[DETAIL PANEL] Select an environment (j/k to navigate)"

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

        # Update placeholders
        self.query_one("#detail-panel", Static).update(self._render_detail_panel_content())
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

    def on_flight_board_env_selected(self, message: FlightBoard.EnvSelected) -> None:
        """Handle env selection in flight board."""
        self._update_detail_panel(message.env_id)

    def on_flight_board_env_expanded(self, message: FlightBoard.EnvExpanded) -> None:
        """Handle env expansion in flight board."""
        pass  # Could update detail panel

    def _update_detail_panel(self, env_id: int | None) -> None:
        """Update detail panel with selected env info."""
        if env_id is None or self._snapshot is None:
            self.query_one("#detail-panel", Static).update(
                "[DETAIL PANEL] Select an environment"
            )
            return

        # Find the env
        env = None
        for e in self._snapshot.flight_board:
            if e.env_id == env_id:
                env = e
                break

        if env is None:
            return

        # Build detail content
        lines = [f"[DETAIL] Env {env.env_id}"]
        lines.append(f"Status: {env.status}")
        lines.append(f"Anomaly: {env.anomaly_score:.2f}")

        if env.anomaly_reasons:
            lines.append("Reasons:")
            for reason in env.anomaly_reasons:
                lines.append(f"  - {reason}")

        self.query_one("#detail-panel", Static).update("\n".join(lines))
```

**Step 2: Verify existing CSS applies**

The existing `styles.tcss` already has `#header` and `#tamiyo-strip` ID selectors that will
apply to our new widgets (since we use `id="header"` and `id="tamiyo-strip"`):

```css
/* Already exists - no changes needed to styles.tcss */
#header {
    background: $surface;
    border-bottom: solid $primary-darken-2;
    padding: 0 1;
}

#tamiyo-strip {
    background: $surface;
    color: $text;
    border-bottom: solid $secondary;
    padding: 0 1;
}
```

The widgets' `DEFAULT_CSS` handles layout (height: 2), while `styles.tcss` handles theming.
This separation keeps widget internals encapsulated while allowing global theme overrides.

**No changes to styles.tcss required for this step.**

**Step 3: Run app tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_app.py -v`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/esper/karn/overwatch/app.py
git commit -m "feat(overwatch): wire RunHeader and TamiyoStrip into app"
```

---

## Task 6: Integration Tests for Header and Strip

**Files:**
- Modify: `tests/karn/overwatch/test_integration.py`

**Step 1: Add integration tests**

Append to `tests/karn/overwatch/test_integration.py`:

```python
class TestHeaderAndStripIntegration:
    """Integration tests for RunHeader and TamiyoStrip."""

    @pytest.fixture
    def tamiyo_replay(self, tmp_path: Path) -> Path:
        """Create replay with Tamiyo data."""
        from esper.karn.overwatch import (
            SnapshotWriter,
            TuiSnapshot,
            ConnectionStatus,
            TamiyoState,
            EnvSummary,
        )

        path = tmp_path / "tamiyo.jsonl"
        with SnapshotWriter(path) as writer:
            snap = TuiSnapshot(
                schema_version=1,
                captured_at="2025-12-18T14:00:00Z",
                connection=ConnectionStatus(True, 1000.0, 0.5),
                tamiyo=TamiyoState(
                    kl_divergence=0.015,
                    entropy=1.5,
                    explained_variance=0.75,
                    kl_trend=0.002,
                    entropy_trend=-0.05,
                    ev_trend=0.01,
                    action_counts={"GERMINATE": 10, "WAIT": 90},
                ),
                run_id="test-001",
                task_name="cifar10",
                episode=5,
                batch=100,
                runtime_s=600.0,
                envs_ok=2,
                envs_warn=1,
                envs_crit=0,
                flight_board=[
                    EnvSummary(env_id=0, device_id=0, status="OK", anomaly_score=0.1),
                    EnvSummary(env_id=1, device_id=0, status="WARN", anomaly_score=0.6),
                ],
            )
            writer.write(snap)
        return path

    @pytest.mark.asyncio
    async def test_run_header_displays_data(self, tamiyo_replay: Path) -> None:
        """RunHeader shows run identity when loaded."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.run_header import RunHeader

        app = OverwatchApp(replay_path=tamiyo_replay)

        async with app.run_test() as pilot:
            header = app.query_one(RunHeader)
            content = header.render_line1()

            assert "test-001" in content
            assert "cifar10" in content
            assert "5" in content  # episode

    @pytest.mark.asyncio
    async def test_tamiyo_strip_displays_vitals(self, tamiyo_replay: Path) -> None:
        """TamiyoStrip shows PPO vitals when loaded."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        app = OverwatchApp(replay_path=tamiyo_replay)

        async with app.run_test() as pilot:
            strip = app.query_one(TamiyoStrip)
            content = strip.render_vitals()

            assert "KL" in content
            assert "Ent" in content
            assert "EV" in content

    @pytest.mark.asyncio
    async def test_tamiyo_strip_shows_trend_arrows(self, tamiyo_replay: Path) -> None:
        """TamiyoStrip shows trend arrows for metrics."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.tamiyo_strip import TamiyoStrip

        app = OverwatchApp(replay_path=tamiyo_replay)

        async with app.run_test() as pilot:
            strip = app.query_one(TamiyoStrip)
            content = strip.render_vitals()

            # entropy_trend=-0.05 should show ↓
            assert "↓" in content

    @pytest.mark.asyncio
    async def test_header_shows_env_counts(self, tamiyo_replay: Path) -> None:
        """RunHeader shows environment health counts."""
        from esper.karn.overwatch import OverwatchApp
        from esper.karn.overwatch.widgets.run_header import RunHeader

        app = OverwatchApp(replay_path=tamiyo_replay)

        async with app.run_test() as pilot:
            header = app.query_one(RunHeader)
            content = header.render_line2()

            # envs_ok=2, envs_warn=1
            assert "OK:2" in content or "2" in content
            assert "WARN:1" in content or "1" in content
```

**Step 2: Run integration tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/test_integration.py::TestHeaderAndStripIntegration -v`

Expected: All 4 tests PASS

**Step 3: Commit**

```bash
git add tests/karn/overwatch/test_integration.py
git commit -m "test(overwatch): add integration tests for RunHeader and TamiyoStrip"
```

---

## Task 7: Run Full Test Suite

**Step 1: Run all Overwatch tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/overwatch/ -v`

Expected: All tests PASS (100+ tests)

**Step 2: Run linting**

Run: `uv run ruff check src/esper/karn/overwatch/`

Expected: No errors

**Step 3: Final commit if needed**

```bash
git status
# If any uncommitted files:
git add -A
git commit -m "chore(overwatch): Stage 3 complete - Header and Tamiyo Strip"
```

---

## Verification Checklist

- [ ] RunHeader displays run_id, task_name, episode, batch, runtime
- [ ] RunHeader shows connection status (Live/Stale/Disconnected)
- [ ] RunHeader shows env health counts (OK/WARN/CRIT)
- [ ] RunHeader shows best metric
- [ ] TamiyoStrip shows KL, Entropy, EV with values
- [ ] TamiyoStrip shows trend arrows (↑↓→)
- [ ] TamiyoStrip shows action distribution percentages
- [ ] TamiyoStrip shows recent actions sequence
- [ ] Health indicators apply correct CSS classes (ok/warn/crit)
- [ ] All tests pass (100+)
- [ ] Linting passes
- [ ] Can launch with fixture: `--replay tests/karn/overwatch/fixtures/tamiyo_active.jsonl`

---

## Files Created/Modified

```
src/esper/karn/overwatch/
├── display_state.py     # Modified: add health/trend functions
├── app.py               # Modified: use real widgets
└── widgets/
    ├── __init__.py      # Modified: export new widgets
    ├── run_header.py    # NEW: run identity widget (with DEFAULT_CSS for layout)
    └── tamiyo_strip.py  # NEW: PPO vitals widget (with DEFAULT_CSS for layout)

tests/karn/overwatch/
├── test_health_indicators.py  # NEW: trend/health tests
├── test_run_header.py         # NEW: RunHeader tests
├── test_tamiyo_strip.py       # NEW: TamiyoStrip tests
├── test_widgets.py            # Modified: add export tests
└── test_integration.py        # Modified: add integration tests
```

---

## Next Stage

After Stage 3 is merged, proceed to **Stage 4: Detail Panel + Event Feed** which will:
- Replace Detail Panel placeholder with real widget
- Add Event Feed with scrollable event list
- Implement event type coloring (GATE, STAGE, PPO, etc.)
- Add event expansion on click
