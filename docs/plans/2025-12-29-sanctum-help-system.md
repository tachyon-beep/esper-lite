# Sanctum Contextual Help System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add contextual help modals to Sanctum TUI that explain how to interpret each panel's metrics, thresholds, and failure modes.

**Architecture:** Create a help content registry with dataclass-based content definitions. Each panel has three sections: overview, field reference, and diagnostics (what to look for). Pressing `?` while focused on a panel shows that panel's help; `??` or `F1` shows global help index. Help modals are scrollable and dismissible via Esc/q/?.

**Tech Stack:** Textual (ModalScreen, Static, VerticalScroll), Rich markup, Python dataclasses

---

## Task 1: Create Help Content Dataclasses

**Files:**
- Create: `src/esper/karn/sanctum/help/__init__.py`
- Create: `src/esper/karn/sanctum/help/content.py`
- Test: `tests/karn/sanctum/test_help_content.py`

**Context:** Define the data structures for panel help content. Each panel has an overview, field definitions with color rules, and diagnostic patterns.

**Step 1: Write failing test for help content dataclasses**

```python
# tests/karn/sanctum/test_help_content.py
"""Tests for help content dataclasses."""
import pytest

from esper.karn.sanctum.help.content import (
    DiagnosticPattern,
    FieldHelp,
    PanelHelp,
)


class TestHelpContentDataclasses:
    """Tests for help content data structures."""

    def test_field_help_creation(self):
        """FieldHelp should store name, description, and optional color rules."""
        field = FieldHelp(
            name="Entropy",
            description="Policy exploration level",
            color_rules="[green >0.3, yellow 0.1-0.3, red <0.1]",
        )
        assert field.name == "Entropy"
        assert field.description == "Policy exploration level"
        assert "green" in field.color_rules

    def test_field_help_no_color_rules(self):
        """FieldHelp should work without color rules."""
        field = FieldHelp(name="Slot", description="Target slot index")
        assert field.color_rules == ""

    def test_diagnostic_pattern_creation(self):
        """DiagnosticPattern should store severity, pattern, meaning, and action."""
        diag = DiagnosticPattern(
            severity="warning",
            pattern="Entropy < 0.1",
            meaning="Policy has collapsed to deterministic",
            action="Increase entropy_coef or restart training",
        )
        assert diag.severity == "warning"
        assert diag.action != ""

    def test_diagnostic_pattern_no_action(self):
        """DiagnosticPattern should work without action."""
        diag = DiagnosticPattern(
            severity="good",
            pattern="Entropy stable 0.3-0.7",
            meaning="Healthy exploration",
        )
        assert diag.action == ""

    def test_panel_help_creation(self):
        """PanelHelp should aggregate all help for one panel."""
        panel = PanelHelp(
            panel_id="tamiyo_brain",
            title="TAMIYO BRAIN",
            overview="Policy agent diagnostics.",
            fields=(
                FieldHelp("Entropy", "Exploration level"),
            ),
            diagnostics=(
                DiagnosticPattern("good", "Stable entropy", "Healthy"),
            ),
            keyboard={"j/k": "Navigate"},
        )
        assert panel.panel_id == "tamiyo_brain"
        assert len(panel.fields) == 1
        assert len(panel.diagnostics) == 1
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'esper.karn.sanctum.help'"

**Step 3: Create help package and content dataclasses**

```python
# src/esper/karn/sanctum/help/__init__.py
"""Sanctum contextual help system."""
from .content import DiagnosticPattern, FieldHelp, PanelHelp

__all__ = ["DiagnosticPattern", "FieldHelp", "PanelHelp"]
```

```python
# src/esper/karn/sanctum/help/content.py
"""Help content dataclasses for Sanctum panels.

Each panel has three sections:
1. overview: 2-3 sentence purpose statement
2. fields: column/field definitions with color rules
3. diagnostics: patterns to look for (good/warning/critical)
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldHelp:
    """Single field/column help entry.

    Attributes:
        name: Column or field name as displayed
        description: What this field shows
        color_rules: Optional threshold/color rules, e.g., "[green <0.1, red >0.5]"
    """

    name: str
    description: str
    color_rules: str = ""


@dataclass(frozen=True)
class DiagnosticPattern:
    """Single diagnostic pattern for the "What to Look For" section.

    Attributes:
        severity: "good", "warning", or "critical"
        pattern: What to look for (e.g., "Entropy < 0.1")
        meaning: What it means in operator terms
        action: What to do about it (empty if just informational)
    """

    severity: str  # "good", "warning", "critical"
    pattern: str
    meaning: str
    action: str = ""


@dataclass(frozen=True)
class PanelHelp:
    """Complete help content for one panel.

    Attributes:
        panel_id: Unique identifier matching registry
        title: Display title for help modal header
        overview: 2-3 sentence purpose statement
        fields: Tuple of FieldHelp entries
        diagnostics: Tuple of DiagnosticPattern entries
        keyboard: Dict of key -> action description
    """

    panel_id: str
    title: str
    overview: str
    fields: tuple[FieldHelp, ...] = field(default_factory=tuple)
    diagnostics: tuple[DiagnosticPattern, ...] = field(default_factory=tuple)
    keyboard: dict[str, str] = field(default_factory=dict)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/help/ tests/karn/sanctum/test_help_content.py
git commit -m "feat(sanctum): add help content dataclasses

Defines FieldHelp, DiagnosticPattern, and PanelHelp dataclasses
for the contextual help system. Each panel will have:
- Overview (purpose statement)
- Field reference (columns with color rules)
- Diagnostics (patterns to watch for)"
```

---

## Task 2: Create TamiyoBrain Help Content

**Files:**
- Create: `src/esper/karn/sanctum/help/panels/__init__.py`
- Create: `src/esper/karn/sanctum/help/panels/tamiyo_brain.py`
- Test: `tests/karn/sanctum/test_help_content.py` (extend)

**Context:** TamiyoBrain is the highest priority panel - it shows PPO health metrics that operators must understand. Content based on DRL specialist recommendations.

**Step 1: Write failing test for TamiyoBrain help content**

```python
# Add to tests/karn/sanctum/test_help_content.py

class TestTamiyoBrainHelpContent:
    """Tests for TamiyoBrain panel help content."""

    def test_tamiyo_brain_help_exists(self):
        """TamiyoBrain help content should be available."""
        from esper.karn.sanctum.help.panels.tamiyo_brain import TAMIYO_BRAIN_HELP

        assert TAMIYO_BRAIN_HELP.panel_id == "tamiyo_brain"
        assert TAMIYO_BRAIN_HELP.title == "TAMIYO BRAIN"

    def test_tamiyo_brain_has_key_fields(self):
        """TamiyoBrain help should explain entropy, KL, clip fraction, EV."""
        from esper.karn.sanctum.help.panels.tamiyo_brain import TAMIYO_BRAIN_HELP

        field_names = {f.name for f in TAMIYO_BRAIN_HELP.fields}
        assert "Entropy" in field_names
        assert "KL Div" in field_names
        assert "Clip Frac" in field_names
        assert "EV" in field_names

    def test_tamiyo_brain_has_diagnostics(self):
        """TamiyoBrain help should include diagnostic patterns."""
        from esper.karn.sanctum.help.panels.tamiyo_brain import TAMIYO_BRAIN_HELP

        assert len(TAMIYO_BRAIN_HELP.diagnostics) >= 3
        severities = {d.severity for d in TAMIYO_BRAIN_HELP.diagnostics}
        assert "good" in severities
        assert "warning" in severities or "critical" in severities

    def test_tamiyo_brain_entropy_thresholds(self):
        """Entropy field should have color threshold rules."""
        from esper.karn.sanctum.help.panels.tamiyo_brain import TAMIYO_BRAIN_HELP

        entropy_field = next(f for f in TAMIYO_BRAIN_HELP.fields if f.name == "Entropy")
        assert "0.1" in entropy_field.color_rules or "0.3" in entropy_field.color_rules
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py::TestTamiyoBrainHelpContent -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create TamiyoBrain help content**

```python
# src/esper/karn/sanctum/help/panels/__init__.py
"""Panel-specific help content."""
from .tamiyo_brain import TAMIYO_BRAIN_HELP

__all__ = ["TAMIYO_BRAIN_HELP"]
```

```python
# src/esper/karn/sanctum/help/panels/tamiyo_brain.py
"""Help content for TamiyoBrain panel.

Based on DRL specialist recommendations for operator-level explanations
of PPO health metrics and policy diagnostics.
"""
from esper.karn.sanctum.help.content import (
    DiagnosticPattern,
    FieldHelp,
    PanelHelp,
)

TAMIYO_BRAIN_HELP = PanelHelp(
    panel_id="tamiyo_brain",
    title="TAMIYO BRAIN",
    overview=(
        "Policy agent diagnostics: 'Is Tamiyo learning?' and 'What is she deciding?' "
        "Watch for entropy collapse (< 0.1) indicating stuck policy, or high KL (> 0.05) "
        "indicating unstable learning. Action distribution shows decision patterns."
    ),
    fields=(
        FieldHelp(
            name="Entropy",
            description="Policy exploration level. High = exploring many actions, Low = committed to specific actions.",
            color_rules="[green >0.3, yellow 0.1-0.3, red <0.1]",
        ),
        FieldHelp(
            name="KL Div",
            description="Policy change rate per update. How much Tamiyo's strategy changed.",
            color_rules="[green <0.015, yellow 0.015-0.03, red >0.03]",
        ),
        FieldHelp(
            name="Clip Frac",
            description="PPO constraint violations. How often updates hit the safety limit.",
            color_rules="[green <0.25, yellow 0.25-0.30, red >0.30]",
        ),
        FieldHelp(
            name="EV",
            description="Explained Variance. How well Tamiyo predicts future rewards. EV<0 = value network harmful.",
            color_rules="[green >0.5, yellow 0-0.5, red <0]",
        ),
        FieldHelp(
            name="Policy Loss",
            description="Main learning signal. Should decrease over time; spikes indicate instability.",
            color_rules="[trend: green decreasing, yellow flat, red increasing]",
        ),
        FieldHelp(
            name="Value Loss",
            description="Critic accuracy. Lower is better; trend matters more than absolute value.",
            color_rules="[trend: green decreasing, yellow flat, red increasing]",
        ),
        FieldHelp(
            name="Action Dist",
            description="Action frequencies: WAIT|GERM|ALPH|FOSS|PRUN. Healthy = WAIT dominant (60-80%) with lifecycle actions.",
            color_rules="",
        ),
    ),
    diagnostics=(
        DiagnosticPattern(
            severity="good",
            pattern="Entropy stable 0.3-0.7",
            meaning="Healthy exploration - Tamiyo is trying different strategies",
        ),
        DiagnosticPattern(
            severity="good",
            pattern="KL < 0.015 with improving returns",
            meaning="Stable learning - policy improving without erratic changes",
        ),
        DiagnosticPattern(
            severity="warning",
            pattern="Entropy < 0.2",
            meaning="Policy narrowing - may be converging or collapsing",
            action="Check accuracy. If low, consider increasing entropy_coef",
        ),
        DiagnosticPattern(
            severity="warning",
            pattern="WAIT > 80%",
            meaning="Policy too conservative - not taking lifecycle actions",
            action="Check if reward penalizes action too harshly",
        ),
        DiagnosticPattern(
            severity="critical",
            pattern="Entropy < 0.1 with low accuracy",
            meaning="Policy collapsed - stuck repeating same action",
            action="Restart training with higher entropy_coef",
        ),
        DiagnosticPattern(
            severity="critical",
            pattern="KL > 0.05",
            meaning="Unstable learning - policy changing too fast",
            action="Lower learning rate or increase batch size",
        ),
        DiagnosticPattern(
            severity="critical",
            pattern="EV < 0",
            meaning="Value network is harmful - worse than no critic",
            action="Check reward scale, value network architecture",
        ),
        DiagnosticPattern(
            severity="warning",
            pattern="GERM -> PRUN cycles",
            meaning="Thrashing - germinating seeds then immediately pruning",
            action="Seeds may need longer to prove themselves",
        ),
    ),
    keyboard={
        "j/k": "Navigate decision cards",
        "?": "Toggle this help",
    },
)
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py::TestTamiyoBrainHelpContent -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/help/panels/
git commit -m "feat(sanctum): add TamiyoBrain help content

DRL-specialist-recommended explanations for PPO metrics:
- Entropy, KL divergence, clip fraction, explained variance
- Diagnostic patterns for collapse, instability, thrashing
- Operator-level language with actionable thresholds"
```

---

## Task 3: Create PanelHelpModal

**Files:**
- Create: `src/esper/karn/sanctum/help/modal.py`
- Test: `tests/karn/sanctum/test_help_modal.py`

**Context:** The modal that renders help content. Scrollable, dismissible via Esc/q/?, supports j/k navigation.

**Step 1: Write failing test for PanelHelpModal**

```python
# tests/karn/sanctum/test_help_modal.py
"""Tests for PanelHelpModal."""
import pytest
from textual.app import App, ComposeResult

from esper.karn.sanctum.help.content import (
    DiagnosticPattern,
    FieldHelp,
    PanelHelp,
)
from esper.karn.sanctum.help.modal import PanelHelpModal


class HelpModalTestApp(App):
    """Test app for help modal."""

    def __init__(self, panel_help: PanelHelp | None = None):
        super().__init__()
        self._panel_help = panel_help

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(PanelHelpModal(self._panel_help))


@pytest.fixture
def sample_help():
    """Sample panel help for testing."""
    return PanelHelp(
        panel_id="test_panel",
        title="TEST PANEL",
        overview="This is a test panel for unit testing.",
        fields=(
            FieldHelp("Field1", "First field", "[green good, red bad]"),
            FieldHelp("Field2", "Second field"),
        ),
        diagnostics=(
            DiagnosticPattern("good", "Good pattern", "Things are working"),
            DiagnosticPattern("warning", "Warning pattern", "Watch this", "Take action"),
        ),
        keyboard={"j/k": "Navigate"},
    )


class TestPanelHelpModal:
    """Tests for the help modal."""

    @pytest.mark.asyncio
    async def test_modal_renders_title(self, sample_help):
        """Modal should display panel title."""
        app = HelpModalTestApp(sample_help)
        async with app.run_test() as pilot:
            # Modal should be showing
            assert app.screen.__class__.__name__ == "PanelHelpModal"
            # Title should be in rendered content
            content = app.screen.query_one("#help-content").renderable
            assert "TEST PANEL" in str(content)

    @pytest.mark.asyncio
    async def test_modal_renders_fields(self, sample_help):
        """Modal should display field definitions."""
        app = HelpModalTestApp(sample_help)
        async with app.run_test() as pilot:
            content = str(app.screen.query_one("#help-content").renderable)
            assert "Field1" in content
            assert "First field" in content

    @pytest.mark.asyncio
    async def test_modal_renders_diagnostics(self, sample_help):
        """Modal should display diagnostic patterns."""
        app = HelpModalTestApp(sample_help)
        async with app.run_test() as pilot:
            content = str(app.screen.query_one("#help-content").renderable)
            assert "Good pattern" in content
            assert "Warning pattern" in content

    @pytest.mark.asyncio
    async def test_modal_dismiss_on_escape(self, sample_help):
        """Modal should close on Escape key."""
        app = HelpModalTestApp(sample_help)
        async with app.run_test() as pilot:
            assert app.screen.__class__.__name__ == "PanelHelpModal"
            await pilot.press("escape")
            # Should be back to main screen
            assert app.screen.__class__.__name__ != "PanelHelpModal"

    @pytest.mark.asyncio
    async def test_modal_dismiss_on_q(self, sample_help):
        """Modal should close on q key."""
        app = HelpModalTestApp(sample_help)
        async with app.run_test() as pilot:
            await pilot.press("q")
            assert app.screen.__class__.__name__ != "PanelHelpModal"

    @pytest.mark.asyncio
    async def test_modal_handles_none_help(self):
        """Modal should handle missing help gracefully."""
        app = HelpModalTestApp(None)
        async with app.run_test() as pilot:
            content = str(app.screen.query_one("#help-content").renderable)
            assert "No help available" in content
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_modal.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create PanelHelpModal**

```python
# src/esper/karn/sanctum/help/modal.py
"""Contextual help modal for Sanctum panels."""
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from .content import PanelHelp


class PanelHelpModal(ModalScreen[None]):
    """Context-sensitive help modal for a specific panel.

    Displays:
    - Overview (what this panel shows)
    - Field reference (columns with color rules)
    - Diagnostics (what to look for)
    - Keyboard shortcuts

    Dismissible via: Escape, q, ?
    Scrollable via: j/k, arrows, g/G
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("question_mark", "dismiss", "Close"),
        Binding("j", "scroll_down", "Scroll Down", show=False),
        Binding("k", "scroll_up", "Scroll Up", show=False),
        Binding("down", "scroll_down", "Scroll Down", show=False),
        Binding("up", "scroll_up", "Scroll Up", show=False),
        Binding("g", "scroll_top", "Top", show=False),
        Binding("G", "scroll_bottom", "Bottom", show=False),
    ]

    DEFAULT_CSS = """
    PanelHelpModal {
        align: center middle;
        background: $surface-darken-1 85%;
    }

    PanelHelpModal > #help-container {
        width: 74;
        height: auto;
        max-height: 85%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    PanelHelpModal #help-scroll {
        height: auto;
        max-height: 100%;
    }
    """

    def __init__(self, panel_help: PanelHelp | None) -> None:
        """Initialize with panel help content.

        Args:
            panel_help: Help content for the panel, or None if unavailable
        """
        super().__init__()
        self._help = panel_help

    def compose(self) -> ComposeResult:
        """Compose the help modal."""
        with Container(id="help-container"):
            with VerticalScroll(id="help-scroll"):
                yield Static(self._render_content(), id="help-content")

    def _render_content(self) -> str:
        """Render help content as Rich markup."""
        if self._help is None:
            return "[dim]No help available for this panel.[/dim]\n\nPress Esc to close."

        lines: list[str] = []

        # Title bar
        lines.append(f"[bold cyan][?] {self._help.title} HELP[/bold cyan]")
        lines.append("")

        # Overview section
        lines.append("[bold]WHAT THIS SHOWS[/bold]")
        lines.append(self._help.overview)
        lines.append("")

        # Separator
        lines.append("[dim]" + "─" * 70 + "[/dim]")

        # Fields section
        if self._help.fields:
            lines.append("[bold]FIELD REFERENCE[/bold]")
            lines.append("")
            for field in self._help.fields:
                color_suffix = f" {field.color_rules}" if field.color_rules else ""
                lines.append(f"  [cyan]{field.name:12}[/cyan] {field.description}")
                if color_suffix:
                    lines.append(f"               {color_suffix}")
            lines.append("")

        # Separator
        lines.append("[dim]" + "─" * 70 + "[/dim]")

        # Diagnostics section
        if self._help.diagnostics:
            lines.append("[bold]WHAT TO LOOK FOR[/bold]")
            lines.append("")
            for diag in self._help.diagnostics:
                icon = {"good": "[green]+[/green]", "warning": "[yellow]![/yellow]", "critical": "[red]X[/red]"}.get(
                    diag.severity, "?"
                )
                lines.append(f"  {icon} [bold]{diag.pattern}[/bold]")
                lines.append(f"     [dim]{diag.meaning}[/dim]")
                if diag.action:
                    lines.append(f"     [italic]Action: {diag.action}[/italic]")
            lines.append("")

        # Separator
        lines.append("[dim]" + "─" * 70 + "[/dim]")

        # Keyboard section
        if self._help.keyboard:
            lines.append("[bold]KEYBOARD[/bold]")
            for key, action in self._help.keyboard.items():
                lines.append(f"  [cyan]{key:12}[/cyan] {action}")
            lines.append("")

        lines.append("[dim]Press Esc, q, or ? to close[/dim]")

        return "\n".join(lines)

    def on_click(self) -> None:
        """Dismiss on click outside content."""
        self.dismiss()

    def action_scroll_down(self) -> None:
        """Scroll content down."""
        scroll = self.query_one("#help-scroll", VerticalScroll)
        scroll.scroll_down()

    def action_scroll_up(self) -> None:
        """Scroll content up."""
        scroll = self.query_one("#help-scroll", VerticalScroll)
        scroll.scroll_up()

    def action_scroll_top(self) -> None:
        """Scroll to top."""
        scroll = self.query_one("#help-scroll", VerticalScroll)
        scroll.scroll_home()

    def action_scroll_bottom(self) -> None:
        """Scroll to bottom."""
        scroll = self.query_one("#help-scroll", VerticalScroll)
        scroll.scroll_end()
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_modal.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/help/modal.py tests/karn/sanctum/test_help_modal.py
git commit -m "feat(sanctum): add PanelHelpModal

Textual ModalScreen that renders panel help content:
- Scrollable via j/k, arrows, g/G
- Dismissible via Esc, q, ?
- Three sections: overview, fields, diagnostics
- Handles missing help gracefully"
```

---

## Task 4: Create Panel ID Registry

**Files:**
- Create: `src/esper/karn/sanctum/help/registry.py`
- Modify: `src/esper/karn/sanctum/help/__init__.py`
- Test: `tests/karn/sanctum/test_help_content.py` (extend)

**Context:** Maps widget classes/IDs to their help content. Used by app to determine which help to show.

**Step 1: Write failing test for registry**

```python
# Add to tests/karn/sanctum/test_help_content.py

class TestHelpRegistry:
    """Tests for help content registry."""

    def test_get_help_for_known_panel(self):
        """Registry should return help for known panel IDs."""
        from esper.karn.sanctum.help.registry import get_help_for_panel

        help_content = get_help_for_panel("tamiyo_brain")
        assert help_content is not None
        assert help_content.panel_id == "tamiyo_brain"

    def test_get_help_for_unknown_panel(self):
        """Registry should return None for unknown panel IDs."""
        from esper.karn.sanctum.help.registry import get_help_for_panel

        help_content = get_help_for_panel("nonexistent_panel")
        assert help_content is None

    def test_get_panel_id_for_widget_class(self):
        """Registry should map widget class names to panel IDs."""
        from esper.karn.sanctum.help.registry import get_panel_id_for_widget_class

        panel_id = get_panel_id_for_widget_class("TamiyoBrain")
        assert panel_id == "tamiyo_brain"

    def test_get_panel_id_for_unknown_widget(self):
        """Registry should return None for unmapped widgets."""
        from esper.karn.sanctum.help.registry import get_panel_id_for_widget_class

        panel_id = get_panel_id_for_widget_class("UnknownWidget")
        assert panel_id is None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py::TestHelpRegistry -v`
Expected: FAIL with "cannot import name 'get_help_for_panel'"

**Step 3: Create registry**

```python
# src/esper/karn/sanctum/help/registry.py
"""Registry mapping panels to help content.

Provides:
- get_help_for_panel(panel_id) -> PanelHelp | None
- get_panel_id_for_widget_class(class_name) -> str | None
"""
from __future__ import annotations

from .content import PanelHelp
from .panels import TAMIYO_BRAIN_HELP

# Panel ID -> Help content
_HELP_REGISTRY: dict[str, PanelHelp] = {
    "tamiyo_brain": TAMIYO_BRAIN_HELP,
}

# Widget class name -> Panel ID
# Used to determine which help to show based on focused widget
_WIDGET_TO_PANEL: dict[str, str] = {
    "TamiyoBrain": "tamiyo_brain",
}


def get_help_for_panel(panel_id: str) -> PanelHelp | None:
    """Get help content for a panel by ID.

    Args:
        panel_id: Panel identifier (e.g., "tamiyo_brain")

    Returns:
        PanelHelp if found, None otherwise
    """
    return _HELP_REGISTRY.get(panel_id)


def get_panel_id_for_widget_class(class_name: str) -> str | None:
    """Get panel ID for a widget class name.

    Args:
        class_name: Widget class name (e.g., "TamiyoBrain")

    Returns:
        Panel ID if mapped, None otherwise
    """
    return _WIDGET_TO_PANEL.get(class_name)


def get_all_panel_ids() -> list[str]:
    """Get all registered panel IDs.

    Returns:
        List of panel IDs with help content
    """
    return list(_HELP_REGISTRY.keys())
```

Update `__init__.py`:

```python
# src/esper/karn/sanctum/help/__init__.py
"""Sanctum contextual help system."""
from .content import DiagnosticPattern, FieldHelp, PanelHelp
from .modal import PanelHelpModal
from .registry import get_help_for_panel, get_panel_id_for_widget_class

__all__ = [
    "DiagnosticPattern",
    "FieldHelp",
    "PanelHelp",
    "PanelHelpModal",
    "get_help_for_panel",
    "get_panel_id_for_widget_class",
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py::TestHelpRegistry -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/help/
git commit -m "feat(sanctum): add help content registry

Maps widget classes to panel IDs and panel IDs to help content.
Currently registers TamiyoBrain; more panels added in future tasks."
```

---

## Task 5: Integrate Context-Sensitive Help with App

**Files:**
- Modify: `src/esper/karn/sanctum/app.py:436-438` (`action_toggle_help`)
- Test: `tests/karn/sanctum/test_app.py` (extend)

**Context:** Make `?` key show context-sensitive help based on focused widget. Falls back to global HelpScreen if no panel-specific help.

**Step 1: Write failing test for context-sensitive help**

```python
# Add to tests/karn/sanctum/test_app.py (or create new test class)

class TestContextSensitiveHelp:
    """Tests for context-sensitive help integration."""

    @pytest.mark.asyncio
    async def test_help_on_tamiyo_brain_shows_panel_help(self):
        """Pressing ? while TamiyoBrain focused should show panel help."""
        from esper.karn.sanctum.app import SanctumApp
        from esper.karn.sanctum.help.modal import PanelHelpModal

        app = SanctumApp()
        async with app.run_test() as pilot:
            # Focus TamiyoBrain widget
            tamiyo = app.query_one("TamiyoBrain")
            tamiyo.focus()

            # Press ?
            await pilot.press("?")

            # Should show PanelHelpModal, not HelpScreen
            assert isinstance(app.screen, PanelHelpModal)

    @pytest.mark.asyncio
    async def test_help_on_unknown_widget_shows_global_help(self):
        """Pressing ? on widget without help should show global HelpScreen."""
        from esper.karn.sanctum.app import HelpScreen, SanctumApp

        app = SanctumApp()
        async with app.run_test() as pilot:
            # Press ? without specific focus
            await pilot.press("?")

            # Should show HelpScreen (global)
            assert isinstance(app.screen, HelpScreen)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::TestContextSensitiveHelp -v`
Expected: FAIL (PanelHelpModal not shown)

**Step 3: Modify action_toggle_help for context sensitivity**

In `src/esper/karn/sanctum/app.py`, replace lines 436-438:

```python
    def action_toggle_help(self) -> None:
        """Toggle context-sensitive help display.

        If a widget with panel-specific help is focused, shows that panel's help.
        Otherwise shows global keyboard shortcuts.
        """
        from esper.karn.sanctum.help import (
            PanelHelpModal,
            get_help_for_panel,
            get_panel_id_for_widget_class,
        )

        # Determine which panel has focus
        focused = self.focused
        panel_id = None

        if focused is not None:
            # Check the focused widget's class and its ancestors
            widget = focused
            while widget is not None and panel_id is None:
                class_name = widget.__class__.__name__
                panel_id = get_panel_id_for_widget_class(class_name)
                widget = widget.parent if hasattr(widget, "parent") else None

        if panel_id is not None:
            # Show panel-specific help
            help_content = get_help_for_panel(panel_id)
            self.push_screen(PanelHelpModal(help_content))
        else:
            # Fall back to global help
            self.push_screen(HelpScreen())
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_app.py::TestContextSensitiveHelp -v`
Expected: PASS (both tests)

**Step 5: Run full Sanctum test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v --tb=short`
Expected: All PASS (no regressions)

**Step 6: Commit**

```bash
git add src/esper/karn/sanctum/app.py tests/karn/sanctum/test_app.py
git commit -m "feat(sanctum): integrate context-sensitive help

Pressing ? now shows panel-specific help when focused on a
widget with registered help content. Falls back to global
HelpScreen for unregistered widgets."
```

---

## Task 6: Add EnvOverview Help Content

**Files:**
- Create: `src/esper/karn/sanctum/help/panels/env_overview.py`
- Modify: `src/esper/karn/sanctum/help/panels/__init__.py`
- Modify: `src/esper/karn/sanctum/help/registry.py`
- Test: `tests/karn/sanctum/test_help_content.py` (extend)

**Context:** EnvOverview is the second highest priority panel - it shows per-environment training status with complex slot columns.

**Step 1: Write failing test for EnvOverview help**

```python
# Add to tests/karn/sanctum/test_help_content.py

class TestEnvOverviewHelpContent:
    """Tests for EnvOverview panel help content."""

    def test_env_overview_help_exists(self):
        """EnvOverview help content should be available."""
        from esper.karn.sanctum.help.panels.env_overview import ENV_OVERVIEW_HELP

        assert ENV_OVERVIEW_HELP.panel_id == "env_overview"

    def test_env_overview_has_key_fields(self):
        """EnvOverview help should explain acc, loss, slots, status."""
        from esper.karn.sanctum.help.panels.env_overview import ENV_OVERVIEW_HELP

        field_names = {f.name for f in ENV_OVERVIEW_HELP.fields}
        assert "Acc" in field_names or "Accuracy" in field_names
        assert "Status" in field_names
        assert "Slots" in field_names or any("slot" in f.name.lower() for f in ENV_OVERVIEW_HELP.fields)

    def test_env_overview_registered(self):
        """EnvOverview should be registered in help registry."""
        from esper.karn.sanctum.help.registry import get_help_for_panel

        help_content = get_help_for_panel("env_overview")
        assert help_content is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py::TestEnvOverviewHelpContent -v`
Expected: FAIL

**Step 3: Create EnvOverview help content**

```python
# src/esper/karn/sanctum/help/panels/env_overview.py
"""Help content for EnvOverview panel.

Based on DRL specialist and UX specialist recommendations for
explaining the per-environment training table.
"""
from esper.karn.sanctum.help.content import (
    DiagnosticPattern,
    FieldHelp,
    PanelHelp,
)

ENV_OVERVIEW_HELP = PanelHelp(
    panel_id="env_overview",
    title="ENVIRONMENT OVERVIEW",
    overview=(
        "Per-environment training table with one row per parallel environment. "
        "Scan for anomalies (red/yellow rows) and identify which environments are "
        "leading (green accuracy) vs struggling (STALLED/DEGRADED status)."
    ),
    fields=(
        FieldHelp(
            name="Env",
            description="Environment ID with cohort pip (color = reward mode). [cyan]▶[/cyan] = received last action.",
            color_rules="",
        ),
        FieldHelp(
            name="Acc",
            description="Current CIFAR validation accuracy. Arrow: ↑improving, ↓declining, →stable.",
            color_rules="[green at best, yellow stale >5ep]",
        ),
        FieldHelp(
            name="Loss",
            description="Host model loss value.",
            color_rules="[green <0.1, yellow 0.5-1.0, red >1.0]",
        ),
        FieldHelp(
            name="CF",
            description="Counterfactual synergy. Positive = seeds helping each other.",
            color_rules="[green +synergy, red -interference]",
        ),
        FieldHelp(
            name="Growth",
            description="Size ratio: (host+fossilized)/host. How much model has grown.",
            color_rules="[green <1.5x, yellow <2.5x, red >2.5x]",
        ),
        FieldHelp(
            name="Reward",
            description="Current step reward (+avg). What Tamiyo received.",
            color_rules="[green >0, red <-0.5]",
        ),
        FieldHelp(
            name="Slots",
            description="Seed state per slot: Stage:Blueprint Curve Tempo Alpha. "
                        "Curve: ⌢concave, ⌣convex, −linear. Tempo: ▸▸▸fast, ▸▸std, ▸slow.",
            color_rules="",
        ),
        FieldHelp(
            name="Stale",
            description="Epochs since accuracy improvement.",
            color_rules="[green +0, yellow !N, red xN]",
        ),
        FieldHelp(
            name="Status",
            description="★EXCL excellent | ●OK healthy | ○INIT | ◐STAL stalled | ▼DEGR degraded",
            color_rules="[green ★●, yellow ◐, red ▼]",
        ),
    ),
    diagnostics=(
        DiagnosticPattern(
            severity="good",
            pattern="★EXCL or ●OK status with ↑ trend",
            meaning="Environment is healthy and improving",
        ),
        DiagnosticPattern(
            severity="good",
            pattern="Low Growth (<1.5x) with high Acc",
            meaning="Efficient seed selection - accuracy without bloat",
        ),
        DiagnosticPattern(
            severity="warning",
            pattern="◐STAL with high stale count",
            meaning="Environment stuck - no improvement in many epochs",
            action="Check seed composition, consider pruning",
        ),
        DiagnosticPattern(
            severity="warning",
            pattern="Negative CF (interference)",
            meaning="Seeds are hurting each other's performance",
            action="May need to prune conflicting seeds",
        ),
        DiagnosticPattern(
            severity="critical",
            pattern="▼DEGR status",
            meaning="Environment actively getting worse",
            action="Check for gradient issues, reward hacking",
        ),
        DiagnosticPattern(
            severity="critical",
            pattern="All slots DORMANT for many epochs",
            meaning="Tamiyo not germinating - policy may be stuck on WAIT",
            action="Check entropy in TamiyoBrain panel",
        ),
    ),
    keyboard={
        "j/k, ↑/↓": "Navigate rows",
        "d": "Open environment detail modal",
        "/": "Filter environments",
        "1-9": "Jump to environment by ID",
        "?": "Toggle this help",
    },
)
```

Update `panels/__init__.py`:

```python
# src/esper/karn/sanctum/help/panels/__init__.py
"""Panel-specific help content."""
from .env_overview import ENV_OVERVIEW_HELP
from .tamiyo_brain import TAMIYO_BRAIN_HELP

__all__ = ["ENV_OVERVIEW_HELP", "TAMIYO_BRAIN_HELP"]
```

Update registry:

```python
# In src/esper/karn/sanctum/help/registry.py
# Add to imports:
from .panels import ENV_OVERVIEW_HELP, TAMIYO_BRAIN_HELP

# Update _HELP_REGISTRY:
_HELP_REGISTRY: dict[str, PanelHelp] = {
    "tamiyo_brain": TAMIYO_BRAIN_HELP,
    "env_overview": ENV_OVERVIEW_HELP,
}

# Update _WIDGET_TO_PANEL:
_WIDGET_TO_PANEL: dict[str, str] = {
    "TamiyoBrain": "tamiyo_brain",
    "EnvOverview": "env_overview",
}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py::TestEnvOverviewHelpContent -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/help/
git commit -m "feat(sanctum): add EnvOverview help content

Explains per-environment training table columns:
- Accuracy, loss, counterfactual synergy, growth ratio
- Slot format: Stage:Blueprint Curve Tempo Alpha
- Status icons and diagnostic patterns"
```

---

## Task 7: Add Scoreboard Help Content

**Files:**
- Create: `src/esper/karn/sanctum/help/panels/scoreboard.py`
- Modify: `src/esper/karn/sanctum/help/panels/__init__.py`
- Modify: `src/esper/karn/sanctum/help/registry.py`
- Test: `tests/karn/sanctum/test_help_content.py` (extend)

**Context:** Scoreboard shows best runs - operators need to understand what makes a "good" run.

**Step 1: Write failing test for Scoreboard help**

```python
# Add to tests/karn/sanctum/test_help_content.py

class TestScoreboardHelpContent:
    """Tests for Scoreboard panel help content."""

    def test_scoreboard_help_exists(self):
        """Scoreboard help content should be available."""
        from esper.karn.sanctum.help.panels.scoreboard import SCOREBOARD_HELP

        assert SCOREBOARD_HELP.panel_id == "scoreboard"

    def test_scoreboard_registered(self):
        """Scoreboard should be registered in help registry."""
        from esper.karn.sanctum.help.registry import get_help_for_panel

        help_content = get_help_for_panel("scoreboard")
        assert help_content is not None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py::TestScoreboardHelpContent -v`
Expected: FAIL

**Step 3: Create Scoreboard help content**

```python
# src/esper/karn/sanctum/help/panels/scoreboard.py
"""Help content for Scoreboard panel."""
from esper.karn.sanctum.help.content import (
    DiagnosticPattern,
    FieldHelp,
    PanelHelp,
)

SCOREBOARD_HELP = PanelHelp(
    panel_id="scoreboard",
    title="BEST RUNS",
    overview=(
        "Top 10 completed episode runs ranked by peak accuracy achieved. "
        "Use this to track 'best ever' performance and see which seed "
        "compositions produced the strongest results."
    ),
    fields=(
        FieldHelp(
            name="#",
            description="Rank position. Pin icon (📌) if pinned.",
            color_rules="",
        ),
        FieldHelp(
            name="Ep",
            description="Episode number when peak was achieved.",
            color_rules="",
        ),
        FieldHelp(
            name="Acc",
            description="Peak accuracy reached in that episode.",
            color_rules="[green = best overall]",
        ),
        FieldHelp(
            name="Growth",
            description="Model size ratio at peak: (host+fossilized)/host.",
            color_rules="[green <1.5x, yellow <2.5x, red >2.5x]",
        ),
        FieldHelp(
            name="Seeds",
            description="Fossilized blueprints at peak. +N if more than shown.",
            color_rules="[green]Foss[/green], [yellow]Blend[/yellow]",
        ),
    ),
    diagnostics=(
        DiagnosticPattern(
            severity="good",
            pattern="High Acc with low Growth",
            meaning="Efficient seed selection - accuracy without bloat",
        ),
        DiagnosticPattern(
            severity="good",
            pattern="Same blueprints across top runs",
            meaning="Robust architecture found - consistent winners",
        ),
        DiagnosticPattern(
            severity="warning",
            pattern="Growth >2.0x without proportional Acc gain",
            meaning="Bloat - seeds adding params without value",
            action="Check if compute rent is too low",
        ),
        DiagnosticPattern(
            severity="warning",
            pattern="All top runs from early episodes",
            meaning="Training may have stalled or regressed",
            action="Check TamiyoBrain for policy issues",
        ),
    ),
    keyboard={
        "j/k, ↑/↓": "Navigate leaderboard",
        "Enter": "View frozen snapshot at peak",
        "p": "Toggle pin (pinned rows preserved)",
        "?": "Toggle this help",
    },
)
```

Update `panels/__init__.py` and registry similarly to Task 6.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_content.py::TestScoreboardHelpContent -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/esper/karn/sanctum/help/
git commit -m "feat(sanctum): add Scoreboard help content

Explains best runs leaderboard:
- Accuracy, growth ratio, seed composition
- What makes a 'good' run (high acc, low growth)
- Warning patterns (bloat, stalled training)"
```

---

## Task 8: Integration Test and Final Verification

**Files:**
- Test: `tests/karn/sanctum/test_help_integration.py` (create)

**Context:** Verify the complete help system works end-to-end.

**Step 1: Write integration test**

```python
# tests/karn/sanctum/test_help_integration.py
"""Integration tests for Sanctum help system."""
import pytest

from esper.karn.sanctum.app import SanctumApp
from esper.karn.sanctum.help import get_all_panel_ids, get_help_for_panel
from esper.karn.sanctum.help.modal import PanelHelpModal


class TestHelpSystemIntegration:
    """Integration tests for the complete help system."""

    def test_all_registered_panels_have_content(self):
        """Every registered panel ID should have help content."""
        from esper.karn.sanctum.help.registry import get_all_panel_ids

        for panel_id in get_all_panel_ids():
            help_content = get_help_for_panel(panel_id)
            assert help_content is not None, f"Missing content for {panel_id}"
            assert help_content.overview, f"Empty overview for {panel_id}"
            assert help_content.fields, f"No fields for {panel_id}"

    def test_all_help_content_has_required_sections(self):
        """All help content should have overview, fields, and diagnostics."""
        from esper.karn.sanctum.help.registry import get_all_panel_ids

        for panel_id in get_all_panel_ids():
            help_content = get_help_for_panel(panel_id)
            assert len(help_content.overview) >= 50, f"Overview too short for {panel_id}"
            assert len(help_content.fields) >= 3, f"Too few fields for {panel_id}"
            assert len(help_content.diagnostics) >= 2, f"Too few diagnostics for {panel_id}"

    @pytest.mark.asyncio
    async def test_help_modal_renders_all_panels(self):
        """PanelHelpModal should render content for all registered panels."""
        from esper.karn.sanctum.help.registry import get_all_panel_ids

        for panel_id in get_all_panel_ids():
            help_content = get_help_for_panel(panel_id)

            # Create minimal test app
            from textual.app import App

            class TestApp(App):
                def on_mount(self):
                    self.push_screen(PanelHelpModal(help_content))

            app = TestApp()
            async with app.run_test() as pilot:
                content = str(app.screen.query_one("#help-content").renderable)
                assert help_content.title in content, f"Title missing for {panel_id}"
                assert "WHAT THIS SHOWS" in content
                assert "FIELD REFERENCE" in content

    def test_minimum_panels_registered(self):
        """At least 3 high-priority panels should be registered."""
        from esper.karn.sanctum.help.registry import get_all_panel_ids

        panel_ids = get_all_panel_ids()
        assert len(panel_ids) >= 3, f"Only {len(panel_ids)} panels registered"
        assert "tamiyo_brain" in panel_ids
        assert "env_overview" in panel_ids
        assert "scoreboard" in panel_ids
```

**Step 2: Run integration tests**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/test_help_integration.py -v`
Expected: PASS (all 4 tests)

**Step 3: Run full Sanctum test suite**

Run: `PYTHONPATH=src uv run pytest tests/karn/sanctum/ -v --tb=short`
Expected: All PASS

**Step 4: Manual verification**

Run: `PYTHONPATH=src python -m esper.karn.sanctum.app --demo`
- Focus TamiyoBrain widget, press `?` → should show panel help
- Focus EnvOverview widget, press `?` → should show panel help
- Press `?` without focus → should show global help
- Verify j/k scrolling, Esc dismissal

**Step 5: Final commit**

```bash
git add tests/karn/sanctum/test_help_integration.py
git commit -m "test(sanctum): add help system integration tests

Verifies:
- All registered panels have complete content
- Modal renders correctly for all panels
- Minimum 3 high-priority panels registered"
```

---

## Summary of Changes

| Component | Files | Purpose |
|-----------|-------|---------|
| Content dataclasses | `help/content.py` | FieldHelp, DiagnosticPattern, PanelHelp |
| Panel content | `help/panels/*.py` | TamiyoBrain, EnvOverview, Scoreboard |
| Help modal | `help/modal.py` | PanelHelpModal (scrollable, dismissible) |
| Registry | `help/registry.py` | Widget → panel_id → content mapping |
| App integration | `app.py` | Context-sensitive `action_toggle_help` |

**Panels with help content (MVP):**
1. TamiyoBrain - PPO metrics, entropy, KL, action distribution
2. EnvOverview - Per-env table, slots, status icons
3. Scoreboard - Best runs, growth ratio, seed composition

**Future panels (add incrementally):**
- RewardHealth - Gaming rate, PBRS fraction
- CounterfactualPanel - Waterfall, synergy
- AnomalyStrip - Anomaly codes
- EventLog - Event types

---

## Execution Checklist

- [ ] Task 1: Help content dataclasses
- [ ] Task 2: TamiyoBrain help content
- [ ] Task 3: PanelHelpModal
- [ ] Task 4: Panel ID registry
- [ ] Task 5: App integration
- [ ] Task 6: EnvOverview help content
- [ ] Task 7: Scoreboard help content
- [ ] Task 8: Integration tests
