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
