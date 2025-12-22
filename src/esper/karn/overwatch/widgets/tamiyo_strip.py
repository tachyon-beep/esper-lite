"""Tamiyo Strip Widget.

Displays PPO vitals, trend arrows, and action distribution in two rows
below the header. Color-coded health indicators show policy status.

Row 1: KL↑ 0.015 | Ent↓ 1.5 | EV→ 0.75 | Clip 8% | ∇ 0.5
Row 2: Actions: G:10% A:20% P:5% W:65% | Recent: GAWPW
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
    "SET_ALPHA_TARGET": "A",
    "PRUNE": "P",
    "WAIT": "W",
    "FOSSILIZE": "F",
}


class TamiyoStrip(Container):
    """Widget displaying Tamiyo PPO vitals and action distribution.

    Two-line format with health-colored indicators:
        KL↑ 0.015 | Ent↓ 1.5 | EV→ 0.75 | Clip 8% | ∇ 0.5
        Actions: G:10% A:20% P:5% W:65% | Recent: GAWPW
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
        """Render vitals line with trend arrows and health-based coloring."""
        if self._snapshot is None or self._snapshot.tamiyo is None:
            return "-- | Waiting for policy data..."

        t = self._snapshot.tamiyo

        # Format each metric with trend arrow
        kl_arrow = trend_arrow(t.kl_trend)
        ent_arrow = trend_arrow(t.entropy_trend)
        ev_arrow = trend_arrow(t.ev_trend)

        # Get health status for each vital
        health = self.get_vitals_health()

        # Map health levels to Rich colors
        color_map = {
            "ok": "green",
            "warn": "yellow",
            "crit": "red",
        }

        kl_color = color_map[health["kl"]]
        ent_color = color_map[health["entropy"]]
        ev_color = color_map[health["ev"]]

        # Apply Rich markup for health coloring
        kl_str = f"[{kl_color}]KL{kl_arrow} {t.kl_divergence:.3f}[/{kl_color}]"
        ent_str = f"[{ent_color}]Ent{ent_arrow} {t.entropy:.2f}[/{ent_color}]"
        ev_str = f"[{ev_color}]EV{ev_arrow} {t.explained_variance:.2f}[/{ev_color}]"
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
