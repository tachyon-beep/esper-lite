"""SlotsPanel - Slot stage distribution across environments.

Displays:
- Border title: Total slots and environments count (static for run)
- Stage distribution with proportional bars (DORM, GERM, TRAIN, BLEND, HOLD, FOSS)
- Fossilization stats (Foss, Prune, Rate, AvgAge)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widgets import Static

from esper.leyline import STAGE_COLORS, STAGE_ABBREVIATIONS

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import SanctumSnapshot


def _format_count(count: int) -> str:
    """Format count with k suffix for values >= 1000.

    Examples:
        999 -> "999"
        1000 -> "1.00k"
        1500 -> "1.50k"
        9999 -> "9.99k"
        12345 -> "12.3k"
    """
    if count >= 10000:
        return f"{count / 1000:.1f}k"
    elif count >= 1000:
        return f"{count / 1000:.2f}k"
    return str(count)


class SlotsPanel(Static):
    """Slot stage summary panel.

    Extends Static directly for minimal layout overhead.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "CURRENT SLOTS"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        # Update border title with slot/env counts (static for the run)
        n_envs = len(snapshot.envs) if snapshot.envs else 0
        self.border_title = f"CURRENT SLOTS ─ {snapshot.total_slots} across {n_envs} envs"
        self.refresh()

    def render(self) -> Text:
        """Render slot summary."""
        if self._snapshot is None:
            return Text("[no data]", style="dim")

        snapshot = self._snapshot
        counts = snapshot.slot_stage_counts
        total = snapshot.total_slots

        if total == 0:
            return Text("[no environments]", style="dim")

        result = Text()

        # Stage distribution - one line per stage
        stages = ["DORMANT", "GERMINATED", "TRAINING", "BLENDING", "HOLDING", "FOSSILIZED"]
        stage_abbrevs = {k: v.upper() for k, v in STAGE_ABBREVIATIONS.items()}

        # Bar width expanded to fill available space (was 8, now 24)
        MAX_BAR_WIDTH = 24

        for stage in stages:
            count = counts.get(stage, 0)
            abbrev = stage_abbrevs.get(stage, stage[:4])
            base_color = STAGE_COLORS.get(stage, "dim")
            # Dim when zero, colored when active
            color = base_color if count > 0 else "dim"

            # Proportional bar (expanded to fill panel width)
            bar_width = int((count / max(1, total)) * MAX_BAR_WIDTH) if total > 0 else 0
            bar_char = "●" if stage != "DORMANT" else "○"
            bar = bar_char * bar_width

            # Left-align abbreviation, right-align count
            result.append(f"{abbrev:<5}", style="dim")
            result.append(f"{count:>3}", style=color)
            result.append(" ", style="dim")
            # Always show at least a placeholder bar character (dim when zero)
            if bar:
                result.append(bar, style=color)
            else:
                result.append("·", style="dim")  # Placeholder for zero count
            result.append("\n")

        # Separator before lifecycle section
        result.append("─" * 40, style="dim")
        result.append("\n")

        # === Seed Lifecycle Section ===
        lifecycle = snapshot.seed_lifecycle

        # Line 1: Active slot count
        result.append("Active:", style="dim")
        result.append(f"{lifecycle.active_count}/{lifecycle.total_slots}", style="cyan")
        result.append("\n")

        # Line 2: Cumulative counts (Germ / Prune / Foss) with k-formatting
        # Column order matches rate line below for vertical alignment
        # Fixed column widths: label(6) + value(6) + spacer(2) = 14 chars per column
        prune_style = "red" if lifecycle.prune_count > lifecycle.fossilize_count else "dim"

        result.append("Germ: ", style="dim")
        result.append(f"{_format_count(lifecycle.germination_count):>6}", style="green")
        result.append("  ", style="dim")
        result.append("Prune:", style="dim")
        result.append(f"{_format_count(lifecycle.prune_count):>6}", style=prune_style)
        result.append("  ", style="dim")
        result.append("Foss: ", style="dim")
        result.append(f"{_format_count(lifecycle.fossilize_count):>6}", style="blue")
        result.append("\n")

        # Line 3: Per-episode rates with trend indicators (same column order)
        def trend_arrow(trend: str) -> tuple[str, str]:
            """Return (arrow, style) for trend."""
            if trend == "rising":
                return "↗", "green"
            elif trend == "falling":
                return "↘", "red"
            else:
                return "→", "dim"

        g_arrow, g_style = trend_arrow(lifecycle.germination_trend)
        p_arrow, p_style = trend_arrow(lifecycle.prune_trend)
        f_arrow, f_style = trend_arrow(lifecycle.fossilize_trend)

        # Align with cumulative counts above: Germ | Prune | Foss
        result.append(f"Germ{g_arrow}", style=g_style)
        result.append(f"{lifecycle.germination_rate:>4.1f}/ep", style="dim")
        result.append("  ", style="dim")
        result.append(f"Prune{p_arrow}", style=p_style)
        result.append(f"{lifecycle.prune_rate:>3.1f}/ep", style="dim")
        result.append("  ", style="dim")
        result.append(f"Foss{f_arrow}", style=f_style)
        result.append(f"{lifecycle.fossilize_rate:>4.2f}/ep", style="dim")
        result.append("\n")

        # Line 3: Quality metrics (Blend success rate + avg lifespan)
        blend_rate = lifecycle.blend_success_rate * 100
        blend_color = "green" if blend_rate >= 70 else "yellow" if blend_rate >= 50 else "red"
        result.append("Lifespan:", style="dim")
        result.append(f"μ{lifecycle.avg_lifespan_epochs:.0f} eps", style="cyan")
        result.append("  Blend:", style="dim")
        result.append(f"{blend_rate:.0f}%", style=blend_color)
        result.append(" success", style="dim")

        return result
