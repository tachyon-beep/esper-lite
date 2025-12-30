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


class SlotsPanel(Static):
    """Slot stage summary panel.

    Extends Static directly for minimal layout overhead.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._snapshot: SanctumSnapshot | None = None
        self.classes = "panel"
        self.border_title = "SLOTS"

    def update_snapshot(self, snapshot: "SanctumSnapshot") -> None:
        """Update with new snapshot data."""
        self._snapshot = snapshot
        # Update border title with slot/env counts (static for the run)
        n_envs = len(snapshot.envs) if snapshot.envs else 0
        self.border_title = f"SLOTS ─ {snapshot.total_slots} across {n_envs} envs"
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
            color = STAGE_COLORS.get(stage, "dim")

            # Proportional bar (expanded to fill panel width)
            bar_width = int((count / max(1, total)) * MAX_BAR_WIDTH) if total > 0 else 0
            bar_char = "●" if stage != "DORMANT" else "○"
            bar = bar_char * bar_width

            # Left-align abbreviation, right-align count
            result.append(f"{abbrev:<5}", style="dim")
            result.append(f"{count:>3}", style=color)
            result.append(" ", style="dim")
            if bar:
                result.append(bar, style=color)
            result.append("\n")

        # Line 7: Summary stats
        foss = snapshot.cumulative_fossilized
        pruned = snapshot.cumulative_pruned
        rate = (foss / max(1, foss + pruned)) * 100 if (foss + pruned) > 0 else 0
        avg_epochs = snapshot.avg_epochs_in_stage

        result.append("Foss:", style="dim")
        result.append(f"{foss}", style="blue")
        result.append("  Prune:", style="dim")
        result.append(f"{pruned}", style="red" if pruned > foss else "dim")
        result.append("  Rate:", style="dim")
        rate_color = "green" if rate >= 70 else "yellow" if rate >= 50 else "red"
        result.append(f"{rate:.0f}%", style=rate_color)
        result.append("  AvgAge:", style="dim")
        result.append(f"{avg_epochs:.1f}", style="cyan")
        result.append(" epochs", style="dim")
        result.append("\n")

        # Line 8: Blueprint breakdown (top 3 by fossilization count)
        result.append("Top: ", style="dim")

        # Aggregate fossilized seeds by blueprint across all envs
        from collections import Counter
        blueprint_counts: Counter[str] = Counter()

        for env in snapshot.envs.values():
            for seed in env.seeds.values():
                if seed.stage == "FOSSILIZED" and seed.blueprint_id:
                    blueprint_counts[seed.blueprint_id] += 1

        if blueprint_counts:
            top_3 = blueprint_counts.most_common(3)
            for i, (bp, count) in enumerate(top_3):
                if i > 0:
                    result.append("  ", style="dim")
                bp_abbrev = bp[:7]  # Abbreviate long blueprint names
                result.append(f"{bp_abbrev}({count})", style="blue")
        else:
            result.append("none yet", style="dim")

        return result
