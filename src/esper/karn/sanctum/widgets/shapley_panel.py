"""ShapleyPanel - Visualization of Shapley value attribution.

Shows per-slot Shapley values with uncertainty bounds and significance indicators.
Shapley values are computed via permutation sampling at PPO batch boundaries.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import ShapleySnapshot, SeedState


class ShapleyPanel(Static):
    """Visualization of Shapley value attribution for slots."""

    def __init__(
        self,
        snapshot: "ShapleySnapshot",
        seeds: dict[str, "SeedState"] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._snapshot = snapshot
        self._seeds = seeds or {}

    def update_snapshot(
        self,
        snapshot: "ShapleySnapshot",
        seeds: dict[str, "SeedState"] | None = None,
    ) -> None:
        """Update the snapshot and refresh display."""
        self._snapshot = snapshot
        if seeds is not None:
            self._seeds = seeds
        self.refresh()

    def render(self) -> Panel:
        """Render the Shapley value panel."""
        if not self._snapshot.values:
            return self._render_unavailable()
        return self._render_values()

    def _render_unavailable(self) -> Panel:
        """Render unavailable state."""
        lines = []
        lines.append(Text("Shapley values unavailable", style="dim"))
        lines.append(Text(""))
        lines.append(Text("Computed at PPO batch boundaries", style="dim"))
        lines.append(Text("when 2+ seeds are active.", style="dim"))

        content = Group(*lines)
        return Panel(content, title="Shapley Attribution", border_style="dim")

    def _render_values(self) -> Panel:
        """Render the Shapley value visualization."""
        lines = []

        # Header with batch info (Shapley is computed at PPO batch boundaries)
        batch_str = f"Batch {self._snapshot.epoch}" if self._snapshot.epoch > 0 else "Latest"
        lines.append(Text(f"[{batch_str}]", style="dim"))
        lines.append(Text(""))

        # Get ranked slots
        ranked = self._snapshot.ranked_slots()
        if not ranked:
            lines.append(Text("No slots computed", style="dim"))
        else:
            # Show column headers
            lines.append(Text("Slot      Shapley    ±Std    Sig", style="bold"))
            lines.append(Text("─" * 35))

            for slot_id, mean in ranked:
                estimate = self._snapshot.values.get(slot_id)
                if estimate is None:
                    continue

                # Get seed stage for color coding
                seed = self._seeds.get(slot_id)
                stage = seed.stage if seed else "DORMANT"

                # Format values
                mean_str = f"{mean:+.3f}" if abs(mean) < 1 else f"{mean:+.1f}"
                std_str = f"±{estimate.std:.3f}" if estimate.std < 1 else f"±{estimate.std:.1f}"

                # Significance indicator
                is_sig = self._snapshot.get_significance(slot_id)
                sig_char = "★" if is_sig else "○"
                sig_style = "bold green" if is_sig and mean > 0 else (
                    "bold red" if is_sig and mean < 0 else "dim"
                )

                # Color based on contribution direction
                if mean > 0.01:
                    value_style = "green"
                elif mean < -0.01:
                    value_style = "red"
                else:
                    value_style = "dim"

                # Build line
                line = Text()
                line.append(f"{slot_id:<10}", style=self._get_stage_style(stage))
                line.append(f"{mean_str:>8}", style=value_style)
                line.append(f"  {std_str:>7}", style="dim")
                line.append(f"  {sig_char}", style=sig_style)
                lines.append(line)

        # Summary statistics
        lines.append(Text(""))
        total_contribution = sum(mean for _, mean in ranked)
        lines.append(Text(f"Total: {total_contribution:+.3f}", style="bold"))

        # Legend
        lines.append(Text(""))
        lines.append(Text("★ = significant (95% CI), ○ = not significant", style="dim"))

        content = Group(*lines)
        return Panel(content, title="Shapley Attribution", border_style="cyan")

    def _get_stage_style(self, stage: str) -> str:
        """Get Rich style for seed stage."""
        stage_styles = {
            "DORMANT": "dim",
            "GERMINATED": "yellow",
            "TRAINING": "cyan",
            "HOLDING": "magenta",
            "BLENDING": "blue",
            "FOSSILIZED": "green bold",
            "PRUNED": "red dim",
            "EMBARGOED": "red",
            "RESETTING": "yellow dim",
        }
        return stage_styles.get(stage, "white")
