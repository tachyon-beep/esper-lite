"""CounterfactualPanel - Waterfall visualization of factorial counterfactual analysis.

Shows baseline → individuals → pairs → combined with synergy calculation.
Displays "detailed counterfactual analysis unavailable" if no full factorial data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import CounterfactualSnapshot


class CounterfactualPanel(Static):
    """Waterfall visualization of counterfactual analysis."""

    def __init__(self, matrix: "CounterfactualSnapshot", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._matrix = matrix

    def update_matrix(self, matrix: "CounterfactualSnapshot") -> None:
        """Update the matrix and refresh display."""
        self._matrix = matrix
        self.refresh()

    def render(self) -> Panel:
        """Render the counterfactual analysis panel."""
        if self._matrix.strategy == "unavailable" or not self._matrix.configs:
            return self._render_unavailable()
        return self._render_waterfall()

    def _render_unavailable(self) -> Panel:
        """Render unavailable state."""
        content = Text("Detailed counterfactual analysis unavailable", style="dim italic")
        return Panel(content, title="Counterfactual Analysis", border_style="dim")

    def _render_waterfall(self) -> Panel:
        """Render the waterfall visualization."""
        lines = []

        baseline = self._matrix.baseline_accuracy
        combined = self._matrix.combined_accuracy
        individuals = self._matrix.individual_contributions()
        pairs = self._matrix.pair_contributions()
        synergy = self._matrix.total_synergy()
        n_seeds = len(self._matrix.slot_ids)
        is_ablation = self._matrix.strategy == "ablation_only"

        # Show strategy indicator for ablation-based estimates
        if is_ablation:
            lines.append(Text("Live Ablation Analysis", style="bold cyan"))
            lines.append(Text("(estimates based on cached baselines)", style="dim italic"))
            lines.append(Text(""))

        # Baseline
        lines.append(self._make_bar_line("Baseline (Host only)", baseline, baseline, combined))
        lines.append(Text(""))

        # Individuals section
        if individuals:
            lines.append(Text("Individual:", style="bold"))
            for slot_id, contrib in individuals.items():
                acc = baseline + contrib
                label = f"  {slot_id} alone"
                lines.append(self._make_bar_line(label, acc, baseline, combined, contrib))

        # Pairs section - show when we have pair data (2-3 seeds inline, 4+ top 5)
        # For 2 seeds, the "all enabled" config IS the pair, so pair_contributions() returns it
        if pairs and n_seeds <= 3:
            lines.append(Text(""))
            lines.append(Text("Pairs:", style="bold"))
            for (s1, s2), contrib in pairs.items():
                acc = baseline + contrib
                label = f"  {s1} + {s2}"
                # Calculate pair synergy
                ind1 = individuals.get(s1, 0)
                ind2 = individuals.get(s2, 0)
                pair_synergy = contrib - ind1 - ind2
                style = "green" if pair_synergy > 0.5 else None
                lines.append(self._make_bar_line(label, acc, baseline, combined, contrib, highlight=style))
        elif pairs and n_seeds > 3:
            # Show top 5 by synergy
            lines.append(Text(""))
            lines.append(Text("Top Combinations (by synergy):", style="bold"))

            # Calculate synergy for each pair
            pair_synergies = []
            for (s1, s2), contrib in pairs.items():
                ind1 = individuals.get(s1, 0)
                ind2 = individuals.get(s2, 0)
                pair_synergy = contrib - ind1 - ind2
                pair_synergies.append(((s1, s2), contrib, pair_synergy))

            # Sort by synergy descending, take top 5
            pair_synergies.sort(key=lambda x: x[2], reverse=True)
            for (s1, s2), contrib, pair_syn in pair_synergies[:5]:
                acc = baseline + contrib
                label = f"  {s1} + {s2}"
                style = "green" if pair_syn > 0.5 else None
                lines.append(self._make_bar_line(label, acc, baseline, combined, contrib, highlight=style))

        # Combined
        lines.append(Text(""))
        lines.append(Text("Combined:", style="bold"))
        improvement = combined - baseline
        lines.append(self._make_bar_line("  All seeds", combined, baseline, combined, improvement))

        # Synergy summary
        lines.append(Text(""))
        expected = sum(individuals.values())
        lines.append(Text(f"Expected (sum of solo): +{expected:.1f}%", style="dim"))
        lines.append(Text(f"Actual improvement:     +{improvement:.1f}%", style="dim"))

        # Interference is MORE critical to surface than synergy - seeds hurting each other
        # Use loud visual treatment for negative cases
        # Show "available at episode end" only when we don't have pair data yet
        if is_ablation and not pairs:
            lines.append(Text("(Pair interactions available at episode end)", style="dim italic"))
        elif synergy < -0.5:
            # INTERFERENCE: Seeds are hurting each other - make this LOUD
            lines.append(Text(""))
            lines.append(Text("✗ INTERFERENCE DETECTED", style="bold red reverse"))
            lines.append(Text(f"  Seeds are hurting each other by {synergy:.1f}%", style="red"))
        elif synergy > 0.5:
            # Synergy: Seeds working together
            lines.append(Text(f"✓ Synergy:              +{synergy:.1f}%", style="bold green"))
        else:
            # Neutral: Seeds are independent
            lines.append(Text(f"  Interaction:          {synergy:+.1f}%", style="dim"))

        content = Group(*lines)
        return Panel(content, title="Counterfactual Analysis", border_style="cyan")

    def _make_bar_line(
        self,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        delta: float | None = None,
        highlight: str | None = None,
    ) -> Text:
        """Create a bar line with label, visual bar, and value."""
        # Normalize to 0-1 range
        range_val = max_val - min_val if max_val > min_val else 1.0
        normalized = (value - min_val) / range_val if range_val > 0 else 0.0
        normalized = max(0.0, min(1.0, normalized))

        bar_width = 30
        filled = int(normalized * bar_width)
        empty = bar_width - filled

        line = Text()
        line.append(f"{label:20s} ", style=highlight or "white")
        line.append("█" * filled, style="cyan")
        line.append("░" * empty, style="dim")
        line.append(f" {value:5.1f}%", style="white")

        if delta is not None:
            delta_style = "green" if delta > 0 else "red" if delta < 0 else "dim"
            line.append(f"  ({delta:+.1f})", style=delta_style)

        return line
