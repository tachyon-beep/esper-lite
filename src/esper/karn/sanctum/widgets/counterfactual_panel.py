"""CounterfactualPanel - Waterfall visualization of factorial counterfactual analysis.

Shows baseline → individuals → pairs → combined with synergy calculation.
Also shows per-seed interaction metrics when available.
Displays "detailed counterfactual analysis unavailable" if no full factorial data.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.widgets import Static

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import CounterfactualSnapshot, SeedState


class CounterfactualPanel(Static):
    """Waterfall visualization of counterfactual analysis."""

    def __init__(
        self,
        matrix: "CounterfactualSnapshot",
        seeds: dict[str, "SeedState"] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._matrix = matrix
        self._seeds = seeds or {}

    def update_matrix(
        self,
        matrix: "CounterfactualSnapshot",
        seeds: dict[str, "SeedState"] | None = None,
    ) -> None:
        """Update the matrix and refresh display."""
        self._matrix = matrix
        if seeds is not None:
            self._seeds = seeds
        self.refresh()

    def render(self) -> Panel:
        """Render the counterfactual analysis panel."""
        if self._matrix.strategy == "unavailable" or not self._matrix.configs:
            return self._render_unavailable()
        return self._render_waterfall()

    def _render_unavailable(self) -> Panel:
        """Render unavailable state with same structure as waterfall.

        All rows are always visible to prevent jarring layout shifts.
        Uses dim "--" placeholders to match waterfall structure.
        """
        lines = []
        dim_placeholder = "--"

        # Baseline (always visible)
        lines.append(Text(f"Baseline (Host only)     {dim_placeholder}", style="dim"))
        lines.append(Text(""))

        # Individual section (always visible)
        lines.append(Text("Individual:", style="bold"))
        lines.append(Text(f"  {dim_placeholder}", style="dim"))

        # Pairs section placeholder
        lines.append(Text(""))
        lines.append(Text("Pairs:", style="bold"))
        lines.append(Text(f"  {dim_placeholder}", style="dim"))

        # Combined (always visible)
        lines.append(Text(""))
        lines.append(Text("Combined:", style="bold"))
        lines.append(Text(f"  All seeds              {dim_placeholder}", style="dim"))

        # Synergy summary (always visible)
        lines.append(Text(""))
        lines.append(Text(f"Expected (sum of solo): {dim_placeholder}", style="dim"))
        lines.append(Text(f"Actual improvement:     {dim_placeholder}", style="dim"))
        lines.append(Text(f"  Interaction:          {dim_placeholder}", style="dim"))

        # Seed Dynamics section (always visible)
        lines.append(Text(""))
        lines.extend(self._render_interaction_metrics())

        content = Group(*lines)
        return Panel(content, title="Counterfactual Analysis", border_style="dim")

    def _render_waterfall(self) -> Panel:
        """Render the waterfall visualization."""
        lines = []

        slot_ids = self._matrix.slot_ids
        n_seeds = len(slot_ids)
        accuracy_by_mask = {
            cfg.seed_mask: cfg.accuracy for cfg in self._matrix.configs
        }

        baseline_mask = tuple(False for _ in range(n_seeds))
        if baseline_mask in accuracy_by_mask:
            baseline = accuracy_by_mask[baseline_mask]
        else:
            baseline = 0.0

        combined_mask = tuple(True for _ in range(n_seeds))
        if combined_mask in accuracy_by_mask:
            combined = accuracy_by_mask[combined_mask]
        else:
            combined = 0.0

        individuals: dict[str, float] = {}
        for i, slot_id in enumerate(slot_ids):
            mask = tuple(j == i for j in range(n_seeds))
            if mask in accuracy_by_mask:
                individuals[slot_id] = accuracy_by_mask[mask] - baseline

        pairs: dict[tuple[str, str], float] = {}
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                mask = tuple(k == i or k == j for k in range(n_seeds))
                if mask in accuracy_by_mask:
                    pairs[(slot_ids[i], slot_ids[j])] = accuracy_by_mask[mask] - baseline

        expected = sum(individuals.values())
        improvement = combined - baseline
        synergy = combined - baseline - expected
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
                if s1 in individuals:
                    ind1 = individuals[s1]
                else:
                    ind1 = 0.0
                if s2 in individuals:
                    ind2 = individuals[s2]
                else:
                    ind2 = 0.0
                pair_synergy = contrib - ind1 - ind2
                if pair_synergy > 0.5:
                    style = "green"
                elif pair_synergy < -0.5:
                    style = "red"
                else:
                    style = None
                line = self._make_bar_line(label, acc, baseline, combined, contrib, highlight=style)
                self._append_interaction(line, pair_synergy)
                lines.append(line)
        elif pairs and n_seeds > 3:
            # Show top 5 by synergy (or worst 5 by interference)
            lines.append(Text(""))
            show_interference = synergy < -0.5
            if show_interference:
                lines.append(Text("Top Interference Pairs (most negative):", style="bold red"))
            else:
                lines.append(Text("Top Combinations (by synergy):", style="bold"))

            # Calculate synergy for each pair
            pair_synergies = []
            for (s1, s2), contrib in pairs.items():
                ind1 = individuals.get(s1, 0)
                ind2 = individuals.get(s2, 0)
                pair_synergy = contrib - ind1 - ind2
                pair_synergies.append(((s1, s2), contrib, pair_synergy))

            # Sort by interaction direction and take top 5
            pair_synergies.sort(key=lambda x: x[2], reverse=not show_interference)
            for (s1, s2), contrib, pair_syn in pair_synergies[:5]:
                acc = baseline + contrib
                label = f"  {s1} + {s2}"
                if pair_syn > 0.5:
                    style = "green"
                elif pair_syn < -0.5:
                    style = "red"
                else:
                    style = None
                line = self._make_bar_line(label, acc, baseline, combined, contrib, highlight=style)
                self._append_interaction(line, pair_syn)
                lines.append(line)

        # Combined
        lines.append(Text(""))
        lines.append(Text("Combined:", style="bold"))
        lines.append(self._make_bar_line("  All seeds", combined, baseline, combined, improvement))

        # Synergy summary
        lines.append(Text(""))
        lines.append(Text(f"Expected (sum of solo): +{expected:.1f}%", style="dim"))
        lines.append(Text(f"Actual improvement:     +{improvement:.1f}%", style="dim"))

        # Interference is MORE critical to surface than synergy - seeds hurting each other
        # Use loud visual treatment for negative cases
        # Show "available at episode end" only when we don't have pair data yet
        # NOTE: Only show interference when n_seeds >= 2 - single seed cannot interfere
        # with itself (mathematically synergy=0). Stale matrix data from pruned seeds
        # can show non-zero synergy but it's misleading for single-seed scenarios.
        if is_ablation and not pairs:
            lines.append(Text("(Pair interactions available at episode end)", style="dim italic"))
        elif synergy < -0.5 and n_seeds >= 2:
            # INTERFERENCE: Seeds are hurting each other - make this LOUD
            lines.append(Text(""))
            interference = Text()
            interference.append("✗", style="bold red")
            interference.append(" INTERFERENCE DETECTED", style="bold red reverse")
            lines.append(interference)
            lines.append(Text(f"  Seeds are hurting each other by {synergy:.1f}%", style="red"))
        elif synergy > 0.5:
            # Synergy: Seeds working together
            lines.append(Text(f"✓ Synergy:              +{synergy:.1f}%", style="bold green"))
        else:
            # Neutral: Seeds are independent
            lines.append(Text(f"  Interaction:          {synergy:+.1f}%", style="dim"))

        # Per-seed interaction metrics section (always visible, greyed out if no data)
        lines.append(Text(""))
        lines.extend(self._render_interaction_metrics())

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

    def _append_interaction(self, line: Text, interaction: float) -> None:
        """Append pair interaction term to a bar line."""
        if interaction > 0.5:
            style = "green"
        elif interaction < -0.5:
            style = "red"
        else:
            style = "dim"
        line.append(f"  I {interaction:+.1f}", style=style)

    def _render_interaction_metrics(self) -> list[Text]:
        """Render aggregate interaction metrics from seeds.

        All rows are always visible to prevent jarring layout shifts.
        Empty/zero values display as dim "--" placeholders.

        Shows:
        - Network synergy: total interaction_sum across all seeds
        - Best partnership: seed with highest boost_received
        - Contribution trends: seeds with positive/negative velocity
        """
        lines: list[Text] = []
        lines.append(Text("Seed Dynamics:", style="bold"))
        dim_placeholder = "--"

        # Filter to active seeds (not DORMANT or PRUNED)
        active_seeds = [
            (slot_id, seed)
            for slot_id, seed in self._seeds.items()
            if seed.stage not in ("DORMANT", "PRUNED", "EMBARGOED", "RESETTING")
        ] if self._seeds else []

        # Network synergy: sum of all interaction_sums (positive = synergy network)
        if active_seeds:
            total_synergy = sum(seed.interaction_sum for _, seed in active_seeds)
            synergy_style = "green" if total_synergy > 0.5 else "red" if total_synergy < -0.5 else "dim"
            synergy_icon = "▲" if total_synergy > 0 else "▼" if total_synergy < 0 else "─"
            lines.append(Text(f"  Network synergy: {synergy_icon} {total_synergy:+.2f}", style=synergy_style))
        else:
            lines.append(Text(f"  Network synergy: {dim_placeholder}", style="dim"))

        # Best partnership: seed receiving highest boost
        best_boost_seed = max(active_seeds, key=lambda x: x[1].boost_received, default=None) if active_seeds else None
        if best_boost_seed and best_boost_seed[1].boost_received > 0.1:
            slot_id, seed = best_boost_seed
            lines.append(Text(
                f"  Best partner: {slot_id} (+{seed.boost_received:.1f}% boost)",
                style="cyan",
            ))
        else:
            lines.append(Text(f"  Best partner: {dim_placeholder}", style="dim"))

        # Contribution trends: show seeds with notable velocity
        trending_up = [(s, sd) for s, sd in active_seeds if sd.contribution_velocity > 0.01]
        trending_down = [(s, sd) for s, sd in active_seeds if sd.contribution_velocity < -0.01]

        trend_line = Text("  Trends: ")
        if trending_up or trending_down:
            if trending_up:
                up_slots = ", ".join(s for s, _ in trending_up[:3])
                trend_line.append(f"↗ {up_slots}", style="green")
            if trending_up and trending_down:
                trend_line.append("  ")
            if trending_down:
                down_slots = ", ".join(s for s, _ in trending_down[:3])
                trend_line.append(f"↘ {down_slots}", style="yellow")
        else:
            trend_line.append(dim_placeholder, style="dim")
        lines.append(trend_line)

        return lines
