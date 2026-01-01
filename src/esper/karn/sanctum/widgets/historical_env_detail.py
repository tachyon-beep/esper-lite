"""HistoricalEnvDetail - Modal for viewing frozen env state from Best Runs.

Shows the complete environment state as it was when the env achieved its
peak accuracy and was added to the leaderboard. Unlike EnvDetailScreen,
this is a static snapshot (no live updates).

Triggered by left-clicking a row in the Best Runs scoreboard.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from esper.karn.constants import DisplayThresholds
from esper.karn.sanctum.formatting import format_params
from esper.karn.sanctum.widgets.counterfactual_panel import CounterfactualPanel
from esper.karn.sanctum.widgets.env_detail_screen import SeedCard

if TYPE_CHECKING:
    from esper.karn.sanctum.schema import BestRunRecord


class HistoricalEnvDetail(ModalScreen[None]):
    """Modal for viewing historical env snapshot from Best Runs leaderboard.

    Shows the complete environment state at the moment it achieved peak
    accuracy and was added to the leaderboard. This is a frozen view -
    no live updates occur.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    HistoricalEnvDetail {
        align: center middle;
        background: $surface-darken-1 90%;
    }

    HistoricalEnvDetail > #modal-container {
        width: 95%;
        height: 95%;
        background: $surface;
        border: thick $secondary;
        padding: 1 2;
    }

    HistoricalEnvDetail .header-bar {
        height: 4;
        padding: 0 1;
        background: $surface-lighten-1;
        margin-bottom: 1;
    }

    HistoricalEnvDetail .seed-grid {
        height: auto;
        min-height: 12;
    }

    HistoricalEnvDetail SeedCard {
        width: 1fr;
        height: auto;
        min-height: 10;
    }

    HistoricalEnvDetail .metrics-graveyard-row {
        height: auto;
        margin-top: 1;
        border-top: solid $secondary-lighten-2;
        padding-top: 1;
    }

    HistoricalEnvDetail .metrics-section {
        width: 1fr;
        height: auto;
        padding-right: 1;
    }

    HistoricalEnvDetail .graveyard-section {
        width: 1fr;
        height: auto;
        padding-left: 1;
        border-left: solid $secondary-lighten-2;
    }

    HistoricalEnvDetail .counterfactual-section {
        height: auto;
        margin-top: 1;
        border-top: solid $secondary-lighten-2;
        padding-top: 1;
    }

    HistoricalEnvDetail .footer-hint {
        height: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(self, record: "BestRunRecord", **kwargs: Any) -> None:
        """Initialize the historical detail screen.

        Args:
            record: The BestRunRecord to display (frozen snapshot).
        """
        super().__init__(**kwargs)
        self._record = record

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Container(id="modal-container"):
            # Header bar
            yield Static(self._render_header(), id="detail-header", classes="header-bar")

            # Seed grid - show all slots (including DORMANT)
            # Uses slot_ids if available, otherwise falls back to seeds.keys()
            slot_ids = self._record.slot_ids or sorted(self._record.seeds.keys())
            if slot_ids:
                with Horizontal(classes="seed-grid"):
                    for slot_id in slot_ids:
                        seed = self._record.seeds.get(slot_id)  # None for DORMANT
                        yield SeedCard(seed, slot_id, id=f"seed-card-{slot_id}")

            # Metrics + Graveyard side by side
            with Horizontal(classes="metrics-graveyard-row"):
                with Vertical(classes="metrics-section"):
                    yield Static(self._render_metrics(), id="detail-metrics")
                with Vertical(classes="graveyard-section"):
                    yield Static(self._render_graveyard(), id="seed-graveyard")

            # Counterfactual analysis section (always visible for stable layout)
            from esper.karn.sanctum.schema import CounterfactualSnapshot
            matrix = self._record.counterfactual_matrix or CounterfactualSnapshot(
                strategy="unavailable"
            )
            with Vertical(classes="counterfactual-section"):
                yield CounterfactualPanel(
                    matrix,
                    seeds=self._record.seeds,
                    id="counterfactual-panel",
                )

            # Footer hint
            pin_status = "📌 Pinned" if self._record.pinned else "Not pinned (right-click to pin)"
            yield Static(
                f"[dim]Press ESC, Q, or click to close  │  {pin_status}[/dim]",
                classes="footer-hint",
            )

    def on_click(self) -> None:
        """Dismiss modal on click."""
        self.dismiss()

    def _render_header(self) -> Text:
        """Render the header bar with record summary."""
        record = self._record

        header = Text()

        # Historical banner
        header.append("📜 HISTORICAL VIEW", style="bold yellow")
        header.append("  │  ")

        # Episode number
        header.append(f"Episode {record.episode + 1}", style="bold")
        header.append("  │  ")

        # Env ID
        header.append(f"Env {record.env_id}", style="dim")
        header.append("  │  ")

        # Peak accuracy (the hero metric)
        header.append(f"Peak: {record.peak_accuracy:.1f}%", style="bold green")
        header.append("  │  ")

        # Final accuracy (what it ended at)
        if record.final_accuracy != record.peak_accuracy:
            header.append(f"Final: {record.final_accuracy:.1f}%", style="cyan")
            header.append("  │  ")

        # Growth ratio
        growth = record.growth_ratio or 1.0
        if growth > 1.0:
            growth_style = (
                "yellow" if growth > DisplayThresholds.GROWTH_RATIO_WARNING else "green"
            )
            header.append(f"Growth: {growth:.2f}x", style=growth_style)
        else:
            header.append("Growth: 1.00x", style="dim")

        # A/B cohort indicator (if available)
        if record.reward_mode:
            header.append("  │  ")
            cohort_color = "cyan" if record.reward_mode == "A" else "magenta"
            header.append(f"Cohort {record.reward_mode}", style=cohort_color)

        # Pin status
        if record.pinned:
            header.append("  │  ")
            header.append("📌 PINNED", style="bold cyan")

        # Second line: parameter info
        header.append("\n")

        header.append(f"Host: {format_params(record.host_params)}", style="dim")
        header.append("  │  ")
        header.append(f"Fossilized: {record.fossilized_count}", style="green")
        header.append("  │  ")
        header.append(f"Pruned: {record.pruned_count}", style="red")

        return header

    def _render_metrics(self) -> Table:
        """Render environment metrics section with historical data.

        All rows are always visible to prevent jarring layout shifts.
        Matches the same row structure as live EnvDetailScreen for consistency.
        Empty/zero values display as dim "--" placeholders.
        """
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Metric", style="dim", width=20)
        table.add_column("Value", style="white")

        record = self._record
        rc = record.reward_components
        dim_placeholder = Text("--", style="dim")

        # Accuracy history sparkline (always visible)
        from esper.karn.sanctum.schema import make_sparkline
        if record.accuracy_history:
            acc_spark = make_sparkline(record.accuracy_history, width=40)
            table.add_row("Accuracy History", acc_spark if acc_spark else dim_placeholder)
        else:
            table.add_row("Accuracy History", dim_placeholder)

        # Reward history sparkline (always visible)
        if record.reward_history:
            rwd_spark = make_sparkline(record.reward_history, width=40)
            table.add_row("Reward History", rwd_spark if rwd_spark else dim_placeholder)
        else:
            table.add_row("Reward History", dim_placeholder)

        # Seed counts (always visible - compute from record data)
        seed_counts = Text()
        # Count active seeds from the seeds dict
        active_count = len([
            s for s in record.seeds.values()
            if s and s.stage not in ("DORMANT", "FOSSILIZED", "PRUNED")
        ])
        seed_counts.append(f"Active: {active_count}", style="cyan")
        seed_counts.append("  ")
        seed_counts.append(f"Fossilized: {record.fossilized_count}", style="green")
        seed_counts.append("  ")
        seed_counts.append(f"Pruned: {record.pruned_count}", style="red")
        table.add_row("Seed Counts", seed_counts)

        # Fossilized params (always visible - not tracked in historical, show --)
        table.add_row("Fossilized Params", dim_placeholder)

        # Action distribution (always visible - not tracked in historical, show --)
        table.add_row("Action Distribution", dim_placeholder)

        # Reward Total with PBRS fraction (always visible)
        reward_text = Text()
        if rc is not None and rc.total != 0:
            total_style = "bold green" if rc.total >= 0 else "bold red"
            reward_text.append(f"{rc.total:+.3f}", style=total_style)
            # Add PBRS fraction
            pbrs_fraction = abs(rc.stage_bonus) / abs(rc.total) if rc.total != 0 else 0.0
            pbrs_healthy = (
                DisplayThresholds.PBRS_HEALTHY_MIN
                <= pbrs_fraction
                <= DisplayThresholds.PBRS_HEALTHY_MAX
            )
            pbrs_icon = "✓" if pbrs_healthy else "⚠" if pbrs_fraction > 0 else ""
            pbrs_style = "green" if pbrs_healthy else "yellow"
            reward_text.append(f"  PBRS: {pbrs_fraction:.0%} ", style="dim")
            if pbrs_icon:
                reward_text.append(pbrs_icon, style=pbrs_style)
        else:
            reward_text.append("0.000", style="dim")
            reward_text.append("  PBRS: --", style="dim")
        table.add_row("Reward Total", reward_text)

        # Signals (always visible)
        signals = Text()
        has_signals = False
        if rc is not None:
            if rc.base_acc_delta != 0:
                style = "green" if rc.base_acc_delta > 0 else "red"
                signals.append(f"ΔAcc: {rc.base_acc_delta:+.3f}", style=style)
                has_signals = True
            if rc.compute_rent != 0:
                if has_signals:
                    signals.append("  ")
                signals.append(f"Rent: {rc.compute_rent:.3f}", style="red")
                has_signals = True
            if rc.alpha_shock != 0:
                if has_signals:
                    signals.append("  ")
                signals.append(f"Shock: {rc.alpha_shock:.3f}", style="red")
                has_signals = True
            if rc.ratio_penalty != 0:
                if has_signals:
                    signals.append("  ")
                signals.append(f"Ratio: {rc.ratio_penalty:.3f}", style="red")
                has_signals = True
        # Gaming rate not tracked in historical
        if has_signals:
            signals.append("  │  ")
        signals.append("Gaming: --", style="dim")
        table.add_row("  Signals", signals if has_signals else dim_placeholder)

        # Credits (always visible)
        credits = Text()
        has_credits = False
        if rc is not None:
            if rc.bounded_attribution != 0:
                style = "green" if rc.bounded_attribution > 0 else "red"
                credits.append(f"Attr: {rc.bounded_attribution:+.3f}", style=style)
                has_credits = True
            if rc.hindsight_credit != 0:
                hind_str = f"Hind: {rc.hindsight_credit:+.3f}"
                if rc.scaffold_count > 0:
                    hind_str += f" ({rc.scaffold_count}x, {rc.avg_scaffold_delay:.1f}e)"
                if has_credits:
                    credits.append("  ")
                credits.append(hind_str, style="blue")
                has_credits = True
            if rc.stage_bonus != 0:
                if has_credits:
                    credits.append("  ")
                credits.append(f"Stage: {rc.stage_bonus:+.3f}", style="blue")
                has_credits = True
            if rc.fossilize_terminal_bonus != 0:
                if has_credits:
                    credits.append("  ")
                credits.append(f"Foss: {rc.fossilize_terminal_bonus:+.3f}", style="blue")
                has_credits = True
        table.add_row("  Credits", credits if has_credits else dim_placeholder)

        # Warnings (always visible)
        warnings = Text()
        has_warnings = False
        if rc is not None:
            if rc.blending_warning < 0:
                warnings.append(f"Blend: {rc.blending_warning:.3f}", style="yellow")
                has_warnings = True
            if rc.holding_warning < 0:
                if has_warnings:
                    warnings.append("  ")
                warnings.append(f"Hold: {rc.holding_warning:.3f}", style="yellow")
                has_warnings = True
        table.add_row("  Warnings", warnings if has_warnings else dim_placeholder)

        # Recent actions (always visible)
        if record.action_history:
            recent = " → ".join(list(record.action_history)[-5:])
            table.add_row("Recent Actions", recent)
        else:
            table.add_row("Recent Actions", dim_placeholder)

        return table

    def _render_graveyard(self) -> Panel:
        """Render the seed graveyard showing per-blueprint lifecycle stats.

        Shows how many seeds of each blueprint type were:
        - Spawned (germinated)
        - Fossilized (successfully integrated)
        - Pruned (removed due to poor performance)

        All rows are always visible to prevent jarring layout shifts.
        Empty state shows header + placeholder row with "--" values.
        """
        record = self._record
        lines = []

        # Header row (always visible)
        header = Text()
        header.append("Blueprint       ", style="dim")
        header.append("  spawn", style="dim")
        header.append("  foss", style="dim")
        header.append("  prun", style="dim")
        header.append("  rate", style="dim")
        lines.append(header)

        # Combine all blueprints seen across spawns, fossilized, pruned
        all_blueprints = set(record.blueprint_spawns.keys())
        all_blueprints.update(record.blueprint_fossilized.keys())
        all_blueprints.update(record.blueprint_prunes.keys())

        if not all_blueprints:
            # Placeholder row when no seeds spawned (matches column structure)
            placeholder = Text()
            placeholder.append("(none)          ", style="dim italic")
            placeholder.append("     --", style="dim")
            placeholder.append("    --", style="dim")
            placeholder.append("    --", style="dim")
            placeholder.append("    --", style="dim")
            lines.append(placeholder)
        else:
            # Build graveyard display (no prefixes - header has column labels)
            for blueprint in sorted(all_blueprints):
                spawned = record.blueprint_spawns.get(blueprint, 0)
                fossilized = record.blueprint_fossilized.get(blueprint, 0)
                pruned = record.blueprint_prunes.get(blueprint, 0)

                line = Text()
                line.append(f"{blueprint:15s} ", style="white")
                line.append(f"    {spawned:2d}", style="cyan")
                line.append(f"    {fossilized:2d}", style="green")
                line.append(f"    {pruned:2d}", style="red")

                # Calculate success rate if any have terminated
                terminated = fossilized + pruned
                if terminated > 0:
                    success_rate = fossilized / terminated
                    if success_rate >= DisplayThresholds.BLUEPRINT_SUCCESS_GREEN:
                        rate_style = "green"
                    elif success_rate >= DisplayThresholds.BLUEPRINT_SUCCESS_YELLOW:
                        rate_style = "yellow"
                    else:
                        rate_style = "red"
                    line.append(f"  {success_rate * 100:3.0f}%", style=rate_style)
                else:
                    line.append("    --", style="dim")

                lines.append(line)

        content = Group(*lines)
        return Panel(content, title="Seed Graveyard", border_style="dim")
